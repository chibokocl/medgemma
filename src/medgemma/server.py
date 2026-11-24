"""HTTP API server (FastAPI).

Run with ``medgemma serve`` or ``uvicorn medgemma.server:create_app --factory``.
"""

import base64
import binascii
import hmac
import io
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager, contextmanager
from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ._version import __version__
from .client import MedGemma
from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ServerSettings

log = logging.getLogger("medgemma.server")
_bearer = HTTPBearer(auto_error=False)


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    image_base64: str | None = Field(
        default=None, description="Base64-encoded image (PNG/JPEG/WebP)."
    )
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class AskResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    elapsed_seconds: float


@contextmanager
def _decoded_image(b64: str | None, max_mb: int) -> Iterator[str | None]:
    """Validate a base64 image and expose it as a temp file path."""
    if not b64:
        yield None
        return
    if len(b64) > max_mb * 1024 * 1024 * 4 // 3 + 4:
        raise HTTPException(413, f"Image exceeds {max_mb} MB limit.")
    try:
        raw = base64.b64decode(b64, validate=True)
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        fmt = (img.format or "").lower()
        img.verify()
    except (binascii.Error, ValueError, OSError) as exc:
        raise HTTPException(422, "image_base64 is not a valid image.") from exc
    if fmt not in {"png", "jpeg", "webp", "gif", "bmp"}:
        raise HTTPException(422, f"Unsupported image format: {fmt or 'unknown'}.")
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as tmp:
        tmp.write(raw)
        tmp.flush()
        yield tmp.name


def create_app(
    client: MedGemma | None = None, settings: ServerSettings | None = None
) -> FastAPI:
    settings = settings or ServerSettings()
    client = client or MedGemma()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload:
            log.info("Loading model…")
            client.load()
            log.info("Model ready.")
        if not settings.api_key:
            log.warning("MEDGEMMA_API_KEY is not set: the API is unauthenticated.")
        yield
        client.unload()

    app = FastAPI(
        title="MedGemma API",
        version=__version__,
        description="Informational/educational use only – not medical advice.",
        lifespan=lifespan,
    )

    def auth(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
        if not settings.api_key:
            return
        supplied = creds.credentials if creds else ""
        if not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
            raise HTTPException(
                401, "Invalid or missing API key.", headers={"WWW-Authenticate": "Bearer"}
            )

    def _limits(req: AskRequest) -> tuple[int, float]:
        max_tokens = min(req.max_tokens or DEFAULT_MAX_TOKENS, settings.max_tokens_limit)
        temp = req.temperature if req.temperature is not None else DEFAULT_TEMPERATURE
        return max_tokens, temp

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        """Liveness probe."""
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz():
        """Readiness probe – 503 until the model is loaded."""
        info = client.info()
        body = {"ready": info.loaded, "backend": info.backend}
        return JSONResponse(body, status_code=200 if info.loaded else 503)

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(auth)], tags=["inference"])
    def ask(req: AskRequest) -> AskResponse:
        max_tokens, temp = _limits(req)
        with _decoded_image(req.image_base64, settings.max_image_mb) as image:
            resp = client.ask(req.prompt, image=image, max_tokens=max_tokens, temperature=temp)
        return AskResponse(**resp.__dict__)

    @app.post("/v1/ask/stream", dependencies=[Depends(auth)], tags=["inference"])
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Server-Sent Events: ``data: {"text": ...}`` chunks, then ``data: [DONE]``."""
        max_tokens, temp = _limits(req)

        # Validate the image before the response starts so errors are real 4xx.
        ctx = _decoded_image(req.image_base64, settings.max_image_mb)
        image = ctx.__enter__()

        def events() -> Iterator[str]:
            try:
                for chunk in client.stream(
                    req.prompt, image=image, max_tokens=max_tokens, temperature=temp
                ):
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                ctx.__exit__(None, None, None)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
