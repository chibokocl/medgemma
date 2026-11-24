import base64
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from medgemma.config import ServerSettings  # noqa: E402
from medgemma.server import create_app  # noqa: E402
from medgemma.types import Response  # noqa: E402


class FakeClient:
    def __init__(self):
        self.loaded = False
        self.last = None

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def info(self):
        return type("I", (), {"loaded": self.loaded, "backend": "fake"})()

    def ask(self, prompt, *, image=None, max_tokens=None, temperature=None):
        self.last = dict(prompt=prompt, image=image, max_tokens=max_tokens)
        return Response(text="answer", completion_tokens=3)

    def stream(self, prompt, *, image=None, max_tokens=None, temperature=None):
        yield from ["he", "llo"]


def _png_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _client(api_key=None, **kw):
    settings = ServerSettings()
    settings.api_key = api_key
    for k, v in kw.items():
        setattr(settings, k, v)
    fake = FakeClient()
    return TestClient(create_app(fake, settings)), fake


def test_health_and_ready():
    with _client()[0] as c:
        assert c.get("/healthz").json()["status"] == "ok"
        assert c.get("/readyz").status_code == 200


def test_not_ready_without_preload():
    c, _ = _client(preload=False)
    with c:
        assert c.get("/readyz").status_code == 503


def test_ask():
    c, fake = _client(max_tokens_limit=100)
    with c:
        r = c.post("/v1/ask", json={"prompt": "hi", "max_tokens": 9999})
    assert r.status_code == 200 and r.json()["text"] == "answer"
    assert fake.last["max_tokens"] == 100  # clamped


def test_ask_accepts_long_rag_prompt_but_caps_at_16000():
    c, fake = _client()
    with c:
        assert c.post("/v1/ask", json={"prompt": "x" * 16000}).status_code == 200
        assert c.post("/v1/ask", json={"prompt": "x" * 16001}).status_code == 422
    assert len(fake.last["prompt"]) == 16000


def test_ask_with_image():
    c, fake = _client()
    with c:
        r = c.post("/v1/ask", json={"prompt": "hi", "image_base64": _png_b64()})
    assert r.status_code == 200
    assert fake.last["image"].endswith(".png")


def test_bad_image_rejected():
    c, _ = _client()
    with c:
        r = c.post("/v1/ask", json={"prompt": "hi", "image_base64": "bm90IGFuIGltYWdl"})
    assert r.status_code == 422


def test_auth():
    c, _ = _client(api_key="secret")
    with c:
        assert c.post("/v1/ask", json={"prompt": "x"}).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert c.post("/v1/ask", json={"prompt": "x"}, headers=bad).status_code == 401
        good = {"Authorization": "Bearer secret"}
        assert c.post("/v1/ask", json={"prompt": "x"}, headers=good).status_code == 200
        assert c.get("/healthz").status_code == 200  # probes stay open


def test_stream():
    c, _ = _client()
    with c:
        r = c.post("/v1/ask/stream", json={"prompt": "hi"})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"text": "he"}' in r.text and r.text.strip().endswith("[DONE]")
