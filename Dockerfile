FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    MEDGEMMA_HOST=0.0.0.0

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[cuda,server]" \
 && useradd --system --uid 10001 --create-home medgemma \
 && mkdir -p /data/hf && chown -R medgemma /data

USER medgemma
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["medgemma", "serve"]
