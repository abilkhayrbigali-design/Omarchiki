# Official verification image: Python 3.11, no network needed at run time.
#   docker build -t auditgym .
#   docker run --rm --network none --cpus 2 --memory 4g auditgym
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Test-only dependencies, pinned with hashes from uv.lock. The runtime itself is stdlib only.
COPY requirements-check.txt ./
RUN pip install --require-hashes -r requirements-check.txt

COPY pyproject.toml LICENSE README.md ./
COPY src ./src
COPY tests ./tests
COPY schemas ./schemas
COPY configs ./configs
COPY episodes ./episodes

RUN useradd --create-home --uid 10001 runner && mkdir -p /app/runs && chown runner /app/runs
USER runner

ENTRYPOINT ["python", "-m", "auditgym"]
CMD ["check"]
