# Official verification image: Python 3.11, no network needed at run time.
#   docker build -t auditgym .
#   docker run --rm --network none --cpus 2 --memory 4g auditgym
# python:3.11-slim, pinned by digest (multi-arch index) for reproducible builds.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Test-only dependencies, pinned with hashes from uv.lock. The runtime itself is stdlib only.
COPY requirements-check.txt ./
RUN pip install --require-hashes -r requirements-check.txt

COPY pyproject.toml LICENSE README.md Makefile Dockerfile ./
COPY src ./src
COPY tests ./tests
COPY schemas ./schemas
COPY configs ./configs
COPY episodes ./episodes
COPY trajectories ./trajectories
COPY docs ./docs
COPY reports ./reports

# Make the package importable in every subprocess, then run as an unprivileged user that may
# write only inside /app (runs/, test caches).
RUN echo /app/src > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/auditgym.pth" \
    && useradd --create-home --uid 10001 runner && mkdir -p /app/runs && chown -R runner /app
USER runner

ENTRYPOINT ["python", "-m", "auditgym"]
CMD ["check"]
