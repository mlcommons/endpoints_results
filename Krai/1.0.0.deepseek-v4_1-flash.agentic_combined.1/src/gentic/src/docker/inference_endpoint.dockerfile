FROM python:3.14.7

# Same uv as upstream scripts/Dockerfile.dev, so uv.lock is read the way it is tested.
COPY --from=ghcr.io/astral-sh/uv:0.7.6 /uv /uvx /bin/

# Cloning pinned endpoints repo
ARG ENDPOINTS_REF=d3c57f2eba127c819e5a8fd83101d9e5f4e15fce

WORKDIR /opt/endpoints
RUN git clone https://github.com/mlcommons/endpoints.git . \
    && git checkout "${ENDPOINTS_REF}" \
    && uv sync --frozen --python /usr/local/bin/python3

# SWE-bench service: own locked env (requires Python 3.12, which uv downloads) + docker CLI for its task containers
COPY --from=docker:28-cli /usr/local/bin/docker /usr/local/bin/docker
RUN uv sync --frozen --python 3.12 --project src/inference_endpoint/evaluation/swebench_service \
    && src/inference_endpoint/evaluation/swebench_service/.venv/bin/python -c "import swebench_service"

# PYTHONUNBUFFERED: stream benchmark output into the job log instead of in buffered chunks.
ENV PATH=/opt/endpoints/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1
