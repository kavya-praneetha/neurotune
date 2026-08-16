# syntax=docker/dockerfile:1.7
#
# CPU-only image for the ai-lab environment.
#
# The point of this file is that `.venv/` never needs to be shared or committed:
# `pyproject.toml` + `uv.lock` reconstruct the exact same 285-package
# environment on any machine, pinned down to the OS and Python build.

FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# xgboost ships libxgboost.so, which links libstdc++.so.6 and libgcc_s.so.1 at
# runtime; a slim Debian base does not guarantee libstdc++6 is present.
# Everything else native is already vendored inside its wheel — libgomp comes in
# faiss_cpu.libs/, xgboost.libs/ and scikit_learn.libs/, and libsndfile comes in
# soundfile's _soundfile_data/ — so nothing further is needed here.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 \
 && rm -rf /var/lib/apt/lists/*

# Bytecode is compiled at build time so container startup isn't paying for it.
# copy link mode avoids hardlink warnings when the uv cache is a mount.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer, deliberately separate from the source layer: it is rebuilt
# only when uv.lock or pyproject.toml change, not on every source edit.
# --locked fails loudly if the lockfile is stale rather than silently resolving
# something different from what was tested here.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app

ENV PATH="/app/.venv/bin:$PATH"

# ai-lab declares no [build-system], so uv treats it as a virtual project:
# dependencies are installed but the project itself is not. `neurotune` is
# therefore imported from the working directory, which needs to be on the path.
ENV PYTHONPATH=/app

EXPOSE 8888

# JupyterLab is the documented entry point. It binds 0.0.0.0 because the
# container's loopback is not reachable from the host; token auth still applies,
# and the token is printed to the container logs on startup.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
