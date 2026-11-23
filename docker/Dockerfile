# Frozen runtime for the solar_uq experiments.
#
# Every Python package is installed offline from docker/wheelhouse/, a local copy of
# the exact wheels in docker/requirements.lock.txt. The PyTorch build is a CUDA 13
# nightly, and nightlies are eventually removed from the PyTorch index, so the
# wheelhouse is what keeps the image rebuildable. It is not in git, and it can no
# longer be rebuilt from the internet: by 2026-11-23 the torch nightly was gone from
# the index listing (only reachable by direct URL) and triton had been republished
# under the same version with different binaries, so the triton wheel was rebuilt
# from the installed copy (docker/wheel_from_venv.py). Check it against
# docker/wheelhouse.sha256 and keep a second copy off this machine.
#
# CUDA itself comes from the nvidia-* wheels, so the base image needs no CUDA. The
# host only needs an NVIDIA driver that supports CUDA 13 (>= 580) and the NVIDIA
# Container Toolkit.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t solar-uq:2026.11 .
# Data, runs and logs are mounted at run time (see docker/compose.yaml).

# syntax=docker/dockerfile:1
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Ubuntu 24.04 ships Python 3.12.3, the same interpreter the experiments ran on.
# procps (ps) is used by scripts/ram_guard.sh and the queue scripts.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ca-certificates git procps libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# --no-deps installs exactly the frozen set, the same way the working venv is made.
# Three pairs of packages write the same files (nccl cu12/cu13, cudnn cu12/cu13,
# nvidia-ml-py/nvidia-ml-py3), and whichever is installed last wins. With the cu12
# NCCL on top, torch fails to import (undefined symbol ncclCommWindowDeregister).
# requirements.overlap.txt lists the copies that won in the working venv; they are
# reinstalled last. Verified 2026-11-23: the result matches the working venv file
# for file (54,724 files; only gpustat's generated _version.py differs).
COPY docker/requirements.lock.txt docker/requirements.overlap.txt /tmp/
RUN --mount=type=bind,source=docker/wheelhouse,target=/wheelhouse \
    pip install --no-index --find-links /wheelhouse --no-deps \
        -r /tmp/requirements.lock.txt \
 && pip install --no-index --find-links /wheelhouse --no-deps --force-reinstall \
        -r /tmp/requirements.overlap.txt

# The package is installed editable so solar_uq.zoo.PROJECT_ROOT resolves to
# /workspace, where data/, runs/ and logs/ are mounted.
WORKDIR /workspace
COPY pyproject.toml README.md CITATION.cff ./
COPY src ./src
RUN pip install --no-index --no-deps --no-build-isolation -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docker/smoke_test.py ./docker/smoke_test.py

CMD ["python", "docker/smoke_test.py"]
