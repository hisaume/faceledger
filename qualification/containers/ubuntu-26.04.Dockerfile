FROM ubuntu@sha256:3131b4cc82a783df6c9df078f86e01819a13594b865c2cad47bd1bca2b7063bb

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0t64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="Faceledger Ubuntu 26.04 qualification runtime"
LABEL org.opencontainers.image.description="Native libraries only; no model weights"
