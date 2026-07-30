FROM debian@sha256:fac46bff2e02f51425b6e33b0e1169f55dfb053d83511ca28aa50c09fd5ed7a4

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0t64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="Faceledger Debian 13 qualification runtime"
LABEL org.opencontainers.image.description="Native libraries only; no model weights"
