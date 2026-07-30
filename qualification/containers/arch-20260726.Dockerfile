FROM archlinux@sha256:3406a568f45d68f0bef35dc80b3eacec8bda59b0292b2e50d5932ba1667f20cf

RUN pacman --sync --refresh --noconfirm glib2 libglvnd \
    && pacman --sync --clean --clean --noconfirm

LABEL org.opencontainers.image.title="Faceledger Arch 2026-07-26 qualification runtime"
LABEL org.opencontainers.image.description="Native libraries only; no model weights"
