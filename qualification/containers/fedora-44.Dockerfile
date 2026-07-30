FROM fedora@sha256:6c75d5bf57cb0fa5aa4b92c6a83c86c791644496d9ac230de7711f5b8ec3b898

RUN dnf install --assumeyes glib2 libglvnd-glx \
    && dnf clean all

LABEL org.opencontainers.image.title="Faceledger Fedora 44 qualification runtime"
LABEL org.opencontainers.image.description="Native libraries only; no model weights"
