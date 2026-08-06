# Faceledger v1 release qualification

The release qualifier exercises both Faceledger's public application operations
and a direct DeepFace probe. The direct probe deliberately remains in the
qualification script: it records raw embedding dimensions independently while
the operation checks prove the supported Faceledger boundary.

For every Facenet512/ArcFace and JPEG/PNG/static-WebP combination, each phase
checks raw inference, uncached comparison, cache build, cached comparison,
cache rebuild, and model-specific trash. The first-use phase starts with an
empty `DEEPFACE_HOME` and verifies announced acquisition. The offline phase
reuses the same assets with package access disabled and HTTP(S) proxies pointed
at an unreachable loopback endpoint.

The four Dockerfiles pin their distribution bases by digest and add only the
native OpenCV runtime libraries missing from bare images. They contain no
Python environment, Faceledger source, qualification fixture, or model weight.
At run time, mount the repository and isolated per-distribution directories for
the managed Python installation, uv cache, virtual environment, images,
DeepFace home, and reports. Create each environment with `uv sync --locked`;
after initially populating a shared download cache, use `uv sync --locked
--offline --no-dev` to prove that each remaining locked environment can be
created without package-network access.

Run the two phases through the locked environment:

```console
uv run --locked --no-dev python scripts/qualify_runtime.py \
  --qualify --phase first-use \
  --image JPEG=/qualification/images/face.jpg \
  --image PNG=/qualification/images/face.png \
  --image WEBP=/qualification/images/face.webp \
  --report /qualification/first-use.json

HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 \
http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
NO_PROXY= no_proxy= uv run --locked --offline --no-dev python \
  scripts/qualify_runtime.py --qualify --phase offline \
  --image JPEG=/qualification/images/face.jpg \
  --image PNG=/qualification/images/face.png \
  --image WEBP=/qualification/images/face.webp \
  --report /qualification/offline.json
```

`DEEPFACE_HOME`, `UV_PROJECT_ENVIRONMENT`, `UV_PYTHON_INSTALL_DIR`, and
`UV_CACHE_DIR` must point to the mounted isolated directories. Set
`UV_NO_MANAGED_PYTHON=1` once the pinned managed-Python artifact has been
populated. The qualifier itself fixes `CUDA_VISIBLE_DEVICES=-1` before DeepFace
or TensorFlow imports.

The checked-in matrix summary and reports are under
[`docs/research/evidence/faceledger-v1/`](../docs/research/evidence/faceledger-v1/).
The reports contain dependency inventory details and hashes, but no model
weights.

## Distribution qualification

The authoritative repository check also runs
`scripts/check_distribution.py`. It builds the source archive and wheel,
exports runtime dependencies from `uv.lock`, installs the wheel into a fresh
CPython 3.12.13 environment, and exercises the complete help grammar through
both the installed launcher and module route from outside the checkout.

The persistent local-tool routes were qualified separately with:

```console
uv tool install --python 3.12.13 .
uv tool install .
```

Both installed Faceledger 0.5.0 and selected Python 3.12.13 on the qualification
host. The bare form may select another compatible Python 3.12 elsewhere, and a
tool environment does not consume the project lock; the checkout plus
`uv.lock` remains the reproducibility authority.
