# Faceledger v1 release-qualification findings

_Research date: 30 July 2026. This note applies the
[two-model scope amendment](../specs/faceledger-v1-model-scope-amendment.md)
and does not repeat the established
[model/weight licensing review](faceledger-v1-model-licensing.md)._

## Release boundary

The supportable v1 claim is CPU execution on glibc x86-64 Linux using managed
CPython 3.12.13 and the committed lock, with `deepface==0.0.100`,
`tensorflow==2.21.0`, and `tf-keras==2.21.0`. The recognition surface is exactly
Facenet512 and ArcFace, both using RetinaFace with alignment, across JPEG, PNG,
and one-frame WebP. These values are encoded in
[`pyproject.toml`](../../pyproject.toml), [`.python-version`](../../.python-version),
[`uv.lock`](../../uv.lock), and the
[runtime qualification contract](../../scripts/qualify_runtime.py).

CPython 3.12.13 is a source-only security release; python.org no longer supplies
binary installers for the 3.12 series. A release therefore needs to identify
and hash the exact managed-Python artifact and record the uv version used, not
merely say “official CPython 3.12.13.”
[Python 3.12.13 release](https://www.python.org/downloads/release/python-31213/),
[uv-managed Python distributions](https://docs.astral.sh/uv/concepts/python-versions/)

The locked CPython 3.12 TensorFlow artifact is
`tensorflow-2.21.0-cp312-cp312-manylinux_2_27_x86_64.whl`. Its tag sets the
effective floor of the locked binary Python dependency set at **glibc 2.27 on
x86-64**; other locked x86-64 wheels have equal or older compatible floors.
Under the packaging specification, `manylinux_2_27_x86_64` means glibc 2.27 or
newer on that architecture, not “every Linux distribution.”
[TensorFlow 2.21.0 release metadata](https://pypi.org/pypi/tensorflow/2.21.0/json),
[platform compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)

`tf-keras==2.21.0` explicitly requires TensorFlow `>=2.21,<2.22`, so the matched
pair is supported by package metadata. DeepFace 0.0.100 is a pure-Python wheel
with broad TensorFlow requirements and does not declare `tf-keras`; its metadata
therefore permits this environment but cannot prove model import or inference.
[tf-keras 2.21.0 metadata](https://pypi.org/pypi/tf-keras/2.21.0/json),
[DeepFace 0.0.100 metadata](https://pypi.org/pypi/deepface/0.0.100/json)

`uv.lock` is the reproducibility boundary: qualification must use `uv sync
--locked`/`uv run --locked`, preserve the recorded lock digest, and fail rather
than re-resolve. The project’s `environments` and `required-environments`
settings constrain and require Linux x86-64 artifacts, but they prove artifact
availability—not successful import or inference on each distribution.
[uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/),
[uv required environments](https://docs.astral.sh/uv/reference/settings/#required-environments)

## Required matrix evidence

The representative release-blocking matrix as of this research date is Ubuntu
26.04 LTS, Debian 13 stable, Fedora 44, and a date/digest-pinned current Arch
image. These are the current releases identified by their publishers; Arch is
rolling, so `latest` alone is not reproducible evidence.
[Ubuntu lifecycle](https://ubuntu.com/about/release-cycle?product=ubuntu&release=ubuntu&version=26.04+LTS),
[Debian releases](https://www.debian.org/releases/),
[Fedora 44 announcement](https://fedoramagazine.org/announcing-fedora-linux-44/),
[Arch official container image](https://hub.docker.com/_/archlinux)

Each matrix run should record the base-image name and digest, glibc and CPU
identity, exact Python and installed distribution versions, installed wheel
tags, `uv.lock` SHA-256, and the three dependency-owned asset hashes. It should
then demonstrate through Faceledger’s outward operation functions:

- an initially empty `DEEPFACE_HOME`, visibly announced first-use acquisition,
  and successful Facenet512/ArcFace inference for JPEG, PNG, and static WebP;
- a second run with the same assets and network access disabled;
- comparison, cache build, cache rebuild, cache reuse, and model-specific trash
  on CPU, including their observable diagnostics and filesystem effects.

[`qualify_runtime.py`](../../scripts/qualify_runtime.py) deliberately combines
a raw dependency probe with the public-operation qualification. It records
versions, wheel tags, assets, the lock digest, and embedding dimensions, then
exercises comparison, cache build, cache reuse, cache rebuild, and trash. Its
separate first-use and offline phases validate the dependency-asset lifecycle.
The existing older checked-in
[runtime evidence](evidence/deepface-runtime-host-offline.json) is CPython 3.11
historical evidence for the superseded eleven-model scope, not v1 release
evidence.

## Licensing and distribution

The Q30 disposition remains controlling: Faceledger may declare/install the
MIT-licensed `deepface` and `retina-face` code subject to their notices, but v1
must not bundle, mirror, or redistribute `facenet512_weights.h5`,
`arcface_weights.h5`, or `retinaface.h5` in source archives, wheels, native
packages, containers, examples, or offline wheelhouses. Library-managed
first-use download avoids Faceledger redistribution but is not a grant of use
rights. No commercial-use or model-redistribution claim is established for the
exact H5 artifacts. The documented ArcFace and RetinaFace lineage gives no
basis for a commercial-use claim and strongly indicates InsightFace’s
non-commercial research restriction, but the missing provenance manifests mean
that restriction cannot be asserted as conclusively attached to each exact H5.
[DeepFace model-assets notice](https://github.com/serengil/deepface_models),
[DeepFace licence notice](https://github.com/serengil/deepface/blob/v0.0.100/README.md#licence),
[InsightFace licence policy](https://github.com/deepinsight/insightface#license)

Release material should identify the upstream downloads and state these limits,
retain applicable MIT notices for redistributed code, and label recorded hashes
as Faceledger observations rather than upstream-published checksums. The exact
artifact identities and attribution conclusions are already captured in the
[licensing review](faceledger-v1-model-licensing.md).

## User-facing claims

Documentation can make the following bounded claims, matching the
[v1 specification](../specs/faceledger-v1.md) and current operation code:

- Results are threshold-qualified **candidate matches ordered by cosine
  distance**, not verified identities, confidence scores, or accuracy claims.
- A structurally compatible cache proves only model-qualified shape and numeric
  readability. It carries no freshness or provenance guarantee; users must run
  rebuild after source changes when freshness matters.
- Images and embeddings are processed locally, with no telemetry or uploads.
  The only permitted network activity is announced inbound acquisition of
  missing dependency assets. “Local” does not mean encrypted: NPY caches and
  application trash are sensitive plaintext governed by the user’s filesystem
  controls.
- Trash is recoverable, manifest-backed application data under the XDG data
  location. V1 neither permanently deletes nor automatically restores it;
  recovery and retention are manual user responsibilities.
- Concurrent read-only comparisons are permitted. Operations use a live tree
  without a lock or snapshot; overlapping build, rebuild, or trash maintenance
  is unsupported, and descendants changing after discovery are handled
  best-effort.

The relevant implemented seams are
[`comparison.py`](../../faceledger/comparison.py),
[`maintenance.py`](../../faceledger/maintenance.py),
[`trash.py`](../../faceledger/trash.py), and
[`deepface_adapter.py`](../../faceledger/deepface_adapter.py).

## Qualification outcome and remaining limits

The [checked-in matrix evidence](evidence/faceledger-v1/) records successful
first-use and network-disabled offline phases on pinned Ubuntu 26.04, Debian 13,
Fedora 44, and Arch 2026-07-26 images. All eight runs used CPython 3.12.13 and
the same lock digest. Each run passed the six two-model/static-format cases and
the six corresponding public-operation sequences without runtime-contract or
asset-lifecycle errors. The qualification images add only the native GLib and
OpenGL libraries needed by OpenCV; model assets remain outside the images and
repository.

Faceledger 0.5.0 now builds as a source archive and pure-Python wheel. The
repository distribution check installs that wheel into a fresh CPython 3.12.13
environment against dependencies exported from `uv.lock`, confirms the import
comes from the installed package, and exercises the complete public grammar
through both the launcher and module route. Neither artifact contains model
weights.

Local `uv tool install --python 3.12.13 .` and bare `uv tool install .` were
also qualified; both selected Python 3.12.13 on the qualification host. The
bare form may select another compatible 3.12 patch, and tool installation does
not consume the lock. The source checkout plus committed lock therefore remains
the reproducibility authority. Registry publication and native distribution
packages remain outside the v1 scope.

The reviewed primary sources still do not establish redistribution or
commercial permission for the exact three H5 files. Only written
artifact-specific permission or requalified replacement weights can close that
rights gap, so the release explicitly excludes bundling and commercial-use
claims for those assets.
