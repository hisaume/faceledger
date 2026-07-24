# DeepFace runtime baseline

_Research date: 2026-07-24_

_Status: Historical eleven-model spike record. Its runtime decision was
superseded by the
[late version-one model-scope amendment](../specs/faceledger-v1-model-scope-amendment.md),
which retains only Facenet512 and ArcFace and returns the active baseline to
CPython 3.12 with matched TensorFlow/tf-keras 2.21._

## Question

Which managed CPython, dependency-locking process, binary compatibility floor,
Linux validation matrix, and model-asset qualification procedure should
Faceledger use for its DeepFace 0.0.100 CPU runtime?

## Decision

Use managed CPython **3.11.15**, the normal GIL-enabled
`x86_64-unknown-linux-gnu` build, for version one. Pin that exact interpreter
patch release rather than only `3.11`: Python 3.11.15 is an upstream security
release and the 3.11 line remains in security-fix support until October 2027.
It is source-only upstream, so the selected managed-Python provider, provider
version, archive identity, and archive digest must be recorded with the lock.
[Python 3.11.15 release](https://www.python.org/downloads/release/python-31115/)

This is a compatibility choice, not a claim that every dependency already has
a usable wheel. DeepFace 0.0.100 advertises Python `>=3.7` and publishes a
`py3-none-any` wheel, while TensorFlow 2.12.1 and ONNX Runtime publish CPython 3.11
x86-64 Linux wheels. The narrower exact Python choice makes the native ABI and
qualification result reproducible.
[DeepFace 0.0.100 metadata](https://pypi.org/pypi/deepface/0.0.100/json),
[TensorFlow 2.12.1 files](https://pypi.org/project/tensorflow/2.12.1/),
[ONNX Runtime files](https://pypi.org/project/onnxruntime/)

### Spike evidence that rejected CPython 3.12

The initial CPython 3.12.13 lock resolved TensorFlow 2.21.0 and passed 30 of the
33 model/format cases. All three `DeepFace` recognition-model cases failed
before weight acquisition because that adapter requires `LocallyConnected2D`,
which TensorFlow removed after 2.12. TensorFlow 2.12.1 has no CPython 3.12 wheel,
so the evidence rejects CPython 3.12 rather than merely preferring 3.11.

The corrected CPython 3.11.15 lock pins TensorFlow 2.12.1 and Keras 2.12.0. On
the spike host it passed all 33 combinations across the eleven recognition
models and JPEG, PNG, and one-frame WebP. The checked-in host evidence report
records the exact lock digest, runtime versions, glibc, acquired-asset hashes,
embedding dimensions, and case outcomes. This host result proves the candidate
graph but does not replace the controlled wheelhouse or distribution matrix.
[DeepFace model guard](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/FbDeepFace.py),
[host qualification evidence](evidence/deepface-runtime-host-offline.json)

## What “fully pinned” must mean

Pinning `deepface==0.0.100` is insufficient. Its metadata contains lower bounds
for TensorFlow, Keras, OpenCV, RetinaFace, NumPy, and the rest of its declared
runtime graph, so a future resolver can legally choose a different environment.
The release also keeps two Faceledger-required model runtimes outside its
metadata: the Dlib adapter imports `dlib` dynamically, and Buffalo_L asks for
InsightFace, ONNX Runtime, typing extensions, Pydantic, and Albumentations.
TensorFlow 2.16 and newer additionally require `tf-keras` in DeepFace's legacy
Keras mode, but those releases cannot load DeepFace's `DeepFace` recognition
model. The selected graph instead pins TensorFlow 2.12.1 and Keras 2.12.0.
[DeepFace 0.0.100 metadata](https://pypi.org/pypi/deepface/0.0.100/json),
[Dlib adapter](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Dlib.py),
[Buffalo_L adapter](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Buffalo_L.py),
[Keras compatibility check](https://github.com/serengil/deepface/blob/v0.0.100/deepface/commons/package_utils.py)

The reproducible baseline therefore needs all of the following:

1. Declare DeepFace plus the omitted Dlib and Buffalo_L runtimes as direct
   project inputs, and pin TensorFlow 2.12.1 as the model-compatibility ceiling.
2. Resolve only for CPython 3.11.15 on glibc x86-64 Linux, with an exact resolver
   version and a fixed `exclude-newer` timestamp. Commit the resulting lock;
   every runtime package, transitive package, and build dependency must have an
   exact version and source/archive hash. `uv pip compile` and `uv lock` are
   designed to turn broad requirements into exact, repeatable resolutions, and
   `uv sync` enforces the complete environment rather than merely adding
   packages.
   [uv locking documentation](https://docs.astral.sh/uv/pip/compile/),
   [uv resolution documentation](https://docs.astral.sh/uv/concepts/resolution/)
3. Materialize a wheelhouse from the frozen lock and install qualification
   environments from that wheelhouse with indexes and source builds disabled.
   A lock that permits an sdist to compile independently on each target distro
   does not define one binary runtime.
4. Build any missing native wheel once in a pinned manylinux builder, pin the
   build frontend/backend and toolchain inputs, run `auditwheel show`, repair to
   the chosen policy where necessary, and record each produced wheel's SHA-256.
   Auditwheel exists specifically to inspect and relabel Linux wheels for
   cross-distribution use.
   [auditwheel](https://github.com/pypa/auditwheel)
5. Save the complete wheel filename/tag inventory and make the release fail if
   a later resolution introduces an sdist, an unapproved external shared
   library, a non-x86-64 artifact, or a platform tag above the declared floor.

The current official `dlib` release is source-only on PyPI, so it is the known
wheelhouse gap. InsightFace 0.7.3 is also source-only, but InsightFace 1.0.1 now
publishes a `py3-none-any` wheel. If the frozen resolution selects 1.0.1 and the
Buffalo_L qualification passes, no local InsightFace wheel build is needed;
otherwise its selected source release must follow the same controlled build
path as Dlib.
[dlib files](https://pypi.org/project/dlib/),
[InsightFace 0.7.3 files](https://pypi.org/project/insightface/0.7.3/),
[InsightFace files](https://pypi.org/project/insightface/)

## Wheel tags and glibc floor

A wheel tag has Python, ABI, and platform components. `py3-none-any` is pure
Python; `cp311-cp311` binds an extension to the CPython 3.11 ABI; `abi3` uses
CPython's stable ABI. For Linux, `manylinux_2_28_x86_64` promises x86-64 glibc
2.28 or newer, while legacy `manylinux2014_x86_64` means glibc 2.17 or newer.
[Python packaging tag specification](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/),
[PEP 600](https://peps.python.org/pep-0600/)

The candidate inventory contains these material tag classes:

| Component class | Expected tag class | Consequence |
| --- | --- | --- |
| DeepFace, RetinaFace, Keras helpers, and other pure Python packages | `py3-none-any` | No CPython or glibc floor |
| CPython extension wheels | `cp311-cp311-manylinux_*_x86_64` | Exact CPython 3.11 ABI |
| Stable-ABI extensions such as OpenCV builds that use it | `cp37-abi3-manylinux*_x86_64` or equivalent | Installable on CPython 3.11, subject to platform tag |
| TensorFlow CPU runtime | `cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64` | TensorFlow 2.12.1 advertises glibc 2.17+ |
| ONNX Runtime CPU runtime | published `cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64` wheel in the selected series | Advertises glibc 2.27+ |
| TensorBoard data server | published `py3-none-manylinux_2_31_x86_64` wheel | Raises the installer's current platform-tag floor to glibc 2.31 |
| Locally built Dlib, and InsightFace only if the chosen release lacks a wheel | target `cp311-cp311-manylinux_2_28_x86_64` | Must not raise the current floor |

The frozen lock's **candidate installation floor is glibc 2.31**, because pip
must honor the TensorBoard data-server wheel's platform tag even though
`readelf --version-info` on its embedded server currently finds no symbol newer
than `GLIBC_2.28`. Build the missing native wheel in a pinned manylinux 2.28
environment, then use `auditwheel show` across the materialized wheelhouse. If
the TensorBoard wheel can be validly repaired or replaced with an audited
`manylinux_2_28` artifact, the final floor may be lowered to 2.28; until then,
the more restrictive published tag controls. Reject any artifact above the
declared final floor. If the audit cannot produce a compliant Dlib wheel, Q17
is not complete and per-distro compilation must not be presented as the same
locked runtime.

## Representative Linux qualification matrix

Run the identical managed interpreter, wheelhouse, and asset procedure on these
release-blocking x86-64 environments:

| Environment | Why it is representative |
| --- | --- |
| Ubuntu 24.04 LTS, fully updated | An older still-standard-supported Ubuntu LTS and the lowest-glibc matrix member (`libc6` 2.39), providing a useful compatibility floor rather than testing only new distributions. [Ubuntu libc6 package](https://packages.ubuntu.com/noble/libc6), [Ubuntu release cycle](https://ubuntu.com/about/release-cycle) |
| Ubuntu 26.04 LTS, fully updated | The current Ubuntu LTS and a check against the newest Debian-family desktop baseline; Canonical lists 26.04 as an April 2026 LTS. [Ubuntu 26.04 release notes](https://documentation.ubuntu.com/release-notes/26.04/), [Ubuntu lifecycle](https://ubuntu.com/about) |
| Debian 13 stable (`trixie`), fully updated | Current Debian stable; Debian 13.6 was current on the research date and the release ships glibc 2.41. [Debian releases](https://www.debian.org/releases/), [Debian 13 announcement](https://www.debian.org/News/2025/20250809) |
| Fedora Linux 44, fully updated | Current Fedora release, exercising the RPM/SELinux family; Fedora 44 ships glibc 2.43. [Fedora 44 release announcement](https://fedoramagazine.org/announcing-fedora-linux-44/), [Fedora glibc package](https://packages.fedoraproject.org/pkgs/glibc/glibc/) |
| Arch Linux snapshot dated 2026-07-24, identified by image/repository digest | Exercises a rolling distribution without pretending “Arch current” is immutable. Arch's package index showed glibc 2.43 in the research window. [Arch glibc package](https://archlinux.org/packages/core/x86_64/glibc/) |

These are behavior and installation checks, not separate native builds. Each
environment must report `platform.machine() == "x86_64"`, its glibc version,
the CPython build identity, installed distribution versions, and wheel hashes.
It must install without a compiler or network from the same wheelhouse. Testing
both Ubuntu LTS releases is deliberate: 24.04 catches unnecessary floor rises,
while 26.04 catches current-LTS integration problems.

## First-use assets and offline behavior

DeepFace stores weights under
`${DEEPFACE_HOME:-$HOME}/.deepface/weights`. Its download helper returns an
existing file without contacting the network; otherwise it downloads with
`gdown`, optionally expands ZIP or BZ2 content, and then loads the file. It
checks existence, not a published content digest. RetinaFace uses the same home
and existence-based behavior. Acquisition failure is therefore distinct from
the Python dependency lock, and a present but truncated or substituted file
can fail later during model loading.
[DeepFace folder helper](https://github.com/serengil/deepface/blob/v0.0.100/deepface/commons/folder_utils.py),
[DeepFace weight helper](https://github.com/serengil/deepface/blob/v0.0.100/deepface/commons/weight_utils.py),
[RetinaFace model loader](https://github.com/serengil/retinaface/blob/master/retinaface/model/retinaface_model.py)

The fixed vector profile needs these first-use artifacts:

| Consumer | File below `.deepface/weights/` | Release source |
| --- | --- | --- |
| RetinaFace | `retinaface.h5` | [RetinaFace loader](https://github.com/serengil/retinaface/blob/master/retinaface/model/retinaface_model.py) |
| VGG-Face | `vgg_face_weights.h5` | [VGGFace.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/VGGFace.py) |
| Facenet | `facenet_weights.h5` | [Facenet.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Facenet.py) |
| Facenet512 | `facenet512_weights.h5` | [Facenet.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Facenet.py) |
| OpenFace | `openface_weights.h5` | [OpenFace.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/OpenFace.py) |
| DeepFace | `VGGFace2_DeepFace_weights_val-0.9034.h5` (expanded from ZIP) | [FbDeepFace.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/FbDeepFace.py) |
| DeepID | `deepid_keras_weights.h5` | [DeepID.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/DeepID.py) |
| Dlib | `dlib_face_recognition_resnet_model_v1.dat` (expanded from BZ2) | [Dlib.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Dlib.py) |
| ArcFace | `arcface_weights.h5` | [ArcFace.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/ArcFace.py) |
| SFace | `face_recognition_sface_2021dec.onnx` | [SFace.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/SFace.py) |
| GhostFaceNet | `ghostfacenet_v1.h5` | [GhostFaceNet.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/GhostFaceNet.py) |
| Buffalo_L | `buffalo_l/webface_r50.onnx` | [Buffalo_L.py](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Buffalo_L.py) |

Q17 should record observed SHA-256 digests after a successful acquisition so a
qualification run can detect content drift. Those observations do not grant
redistribution rights and should not cause Faceledger to bundle weights before
Q30. One upstream Dlib URL is HTTP in the DeepFace adapter, and the general
download helper has no digest verification; the announced acquisition and
post-download digest record are therefore material qualification evidence, not
optional diagnostics.

## Qualification procedure

Use one permissibly obtained, known single-face fixture and derive three files
from the same source pixels: high-quality JPEG, PNG, and a one-frame static
WebP. Verify the WebP container has exactly one frame before inference; Pillow
exposes `n_frames` and `is_animated` for this distinction.
[Pillow image file formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#webp)

For every matrix environment:

1. Install CPython and the locked wheelhouse offline. Assert that no compiler
   or package-index access is needed and that TensorFlow runs with CPU only.
2. Point `DEEPFACE_HOME` at a new empty directory, enable only announced inbound
   asset acquisition, and call `DeepFace.represent` for every combination of
   the 11 model names and three image files with
   `detector_backend="retinaface"`, `align=True`, and
   `enforce_detection=True`.
3. For all 33 cases, require exactly one representation, a finite numeric
   embedding, and the model's expected dimension: VGG-Face 4096, Facenet 128,
   Facenet512 512, OpenFace 128, DeepFace 4096, DeepID 160, Dlib 128, ArcFace
   512, SFace 128, GhostFaceNet 512, and Buffalo_L 512. DeepFace exposes the
   eleven names in its 0.0.100 registry, while the adapter classes define their
   output shapes.
   [DeepFace 0.0.100 model registry](https://github.com/serengil/deepface/blob/v0.0.100/deepface/modules/modeling.py),
   [facial-recognition adapters](https://github.com/serengil/deepface/tree/v0.0.100/deepface/models/facial_recognition)
4. Inventory every acquired file with relative path, size, and SHA-256; compare
   it with the expected artifact list and fail on missing or unexpected files.
5. Reuse the same `DEEPFACE_HOME` in a process or container with networking
   disabled and repeat all 33 cases. Success proves that model loading and
   inference are offline after first use. A separate empty-home,
   network-disabled check should prove that acquisition failure is surfaced as
   an operation error rather than a hang or partial success.
6. Save a machine-readable report containing distro/image digest, glibc,
   interpreter, lock identity, wheel hashes and tags, asset hashes, each case's
   dimension, and pass/fail result. A release qualifies only when every case
   passes on every matrix member.

Do not use `download_all_models_in_one_shot()` as proof of this profile. That
helper downloads unrelated demography, spoofing, and detector assets and does
not include Buffalo_L; exercising the actual RetinaFace-plus-recognition calls
is both narrower and more complete for Faceledger.
[DeepFace bulk-download helper](https://github.com/serengil/deepface/blob/v0.0.100/deepface/commons/weight_utils.py)

## Completion boundary

This note selects the interpreter, candidate ABI policy, matrix, and evidence
required from qualification. Q17 is complete only after the repository records
the frozen full graph, controlled Dlib wheel, final audited tag inventory and
effective glibc floor, and passing online-then-offline reports for the entire
matrix. Resolver success or one Facenet512 smoke test alone does not satisfy
that boundary.
