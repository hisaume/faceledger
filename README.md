# Faceledger

Faceledger compares one source face with a user-managed local face tree and
returns threshold-qualified **candidate matches** ordered by cosine distance.
A result is not a verified identity, confidence score, or accuracy claim.

Version one exposes presentation-neutral Python operations for comparison,
cache build, cache rebuild, and model-specific recoverable trash. Final command
names and option spelling are not yet a stable interface.

## Supported runtime

The supported v1 runtime is CPU-only glibc x86-64 Linux with:

- CPython 3.12.13 managed through uv;
- the exact dependency graph in `uv.lock`;
- DeepFace 0.0.100 with RetinaFace detection and alignment;
- Facenet512 (default) and ArcFace recognition; and
- JPEG, PNG, and one-frame static WebP input.

The locked TensorFlow wheel establishes a glibc 2.27 minimum. Release
qualification passed on Ubuntu 26.04 LTS, Debian 13, Fedora 44, and a pinned
Arch 2026-07-26 image. OpenCV also needs distribution-native GLib and OpenGL
runtime libraries:

| Distribution family | Required packages |
| --- | --- |
| Ubuntu 26.04 / Debian 13 | `libgl1 libglib2.0-0t64` |
| Fedora 44 | `glib2 libglvnd-glx` |
| Arch | `glib2 libglvnd` |

Install uv, check out the source release, and create the locked environment:

```console
uv sync --locked
./scripts/check.sh
```

This source checkout and its lock are the v1 delivery boundary. The repository
is not currently built as a Python wheel or native distribution package.

## Operation boundary

Application code calls these public operations and their request objects:

- `faceledger.comparison.compare(ComparisonRequest(...))`
- `faceledger.maintenance.build_vector_cache(CacheBuildRequest(...))`
- `faceledger.maintenance.rebuild_vector_cache(CacheBuildRequest(...))`
- `faceledger.trash.trash_vector_cache(TrashRequest(...))`

The returned outcomes keep candidate results, diagnostics, progress, success,
and completeness separate so a caller can present them without relying on
internal implementation details.

## Data, cache, and concurrency boundaries

Faceledger processes images and embeddings locally and sends no telemetry or
uploads. Its sole permitted network activity is an announced, inbound
dependency-managed download when a required model asset is missing. Those
assets can then be reused offline.

Local does not mean encrypted. NPY caches and recoverable trash are sensitive
plaintext protected only by the user's filesystem controls. Trash is stored
under the XDG application data location with a recovery manifest; v1 neither
permanently deletes nor automatically restores it, and retention and recovery
are manual responsibilities.

A structurally compatible cache proves the selected model and vector shape are
readable. It provides no freshness or provenance guarantee. Run an explicit
cache rebuild after source-image changes whenever freshness matters.

Concurrent read-only comparisons are permitted. Faceledger does not lock or
snapshot the live face tree, so descendants changing after discovery are
handled best-effort. Overlapping build, rebuild, or trash maintenance is not
supported.

## Model assets and exclusions

Faceledger does not bundle, mirror, or redistribute `facenet512_weights.h5`,
`arcface_weights.h5`, or `retinaface.h5`. Dependency-managed download is not a
grant of use rights. The reviewed sources do not establish commercial-use or
redistribution permission for the exact H5 files; users must ensure their use
complies with applicable upstream terms. See the
[licensing review](docs/research/faceledger-v1-model-licensing.md) and
[third-party notice](THIRD_PARTY_NOTICES.md).

V1 makes no support claim for GPU acceleration, ARM or other non-x86
architectures, musl/Alpine, non-Linux systems, encryption, managed permissions
or retention, secure erasure, identity verification, or model-weight
redistribution.

The complete qualification method and evidence are in
[`qualification/README.md`](qualification/README.md) and
[`docs/research/evidence/faceledger-v1/`](docs/research/evidence/faceledger-v1/).
