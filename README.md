# Faceledger

Faceledger compares one source face with a user-managed local face tree and
returns threshold-qualified **candidate matches** ordered by cosine distance.
A result is not a verified identity, confidence score, or accuracy claim.

Version one provides an installable command-line application and
presentation-neutral Python operations for comparison, cache build, cache
rebuild, and model-specific recoverable trash.

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

## Install

Install uv, check out the source release, and create the authoritative locked
environment:

```console
uv sync --locked --python 3.12.13
./scripts/check.sh
```

The installed launcher and module route share the same application entry point:

```console
uv run --locked faceledger --help
uv run --locked python -m faceledger --help
```

For a persistent launcher outside the checkout environment, the qualified local
tool route is:

```console
uv tool install --python 3.12.13 .
faceledger --version
```

`uv tool install .` is also supported and was verified to select a compatible
Python 3.12. It may select a different 3.12 patch on another machine; only the
explicit 3.12.13 command matches the tool-install qualification. A tool install
also resolves its own environment rather than consuming `uv.lock`, so the
source checkout and lock remain the v1 reproducibility boundary.

`uv build` creates the source archive and pure-Python wheel. Publishing them to
a registry and producing native distribution packages remain outside the
version-one scope.

## Command line

```text
faceledger [--version] {compare,cache} ...
faceledger compare SOURCE TARGET_ROOT [--model {facenet512,arcface}]
    [--threshold VALUE] [--no-cache] [--no-recursive]
    [--result-file PATH] [--log-file PATH] [--no-progress]
faceledger cache {build,rebuild,trash} ROOT
    [--model {facenet512,arcface}] [--recursive] [--no-progress]
```

`SOURCE` is one supported image or one identity folder. Comparison searches the
complete `TARGET_ROOT` hierarchy by default; `--no-recursive` limits it to the
root identity. Cache maintenance changes only the selected root by default;
`--recursive` deliberately includes descendant identities.

| CLI model | Recognition model | Default cosine-distance threshold |
| --- | --- | --- |
| `facenet512` | Facenet512 | 0.30 |
| `arcface` | ArcFace | 0.68 |

Facenet512 is the default. Comparison accepts a finite `--threshold` override
from 0 through 2 inclusive. Lower distances are closer; results at or below the
active threshold are candidate matches, not verified identities.

Comparison reads compatible selected-model caches by default but never creates,
repairs, or removes them. `--no-cache` calculates transient vectors instead.
`cache build` creates missing entries, replaces structurally invalid entries,
and retains compatible ones; `cache rebuild` refreshes every in-scope entry.
Structural compatibility is not a freshness or provenance guarantee, so rebuild
after source-image changes whenever freshness matters.

`cache trash` moves exact selected-model entries to manifest-backed recovery
storage below the XDG application data root. It prints the recovery directory
and manifest on standard error. V1 has no automatic restore or permanent-delete
command: inspect the manifest and recover or retain files manually.

Successful comparison results and maintenance summaries use standard output;
diagnostics, warning summaries, progress, and trash recovery locations use
standard error. Progress appears only on an interactive terminal and can be
disabled with `--no-progress`. Process statuses are:

| Status | Meaning |
| --- | --- |
| 0 | Completed success, including warnings, no matches, and maintenance no-ops |
| 1 | Valid command with validation, operation, output, or unexpected failure |
| 2 | Command grammar, choice, threshold, or conflicting-option error |
| 130 | User cancellation |

Ctrl+C requests cancellation at the next safe item boundary. A cancelled
comparison emits no partial candidates or result artifact; completed maintenance
effects remain in place and trash keeps its manifest state.

`--result-file` writes only a successful complete comparison. `--log-file` is
attempted for successful, failed, and cancelled comparisons and contains
metadata, status, counts, and diagnostics without candidate matches or progress.
The two destinations must differ, may overwrite regular files, and require
existing parent directories.

## Operation boundary

Application code calls these public operations and their request objects:

- `faceledger.comparison.compare(ComparisonRequest(...))`
- `faceledger.maintenance.build_vector_cache(CacheBuildRequest(...))`
- `faceledger.maintenance.rebuild_vector_cache(CacheBuildRequest(...))`
- `faceledger.trash.trash_vector_cache(TrashRequest(...))`

The returned outcomes keep candidate results, diagnostics, progress, success,
and completeness separate so a caller can present them without relying on
internal implementation details. See the concise
[core API reference](docs/reference/core-api.md) for function signatures and
their request and outcome structures.

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

## License

Faceledger source code is available under the [MIT License](LICENSE). Model
weights remain subject to the separate terms described above.
