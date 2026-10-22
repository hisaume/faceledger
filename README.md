# Faceledger : A face-comparison CLI tool

Faceledger compares one face image (source) with a local folder containing face images (targets), and returns **candidate matches** ordered by similarity. This is achieved by comparing face embeddings using cosine distance, with the threshold of your choosing.

The result is not a verified identity, confidence score, or accuracy claim; it is merely a resemblance league-table.

This version provides an installable command-line application, and Python operations for comparison, cache build, cache rebuild, and model-specific recoverable trash.

It is intended for Linux, but should work on other platforms where Python is available.

## Install

- ### `uv` is a pre-requisite:

  Install `uv` following the official instructions: https://docs.astral.sh/uv/getting-started/installation/

- ### Install the application

  A typical installation (a persistent launcher outside the checkout environment):

  ```console
  uv tool install --python 3.12.13 .
  ```

  Then you can check it is working:

  ```console
  faceledger --version
  ```

  And display help:

  ```console
  faceledger --help
  ```

  - On first use of a recognition model, Faceledger may download its required model weights (which enable offline face comparison). Your images and face data remain local; Faceledger does not upload them or send telemetry. [Privacy details](#privacy-and-data-handling)

- ### Alternative to a typical installation

  Run directly from the checkout environment without the persistent launcher:

  ```console
  uv run --locked faceledger --help
  ```

- ### Troubleshooting

  If the locked checkout environment errors or cannot be created, re-resolve dependencies for this checkout:

  ```console
  uv sync --python 3.12.13
  ```

  This updates `uv.lock`; the resulting environment differs from the release-qualified dependency set. Then try again:

  ```console
  uv run --locked faceledger --help
  ```

## Command line

- ### Usage

  There are 2 primary modes:
  - `faceledger compare` : Compare a source image or folder to a target directory
  - `faceledger cache` : Manage the saved embedding files (cache)

  ```console
  faceledger [--version] {compare,cache} ...

  faceledger compare SOURCE TARGET_ROOT [--model {facenet512,arcface}]
      [--threshold VALUE] [--no-cache] [--no-recursive]
      [--result-file PATH] [--log-file PATH] [--no-progress]

  faceledger cache {build,rebuild,trash} TARGET_ROOT
      [--model {facenet512,arcface}] [--recursive] [--no-progress]
  ```

  - `SOURCE` is one supported image or one identity folder.
  - Comparison searches the complete `TARGET_ROOT` hierarchy by default; `--no-recursive` limits it to the root identity.
  - Cache maintenance applies only to the selected root by default; `--recursive` deliberately includes descendant identities.
  - `CTRL + C` cancels the current operation and exits gracefully.

- ### MCP support

  Faceledger includes a local stdio MCP server for use with MCP-compatible hosts such as Codex CLI.

  Configure your MCP host to launch:

  ```bash
  faceledger-mcp
  ```

  When working from the repository instead:

  ```bash
  uv run --locked faceledger-mcp
  ```

  The server exposes Faceledger comparison functionality as MCP tools, for a pair-of-image comparison workflow. When started manually, it waits for an MCP client on stdin/stdout; press `Ctrl-C` to stop it.

- ### Recognition Models

  | CLI model    | Recognition model | Default cosine-distance threshold |
  | ------------ | ----------------- | --------------------------------- |
  | `facenet512` | Facenet512        | 0.30                              |
  | `arcface`    | ArcFace           | 0.68                              |

  Facenet512 is the default. Comparison accepts a `--threshold` override
  from 0 through 2 inclusive. Lower distances are closer; results at or below the
  active threshold are candidate matches, not verified identities.

  Relaxing the threshold will result in more hits, but also more false positives:

  ```console
  faceledger compare --threshold 0.5 source target
  ```

- ### Cache Management
  - #### Create cache to speed up the comparison:
    - `compare` mode uses selected-model caches by default if they exist, but you must manually create the cache with `cache` mode first.
    - If cache does not exist, or if `--no-cache` flag is set, `compare` calculates temporary vectors instead.
    - `cache build` creates missing entries, replaces structurally invalid entries,
      and retains compatible ones.
    - `cache rebuild` refreshes every entry.
    - A compatible cache file can still be out of date, if you change a source
      image. Run `cache rebuild` if in doubt.
  - #### Delete cache:
    - `cache trash` _moves_ selected-model files to manifest-backed recovery folder below the XDG application data root (typically `~/.local/share/faceledger/trash/`). It prints the recovery directory and manifest on standard error.
    - V1 has no automatic restore or permanent-delete command: inspect the manifest and recover or retain files manually.
    - To remove trashed files permanently, _DELETE THE TRASH FOLDER MANUALLY_.

- ### Output & Results

  Successful comparison results and cache maintenance summaries use standard output;
  diagnostics, warning summaries, progress, and trash recovery locations use
  standard error. Progress appears only on an interactive terminal and can be
  disabled with `--no-progress`. Process statuses are:

  | Status | Meaning                                                                   |
  | ------ | ------------------------------------------------------------------------- |
  | 0      | Completed success, including warnings, no matches, and maintenance no-ops |
  | 1      | Valid command with validation, operation, output, or unexpected failure   |
  | 2      | Command grammar, choice, threshold, or conflicting-option error           |
  | 130    | User cancellation                                                         |
  - `Ctrl + C` requests cancellation at the next safe item boundary:
    - A cancelled `compare` emits no partial candidates or result artifact.
    - Completed cache management effects (build, rebuild, trash) remain, and trash keeps its manifest state.

  - Output Files (if specified)
    - `--result-file` writes only a successful complete comparison. `--log-file` is attempted for successful, failed, and cancelled comparisons and contains
      metadata, status, counts, and diagnostics without candidate matches or progress.
    - Previous output files will be overwritten if the path and filename are the same. The path must exist.

## Scan Target (TARGET_ROOT)

Expected face files at a glance:

- Name face files `name.face0.jpg` through `name.face9.jpg`. JPEG, PNG, and static WebP are supported.
- Extension case does not matter. `.png`, `.PNG`, and `.pNg` are all recognized equally.
- Add an exact lowercase `folder.jpg` to mark a folder as one-identity. In such case:
  - All recognized face files are combined into one embedding, as a single identity.
  - `folder0.jpg` through `folder9.jpg` are recognized as additional images for that identity.
  - `name.face0.jpg` through `name.face9.jpg` are also recognized as additional images for that identity.
  - `Folder.JPG`, `folder.png`, and similar names do NOT mark a one-identity folder; conform to JPEG only, and case-sensitive.
- Without `folder.jpg` in a folder, each numbered face file is treated as a separate identity.

## TensorFlow Configuration / CUDA support

- Faceledger uses CPU by default by setting `CUDA_VISIBLE_DEVICES=-1` when the caller has not selected a value. To try a CUDA device, set TensorFlow's native variable before starting Faceledger:

  ```console
  CUDA_VISIBLE_DEVICES=0 uv run --locked faceledger compare SOURCE TARGET_ROOT
  ```

  GPU execution is an unqualified override: Faceledger makes no support, performance, output-compatibility, or vector-cache compatibility promise for it.

- Faceledger reduces routine TensorFlow startup output by default. To restore
  TensorFlow's native diagnostics while troubleshooting, set:

  ```console
  TF_CPP_MIN_LOG_LEVEL=0 uv run --locked faceledger compare SOURCE TARGET_ROOT
  ```

---

## Technical & Legal

### Supported runtime

Faceledger is supported on CPU-only, glibc x86-64 Linux with CPython 3.12.13
managed by uv. It depends on DeepFace 0.0.100 with RetinaFace detection and
alignment, and accepts JPEG, PNG, and one-frame static WebP images.

OpenCV also needs distribution-native GLib and OpenGL runtime libraries:

| Distribution family      | Required packages        |
| ------------------------ | ------------------------ |
| Ubuntu 26.04 / Debian 13 | `libgl1 libglib2.0-0t64` |
| Fedora 44                | `glib2 libglvnd-glx`     |
| Arch                     | `glib2 libglvnd`         |

The checked-in `uv.lock` defines the release-qualified dependency graph. See
[`qualification/README.md`](qualification/README.md) for the complete tested runtime matrix and method.

### Privacy and data handling

Faceledger processes images and embeddings locally and sends no telemetry or uploads. Its sole permitted network activity is an announced, inbound dependency-managed download when a required model asset is missing. Those assets can then be reused offline.

Its design intent is to be thin and transient; it attempts to catalogue as little as possible. No database is used or saved. The directory structure of the scan-target itself is your database.

Local does not mean encrypted. NPY caches and recoverable trash are sensitive plaintext protected only by the user's filesystem controls. Trash is stored under the XDG application data location with a recovery manifest; v1 neither permanently deletes nor automatically restores it, and retention and recovery are manual responsibilities.

Concurrent read-only comparisons are permitted. Faceledger does not lock or snapshot the live face tree, so descendants changing after discovery are handled best-effort. Overlapping build, rebuild, or trash maintenance is not supported.

### Model assets and exclusions

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

### API: getting started

For integrators & engineers: application code calls these public operations and their request objects:

- `faceledger.comparison.compare(ComparisonRequest(...))`
- `faceledger.maintenance.build_vector_cache(CacheBuildRequest(...))`
- `faceledger.maintenance.rebuild_vector_cache(CacheBuildRequest(...))`
- `faceledger.trash.trash_vector_cache(TrashRequest(...))`

See the concise
[core API reference](docs/reference/core-api.md) for function signatures and their request and outcome structures.

### License

Faceledger source code is available under the [MIT License](LICENSE). Model
weights remain subject to the separate terms described above.
