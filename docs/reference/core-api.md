# Core API Reference

Public types and operations used across Faceledger's core and presentation
boundaries. Signatures mirror the source; private helpers are omitted.

## comparison.py

### `Embedding`

```python
type Embedding = tuple[float, ...]
```

- Immutable internal representation of a face vector.

### `RecognitionAdapter`

```python
class RecognitionAdapter(Protocol):
    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> Sequence[float]: ...
```

- Boundary for calculating one face embedding. Operations use
  `DeepFaceRecognition` when no adapter is supplied.
- Implementations raise `RecognitionFailure` for unusable images and
  `AssetAcquisitionFailure` when dependency assets cannot be acquired.

### `RecognitionFailure`

- Signals that a selected image could not produce exactly one valid face
  vector.

### `AssetAcquisitionFailure`

- Signals that DeepFace could not acquire a required model or detector asset.

### `InvalidCacheEntry`

- Signals a structurally incompatible vector-cache entry. Public operations
  translate it into a diagnostic and recalculate or replace the entry.

### `ComparisonRequest`

```python
@dataclass(frozen=True)
class ComparisonRequest:
    target_root: Path
    source: Path | None = None
    source_folder: Path | None = None
    model_name: str = DEFAULT_MODEL_NAME
    threshold: float | None = None
    single_target_folder: bool = False
    reuse_cache: bool = True
```

- Selects exactly one source image or source folder and defines the target,
  model, threshold, traversal, and cache-reuse choices.

### `CandidateMatch`

```python
@dataclass(frozen=True)
class CandidateMatch:
    identity_path: Path
    cosine_distance: float
```

- Identifies one threshold-qualified candidate without asserting a verified
  identity.

### `Diagnostic`

```python
@dataclass(frozen=True)
class Diagnostic:
    severity: str
    category: str
    code: str
    path: Path | None
    message: str
```

- Presentation-neutral warning, error, or information notice produced by an
  operation.

### `ComparisonMetadata`

```python
@dataclass(frozen=True)
class ComparisonMetadata:
    source: Path
    target_root: Path
    model_name: str
    threshold: float
```

- Resolved inputs and active vector-profile choices used for a comparison.

### `ProgressNotification`

```python
@dataclass(frozen=True)
class ProgressNotification:
    category: str
    completed_items: int
    path: Path
    message: str
```

- Reports one completed work item independently of results and diagnostics.
  `completed_items` is a count so far, not a percentage or total estimate.

### `ComparisonOutcome`

```python
@dataclass(frozen=True)
class ComparisonOutcome:
    matches: tuple[CandidateMatch, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    progress: tuple[ProgressNotification, ...] = ()
    successful: bool = True
    complete: bool = True
    target_identities_compared: int = 0
    metadata: ComparisonMetadata | None = None
```

- Candidates ranked by cosine distance and then canonical result identity path,
  together with diagnostics, progress, status, completeness, comparison count,
  and resolved metadata.

### `compare`

```python
def compare(
    request: ComparisonRequest,
    recognition: RecognitionAdapter | None = None,
    *,
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
    on_progress: Callable[[ProgressNotification], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> ComparisonOutcome
```

- Compares one source identity with identities in the selected target root.
  Compatible caches may be reused but are never written, replaced, or removed.
- Diagnostic and progress callbacks receive notices as they arise. Cancellation
  is checked at safe item boundaries and suppresses partial candidate results.
  Callback exceptions propagate.

## console.py

### `ConsolePresentationFailure`

- Signals that a terminal stream could not render or flush operation feedback.

### `ComparisonConsole`

```python
class ComparisonConsole:
    def __init__(
        self,
        stdout: TextIO,
        stderr: TextIO,
        *,
        show_progress: bool = False,
    ) -> None: ...

    def diagnostic(self, diagnostic: Diagnostic) -> None: ...
    def progress(self, notification: ProgressNotification) -> None: ...
    def present(self, outcome: ComparisonOutcome) -> int: ...
    def report_presentation_failure(
        self,
        error: ConsolePresentationFailure,
    ) -> int: ...
```

- Owns live terminal diagnostics, transient completed-item progress, final
  comparison output, warning summaries, and stream-failure reporting.
- `present` does not replay retained diagnostics. The caller streams each
  diagnostic once through `diagnostic` and enables progress only for an
  interactive standard-error stream when the user has not suppressed it.

## deepface_adapter.py

### `DeepFaceRecognition`

```python
class DeepFaceRecognition:
    def __init__(
        self,
        announce_missing_asset: Callable[[Path], None],
    ) -> None: ...

    def vector_for(
        self,
        image_path: Path,
        profile: VectorProfile,
    ) -> Sequence[float]: ...
```

- Concrete `RecognitionAdapter` used by default. It isolates DeepFace, locks
  recognition to the selected CPU profile, validates its output, and announces
  missing dependency assets before DeepFace may acquire them.

## maintenance.py

### `CacheBuildRequest`

```python
@dataclass(frozen=True)
class CacheBuildRequest:
    root: Path
    model_name: str = DEFAULT_MODEL_NAME
    recursive: bool = False
```

- Selects one root, recognition model, and optional recursive scope for cache
  build or rebuild.

### `CacheBuildOutcome`

```python
@dataclass(frozen=True)
class CacheBuildOutcome:
    created: tuple[Path, ...]
    retained: tuple[Path, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    successful: bool = True
    progress: tuple[ProgressNotification, ...] = ()
    complete: bool = True
```

- Newly written and retained compatible caches together with diagnostics,
  progress, status, and completeness.

### `CacheRebuildOutcome`

```python
@dataclass(frozen=True)
class CacheRebuildOutcome:
    rebuilt: tuple[Path, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    successful: bool = True
    progress: tuple[ProgressNotification, ...] = ()
    complete: bool = True
```

- Successfully replaced caches together with diagnostics, progress, status,
  and completeness.

### `build_vector_cache`

```python
def build_vector_cache(
    request: CacheBuildRequest,
    recognition: RecognitionAdapter | None = None,
    *,
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
    on_progress: Callable[[ProgressNotification], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> CacheBuildOutcome
```

- Creates missing selected-model entries, replaces structurally invalid ones,
  and retains compatible entries. Item failures do not stop later work unless
  the operation itself cannot continue.
- Callbacks stream retained diagnostics and completed-item progress. Callback
  exceptions propagate.

### `rebuild_vector_cache`

```python
def rebuild_vector_cache(
    request: CacheBuildRequest,
    recognition: RecognitionAdapter | None = None,
    *,
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
    on_progress: Callable[[ProgressNotification], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> CacheRebuildOutcome
```

- Recalculates every in-scope selected-model cache and installs a replacement
  only after successful calculation and persistence. Completed replacements
  remain if the operation is cancelled.
- Callbacks stream retained diagnostics and completed-item progress. Callback
  exceptions propagate.

## trash.py

### `TrashRequest`

```python
@dataclass(frozen=True)
class TrashRequest:
    root: Path
    model_name: str = DEFAULT_MODEL_NAME
    recursive: bool = False
```

- Selects the root, model-specific cache suffix, and optional recursive scope
  for one recoverable trash action.

### `TrashOutcome`

```python
@dataclass(frozen=True)
class TrashOutcome:
    action_directory: Path | None
    manifest_path: Path | None
    moved: tuple[Path, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    message: str = ""
    successful: bool = True
    progress: tuple[ProgressNotification, ...] = ()
    complete: bool = True
```

- Recovery directory, durable manifest, moved entries, diagnostics, progress,
  status, and completeness of a trash action.

### `trash_vector_cache`

```python
def trash_vector_cache(
    request: TrashRequest,
    *,
    now: Callable[[], datetime] | None = None,
    on_diagnostic: Callable[[Diagnostic], None] | None = None,
    on_progress: Callable[[ProgressNotification], None] | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> TrashOutcome
```

- Moves exact selected-model cache entries into XDG application trash and
  records every planned, moved, or failed item in a recovery manifest. An empty
  selection succeeds without creating a trash action.
- Callbacks stream retained diagnostics and completed-item progress. Callback
  exceptions propagate.

## presentation.py

### `ComparisonArtifactRequest`

```python
@dataclass(frozen=True)
class ComparisonArtifactRequest:
    result_path: Path | None = None
    log_path: Path | None = None
```

- Selects optional human-readable result and troubleshooting-log destinations;
  neither file is created unless requested.

### `render_matches`

```python
def render_matches(outcome: ComparisonOutcome) -> str
```

- Formats ranked candidates as a table, or reports that no matches were found.
  Unsuccessful outcomes produce no result text.

### `render_comparison_result`

```python
def render_comparison_result(outcome: ComparisonOutcome) -> str
```

- Formats resolved comparison metadata followed by ranked candidates.

### `render_comparison_log`

```python
def render_comparison_log(outcome: ComparisonOutcome) -> str
```

- Formats operation metadata, status, counts, and diagnostics without
  duplicating the candidate table.

### `write_comparison_artifacts`

```python
def write_comparison_artifacts(
    outcome: ComparisonOutcome,
    request: ComparisonArtifactRequest,
) -> ComparisonOutcome
```

- Writes only requested files and returns an updated outcome when a write
  failure changes diagnostics or status.

### `present_comparison`

```python
def present_comparison(
    outcome: ComparisonOutcome,
    stdout: TextIO,
    stderr: TextIO,
) -> int
```

- Writes successful results to standard output and diagnostics and warning
  summaries to standard error through `ComparisonConsole`. Returns zero on
  success and one otherwise.

## paths.py

### `application_data_root`

```python
def application_data_root() -> Path
```

- Resolves the Faceledger application-data directory from `XDG_DATA_HOME`,
  falling back to `~/.local/share/faceledger`, without creating it.

## vector_profiles.py

### `VectorProfile`

```python
@dataclass(frozen=True)
class VectorProfile:
    model_name: str
    cache_slug: str
    expected_dimensions: int
    cosine_threshold: float
    detector_backend: str = "retinaface"
    align: bool = True
```

- Fixed recognition, cache compatibility, threshold, detector, and alignment
  settings shared by comparison and maintenance.

### `DEFAULT_MODEL_NAME`

```python
DEFAULT_MODEL_NAME = "Facenet512"
```

- Default recognition model used by requests.

### `VECTOR_PROFILES`

```python
VECTOR_PROFILES: Mapping[str, VectorProfile]
```

- Immutable supported-profile mapping for `Facenet512` and `ArcFace`.

## cli.py

### `main`

```python
def main(
    arguments: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    recognition: RecognitionAdapter | None = None,
) -> int
```

- Shared application entry point used by the installed `faceledger` launcher
  and `python -m faceledger`. Optional streams and recognition adapter support
  embedding and deterministic tests without changing command syntax.
