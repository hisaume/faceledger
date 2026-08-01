# comparison.py

## `Embedding`

- An immutable `tuple[float, ...]` used as Faceledger's internal representation
  of a face vector.

## `RecognitionAdapter.vector_for(image_path, profile)`

- Calculates one face embedding for an image using the supplied vector profile.
  Custom adapters may implement this protocol; operations use the locked
  DeepFace adapter when none is supplied.
- Adapters signal unusable images with `RecognitionFailure` and dependency-asset
  acquisition problems with `AssetAcquisitionFailure`.

## `ComparisonRequest(target_root, source=None, source_folder=None, model_name="Facenet512", threshold=None, single_target_folder=False, reuse_cache=True)`

- Selects exactly one source image or source folder and defines the target,
  model, threshold, traversal, and cache-reuse choices for a comparison.

## `CandidateMatch(identity_path, cosine_distance)`

- Identifies one threshold-qualified candidate and its cosine distance from the
  source, without asserting that the candidate is a verified identity.

## `Diagnostic(severity, category, code, path, message)`

- Carries a presentation-neutral warning, error, or information notice produced
  by an operation.

## `ComparisonMetadata(source, target_root, model_name, threshold)`

- Records the resolved inputs and active vector-profile choices used for a
  comparison.

## `ProgressNotification(category, completed_items, path, message)`

- Reports one completed work item independently of results and diagnostics.
  `completed_items` is a count so far, not a percentage or total-work estimate.

## `ComparisonOutcome(matches, diagnostics=(), progress=(), successful=True, complete=True, target_identities_compared=0, metadata=None)`

- Returns ranked candidates together with diagnostics, progress, status,
  completeness, comparison count, and resolved metadata.

## `compare(request, recognition=None, *, on_diagnostic=None, on_progress=None, cancellation_requested=None)`

- Compares one source identity with identities in the selected target root and
  returns a `ComparisonOutcome`. Comparison may reuse compatible caches but
  never writes, replaces, or removes them.
- `on_diagnostic` receives each diagnostic as it arises while the same notice is
  retained in the outcome; callback exceptions propagate to the caller.
- `on_progress` receives completed-item notifications; cancellation is checked
  at safe item boundaries and suppresses partial candidate results.

# maintenance.py

## `CacheBuildRequest(root, model_name="Facenet512", recursive=False)`

- Selects one maintenance root, recognition model, and optional recursive scope
  for cache build or rebuild.

## `CacheBuildOutcome(created, retained=(), diagnostics=(), successful=True, progress=(), complete=True)`

- Reports newly written and retained compatible caches together with operation
  diagnostics, progress, status, and completeness.

## `CacheRebuildOutcome(rebuilt, diagnostics=(), successful=True, progress=(), complete=True)`

- Reports successfully replaced caches together with operation diagnostics,
  progress, status, and completeness.

## `build_vector_cache(request, recognition=None, *, on_diagnostic=None, on_progress=None, cancellation_requested=None)`

- Creates missing selected-model cache entries and replaces structurally invalid
  ones while retaining compatible entries. Item failures are reported and do
  not stop later work unless the operation itself cannot continue.
- `on_diagnostic` streams retained diagnostics and propagates callback failures.

## `rebuild_vector_cache(request, recognition=None, *, on_diagnostic=None, on_progress=None, cancellation_requested=None)`

- Recalculates every in-scope selected-model cache, installing a replacement
  only after its vector is successfully calculated and persisted. Completed
  replacements remain in place if the operation is cancelled.
- `on_diagnostic` streams retained diagnostics and propagates callback failures.

# trash.py

## `TrashRequest(root, model_name="Facenet512", recursive=False)`

- Selects the root, model-specific cache suffix, and optional recursive scope
  for one recoverable trash action.

## `TrashOutcome(action_directory, manifest_path, moved=(), diagnostics=(), message="", successful=True, progress=(), complete=True)`

- Reports the recovery directory, durable manifest, moved entries, diagnostics,
  progress, status, and completeness of a trash action.

## `trash_vector_cache(request, *, now=None, on_diagnostic=None, on_progress=None, cancellation_requested=None)`

- Moves only exact selected-model cache entries into XDG application trash and
  records every planned, moved, or failed item in a recovery manifest. An empty
  selection succeeds without creating a trash action.
- `on_diagnostic` streams retained diagnostics and propagates callback failures.

# presentation.py

## `ComparisonArtifactRequest(result_path=None, log_path=None)`

- Selects optional destinations for a human-readable result and troubleshooting
  log; neither file is created unless requested.

## `render_matches(outcome)`

- Formats ranked candidate matches as a table, or reports that no matches were
  found. Unsuccessful outcomes produce no result text.

## `render_comparison_result(outcome)`

- Formats resolved comparison metadata followed by the ranked candidate result.

## `render_comparison_log(outcome)`

- Formats operation metadata, status, counts, and diagnostics without duplicating
  the ranked candidate table.

## `write_comparison_artifacts(outcome, request)`

- Writes only the requested result and log files and returns an updated outcome
  when a write failure changes the operation's diagnostics or status.

## `present_comparison(outcome, stdout, stderr)`

- Writes successful results to standard output and diagnostics and warning
  summaries to standard error. It returns zero for success and one for an
  unsuccessful operation.

# paths.py

## `application_data_root()`

- Resolves the Faceledger application-data directory from `XDG_DATA_HOME`,
  falling back to `~/.local/share/faceledger`, without creating it.

# vector_profiles.py

## `VectorProfile(model_name, cache_slug, expected_dimensions, cosine_threshold, detector_backend="retinaface", align=True)`

- Defines the fixed recognition, cache compatibility, threshold, detector, and
  alignment settings shared by comparison and maintenance.

## `DEFAULT_MODEL_NAME`

- Names `Facenet512` as the default recognition model.

## `VECTOR_PROFILES`

- Provides the immutable supported-profile mapping for `Facenet512` and
  `ArcFace`.
