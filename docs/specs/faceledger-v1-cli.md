# Faceledger Version One CLI Specification

Approved on 31 July 2026. This is the authoritative product specification for
Faceledger version one. It consolidates the implemented CORE domain contract,
the final two-model runtime scope, and the command-line frontend and packaging
decisions. Earlier `faceledger-v1` specifications are retained as read-only
project history and are superseded wherever this document differs.

## Problem Statement

People with local photo collections need a convenient way to find plausible
appearances of one person across a user-managed folder tree. Face-recognition
libraries can calculate embeddings, but they do not provide Faceledger's face
tree conventions, reusable model-specific vector cache, ranked candidate
workflow, recoverable cache maintenance, or careful distinction between visual
similarity and verified identity.

The completed CORE exposes these capabilities as presentation-neutral Python
operations, but users still need a coherent command-line application. That
application must translate concise commands into the CORE contract, validate
directly selected source content, preserve useful work when individual files
fail, separate results from diagnostics, make long work observable and safely
cancellable, and provide predictable process statuses for scripts.

Version one must remain a local, best-effort convenience utility. The face tree
is the source of truth; comparison is read-only; warnings keep the user
informed without discarding useful results; maintenance retains safely completed
changes; and the sole permitted network activity is announced acquisition of
missing DeepFace assets. The application must be installable as a Python
package while keeping the locked source checkout as the reproducible reference
environment.

## Solution

Deliver Faceledger as an installable local Python command-line application with
comparison, cache build, cache rebuild, and recoverable model-specific cache
trash commands. Both the installed `faceledger` launcher and module execution
route call the same application entry point.

A comparison accepts exactly one source path and one target root. The source is
either a supported image file selected directly by the user or a single-person
folder anchored by exact lowercase `folder.jpg`. Target comparison includes the
selected root and is recursive by default. Results contain only threshold-
qualified candidate matches ordered by ascending cosine distance, with the
canonical result identity path as a deterministic tie-breaker.

Cache maintenance is explicit and operates on one model at a time. Build
creates missing or incompatible entries, rebuild safely replaces every in-scope
entry, and trash moves exact selected-model cache entries into recoverable,
manifest-backed application trash. Maintenance is limited to the selected root
unless recursion is explicitly requested.

Standard output carries completed comparison results or maintenance summaries.
Standard error carries live diagnostics, transient interactive progress,
warning summaries, failure detail, and trash recovery locations. Requested
result and log files are additional artifacts rather than substitutes for
normal terminal output. Process statuses distinguish success, operation
failure, command-line misuse, and cancellation.

Version one supports exactly Facenet512 and ArcFace under the qualified
DeepFace 0.0.100, CPython 3.12, TensorFlow 2.21, and tf-keras 2.21 runtime.
Facenet512 is the default. RetinaFace detection and alignment remain fixed parts
of the vector profile.

## User Stories

1. As a collection owner, I want an installed `faceledger` command, so that I can use Faceledger without invoking an internal Python file.
2. As a collection owner, I want module execution to provide the same application, so that source checkouts and installed packages have a standard Python entry route.
3. As a collection owner, I want top-level help, so that I can discover commands and options without external documentation.
4. As a collection owner, I want a top-level version option, so that I can identify the installed Faceledger package when troubleshooting.
5. As a collection owner, I want to select one standalone source image, so that I can search for a person without reorganizing the source file.
6. As a collection owner, I want to select one single-person source folder, so that several images of the same person can represent the source identity together.
7. As a collection owner, I want the CLI to infer source image versus source folder from the resolved path, so that I do not need a second source-type option.
8. As a collection owner, I want a directly selected source to be a regular file, so that special files and invalid paths are rejected before recognition.
9. As a collection owner, I want direct source support determined from image content rather than filename extension, so that valid extensionless or unusually named images work.
10. As a collection owner, I want JPEG, PNG, and one-frame static WebP accepted as supported image files, so that common local formats work.
11. As a collection owner, I want unidentified, corrupt, unreadable, unsupported, and animated WebP source files rejected before CORE dispatch, so that unsuitable direct input does no recognition work.
12. As a collection owner, I want CLI validation to inspect but never decode into a replacement, convert, or rewrite my source, so that the original file remains untouched.
13. As a collection owner, I want passing source-format validation to remain distinct from having a usable face image, so that no-face and multi-face failures are still reported accurately by recognition.
14. As a collection owner, I want a source folder to require exact lowercase `folder.jpg`, so that its single-person meaning is deterministic.
15. As a collection owner, I want source-folder processing limited to that folder root, so that descendants cannot silently alter the source identity.
16. As a collection owner, I want every unusable source-folder image reported, so that I can understand a weakened or failed source identity.
17. As a collection owner, I want an unusable structural anchor to leave the source folder usable when another recognized image works, so that `folder.jpg` remains structural rather than mandatory evidence.
18. As a collection owner, I want a source identity with no usable face to stop comparison before target work, so that partial target activity cannot be mistaken for a meaningful search.
19. As a collection owner, I want a standalone source vector calculated freshly on every comparison, so that nearby cache files cannot replace my direct selection.
20. As a collection owner, I want compatible source-folder caches reused by default, so that repeat comparisons avoid unnecessary CPU work.
21. As a collection owner, I want source-folder vectors calculated transiently when no cache is reused, so that comparison never writes cache state.
22. As a collection owner, I want to select a target root positionally, so that every comparison clearly states its search boundary.
23. As a collection owner, I want the target root itself included in comparison, so that identities stored directly in it are not missed.
24. As a collection owner, I want target comparison recursive by default, so that an ordinary search covers the complete selected hierarchy.
25. As a collection owner, I want to disable target recursion explicitly, so that I can restrict a comparison to the selected root.
26. As a collection owner, I want exact lowercase `folder.jpg` to classify a target folder as single-person, so that one folder represents one target identity predictably.
27. As a collection owner, I want differently cased or formatted anchor lookalikes to lack anchor meaning, so that folder classification remains unambiguous.
28. As a collection owner, I want recognized folder images in an anchored folder to contribute to one folder vector, so that several views represent the same target identity.
29. As a collection owner, I want each numbered face image in an unanchored multi-person folder to remain an independent target identity, so that distinct people are never averaged together.
30. As a collection owner, I want sparse and unordered one-digit face numbers accepted, so that face-tree files do not require artificial renumbering.
31. As a collection owner, I want miscellaneous folders skipped even when they contain NPY files, so that cache files alone cannot invent identities.
32. As a collection owner, I want JPEG, PNG, and static WebP recognized under the face-tree naming rules, so that target and maintenance discovery are consistent.
33. As a collection owner, I want animated WebP face-tree images warned about and skipped, so that Faceledger never silently chooses a frame.
34. As a collection owner, I want every processed face image to contain exactly one detected face, so that Faceledger never guesses which face I intended.
35. As a collection owner, I want every usable image in a single-person folder weighted equally in a normalized centroid, so that no image receives implicit priority.
36. As a collection owner, I want unusable target images warned about and skipped, so that one bad item does not erase useful results elsewhere.
37. As a collection owner, I want an anchored target folder to remain usable from its remaining images after partial failures, so that best-effort processing preserves useful identities.
38. As a collection owner, I want discovered symlinks skipped with warnings, so that traversal does not unexpectedly escape or alias the face tree.
39. As a collection owner, I want explicitly selected source and root paths resolved once, so that their meaning remains stable during an operation.
40. As a collection owner, I want files that disappear or change during traversal handled best-effort, so that a live face-tree view can still produce useful work.
41. As a collection owner, I want concurrent read-only comparisons allowed, so that independent searches need no global lock.
42. As a collection owner, I want overlapping maintenance documented as unsupported, so that the live face-tree consistency boundary is honest.
43. As a collection owner, I want compatible selected-model caches reused by default, so that hybrid cached and uncached trees remain efficient.
44. As a collection owner, I want to disable all cache reuse for one comparison, so that current images produce transient vectors without modifying persistent caches.
45. As a collection owner, I want an invalid cache entry warned about and bypassed during comparison, so that useful comparison continues without silent repair.
46. As a collection owner, I want cache compatibility limited to numeric vectors with the selected model's expected dimensions, so that structurally invalid data is never compared.
47. As a collection owner, I want cache model suffixes matched case-sensitively, so that ambiguous or unrelated files remain untouched.
48. As a collection owner, I want compatible cache entries to make no freshness or provenance promise, so that explicit rebuild remains my responsibility after source changes.
49. As a collection owner, I want Facenet512 selected by default, so that ordinary commands have a settled profile.
50. As a collection owner, I want ArcFace available explicitly, so that I can use the second qualified version-one model and its cache.
51. As a collection owner, I want lowercase model spelling at the CLI, so that command syntax is conventional and predictable.
52. As a collection owner, I want CLI model names translated to canonical CORE model names, so that frontend spelling does not weaken vector-profile compatibility.
53. As a collection owner, I want RetinaFace detection and alignment fixed, so that cache compatibility does not depend on hidden user-selectable extraction options.
54. As a collection owner, I want Facenet512's default cosine threshold of 0.30, so that its candidate shortlist uses the qualified baseline.
55. As a collection owner, I want ArcFace's default cosine threshold of 0.68, so that its candidate shortlist uses the qualified baseline.
56. As a collection owner, I want to override comparison threshold with a finite value from zero through two inclusive, so that I can make a shortlist stricter or broader.
57. As a collection owner, I want malformed and out-of-range threshold options rejected as command-line errors, so that invalid values never begin target traversal.
58. As a collection owner, I want only targets at or below the active threshold returned, so that results remain a candidate shortlist rather than all comparisons.
59. As a collection owner, I want candidate matches ordered by ascending cosine distance, so that the most visually similar candidate appears first.
60. As a collection owner, I want exact-distance ties ordered by canonical result identity path, so that output is stable across runs and frontends.
61. As a collection owner, I want the tie-breaker to use target-root-relative identity paths rather than machine-specific absolute paths, so that determinism remains portable.
62. As a collection owner, I want results described as candidate matches rather than verified identities, so that visual similarity is not overstated.
63. As a collection owner, I want an empty candidate list to be successful and say `No matches found`, so that absence of candidates is not confused with failure.
64. As a collection owner, I want result identity paths relative to the selected target root with original casing preserved, so that duplicate names in different branches remain distinguishable.
65. As a collection owner, I want the selected source identity excluded when it lies inside the target tree, so that Faceledger does not return the source as its own candidate.
66. As a collection owner, I want comparison to remain read-only with respect to the face tree and vector cache, so that searching cannot create, replace, or remove cache entries.
67. As a collection owner, I want an explicit cache build command, so that I can prepare reusable selected-model vectors deliberately.
68. As a collection owner, I want cache build to retain compatible entries, so that preparation avoids unnecessary recalculation.
69. As a collection owner, I want cache build to replace missing or structurally incompatible entries after warning, so that broken caches can be repaired explicitly.
70. As a collection owner, I want an explicit cache rebuild command, so that I can deliberately refresh every in-scope selected-model identity.
71. As a collection owner, I want rebuild to preserve an existing entry until a replacement is fully calculated and safely persisted, so that failed refresh does not destroy useful data.
72. As a collection owner, I want build and rebuild to use the same discovery and folder-vector rules as comparison, so that persistent and transient vectors mean the same thing.
73. As a collection owner, I want cache maintenance limited to the selected root by default, so that state-changing work does not unexpectedly affect descendants.
74. As a collection owner, I want recursive maintenance to require an explicit option, so that broad changes are deliberate.
75. As a collection owner, I want each maintenance invocation to select exactly one model, so that other model caches remain untouched.
76. As a collection owner, I want individual maintenance failures warned about while remaining items continue, so that one problem does not waste successful work.
77. As a collection owner, I want an explicit cache trash command, so that cleanup is intentional and scriptable.
78. As a collection owner, I want trash to select only exact selected-model cache suffixes, so that arbitrary NPY files and other model caches are never moved.
79. As a collection owner, I want trash to require no confirmation prompt, so that an explicit recoverable command remains automation-friendly.
80. As a collection owner, I want an empty trash selection to succeed without creating recovery clutter, so that a no-op remains harmless.
81. As a collection owner, I want the complete trash plan persisted before movement begins, so that recovery information exists if interruption occurs.
82. As a collection owner, I want each attempted trash entry recorded as moved or failed, so that any remaining planned entry is visibly unconfirmed.
83. As a collection owner, I want collision-safe UTC trash action identifiers, so that one cleanup never overwrites another.
84. As a collection owner, I want cross-filesystem trash to verify the copied destination before removing the source, so that failed moves preserve original caches.
85. As a collection owner, I want trash stored below the XDG application data root, so that recovery follows mainstream Linux conventions.
86. As a collection owner, I want recovery directory and manifest locations printed on standard error, so that I can manually inspect or restore moved caches.
87. As a collection owner, I want standard output reserved for completed comparison results and maintenance summaries, so that diagnostics do not corrupt useful output.
88. As a collection owner, I want every diagnostic to carry severity, category, stable code, path when applicable, and readable message, so that I know what failed and where to look.
89. As a collection owner, I want diagnostics streamed as they arise and retained in operation outcomes, so that terminal feedback is immediate while logs remain complete.
90. As a collection owner, I want missing model assets announced before dependency acquisition begins, so that Faceledger's only network activity is visible in advance.
91. As a collection owner, I want streamed diagnostics rendered only once, so that long runs are not followed by repetitive terminal output.
92. As a collection owner, I want callback failures to stop rather than disappear silently, so that presentation failure cannot masquerade as a clean operation.
93. As a collection owner, I want item-level warnings not to make an otherwise completed operation unsuccessful, so that best-effort results remain useful.
94. As a collection owner, I want a prominent final warning summary, so that problems remain noticeable after a long operation.
95. As a collection owner, I want interactive progress on standard error, so that I can see completed work without contaminating results.
96. As a collection owner, I want progress shown as one transient line with completed count and current path, so that feedback remains informative without scrolling one line per item.
97. As a collection owner, I want progress suppressed automatically when standard error is not a terminal, so that redirected diagnostics remain clean.
98. As a collection owner, I want an explicit no-progress option, so that I can suppress interactive progress without suppressing diagnostics.
99. As a collection owner, I want no runtime estimate or percentage claim, so that progress does not imply unknowable total work.
100. As a collection owner, I want Ctrl+C to request cancellation at safe item boundaries, so that persistent work is not interrupted inside an unsafe replacement.
101. As a collection owner, I want repeated Ctrl+C signals to remain graceful, so that impatience does not bypass the safety boundary.
102. As a collection owner, I want cancelled comparison to suppress partial candidates, so that incomplete data is never presented as a completed result.
103. As a collection owner, I want safely completed maintenance changes retained after cancellation, so that interruption does not roll back valid work.
104. As a collection owner, I want maintenance summaries to identify successful, incomplete, or cancelled status, so that CORE completeness and user-visible completion are not conflated.
105. As a collection owner, I want warnings reported separately from successful maintenance status, so that useful best-effort completion remains successful while imperfect items stay visible.
106. As a collection owner, I want every maintenance outcome summarized on standard output, so that safely completed counts remain available after incomplete or cancelled work.
107. As a collection owner, I want model, root, scope, and counts stated once in a maintenance summary, so that output is useful without repeating invariant information line by line.
108. As a collection owner, I want an optional result file containing the complete human-readable comparison result, so that I can retain or share the same result shown on standard output.
109. As a collection owner, I want an optional troubleshooting log containing current CORE metadata, status, counts, and diagnostics without ranked matches or progress, so that failures can be investigated separately from results.
110. As a collection owner, I want result and log files created only when requested, so that ordinary comparison creates no frontend artifacts.
111. As a collection owner, I want existing regular artifact files overwritten but missing parent directories left uncreated, so that destinations remain explicit and predictable.
112. As a collection owner, I want result and log destinations to identify different files, so that one requested artifact cannot overwrite the other.
113. As a collection owner, I want conflicting artifact paths rejected before comparison, so that expensive work does not begin for an impossible request.
114. As a collection owner, I want artifact writes attempted only after comparison traversal finishes, so that a bad output destination cannot interrupt recursive comparison work.
115. As a collection owner, I want any requested artifact failure to return operation failure while preserving a completed stdout result, so that the useful comparison is not discarded.
116. As a collection owner, I want no result file written for failed or cancelled comparison, so that partial work is never recorded as a result.
117. As a collection owner, I want a requested log attempted even when direct-source validation fails before CORE dispatch, so that early failures remain troubleshootable.
118. As a collection owner, I want early-validation logs to record unavailable CORE metadata and zero processed counts, so that they never invent work that did not occur.
119. As a script author, I want status 0 for completed success including warnings, no matches, and maintenance no-ops, so that useful completion is automation-friendly.
120. As a script author, I want status 1 for syntactically valid commands that fail validation, operation, requested output, or unexpected execution, so that failure is distinguishable from no matches.
121. As a script author, I want status 2 for grammar, choices, malformed or out-of-range option values, and conflicting options, so that command-line misuse follows argparse convention.
122. As a script author, I want status 130 for user cancellation, so that interruption remains distinguishable from ordinary operation failure.
123. As a collection owner, I want unexpected failures rendered concisely without a traceback, so that ordinary CLI use is not overwhelmed by internal details.
124. As a privacy-conscious collection owner, I want images and vectors processed locally with no uploads or telemetry, so that Faceledger respects the local processing boundary.
125. As a privacy-conscious collection owner, I want cache and trash data described as ordinary sensitive local files, so that local processing is not mistaken for encryption or managed retention.
126. As a Linux user, I want CPU support across the qualified mainstream glibc x86-64 envelope, so that the platform promise is based on representative release testing.
127. As a Linux user, I want the locked CPython 3.12.13 checkout to remain the authoritative reproducible installation, so that exact dependencies and interpreter are repeatable.
128. As a Linux user, I want a local uv tool installation route, so that the launcher can be installed persistently without activating the repository environment.
129. As a Linux user, I want the qualified tool-install command to select CPython 3.12.13 explicitly, so that it matches release qualification.
130. As a Linux user, I want bare local tool installation supported with any compatible Python 3.12, so that convenience remains available with an honest qualification caveat.
131. As a release maintainer, I want source and wheel distributions to build successfully, so that installability is demonstrated without requiring publication.
132. As a release maintainer, I want both entry routes smoke-tested from an isolated built package, so that packaging does not work only from the repository tree.

## Implementation Decisions

- Use standard-library argparse with required command and subcommand parsers. The public grammar is comparison with positional source and target root; cache build, rebuild, and trash with positional maintenance root; and a top-level version option.
- Comparison options are lowercase model choice, threshold override, no-cache, no-recursive, result-file, log-file, and no-progress. Maintenance options are lowercase model choice, recursive, and no-progress.
- CLI model spelling maps `facenet512` to canonical `Facenet512` and `arcface` to canonical `ArcFace`. No other recognition model is accepted in version one.
- Validate CLI option choices, threshold syntax and range, and conflicting artifact destinations before operation dispatch. These are parser failures with status 2. Filesystem, source-content, dependency, operation, and artifact failures use status 1.
- Resolve the selected source once. When it is a directory, populate only the CORE source-folder request field. When it is a file, populate only the source-image field after CLI content validation.
- Use Pillow to inspect a directly selected regular source file. Require detected JPEG, PNG, or WebP content; reject animated or multi-frame WebP; verify readable image structure; close the image; and pass the unchanged original path to CORE. Filename extension has no role.
- Keep direct-source supported-format validation at the CLI boundary for version one. The public CORE comparison operation remains permissive for other Python callers; consolidating the invariant there is deferred.
- Preserve the CORE's existing face-tree discovery, folder classification, normalized centroid, cache compatibility, best-effort warning, live traversal, symlink, source exclusion, maintenance safety, trash manifest, XDG, and local-processing contracts as normative behavior.
- Add a deterministic secondary candidate sort key using the canonical result identity path after cosine distance. Do not use the absolute target path or original user spelling.
- Extend all public comparison and maintenance operations with an optional diagnostic callback. Retain each diagnostic in its outcome and invoke the callback once as the diagnostic arises. Missing-asset notices must be emitted before DeepFace may acquire the asset.
- Do not swallow diagnostic callback exceptions. Allow them to propagate to the CLI, which reports a concise presentation failure and returns status 1. Progress callback failures follow the same non-silent principle.
- Keep parsing, CLI-only validation, request construction, cancellation coordination, operation dispatch, artifact sequencing, unexpected-error translation, and final process-status selection in the CLI application adapter.
- Keep terminal-specific live diagnostic, transient progress, comparison summary, maintenance summary, warning summary, and recovery-location rendering in a dedicated console presentation component.
- Keep persistent human-readable comparison-result and troubleshooting-log formatting in the existing presentation component. Neither presentation component owns domain behavior.
- The installed launcher and module route independently call the same CLI main callable. The callable accepts an optional argument sequence and returns the operation status; test-only dependency injection may provide a recognition adapter and text streams without changing user syntax.
- Install an application SIGINT handler only while an operation runs. Every SIGINT sets the same cancellation flag. Pass that flag through the CORE cancellation callback, allow the operation to stop at its next safe boundary, and restore the previous handler afterward.
- Render every diagnostic live through the callback. Clear a visible progress line before a diagnostic and do not replay outcome diagnostics afterward. The outcome remains the source for requested logs and warning counts.
- Enable progress only when standard error is interactive and no-progress is absent. Render one carriage-returned line with completed count and current path, clear it before diagnostics and final output, and provide no percentage or estimate.
- Comparison standard output contains resolved source, resolved target root, canonical model, active threshold, and either a ranked candidate table or `No matches found`. An unsuccessful or cancelled comparison writes no result body to standard output.
- Maintenance standard output is produced whenever a maintenance outcome exists. Derive public status as cancelled when CORE completeness is false, incomplete when the outcome is otherwise unsuccessful, and successful otherwise. Warnings do not change successful status.
- State maintenance operation, status, resolved root, canonical model, recursion scope, and relevant safely completed counts once. Build reports created and retained; rebuild reports rebuilt; trash reports moved. Diagnostics and warning summaries remain on standard error.
- When trash creates an action, render recovery directory and manifest locations on standard error. A no-op creates and reports neither.
- Detect result/log collisions before source validation or comparison. Treat equal normalized destinations, aliases, and existing hard links as the same file.
- Overwrite requested regular artifact files using UTF-8 and require their parent directories to exist. Do not create missing parents. Attempt artifact writes only after the comparison operation returns.
- Always present a completed successful comparison on standard output even if a subsequent requested artifact write fails. Make every requested artifact failure an error and final status 1.
- Write a result artifact only for a successful completed comparison. Attempt a requested log for successful, failed, or cancelled CORE outcomes and for CLI direct-source validation failure.
- An early CLI-validation log records unavailable CORE metadata, unsuccessful status, zero target and candidate counts, and the CLI validation diagnostic. If logging also fails, render both failures once and return status 1.
- Keep the troubleshooting log at the current metadata contract: source, target root, model, threshold when available; status; target and candidate counts; diagnostic count and diagnostics. Do not include ranked candidates, progress events, CLI version, recursion, or cache-reuse choices.
- Translate unexpected exceptions outside structured operation and callback outcomes into one concise internal-error diagnostic without a traceback and status 1.
- Package the existing flat pure-Python module with `uv_build` constrained to the compatible 0.11 minor line, configure the repository root as the module root, and set uv package mode true.
- Declare the installed launcher entry point and module execution shim. Read the top-level version from installed package metadata; retain project version 0.5.0 until a separate release step changes it.
- Keep the locked checkout authoritative through the pinned CPython 3.12.13 interpreter, project metadata, and lockfile. Support local tool installation from the project; recommend explicit CPython 3.12.13 and document that bare installation may select another compatible 3.12 patch.
- Build source and wheel distributions as acceptance evidence. Publishing, registry configuration, and native packaging are not part of this delivery.
- Add the domain term `Supported image file` to distinguish CLI format acceptance from the existing `Usable face image` recognition concept.
- No new ADR is required: the resolved decisions are frontend contracts or reversible implementation choices, not surprising hard-to-reverse architectural trade-offs.

## Testing Decisions

- Test observable behavior at the highest stable seams. The primary seam is the public CLI callable and installed process interface; the supporting seam is the existing public comparison and maintenance operation boundary where diagnostic streaming and canonical result ordering are application contracts.
- Use deterministic recognition adapters only at the external DeepFace boundary. Do not mock internal CLI, console, presentation, comparison, maintenance, or trash collaborators.
- Drive CLI tests with argument sequences, temporary face trees, injected text streams, and deterministic recognition. Assert returned status, stdout, stderr, requested artifacts, cache effects, and trash effects without asserting private call order.
- Preserve the repository's existing public-operation tests as prior art for comparison requests/outcomes, best-effort warnings, cache build/rebuild, trash manifests, progress, cancellation, and presentation.
- Test parser help, top-level version, required commands, both nested maintenance levels, lower-case model choices, threshold boundaries, recursion switches, cache-reuse switch, no-progress, and status 2 failures.
- Test direct source validation by content with extensionless and misleadingly named JPEG/PNG/WebP, unsupported formats, unidentified or corrupt files, unreadable paths, directories, static WebP, and animated WebP. Assert that rejected content performs no CORE recognition work.
- Test comparison dispatch for standalone and folder sources, Facenet512 and ArcFace mapping, default and overridden thresholds, recursive and single-root target scope, cache reuse choices, no matches, warnings, and operation errors.
- Test exact-distance candidate ties where traversal order differs from canonical result identity path order.
- Test every public operation's diagnostic callback for early validation errors, item warnings, cancellation, and first-use asset notices. Assert callback order matches outcome order, each notice is emitted once, and callback exceptions propagate.
- Test interactive progress with a TTY-like error stream, non-interactive suppression, explicit suppression, clearing around live diagnostics, and final line cleanup. Do not assert terminal width-dependent decoration.
- Test SIGINT during comparison, build, rebuild, and trash. Assert safe cancellation status 130, no partial comparison result, completed maintenance effects retained, manifest state preserved, repeated signals remain graceful, and the previous signal handler is restored.
- Test comparison stdout and stderr separation, resolved header fields, ranked results, no-match text, diagnostics, final warning summary, and concise unexpected-error rendering without exact spacing beyond the semantic contract.
- Test maintenance summaries for successful, warning-bearing, incomplete, and cancelled outcomes. Assert one model/root/scope statement, relevant counts, stdout availability whenever an outcome exists, diagnostic separation, and trash recovery paths on stderr.
- Test artifact path equality, symlink aliases, existing hard links, overwrite behavior, missing parents, result failure, log failure, simultaneous failures, and the rule that write failures occur after recursive comparison work.
- Test requested results for successful matches, no matches, warning-bearing success, failure, and cancellation. Test requested logs for successful CORE outcomes, CORE failure, cancellation, CLI source rejection, artifact failure, and unavailable metadata.
- Test the shell launcher and module route from the locked checkout. Build source and wheel distributions, install the built package into an isolated environment, and smoke-test both entry routes there.
- Test qualified local tool installation with CPython 3.12.13 and the installed launcher. Also verify bare local tool installation selects an interpreter satisfying Python 3.12, while documenting that only the exact command matches release qualification.
- Run the repository's authoritative check script after Python changes and again immediately before commit or push. It must cover Ruff lint, Ruff formatting, strict mypy, and unit tests without weakening configuration or adding broad suppressions.

## Out of Scope

- Identity verification, confidence or accuracy claims, and presentation of candidate matches as confirmed identities.
- Recognition models other than Facenet512 and ArcFace for version one.
- User-selectable detector backend, alignment, distance metric, GPU execution, or vector-profile components.
- Comparison-time cache creation, repair, replacement, refresh, trash, or other persistent face-tree mutation.
- Multi-person source folders, recursive source-folder discovery, and miscellaneous source folders.
- JSON or any other machine-readable result schema.
- A debug option or routine tracebacks for expected or unexpected CLI failures.
- A graphical frontend or stable GUI integration contract.
- Permanent cache deletion, application-managed restore, conflict-aware restoration, and all-model maintenance actions.
- A central catalogue, database, provenance index, freshness tracker, scan history, or filesystem snapshot.
- Inter-process maintenance locks or support for overlapping maintenance.
- Automatic parent-directory creation for requested result or log files.
- Faceledger-managed model-asset downloads, asset bundling, or a second asset cache.
- Model-weight redistribution beyond the already reviewed first-use dependency acquisition behavior.
- Encryption, managed permissions, retention enforcement, secure erasure, or a managed biometric vault.
- GPU, ARM64, musl/Alpine, NixOS-native, immutable-system, Windows, or macOS runtime support.
- Publishing to PyPI or another registry, hosted release automation, native executables, distribution packages, or GUI installers.
- A version bump from 0.5.0; release versioning is a separate step.
- Exact byte-for-byte terminal layout compatibility. Streams, required fields, ordering, statuses, and counts are stable; harmless spacing refinements are permitted.
- Moving content-based standalone-source validation into CORE; that remains a possible future cleanup.

## Further Notes

- This document is the final version-one source of truth. The earlier core
  specification and late model-scope amendment remain useful for tracing how
  the project reached this design, but future implementation and acceptance
  work should cite this specification.
- The implemented CORE overview remains useful evidence when checking whether
  code matches this specification, but this specification controls desired
  version-one behavior when documentation differs.
- CORE `complete` records whether processing reached its own concluded state;
  the CLI's public maintenance status is intentionally user-oriented. A
  complete but unsuccessful CORE outcome renders as incomplete, while
  `complete=False` renders as cancelled.
- Best-effort does not mean silent failure. Individual item problems are
  warnings and useful work continues; failures that prevent meaningful
  continuation remain unsuccessful and visible.
- Comparison may cause DeepFace to persist missing dependency-owned model
  assets after announcing them. That acquisition is outside the read-only
  face-tree and vector-cache boundary.
- The locked checkout is the reproducibility authority. A local uv tool install
  resolves from package metadata rather than reproducing the repository's
  complete transitive lock, even when it uses the qualified interpreter.
