# Faceledger Version One Specification

Approved on 22 July 2026. The implementation-planning parent is Beads issue
`fl-repo-60a`. This file is the durable, human-readable snapshot of that
approved specification.

## Problem Statement

People with local photo collections need a practical way to find plausible appearances of one person across a user-managed folder tree. Existing face-recognition libraries can calculate embeddings, but they do not provide Faceledger's folder conventions, reusable model-specific vector cache, ranked candidate workflow, recoverable maintenance, or clear distinction between similarity and verified identity. Users also need malformed images, unreadable descendants, and stale or invalid cache files to remain visible without causing an otherwise useful operation to stop.

The first version must preserve the user's face tree as the source of truth, keep comparison read-only, process sensitive images and vectors locally, and work predictably on mainstream glibc-based x86-64 Linux without claiming broader platform, identity-verification, or model-licensing guarantees than have been established.

## Solution

Provide Faceledger as a local command-line application that accepts exactly one source identity, compares it with target identities discovered in a selected target root, and returns only threshold-qualified candidate matches ordered by ascending cosine distance. A source identity may be an arbitrary standalone image or a single-person folder anchored by exact lowercase `folder.jpg`. Target folders are classified from their recognized filenames as single-person, multi-person, or miscellaneous, and result identity paths remain relative to the selected target root.

Faceledger uses a fixed DeepFace 0.0.100 vector profile with the selected recognition model, RetinaFace detection, and alignment enabled. Facenet512 is the default, all eleven recognition models exposed by that DeepFace release are supported, and each model uses its pre-tuned cosine threshold unless the user supplies a finite override from zero through two inclusive.

Comparison may reuse compatible model-specific NPY entries but never changes the vector cache. Separate cache build, cache rebuild, and trash actions own persistence. Maintenance is limited to the selected root unless recursion is explicitly requested, operates on one model at a time, and continues best-effort after individual item failures. Trash is recoverable and manifest-backed rather than permanent deletion.

Results, diagnostic notifications, and progress notifications remain conceptually distinct so the CLI can render clean human-readable output without making terminal prose the application's only behavioral representation. Required-input and fatal source failures remain operation errors; target-local and maintenance-item failures become warnings with stable diagnostic information.

## User Stories

1. As a collection owner, I want to select one standalone source image, so that I can look for plausible appearances of that person without reorganizing the source file.
2. As a collection owner, I want to select one single-person source folder, so that several photos of the same person can represent the source identity together.
3. As a collection owner, I want Faceledger to reject ambiguous or missing source selection, so that every comparison has exactly one source identity.
4. As a collection owner, I want a source folder to require exact lowercase `folder.jpg`, so that its single-person meaning is deterministic.
5. As a collection owner, I want source-folder processing to stay within that folder, so that descendants cannot silently change the source identity.
6. As a collection owner, I want an arbitrary source image to be exempt from face-tree filename conventions, so that I can compare an image from anywhere.
7. As a collection owner, I want a standalone source image to be calculated freshly for each comparison, so that unrelated nearby cache files cannot replace the selected image.
8. As a collection owner, I want source-folder cache reuse enabled by default, so that repeat comparisons avoid unnecessary CPU work.
9. As a collection owner, I want missing source-folder vectors to remain transient during comparison, so that comparison never modifies my vector cache.
10. As a collection owner, I want an unusable source identity to stop the comparison before target work, so that no result is presented without a meaningful source vector.
11. As a collection owner, I want every unusable image in a source folder reported before the fatal summary, so that I can find each file that prevented or weakened source construction.
12. As a collection owner, I want to choose a target root, so that I control which face tree participates in a comparison.
13. As a collection owner, I want the target root itself included in the operation, so that identities stored directly in it are not missed.
14. As a collection owner, I want target comparison recursive by default, so that a hierarchy can be searched without selecting each folder separately.
15. As a collection owner, I want a single-target-folder mode, so that I can deliberately restrict comparison to the selected target root.
16. As a collection owner, I want exact lowercase `folder.jpg` to classify a folder as single-person, so that one folder represents one target identity predictably.
17. As a collection owner, I want differently cased or formatted lookalikes such as `Folder.JPG` or `folder.png` not to gain anchor meaning, so that classification stays unambiguous.
18. As a collection owner, I want numbered folder images to contribute to an anchored single-person folder, so that several views of that identity improve its folder vector.
19. As a collection owner, I want a numbered folder image without `folder.jpg` not to create a single-person folder, so that partial naming conventions do not misclassify scaffolding.
20. As a collection owner, I want each numbered face image in an unanchored multi-person folder to remain an independent target identity, so that distinct people are never averaged together.
21. As a collection owner, I want sparse and unordered one-digit face numbers accepted, so that filenames do not require artificial renumbering.
22. As a collection owner, I want JPEG, PNG, and static WebP face images recognized with the agreed case rules, so that common local image formats work consistently.
23. As a collection owner, I want animated WebP excluded, so that Faceledger does not silently choose an arbitrary frame.
24. As a collection owner, I want miscellaneous folders skipped even when they contain NPY files, so that cache files alone cannot invent target identities.
25. As a collection owner, I want every usable image in a single-person folder weighted equally in a normalized centroid, so that no image receives implicit priority.
26. As a collection owner, I want an unusable structural anchor to leave the folder usable when other recognized images work, so that `folder.jpg` can remain structural rather than mandatory evidence.
27. As a collection owner, I want every processed face image to contain exactly one detected face, so that Faceledger never guesses which face I intended.
28. As a collection owner, I want target-local failures reported and skipped, so that one damaged image does not discard useful results from the rest of the face tree.
29. As a collection owner, I want a single-person target folder to average its remaining usable images after partial failures, so that one bad file does not erase the whole identity.
30. As a collection owner, I want symlinked descendants and files skipped with warnings, so that traversal does not unexpectedly escape or alias the selected face tree.
31. As a collection owner, I want explicitly selected paths resolved once before validation, so that my requested roots and source have stable meanings at operation start.
32. As a collection owner, I want Faceledger to tolerate files changing or disappearing during traversal, so that a live face-tree view can complete best-effort.
33. As a collection owner, I want compatible model-specific vectors reused by default, so that hybrid cached and uncached trees remain efficient.
34. As a collection owner, I want to disable existing-vector reuse for a comparison, so that I can deliberately calculate transient vectors from current images.
35. As a collection owner, I want invalid cache entries warned about and bypassed during comparison, so that useful comparison continues without silently repairing persistent data.
36. As a collection owner, I want cache compatibility limited to numeric vectors of the selected model's expected dimensions, so that structurally unusable NPY data is not compared.
37. As a collection owner, I want cache filenames to contain the fixed lowercase model slug, so that vectors from different recognition models cannot be confused.
38. As a collection owner, I want cache suffix matching to remain case-sensitive, so that ambiguous duplicates and unrelated files are left untouched.
39. As a collection owner, I want Facenet512 selected by default, so that ordinary comparisons have a settled recognition model.
40. As a collection owner, I want to select any recognition model supported by DeepFace 0.0.100, so that I can use a model appropriate to my existing cache or preferences.
41. As a collection owner, I want RetinaFace detection with alignment consistently enabled, so that cached vectors share one predictable vector profile.
42. As a collection owner, I want each model's DeepFace 0.0.100 cosine threshold used by default, so that candidate qualification follows a documented model-specific baseline.
43. As a collection owner, I want to override the match threshold with a finite value from zero through two, so that I can make the candidate list stricter or broader.
44. As a collection owner, I want invalid threshold overrides rejected before target traversal, so that an invalid operation does no expensive or confusing work.
45. As a collection owner, I want only targets within the active match threshold returned, so that the result list remains a shortlist rather than a dump of all comparisons.
46. As a collection owner, I want candidate matches ordered by ascending cosine distance, so that the most visually similar candidate appears first.
47. As a collection owner, I want Faceledger to describe results as candidate matches rather than verified identities, so that similarity is not overstated.
48. As a collection owner, I want an empty result list to be successful and clearly say `No matches found`, so that absence of candidates is not confused with failure.
49. As a collection owner, I want result identity paths relative to the target root with original casing preserved, so that duplicate names in different branches remain distinguishable.
50. As a collection owner, I want the selected source identity excluded when it lies inside the target tree, so that Faceledger does not report the source as its own candidate.
51. As a collection owner, I want comparison to leave the vector cache unchanged, so that a read-only search cannot create, overwrite, or remove persistent entries.
52. As a collection owner, I want no persistent catalogue or directory index, so that my face tree remains the source of truth.
53. As a collection owner, I want a cache build action that creates only missing or incompatible entries for one model, so that I can prepare a tree without refreshing valid entries.
54. As a collection owner, I want a cache build to warn before replacing a corrupt, non-numeric, or wrong-dimension entry, so that the exceptional replacement is visible.
55. As a collection owner, I want a cache rebuild action that recalculates every in-scope entry for one model, so that I can explicitly refresh stale vectors.
56. As a collection owner, I want rebuild to preserve an existing entry unless its replacement is calculated and safely persisted, so that a failed refresh does not destroy usable cached data.
57. As a collection owner, I want cache build and rebuild to use the same classification and folder-vector rules as comparison, so that persistent and transient vectors have consistent meaning.
58. As a collection owner, I want maintenance restricted to the selected root by default, so that state-changing work does not unexpectedly affect descendants.
59. As a collection owner, I want recursion to be explicit for every maintenance action, so that broad cache changes are deliberate.
60. As a collection owner, I want every maintenance action to operate on exactly one selected model, so that caches for other models remain untouched.
61. As a collection owner, I want individual maintenance failures warned about while remaining items continue, so that partial filesystem problems do not waste successful work.
62. As a collection owner, I want a trash action to select only cache entries whose suffix matches the chosen model, so that arbitrary NPY files are never moved.
63. As a collection owner, I want an empty trash selection to succeed without creating an action directory or manifest, so that no-op maintenance leaves no clutter.
64. As a collection owner, I want a complete trash plan recorded before any file moves, so that recovery information exists even if the process is interrupted.
65. As a collection owner, I want every trash manifest entry finalized as `moved` or `failed` when attempted, so that any remaining `planned` entry is visibly unconfirmed.
66. As a collection owner, I want trash action identifiers to use collision-safe UTC timestamps, so that one action never overwrites another.
67. As a collection owner, I want trash to preserve only the relative directories needed for moved files, so that recovery remains understandable without duplicating empty tree structure.
68. As a collection owner, I want cross-filesystem trash to verify a copied destination before removing its source, so that a failed move preserves the original cache entry.
69. As a collection owner, I want cleanup to move entries into recoverable application trash rather than permanently delete them, so that I can recover them manually.
70. As a collection owner, I want application trash under the XDG data location, so that Faceledger follows mainstream Linux desktop conventions.
71. As a collection owner, I want standard output reserved for comparison results, so that warnings and debug information do not corrupt the result presentation.
72. As a collection owner, I want each warning to include severity, category, stable code, affected path when applicable, and a readable reason, so that failures can be understood and acted upon.
73. As a collection owner, I want a prominent final warning summary with relevant counts, so that problems remain noticeable after a long operation.
74. As a collection owner, I want required-input failures reported as operation errors with unsuccessful process status, so that scripts and humans can distinguish failure from a successful empty result.
75. As a collection owner, I want result and diagnostic files created only when explicitly requested, so that ordinary comparisons leave no application data behind.
76. As a collection owner, I want an optional human-readable run log containing metadata, diagnostics, and final counts, so that I can retain troubleshooting information without duplicating the ranked result table.
77. As a collection owner, I want long uncached operations to expose progress outside the result stream, so that I can see advancement without receiving a runtime guarantee.
78. As a collection owner, I want cancellation to stop at a safe item boundary where possible and report an incomplete operation, so that partial comparison data is never presented as complete.
79. As a collection owner, I want safely completed maintenance items retained after cancellation, so that interruption does not roll back valid persistent work.
80. As a privacy-conscious collection owner, I want images and vectors processed locally with no telemetry or uploads, so that Faceledger respects the local processing boundary.
81. As a collection owner, I want missing DeepFace assets acquired through the dependency's normal mechanism only after Faceledger announces it, so that the sole permitted network activity is visible.
82. As a collection owner, I want asset-acquisition failures to include retry and offline guidance, so that dependency bootstrap problems are actionable.
83. As a Linux user, I want CPU support across the declared mainstream glibc x86-64 envelope, so that the support promise is based on a tested ABI and platform boundary.
84. As a Linux user, I want each release validated on representative Debian/Ubuntu, Fedora, and Arch environments, so that broad compatibility is exercised across major ecosystems.
85. As a release maintainer, I want one managed CPython version and a fully locked dependency graph chosen before implementation is declared platform-ready, so that the Linux promise is reproducible.
86. As a release maintainer, I want every supported recognition model and image format exercised against the locked runtime, so that support is proven beyond the Facenet512 happy path.
87. As a release maintainer, I want upstream model and weight licences reviewed before assets are distributed, so that runtime support is not mistaken for redistribution permission.
88. As a collection owner, I want Faceledger to make no provenance or freshness guarantee for structurally compatible cache entries, so that I know when an explicit rebuild is my responsibility.
89. As a collection owner, I want concurrent read-only comparisons allowed but overlapping maintenance documented as unsupported, so that the live face-tree consistency boundary is clear.
90. As a collection owner, I want sensitive cache and trash files described as ordinary plaintext under my filesystem controls, so that local processing is not mistaken for encryption or managed retention.

## Implementation Decisions

- Deliver version one as a local command-line application with four user-visible capabilities: comparison, cache build, cache rebuild, and model-specific trash. Exact command and option syntax remains deferred.
- Keep application operation results, diagnostic notifications, and progress notifications presentation-neutral before CLI rendering. A diagnostic notification includes severity, category, stable code, an affected path when applicable, and a human-readable message.
- Treat the user-managed face tree, recognized filenames, and model-qualified NPY files as the source of truth. Do not introduce a central catalogue, provenance database, or scan index.
- Resolve explicitly selected source and root paths once before validation. Include each selected root in its own operation. Do not follow discovered symlinked directories, face images, or cache files; warn and continue when encountered.
- Recognize `.jpg`, `.jpeg`, `.png`, and static `.webp` case-insensitively for ordinary and numbered face files. Recognize the one-digit `.face#` marker case-insensitively. Reserve single-person anchor power exclusively for exact case-sensitive lowercase `folder.jpg`.
- Classify a folder with exact `folder.jpg` as single-person and produce one folder vector from every usable recognized image in that folder root. Classify an unanchored folder with one or more numbered face images as multi-person and produce one independent vector per usable face image. Treat every other folder as miscellaneous/scaffolding and skip it.
- Treat `folder.jpg` as a structural anchor, not a mandatory vector contributor. If it is unusable, warn and use other usable recognized images. Skip a target or fail a source only when no usable image remains.
- Form a folder vector by L2-normalizing each usable embedding, calculating their equal-weight arithmetic mean, and L2-normalizing the centroid. Do not perform semantic outlier rejection.
- Accept exactly one source: an arbitrary standalone image or a single-person folder. Never reuse a cache entry for a standalone source image. Permit default-on reuse for a source folder. Keep all comparison-generated source vectors transient.
- Require every usable face image to yield exactly one face. Do not choose among multiple faces. Fail immediately for an unusable standalone source; warn per item in a source folder and emit a fatal summary only if the folder produces no usable source vector.
- Pin the vector profile to DeepFace 0.0.100, the selected recognition model, RetinaFace detection, and alignment enabled. Do not expose detector or alignment overrides in version one.
- Support VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace, GhostFaceNet, and Buffalo_L. Use Facenet512 by default.
- Use the fixed lowercase cache slugs `vgg-face`, `facenet`, `facenet512`, `openface`, `deepface`, `deepid`, `dlib`, `arcface`, `sface`, `ghostfacenet`, and `buffalo_l` respectively.
- Use DeepFace 0.0.100's pre-tuned cosine thresholds verbatim: VGG-Face 0.68, Facenet 0.40, Facenet512 0.30, OpenFace 0.10, DeepFace 0.23, DeepID 0.015, Dlib 0.07, ArcFace 0.68, SFace 0.593, GhostFaceNet 0.65, and Buffalo_L 0.55.
- Accept an optional comparison threshold override only when it is finite and in the inclusive cosine-distance range `[0, 2]`. Reject invalid values before target traversal.
- Compare every usable target vector with the source using cosine distance. Include a target identity only when its distance is less than or equal to the active match threshold, then order candidate matches by ascending distance.
- Identify a single-person target by its target-root-relative directory path and a multi-person target by its target-root-relative image path. Preserve original filesystem casing.
- When the source lies in the target tree, exclude its own target identity while keeping other identities in scope. Exclude the complete target identity containing a standalone source image.
- Make comparison recursive by default, with an explicit single-target-folder mode. Make all state-changing maintenance non-recursive by default, with explicit recursive maintenance.
- Enable existing-vector reuse by default for applicable identities. A corresponding cache entry appends `.<cache-model-slug>.npy` to the complete associated image filename. Associate a single-person folder's aggregate cache with `folder.jpg`, even if the anchor did not contribute a usable embedding.
- Consider a cache entry compatible only when its case-sensitive model-qualified filename is correct and it loads as numeric vector data with the selected model's expected dimensions. Do not infer freshness or provenance from content, timestamps, or source images.
- During comparison, warn on an invalid cache entry and calculate the vector transiently without repairing or overwriting the cache. Comparison never creates, overwrites, or removes cache entries.
- Cache build creates entries only when a compatible entry is absent. Treat corrupt, non-numeric, or wrong-dimension entries as missing, warn, and replace them. Leave structurally compatible entries untouched even when their source images may have changed.
- Cache rebuild recalculates every in-scope identity for one selected model. Preserve an existing cache entry until its complete replacement has been calculated and safely persisted. On failure, warn, retain the old entry, and continue.
- Apply identical discovery, classification, face-count, and folder-vector rules to transient comparison calculation, cache build, and cache rebuild.
- Trash discovery matches only the exact case-sensitive suffix for the selected model. If no matching entries exist, report a successful no-op and create neither an action directory nor a manifest.
- Create one collision-safe trash action directory per non-empty cleanup operation under the XDG application data root's `faceledger/trash` subtree. Use a UTC timestamp with microsecond precision as the action ID and append a numeric suffix on collision.
- Before moving any entry, persist one complete trash manifest containing every original and trash-relative path with `planned` status. After each attempt, update the same entry to `moved` or `failed`, including a known failure reason. Treat a remaining `planned` status as unconfirmed.
- Recreate only relative directories needed for moved entries. For cross-filesystem movement, preserve the source until the copied destination has been verified as faithfully representing it. A failed copy or verification leaves the original in place, records failure, warns, and does not stop other entries.
- Resolve the application data root centrally, honoring `XDG_DATA_HOME` and otherwise using the user's `.local/share` data location. Cleanup moves files into application-managed trash; version one does not permanently delete or automatically restore them.
- Validate required source and root inputs before the main operation. Treat missing, non-directory, or wholly unreadable selected roots and failed dependency acquisition as operation errors. After validation, treat descendant and individual-item failures as warnings under best-effort processing.
- Reserve standard output for the resolved comparison header and ranked human-readable result table. Render diagnostic notifications on standard error. Emit a prominent warning summary with relevant counts after a successful operation containing warnings.
- Create result and log files only when explicitly requested. Keep an optional human-readable UTF-8 log separate from a requested result file; include operation metadata, diagnostics, and final counts without duplicating the ranked result table.
- Expose long-running progress separately from result data. On user cancellation, stop further work at a safe item boundary where possible, report the operation as incomplete, do not present a partial comparison as completed, and do not roll back safely completed maintenance items.
- Let DeepFace manage its normal model-weight acquisition and persistence. Announce missing-asset acquisition and translate acquisition failure into an actionable operation error. Do not introduce a parallel Faceledger asset manager.
- Enforce the local processing boundary: never upload images or vectors and send no telemetry. The sole permitted version-one network activity is announced inbound acquisition of missing dependency assets.
- Support CPU execution on mainstream glibc x86-64 Linux. Validate representative Ubuntu LTS, Debian stable, current Fedora, and current Arch environments for each release. Treat other qualifying distributions as expected-compatible rather than release-blocking.
- Choose one application-managed CPython version and fully lock the runtime dependency graph through the Q17 packaging spike before declaring the supported Linux envelope implementation-ready.
- Complete Q30 upstream licence and weight-redistribution review before choosing a distribution format or bundling model assets. Runtime model support does not establish redistribution permission.
- Operate on a live face-tree view without an inter-process lock or snapshot. Allow concurrent read-only comparisons; document overlapping maintenance as unsupported and handle descendants that change after discovery best-effort.

## Testing Decisions

- The single primary testing seam is the public CLI/application operation boundary. Tests request comparison, cache build, cache rebuild, or trash behavior against temporary face trees and assert only observable outcomes: result data, diagnostic and progress notifications, process outcome, requested files, vector-cache effects, and trash contents.
- Most behavior tests use a deterministic recognition adapter at the DeepFace integration boundary. This keeps TDD fast and repeatable while still driving the core operation from its highest stable boundary. Tests must not depend on private helper calls, temporary class layouts, or internal call ordering.
- The same outward operation seam is exercised with the real locked DeepFace runtime for dependency and release qualification. That suite covers RetinaFace with alignment, all eleven supported recognition models, each supported static image format, first-use asset acquisition, and a subsequent offline run with assets already present.
- Comparison tests cover standalone and folder sources, recursive and single-folder targets, exact filename classification, normalized folder vectors, hybrid cached/uncached trees, threshold boundaries, ascending result ordering, source overlap, empty successful results, fatal source failures, and best-effort target warnings.
- Maintenance tests cover non-recursive defaults, explicit recursion, one-model isolation, build versus rebuild semantics, invalid cache replacement, preservation on failed rebuild, model-specific trash discovery, manifest state changes, collisions, cross-filesystem failure safety, cancellation, and successful no-ops.
- Filesystem scenarios include symlinks, permission failures, disappearing descendants, duplicate names, unusual casing, and paths sufficient to demonstrate that result and trash paths remain relative and unambiguous.
- CLI rendering tests verify separation of standard output from standard error, successful versus unsuccessful process status, explicit-only result/log creation, warning summaries, debug behavior, and incomplete cancellation reporting without treating exact internal formatting choices as settled before the CLI contract is specified.
- Platform qualification runs on the published representative Debian/Ubuntu, Fedora, and Arch environments using CPU inference and the final locked CPython and dependency graph. The final wheels' ABI tags establish the supported glibc floor.
- This repository has no executable test suite or implementation precedent yet. The design seed's representative scenarios provide the initial behavioral prior art; implementation slices will establish the testing conventions through TDD.

## Out of Scope

- Identity verification, confidence or accuracy claims, and presentation of candidate matches as confirmed identities.
- A persistent database, catalogue, provenance index, freshness tracker, or comprehensive record of prior face-tree scans.
- Comparison-time creation, repair, refresh, overwrite, or removal of vector-cache entries.
- Recursive source-folder discovery, multi-person source folders, or miscellaneous source folders.
- Permanent cache deletion, application-managed restore, conflict-aware restoration, or all-models maintenance actions.
- Arbitrary NPY deletion or case-insensitive cache-suffix matching.
- Semantic outlier detection for technically valid vectors in a single-person folder.
- JSON or another machine-readable CLI schema and a graphical frontend.
- A stable version-one GUI integration mechanism; only presentation-neutral operation meaning is preserved.
- GPU acceleration, ARM64 or other non-x86 architectures, 32-bit Linux, musl/Alpine, NixOS-native packaging, and host-level packaging for immutable or appliance-style systems.
- Native packages for every Linux distribution or a claim to support every Linux environment.
- Animated WebP processing or frame selection.
- Inter-process maintenance locking, filesystem snapshots, stale-lock recovery, or support for overlapping maintenance actions.
- Encryption, managed permissions, retention enforcement, secure erasure, or a managed biometric vault.
- Fixed runtime guarantees, optimization for tens of thousands of images, or cross-hardware performance commitments.
- A Faceledger-specific model-asset manager or unreviewed bundling of upstream model weights.
- Final command names, option spelling, argument syntax, internal module layout, function/class boundaries, precise atomic-write primitives, cross-filesystem verification mechanics, exact log rendering, and exact interrupt mechanics; these remain implementation-level specification work.

## Further Notes

- The settled architecture decisions pin DeepFace 0.0.100, separate read-only comparison from vector-cache maintenance, retain the face tree as the source of truth, fix RetinaFace plus alignment as part of the vector profile, and define mainstream glibc x86-64 Linux CPU support as the version-one platform envelope.
- Q17 remains unresolved: a packaging spike must choose and lock the managed CPython version, glibc ABI floor, TensorFlow, Keras, OpenCV, RetinaFace, and remaining transitive dependency versions before the platform promise is implementation-ready.
- Q30 remains unresolved: every supported recognition model and RetinaFace require upstream licence and weight-redistribution review before Faceledger chooses a distribution format or bundles assets.
- Facenet512 is the default model. The full supported model set is VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace, GhostFaceNet, and Buffalo_L.
- Comparison and maintenance are best-effort only after their required inputs are validated. Warnings do not make such an operation unsuccessful; an operation error or cancelled operation has a distinct unsuccessful/incomplete outcome.
- Cache entries deliberately provide structural compatibility rather than freshness or provenance. Users rebuild after source-image changes when freshness matters.
- Face embeddings and recoverable trash are sensitive plaintext local data. The local processing boundary prevents uploads and telemetry but does not add encryption, access control, or retention guarantees.

## Acceptance Criteria

Comparison, folder classification, vector reuse, ranked candidate results, cache build/rebuild/trash, diagnostics, cancellation, and local-processing behavior satisfy this specification; Q17 locks and validates the supported Linux runtime across all eleven models and supported static image formats; Q30 resolves licence and weight-redistribution constraints before assets are bundled or distribution implies redistribution rights.
