# Faceledger design seed

## 1. Product sentence

Faceledger compares one source identity with face images found in a user-selected
folder tree by creating or reusing model-specific face vectors, calculating
cosine distances, and returning a ranked shortlist of candidate matches without
building a persistent catalogue of the scanned tree.

## 2. User and operating context

- The intended first version is a local command-line tool.
- Version one officially supports CPU execution on mainstream glibc-based
  x86-64 Linux. Its release-blocking validation spans representative
  Debian/Ubuntu, Fedora, and Arch environments; other qualifying glibc x86-64
  distributions are expected-compatible but not individually certified.
- A graphical frontend is a likely later presentation layer. Version-one
  comparison and maintenance outcomes should therefore remain conceptually
  separate from their human-readable CLI rendering, so a later frontend does
  not have to reinterpret terminal prose as application logic.
- A user selects exactly one source:
  - an arbitrary image file; or
  - a folder representing one person, whose recognized face images are combined
    into one source vector.
- The user also selects a target root folder. The root itself is scanned, followed
  by its descendants unless single-target-folder mode is selected.
- Users commonly retain a model-specific vector cache beside their face images
  and reuse it on later comparison runs. A tree may therefore contain a mixture
  of reusable vectors and newly added images without vectors.
- Comparison runs never create, overwrite, or remove vector-cache files. A
  separate maintenance capability owns vector-cache creation and cleanup.
- Users may move vector-cache files from a selected tree into
  application-managed trash. Cleanup is recoverable rather than permanent.
- Face images are expected to be local files organized according to the naming
  conventions in this seed. A single source image is exempt from those naming
  conventions.
- Faceledger does not upload images or vectors. DeepFace may make an inbound
  network request to acquire a missing detector or recognition-model asset as
  part of dependency initialization.
- Faceledger sends no telemetry. Its product data processing is local; announced
  inbound model-asset acquisition is the only permitted network activity in
  version one.
- The user-managed folder tree and filename conventions are the source of truth.
  Faceledger does not maintain centralized provenance, freshness, or catalogue
  metadata.
- Version one is intended for personal collections typically containing hundreds
  and potentially a few thousand recognized images. Performance at
  tens-of-thousands scale is not a version-one objective.

## 3. Primary user journeys

### Compare from a single source image

1. Select one image file and a target root folder.
2. Faceledger calculates a temporary source vector for the selected model,
   regardless of the existing-vector reuse setting. Comparison has no cache
   saving setting.
3. It discovers and classifies target folders, obtains their target vectors, and
   compares each target vector with the source using cosine distance.
4. It returns a result list containing threshold-qualified candidate matches and
   their distances, ordered from most to least similar.
5. Any source-image failure terminates the run; an ordinary target failure is
   reported as a warning and does not stop the remaining scan.

### Compare from a single-person source folder

1. Select a source folder containing exact lowercase `folder.jpg` and select a
   target root folder.
2. Faceledger applies the single-person folder rules to obtain one averaged
   source vector from all recognized face images in the source folder root.
   It does not descend into source subfolders.
3. Existing-vector reuse applies to this source. Missing vectors are calculated
   transiently and the comparison run never persists them.
4. The target scan and comparison proceed as in the single-image journey.
5. A miscellaneous or multi-person source folder terminates the run with a
   diagnostic equivalent to
   `Select a single-person folder containing exact folder.jpg`.

### Compare a target tree

1. Select a target root and optionally restrict scanning to that folder alone.
2. For each folder in scope, classify it as single-person, multi-person, or
   miscellaneous.
3. Reuse a corresponding model-specific NPY when enabled and available;
   otherwise calculate the required vector.
4. Compare every resulting target vector with the source and add its identity
   and distance to the main result list.

### Maintain the vector cache

1. Select a root folder, model, and maintenance action. The selected folder
   alone is in scope unless recursion is explicitly requested.
2. A cache build discovers recognized face images and creates only missing
   vectors for the selected model without performing a comparison. Compatible
   existing cache entries remain untouched.
3. An explicitly requested cache rebuild recalculates every in-scope vector for
   the selected model and overwrites its existing cache entry.
4. A cleanup action discovers only NPY files whose suffix identifies the
   selected model, records them in a manifest, and moves them to recoverable
   application trash.

### Move vector-cache files to trash

1. Select a root folder. The selected folder alone is in scope unless recursion
   is explicitly requested.
2. Discover model-qualified NPY paths. If none exist, report a successful no-op
   without creating files.
3. Otherwise create a collision-safe UTC-timestamped action directory under the
   application's trash root and record the complete `planned` manifest.
4. Move the discovered NPY files into that action directory, recreating only the
   relative directories needed to hold them and recording each final outcome.

## 4. Domain concepts and glossary

| Term | Meaning |
| --- | --- |
| Source | The single identity against which targets are compared. It is supplied as one arbitrary image or one single-person folder. |
| Target root | The top folder selected for comparison. |
| Maintenance root | The top folder selected for cache build, rebuild, or trash. |
| Face image | A recognized target/source-folder image named `folder.jpg`, single-digit `folder#.jpg`, or `*.face#.ext`, where `#` is one digit. It is usable only when exactly one face is detected and a valid embedding is produced. |
| Folder anchor | The exact, case-sensitive filename `folder.jpg`, supported for compatibility with Windows thumbnail conventions. It alone classifies a folder as single-person. |
| Numbered folder image | `folder0.jpg` through `folder9.jpg`. It may coexist with `folder.jpg`. |
| Numbered face image | An image whose basename ends in `.face0` through `.face9` before its supported image extension. Digits need not start at zero or be consecutive. |
| Single-person folder | A folder containing exact lowercase `folder.jpg`. It represents one identity and produces one averaged vector even when other recognized face images are present. |
| Multi-person folder | A folder with one or more numbered face images but no exact lowercase `folder.jpg`. Each face image represents a distinct identity and produces its own vector. |
| Miscellaneous/scaffolding folder | A folder with neither exact lowercase `folder.jpg` nor a numbered face image. It is skipped, even if it contains NPY files. |
| Model/backend | A DeepFace face-recognition model used to generate and interpret vectors. Facenet512 is the default. |
| Vector profile | DeepFace 0.0.100, the selected recognition model, RetinaFace detection, and enabled alignment; together these define embedding compatibility. |
| Vector/NPY | The model-specific numerical face representation used for cosine-distance comparison. NPY is the persisted representation used by the vector cache. |
| Vector cache | Model-specific NPY vectors persisted beside their associated face images for reuse. Comparison runs do not modify it. |
| Vector-cache maintenance | A dedicated capability for creating or cleaning vector-cache entries outside a comparison run. |
| Folder vector | The equal-weight arithmetic mean of a single-person folder's usable, individually L2-normalized embeddings, L2-normalized again after averaging. |
| Existing-vector reuse | Loading an associated model-specific NPY instead of recalculating it. This is enabled by default. |
| Compatible cache entry | A model-qualified NPY that loads as a numeric vector with the selected model's expected dimensions. This does not guarantee freshness or provenance. |
| Candidate match | A target identity whose cosine distance is less than or equal to the active model-specific match threshold. It is a plausible visual match, not a verified identity. |
| Result list | The candidate matches from one comparison run, ordered from smallest distance (most similar) to largest distance. An empty list is successful. |
| Result identity path | A target identity's path relative to the selected target root: the directory path for a single-person folder or the image path for a multi-person face. Original casing is preserved. |
| Single-target-folder mode | Restricts a comparison to the selected target root rather than its descendants; comparison is recursive by default. |
| Recursive maintenance | An explicit request for a cache build, rebuild, or trash action to include descendants of its selected root; maintenance otherwise processes only that root. |
| Trash action | One recoverable cleanup operation, with its own timestamped folder and manifest. |

`ext` means `.jpg`, `.jpeg`, `.png`, or static `.webp`. Except for the exact
lowercase `folder.jpg` anchor, extensions and numbered `.face#` naming markers
are recognized case-insensitively while actual filenames are preserved in
output. Animated WebP is not supported in version one.

## 5. Behavioural rules and invariants

### Folder classification and vector production

Classification is based on recognized face images, not on the presence of NPY
files.

| Condition in a folder | Classification | Vector production | Result identity |
| --- | --- | --- | --- |
| Exact lowercase `folder.jpg` exists | Single-person | One averaged vector using every usable `folder.jpg`, `folder#.jpg`, and `*.face#.ext` in that folder | Target-root-relative directory path |
| No exact lowercase `folder.jpg`, but one or more `*.face#.ext` exist | Multi-person | One independent vector per usable face image; no averaging | Target-root-relative image path |
| Neither condition holds | Miscellaneous/scaffolding | None | None; folder is skipped |

- Recognition and averaging concern only files in the folder being classified.
- Faceledger resolves explicitly selected source and root paths once before
  validation. During traversal it never follows discovered symlinked directories,
  face images, or cache files; it warns and skips them.
- Faceledger recognizes `.jpg`, `.jpeg`, `.png`, and static `.webp`
  case-insensitively for ordinary and numbered face files. The numbered
  `.face#` marker is also case-insensitive, so `photo.FACE3.PNG` is valid. The
  folder anchor is the deliberate exception: only the exact case-sensitive
  filename `folder.jpg` has the power to classify a folder as single-person;
  `Folder.JPG`, `folder.png`, and `folder.webp` do not.
- A numbered folder image alone does not make a valid single-person folder.
- The exact `folder.jpg` anchor is structural rather than a mandatory vector
  contributor. If it is unusable, Faceledger warns for that image and forms the
  folder vector from any other usable recognized images. The identity is skipped
  as a target—or fails as a source—only when no usable image remains.
- In a multi-person folder, face-number digits may be sparse or unordered and
  each recognized file remains independent.
- For an averaged vector, the persisted NPY is associated with `folder.jpg`.
  Therefore removing the final `.npy` suffix from any Faceledger NPY name
  identifies the first/anchor face image associated with that vector.
- The aggregate cache remains associated with and named after `folder.jpg` even
  when that structural anchor was unusable and did not contribute to the
  calculated folder vector.
- A single-person folder's vector is produced by L2-normalizing every usable
  per-image embedding, taking their equal-weight arithmetic mean, and
  L2-normalizing the resulting centroid. No usable image receives extra weight.

### Source acquisition

| Source kind | Naming rules | Reuse existing NPY | Persistence during comparison | Failure |
| --- | --- | --- | --- | --- |
| Single image | Exempt from folder face-file naming rules | Never | Never; vector is temporary for the run | Terminate immediately |
| Source folder | Must classify as single-person and is not scanned recursively | Enabled by default when a corresponding NPY exists | Never; missing vectors are temporary for the run | Invalid folder or inability to produce a usable source vector terminates the run |

### Existing-vector reuse and generation

| Corresponding model NPY exists? | Reuse enabled? | Behaviour |
| --- | --- | --- |
| Yes | Yes | Load the compatible existing NPY without recalculating the source image or images |
| Yes | No | Recalculate from the applicable face image or images |
| No | Either | Calculate the vector |

- Reuse is enabled by default.
- NPY data is model-specific. A vector for one backend must not be treated as a
  vector for another backend.
- A cache entry is compatible when its model-qualified filename is correct and
  it loads as a numeric vector with the selected model's expected dimensions.
  Faceledger does not infer freshness from timestamps or image content.
- During comparison, an invalid cache entry produces a warning and Faceledger
  calculates the required vector transiently. It does not repair the cache.
- A comparison run never persists or overwrites a calculated vector. Persistence
  belongs exclusively to vector-cache maintenance.
- Cache names append the selected model's fixed lowercase cache slug and `.npy`
  to the complete face-image filename, for example
  `suspect.jpg.arcface.npy`. Reuse and trash discovery match this suffix
  case-sensitively. A differently cased suffix is not a compatible entry and is
  left untouched, avoiding ambiguous duplicate cache entries.

### Comparison and failure handling

- Facenet512 is the default model. The user may select any other supported
  DeepFace face-recognition model and may override that model's default cosine
  threshold.
- A threshold override must be a finite cosine distance from `0` through `2`
  inclusive. Zero admits only zero-distance candidates; two is the loosest
  mathematically meaningful bound. Negative values, values above two, `NaN`,
  and infinities are operation input errors rejected before scanning.
- Face extraction always uses RetinaFace with alignment enabled. Version one
  does not expose detector or alignment overrides.
- Every usable target vector is compared with the source vector using cosine
  distance. A target joins the result list only when its distance is less than
  or equal to the active match threshold.
- The source may be located inside the target tree, but Faceledger excludes its
  own target identity: a source folder is skipped as a target, and a target
  identity containing the standalone source image is skipped in full. Other
  identities in the overlapping tree remain in scope.
- Candidate matches are ordered by ascending cosine distance, which presents the
  most similar candidate first. The user interprets this comparative shortlist;
  Faceledger does not claim to verify identity.
- An empty result list (`No matches found`) is a successful comparison run.
- A source failure is the exceptional fatal condition: Faceledger exits as soon
  as it determines that it cannot produce a usable source vector, because no
  meaningful target comparisons can follow.
- Every face image must yield exactly one detected face. Faceledger never chooses
  the first, largest, or most similar face from a multi-face image.
- A standalone source image with zero or multiple detected faces terminates the
  run. In a source folder, such an image is warned about and excluded; the run
  terminates only if no usable source image remains.
- Every unusable image in a source folder produces its own warning identifying
  that file and the known reason. If no usable image remains, Faceledger then
  emits one fatal summary identifying the source folder and the examined and
  usable image counts, and exits before scanning any targets. The per-file
  warnings let the user find the otherwise easily overlooked problem files.
- Common target failures—including image loading/format errors, face-detection
  failures, zero or multiple detected faces, and vector calculation failures—
  produce warnings on standard error, then scanning continues. A log file is
  written only when the user explicitly requests one.
- When DeepFace needs a missing RetinaFace or selected recognition-model asset,
  Faceledger announces the dependency acquisition through its diagnostics and
  allows DeepFace to perform its normal download and persistence. Faceledger
  does not implement a parallel model-asset manager. An acquisition failure is
  an operation error with retry/offline guidance.
- After a usable source vector exists, comparison is best-effort. Unreadable
  descendants, permission problems, and other individual target failures are
  warnings rather than failed runs; the operation completes successfully with
  whatever results it could produce.
- When some images in a single-person target folder fail, the average uses the
  remaining usable vectors. If no usable vector remains, that target is warned
  about and skipped.
- No persistent database, directory index, or catalogue of scanned files is
  created.

### Cleanup

- A trash action processes only its selected root by default. Descendants enter
  scope only when the user explicitly requests recursive maintenance.
- Cleanup detection must match Faceledger's model-qualified NPY suffix, not every
  file ending in `.npy`.
- If discovery finds no matching cache entry for the selected model, trash is a
  successful no-op. Faceledger reports `No matching <model> cache entries found`
  and creates neither an action directory nor a manifest; traversal warnings,
  if any, remain visible separately.
- A cleanup action writes its complete discovered plan to one manifest before
  moving files. Every entry records its original and trash-relative paths with
  an initial `planned` outcome.
- After each move attempt, the same manifest records that entry as `moved` or
  `failed`, including a reason when known. A process interruption therefore
  leaves any still-`planned` entry visibly unconfirmed. Move failures are also
  emitted as standard-error warnings, and the action remains successful under
  the best-effort rule.
- Each action has a distinct UTC timestamp with microsecond precision as its
  action ID. If that directory already exists, Faceledger appends a numeric
  suffix rather than reusing or overwriting it.
- On supported Linux systems, the application data root is
  `${XDG_DATA_HOME:-$HOME/.local/share}/faceledger`; the trash root is its
  `trash/` child. `XDG_DATA_HOME` must be honored when set.
- The application data root is resolved in one centralized, readily visible
  configuration point so a future user override does not require finding path
  literals throughout the codebase.
- The moved layout is relative to the selected cleanup root and contains no
  directories that are unnecessary for the moved files.
- Trash works when the selected cache tree and application trash are on
  different filesystems. Faceledger copies an entry to its planned trash
  location, verifies that the destination faithfully represents the source,
  and only then removes the original. A copy or verification failure leaves the
  original untouched, records `failed` in the manifest, emits a warning, and
  does not stop other entries.
- Cleanup moves files; it does not permanently delete them.
- Version one provides no restore operation. Recovery is manual using the action
  directory and `manifest.txt`.

### Vector-cache creation

- Cache build and rebuild process only their selected root by default.
  Descendants enter scope only when the user explicitly requests recursive
  maintenance.
- A cache build creates a model-specific cache entry only when a compatible
  entry is not already available. It does not refresh all existing entries.
- A corrupt, non-numeric, or wrong-dimension cache entry is not compatible. A
  cache build replaces it as though its entry were missing and emits a warning.
- A structurally compatible entry remains untouched even if its source image may
  have changed. Users explicitly request a cache rebuild to manage freshness.
- A cache rebuild is an explicit, more destructive action. It recalculates every
  in-scope entry for the selected model and overwrites corresponding cache files.
- Rebuild replacement is non-destructive at the individual-entry boundary. An
  existing cache entry remains intact unless its replacement vector has been
  calculated and safely persisted in full. Any calculation or persistence
  failure warns, preserves the old entry, and allows the broader rebuild to
  continue successfully.
- Build and rebuild classify folders and average single-person folders using the
  same face-image rules as comparison, but do not perform comparisons.
- Vector-cache maintenance is also best-effort. An individual read, write, or
  move failure produces a warning and leaves that item unchanged where
  possible; processing continues and the overall action is still successful.
  The user decides whether and how to remedy warned-about items.

## 6. Inputs and outputs

### Comparison inputs

| Input | Requirement/default |
| --- | --- |
| Source image or source folder | Exactly one is required |
| Target root folder | Required |
| Model/backend | Optional; defaults to Facenet512 |
| Cosine threshold | Optional finite override in `[0, 2]`; otherwise a model-specific default |
| Reuse existing NPY | Optional setting; defaults to enabled |
| Single-target-folder mode | Optional; otherwise the target root is scanned recursively |

### Comparison outputs

- The main output is a result list containing only candidate matches. Entries are
  ordered from smallest to largest cosine distance. Each entry contains:
  - the cosine distance; and
  - the target-root-relative image path for a multi-person target or relative
    directory path for an averaged single-person target.
- Result identity paths preserve the filesystem's original casing and remain
  unambiguous when different branches reuse the same folder or image names.
- Standard output is reserved for the ranked result header and table. It does
  not contain warnings, errors, or debugging diagnostics.
- Each individual item that Faceledger cannot process produces a warning on
  standard error containing a warning severity, category, stable diagnostic
  code, affected path, and human-readable reason. Warnings do not make a
  best-effort operation unsuccessful.
- At the end of a successful run with warnings, standard error includes a
  prominent summary with the number of warnings and relevant examined,
  successful, or compared counts. This keeps problems noticeable even when the
  individual warning list is long.
- A condition that prevents the requested operation from meaningfully running
  or continuing is an error. It is written to standard error, leaves standard
  output free of a result, and produces an unsuccessful process status. An
  unusable comparison source is one such condition. A missing, non-directory,
  or wholly unreadable explicitly selected target root is another and is
  rejected before scanning.
- A result file or log file is created only when the user explicitly supplies a
  destination.
- Once source validation succeeds, individual target problems do not make the
  comparison process unsuccessful. An unusable source terminates immediately
  with an unsuccessful process result.
- A comparison run does not persist calculated vectors or otherwise modify the
  vector cache.
- Standard output begins with the source, target root, selected model, and active
  threshold, followed by a human-readable table. The header provides the
  resolved source and target-root paths; each row contains rank,
  target-root-relative identity path, and cosine distance. An empty table is represented as
  `No matches found`.
- An explicitly requested log is one human-readable UTF-8 diagnostic record for
  the run. It contains operation metadata, every warning and error, and final
  counts, but does not duplicate the ranked result table. An explicitly
  requested result destination remains separate.
- Opt-in debugging adds technical diagnostics, including an available traceback
  for an unexpected application error, to standard error and to the same log
  when one is requested. Version one does not split warnings, errors, and debug
  details into separate log files.
- Long-running uncached work exposes progress separately from standard-output
  result data, so interactive CLI users can tell that Faceledger is advancing
  and a future frontend can present the same progress meaning differently.
- User cancellation stops further work at a safe item boundary where possible,
  clearly reports that the operation is incomplete, and does not present a
  partial comparison as a completed result. Maintenance work already completed
  remains in place; it is not rolled back.

### Cleanup inputs and outputs

- Inputs:
  - a selected root folder;
  - an optional request for recursion, disabled by default; and
  - the model whose NPY suffix is to be matched.
- Outputs:
  - when matching entries exist, a collision-safe UTC-timestamped action
    directory below the application trash root, `manifest.txt` inside it as the
    planned-and-finalized per-entry recovery record, and the successfully moved
    NPY files in the replicated relative structure; or
  - when no matching entries exist, a successful no-op message and no files.

### Vector-cache maintenance inputs and outputs

- A cache build or rebuild takes a selected root, model, and optional request
  for recursion, disabled by default.
- Every build, rebuild, or trash action operates on exactly one model. The model
  defaults to Facenet512 and may be overridden; version one has no all-models
  maintenance action.
- Build outputs newly created model-qualified NPY files only for missing cache
  entries.
- Rebuild outputs a freshly calculated model-qualified NPY for every in-scope
  entry and safely replaces an existing corresponding file only after the new
  entry is fully persisted.

## 7. Constraints and provisional technical choices

### Product and domain constraints

- Source selection and target-root selection are mandatory.
- Explicitly selected required inputs are operation preconditions. An invalid
  source or selected root is an error; a problem discovered with an individual
  descendant beneath a valid root is a warning handled best-effort.
- Source-folder averaging and target-folder classification must obey the naming
  and non-recursion rules in this seed.
- Model identity must be present in persisted NPY filenames and cleanup matching.
- The first version supports every face-recognition backend in the researched
  DeepFace registry: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
  Dlib, ArcFace, SFace, GhostFaceNet, and Buffalo_L.
- The system must tolerate hybrid trees containing both existing vectors and
  images that still require calculation.
- Comparison and vector-cache maintenance are separate capabilities. Comparison
  must not create, overwrite, or remove cache entries.
- A comparison creates no Faceledger result, log, vector-cache, or other
  user-data files by default. DeepFace's visible first-use persistence of a
  required model asset is dependency bootstrap activity and is the sole narrow
  exception.
- Result data and diagnostic notifications are distinct outputs. CLI rendering
  must not be the only representation available to the application logic.
- Target-local failures must not abort the broader scan.
- Cleanup must remain recoverable and manifest-backed.
- No persistent database of the scanned tree may be introduced.
- Version one provides no inter-process locking or filesystem snapshot.
  Overlapping cache build, rebuild, or trash actions on the same face tree are
  unsupported and users are responsible for avoiding them. Concurrent read-only
  comparisons are allowed but observe the tree as it changes.
- A descendant that disappears or changes after discovery produces a warning
  and best-effort continuation where possible; Faceledger does not attempt
  centralized coordination or stale-lock recovery.
- Faceledger never uploads source images, target images, or vectors and sends no
  telemetry. Model-asset acquisition remains the sole announced inbound network
  exception.
- The version-one platform envelope is mainstream glibc-based x86-64 Linux with
  CPU inference. Each release publishes the exact environments used for
  release-blocking validation across Ubuntu LTS, Debian stable, current Fedora,
  and current Arch.
- Other current glibc x86-64 distributions may be described as
  expected-compatible, but do not block a release unless deliberately promoted
  into the validation matrix.

### Provisional choices

- **Provisional:** deliver the first version as a command-line application.
- **Provisional:** represent results and diagnostics in a presentation-neutral
  form before rendering them for the CLI. A diagnostic carries severity,
  category, stable code, affected path when applicable, and a human-readable
  message. The eventual internal interface is deliberately unspecified.
- **Provisional pending a packaging spike:** use one application-managed CPython
  version and a fully locked runtime dependency set rather than relying on each
  distribution's system Python. The exact Python version, glibc ABI floor, and
  transitive dependency versions are not yet selected.
- Use DeepFace to load the supported face-recognition models, detect faces, and
  create embeddings.
- Delegate model-weight acquisition and storage to DeepFace. Faceledger surfaces
  that activity and its failures but does not duplicate the dependency's asset
  management.
- Pin DeepFace exactly to version 0.0.100. The supported model registry and
  hard-coded thresholds belong to that release and expand only through an
  intentional Faceledger dependency upgrade.
- Use RetinaFace detection with alignment enabled for every vector. These fixed
  settings are part of cache compatibility and are not user-overridable in
  version one.
- Hard-code DeepFace 0.0.100's pre-tuned cosine threshold verbatim for every
  supported model. Users may override the active threshold for a comparison run
  to broaden or narrow the candidate list.
- **Provisional:** keep model names and their threshold defaults together in one
  readily reviewable configuration area near the primary configuration surface.

These choices do not commit the final architecture, module layout, classes,
function signatures, or CLI syntax.

DeepFace 0.0.100's supported models, canonical cache slugs, and pre-tuned cosine
thresholds are:

| DeepFace model | Cache slug | Default cosine threshold |
| --- | --- | ---: |
| VGG-Face | `vgg-face` | 0.68 |
| Facenet | `facenet` | 0.40 |
| Facenet512 | `facenet512` | 0.30 |
| OpenFace | `openface` | 0.10 |
| DeepFace | `deepface` | 0.23 |
| DeepID | `deepid` | 0.015 |
| Dlib | `dlib` | 0.07 |
| ArcFace | `arcface` | 0.68 |
| SFace | `sface` | 0.593 |
| GhostFaceNet | `ghostfacenet` | 0.65 |
| Buffalo_L | `buffalo_l` | 0.55 |

## 8. Non-goals for the first version

- Building or saving a comprehensive database/index of directories, images,
  vectors, or prior scans.
- Recursively finding images beneath a selected source folder.
- Treating a multi-person or miscellaneous folder as a valid source folder.
- Permanently deleting NPY files through the cleanup feature.
- Automatically restoring a trash action or resolving conflicts with cache files
  created after that action.
- Creating or refreshing the vector cache as a side effect of comparison.
- Comparing or deleting arbitrary `.npy` files without a matching model name.
- Averaging distinct identities in a multi-person folder.
- Defining the final application architecture, source-file organization,
  function boundaries, or complete command syntax in this seed.
- Adding future features not stated in the raw notes.
- Providing JSON or another machine-readable result schema.
- Providing a graphical frontend in version one.
- Supporting GPU acceleration, non-x86 architectures including ARM64, 32-bit
  Linux, or musl-based systems such as Alpine in version one.
- Promising native packages for every Linux distribution, NixOS-native
  packaging, or host-level installation on immutable/appliance-style systems
  such as SteamOS in version one.
- Coordinating concurrent maintenance processes, providing a consistent
  filesystem snapshot, or recovering stale inter-process locks.
- Guaranteeing a fixed runtime, completing accurate cross-hardware benchmarks,
  or optimizing version one for tens of thousands of recognized images.
- Processing animated WebP or selecting a frame from an animated image.
- Detecting or rejecting technically valid face vectors as semantic outliers in
  a single-person folder.
- Encrypting NPY caches or recoverable-trash contents, managing filesystem
  ownership or access controls, enforcing retention, or providing secure
  erasure.

## 9. Known unknowns and risks

### Unresolved questions

- **Q17 — Locked Linux runtime baseline:** Which managed CPython version, glibc
  ABI floor, TensorFlow/Keras/OpenCV/RetinaFace versions, and remaining
  transitive dependency versions form the tested version-one lock? A packaging
  spike must resolve this before the platform promise is implementation-ready.
- **Q30 — Model licensing and redistribution:** Before Faceledger is packaged
  for distribution, which licences govern each supported model and its weight
  files, and may those weights be bundled or only acquired separately through
  DeepFace? Version one must not imply redistribution rights that have not been
  reviewed.

### Normalization assumptions

- **A1:** A selected root always participates in its operation. A recursive
  comparison adds descendants rather than excluding its target root; explicitly
  recursive maintenance likewise adds descendants to its maintenance root.
- **A2:** `folder#.jpg` uses exactly one digit, consistent with the explicit
  single-digit convention for all numbered face files.
- **A3:** A source folder must contain the exact case-sensitive filename
  `folder.jpg`; a folder containing only numbered face images or differently
  cased/lookalike anchors is multi-person or miscellaneous and therefore invalid
  as a source.
- **A4 (resolved):** “Average NPY” was normalized as a single aggregate
  embedding at the product level. Grilling subsequently settled its calculation
  as the L2-normalized equal-weight centroid defined in this seed.
- **A6:** A single-person folder's `folder.jpg` is the persisted-vector anchor,
  even when other images contribute to its average.
- **A7:** A recursive scan processes folders independently; recognized images in
  a descendant never contribute to its ancestor's average.
- **A8:** The model selected for a run governs vector reuse, new NPY naming, and
  cleanup suffix matching.
- **A9 (superseded):** The raw notes' “outlier NPY” wording was initially
  normalized as a newly calculated per-image embedding considered for an
  average. Semantic outlier checking was later removed from version one, so the
  distinction no longer drives version-one behaviour.

### Principal risks

- Reusing NPY files without provenance or freshness checks can silently compare
  outdated embeddings; users must rebuild after changing source images.
- A technically valid image of the wrong person can corrupt an averaged identity;
  version one deliberately trusts the user's face-tree organization instead of
  attempting semantic outlier detection.
- A recursive rebuild can safely replace many cache files as explicitly
  requested, but incorrect scope selection still has a broad impact and must be
  clearly visible to the user.
- Loose filename matching could misclassify ordinary images or move unrelated
  numerical data.
- Recursive filesystem operations and replicated trash paths require explicit
  symlink, collision, and error policies to prevent surprising movement.
- Model-specific thresholds can still produce misleading candidate matches;
  Faceledger must identify them as DeepFace 0.0.100's pre-tuned values and avoid
  presenting them as guarantees of identity.
- DeepFace 0.0.100's broad lower-bounded dependencies do not by themselves form
  a reproducible Linux runtime. An untested resolver outcome could break an
  otherwise supported distribution or model unless the complete environment is
  locked and exercised across the release matrix.
- Face embeddings are sensitive local data. Vector-cache entries and files moved
  to recoverable trash remain ordinary plaintext files that inherit local
  filesystem ownership and permissions; trash deliberately extends their
  retention until the user removes it.
- The face tree can change while a comparison or maintenance action is running.
  Without inter-process coordination or a snapshot, results and manifests may
  reflect different moments; users must avoid overlapping maintenance on the
  same tree.
- RetinaFace CPU processing of uncached images may be slow and varies materially
  by hardware. Version one mitigates uncertainty with reusable caches, visible
  progress, and cancellation rather than promising a runtime SLA.
- DeepFace wraps models and weights governed by separate upstream licences.
  Supporting a model at runtime does not by itself establish the right to bundle
  or redistribute its assets with Faceledger.

## 10. Representative scenarios

### Arbitrary source image and hybrid target tree

The user selects `/outside/photo.png` as the source, chooses ArcFace, enables
existing-vector reuse, and scans `/people`. Faceledger always calculates a
temporary source vector. A target with `folder.jpg.arcface.npy` reuses it; a new
target with no NPY is calculated. Each target within the threshold produces a
candidate result. The temporary source vector is not saved, and no qualifying
targets is still a successful run.

### Single-person folder with several face images

A folder contains:

```text
folder.jpg
folder0.jpg
folder3.jpg
holiday.face1.png
```

It represents one identity. All four usable images contribute to one aggregate
vector, persisted when applicable as `folder.jpg.arcface.npy`, and the result
uses the folder's target-root-relative directory path.

### Multi-person folder with sparse numbers

A folder contains:

```text
image1.face0.jpg
image1.face3.jpg
image4.face0.png
```

Because literal `folder.jpg` is absent, each file represents a separate target.
The numbers need not be sequential. Each has its own vector, such as
`image1.face3.jpg.arcface.npy`, and its own result entry.

### Miscellaneous folder containing an NPY

A folder contains ordinary images and `old.arcface.npy`, but no `folder.jpg` or
numbered face image. Comparison classifies it as miscellaneous and skips it;
the NPY alone does not make the folder valid.

### Duplicate names in different branches

The target tree contains `actors/classic/Alice/folder.jpg` and
`staff/Alice/folder.jpg`, and both qualify as candidates. The result list shows
the distinct identity paths `actors/classic/Alice` and `staff/Alice`, relative
to the selected target root, rather than two ambiguous `Alice` rows.

### Partial averaging failure

A target single-person folder contains three recognized images. One cannot be
loaded, while two produce acceptable vectors. Faceledger warns on standard
error, averages the two usable vectors, and continues. It does not attempt to
decide whether either technically valid vector depicts the wrong person.

### Unusable structural anchor

A single-person folder contains exact `folder.jpg` plus two numbered face
images. The anchor cannot be loaded, but both numbered images yield usable
vectors. Faceledger warns for `folder.jpg`, averages the two usable vectors, and
keeps the folder as one identity. Cache maintenance associates the aggregate
with `folder.jpg.<model>.npy`; if all three images were unusable, the target
would instead be skipped or the source would terminate.

### Partial maintenance failure

A cache build can write most discovered entries but lacks permission to write
one entry. Faceledger warns about that entry, leaves it unchanged, continues
building the remaining entries, and reports the action as successful. The user
may leave the warned-about item as-is or correct it and run maintenance again.

### Invalid source folder

The selected source folder contains only `person.face0.jpg` and
`person.face1.jpg`. It is a multi-person folder, not a valid single-person
source. The run terminates before target comparison with the source-folder
selection diagnostic.

### Source folder with no usable images

The selected single-person source folder contains several recognized images,
but every image fails loading, face-count validation, or vector calculation.
Faceledger emits one path-and-reason warning for each image, then a fatal summary
such as `No usable source faces found in <path> (examined: N, usable: 0)` and
exits without scanning the target tree.

### Invalid selected target root

The source is usable, but the explicitly selected target root does not exist,
is not a directory, or cannot be read at all. Faceledger emits an operation
error on standard error and exits unsuccessfully before scanning. If the root
itself is valid and only one descendant is unreadable, that descendant instead
produces a warning and the scan succeeds best-effort.

### Recoverable model-specific cleanup

The selected cleanup tree contains ArcFace NPY files in two branches and an
unrelated `.npy` file. Faceledger records only the ArcFace-qualified paths in a
timestamped manifest, creates one timestamped trash action folder, recreates the
two required relative branches, and moves the matching files. The unrelated NPY
remains in place.

### Empty model-specific cleanup

A valid trash operation discovers no NPY suffix matching its selected model.
Faceledger reports that no matching entries were found and succeeds without
creating a timestamped action directory or `manifest.txt`. Any warnings from
unreadable descendants are still emitted.

### Partially completed trash action

A trash action discovers three matching entries and records all three as
`planned` before movement. Two moves succeed and become `moved`; one permission
failure becomes `failed` with its reason and also emits a warning. The action is
successful overall, while `manifest.txt` gives the user enough information to
recover the two moved files and investigate the unchanged third file. If the
process stops before recording an outcome, that entry remains `planned` and is
understood to be unconfirmed rather than moved.

### Failed cache replacement

A rebuild calculates replacements for three identities. Two replacements are
safely persisted, while calculation fails for the third. Faceledger warns about
that identity, leaves its old cache entry unchanged, completes the other
replacements, and reports the best-effort rebuild as successful.

### Trash across filesystems

The selected photo tree is on an external volume while application trash is in
the user's home filesystem. Faceledger copies each selected cache entry into
its planned trash path and verifies the destination before removing the
original. If one destination cannot be verified, its original remains in the
photo tree, its manifest outcome becomes `failed`, and other entries continue.

### Concurrent change during comparison

A comparison discovers a target image that another process removes before it is
loaded. Faceledger warns for that path, continues with the remaining targets,
and succeeds best-effort. Two overlapping maintenance actions against the same
tree are unsupported; version one neither locks the tree nor promises a
consistent snapshot.

### User cancels a long uncached operation

An uncached operation over a few thousand images reports progress outside the
standard-output result stream. The user cancels it while an image is being
processed. Faceledger stops further work at a safe item boundary, reports that
the run is incomplete, and does not label partial comparison rows as a completed
result. Cache entries or trash moves already completed by maintenance remain in
their safe recorded state rather than being rolled back.
