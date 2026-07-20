# Faceledger design seed

## 1. Product sentence

Faceledger compares one source identity with face images found in a user-selected
folder tree by creating or reusing model-specific face vectors, calculating
cosine distances, and reporting a result for each target identity without
building a persistent catalogue of the scanned tree.

## 2. User and operating context

- The intended first version is a local command-line tool.
- A user selects exactly one source:
  - an arbitrary image file; or
  - a folder representing one person, whose recognized face images are combined
    into one source vector.
- The user also selects a target root folder. The root itself is scanned, followed
  by its descendants unless single-target-folder mode is selected.
- Users commonly retain model-specific NPY vectors beside their face images and
  reuse them on later scans. A tree may therefore contain a mixture of reusable
  vectors and newly added images without vectors.
- Users may move generated NPY files from a selected tree into application-managed
  trash. This is recoverable cleanup rather than permanent deletion.
- Face images are expected to be local files organized according to the naming
  conventions in this seed. A single source image is exempt from those naming
  conventions.

## 3. Primary user journeys

### Compare from a single source image

1. Select one image file and a target root folder.
2. Faceledger calculates a temporary source vector for the selected model,
   regardless of vector reuse or saving settings.
3. It discovers and classifies target folders, obtains their target vectors, and
   compares each target vector with the source using cosine distance.
4. It returns a result list containing the target identity and distance.
5. Any source-image failure terminates the run; an ordinary target failure is
   reported as a warning and does not stop the remaining scan.

### Compare from a single-person source folder

1. Select a source folder containing literal `folder.jpg` and select a target
   root folder.
2. Faceledger applies the single-person folder rules to obtain one averaged
   source vector from all recognized face images in the source folder root.
   It does not descend into source subfolders.
3. Existing-vector reuse and vector-saving settings apply to this source.
4. The target scan and comparison proceed as in the single-image journey.
5. A miscellaneous or multi-person source folder terminates the run with a
   message equivalent to `Select a single-suspect folder (with folder.jpg)`.

### Compare a target tree

1. Select a target root and optionally restrict scanning to that folder alone.
2. For each folder in scope, classify it as single-person, multi-person, or
   miscellaneous.
3. Reuse a corresponding model-specific NPY when enabled and available;
   otherwise calculate the required vector.
4. Optionally save or overwrite calculated vectors beside their associated face
   images.
5. Compare every resulting target vector with the source and add its identity
   and distance to the main result list.

### Move generated vectors to trash

1. Select a root folder and optionally restrict the operation to that folder.
2. Discover only NPY files whose suffix identifies the selected model, and record
   the discovered paths in a timestamped text manifest.
3. Create a timestamped action folder under the application's main `trash`
   folder.
4. Move the discovered NPY files there, recreating only the relative directories
   needed to hold them.

## 4. Domain concepts and glossary

| Term | Meaning |
| --- | --- |
| Source | The single identity against which targets are compared. It is supplied as one arbitrary image or one single-person folder. |
| Target root | The top folder selected for comparison or NPY cleanup. |
| Face image | A recognized target/source-folder image named `folder.jpg`, single-digit `folder#.jpg`, or `*.face#.ext`, where `#` is one digit. |
| Exact folder image | Literal `folder.jpg`, supported for compatibility with Windows thumbnail conventions. |
| Numbered folder image | `folder0.jpg` through `folder9.jpg`. It may coexist with `folder.jpg`. |
| Numbered face image | An image whose basename ends in `.face0` through `.face9` before its supported image extension. Digits need not start at zero or be consecutive. |
| Single-person folder | A folder containing literal `folder.jpg`. It represents one identity and produces one averaged vector even when other recognized face images are present. |
| Multi-person folder | A folder with one or more numbered face images but no literal `folder.jpg`. Each face image represents a distinct identity and produces its own vector. |
| Miscellaneous/scaffolding folder | A folder with neither literal `folder.jpg` nor a numbered face image. It is skipped, even if it contains NPY files. |
| Model/backend | The face-embedding model used to generate and interpret vectors. ArcFace is the default. |
| Vector/NPY | The model-specific numerical face representation used for cosine-distance comparison. NPY is the persisted representation. |
| Averaged vector | The one vector representing all usable face images in a single-person folder. The exact aggregation method is unresolved. |
| Existing-vector reuse | Loading an associated model-specific NPY instead of recalculating it. This is enabled by default. |
| Result | At minimum, a target identity and its cosine distance from the source. |
| Single-target-folder mode | Restricts a target comparison or cleanup operation to the selected root rather than its descendants. |
| Trash action | One recoverable cleanup operation, with its own timestamped folder and manifest. |

`ext` means a commonly supported image extension. JPEG and PNG are the expected
common cases; the authoritative extension set is unresolved.

## 5. Behavioural rules and invariants

### Folder classification and vector production

Classification is based on recognized face images, not on the presence of NPY
files.

| Condition in a folder | Classification | Vector production | Result identity |
| --- | --- | --- | --- |
| Literal `folder.jpg` exists | Single-person | One averaged vector using every usable `folder.jpg`, `folder#.jpg`, and `*.face#.ext` in that folder | Folder name |
| No `folder.jpg`, but one or more `*.face#.ext` exist | Multi-person | One independent vector per usable face image; no averaging | Face filename |
| Neither condition holds | Miscellaneous/scaffolding | None | None; folder is skipped |

- Recognition and averaging concern only files in the folder being classified.
- A numbered folder image alone does not make a valid single-person folder.
- In a multi-person folder, face-number digits may be sparse or unordered and
  each recognized file remains independent.
- For an averaged vector, the persisted NPY is associated with `folder.jpg`.
  Therefore removing the final `.npy` suffix from any Faceledger NPY name
  identifies the first/anchor face image associated with that vector.

### Source acquisition

| Source kind | Naming rules | Reuse existing NPY | Save NPY | Failure |
| --- | --- | --- | --- | --- |
| Single image | Exempt from folder face-file naming rules | Never | Never; vector is temporary for the run | Terminate immediately |
| Source folder | Must classify as single-person and is not scanned recursively | Enabled by default when a corresponding NPY exists | Governed by the save setting | Invalid folder or inability to produce a usable source vector terminates the run |

### Existing-vector reuse and generation

| Corresponding model NPY exists? | Reuse enabled? | Behaviour |
| --- | --- | --- |
| Yes | Yes | Load the existing NPY; do not perform image outlier checks for that loaded vector |
| Yes | No | Recalculate from the applicable face image or images |
| No | Either | Calculate the vector |

- Reuse is enabled by default.
- NPY data is model-specific. A vector for one backend must not be treated as a
  vector for another backend.
- A calculated vector selected for persistence overwrites the corresponding
  existing NPY.
- Persisted names append the normalized model name and `.npy` to the complete
  face-image filename, for example `suspect.jpg.arcface.npy`.
- The save default is contradictory in the source notes and remains unresolved
  as **Q3** rather than being silently selected.

### Comparison and failure handling

- ArcFace is the default model. The user may select another supported model and
  may override that model's default cosine threshold.
- Every usable target vector is compared with the source vector using cosine
  distance and produces a result entry.
- A source failure is fatal because no meaningful target comparisons can follow.
- Common target failures—including image loading/format errors, face-detection
  failures, vector calculation failures, and rejected averaging outliers—produce
  warnings in both the CLI and a log, then scanning continues.
- When some images in a single-person target folder fail, the average uses the
  remaining usable vectors. If no usable vector remains, that target is warned
  about and skipped.
- No persistent database, directory index, or catalogue of scanned files is
  created.

### Cleanup

- Cleanup detection must match Faceledger's model-qualified NPY suffix, not every
  file ending in `.npy`.
- A cleanup action writes its discovered-file manifest before moving files.
- Each action has a distinct timestamped trash folder.
- The moved layout is relative to the selected cleanup root and contains no
  directories that are unnecessary for the moved files.
- Cleanup moves files; it does not permanently delete them.

## 6. Inputs and outputs

### Comparison inputs

| Input | Requirement/default |
| --- | --- |
| Source image or source folder | Exactly one is required |
| Target root folder | Required |
| Model/backend | Optional; defaults to ArcFace |
| Cosine threshold | Optional override; otherwise a model-specific default |
| Reuse existing NPY | Optional setting; defaults to enabled |
| Save/overwrite calculated NPY | Optional setting; default unresolved in **Q3** |
| Single-target-folder mode | Optional; otherwise the target root is scanned recursively |

### Comparison outputs

- The main output is a result list. Each entry contains:
  - the cosine distance; and
  - the face filename for a multi-person target or the folder name for an
    averaged single-person target.
- Non-fatal problems appear as both CLI warnings and log entries.
- When enabled, calculated vectors are persisted beside the associated face
  image using the model-qualified filename.
- The result's serialization, ordering, match label, threshold presentation, and
  output destination are unresolved in **Q1** and **Q2**.

### Cleanup inputs and outputs

- Inputs:
  - a selected root folder;
  - the recursive/single-folder setting; and
  - the model whose NPY suffix is to be matched.
- Outputs:
  - a timestamped text manifest in the main application `trash` folder;
  - a timestamped per-action folder below `trash`; and
  - the discovered NPY files moved into the replicated relative structure.

## 7. Constraints and provisional technical choices

### Product and domain constraints

- Source selection and target-root selection are mandatory.
- Source-folder averaging and target-folder classification must obey the naming
  and non-recursion rules in this seed.
- Model identity must be present in persisted NPY filenames and cleanup matching.
- The system must tolerate hybrid trees containing both existing vectors and
  images that still require calculation.
- Target-local failures must not abort the broader scan.
- Cleanup must remain recoverable and manifest-backed.
- No persistent database of the scanned tree may be introduced.

### Provisional choices

- **Provisional:** deliver the first version as a command-line application.
- **Provisional:** use DeepFace to load models, detect faces, and create
  embeddings. Its suitability remains an investigation because the notes invite
  alternatives.
- **Provisional:** maintain model defaults, including ArcFace and per-model
  cosine thresholds, in one readily reviewable configuration area.
- **Provisional:** provide hard-coded reasonable default thresholds for each
  supported model.
- **Provisional:** use literal `folder.jpg` as an anchor when assessing whether
  other images are drastic outliers during averaging.

These choices do not commit the final architecture, module layout, classes,
function signatures, or CLI syntax.

## 8. Non-goals for the first version

- Building or saving a comprehensive database/index of directories, images,
  vectors, or prior scans.
- Recursively finding images beneath a selected source folder.
- Treating a multi-person or miscellaneous folder as a valid source folder.
- Permanently deleting NPY files through the cleanup feature.
- Comparing or deleting arbitrary `.npy` files without a matching model name.
- Averaging distinct identities in a multi-person folder.
- Defining the final application architecture, source-file organization,
  function boundaries, or complete command syntax in this seed.
- Adding future features not stated in the raw notes.

## 9. Known unknowns and risks

### Unresolved questions

- **Q1 — Meaning of an outcome:** Does a threshold produce a match/non-match
  label, filter the result list, or merely accompany a list that always contains
  every calculated distance?
- **Q2 — Result contract:** What format, ordering, destination, and metadata
  should the result list use? Should skipped targets and warnings also be
  represented in it?
- **Q3 — Save default and precedence (direct contradiction):** One rule says
  calculated target NPY files are saved by default, while later rules describe
  saving only when the user explicitly sets the save flag and otherwise purging
  the calculated NPY. Which rule has precedence, and does the answer differ for
  source folders and targets?
- **Q4 — Image extensions:** Which extensions and filename case variants are
  supported? Are `.jpeg`, uppercase extensions, and formats beyond JPEG/PNG
  recognized?
- **Q5 — Default thresholds:** Which models are supported initially, and what
  cosine threshold belongs to each one?
- **Q6 — Averaging:** Are embeddings normalized before and/or after aggregation,
  and is the aggregate an arithmetic mean or another representation?
- **Q7 — Outlier rejection:** How is a “drastic outlier” measured, what threshold
  applies, what minimum sample count is needed, and what happens when the anchor
  image itself fails?
- **Q8 — Existing NPY validity:** How should corrupt, stale, dimensionally
  incompatible, or incorrectly modelled NPY files be detected and handled?
- **Q9 — Complete source-folder failure:** What precise diagnostic and recovery
  guidance should be given when every source-folder image fails?
- **Q10 — Logging:** Where is the log written, how is it named/formatted, and is
  it per run or cumulative?
- **Q11 — Trash collisions:** What timestamp precision is required, and what
  happens if an action folder or replicated destination file already exists?
- **Q12 — Model-name canonicalization:** What canonical text represents each
  model in NPY filenames, and is cleanup matching case-sensitive?
- **Q13 — Filesystem boundaries:** How should symlinks, permission failures,
  unreadable folders, and paths outside the selected root be treated?
- **Q14 — Face extraction semantics:** What happens if a recognized face image
  contains zero or multiple detectable faces, and which DeepFace detection
  settings, if any, are product requirements?
- **Q15 — Source/target overlap:** Is it valid for the source folder to be inside
  the target tree, and if so should it be compared with itself?
- **Q16 — Cleanup scope:** Does cleanup target only the currently selected model,
  or may one action select multiple/all supported model-qualified NPY files?

### Normalization assumptions

- **A1:** The selected target root participates in comparison and cleanup scans;
  “recursively downward” adds its descendants rather than excluding the root.
- **A2:** `folder#.jpg` uses exactly one digit, consistent with the explicit
  single-digit convention for all numbered face files.
- **A3:** A source folder must contain literal `folder.jpg`; a folder containing
  only numbered face images is multi-person and therefore invalid as a source.
- **A4:** “Average NPY” means a single aggregate embedding at the product level;
  no aggregation algorithm is selected by this wording.
- **A5:** A result “outcome” contains at least target identity and numeric cosine
  distance. Any match label or threshold-based filtering remains unresolved.
- **A6:** A single-person folder's `folder.jpg` is the persisted-vector anchor,
  even when other images contribute to its average.
- **A7:** A recursive scan processes folders independently; recognized images in
  a descendant never contribute to its ancestor's average.
- **A8:** The model selected for a run governs vector reuse, new NPY naming, and
  cleanup suffix matching.
- **A9:** References to rejecting an “outlier NPY” mean rejecting a newly
  calculated per-image embedding before it contributes to an average, because
  the notes explicitly exclude reused NPY files from that check.

### Principal risks

- Reusing NPY files without provenance or freshness checks can silently compare
  incompatible or outdated embeddings.
- A poorly chosen outlier rule can discard legitimate variation or allow a wrong
  face to corrupt an averaged identity.
- Ambiguous save defaults could unexpectedly overwrite many files or, in the
  opposite direction, defeat the expected caching workflow.
- Loose filename matching could misclassify ordinary images or move unrelated
  numerical data.
- Recursive filesystem operations and replicated trash paths require explicit
  symlink, collision, and error policies to prevent surprising movement.
- Model-specific thresholds may create misleading match decisions if their
  source and calibration are not documented.

## 10. Representative scenarios

### Arbitrary source image and hybrid target tree

The user selects `/outside/photo.png` as the source, chooses ArcFace, enables
existing-vector reuse, and scans `/people`. Faceledger always calculates a
temporary source vector. A target with `folder.jpg.arcface.npy` reuses it; a new
target with no NPY is calculated. Both produce distance results. The temporary
source vector is not saved.

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
uses the folder name.

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

### Partial averaging failure

A target single-person folder contains three recognized images. One cannot be
loaded, while two produce acceptable vectors. Faceledger warns in the CLI and
log, averages the two usable vectors, and continues. The policy for deciding
whether a calculated vector is an outlier remains **Q7**.

### Invalid source folder

The selected source folder contains only `person.face0.jpg` and
`person.face1.jpg`. It is a multi-person folder, not a valid single-person
source. The run terminates before target comparison with the source-folder
selection diagnostic.

### Recoverable model-specific cleanup

The selected cleanup tree contains ArcFace NPY files in two branches and an
unrelated `.npy` file. Faceledger records only the ArcFace-qualified paths in a
timestamped manifest, creates one timestamped trash action folder, recreates the
two required relative branches, and moves the matching files. The unrelated NPY
remains in place.
