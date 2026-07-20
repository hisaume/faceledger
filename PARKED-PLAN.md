# Faceledger planning parking lot

This file preserves useful source material that is too implementation-specific,
premature, investigatory, future-facing, or contradictory for the current
product-level design seed. Nothing here is a final architectural commitment.

## 1. Provisional implementation ideas

- Use DeepFace as the embedding and face-detection dependency, subject to
  confirming that its model selection, output stability, supported formats, and
  error behaviour meet Faceledger's needs.
- Make the first version a CLI with required source and target selections and
  optional model, threshold, reuse, save, scan-depth, and cleanup controls.
- Keep model defaults together in one easily reviewed configuration namespace or
  equivalent configuration area near the principal program entry point.
- Supply hard-coded cosine thresholds per supported model, with ArcFace as the
  default.
- Use `folder.jpg` as the initial trusted/anchor embedding for a single-person
  folder and reject other calculated embeddings that are too distant from it.
  This needs investigation before adoption because the anchor can itself be
  wrong or unusable.
- Treat generated vectors as ephemeral in memory unless persistence is enabled.
  The raw notes also mention “purging” an internally calculated NPY, but no
  temporary on-disk representation is required at the product level.

## 2. Premature specification details

- The proposed cleanup call sequence is:
  1. call a scanner that returns a complete list of matching NPY paths;
  2. write that list to a timestamp-named text file in the main `trash` folder;
  3. pass the in-memory list to another operation that creates the timestamped
     action folder, recreates necessary relative folders, and moves each file.
  The observable ordering and manifest are retained in the seed, but these
  function boundaries are deferred.
- Exact function/class structures, exact call sequences beyond required
  observable ordering, source-file organization, and exception boundaries are
  intentionally excluded from the seed.
- Exact CLI option names and syntax are deferred. This includes how users express
  the mutually exclusive source image/source folder, negated default-on options,
  threshold overrides, model selection, single-folder mode, and cleanup.
- The physical placement of a centralized defaults namespace and the request to
  keep it “near the top of the code” are implementation-layout concerns.
- Detailed tactics for releasing calculated vectors, deciding whether an
  intermediate NPY ever exists on disk, and cleaning it after comparison are
  deferred.
- A log must expose target warnings, but its library, schema, rotation, output
  stream handling, and file naming are premature specification.

## 3. Future-scope ideas

- No concrete future product feature was stated beyond possible expansion to
  additional DeepFace-supported model backends.
- A persistent “grand database” or directory-tree catalogue was mentioned only
  to exclude it; it is not a planned future feature.
- Permanent deletion was not requested. The first cleanup capability remains a
  recoverable move to application-managed trash.

## 4. Technical investigations to revisit

- Compare DeepFace with viable alternatives on embedding quality, deterministic
  model/version identification, face-selection controls, supported image
  formats, CPU/GPU installation burden, and error reporting.
- Establish and document calibrated cosine thresholds for every supported
  backend rather than selecting “reasonable” values without evidence.
- Determine whether model name alone is sufficient cache provenance or whether
  an NPY needs sidecar/embedded metadata for model version, detector settings,
  source fingerprint, dimensions, and normalization.
- Define safe embedding aggregation: normalization order, averaging method,
  minimum valid samples, and post-aggregation normalization.
- Evaluate robust outlier strategies. Compare a `folder.jpg` anchor rule with
  pairwise/centroid methods and define behaviour when the anchor fails or is the
  outlier.
- Determine how DeepFace should handle images containing no face or multiple
  faces and whether target images must resolve to exactly one face.
- Establish supported extension and case rules using the actual image loader's
  capabilities.
- Define safe recursive traversal and trash movement across symlinks, permission
  failures, mount boundaries, filename collisions, interrupted moves, and paths
  containing unusual characters.
- Decide how to validate existing NPY shape, dtype, readability, backend
  compatibility, and freshness before reuse.
- Determine stable output and logging formats suitable for CLI display and later
  machine consumption without prematurely designing a broader API.

## 5. Material excluded because it conflicts with the current seed

- **NPY saving default:** The notes first specify that internally calculated
  target NPY files are saved by default, but later say that calculated vectors
  are saved only when the user sets the save flag and otherwise may be purged.
  The seed records this as `Q3`; neither behaviour is selected here.
- **“Flag” wording versus default-on reuse:** Early wording can be read as reuse
  occurring only when a flag is set, while later wording explicitly says reuse
  is on by default and is the common workflow. The explicit default is retained
  as product behaviour; the eventual CLI must decide whether users express this
  as `--use-existing`, `--no-use-existing`, or another form.
- **“Flag” wording versus default-on saving:** A positive “save” flag is awkward
  if saving is ultimately confirmed as default-on. Exact CLI polarity is parked
  until `Q3` is resolved.
- **Outlier NPY terminology:** The notes describe ignoring an “outlier NPY” while
  also saying the condition is not relevant when an existing NPY is used. The
  seed interprets the concern as rejecting a newly calculated per-image
  embedding before averaging, marked as normalization assumption `A9`, without
  choosing the algorithm.
- **Numbered folder images in otherwise invalid folders:** The notes recognize
  `folder#.jpg` face files but classify a target as single-person only when
  literal `folder.jpg` exists, and define multi-person folders by numbered
  `*.face#.ext` files. Accordingly, `folder0.jpg` without `folder.jpg` does not
  establish a valid folder. This consequence is retained visibly rather than
  broadening the classification rule.
