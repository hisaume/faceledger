# Faceledger planning parking lot

This file preserves useful source material that is too implementation-specific,
premature, investigatory, future-facing, or contradictory for the current
product-level design seed. Nothing here is a final architectural commitment.

## 1. Provisional implementation ideas

- Make the first version a CLI with a comparison capability and a separate
  vector-cache maintenance capability. Exact command and option names remain
  provisional.
- Keep model defaults together in one easily reviewed configuration namespace or
  equivalent configuration area near the principal program entry point.
- Keep the complete supported DeepFace model-name and default-threshold mapping
  together near the primary configuration surface for easy maintenance.
- Resolve one centralized application data root at runtime rather than embedding
  a literal `~` path. On supported Linux it honors `XDG_DATA_HOME` and otherwise
  resolves to `~/.local/share/faceledger`; the trash root is its `trash/` child. A
  platform-aware helper such as `platformdirs` is a provisional implementation
  option, and exact variable/module placement remains an architectural choice.
- Treat comparison-generated vectors as ephemeral in memory. Vector persistence
  belongs only to the dedicated maintenance capability.
- Let DeepFace own its normal model-weight acquisition and storage rather than
  introducing a Faceledger asset subsystem. Faceledger should announce
  first-use acquisition and translate a dependency failure into an actionable
  diagnostic; exact interception and progress-reporting mechanics remain
  provisional.

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
  the mutually exclusive source image/source folder, negated default-on reuse,
  threshold overrides, model selection, single-folder mode, and vector-cache
  build/rebuild/trash actions. The exact spelling of an explicit rebuild control
  (for example, a subcommand or a `--rebuild`/`--force` flag) is not yet chosen.
- Exact progress rendering, terminal-detection behaviour, cancellation exit
  codes, and interrupt handling remain premature CLI/runtime details. The seed
  requires presentation-neutral progress, clear incomplete status, safe-boundary
  cancellation where possible, and no rollback of completed maintenance items.
- The physical placement of a centralized defaults namespace and the request to
  keep it “near the top of the code” are implementation-layout concerns.
- Detailed tactics for releasing calculated vectors, deciding whether an
  intermediate NPY ever exists on disk, and cleaning it after comparison are
  deferred.
- The settled rebuild invariant preserves an existing cache entry until its
  complete replacement is safely persisted. Temporary-file naming, flush and
  durability behaviour, and the atomic replacement mechanism remain premature
  implementation details.
- Warnings and errors go to standard error, and an explicitly requested log is
  one human-readable UTF-8 run record containing operation metadata,
  diagnostics, and final counts. The logging library, exact text layout,
  escaping rules, CLI spelling, and destination handling remain premature
  specification.

## 3. Future-scope ideas

- A future `verify`-style capability may scan a selected face tree specifically
  to report every detectable image and NPY problem. Unlike version-one
  comparison and maintenance, its purpose may justify treating any discovered
  problem as an unsuccessful verification result. Its scope, checks, and status
  semantics require a separate design exercise.
- A persistent “grand database” or directory-tree catalogue was mentioned only
  to exclude it; it is not a planned future feature.
- Permanent deletion was not requested. The first cleanup capability remains a
  recoverable move to application-managed trash.
- Semantic outlier detection for technically valid images in a single-person
  folder is deferred beyond version one. A future design would need to handle an
  incorrect or unusable `folder.jpg` anchor rather than assuming it is trusted.
- A future output-format option may add JSON or another machine-readable result
  schema. Version one promises only the human-readable stdout table.
- A future graphical frontend should consume the same conceptual result and
  diagnostic information as the CLI. A future machine-readable CLI mode could
  provide a quick subprocess integration, while direct reuse of application
  logic remains an architectural decision for later.
- An all-models vector-cache maintenance action is deferred. Version one handles
  exactly one selected model per build, rebuild, or trash action.
- An application-managed restore action is deferred. Version one supports manual
  recovery from each action's `manifest.txt` and preserved files.
- Encryption, application-managed permissions, retention enforcement, and secure
  erasure for NPY caches or recoverable trash are outside version one. Any later
  privacy feature needs separate threat modeling rather than an implied promise
  from local-only processing.
- Inter-process maintenance locking, consistent filesystem snapshots, and stale
  lock recovery are outside version one. Revisit them only if concurrent use
  becomes a real requirement for the folder-backed datastore.

## 4. Technical investigations to revisit

- Run the packaging spike described in
  `docs/research/linux-v1-support-scope.md`: choose one managed CPython version,
  resolve and lock DeepFace 0.0.100's complete runtime graph, establish the
  resulting glibc ABI floor, and validate CPU operation on representative
  Ubuntu LTS, Debian stable, Fedora, and Arch environments. The spike must also
  exercise all eleven promised recognition models and every supported image
  format, including both first-use model acquisition and a later offline run
  using already-present assets.
- Review the upstream licence and weight-redistribution terms for every supported
  DeepFace recognition model and RetinaFace before choosing a distribution
  format or bundling any model asset. Runtime support must not be mistaken for
  redistribution permission.
- Confirm the pinned DeepFace release's output stability, model/version
  identification, face-selection controls, supported image formats, CPU/GPU
  installation burden, and error reporting.
- On any future DeepFace upgrade, compare its pre-tuned cosine thresholds with
  the pinned Faceledger mapping before accepting the new compatibility boundary.
- Determine whether model name alone is sufficient cache provenance or whether
  an NPY needs sidecar/embedded metadata for model version, detector settings,
  source fingerprint, dimensions, and normalization.
- Verify the agreed equal-weight normalized-centroid calculation against
  DeepFace 0.0.100 during implementation.
- Benchmark the fixed RetinaFace-with-alignment profile on representative local
  photo trees only as a later optimization exercise; version one has no fixed
  runtime SLA or accurate cross-hardware benchmarking commitment.
- Verify static WebP decoding in the pinned dependency environment on every
  supported operating system; animated WebP remains out of scope.
- Revisit GPU, ARM64, musl/Alpine, NixOS-native, and immutable-system delivery
  independently after version one; each is a distinct support investigation,
  not an implied part of mainstream Linux compatibility.
- Define safe recursive traversal and trash movement across symlinks, permission
  failures, mount boundaries, interrupted moves, and paths containing unusual
  characters. Cross-filesystem trash is required to preserve the original until
  its copied destination is verified, but the verification and durability
  mechanism remains provisional. Action-directory name collisions are already
  resolved by numeric suffixing.
- Validate NPY readability, numeric content, and expected model dimensions only.
  Provenance and freshness tracking are deliberately outside the first version.
- Define the exact text rendering and escaping rules for the human-readable
  version-one result table and diagnostic log; defer a stable machine-readable
  schema until a later version.

## 5. Material excluded because it conflicts with the current seed

- **NPY saving default:** The original contradiction is superseded. Comparison
  never writes vector-cache files; explicit vector-cache maintenance owns all
  creation and cleanup.
- **“Flag” wording versus default-on reuse:** Early wording can be read as reuse
  occurring only when a flag is set, while later wording explicitly says reuse
  is on by default and is the common workflow. The explicit default is retained
  as product behaviour; the eventual CLI must decide whether users express this
  as `--use-existing`, `--no-use-existing`, or another form.
- **“Save flag” wording:** The proposed comparison save flag is intentionally
  excluded because persistence is now a separate maintenance capability.
- **Cleanup recursion default:** The raw cleanup notes make recursive traversal
  the default with a single-folder override. The settled seed supersedes this:
  every state-changing cache build, rebuild, or trash action processes only its
  selected root unless recursion is explicitly requested. Comparison remains
  recursive by default because it is read-only.
- **Outlier NPY terminology:** The notes describe ignoring an “outlier NPY” while
  also saying the condition is not relevant when an existing NPY is used. The
  initial seed interpreted the concern as a newly calculated per-image embedding
  considered for an average. Grilling subsequently removed semantic outlier
  detection from version one, so this ambiguous mechanism remains excluded
  rather than driving an algorithm.
- **Numbered folder images in otherwise invalid folders:** The notes recognize
  `folder#.jpg` face files but classify a target as single-person only when
  literal `folder.jpg` exists, and define multi-person folders by numbered
  `*.face#.ext` files. Accordingly, `folder0.jpg` without `folder.jpg` does not
  establish a valid folder. This consequence is retained visibly rather than
  broadening the classification rule.
- **Case-insensitive folder-anchor example:** An earlier normalized rule treated
  `Folder.JPG` as equivalent to `folder.jpg`. Grilling superseded it: only the
  exact case-sensitive lowercase filename `folder.jpg` is the structural anchor.
  Other casing and formats remain ordinary or unrecognized according to the
  numbered face-file rules and never classify a folder as single-person.
