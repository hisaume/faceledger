# Faceledger

Faceledger is a casual face-comparison context that ranks plausible target
identities for a user-selected source. Its language deliberately describes
similarity candidates rather than verified identity.

## Language

**Source identity**:
The one person represented by the image or single-person folder against which
targets are compared.
_Avoid_: Suspect, query person

**Target identity**:
One person represented by a target face image or single-person folder.
_Avoid_: Suspect, record

**Face tree**:
A user-managed folder hierarchy whose recognized filenames and placement are
Faceledger's source of truth for target identities and cached vectors.
_Avoid_: Database, catalogue

**Live face-tree view**:
The non-snapshot view of a face tree observed while an operation traverses it.
Concurrent read-only comparisons are allowed, but overlapping maintenance is
unsupported and changes encountered mid-operation are handled best-effort.
_Avoid_: Transactional snapshot, locked database view

**Supported Linux envelope**:
Mainstream glibc-based x86-64 Linux using CPU inference. Faceledger validates
representative Debian/Ubuntu, Fedora, and Arch environments for each release;
other qualifying distributions may be expected-compatible without being
release-blocking.
_Avoid_: Ubuntu-only support, every Linux system, GPU support

**Local processing boundary**:
Faceledger never uploads images or vectors and sends no telemetry. Its only
network activity is announced inbound acquisition of missing DeepFace model
assets; persisted caches and recoverable trash remain ordinary local files under
the user's filesystem controls.
_Avoid_: Cloud comparison, managed biometric vault

**Usable face image**:
A recognized image from which exactly one face is detected and a structurally
valid embedding can be calculated.
_Avoid_: Multi-face image, merely loadable image

**Supported image file**:
A regular local file whose content is JPEG, PNG, or one-frame static WebP. It
may still fail to be a usable face image.
_Avoid_: Supported extension, usable face image

**Candidate match**:
A target identity whose model-specific distance is within the match threshold.
It is a plausible visual match, not a verified identification.
_Avoid_: Verified identity, confirmed match

**Match threshold**:
The greatest model-specific cosine distance admitted to the result list. A user
may override the model's default for a comparison run with any finite value from
zero through two inclusive.
_Avoid_: Confidence, accuracy

**Result list**:
The candidate matches from one comparison run, ordered from smallest distance
to largest distance. An empty result list is a successful outcome.
_Avoid_: Verification report, all comparisons

**Result identity path**:
The path that locates a target identity relative to the selected target root.
It is the directory path for a single-person folder and the image path for a
multi-person face, with original casing preserved.
_Avoid_: Bare folder name, bare face filename, absolute result path

**Comparison run**:
A comparison of one source identity against target identities that is read-only
with respect to the face tree and vector cache. It may reuse cached vectors or
calculate transient ones but never changes the cache; dependency bootstrap may
persist missing model assets separately.
_Avoid_: Scan, cache build

**Fatal source failure**:
The inability to produce a usable vector for the selected source identity. It
ends a comparison immediately because no meaningful target work can follow.
Individual target problems are warnings, not fatal source failures.
_Avoid_: No matches, skipped target

**Best-effort operation**:
An operation that emits a separate warning for each individual item it cannot
process, continues with the remaining items, and is still considered
successful. Comparison uses this behaviour after source validation;
vector-cache maintenance uses it for individual items after its required inputs
have been validated.
_Avoid_: Atomic batch, fail-fast operation

**Diagnostic notification**:
A structured notice about an operation, kept distinct from its result data. It
has a severity and category, a stable diagnostic code, an affected path when
applicable, and a human-readable message. The CLI renders notifications on
standard error; a later presentation layer may render the same meaning
differently.
_Avoid_: Result row, unstructured print statement

**Progress notification**:
A presentation-neutral notice that a potentially long-running operation is
advancing. It is separate from result data and carries no claim about a fixed
completion time.
_Avoid_: Candidate result, performance guarantee

**Cancelled operation**:
An operation stopped by the user before completion. It is reported as
incomplete, does not present partial comparison data as a completed result, and
does not roll back maintenance items already completed safely.
_Avoid_: Successful partial result, fatal item failure

**Warning**:
A diagnostic notification that one item could not be processed during a
best-effort operation. It does not make the operation unsuccessful.
_Avoid_: Fatal error, candidate result

**Operation error**:
A diagnostic notification that the requested operation cannot meaningfully
start or continue. It produces an unsuccessful outcome; an unusable source is a
comparison-specific example, as is an invalid explicitly selected target root.
Problems with individual descendants beneath a valid root are warnings instead.
_Avoid_: Warning, no matches

**Vector cache**:
Model-specific face vectors persisted as NPY files beside their associated face
images for reuse by later comparison runs.
_Avoid_: Database, saved NPY files

**Cache model slug**:
The fixed lowercase model identifier embedded in a vector-cache filename. Cache
reuse and trash matching are case-sensitive, so differently cased suffixes are
not cache model slugs and remain untouched.
_Avoid_: DeepFace display name, case-insensitive model suffix

**Compatible cache entry**:
A model-qualified NPY file that loads as a numeric vector with the expected
dimensions for Faceledger's fixed vector profile. Compatibility does not assert
freshness or provenance.
_Avoid_: Current vector, verified cache entry

**Vector profile**:
The embedding compatibility boundary formed by DeepFace 0.0.100, the selected
recognition model, RetinaFace detection, and enabled face alignment.
_Avoid_: Model alone, user-selectable detector

**Vector cache maintenance**:
An explicit operation that creates entries in the vector cache or moves them to
recoverable trash, separate from a comparison run.
_Avoid_: Save flag, comparison option

**Recursive maintenance**:
An explicit expansion of a cache build, rebuild, or trash action from its
selected root to that root's descendants. Without it, maintenance processes the
selected root only; this differs from comparison, which is recursive by default.
_Avoid_: Default maintenance scope, source-folder recursion

**Cache build**:
A vector-cache maintenance action that creates missing entries for one model
without replacing compatible entries already present.
_Avoid_: Save, rebuild

**Cache rebuild**:
An explicitly requested vector-cache maintenance action that recalculates every
in-scope entry for one model. An existing entry is replaced only after its new
vector is fully calculated and safely persisted; a failed replacement preserves
the old entry.
_Avoid_: Build, automatic refresh

**Trash action**:
One vector-cache maintenance action that moves selected entries into its own
collision-safe UTC-timestamped recovery directory containing `manifest.txt` and
the moved files' necessary relative structure. Recovery is manual in version
one. An empty selection is a successful no-op and does not create a trash
action. Across filesystems, “moves” preserve the original until the copied
destination is verified.
_Avoid_: Delete, purge

**Trash manifest**:
The single recovery record for a trash action. It records every intended
original and trash-relative path as `planned` before movement, then records each
attempt as `moved` or `failed`; a remaining `planned` outcome is unconfirmed.
_Avoid_: Discovery list only, separate error manifest

**Single-person folder**:
A folder anchored by the exact case-sensitive filename `folder.jpg` that
represents one target identity, possibly using several face images to form its
vector. The anchor is structural and need not itself contribute a usable vector
when another recognized image can do so. No other image format or casing has
anchor power.
_Avoid_: Single folder, suspect folder

**Multi-person folder**:
A folder without the exact lowercase `folder.jpg` anchor in which each numbered
face image represents a separate target identity.
_Avoid_: Multi folder, suspects folder

**Folder vector**:
The equal-weight centroid of a single-person folder's usable, individually
L2-normalized face vectors, itself L2-normalized after averaging.
_Avoid_: Raw average, outlier-filtered vector
