# Linux version-one support scope

_Research date: 2026-07-22_

## Question

What would it take for Faceledger version one to support a broad majority of
mainstream Linux desktop installations rather than Ubuntu alone?

## Finding

An exact claim such as “75% of Linux installations” is not defensible. Linux has
no authoritative installation census, and public samples have strong audience
biases. Valve's June 2026 Linux-only hardware survey is useful as a consumer
desktop signal, but participation is optional and its population is Steam
users. Grouping its named entries loosely into Arch/SteamOS, Debian/Ubuntu, and
Fedora-family systems accounts for about 81% of that sample; this demonstrates
the value of testing those three ecosystems, not an 81% market-coverage
guarantee. [Valve Steam Hardware & Software Survey](https://store.steampowered.com/hwsurvey/?platform=linux)

The more meaningful promise is a tested runtime envelope: modern glibc-based
x86-64 Linux, an application-managed CPython version, a fully locked dependency
set, and CPU inference. Validate that envelope across representative
Debian/Ubuntu, Fedora, and Arch environments.

## Why distro names are not the main boundary

Python's `manylinux` standard intentionally targets mainstream glibc-based
distributions including Debian, Ubuntu, RHEL-family systems, and openSUSE. A
wheel tagged for a glibc version and architecture promises compatibility with
mainstream distributions meeting that ABI floor. It explicitly does not cover
non-glibc platforms or external systems such as CUDA.
[PEP 600](https://peps.python.org/pep-0600/)

This means a single wheel-based installation can cross package-manager families
when all binary dependencies publish compatible wheels. Alpine and other
musl-based systems form a separate compatibility family represented by
`musllinux`, so they should not be implied by a `manylinux` support claim.
[Python Packaging platform compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)

OpenCV's official Python wheels use `manylinux2014` and are intended to work on
most GNU-libc distributions. That supports the broad-glibc approach, although
Faceledger must still test its exact locked OpenCV build and static WebP path.
[opencv-python package metadata](https://pypi.org/project/opencv-python/)

## The actual compatibility bottleneck

`deepface==0.0.100` is distributed as a platform-independent Python wheel and
declares Python 3.7 or newer, but its runtime requirements are broad rather than
reproducibly locked. They include `tensorflow>=1.9.0`,
`opencv-python>=4.5.5.64`, `keras>=2.2.0`, and `retina-face>=0.0.14`.
[DeepFace 0.0.100 PyPI metadata](https://pypi.org/pypi/deepface/0.0.100/json)

Consequently, pinning DeepFace alone does not define a supportable environment.
Faceledger needs one tested, locked transitive dependency set and a declared
CPython version. Current TensorFlow wheels are large compiled artifacts with
specific Python, architecture, and glibc tags; the current release provides
x86-64 and AArch64 Linux wheels for supported Python versions, with a glibc 2.27
floor visible in its wheel metadata. [TensorFlow PyPI files](https://pypi.org/project/tensorflow/)

TensorFlow's installation documentation officially tests Ubuntu and documents
additional CUDA requirements for GPU use. It also notes that Linux AArch64 CPU
packages involve a third-party build arrangement. Broad CPU compatibility on
glibc distributions is therefore much cheaper to promise than broad GPU or ARM
support. [TensorFlow pip installation guide](https://www.tensorflow.org/install/pip)

## Recommended version-one envelope

### Release-blocking support

- 64-bit x86 Linux using glibc at or above the final locked wheels' ABI floor.
- CPU inference. GPU acceleration is not part of the version-one support
  promise.
- One application-managed CPython version chosen after a dependency-resolution
  spike; Python 3.12 or 3.13 is a plausible candidate, not yet a decision.
- A fully locked runtime dependency set, including TensorFlow, Keras, OpenCV,
  RetinaFace, NumPy, and their material binary dependencies.
- Validation on Ubuntu LTS, Debian stable, current Fedora, and current Arch at
  each release. The purpose is to exercise the Debian/Ubuntu, RPM/SELinux, and
  rolling-release ecosystems, not to build separate native packages initially.
- CPU smoke tests that load RetinaFace and Facenet512, decode every supported
  image format including static WebP, calculate and reuse an NPY, and exercise
  the XDG data/trash path.
- A release-level compatibility check for every promised DeepFace recognition
  model, because “all models supported” is broader than proving the default
  model works.

### Compatible but not release-blocking

Other current glibc x86-64 distributions and derivatives may be documented as
expected to work when they meet the runtime envelope. Examples include Linux
Mint, Pop!_OS, Ubuntu derivatives, RHEL-family distributions, openSUSE, and
Arch derivatives. A report from one of these systems is a compatibility issue,
but the release need not be blocked unless the system is promoted into the
tested matrix.

### Outside the version-one promise

- musl-based systems such as Alpine;
- 32-bit Linux and non-x86 architectures, including ARM64;
- supported GPU/CUDA acceleration;
- native packages for every distribution;
- immutable or appliance-style systems such as SteamOS, where host-level Python
  tool installation may not be the appropriate delivery path; and
- NixOS-native packaging, whose non-FHS model deserves separate validation.

These exclusions can be revisited individually. They should not be bundled into
a vague “Linux support” task because each has a different dependency and
delivery problem.

## Filesystem portability

The XDG application-data rule already chosen for Faceledger is portable across
mainstream Linux desktops: `$XDG_DATA_HOME` is the user data base directory and
defaults to `$HOME/.local/share` when unset.
[XDG Base Directory Specification](https://specifications.freedesktop.org/basedir/0.8/)

Faceledger's recoverable cache trash remains its own application-data subtree.
It is not the freedesktop desktop Trash service, so broad Linux support does not
require desktop-environment integration or change the settled manifest-backed
recovery behaviour.

## Recommendation

Replace “Ubuntu-only” with “tested on mainstream glibc x86-64 Linux” and publish
the exact validated environments with each release. Treat the suggested 75%
goal as motivation for the three-family test matrix, not as a measurable product
claim. Before committing that scope, run a packaging spike that resolves and
locks DeepFace 0.0.100's full dependency graph, chooses the managed Python
version and ABI floor, and proves all eleven recognition models on the proposed
matrix.
