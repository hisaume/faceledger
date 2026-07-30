# Third-party notices and model-asset boundary

Faceledger's locked environment installs third-party Python distributions from
their upstream packages. The project does not vendor their source or binaries;
their installed package metadata retains the applicable notices. In
particular, DeepFace 0.0.100 and RetinaFace 0.0.17 publish MIT-licensed wrapper
code. Their licences do not by themselves establish rights in separately
downloaded model weights.

Faceledger does not ship these dependency-owned assets:

- `facenet512_weights.h5`
- `arcface_weights.h5`
- `retinaface.h5`

DeepFace downloads missing assets from its normal upstream locations after
Faceledger announces the acquisition. This mechanism is not a grant of rights.
The exact H5 files have no reviewed provenance manifest establishing commercial
use or redistribution permission. Do not include them in a Faceledger source
archive, wheel, native package, container, mirror, example, or offline
wheelhouse without separate artifact-specific permission.

The complete source and artifact review, including upstream links, observed
hashes, and the limits of the available evidence, is recorded in
[`docs/research/faceledger-v1-model-licensing.md`](docs/research/faceledger-v1-model-licensing.md).
