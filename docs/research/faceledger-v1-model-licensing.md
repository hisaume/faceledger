# Faceledger v1 model and weight licensing review

_Research date: 24 July 2026_

_Scope: the late v1 amendment only — DeepFace 0.0.100 with Facenet512 and
ArcFace recognition, plus `retina-face` 0.0.18 detection/alignment. This is a
source and terms review, not legal advice._

## Conclusion for v1

Faceledger can distribute its integration code and declare the `deepface` and
`retina-face` Python dependencies subject to their MIT notices. It should **not
bundle, mirror, or redistribute** `facenet512_weights.h5`,
`arcface_weights.h5`, or `retinaface.h5` in v1. The three H5 files are separate
release assets, and their host repository expressly says that its MIT licence
does not settle the wrapped models: model licences are inherited from their
original sources. [DeepFace model-assets repository](https://github.com/serengil/deepface_models),
[DeepFace 0.0.100 licence notice](https://github.com/serengil/deepface/blob/v0.0.100/README.md#licence)

Use DeepFace's existing first-use, library-managed download path and make that
acquisition visible to the user. Record the expected filename, source URL, and
observed digest, but do not describe a digest as an upstream licence or an
upstream-published checksum. This avoids Faceledger itself redistributing the
files; it does **not** establish that every downstream use is licensed.

The reviewed sources do not establish commercial-use permission for any of the
three exact H5 artifacts. ArcFace and RetinaFace are especially unsuitable for
a commercial-use claim because their stated lineage reaches InsightFace, whose
current policy limits provided training data and trained models to
non-commercial research. Facenet512's exact checkpoint/data mapping and weight
terms are not documented sufficiently to establish commercial permission.
[InsightFace licence policy](https://github.com/deepinsight/insightface#license)

Because the v1 amendment fixes RetinaFace as the detector, this is not merely
an optional-model caveat: the reviewed runtime as a whole is not commercially
cleared without separate permission or a replacement detector weight.

Accordingly, v1 documentation should make no model-licensing or commercial-use
guarantee. A distribution that promises commercial use needs written clearance
for these exact artifacts, or replacement weights trained from data and code
with documented compatible rights. Merely moving the download to the user's
machine is not that clearance.

Because RetinaFace is fixed for detection and alignment in every v1 vector
profile, this limitation cannot be avoided by choosing Facenet512 instead of
ArcFace. Faceledger's code may be distributed, but the reviewed v1 inference
runtime as a whole is not cleared for a commercial-use claim without separate
permission or a requalified replacement detector.

## Exact runtime artifacts

DeepFace 0.0.100's adapters and RetinaFace loader resolve these files from the
same GitHub release. The sizes and SHA-256 values are observations from the Q17
spike, retained to identify what was tested.

| Consumer | Runtime file | Exact adapter URL | Observed identity |
| --- | --- | --- | --- |
| Facenet512 | `facenet512_weights.h5` | `https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5` | 94,955,648 bytes; SHA-256 `3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f` |
| ArcFace | `arcface_weights.h5` | `https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5` | 137,026,640 bytes; SHA-256 `6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93` |
| RetinaFace | `retinaface.h5` | `https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5` | 118,667,368 bytes; SHA-256 `ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf` |

Sources: [DeepFace 0.0.100 Facenet adapter](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Facenet.py),
[DeepFace 0.0.100 ArcFace adapter](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/ArcFace.py),
[RetinaFace loader](https://github.com/serengil/retinaface/blob/master/retinaface/model/retinaface_model.py),
[v1.0 release assets](https://github.com/serengil/deepface_models/releases/tag/v1.0),
[Q17 observed evidence](evidence/deepface-runtime-host-offline.json).

The adapters download a missing file into `.deepface/weights` and reuse a
present file. They do not bring model weights inside the Python wheel. DeepFace
0.0.100's common downloader checks presence but does not pin or verify a
published content digest. [DeepFace weight downloader](https://github.com/serengil/deepface/blob/v0.0.100/deepface/commons/weight_utils.py)

## Code and model-definition licences

- `deepface` 0.0.100 code is MIT. If Faceledger redistributes that code or a
  substantial copy, retain its copyright and MIT permission notice. The
  project itself warns that external model licences are inherited and must be
  checked for production use. [DeepFace MIT licence](https://github.com/serengil/deepface/blob/v0.0.100/LICENSE),
  [DeepFace 0.0.100 package page](https://pypi.org/project/deepface/0.0.100/)
- The FaceNet adapter says its Inception-ResNet-v1 definition is heavily
  inspired by David Sandberg's FaceNet implementation. That implementation is
  MIT. This clears the cited software implementation subject to its notice; it
  does not by itself license the separately hosted DeepFace H5 file.
  [DeepFace FaceNet adapter](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Facenet.py),
  [Sandberg FaceNet licence](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md)
- The ArcFace adapter constructs a ResNet34-derived Keras model. DeepFace's
  author describes it as a third-party Keras reimplementation of InsightFace's
  MXNet work and describes extracting the separately shared weights from a
  monolithic model. InsightFace code is MIT, but InsightFace expressly
  separates permissively licensed code from restricted data and trained
  models. [DeepFace ArcFace adapter](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/ArcFace.py),
  [author's ArcFace implementation note](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/),
  [InsightFace licence policy](https://github.com/deepinsight/insightface#license)
- `retina-face` package code is MIT and credits the original InsightFace model
  and Stanislas Bertrand's MIT TensorFlow 2 reimplementation. Its README says
  the main structure and pretrained weights remain the same as the reference
  model. Those statements establish code provenance, but they do not attach a
  separate, unambiguous licence to the exact `retinaface.h5` release asset.
  [RetinaFace project and licence](https://github.com/serengil/retinaface),
  [RetinaFace-tf2 project](https://github.com/StanislasBertrand/RetinaFace-tf2),
  [RetinaFace-tf2 MIT licence](https://github.com/StanislasBertrand/RetinaFace-tf2/blob/master/LICENSE)

## Weight and training-data lineage

### Facenet512

The model-assets repository says the author converted FaceNet's original
weights to Keras and that the original model's licence is inherited. It does
not give the release asset its own model card, checkpoint identifier, training
recipe, dataset manifest, or weight-specific licence.
[DeepFace model-assets repository](https://github.com/serengil/deepface_models)

The cited Sandberg project offers two Inception-ResNet-v1 pretrained models: a
CASIA-WebFace model and the higher-performing 512-dimensional model trained on
VGGFace2. It asks users of the models to credit the training-data providers.
That makes the VGGFace2 checkpoint the best available lineage for DeepFace's
512-dimensional conversion, but DeepFace does not explicitly map
`facenet512_weights.h5` to checkpoint `20180402-114759`; treat that mapping as
probable, not proven. [Sandberg FaceNet pretrained models](https://github.com/davidsandberg/facenet#pre-trained-models)

The surviving official VGGFace2 page says the images were downloaded from
Google Image Search and that dataset downloads are no longer available, but it
does not state terms for derivative model weights. The DeepFace asset therefore
has no affirmative weight-redistribution or commercial-use grant traceable
through the reviewed first-party sources. [Official VGGFace2 page](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)

**Disposition:** allow first-use upstream download with notices; do not bundle
or mirror; credit DeepFace, FaceNet, and VGGFace2; do not claim commercial use.
Commercial permission and the exact checkpoint mapping remain blockers.

### ArcFace

The model-assets repository again says the weight was converted from its
original source and inherits that source's licence. The architecture, training,
and accuracy values match the ResNet34
`CASIA, E40` entry in the third-party Keras InsightFace project that the
DeepFace author says supplied the monolithic model from which he extracted
weights. This is strong lineage evidence, but it is still an inference because
the H5 release has no provenance manifest: no reviewed source defines `E40`,
identifies the exact source checkpoint, or attaches its terms to the H5
artifact. That Keras project draws a clear line between MIT implementation code
and training data/models restricted to non-commercial research.
[DeepFace model-assets repository](https://github.com/serengil/deepface_models),
[Keras InsightFace licence and model table](https://github.com/leondgarse/Keras_insightface#license),
[DeepFace author's conversion note](https://sefiks.com/2020/12/14/deep-face-recognition-with-arcface-in-keras-and-python/)

InsightFace's current policy states that its code is MIT but its provided
training data, annotations, and models trained with that data are available
only for non-commercial research; it says this applies to both manual and
automatic model downloads. The model zoo repeats that all its models are for
non-commercial research only. The exact DeepFace conversion is not clearly
identified as an InsightFace-hosted checkpoint, so this policy cannot repair
the missing chain; it is, at minimum, a strong reason not to claim commercial
permission. [InsightFace licence policy](https://github.com/deepinsight/insightface#license),
[InsightFace model-zoo terms](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)

**Disposition:** allow first-use upstream download only for a distribution that
makes the unresolved/non-commercial lineage clear; do not bundle or mirror;
credit DeepFace and the ArcFace/InsightFace work; do not claim commercial use.
Written permission for the exact artifact, or a replacement with documented
commercially compatible training provenance, is the blocker.

### RetinaFace

`retina-face` says it is a simplified TensorFlow packaging of the InsightFace
detector, heavily inspired by RetinaFace-tf2, and that the reference model's
structure and pretrained weights are the same. RetinaFace-tf2 identifies a
ResNet50 model evaluated on WIDER FACE. The original InsightFace training
instructions also identify WIDER FACE and a RetinaFace-R50 pretrained model.
[Serengil RetinaFace project](https://github.com/serengil/retinaface),
[RetinaFace-tf2 README](https://github.com/StanislasBertrand/RetinaFace-tf2),
[original InsightFace RetinaFace README](https://github.com/deepinsight/insightface/blob/master/detection/retinaface/README.md)

Neither the `deepface_models` release nor the wrapper supplies an
artifact-specific model card or terms for `retinaface.h5`. The wrapper and
reimplementation repositories being MIT clears their software under the MIT
conditions, but does not resolve the separate hosted weight after
`deepface_models` has expressly deferred model licensing to original sources.
InsightFace's current provided-model policy is non-commercial research only.

**Disposition:** allow first-use upstream download only with the same licence
warning; do not bundle or mirror; retain MIT notices for code and cite the
RetinaFace works; do not claim commercial use. Written permission or a detector
weight with documented commercially compatible training provenance is the
blocker.

## Distribution decision matrix

| Action | Facenet512 | ArcFace | RetinaFace |
| --- | --- | --- | --- |
| Declare/install Python package dependency | Yes, subject to package notices | Yes, subject to package notices | Yes, subject to package notices |
| Upstream first-use download by the installed library | Mechanically supported v1 path; not a rights clearance | Mechanically supported; only for uses allowed by upstream terms | Mechanically supported; only for uses allowed by upstream terms |
| Faceledger bundle, installer payload, mirror, or offline wheelhouse containing H5 | No | No | No |
| Claim that redistribution is permitted | Not established | Not established | Not established |
| Claim commercial model use is permitted | Not established | No basis; upstream lineage is non-commercial/restricted | No basis; upstream lineage is non-commercial/restricted |
| Attribution/notice | DeepFace, FaceNet, VGGFace2; MIT notices for copied code | DeepFace, ArcFace/InsightFace; MIT notices for copied code | `retina-face`, RetinaFace/InsightFace and RetinaFace-tf2; MIT notices for copied code |

## Required product treatment

1. Keep all three weights out of Faceledger source archives, wheels, native
   packages, container images, caches shipped as examples, and release mirrors.
2. Announce first-use acquisition before allowing the dependency to start it,
   identify the three upstream artifacts in product documentation, and state
   that Faceledger has not established redistribution or commercial-use rights.
   Preserve the existing cancellation behavior; do not add a licence-acceptance
   prompt that implies Faceledger can grant the upstream rights.
3. Do not silently substitute another mirror.
4. Record and check the observed digests in release qualification evidence,
   while clearly labelling them as Faceledger observations rather than
   upstream-published checksums. Do not turn this into a second runtime asset
   manager alongside DeepFace.
5. Ship the MIT notices for redistributed DeepFace/RetinaFace code as required,
   and include the attribution named above in third-party notices.
6. For a future commercial-capable or offline-bundled release, obtain terms tied
   to each exact file and its training provenance, or replace it and requalify
   embeddings, thresholds, and cache compatibility.
