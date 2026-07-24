# Latest DeepFace with TensorFlow/tf-keras

**Research date:** 2026-07-24  
**Scope:** released packages and upstream source only; no runtime experiment was performed for this note.

This review informed the
[late version-one model-scope amendment](../specs/faceledger-v1-model-scope-amendment.md),
which removes the `DeepFace` recognition model while retaining the DeepFace
library.

## Conclusion

No. The latest released DeepFace is still **0.0.100** (released 2026-05-09), and pairing it with matching **TensorFlow 2.21 / tf-keras 2.21** does not remove the Q17 blocker for the recognition model named `DeepFace`. PyPI's [release history](https://pypi.org/project/deepface/) identifies 0.0.100 as current.

`tf-keras` solves the general Keras 3 routing problem, but DeepFace 0.0.100 contains its own earlier failure:

1. TensorFlow 2.16+ uses Keras 3 by default. The Keras project says legacy Keras 2 requires installing `tf_keras`, setting `TF_USE_LEGACY_KERAS=1`, and doing so **before any TensorFlow import** ([Keras guidance](https://keras.io/getting_started/#tensorflow-keras-2-backwards-compatibility); [TensorFlow 2.16 announcement](https://blog.tensorflow.org/2024/03/whats-new-in-tensorflow-216.html)).
2. DeepFace sets that variable before its own TensorFlow import and checks that `tf_keras` is importable ([bootstrap](https://github.com/serengil/deepface/blob/v0.0.100/deepface/DeepFace.py#L1-L41); [validation](https://github.com/serengil/deepface/blob/v0.0.100/deepface/commons/package_utils.py#L31-L49)).
3. Nevertheless, the `DeepFace` model adapter unconditionally raises for every TensorFlow 2.x minor greater than 12, before constructing the model ([adapter guard](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/FbDeepFace.py#L42-L68)). TensorFlow 2.21 therefore fails even when correctly routed to tf-keras 2.21.
4. That guard is now broader than the API it is trying to protect: tf-keras 2.21 still exports `LocallyConnected2D` ([tf-keras 2.21 source](https://github.com/keras-team/tf-keras/blob/v2.21.0/tf_keras/layers/__init__.py#L44)). This does not make the released adapter usable, because the guard runs first. Removing the guard might permit construction, but source inspection alone cannot establish H5-weight loading or numerical compatibility.

## Recognition-model assessment

| Models | Source-based assessment for TensorFlow/tf-keras 2.21 | Exact reason |
| --- | --- | --- |
| `DeepFace` | **Definitely fails** | Its constructor raises `ValueError` whenever TensorFlow is 2.13 or newer, before importing `LocallyConnected2D` or loading weights. |
| `VGG-Face`, `Facenet`, `Facenet512`, `OpenFace`, `DeepID`, `ArcFace`, `GhostFaceNet` | **No comparable model-specific blocker found; not proven compatible by this review** | Their released adapters select `tensorflow.keras` on TensorFlow 2 and contain no TensorFlow-minor rejection like `DeepFace` ([VGG-Face](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/VGGFace.py), [FaceNet](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Facenet.py), [OpenFace](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/OpenFace.py), [DeepID](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/DeepID.py), [ArcFace](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/ArcFace.py), [GhostFaceNet](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/GhostFaceNet.py)). They still require correct legacy-Keras routing and runtime qualification of model/weight behavior. |
| `Dlib`, `SFace`, `Buffalo_L` | **Not affected by the Keras 2/3 switch** | These adapters use dlib, OpenCV DNN, and InsightFace/ONNX Runtime respectively, rather than `tensorflow.keras` ([Dlib](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Dlib.py), [SFace](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/SFace.py), [Buffalo_L](https://github.com/serengil/deepface/blob/v0.0.100/deepface/models/facial_recognition/Buffalo_L.py)). Their optional native/runtime dependencies remain separate qualification concerns. |

There is also a package-wide integration risk for every TensorFlow-backed adapter: if the host imports TensorFlow before importing DeepFace, DeepFace's later environment assignment cannot switch that already-loaded runtime to legacy Keras. Its validation only proves that the `tf_keras` package is installed, not that `tf.keras` was routed to it.

Finally, installation is not self-contained. DeepFace 0.0.100 declares broad `tensorflow>=1.9.0` and `keras>=2.2.0` dependencies but does **not** declare `tf-keras` ([released requirements](https://github.com/serengil/deepface/blob/v0.0.100/requirements.txt)). Conversely, `tf-keras==2.21.0` requires `tensorflow>=2.21,<2.22` ([PyPI metadata](https://pypi.org/pypi/tf-keras/2.21.0/json)). A consumer must therefore add and align `tf-keras` explicitly.

## Decision impact

DeepFace 0.0.100 plus tf-keras 2.21 is not a replacement for Q17's TensorFlow 2.12 compatibility lock if all eleven recognition models remain required. A future upgrade candidate would need at least an upstream release that removes or narrows the `DeepFace` adapter guard, followed by runtime qualification of all required models and weight-loading paths.
