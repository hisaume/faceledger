# Use managed CPython 3.11 and TensorFlow 2.12 for the locked runtime

_Status: Superseded by the
[late version-one model-scope amendment](../specs/faceledger-v1-model-scope-amendment.md)._

Faceledger version one uses managed CPython 3.11.15 with TensorFlow 2.12.1 and
Keras 2.12.0. A CPython 3.12 spike resolved TensorFlow 2.21 and passed ten
recognition models, but DeepFace's `DeepFace` adapter requires the
`LocallyConnected2D` layer removed after TensorFlow 2.12; CPython 3.11 is the
newest interpreter line with a compatible TensorFlow wheel, so the older
runtime is required to preserve all eleven models in the pinned DeepFace
0.0.100 vector profile.
