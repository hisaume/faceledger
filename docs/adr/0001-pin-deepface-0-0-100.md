# Pin DeepFace 0.0.100 as the embedding compatibility boundary

Faceledger pins `deepface==0.0.100` and supports that release's complete
face-recognition model registry. This trades automatic access to future DeepFace
models for reproducible model names, thresholds, and persisted NPY embeddings;
any DeepFace upgrade must deliberately review those compatibility assumptions
before expanding or changing the supported set.

The version pin remains active, but the complete-registry scope was superseded
for version one by the
[late model-scope amendment](../specs/faceledger-v1-model-scope-amendment.md).
