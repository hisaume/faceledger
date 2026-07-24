# Late amendment: narrow the version-one recognition-model scope

Amended on 24 July 2026. This file is kept beside the original
[Faceledger version-one specification](faceledger-v1.md) so the approved plan
and the later change of direction remain visible together. Where they differ,
this amendment controls; every other part of the original specification is
unchanged.

## Decision

Version one supports exactly two recognition models:

- `Facenet512`, which remains the default model, with cache slug `facenet512`
  and DeepFace 0.0.100's cosine threshold `0.30`.
- `ArcFace`, with cache slug `arcface` and DeepFace 0.0.100's cosine threshold
  `0.68`.

The recognition model named `DeepFace` is removed from the version-one model
set. This does **not** remove the DeepFace library: Faceledger remains pinned to
the `deepface` package at version 0.0.100. The other recognition models
advertised by that library are deferred. They may be introduced on demand in a
later Faceledger version when a model provides a distinct benefit worth its
compatibility, dependency, licensing, and release-qualification cost.

The version-one runtime baseline is:

| Dependency | Version |
| --- | --- |
| CPython | 3.12 |
| `deepface` | 0.0.100 |
| `tensorflow` | 2.21.0 |
| `tf-keras` | 2.21.0 |

The repository selects managed CPython 3.12.13 as the current exact patch
release for the Python 3.12 baseline. TensorFlow and tf-keras are an explicitly
matched pair; `tf-keras` is a direct dependency because DeepFace does not
declare it itself.

## What this supersedes

References in the original specification and planning tickets to the complete
eleven-model DeepFace registry now mean the two-model version-one set above.
Release qualification covers RetinaFace with alignment plus Facenet512 and
ArcFace across JPEG, PNG, and static WebP. Cache-model choices, cache slugs,
default thresholds, licensing review, and model-asset acquisition likewise
apply only to those two recognition models for version one.

This amendment supersedes the complete-registry clause of
[ADR 0001](../adr/0001-pin-deepface-0-0-100.md) and all of
[ADR 0006](../adr/0006-use-cpython-3-11-and-tensorflow-2-12.md). It does not
alter RetinaFace detection, enabled alignment, vector-cache semantics,
comparison behavior, filesystem conventions, diagnostics, maintenance,
privacy, or the supported Linux envelope.

## Reason

The packaging spike showed that DeepFace's advertised model registry is wider
than the compatibility surface qualified by its release. Supporting every
advertised adapter would make Faceledger carry dependency and release-testing
cost without a corresponding version-one benefit. Facenet512 and ArcFace were
the primary target models from the outset, and both work with the modern
TensorFlow/tf-keras route. Narrowing the set preserves the intended product
while making future runtime upgrades and qualification materially simpler.

The evidence leading to this change is retained in the
[runtime spike note](../research/deepface-runtime-baseline.md) and the
[DeepFace/tf-keras compatibility review](../research/latest-deepface-tf-keras-compatibility.md).
