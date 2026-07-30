# Faceledger v1 Linux qualification evidence

Qualification completed on 30 July 2026 with uv 0.11.32 and `uv.lock`
SHA-256 `fa8bb82406abb104dacd23bf4916082278afdfbb8b2e8a7ab9e40036e5146809`.
All environments used CPython 3.12.13, DeepFace 0.0.100, TensorFlow 2.21.0,
tf-keras 2.21.0, CPU-only execution, and the two-model scope amendment.

| Environment | Pinned base image | glibc | First use | Offline |
| --- | --- | --- | --- | --- |
| Ubuntu 26.04 LTS | `ubuntu@sha256:3131b4cc82a783df6c9df078f86e01819a13594b865c2cad47bd1bca2b7063bb` | 2.43 | pass | pass |
| Debian 13 | `debian@sha256:fac46bff2e02f51425b6e33b0e1169f55dfb053d83511ca28aa50c09fd5ed7a4` | 2.41 | pass | pass |
| Fedora 44 | `fedora@sha256:6c75d5bf57cb0fa5aa4b92c6a83c86c791644496d9ac230de7711f5b8ec3b898` | 2.43 | pass | pass |
| Arch 2026-07-26 | `archlinux@sha256:3406a568f45d68f0bef35dc80b3eacec8bda59b0292b2e50d5932ba1667f20cf` | 2.43 | pass | pass |

Each of the eight reports records six successful raw model/format checks, six
successful public-operation sequences, no failed checks, no asset-lifecycle
errors, and no runtime-contract errors. Every public sequence included
uncached comparison, cache build, cache reuse without recalculating the target,
cache rebuild, and recoverable trash.

The first-use environments were independently empty and downloaded the same
three assets after Faceledger announced them. Their observed values were:

| Asset | Bytes | Observed SHA-256 |
| --- | ---: | --- |
| `facenet512_weights.h5` | 94,955,648 | `3f76b5117a9ca574d536af8199e6720089eb4ad3dc7e93534496d88265de864f` |
| `arcface_weights.h5` | 137,026,640 | `6336979c0c602cae08d1122a66f4dfb862d059bbcd8ef80306aef2b2249b0c93` |
| `retinaface.h5` | 118,667,368 | `ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf` |

These are Faceledger observations, not upstream-published checksums or rights
claims. The H5 files remain only in external qualification workspaces; they are
not present in this repository or any qualification image.

The bare Ubuntu image first exposed an `ImportError` for `libGL.so.1` while
importing OpenCV. The checked-in Dockerfiles add the minimum distribution
packages subsequently used by the successful matrix. The local derived-image
IDs used for this run were:

- Ubuntu: `sha256:5ad196e3d9dc852c2ea554f45abe970721972c00e53a0301c5a065aae1a425b1`
- Debian: `sha256:d4e6baff40a4bbd382103a9a269578e53d5f9f5ca9331cff471b47475e56c4cc`
- Fedora: `sha256:924bc58747ed587c4b98c49890903d2aa7fa3c858776060e98f753ff537f51d7`
- Arch: `sha256:255ad08b4e4bb00ca6113e6dd1610dd81698b3eab4bf6886c7523fbcc5b9f924`

The qualification fixture has SHA-256
`ea7521adfaae94354cb73fbca6ff1def690767881fe8825be5d27c8f2b30aff9`;
its generation provenance is recorded in
[`tests/runtime/fixtures/README.md`](../../../../tests/runtime/fixtures/README.md).
