# Late amendment: runtime device selection and TensorFlow startup output

Amended on 15 September 2026. This document overlays
[the version-one CLI specification](faceledger-v1-cli.md) where it differs.
All other version-one behavior remains unchanged.

## Decision

Faceledger defaults to CPU execution by assigning `CUDA_VISIBLE_DEVICES=-1`
only when the caller has not provided that environment variable. A caller may
provide any CUDA-native visibility value before invoking Faceledger. The value
is passed through without validation and must be set before TensorFlow is
imported into the process.

This is a best-effort, unqualified override. Faceledger continues to support
and release-qualify only the locked CPU runtime; it makes no GPU compatibility,
performance, output-equivalence, or vector-cache compatibility claim.
Release qualification continues to force CPU visibility regardless of the
caller environment.

Before DeepFace imports TensorFlow, Faceledger defaults
`TF_CPP_MIN_LOG_LEVEL` to `3` when the caller has not set it. A caller can set
`TF_CPP_MIN_LOG_LEVEL=0` before launch to restore native TensorFlow startup
diagnostics. Faceledger filters only the known `tf.losses.sparse_softmax_cross_entropy`
legacy deprecation warning. It does not alter `TF_ENABLE_ONEDNN_OPTS` and does
not persist, deduplicate, or file-descriptor-filter native TensorFlow output.

## What this supersedes

This amendment supersedes the version-one CLI specification's exclusion of
user-selectable GPU execution only to the extent of caller-owned
`CUDA_VISIBLE_DEVICES` selection. It does not add a Faceledger GPU option,
configuration file, GPU support promise, or GPU release qualification.

## Reason

The standard TensorFlow environment variables work consistently for shells,
debuggers, containers, services, and programmatic callers. CPU remains the
safe default while advanced callers retain control of the runtime they provide.
Quiet defaults remove routine startup noise without risking a persistent native
stderr filter that could conceal meaningful diagnostics.
