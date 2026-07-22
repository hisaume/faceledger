# Use the face tree as the source of truth

Faceledger treats the user-managed folder hierarchy, recognized filenames, and
model-qualified NPY files as its source of truth rather than building a central
catalogue. It validates cache entries only for basic readability, numeric type,
and expected model dimensions, leaving provenance and freshness management to
the user; this preserves a simple portable folder-based workflow at the known
cost of weak stale-cache detection and database-management capabilities.
