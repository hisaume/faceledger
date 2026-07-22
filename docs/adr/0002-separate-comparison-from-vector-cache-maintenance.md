# Separate comparison from vector-cache maintenance

Comparison runs may reuse compatible cached vectors or calculate missing vectors
transiently, but they never create, overwrite, or remove cache entries. Explicit
vector-cache maintenance owns persistence, and cleanup writes a manifest before
moving selected entries into recoverable timestamped trash; this trades the
convenience of automatic caching for predictable read-only comparison and makes
the destructive-looking operation reversible.
