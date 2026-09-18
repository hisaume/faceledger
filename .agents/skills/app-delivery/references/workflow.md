# Faceledger Beads workflow

## Shape and publish work

1. Use `grill-with-docs` where product or domain choices remain unresolved.
2. Use `to-spec` once scope is settled.
3. Use `to-tickets` to create independently verifiable vertical slices.
4. Record only genuine prerequisite relationships with `bd dep add`.
5. Confirm the frontier with `bd ready --explain`.

## Implement a claimed slice

1. Refresh tracker context with `bd prime`.
2. Select ready work with `bd ready`, inspect it with `bd show <id>`, and claim
   it atomically with `bd update <id> --claim`.
3. Keep discoveries durable: update the issue or create a linked Beads follow-up.
4. Use test-first changes for behavioural work.
5. Run focused feedback during development, then `./scripts/check.sh`.

Do not claim blocked or already-assigned work. Do not edit Beads JSONL directly;
the local Dolt database is authoritative.

## Delivery and completion

When the current task authorises a pull request:

1. Create one branch and one pull request for the claimed slice.
2. Store the pull-request URL with `bd update <id> --external-ref="<PR URL>"`.
3. Keep the issue assigned and `in_progress` while review or checks are pending.
4. Close it only after the pull request has merged and its required checks pass:
   `bd close <id> --reason="Merged in <PR URL>"`.
5. Use `bd close <id> --suggest-next` to identify newly unblocked work.

Use `bd dolt push` or `bd dolt pull` only when explicitly authorised.
