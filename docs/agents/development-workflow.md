# Planning and delivery workflow

This repository uses Beads as the durable coordination layer from product
planning through pull-request integration. The purpose of the dependency graph
is to let multiple agents work safely without exposing a downstream slice as
ready before its prerequisite has actually merged.

## 1. Shape the product and domain

1. Run `grill-with-docs` against the current design seed.
2. Resolve consequential product and domain questions through the interview.
3. Let the domain-modeling workflow update `CONTEXT.md` and create ADRs only as
   terms and decisions settle.
4. Keep unresolved matters visible; do not turn them into implementation
   assumptions silently.

## 2. Publish the specification and slices

1. Run `to-spec` after grilling is complete. Publish the resulting specification
   to Beads according to `docs/agents/issue-tracker.md`.
2. Run `to-tickets` to create structured, independently valuable vertical-slice
   issues in Beads.
3. Give every slice observable acceptance criteria and explicit dependencies.
   When slice B must follow slice A, record B as depending on A. Independent
   slices should not be linked merely to force serialization.
4. Confirm with `bd ready --explain` that the graph exposes only genuinely
   claimable work.

## 3. Implement one slice test-first

1. Select work through `bd ready` and inspect it with `bd show <id>`.
2. Claim it atomically with `bd update <id> --claim` before changing code. Do not
   take blocked work or work claimed by another agent.
3. Invoke the `tdd` skill for every implementation slice and follow its
   red-green-refactor workflow throughout that slice.
4. Keep discoveries durable: update the current issue or create linked Beads
   follow-up issues rather than maintaining markdown TODO lists.
5. Run the slice's relevant tests and repository quality gates before delivery.

## 4. Deliver through a pull request

For a claimed slice, the implementing agent has standing authority to:

- commit the completed slice on its working branch;
- push that slice branch; and
- open one ready-for-review pull request for the slice.

This authority is limited to the claimed slice. It does not authorize merging
the pull request, pushing unrelated work, rewriting shared history, or syncing
the Beads Dolt remote without authorization.

After opening the pull request:

1. Store its URL on the same slice with
   `bd update <id> --external-ref="<PR URL>"` and append a concise handoff note.
2. Leave the slice `in_progress` and assigned. Do not close it merely because
   implementation is complete or review has begun.
3. Address review feedback under the same slice and pull request, continuing to
   use test-first changes where behaviour changes.

The open slice remains an active blocker. Any downstream slice that depends on
it stays out of `bd ready`, while unrelated ready slices remain available to
other agents.

## 5. Merge gate and completion

- A human reviews and merges the pull request. Implementing agents must not
  merge their own pull requests under this standing workflow.
- Approval alone is not completion. The pull request must be verified as merged
  and all required checks must have passed.
- After merge verification, close the original slice issue with a reason that
  identifies the merged pull request. `bd close <id> --suggest-next` may be used
  to reveal newly unblocked work.
- If review requests changes, checks fail, or the pull request closes without
  merging, keep the slice open. It continues to block its dependants until the
  intended integration outcome is resolved.

There is no separate review-gate issue by default. The original vertical-slice
issue is the single source of truth from claim through merge.
