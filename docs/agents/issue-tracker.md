# Issue tracker: Beads

Issues, specifications, and durable project tasks for this repository live in
the local Beads Dolt database under `.beads/`. Use the `bd` CLI for all issue
operations.

The GitHub remote stores code under normal Git refs and Beads synchronization
data separately under `refs/dolt/data`. `.beads/issues.jsonl` is a passive
export, not the source of truth or the normal synchronization mechanism.

## Conventions

- Refresh workflow context: `bd prime`
- Find ready work: `bd ready`
- List open work: `bd list --status=open`
- Read an issue: `bd show <id>`
- Create an issue: `bd create --title="..." --description="..." --type=task|bug|feature --priority=2`
- Claim an issue: `bd update <id> --claim`
- Update details: `bd update <id> --title="..." --description="..." --notes="..." --design="..."`
- Link a pull request: `bd update <id> --external-ref="<PR URL>"`
- Add a dependency: `bd dep add <blocked-id> <blocker-id>`
- Close completed work: `bd close <id> --reason="..."`
- Synchronize when authorized: `bd dolt push` or `bd dolt pull`

Do not use interactive `bd edit`. Do not replace Beads with markdown TODO
files, ad hoc memory files, GitHub Issues, or direct JSONL editing.

## When a skill says "publish to the issue tracker"

Create a Beads issue. Put the durable problem and requested outcome in its
description, implementation decisions in its design field, and measurable
completion conditions in its acceptance criteria when applicable.

## When a skill says "fetch the relevant ticket"

Run `bd show <id>`.

## Work lifecycle

Inspect and claim an issue before implementation. Create follow-up Beads issues
for durable work discovered during implementation. Close an issue only after
its outcome is complete and its relevant quality and integration gates pass.

The repository uses the conservative profile by default. The slice-delivery
workflow grants limited standing authority to commit, push the slice branch,
and open its pull request. It does not authorize merging, unrelated pushes, or
Dolt synchronization. See `docs/agents/development-workflow.md`.
