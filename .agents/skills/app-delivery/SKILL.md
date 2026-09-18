---
name: app-delivery
description: Use when planning, implementing, reviewing, or handing off a durable application change tracked in Beads, including shaping a feature, turning approved scope into tickets, or delivering a claimed vertical slice. Do not use for an informational question or a read-only code inspection.
---

# Application delivery

This skill composes the repository’s generic Beads, domain-modeling,
grill-with-docs, to-spec, to-tickets, TDD, and code-review skills. It owns the
application-specific order of work; those skills retain ownership of their
individual procedures.

## First read

1. Read `CONTEXT.md`.
2. Read ADRs relevant to the proposed change.
3. Read `docs/specs/README.md`.
4. Read [the workflow reference](references/workflow.md).

## Choose the route

- For unclear or consequential product/domain scope, use `grill-with-docs`.
  Capture settled vocabulary in `CONTEXT.md` and only durable trade-offs in ADRs.
- For approved scope without a specification, use `to-spec`.
- For an approved specification, use `to-tickets`, then record real Beads
  dependency edges.
- For a claimed implementation slice, use `tdd`; use `code-review` before
  delivery when independent review is requested or the change is substantial.

## Boundaries

- Beads records durable work. Local plans are only temporary execution aids.
- A pull request is not completion. Keep its Beads issue open until its intended
  change is merged and verified.
- This skill never expands the authority stated in `AGENTS.md`.
