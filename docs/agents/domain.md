# Domain docs

How the engineering skills should consume this repository's domain
documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repository root.
- `docs/adr/` for decisions touching the area about to be changed.

If either location does not exist, proceed silently. Do not create it merely
because it is absent. The domain-modeling workflow creates domain terms and
ADRs lazily when decisions are actually resolved.

## File structure

This is a single-context repository:

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
│       ├── 0001-example-decision.md
│       └── 0002-another-decision.md
└── src/
```

`CONTEXT.md` holds the shared domain glossary. `docs/adr/` holds project-wide
architectural and design decisions.

## Use the glossary's vocabulary

When output names a domain concept—in an issue, design proposal, test, or
implementation—use the term defined in `CONTEXT.md`. Do not drift to synonyms
that the glossary explicitly avoids.

If a required concept is missing, reconsider whether new language is necessary.
When it represents a genuine domain gap, record it through the domain-modeling
workflow.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly
rather than silently overriding the prior decision.
