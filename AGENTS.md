# Faceledger agent guide

## Sources of truth

- Follow the current user's instructions and the repository's active safety rules.
- Use the terminology and domain invariants in `CONTEXT.md`.
- Before changing behaviour, read relevant ADRs under `docs/adr/` and the
  specification status in `docs/specs/README.md`.
- Durable project work is tracked in Beads. For a tracked product or engineering
  change, use the `app-delivery` skill.

## Python and validation

- Use `uv` only. `pyproject.toml` and `uv.lock` are the dependency source of truth.
- The supported project interpreter is CPython 3.12; create environments with
  `uv sync --locked` and run commands through `uv run --locked`.
- Treat `.venv/` as disposable generated state. Do not use system or user-site
  `pip`, and do not invoke `.venv/bin/python` directly.
- Run `./scripts/check.sh` after changing Python code and before handoff. It is
  the authoritative full lint, format, type, test, and distribution check.
- Keep `scripts/check.sh` and CI aligned when adding or moving maintained Python
  code or quality checks. Do not weaken checks merely to make them pass.

## Engineering policy

- Keep direct `deepface` imports and DeepFace-specific behaviour behind the
  adapter boundary. Return typed, application-level data from that boundary.
- Use the glossary’s canonical terms in code, tests, issues, and user-facing
  text. Surface conflicts with an ADR or invariant rather than silently changing
  the meaning.
- Add concise docstrings to non-trivial production functions when their intent
  is not clear from their name and structure. Do not add routine docstrings to
  tests, fixtures, callbacks, or trivial accessors.
- Update `docs/reference/core-api.md` when a documented core module’s
  responsibility or public behaviour changes.

## Delivery authority

- Beads is the durable source for issues, dependencies, discoveries, and
  handoff state; do not create project markdown TODO or memory files.
- Do not commit, push, open or merge pull requests, or run `bd dolt push/pull`
  unless the current user request explicitly includes that action.
- Report changed files, validation performed, issue state, and remaining work
  at handoff.
