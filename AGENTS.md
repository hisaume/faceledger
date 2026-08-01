# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [SYNC_CONCEPTS.md](https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Python Environment

- Use **uv** for Python version, virtual-environment, and dependency management.
- The project Python version is pinned by `.python-version` to **CPython 3.12.13**.
- `pyproject.toml` and `uv.lock` are the dependency source of truth; commit any resulting changes to them when dependencies change.
- Create or restore the environment with `uv sync --locked`.
- Run Python commands and tests through `uv run --locked`, for example:
  - `uv run --locked python -m unittest discover -v`
  - `uv run --locked python <script>`
- Do not depend on shell activation or call `.venv/bin/python` directly. Activating `.venv` is optional convenience only.
- Do not install project dependencies with system `pip`, user-site `pip`, or `sudo pip`.
- Treat `.venv/` as disposable generated state. Recreate it with uv rather than repairing or committing it.
- Add and remove dependencies with `uv add` and `uv remove`, then run the locked test suite.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->

## Agent skills

### Issue tracker

Issues, specifications, and durable project work are tracked in the repository's Beads database using `bd`. See `docs/agents/issue-tracker.md`.

### Domain docs

This repository uses a single-context domain-documentation layout. See `docs/agents/domain.md`.

### Delivery workflow

3-step planning in this order:

1. `grill-with-docs`
2. `to-spec`
3. `to-tickets`
4. Implement each claimed vertical slice with `tdd`; and keep the slice open until its pull request is merged. See `docs/agents/development-workflow.md`.

- #### Versioning note

    The files `faceledger-v1.md` and `faceledger-v1-model-scope-amendment.md` (in that chronological order) under `docs/specs/` document the original **core functionality** only. Treat those specifications as historical design references only.

    Treat `faceledger-v1-cli.md` as the primary specification.
    Also, `docs/reference/core-api.md` provides a good overview of the core files.

## General Notes

### Sandbox problems

Consider sandbox issues when a command fails.

If gh reports an invalid token or cannot reach api.github.com, retry the command with escalated network access before concluding authentication is broken; sandbox restrictions can produce misleading auth errors.

### Docker images & Large assets

Before downloading release-qualification assets again, check the preserved Docker images faceledger-qualification:{ubuntu-26.04,debian-13,fedora-44,arch-20260726} and their corresponding base images.

---
## Implementation Phase

Implementation happens per-ticket in Beads, called a cycle.

### Python Validation

Run `./scripts/check.sh` after modifying Python code and before declaring the task complete. Run it again immediately before committing or pushing when requested.

Treat this script as the authoritative project-wide validation command for Ruff linting, Ruff formatting, mypy, and unit tests. Fix underlying failures; do not bypass the script, weaken its configuration, or add broad suppressions merely to make it pass.

Targeted Ruff, mypy, or test commands may be used during development for faster feedback, but they do not replace the final full check.

Ruff’s safe automatic fixes may be used during development with `uv run ruff check --fix <targets>`. Do not use `--unsafe-fixes`. Review all resulting changes, then run `./scripts/check.sh` for final non-mutating validation.

- #### Validation Script Maintenance

  Keep `./scripts/check.sh` aligned with the project’s maintained Python code and required validation steps. When adding, moving, or removing Python packages, tests, scripts, or quality tools, update the script as necessary so it remains the single authoritative command used locally, by Codex, and by CI.

  Do not exclude relevant code or remove checks merely to make validation pass. When changing the script’s scope or behaviour, update CI to invoke the same script rather than duplicating its commands.

### Docstrings

When creating or modifying a non-trivial production function, add a docstring immediately below its declaration.

* Describe what the function achieves in one concise sentence.
* Explain architectural intent only when it is not apparent from the implementation.
* Do not restate the function name, implementation steps, parameter types, or return type.
* Document parameters, return values, side effects, or edge cases only when they are non-obvious or significant.
* Do not add docstrings to test functions, fixtures, callbacks, trivial accessors, or other functions whose purpose is already clear from their name and context.
* Use the language’s conventional docstring format.
* Keep the complete docstring to four lines or fewer.

### Cycle start

- Check remote to see if PR has succeeded for previous tickets, and update Beads accordingly.
- Check and fast-forward or rebase `main` branch as needed.
- At the start of each ticket, create and check out a new ticket-specific Git branch. Work for different tickets must not share a branch. You may name the branch without a prefix i.e. `ticket/fl-repo-60a.17-trash-cache` should just be `fl-repo-60a.17-trash-cache`. You may name the branch after the corresponding bead ticket name.

### Cycle end

- Make a commit with a ticket name and a very short description.
- Push the branch to remote with `git push -u origin HEAD`.
- Create a PR with a brief summary. You may reuse the commit title as the title.
- Report errors and standby, if necessary. Otherwise move on.
- Report status by stating the next tickets which are open or blocked. Distinguish the tickets which previously became open and has remained open, from the ones which has become unlocked more recently.
- Standby.

### DeepFace Boundary & Known Issues

- `qualify_runtime.py` deliberately tests raw DeepFace behavior, while application code uses `DeepFaceRecognition`.

- Generally, DeepFace is an untyped external dependency. Keep all direct `deepface` imports and DeepFace-specific logic isolated behind the adapter boundary, preserve the targeted `# type: ignore[import-untyped]`, and return normalized typed application data. If maintaining this boundary in one module would mix distinct responsibilities or create excessive complexity, propose the additional adapter module and explain the intended separation before implementing it.

### Core files

Core functionality files are described in `docs/reference/core-api.md`. Update this reference when the implementation of these core files is changed.

### Live Scoping Doc during Implementation

Treat the following scoping document as an overlay on every original spec, ADR, and ticket.

For version 1:

      None so far.

- Key points:

      None so far