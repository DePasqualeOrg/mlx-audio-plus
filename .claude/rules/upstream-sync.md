# Upstream Sync

This project is a fork of [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) (remote: `upstream`).

**Last sync point:** `19f5aba` (Add Echo TTS, 2026-02-27) – synced on 2026-02-28.

## Planning

1. `git fetch upstream` to get the latest commits.
2. Compare `git log --oneline <last-sync-hash>..upstream/main` to identify new commits.
3. Categorize each commit as cherry-pick or skip:
   - **Cherry-pick:** Models, bug fixes, server changes, infrastructure, dependency updates.
   - **Skip:** README-only, examples-only, UI changes (`mlx_audio/ui/`), version bumps, empty merge commits.

## Branch workflow

1. Create a branch: `git checkout -b sync/upstream-YYYY-MM-DD`
2. Cherry-pick commits oldest-first: `git cherry-pick <hash>`
3. After all cherry-picks, fast-forward merge: `git checkout main && git merge --ff-only sync/upstream-YYYY-MM-DD`

## Conflict resolution

- **`pyproject.toml`:** Keep fork identity (package name `mlx-audio-plus`, version, authors, URLs). Keep self-referencing optional dependency groups (e.g., `mlx-audio-plus[stt,tts]`) rather than flattening. Take new dependencies from upstream.
- **`mlx_audio/version.py`:** Always keep the fork's version.
- **`uv.lock`:** Delete with `git rm --force uv.lock` during cherry-picks (it's gitignored in the fork). Regenerate at the end with `uv lock`.
- **`README.md`:** Take upstream's version with `git checkout --theirs README.md`.
- **UI files (`mlx_audio/ui/`):** Drop with `git rm --force` (UI was removed from the fork).
- **Code files (`utils.py`, model files, etc.):** Take upstream changes, preserve any fork-specific code.

## Cherry-pick edge cases

- **Empty cherry-picks** (e.g., reverts of commits we never had): Skip with `git cherry-pick --skip`.
- **Authorship:** When a commit is primarily someone else's work, set them as author with `git commit --amend --author="Name <email>"`.
- **Avoid `git cherry-pick --abort`** after resolving earlier commits in a session – it can revert all prior cherry-picks. Instead, resolve conflicts manually and continue.

## Verification

1. `uv lock` – regenerate the lockfile.
2. `uv run python -c "import mlx_audio"` – basic import check.
3. Run the test suite. Fix any broken imports from files that diverged between fork and upstream.

## After syncing

1. Update "Last sync point" above with the hash and date of the most recent cherry-picked commit.
2. Delete the sync branch: `git branch -d sync/upstream-YYYY-MM-DD`
