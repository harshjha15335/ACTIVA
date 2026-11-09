# Landing the role branches on `main`

`main` starts from the M4 foundation. The role branches were created
separately and share no history with it, so each owner merges `main` into
their branch once, resolves a few predictable conflicts, and opens a PR.
After this first merge, normal `git merge origin/main` works.

The conflict lists below come from trial merges of each branch as of
2026-11-09 (m1 `fccad75`, m2 `e70b55f`, m3 `cc6f36a`).

## Steps (every owner)

```sh
git fetch origin
git switch m1-actions                      # your branch
git merge origin/main --allow-unrelated-histories
# resolve the files listed for your branch (below), then:
git add -A
python -m pytest -q                        # all tests, yours and M4's
git commit                                 # the merge commit
git push
gh pr create --base main --title "Land m1-actions on main"
```

During this merge, `--ours` is **your branch** and `--theirs` is **main**.

- Shared files (M4-owned): `git checkout --theirs <file>`
- Your module files (replacing M4's placeholder stubs): `git checkout --ours <file>`

## M1 (`m1-actions`)

| File | Take |
|------|------|
| `.gitignore` | `--theirs` (main's is the union of all branches) |
| `code/actions.py` | `--ours` |

Trial result: merges cleanly with these two choices; M1's 17 tests and M4's
tests pass together. `tests/conftest.py` now finds the real `contracts.py`,
so its test-only fallback is no longer used.

## M3 (`m3-selector`)

| File | Take |
|------|------|
| `.gitignore`, `README.md` | `--theirs`; first save yours: `git show HEAD:README.md > docs/M3_README.md` |
| `code/optimizer.py` | `--ours` |

Also add `-r requirements-m3.txt` to `requirements-dev.txt` (the scorers import
pandas; they are dev-only, not part of inference).
Trial result: M3's 49 tests and M4's tests pass together; the pipeline runs
M3's selector stub with no fallback.

## M2 (`m2-perception`)

| File | Take |
|------|------|
| `.gitignore`, `README.md` | `--theirs`; first save yours: `git show HEAD:README.md > docs/M2_README.md` |
| `code/contracts.py`, `code/config.py`, `code/video_io.py` | `--theirs` (your behaviour is kept; see `docs/decisions.md` D2-D4) |
| `code/clip_features.py`, `code/importance.py`, `code/category.py` | `--ours` |
| `requirements.txt` | main's file plus your pins (Pillow, torch, torchvision, open_clip_torch) |
| `requirements-dev.txt` | main's file plus `-r requirements-m2.txt` |
| `docs/decisions.md` | keep both: M4 section, then yours |

CI installs CPU-only torch automatically once `requirements.txt` pins it.

Trial result (all four branches merged, Windows, CPU torch 2.15.0, no CLIP
weights): **230 passed, 2 skipped, 1 failed** after the patch below.

- **Needed M2 change, `code/category_vocab.py`:** the adapter treats "a `vocab`
  module exists" as "the official vocabulary is installed". On `main`,
  `vocab.load()` raises `vocab.VocabError` until the kit file is present, so 3
  M2 tests fail. This patch fixes all three (tested in the trial merge):

  ```diff
       if module is not None:
  -        loaded=module.load()
  +        try:
  +            loaded=module.load()
  +        except module.VocabError:
  +            if require_canonical:raise
  +            module=None  # kit vocabulary not installed: use explicit dev labels
  +    if module is not None:
           if not isinstance(loaded,(tuple,list)) or len(loaded)!=4:
  ```

- The remaining failure, `test_two_minute_black_motion_cli_and_invalid_input`,
  runs `perceive.sh` directly, which cannot start on Windows (M2's README notes
  Windows is untested). Expected to pass on Linux CI; not yet confirmed.
- Without CLIP weights, `./run.sh` still exits 0 with valid JSON: clip,
  importance and category log their errors and use fallbacks.

## After each PR lands

M4 runs the full suite plus `./run.sh` on the fixture videos, posts a
DECISION/RESOLVED note on the board, and the next owner merges the new `main`.
