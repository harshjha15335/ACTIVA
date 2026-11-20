# Landing the role branches on `main`

`main` starts from the M4 foundation (cc3d61d). The role branches were created
separately and share no history with it. Instead of merging unrelated
histories, each member's own files are copied onto a branch cut from `main`
and opened as a PR. The original role branches are left untouched.

## Landing PRs (opened 2026-11-09)

| Member | PR | Source | Tests on the branch (Windows) |
|--------|----|--------|-------------------------------|
| M1 | [#6](https://github.com/harshjha15335/ACTIVA/pull/6) `m1-landing` | `m1-actions` fccad75, files copied | 110 passed |
| M2 | [#5](https://github.com/harshjha15335/ACTIVA/pull/5) `m2-landing` | `m2-perception` 944732f, M2's own `docs/integration/m2-to-main.patch.gz` | 169 passed, 2 skipped (need real CLIP weights) |
| M3 | [#7](https://github.com/harshjha15335/ACTIVA/pull/7) `m3-landing` | `m3-selector` cc6f36a, files copied | 142 passed |

`./run.sh` exits 0 with valid JSON on every landing branch. On `m2-landing`,
with no CLIP weights in `weights/`, the clip, importance and category stages
log the error and fall back.

Each PR merges after CI passes and its owner confirms the files are theirs and
complete. If two PRs touch `requirements-dev.txt`, keep every `-r` line.

## After your PR lands

Work on a branch cut from the new `main`:

```sh
git fetch origin
git switch -c m1-actions-v2 origin/main     # or merge origin/main into your landing branch
python -m pytest -q                         # before every push
gh pr create --base main
```

If you pushed more commits to your old role branch after the landing commit,
copy those files over again (`git checkout origin/<your-branch> -- <paths>`)
and open a follow-up PR.

## After each PR lands

M4 runs the full suite plus `./run.sh` on the fixture videos and posts a
RESOLVED note on the board ([#4](https://github.com/harshjha15335/ACTIVA/issues/4)).
