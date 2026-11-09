# M2 source transfer onto canonical main

## Why this route

Canonical main is `cc3d61d23c57be326eeca80baf19a8f61141fd67`. The original role
branches have unrelated roots. M4's published plan proposes an unrelated-history
merge, but the user's explicit instruction forbids it. No such merge, history
rewrite, canonical push or self-merge was performed.

A source-only patch is prepared against the exact canonical base. It replaces
M2's own placeholders and adds its modules/tests/tools, preserves M4's protected
source byte for byte, retains the canonical README and decisions, and proposes
the required dependency union for review. It is not an already accepted migration.

## Reproduce the review tree and patch

From a clean/current M2 checkout with Git, Python dependencies and FFmpeg:

```sh
git fetch origin refs/heads/main:refs/remotes/origin/main refs/heads/m1-actions:refs/remotes/origin/m1-actions refs/heads/m3-selector:refs/remotes/origin/m3-selector
python scripts/prepare_integration_review.py --base origin/main --m1-ref origin/m1-actions --m3-ref origin/m3-selector --output /tmp/activa-m2-review
```

Choose a new output directory: existing user work is never overwritten. The tool
creates plain source copies, not Git worktrees or history merges:

- `canonical/`: immutable main snapshot.
- `review/`: proposed M2-owned changes and dependency additions only.
- `combined/`: disposable M1/M2/M3/M4 rehearsal, not proposed owner-source edits.
- `m2-to-main.patch`: M2-only source patch, checked with `git apply --check`.
- `review-manifest.json`: exact source refs and protected-file checks.

Protected canonical files include contracts.py, config.py, vocab.py, video_io.py,
main.py, schema.py, run.sh and AGENTS.md. M1/M3 implementations are copied only into
the rehearsal; they are not included as changes in M2's patch. Model weights,
media and development feature payloads are never included in the source patch.

## Reproduce real smoke assets and tests

```sh
python scripts/prepare_weights.py --preset laion400m_e32
python scripts/setup_validation_assets.py --output /tmp/activa-m2-smoke
```

The model candidate is explicit; see weights/README_weights.md for source and
verification limits. Smoke assets are small public examples with pinned observed
content hashes, not evaluation/training data. They are fetched only by this
explicit setup script. Source licenses require review before redistribution.

Run the combined suite with real assets explicitly selected:

```sh
cd /tmp/activa-m2-review/combined
ACTIVA_REAL_CLIP_WEIGHTS=/ABSOLUTE/PATH/ACTIVA/weights/ViT-B-32-laion400m_e32.safetensors ACTIVA_REAL_MEDIA=/tmp/activa-m2-smoke python -m pytest -q
```

Then use the actual integration driver, passing the canonical source copy:

```sh
python /ABSOLUTE/PATH/ACTIVA/scripts/verify_team_interfaces.py --review-root /tmp/activa-m2-review/combined --video /tmp/activa-m2-smoke/walking-smoke.mp4 --weights /ABSOLUTE/PATH/ACTIVA/weights/ViT-B-32-laion400m_e32.safetensors --output /tmp/activa-m2-review/team-interfaces.json
```

The driver uses M4's explicitly marked TEST vocabulary unless an explicit file is
supplied. It requires real M2 stages without fallback, checks M1 conversion and
Action compatibility, and exports controlled M3 importance inputs. This is not
an official validator or proof of a real M1 action model/B0/B2 selector.

## Acceptance checklist for M4

1. Review the exact base SHA; regenerate if main changes. Do not blindly apply a
   patch against an unverified newer foundation.
2. Approve the explicit model/preprocess/cache identity and asset distribution.
3. Review dependency additions, M2 placeholder replacements and tests. Shared
   contracts/config/vocab/decoder must stay unchanged unless separately approved.
4. On an approved new review branch based on canonical main, apply the checked
   source patch or transfer the owned files; preserve existing role commits.
5. Run all relevant tests and genuine-asset smoke checks. Obtain owner review.
6. Open a related-history PR only after the target/foundation and authorization
   are agreed. Link Issues #1/#2/#3; never self-merge or close them prematurely.
7. Supply actual official vocabulary/manifest/validator before claiming challenge
   compatibility or starting data-dependent model selection.

The local rehearsal passed 237 tests, including 78 M2 tests. The real runner used
no perception fallback. Official validation and recipient acceptance remain open.
