# ACTIVA M2 — Issue #14 Handoff and Release Coordination

Status: 14 November 2026. This document records source-backed facts and the remaining M2 acceptance gates. Reply on [issue #14](https://github.com/harshjha15335/ACTIVA/issues/14), not on the old board issue #4.

## Official deadlines and submission

Samsung PRISM's 12–13 November final-submission emails confirm:

| Milestone | Official date |
| --- | --- |
| Final submission | **22 November 2026** |
| Top-12 announcement | 29 November 2026 |
| Final demo | 9 December 2026 |
| Results | 12 December 2026 |

Official form: https://forms.gle/86xENrtqhx988hBt8.
The organizer requests the solution, code, resources, documentation, demo, presentation and README through a **public GitHub repository** linked in that form. The email does not settle redistribution rights for third-party datasets or the organizer-provided kit; check permissions before making those public.

**Internal code freeze is not an organizer-set deadline.** M2 proposes 20 November, subject to M4/team confirmation. The registered team name is LensLog; a separate unique team ID has not been confirmed.

An initial clarification was sent on 11 November to prism@samsung.com. A concise follow-up was sent **14 November** in the same Gmail thread to m.shankar@samsung.com and prakhar.c@samsung.com. It requests only (1) missing official validate.py, (2) handling the 4-second/15% conflict on short videos, and (3) the exact team_id. A sent email is **not** an organizer answer.

## CLIP tensor-content SHA-256 check

The M4 reference tensor-content SHA-256 recorded on the board is:

    4423d64febb7bfb00ddc8c77fd2c38b9047d5183c0e45f14e56847295ff8fedf

The original LAION-400M e32 source archive SHA-256 is:

    46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa

Different converted safetensors **file** hashes can occur even when tensor content is identical. On the actual local machine with the checkpoint installed, run:

    .\.venv\Scripts\python.exe scripts\check_m2_tensor_hash.py --report tmp\m2\tensor-hash.json

Report the tensor-content digest and match flag to M4, not the checkpoint bytes. CI's synthetic tests **cannot** establish that the real checkpoint's tensors match.

## Genuine cache transfer for Anuj (M3)

The genuine 79.5-second cache was reported under the ignored local path:

    dev_cache/manifests/vtest_full.ab378ab08665.074b2b49db4a.json

This manifest and its NPZ arrays are **not** on public GitHub. On the machine holding them, run:

    .\.venv\Scripts\python.exe scripts\package_m2_cache.py --manifest dev_cache\manifests\vtest_full.ab378ab08665.074b2b49db4a.json --output tmp\m2\vtest_full_handoff.zip

The package includes manifests/, clip/, importance/ and SHA256SUMS.json. It checks all original hashes and performs a recipient-side reader roundtrip, normalizes Windows file separators, and redacts the absolute source media path. It does **not** include the original video, the model checkpoint, organizer files, or annotations.

Share the ZIP privately via a team-approved channel. Provide Anuj the printed archive SHA-256 and the extracted manifest location. His consumer acceptance command is:

    python scripts/check_m3_handoff.py --optimizer code/optimizer.py --manifest <extracted-dir>/manifests/vtest_full.ab378ab08665.074b2b49db4a.json --output tmp/m2/m3-handoff.json --require-importance-sensitive

The existence of packaging code does not prove the actual private archive was created or delivered.

## Category misses: evidence and next experiment

M4 reports 39/49 FPVSum name-label matches on the manifest video set. These are **diagnostic outcomes on evaluation-set predictions**, not permissible tuning or official scores. Do not tune against the 49 manifest labels/IDs.

Hypotheses to test only with legitimate **non-manifest**, reviewed-safe data:
- Current prompts include constructions such as “a person is bike” or “a person is bikepolo”, which may poorly describe the activity.
- Compare existing mean, importance-weighted and top-30% pooling with the same checkpoint, prompts and preprocessing.
- Require human-reviewed exact category aliases; do not invent bike/bikepolo or ski/skiing mappings.
- Verify softmax score reliability separately from class accuracy.

An existing guarded study runner already evaluates all four M2 importance baselines and three pooling variants:

    python scripts/evaluate_perception.py --study <study.json> --inventory <inventory.csv> --annotations <annotations.mat> --review <review.json> --official-manifest kit/manifest.csv --weights weights/ViT-B-32-laion400m_e32.safetensors --output tmp/m2/approved-study.json

It requires a human-reviewed split bound to the official manifest; its sample-grid importance F1 is **not** the organizer's final highlight F1. No production-category prompt or pooling default should change until held-out results support it.

## Remaining dependencies

- M2 local machine: run the real tensor hash check and create the genuine private cache archive.
- M3: receive and independently verify cache, and run U/I/A/J on approved matched data.
- M1: agree the 10 off-manifest EPIC train videos and provide action predictions with provenance.
- M4: confirm internal freeze date, safe-label development data, organizer responses, and permission to share private artifacts.
