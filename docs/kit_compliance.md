# Student kit compliance (Track 04)

Source: `Student package_track04 (1).zip`, SHA256
`ddf7ee8d2b8b9fb1363c1e72ae95da9144f64573cbca5a1b7a2378f41bbd0c09` (same file M2 has).
Contents: `TEAM_README.md`, `manifest.csv`, `supported_classes.json`, `P02_109.json`,
`skiing_03.json`. Kept local in `kit/` (git-ignored); not committed.

## What the kit says

- **Videos:** 187 in `manifest.csv` (`video_id,duration_sec,dataset`).
  - 138 `epic_kitchens_100`, 11.1-1968.6 s, 13.2 h in total. 46 of them are under 2 min and 17 are over 10 min.
  - 49 `fpvsum`, 37-377 s, 2.4 h in total.
  - Raw videos are not supplied. EPIC comes from the official downloader (~184 GB), FPVSum from YouTube via yt-dlp.
- **Vocabulary:** `supported_classes.json` (version `prism2026-theme01-v1`).
  - Keys: `labels` (16), `verbs` (98), `nouns` (301), `notes`.
  - Labels are the 14 FPVSum categories (lowercase, e.g. `ski`) plus `cooking` and `other`.
  - Verbs and nouns are EPIC-100 class keys, plus `other`. No valid-pair list is given.
- **Output:** one `<video_id>.json` per manifest video, with exactly 6 top-level fields:
  - `team_id`: identical across all files.
  - `video_id`: must match the file name and the manifest.
  - `duration_sec`: within ±0.5 s of the manifest.
  - `label {name, score}`
  - `actions [{segment, verb, noun, score}]`: 0-2000 items, `0 <= start < end <= duration_sec + 0.5`.
  - `highlights [{segment}]`
  - Timestamps "up to 2 decimal places".
- **Highlights:**
  - At least 1, sorted, "strictly non-overlapping".
  - At least one segment must be >= 4.0 s. Shorter segments are ignored when scoring.
  - Total "target" <= 15% of `duration_sec`. Going over is a validator warning plus a soft scoring penalty.
- **Validity:** a missing or invalid JSON scores 0 on every metric for that video.
- **Scoring:**

  | Metric | Weight | Dataset | Based on |
  |--------|--------|---------|----------|
  | Action Coverage | 40% | EPIC | `highlights` vs distinct ground-truth action pairs |
  | Action mAP | 25% | EPIC | `actions`, tIoU 0.1-0.5 |
  | Highlight F1 | 25% | FPVSum | frame-level, vs the top 15% of consensus importance |
  | Label accuracy | 10% | FPVSum | `label.name` |

  Only a hidden subset of the manifest is scored.
- **Submission:** a folder named after the team ID, containing:
  - `run.sh`, `code/`, `weights/`, `requirements.txt`
  - `outputs/`: 187 files
  - `README.md`

  Organisers re-run `./run.sh` on sample videos. Hardcoded IDs or copied ground truth mean disqualification.

## Gap table (main d874a3a vs kit) and what this change does

| Area | Before | Kit | Now | Status |
|------|--------|-----|-----|--------|
| Vocabulary keys | expected `categories` | `labels`, `verbs`, `nouns` | `vocab._parse` reads `labels`; the `Vocab` 4-tuple is unchanged (`categories` = labels) | DONE |
| Valid pairs | optional list | none given | every verb x noun accepted | DONE |
| Top-level JSON | 4 keys | 6 keys | `schema.to_output` writes all 6 | DONE |
| `team_id` | none | required, identical | `config.TEAM_ID = "LensLog"` (registered team name); `ACTIVA_TEAM_ID` overrides; missing means exit 5 | PROVISIONAL: no official ID issued yet (Q9) |
| `duration_sec` | none | ±0.5 s of manifest | probed duration, rounded to 2 decimals; checked by the batch runner and validator | DONE (checked on real videos once downloaded) |
| `label` | string | `{name, score}` | score = `CategoryResult.scores[label]` clipped to [0, 1] | DONE; M2 asked to make scores calibrated |
| Action fields | `start`/`end`, no score | `segment` + `score` | `segment` + `score` always | DONE |
| Action count | unlimited | <= 2000 | top 2000 by score kept | DONE |
| Highlight fields | `start`/`end`/`importance` | `segment` only | `segment` only; importance stays internal | DONE |
| Minimum clip | 1 s | >= 4 s counts | clips < 4 s dropped before checks; fallback uses 6 s clips | DONE; M3 still emits 3 s clips |
| Short videos (15% < 4 s) | n/a | rules conflict | one 4.0 s clip (`config.SHORT_VIDEO_SINGLE_CLIP`) | PROVISIONAL: ask organisers |
| Touching clips | allowed | "strictly non-overlapping" | merged into one clip; the validator warns on touching | DONE |
| Fallback label | first category | n/a | `other` | DONE |
| Validator | 4-key spec | `validate.py` (missing from ZIP) | `scripts/validate_outputs.py`, the "ACTIVA local spec validator" | PARTIAL: official script needed |
| Release gate | fallbacks always on | n/a | `ACTIVA_STRICT=1`: a failing stage means exit 6, no JSON | DONE |
| Batch run | none | 187 outputs | `scripts/run_manifest.py`: resumable, per-video report | DONE (not yet run on real videos) |
| Durations | warned outside 2-10 min | 11-1969 s | warn outside 10-2000 s | DONE |
| Kit in Git | n/a | unknown | `kit/` ignored | OPEN (#9) |

## Questions for the organisers (Aarush to send)

1. **`validate.py`:** the README lists it, but our ZIP doesn't contain it, and `examples/` is a flat folder. Can you share the official validator?
2. **Short videos:** for P03_26 (11.13 s), P26_30 (15.28 s) and P28_23 (25.16 s), one segment of 4.0 s or more is longer than 15% of the video. Is one 4.0 s segment over budget accepted, with only the soft penalty?
3. **Touching segments:** is `[a, b]` followed by `[b, c]` "strictly non-overlapping"?
4. **Example label:** `skiing_03.json` uses the label `skiing`, but `supported_classes.json` has `ski`. We assume `ski` is correct.
5. **Sample IDs:** `P02_109` and `skiing_03` are not in the manifest. We assume the examples are format samples only.
6. **Vocabulary version:** `supported_classes.json` says `prism2026-theme01-v1`. Is that the right version for Track 04?
7. **Publishing:** may the kit files (manifest, classes) be committed to a public team repository?
8. **Reproduction:** what hardware and time limits apply when you re-run `./run.sh`? Is network access available?
9. **Team ID:** we haven't received one. Should `team_id` and the submission folder use our team name, `LensLog`?

## Not done here (other owners or later M4 work)

- M1: real detections (`detect_actions` still returns `[]`).
- M2: CLIP weights installed offline; calibrated label scores.
- M3: clips of 4 s or more, and selection that uses actions and importance.
- M4:
  - run the batch on the downloaded videos;
  - an opt-in real-model end-to-end test;
  - `TEAM_ID/` packaging, Docker, the demo.
