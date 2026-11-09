# M1 (Action Engine) status

## Done in this increment (G0)

- `code/actions.py`: `detect_actions(info) -> list[Action]` **stub** (returns `[]`) with the
  fixed A5 signature, plus tested building blocks for the real engine:
  feature-grid <-> seconds conversion (copied from ActionFormer), tIoU, EPIC-train verb-noun
  prior, verb+noun pairing, class-wise NMS, and `finalize` (clamp, filter, sort, cap, convert).
- `code/action_config.py`: every M1 setting in one place (fixed ActionFormer grid values and
  tunable pairing values).
- `tests/test_actions.py`: 17 tests. `python -m pytest -q` -> 17 passed (CPU, no data needed).
- `kaggle/m1_step1_3_actionformer_reference.ipynb`: Steps 1–3 on a Kaggle GPU (install,
  reproduce published mAP, export raw detections in seconds, one-video sanity check).

`contracts.py` is M4-owned and not committed here; tests fall back to a test-only copy of the
A5 dataclasses when it is missing (`tests/conftest.py`).

## Facts confirmed from ActionFormer (commit 61ea7eb)

| Item | Value |
|---|---|
| Features | SlowFast trained on EPIC-100, 32 frames @ 30 fps, stride 16, `input_dim: 2304` |
| Time of grid index i | `(16 i + 16) / 30` s (`segs * stride + 0.5 * nframes) / fps`) |
| Published validation mAP (avg tIoU 0.1–0.5) | verb 23.51, noun 21.88 |
| Default test output | up to 2000 segments/video, `min_score 0.001`, Soft-NMS sigma 0.4 |
| GPU memory | README: inference may exceed 10 GB, 12 GB recommended (we use Kaggle T4/P100, 16 GB) |
| Feature extractor checkpoint | **not stated** in the README -> first task of the Step 4 gate |

## Not done / open

- Steps 1–3 have not been run yet (needs Kaggle). No mAP has been measured by us.
- Step 4 gate (raw MP4 -> matching features) not started.
- No tuning data plan yet: the pretrained models were trained on all of EPIC train, so
  thresholds tuned on train are biased and tuning on validation is leakage. Option to decide:
  retrain ActionFormer heads on train minus a few held-out participants (training needs
  ~4.5 GB GPU per the README).

## Reply draft to M2's request (not posted)

```text
[FROM: M1] [TO: M2] [TYPE: ACK] [TASK: M2-TIMELINE-RUNTIME]
Timestamp origin: seconds from the first decoded frame, same as your PTS-minus-start
convention. SlowFast windows: video resampled to 30 fps, window i centre = (16i+16)/30 s.
Action.score: internal confidence in [0,1], sorted descending; not yet known whether the
JSON may carry it (A3 #5).
Vocabulary: EPIC-100 97 verbs / 300 nouns until the kit's supported_classes.json arrives.
Runtime, dependencies and a real dev action cache: after the Step 4 gate.
CLIP fallback: will request only if the gate fails.
Artifact: branch m1-actions, docs/m1_status.md.
Status: OPEN
```
