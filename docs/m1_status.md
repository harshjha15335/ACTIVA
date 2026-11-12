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

## Steps 1–3 result (measured 2026-11-10, Kaggle T4, torch 2.11.0+cu130, numpy 2.1.3)

Run: notebook `kaggle/m1_step1_3_actionformer_reference.ipynb`, Kaggle version 2,
"Successfully ran in 502.4s". Published features + pretrained checkpoints, 138 EPIC-100
validation videos.

| Model | tIoU 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | **Avg** | README avg |
|---|---|---|---|---|---|---|---|
| Verb | 26.58 | 25.42 | 24.15 | 22.28 | 19.01 | **23.49** | 23.51 |
| Noun | 25.21 | 24.11 | 22.66 | 20.47 | 16.97 | **21.88** | 21.88 |

Reference reproduced (verb within 0.02, noun exact). Compatibility patches needed on new
torch/numpy: `np.float/int/bool` aliases in `libs/utils/metrics.py`, and `eval.py` checkpoint
loading (`map_location=cfg['devices'][0], weights_only=False`).

Raw detections: ~2000 per video (verb 275,928 / noun 276,000 over 138 videos), saved as
Kaggle output `m1_raw_detections_val.pkl`. Validation data: code checks only, no tuning.

Findings that matter for Step 4:
- Published `.npz` holds `feats [T, 2304]` **plus `v_prob [T, 97]` and `n_prob [T, 300]`**:
  per-window verb/noun probabilities from the same SlowFast model. So the extractor is the
  EPIC-100 SlowFast action-recognition model with its classification heads, and our own
  extractor can be checked on both features and these probabilities.
- Window count: P01_11 (561.6 s) has **T = 1053**; `(n_frames - 32) // 16 + 1` gives 1052.
  The published count looks like `ceil(n_frames / 16)` (the last window padded). Confirm on
  more videos before relying on `expected_num_windows`.
- Annotation JSON `fps` is null (config `default_fps: 30` is used) and labels are numeric
  class ids; verb/noun names must come from the EPIC-100 class CSVs.

## Real inference engine (branch m1-real-action-inference, 2026-11-11)

Board request M1-REAL-ACTION-INFERENCE (#4). Kit: `Student package_track04 (1).zip`,
sha256 ddf7ee8d...0c09 (matches the board), kept in git-ignored `kit/`.

Extractor provenance (ActionFormer authors): EPIC-KITCHENS fine-tuned SlowFast from
epic-kitchens-slowfast (actionformer issue #116); 2304-D after 3D global average pooling
(#70); whole 16:9 frame, no centre crop (#91). From epic-kitchens-slowfast @ bf50519:
frames are EPIC rgb_frames read with `cv2.imread` (**BGR**, never converted), /255,
-0.45, /0.225; 32 frames at 60 fps sampling rate 2 (= 30 fps); slow path = 8 of 32.

| Piece | File | State |
|---|---|---|
| Raw MP4 -> features (+ v_prob/n_prob) | `code/slowfast_features.py` | built, unit-tested; parity with published features **pending gate** |
| ActionFormer verb/noun inference | `code/actions.py` `ActionFormerRunner` | EMA weights load strict (CPU, torch 2.15) |
| NMS without C++ build | `code/nms_1d_py.py` | unit-tested; parity vs compiled **pending gate** |
| Pairing, NMS, cap 2000, official vocab | `code/actions.py` | all 97 verbs / 300 nouns are in kit vocabulary (kit adds `other`) |
| `detect_actions(info)` | `code/actions.py` | real chain; raises `ActionEngineUnavailable` if assets missing |
| Local assets, pinned + hashed | `scripts/setup_m1_assets.py` -> `weights/m1/ASSETS.json` | done (needs M4 OK: new file in scripts/) |

Asset hashes (sha256): SlowFast.pyth `0fc9e83d...0cf7`, verb `88486434...687b`,
noun `8d313a2d...bbf9`, verb classes csv `aab8bf21...033e`, noun classes csv `40f0a81c...e893`.

Runtime: SlowFast on laptop CPU (torch 2.15) **7.1 s per window** -> a 10-min video
(~1125 windows) would take >2 h on CPU. A GPU is required; GPU numbers pending.

Tests: M1 suites 50 passed. Full suite in project venv: 5 failed / 3 errors, identical
on plain `main` (M2 tests: no ffmpeg on this laptop + kit label conflict noted by M4).

### Step 4 gate result (measured 2026-11-11, Kaggle T4 x2, torch 2.11.0+cu130)

Kaggle notebook `notebookdbcfaaeed9` v1 ("Successfully ran in 1420.9s"), from
`kaggle/m1_step4_feature_gate.ipynb` at ab0c43f. Raw MP4s from the official EPIC downloader:
P02_13, P03_21, P26_30, P28_23 (all 1920x1080, 59.94 fps). Compared with the published
ActionFormer features of the same videos. No annotations used.

| Variant | feat cos mean | feat cos min | verb top-1 agree | noun top-1 agree | v_prob MAE |
|---|---|---|---|---|---|
| **bgr_full** (456x256, BGR, no crop) | **0.937** | **0.860** | **0.835** | **0.790** | 0.0027 |
| bgr_crop256 | 0.930 | 0.821 | 0.793 | 0.717 | 0.0043 |
| bgr_crop224 | 0.912 | 0.803 | 0.768 | 0.669 | 0.0047 |
| rgb_full | 0.826 | 0.579 | 0.791 | 0.618 | 0.0036 |

Best temporal shift 0 on every video (time base aligned).

ActionFormer on our bgr_full features vs on published features, top-100 detections with a
same-class match at tIoU >= 0.5 (ours-in-published / published-in-ours):

| Video | verb | noun |
|---|---|---|
| P02_13 | 0.80 / 0.81 | 0.86 / 0.85 |
| P03_21 | 0.86 / 0.85 | 0.74 / 0.73 |
| P26_30 | 0.78 / 0.83 | 0.74 / 0.74 |
| P28_23 | 0.75 / 0.78 | 0.68 / 0.70 |

NMS parity (P02_13 verb, published features): compiled `nms_1d_cpu` vs `nms_1d_py`:
2000 / 2000 detections, top-100 match 1.00, max score difference 4.5e-8. **Identical.**

**Verdict: PARTIAL.** Raw MP4 -> real ActionFormer detections works and is time-aligned,
but features reach cos 0.937 (guide PASS rule: >= 0.95) and ~70-86% of top detections are
reproduced. Remaining gap is in the frames themselves (the model's own verb/noun top-1
differs on ~16-21% of windows): EPIC rgb_frames are ffmpeg-scaled JPEGs read by cv2,
we use PyAV + PIL bicubic without JPEG. Next: emulate that path and re-measure.

Other findings:
- Published window count is **floor(n_frames / 16)** (55/59/28/47 here, 1053 for P01_11);
  our pad rule emits ceil -> one extra trailing window. Fix pending.
- Decode is the bottleneck: ~33 s CPU to decode+resize a 30 s 1080p60 clip with PyAV+PIL.

### Step 4 gate v2: frame pipeline variants (measured 2026-11-11, Kaggle T4 x2)

Kaggle notebook `notebookc3973e8913` v1 ("Successfully ran in 1278.7s") from
`kaggle/m1_step4_feature_gate_v2.ipynb` at 07bd0b8. Same 4 videos, BGR, whole frame,
`floor` window rule. On Kaggle (ffmpeg 6.1.1) the M1 tests ran with the ffmpeg path: 69 passed.

| Frame pipeline | feat cos mean | cos min | verb top-1 | noun top-1 | decode s/video | model s/video |
|---|---|---|---|---|---|---|
| ff_jpg4_even (ffmpeg 60 fps, JPEG q4, even frames) | **0.9405** | 0.869 | 0.844 | 0.819 | 13.1 | 11.5 |
| ff_jpg2_even | 0.9402 | 0.870 | 0.821 | 0.805 | 13.2 | 11.9 |
| ff_jpg4_odd | 0.9389 | 0.863 | 0.854 | 0.785 | 13.1 | 11.7 |
| ff_raw_even (**production default**) | 0.9383 | 0.867 | 0.826 | 0.820 | 13.1 | 11.1 |
| ff30_raw (ffmpeg fps=30) | 0.9374 | 0.861 | 0.830 | 0.799 | 11.3 | 12.1 |
| pyav_pil (v1) | 0.9372 | 0.860 | 0.835 | 0.790 | 23.5 | 12.0 |
| ff_raw_odd | 0.9365 | 0.857 | 0.811 | 0.804 | 13.0 | 11.3 |

Window counts now match the published files exactly (55/55, 59/59, 28/28, 47/47).
Downstream, best variant, top-100 ours-in-published: verb 0.81 / 0.89 / 0.77 / 0.76,
noun 0.86 / 0.76 / 0.68 / 0.69 (P02_13 / P03_21 / P26_30 / P28_23).

Reading: every frame pipeline lands at cos 0.937-0.941, so decoding, scaling, JPEG and
frame parity are **not** the cause of the gap. Production uses `ff_raw_even` (1.8x faster
decode than PyAV; the JPEG round trip gains 0.002, within noise, and needs GBs of temp files on
20-min videos). Open hypothesis: a sub-window frame offset in the original extraction
(not detectable by the whole-window shift test), or a different checkpoint than
`SlowFast.pyth`. Verdict unchanged: **PARTIAL**.

### Step 4 gate v3: sub-window frame offset (measured 2026-11-11, Kaggle T4 x2)

Kaggle notebook `notebook919773d50a` v1 ("Successfully ran in 1173.0s") from
`kaggle/m1_step4_feature_gate_v3.ipynb` at 88c184e. Production ffmpeg frames, shifted by k
frames at 30 fps before windowing (k < 0: windows start earlier). Means over the 4 videos:

| k | -12 | -8 | -6 | -4 | **-2** | **-1** | 0 | +1 | +2 | +4 | +6 | +8 | +12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| feat cos | 0.878 | 0.908 | 0.923 | 0.934 | **0.945** | 0.943 | 0.938 | 0.935 | 0.930 | 0.917 | 0.905 | 0.892 | 0.863 |
| verb top-1 | 0.725 | 0.739 | 0.794 | 0.823 | 0.850 | **0.860** | 0.826 | 0.798 | 0.810 | 0.790 | 0.699 | 0.680 | 0.606 |
| noun top-1 | 0.688 | 0.712 | 0.773 | 0.804 | 0.814 | **0.853** | 0.820 | 0.767 | 0.780 | 0.770 | 0.735 | 0.668 | 0.617 |

A smooth peak at k = -1..-2 (~50 ms): the published windows start slightly earlier than
ours. Production now uses `FrameSource.shift = -1` (best verb/noun agreement; cos 0.943,
within 0.002 of the peak); the window count still follows the real frame count.

**Verdict after v1-v3: PARTIAL, with the cause narrowed down.** Decode backend, scaling,
JPEG, frame parity and window count are matched or ruled out; the time offset is fixed.
The remaining ~0.055 cosine gap (peak 0.945 < 0.95) and ~15% top-1 disagreement are most
likely the checkpoint or the authors' private extraction setup, which cannot be reproduced
from public files. The engine is real, offline and time-aligned; it is an approximation of
ActionFormer's published input, and that should be stated in the README and deck.

### Integration with current `main` (2026-11-12)

Merged `origin/main` (f01392f: kit 6-field JSON, official vocab, M2 finalization) into
`m1-real-action-inference`. With the official kit installed: the engine loads against
`vocab.load()` (97/300 EPIC classes all present), emits the 2000-action cap and
`main.sanitize_actions` keeps all of them. Full suite: 280 passed, 50 skipped; the only
failures (3 failed + 3 errors) are M2 tests that need `ffmpeg` on PATH, absent on this laptop.

Runtime for the 138 EPIC manifest videos (13.2 h of video, 89,058 windows, longest
P24_09 32.8 min), extrapolated from measured speeds:

| Hardware | Estimate |
|---|---|
| Kaggle T4 (measured: model 0.21 s/window, ffmpeg decode ~0.5x real time) | ~6 h model + ~6.8 h decode, ~12.8 h serial; decode now runs in a background thread, so less |
| Laptop CPU (measured 7.1 s/window) | ~176 h: not feasible |

So the EPIC action stage needs a GPU (ActionFormer README: >10 GB for its reference
setting; Kaggle T4 16 GB works). FPVSum videos get the same engine (actions may be [] per kit).

Dependencies beyond `requirements.txt` (M4-owned, request on the board): `fvcore`,
`iopath`, `simplejson`, `pyyaml` (SlowFast / ActionFormer code), plus `gdown` only for the
one-time `scripts/setup_m1_assets.py`. FFmpeg must be on PATH (already true for video_io).
M1 does not need CLIP features.

## Not done / open

- Close the feature gap (JPEG/ffmpeg frame path), fix window count (floor), speed up decode.
- Time-alignment check on real video, GPU runtime for 2/5/10-min videos, real dev action cache.
- Opt-in real-checkpoint integration test; M4 strict end-to-end run.
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
