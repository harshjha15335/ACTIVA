# ACTIVA - Member 3: Selector & Evaluation

First increment: local development scorers and the G0 uniform selector stub.
The combined optimizer, model pipeline and real-data ablations are not yet built.

## Run locally (PowerShell)

```powershell
cd C:\Users\adita\Desktop\ACTIVA
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-m3.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Python 3.10+ is required. If NumPy, pandas and pytest are already installed,
`python -m pytest -q` works directly. These tests need no GPU, videos, weights,
network connection, student kit or dataset downloads.

## What is available

- `eval/metrics.py`: budget validation, union duration, distinct action coverage,
  action/verb/noun mAP, highlight precision/recall/F1, label accuracy, combined report.
- `code/optimizer.py`: deterministic, evenly spaced selector stub with the specified
  `select_highlights(info, actions, imp, clip)` signature. It needs M4's
  `code/contracts.py` to return the canonical `Highlight` type. Unit tests inject
  a test-only type; the repository does not redefine shared contracts.
- `tests/`: known-answer metrics, interval properties, invalid inputs, deterministic
  stub checks and random budget checks.
- `docs/m3_status.md`: acceptance evidence, open dependencies and next steps.
- `docs/m3_handoff_draft.md`: local request draft for the integration owner.

## Call the scorers

```python
from eval.metrics import action_map, budget_ok, coverage, highlight_f1

# Synthetic example only, not a measured video result.
gt = [{"verb": "cut", "noun": "tomato", "start": 2.0, "end": 6.0}]
pred = [dict(gt[0], score=0.8)]
highlights = [{"start": 2.0, "end": 6.0, "importance": 0.5}]

assert budget_ok(highlights, duration=60)
cov = coverage(gt, highlights, rule="frac>=0.5")
maps = action_map(gt, pred)
pair_map = maps["pair"]["mAP"]
```

Actions and highlights can also be dataclass instances. Prediction `score` is
required for mAP; the scorer never invents a confidence. For dataset-level mAP,
pass `video_id -> list[Action]` mappings for both GT and predictions. AP pools
ranked detections across videos, rather than averaging per-video AP values.

All intervals use seconds and half-open `[start, end)` boundaries. Coverage
requires an explicit rule (`any`, `centre`, `frac>=X`); there is no assumed official
rule. Fractional coverage uses the highlight union, and repeated action instances
count once per verb-noun pair. Empty GT returns zero. ANet mAP averages over GT
classes; classes appearing only in predictions do not enter that average.

`highlight_f1(human_importance, times, highlights, top_ratio=0.15)` uses the supplied
annotation sample grid. The highest `ceil(top_ratio * N)` samples form the human
target; equal importance prefers earlier samples. Importance must be normalized
to `[0, 1]`. This is sample-based and does not time-weight uneven grids.

`score_all` is a single-video, keyword-only report. Supply `duration`, `gt_actions`,
`pred_actions`, `highlights` and `coverage_rule`. Optional paired inputs are
`human_importance`/`times` and `gt_labels`/`pred_labels`. Its weighted total is
unavailable when any metric is missing or the highlight budget is invalid.
The weights are a development proxy, not verified official score aggregation.

## Integration

M4 should put `code/` on the module search path (as when running `python code/main.py`)
and provide the guide's canonical `contracts.py`. The stub then returns canonical
`Highlight` instances, sorted, non-overlapping and within a strict 15% budget.
It returns no clips when the budget cannot accommodate a 1-second clip.

The official student kit, validator, vocabulary and metric rules are still needed
before claiming schema compatibility or completion of the official G1 gate.
No real benchmark scores have been measured. No model pipeline or `run.sh` is
included in this M3 increment. M4 owns that integration work.

## Metric source

The mAP core is adapted from [ActionFormer](https://github.com/happyharrycn/actionformer_release/blob/61ea7eb9308a568b0cf45e3804830836e30061de/libs/utils/metrics.py).
Its MIT license and exact adaptation notes are in `eval/_vendor/`.
