# Student kit checklist (guide A3)

Nothing below is answered yet. The code keeps each item as a setting in
`code/config.py` or `code/vocab.py`; when the kit arrives, fill in the answer,
change the setting, and record it in `docs/decisions.md`.

| # | Question | Where it lives now | Status |
|---|----------|--------------------|--------|
| 1 | Exact verb / noun / category strings; are valid verb-noun pairs listed? | `vocab._parse()` (assumed layout), `kit/supported_classes.json` | OPEN |
| 2 | Which dataset feeds which metric (EPIC: coverage + mAP? FPVSum: F1 + label?) | M3 scorers | OPEN |
| 3 | Overlap rule for action coverage | M3 `coverage(rule=...)` | OPEN |
| 4 | Budget tolerance (is 15.5% penalised? soft penalty?) | `config.BUDGET_RATIO`, `BUDGET_SAFETY` | OPEN |
| 5 | May actions carry `score`? How is mAP ranked without it? | `config.INCLUDE_ACTION_SCORE` (off) | OPEN |
| 6 | Hardware, time limit, network access, weight-size limits | `docs/decisions.md` | OPEN |
| 7 | Full EPIC videos or trimmed 2-10 min clips? | `config.EXPECTED_DURATION_RANGE` (warn only) | OPEN |
| 8 | JSON rules: rounding, extra keys, ordering, `video_id` format | `config.OUTPUT_DECIMALS`, `ALLOW_EXTRA_KEYS`, `schema.to_output` | OPEN |
| 9 | Can kit files be committed to this public repo? | `.gitignore` ignores `kit/` | OPEN |
| 10 | Official `validate.py`: how to call it; add to CI and `scripts/validate_all.sh` | not yet wired | OPEN |
