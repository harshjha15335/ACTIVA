# Student kit checklist (guide A3)

Answered from the student kit received 2026-11-11 (see `docs/kit_compliance.md`).

| # | Question | Answer | Status |
|---|----------|--------|--------|
| 1 | Exact verb / noun / category strings; valid pairs? | `supported_classes.json`: 16 `labels`, 98 `verbs`, 301 `nouns`; no pair list | ANSWERED |
| 2 | Which dataset feeds which metric | EPIC: coverage 40% + mAP 25%; FPVSum: highlight F1 25% + label 10% | ANSWERED |
| 3 | Overlap rule for action coverage | "share of distinct GT action pairs captured within the selected highlights"; exact overlap rule not given | OPEN (no scorer in kit) |
| 4 | Budget tolerance | over 15% is a validator warning plus a "soft length penalty"; each segment needs >= 4 s to count | ANSWERED; short-video conflict OPEN |
| 5 | May actions carry `score`? | required, in [0, 1] | ANSWERED |
| 6 | Hardware, time limit, network, weight size | not stated; organisers re-run `./run.sh` "in an isolated environment" | OPEN (asked) |
| 7 | Full EPIC videos or trimmed clips? | full EPIC-100 validation videos, 11 s to 33 min | ANSWERED |
| 8 | JSON rules | exactly 6 fields, timestamps up to 2 decimals, file `<video_id>.json`; extra keys not addressed | ANSWERED (extra keys: we emit none) |
| 9 | Can kit files be committed? | not stated | OPEN (asked) |
| 10 | Official `validate.py` | listed in the README but missing from the ZIP; we use `scripts/validate_outputs.py` meanwhile | OPEN (asked) |
