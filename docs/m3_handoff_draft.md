# M3 dependency request draft

This draft has not been posted or sent. The repository currently has no known
message board issue number, official kit or shared contracts.

```text
[FROM: M3] [TO: M4] [TYPE: REQUEST] [TASK: startup]
Context: Local development scorers and a G0 uniform selector stub are ready.
Needed: official student kit/evaluator, code/contracts.py, supported_classes.json,
coverage overlap rule, F1 frame/ranking/tie rules, allowed confidence fields,
schema constraints and exact budget rule. Please identify the message board issue.
Artifact: branch m3-selector; eval/metrics.py; code/optimizer.py;
requirements-m3.txt; tests/; docs/m3_status.md.
Interface: select_highlights(info, actions, imp, clip) -> list[Highlight];
canonical Highlight is imported from M4's contracts.py at invocation.
Tests: python -m pytest -q -> 49 passed (local toy/property tests only).
Acceptance: shared contract imported and official evaluator parity checked on
agreed examples. This draft does not claim official G1 validation.
Status: OPEN
```

Later M1/M2 requests should specify the actual agreed development split and cache
format. No cache format or safe evaluation split is assumed in this increment.
