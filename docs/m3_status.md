# M3 startup status

## Scope

Started from the supplied Member 3 evaluation guide. Both the local directory and
the referenced GitHub repository were empty when inspected. Work is on
`m3-selector`. This increment builds Step 1 and the G0 selector stub; it does not
claim a complete application or the later optimization/ablation milestones.

No teammate agents were launched and no messages were sent to teammates.
The request draft is a local artifact for a future handoff.

## Validation evidence

Command: `python -m pytest -q`

Observed result on 2026-11-08: **49 passed**. This is a software test count,
not benchmark performance. Tested with the existing Windows Python environment:
NumPy 2.3.5, pandas 2.3.3, pytest 8.4.2. No GPU or datasets were used.

Checks include known-answer scores, AP ranking and duplicate locks, cross-video
matching, per-class and per-tIoU scores, stable score ties, finite values, explicit
coverage rules, half-open boundaries, correct union duration, and 500 seeded
random video durations for the uniform stub's budget and determinism properties.
Interval unions are independently checked against an integer-grid set oracle.

The canonical M4 contracts are absent. Typed stub tests use an injected test-only
Highlight, so shared-contract compatibility still needs an integration check.
The official evaluator has not been provided or compared with these local proxies.

## Decisions and assumptions

- Use the ActionFormer ANet-style AP implementation with retained MIT notice.
  Modernize deprecated NumPy scalar alias and stabilize equal-confidence ordering.
- Ground-truth class macro averaging follows that implementation; keep pair,
  verb and noun reports separate. Dataset AP ranks predictions across videos.
- Coverage overlap rule must be supplied. No official coverage rule is inferred.
- Local F1 ranks annotation samples, rounds target count up, and resolves ties by
  time. These conventions need official verification, especially on tied curves.
- Budget validation requires sorted, disjoint intervals and a strict 15% maximum,
  with no assumed organizer tolerance. The stub leaves numerical headroom.
- Missing metrics produce no combined proxy total. No partial metric reweighting.
- Shared contracts, config, vocabulary, runner and model files remain M4/M1/M2
  dependencies. M3 dependencies are isolated in requirements-m3.txt.
- No validation/test split has been downloaded or used for tuning. No video IDs,
  annotation answers or development cache lookups enter the selector stub.

## Next inputs and work

1. M4: official student kit, evaluator/validator, canonical contracts, vocabulary,
   coverage overlap rule, score ranking/schema rules, F1 target construction and
   budget tolerance. Compare local scores with official toy cases once available.
2. Implement U/B0/B1 baselines with one interface and compare on designated safe
   development data. Keep oracle-action experiments explicitly labeled.
3. Implement candidate generation, then the coverage-plus-importance optimizer.
4. M1: predicted actions with confidence and precise timestamp semantics.
   M2: importance and CLIP features on the agreed time grid, plus safe split IDs.
5. Run held-out ablations on actual predictions, record commands/hardware/inputs,
   and choose settings only after that evidence exists.

No settings have been frozen and no benchmark winner is claimed.
