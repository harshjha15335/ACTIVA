# M3 status

## Review handoff - 2026-11-13

M4 reviewed PR #9 and requested it be marked ready; no code changes were
requested. The original PR CI re-run is verified successful. Current main
f01392f (including M2's official-kit test isolation fixes and category confidence
update) is merged into this feature branch without changing M3's algorithms.
Fresh full-suite evidence with the actual kit installed is in docs/results.md.
Only the feature branch is pushed; M4 retains the human merge decision.

Fresh local verification: **325 passed, 3 skipped, 0 failed** with the authentic
kit installed. The previous three M2 isolation failures are resolved by the
already-merged M2 changes. Skips are the configured-team-ID case and the two
opt-in genuine-model tests whose local weights/media remain absent.

The November 11 handoff was posted to the message board:
https://github.com/harshjha15335/ACTIVA/issues/4#issuecomment-6801054556
The older draft text below is historical, not the current coordination state.

M2 reports successful genuine-cache acceptance on the 79.5-second vtest_full
fixture in docs/M2_FINAL_REPORT.md and board comment 6802382981. That is M2's
reported evidence, not a local rerun by M3. The manifest and its two NPZ payloads
are still absent from this checkout. Request their transfer with original hashes
and model identity; never create substitute features or pad cached timestamps.

M1 offers real predictions and GT for ten non-manifest EPIC-100 train videos.
M3 accepts that offer and asks M1 to select/post the IDs, source hashes, participant
groups, durations, timestamp conventions and checkpoint/preprocessing identity.
The source list and leakage review must be checked before experiments. Separate
development/tuning inputs from held-out comparisons and report actual predicted
inputs separately from oracle inputs. No list, review or results are fabricated.

PR readiness covers the tested implementation, not completion of held-out quality
evaluation or organizer scorer parity. The official validator/scorer, short-video
clarification and reviewed labeled artifacts remain dependencies for those gates.

## Implementation record - 2026-11-11

Initial files landed through PR #7; no source files were missing. New work starts
from shared main f16179c on m3-action-aware-selector. The board assigns replacing
the uniform stub, whose 3-second clips M4 now drops. Production now uses joint
predicted-action coverage and importance. U/I/A/J, candidates, sensitivity tests
and an ablation adapter are implemented; see M3_SELECTOR.md for conventions.

The official ZIP is installed locally and matches the shared SHA-256. It lacks
validate.py; organizer files are not committed. M4's existing D13 short-video
exception is logged and reported as a strict-budget violation pending organizers.

Controlled action/importance location changes and the existing M2 handoff checker
pass on an explicitly SYNTHETIC v1 cache. This is interface evidence. Genuine M2
caches, raw-MP4 M1 prediction artifacts, reviewed labeled development data and
an official scorer are unavailable here. No real-data quality winner is claimed.
Commands/evidence are in docs/results.md. Local handoff drafts have not been sent.

Final checks: 99 M3 tests pass with the kit; full CI-configuration suite has
317 passed/6 skipped. Three M2 tests fail with the kit installed and reproduce
on unchanged main, as already flagged by M4. No M2 files were changed to hide
those failures. The official kit was restored after the CI-configuration run.

## Historical startup record - 2026-11-08

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
