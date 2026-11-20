# Action-density Study Plan

**Goal:** Execute the fixed request in #28 using existing matched private inputs.
**Spec:** #28; variants U/I/A/J0.6/J0.8, densities 2/3.5/6/all, six tune then four
held-out, keep J0.6 unless one alternative wins every density on both splits.
**Architecture:** Extend only the M3 evaluation runner. Production selectors and
defaults do not change. Confidence thinning is score-only, deterministic, and
uses floor(density * duration / 60) among predictions >=0.2. “All” preserves all
confident detections. This is a density sensitivity study, not a new detector.

The earlier full-density held-out outcomes are already published. The new sparse
conditions get one frozen held-out evaluation; this is reuse of the existing
partition and must not be described as a fresh independent holdout.

- [x] Add failing tests for thinning, ties, bad inputs and mode/signal separation.
- [x] Add thin_actions and run_density_cases plus --densities to run_ablation.py.
- [x] Run focused tests. Execute tuning densities, save fixed rule/choice lock.
- [x] Execute new held-out density conditions once; report all conditions.
- [x] Review #29 in an isolated source copy; run tune-only no-harm checks if its
      actual checkpoint/text inputs are available. Do not activate the mask.
- [x] Report evidence/limitations, full tests, push PR and update #28/#25.

Future D-a is report-only after frozen final outputs are committed/hashed; do not
score manifest annotations now. Human card review and source-disjoint FPVSum
gates belong to M2/M4 and are not replaced by a no-harm coverage result.
