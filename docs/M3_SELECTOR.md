# M3 action-aware selector

This increment replaces the uniform production stub with U/I/A/J selectors.
It is a development implementation; no held-out quality winner is claimed.
Board task: [M3-ACTION-AWARE-HIGHLIGHT-OPTIMIZER](https://github.com/harshjha15335/ACTIVA/issues/4#issuecomment-6794751174).

## Interfaces

`select_highlights(info, actions, imp, clip) -> list[contracts.Highlight]` is
unchanged. Production calls J (joint). `select_uniform`, `select_importance`
and `select_coverage` take the same four arguments. Experiments can call
`select_variant(..., mode="U"|"I"|"A"|"J", settings=SelectorSettings(...))`.
Settings are development defaults, not tuned/frozen benchmark settings.

Actions use canonical labels, seconds and confidence [0,1]. M4 owns vocabulary
checks; invalid intervals/confidences fail visibly here. Importance is finite
[0,1] on strictly increasing actual PTS seconds, not necessarily an even grid.
When CLIP features are present their time grid must match exactly. Empty grids
are supported. No embedding dimension or checkpoint is hardcoded.

M2 v1 cache is accepted for DEVELOPMENT use through perception_cache.read_cache,
including payload hashes, model identity, aligned grids and normalized float32
features. The documented 512-D ViT-B-32 identity can be checked with that reader's
expected_model argument. This does not approve the checkpoint for submission.
No genuine M2 cache is installed here. Production never reads caches.

13 November update: M2 reports a genuine 79.5-second vtest_full cache passing
this selector's handoff check (docs/M2_FINAL_REPORT.md). Its manifest/NPZ files
are not present in M3's checkout yet, so that result remains attributed to M2.
The reader uses each artifact's actual model_name and payload hashes; it does
not require one machine's older converted-checkpoint file hash.

15 November update: M3 received the replacement vtest_handoff.zip, verified its
archive/payload checksums, loaded its actual 159x512 cache, and passed the existing
handoff checker and canonical rounding/minimum checks. This is now local genuine
cache-consumer evidence; see docs/results.md. Real-action quality comparison is
still pending the ten agreed EPIC train sources and matched M1/M2 artifacts.

17 November update: the real matched-input U/I/A/J study is complete on the
fixed six-tuning/four-held-out selector split. J reaches 37.41% held-out local
coverage versus U's 24.89%; A reaches 40.78%. Retain J because the EPIC-only
coverage winner does not establish the full coverage/F1 trade-off and A's
no-action behavior is uniform. See the dated results/evidence for limitations,
separate oracle results and exact provenance. No production defaults changed.

The objective uses M2 importance directly, including novelty if M2 selects that
baseline. It does not directly score CLIP embeddings. Distinct action pairs
provide diversity. Facility-location embedding diversity is deferred until
safe held-out evidence justifies it.

## Objective and search

1. Generate tight/padded action clips, 4/6/12/18-second sliding windows,
   importance peaks and bounded boundary-snapped variants. Expand short action
   candidates to 4 seconds; clip to video boundaries and quantize to M4's output
   grid (currently 0.01 seconds). Remove exact duplicates.
2. Retain at most 500 candidates. Preserve best candidates per distinct pair,
   high-confidence coverage, high-importance density and temporal spread.
3. For each detected instance, r is the fraction intersecting the selected UNION.
   Credit is zero below tau=0.5, otherwise confidence*r. Each distinct pair gets
   the MAX instance credit. C is the average across pairs. Identical detections
   are deduplicated at maximum confidence; confidence <0.2 gets no credit. This
   is the optimizer's predicted-coverage proxy, not an official overlap rule.
4. Integrate piecewise-linear importance analytically on actual timestamps;
   endpoint values extend to video boundaries, empty curves mean zero.
   Normalize I by optimistic best-budget mass over at most 20,000 time bins.
5. Joint F=0.6*C+0.4*I. Try marginal gain/cost^alpha for alpha 0, 0.5, 1; retain
   the largest true F. Only nonoverlapping candidates are feasible. Ties use
   true gain, shorter duration, earlier start/end. Merge touching selected clips
   and assign integrated mean importance to each returned highlight.

Coverage-only removes importance from candidate generation and the objective.
Importance-only removes actions from both. With no confident actions J becomes
importance-only. With no signal, even spacing is an explicit deterministic
policy. A-mode with no actions uses uniform. Positive-gain search can leave
budget unused; it does not add clips solely to fill time.

This is bounded greedy interval packing, with no global optimum or submodularity
guarantee. For P raw candidates, A actions, C<=500 retained candidates and K
selected clips, scoring is O(3*K*C*A), overlap memory O(C*A). Pool pruning is
O(P*A) plus sorting. Local search is not enabled.

## Budget and ambiguity

Output timestamps already sit on M4's grid. Inward rounding is idempotent and
does not shorten a minimum clip. Normal target: 14.8%, hard maximum: 15%.
When 15% can fit 4 seconds but 14.8% cannot, use that minimum within the hard cap.

For duration>=4 seconds whose 15% budget is under 4 seconds, preserve M4's
existing D13/config.SHORT_VIDEO_SINGLE_CLIP: exactly one 4-second clip with an
explicit warning. This IS a strict budget violation, never a budget pass.
It remains provisional pending organizers. Disabling that config raises.
A positive duration under 4 seconds raises because no valid minimum fits.
Zero duration returns [] for helper compatibility; M4 rejects invalid inputs
before invoking the selector. No video IDs appear in this policy.

## Reproduce

Use the project environment. On this Windows checkout replace `python` with
`.\.venv\Scripts\python.exe` (the global Anaconda environment lacks dependencies).

```powershell
python -m pytest -q
python scripts/run_ablation.py --synthetic --coverage-rule 'frac>=0.5' --output tmp/m3/synthetic-ablation.json
```

Synthetic controls use synthetic actions/features/labels. They test behavior,
not video quality. --oracle produces a separately labeled oracle-input report;
detector AP still uses predictions. Reports include source hashes, settings,
command, hardware, runtimes, sample counts, budget violations, clip counts,
cut-action fractions and per-video timeline highlights.

For genuine cache acceptance, use the existing M2 checker:

```powershell
python scripts/check_m3_handoff.py --optimizer code/optimizer.py --manifest <real-M2-manifest.json> --output tmp/m3/real-handoff.json --require-importance-sensitive
```

M2's original 10-second smoke cache cannot satisfy 4 seconds AND 15%. That
checker correctly fails strict budget on it. Use a genuine >=26.67-second input
for normal-budget acceptance and test short policy separately. Never extend
cached timestamps/features to pretend a short video is longer.

## Real ablation adapter

This is an explicitly defined M3 DEVELOPMENT adapter, not an organizer format.
--study reads an activa-m3-study-v1 JSON with nonempty videos entries:

- dataset: epic_kitchens_100 or fpvsum;
- m2_manifest: local v1 cache path;
- predictions: JSON with source="predicted", input_sha256, checkpoint_sha256
  and actions with explicit score plus start/end or segment;
- annotations: JSON with matching input_sha256; EPIC uses actions, FPVSum uses
  aligned times/normalized importance, and optionally a GT label;
- optional pred_label for category accuracy.

Also supply --review using the existing activa-leakage-review-v1 format, bound
to --official-manifest by SHA-256, with reviewed-safe decisions, exact hashes
and human content-review evidence. The loader rejects evaluation IDs,
unreviewed inputs and duplicate IDs/hashes. The human review is responsible
for source identity, aliases, grouping and split isolation; it is not generated
by this agent. Organizer data, annotations and generated real outputs stay local.

```powershell
python scripts/run_ablation.py --study <study.json> --review <review.json> --official-manifest kit/manifest.csv --coverage-rule 'frac>=0.5' --output tmp/m3/real-ablation.json
```

Coverage is averaged on EPIC; F1/label on FPVSum. Detector AP pools scored
predictions across EPIC. Missing metrics or strict budget violations make the
combined proxy total unavailable. No reviewed study was supplied, so real
quality evaluation was initially blocked. The verified 17 November real study
and fixed density sensitivity report now appear in docs/results.md; current
J0.6/0.4 is retained under #28's fixed decision rule.

For EPIC density sensitivity, add `--densities 2,3.5,6,all`. This evaluates only
U/I/A/J0.6/J0.8. Filtering is score >=0.2, then floor(density*duration/60) top
rows with start/end/verb/noun tie breaks. It changes evaluation inputs only.
Synthetic density controls use the EPIC fixture and remain non-benchmark tests.

To review an explicit M2 card-mask candidate without replacing inference code:

```powershell
python scripts/m3_card_mask_check.py --candidate-source <extracted-importance.py> --candidate-commit <full-sha> --tune-study <six-tune.json> --other-study <four-other.json> --review <review.json> --official-manifest kit/manifest.csv --baseline-report <density-tune.json> --output tmp/m3/card-mask-check.json
```

Use the exact checkpoint matching the cached image embeddings. All ten source
hashes must be distinct. The other four sources get mask-rate checks only;
their coverage is not scored. The no-harm gate requires J0.6 to change by <=0.5
percentage points at every density relative to both the original cache baseline
and recomputed candidate OFF. This isolates mask effects from numerical or
implementation differences. FPVSum human frame review is still mandatory for GO.

## Metric fidelity

The kit confirms weights/datasets and tIoU 0.1-0.5, but provides neither the
promised validate.py nor a scorer. Coverage overlap, F1 tie/rank construction
and soft budget penalty remain unverified. Local F1 uses the annotation sample
grid, top ceil(.15*N), earliest ties and half-open intervals. On irregular grids
it is sample-weighted, not original-frame F1. Scorers now accept kit segment
fields. score_all remains a single-video diagnostic; dataset aggregation lives
in run_ablation.py. M1 predictions, genuine M2 artifacts, reviewed labeled data
and organizer parity remain explicit dependencies.

## Whiteboard walkthrough of select_variant (#28)

1. Validate the time grid and action intervals, filter confidence <0.2 and
   deduplicate identical instances by maximum confidence. Round feasible budget
   down to centisecond ticks, aiming for 14.8% of duration. The provisional
   short-video exception remains separately documented.
2. Generate at most 500 candidates from action boundaries, importance peaks and
   sliding windows. A removes importance from candidate generation; I removes
   actions. Candidate generation is part of the ablation, not a shared oracle.
3. For selected intervals S, each action instance receives its confidence times
   covered fraction, but only once the union fraction reaches tau=0.5. For each
   distinct (verb,noun), take the best instance credit, then average across
   distinct predicted pairs. Repeated detections do not add class credit.
4. Importance is the integral of the interpolated curve on S, divided by an
   optimistic budget-mass estimate: split time into at most 20,000 bins, rank
   bin-average importance and fill the duration budget, allowing a fractional
   final bin. This comparison scale ignores interval packing and minimum clip
   length. It is an approximation, not an exact feasible optimum.
5. J uses 0.6*C + 0.4*I. At each step, gain is the increase in that objective
   after adding one feasible candidate. Rank gain/length^alpha for each of
   alpha=0,0.5,1 in three independent greedy runs. Alpha changes the preference
   for short clips; it is not a fourth objective weight. Reject overlaps and
   clips exceeding the remaining tick budget after each choice.
6. Choose the run with the largest actual objective, not its length-adjusted
   ranking value. Ties are deterministic. Merge touching clips and return their
   mean importance internally; exported highlights contain only segment.

This is a bounded heuristic: the thresholded coverage term has jumps, and no
global-optimum or submodularity guarantee is claimed. J with no confident
actions becomes importance-only; A with none becomes uniform. The density study
changes the available predicted actions, not these definitions or the GT metric.
