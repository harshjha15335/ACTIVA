# Current M2 results — 9 November 2026

These are fresh checks, not the historical74-test report. Detailed evidence:
`docs/evidence/M2_team_validation_2026-11-09.json`. Earlier implementation history
remains in commit e70b55f; its measurements are not relabeled as new runs.

## Source and tests

M2 started clean at e70b55f. Tested canonical main `cc3d61d23c57be326eeca80baf19a8f61141fd67`, M1
`fccad758b3e71cf1c816326eeae1ab965895c0db`, M3 `cc6f36a3c4d567235b5466666d3d6e0848ed5acf` in disposable source copies. No owner files or
shared configuration/contracts/decoder were modified. No Git histories were merged.

```sh
ACTIVA_REAL_CLIP_WEIGHTS=/workspace/ACTIVA/weights/ViT-B-32-laion400m_e32.safetensors ACTIVA_REAL_MEDIA=/workspace/validation-assets/activa-m2 /workspace/activa-venv/bin/python -m pytest -q -p no:cacheprovider
```

- Baseline before edits:74 passed,0 failed,0 skipped.
- Current standalone M2:78 passed,0 failed,0 skipped,4 warnings,19.38 seconds.
- Combined canonical/M1/M2/M3 copy:237 passed,0 failed,0 skipped,4 warnings,48.46 seconds.
- The first combined attempt found one newly written toy-test fixture error (blank
  config values);236 passed/1 failed. The fixture was corrected, not the validation
  rule. Logs retain that failure; final results above passed.
- M2's missing-kit vocabulary adapter now handles M4's actual module correctly.
  Strict mode, malformed installed vocabulary and unrelated implementation errors
  still fail. The functional CLI test now invokes Python directly for Windows
  development portability; actual Windows execution remains unverified.

## Real pretrained checkpoint

Full raw and converted hashes were recalculated and match their local manifest.
Identity is unchanged:
`ViT-B-32/sha256:9bd6bea2e0b7f343b17347af1854d7fd2667a227b0997801e5d297d46808854b`.
This is the explicit OpenCLIP LAION-400M e32 QuickGELU candidate, not original
OpenAI weights. Published prefix verification and full locally recorded hashes
retain the limits described in weights/README_weights.md. No new model was selected.
Fresh real-image and MP4 tests passed with512-D normalized float32 features and
blocked socket connections. No GPU is available (CPU-only Torch 2.15.0+cpu).
Code license reference is MIT; final model distribution/competition approval is
still M4/team work, not inferred from passing tests.

## Actual canonical execution

M4's actual main.run() and run.sh, with real M2 and actual M1/M3 source copies,
processed a 10-second public smoke MP4. No stage fell back. stdout was empty for
run.sh; output contained only video_id,label,actions,highlights and passed M4's
local schema validation. Vocabulary was the explicitly marked M4 TEST fixture,
not official Samsung classes; no official validator was available.

M1's public detect_actions remains an empty stub. Its real grid conversion and
finalize helpers produced an Action accepted by M4 (controlled confidence 0.8).
This is helper/API evidence, not raw-video action-model alignment or accuracy.
M3 accepted genuine 20 x 512 features and empty actions; early/late/uniform importance
still produced identical locations. Its uniform mode is not an importance-only
optimizer. Controlled inputs and reproduction script are available for its owner.

## Fresh actual pretrained runtime

CPU,5 visible threads,batch16; public surveillance footage looped to768 x 576 / 10 fps.
These are existing valid runtime fixtures, not first-person quality evaluation.
Fresh processes used M4's shared decoder and explicit development labels. Socket
connections were blocked. Results are observed shared-machine runtime, not a
throughput guarantee.

| Duration | Vectors | Load s | Extract s | Importance s | Category s | Total incl. imports s | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 120 s | 240 | 3.331 | 14.328 | 0.0044 | 0.538 | 22.310 | 1,567,632 |
| 300 s | 600 | 2.076 | 34.446 | 0.0099 | 0.483 | 39.980 | 1,567,580 |
| 600 s | 1200 | 1.922 | 73.794 | 0.0235 | 0.788 | 79.648 | 1,567,372 |

## Data, quality and coordination

Fresh inspection again found98 FPVSum videos:56 labeled/42 unlabeled. Strict parsing
reports the unresolved Jumping ID and BikePolo category spelling discrepancies.
Fractional raw values are preserved. No official manifest, approved safe split,
held-out F1/category accuracy, pooling winner or trained TCN exists. The complete
evaluation workflow now has a passing explicitly labeled toy orchestration test
and records dataset/input hashes,scorer version,precision/recall/F1,runtime,prompts
and reproduction command. Toy values are not reported as model quality.

Issues#1/#2/#3 exist and are OPEN. Their inspected pages had zero comments and no
linked PRs. GitHub API reads returned Forbidden; api.github.com is absent from the
enforced allowlist. Its domain addition was saved in the environment draft but is
not applied. Progress comments remain prepared until posting succeeds.

## M3 action-aware selector - 11 November 2026

Base: main f16179c. Branch: m3-action-aware-selector. U/I/A/J are implemented;
production J uses predicted distinct-action coverage plus integrated importance.
No M1/M2 or shared contracts/config/schema/runner source was modified.
Design, complexity, limitations and the real-data adapter: docs/M3_SELECTOR.md.

### Test evidence

Windows, Intel Core i7-10750H CPU, Python 3.13.9, NumPy 2.3.5, pandas 2.3.3,
pytest 9.2.0, PyAV 19.0.2, Torch 2.15.0+cpu. Project .venv uses local PyPI
numerics to avoid the inherited Anaconda MKL/Torch OpenMP conflict. No unsafe
OpenMP override or shared dependency-file edit was used.

Commands (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_budget.py tests/test_intervals.py tests/test_metrics.py
.\.venv\Scripts\python.exe -m pytest -q
python scripts/run_ablation.py --synthetic --coverage-rule 'frac>=0.5' --output docs/evidence/M3_synthetic_ablation_2026-11-11.json
```

- M3 suite with the real kit installed: **99 passed** in 4.50 s.
- Final full suite in CI conditions (default kit vocabulary temporarily absent,
  restored afterward): **317 passed, 6 skipped**, 92.08 s. Skips: 3 local-kit
  checks, 2 opt-in genuine pretrained checks, 1 unset-team-ID test because the
  team ID is already configured.
- An installed-kit full run had **316 passed, 3 failed, 3 skipped**, 102.15 s.
  The final CI run additionally includes the new study-leakage guard test.
  All three failures are the existing M2 test-isolation issues M4 already listed:
  test_category_uses_exact_configured_labels, test_missing_categories_and_weights_fail,
  test_real_openclip_offline_with_tiny_random_checkpoint in test_perception.py.
  A detached worktree of UNCHANGED main f16179c with the same kit reproduced
  **3 failed** on exactly those three tests (9.01 s). No M2 tests were altered.

Synthetic acceptance proves early/late importance changes clip locations,
moving action detections changes locations, distinct pairs beat repeated pairs,
duplicate copies/action ordering do not inflate coverage, the I/A ablations
ignore the removed signal, canonical inward rounding retains >=4 s clips, and
random scored actions/irregular PTS preserve constraints and determinism.
The original 500-duration helper properties remain; 30 random joint-selector
cases with 20 actions each are also checked.

The existing scripts/check_m3_handoff.py --require-importance-sensitive passed
on an explicitly SYNTHETIC activa-m2-cache-v1 fixture (240 x 512). Early peak:
[0,4]; late peak: [116,120]. Model identity is SYNTHETIC-not-pretrained.
This is interface evidence, not genuine pretrained-model evidence.

### Synthetic ablations and runtime

Evidence: docs/evidence/M3_synthetic_ablation_2026-11-11.json. Exactly two
handwritten controls, one EPIC-shaped and one FPVSum-shaped. They are NOT actual
dataset videos. U/I/A/J have zero strict budget violations on these controls.
The planted distinct-action/importance checks favor J; this cannot establish
real video quality or justify freezing settings. The separately labeled oracle
command also runs (`--synthetic --oracle`); it does not replace detector AP with
oracle AP. No real-data score table or official metric parity is claimed.

One observed stress run on synthetic arrays (no model execution):

| Duration (s) | Scored actions | Selector time (s) | Clips | Selected seconds |
|---|---:|---:|---:|---:|
| 120 | 100 | 0.053 | 3 | 16.00 |
| 600 | 500 | 0.388 | 17 | 87.78 |
| 1968.6 | 2000 | 2.416 | 31 | 288.22 |

Reproduce inputs with NumPy default_rng(1108), sequentially for the three rows:
2-fps np.arange times, normalized one-hot 512-D embeddings, uniform random
importance [0,1], then starts=rng.uniform(0,duration-1,count), and for action i
use verb='v'+str(i%20), noun='n'+str(i%31), end=min(duration,start+rng.uniform(1,30)),
score=rng.random(). Time select_highlights with time.perf_counter. This is a
single runtime observation, not a latency percentile or benchmark accuracy.

### Unresolved evidence

The authentic kit SHA-256 is verified and files remain ignored/local. It has
no validate.py/scorer. M4's D13 short-video policy is retained with warnings:
one 4-second clip when 15% is under 4 seconds, explicitly NOT a strict-budget
pass. Organizer clarification, genuine M1 predictions/M2 caches, approved
leakage-safe labeled data and real held-out comparisons remain pending.
M3's handoff/dependency requests are local drafts, not posted messages.

## M3 review readiness - 13 November 2026

M4's PR #9 review requests no code changes and asks for the draft to be marked
ready. Its original CI run is now verified successful:
https://github.com/harshjha15335/ACTIVA/actions/runs/43055843867
The earlier runner failure was infrastructure; M4's review records the reruns.

Merged approved shared main f01392f into m3-action-aware-selector (merge f2d8394).
This incorporates M2's official-kit test isolation and softmax confidence changes
and M4's default weight path. No new M1/M2/shared-owner source changes were made
by M3; the selector algorithm is unchanged from 8a047d4.

Fresh verification, actual kit installed throughout (not temporarily hidden):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Observed: **325 passed, 3 skipped, 0 failed**, 3 warnings, 106.72 seconds.
Same Windows CPU/Python environment as the November 11 record. Skips are the
unset-team-ID scenario (LensLog is configured) and the two explicit opt-in
pretrained tests (local genuine-model weights/media are absent). No custom test
exclusions or synthetic substitutions were used. The prior three kit-related
M2 failures no longer occur. This supersedes them as current blockers; their
historical results above remain intact.

M2 reports genuine-cache acceptance against this selector on its 79.5-second
fixture in docs/M2_FINAL_REPORT.md and board comment 6802382981. The supplied
path identifies ignored files on M2's machine, not a downloadable artifact in
this checkout. M3 has not independently rerun that genuine cache here. Request
the manifest and both NPZ files with source/model/payload hashes. Keep cache
acceptance distinct from held-out egocentric quality evaluation.

M1 offers ten non-manifest EPIC-100 train videos with real predictions and GT.
Accept M1 selecting/posting the candidate list, then check manifest exclusion,
source hashes and participant groups and agree a separate held-out selector
partition before measuring/tuning. If trimmed, require offsets and rebased
prediction/annotation timestamps. No candidate IDs, approvals or quality results
have been invented. Real U/I/A/J comparisons and official scorer parity remain
follow-up gates, not prerequisites for reviewing the implemented selector.

The earlier handoff was actually posted on November 11:
https://github.com/harshjha15335/ACTIVA/issues/4#issuecomment-6801054556
It supersedes the historical "not posted" statements in that day's draft notes.

## M3 split agreement and genuine cache acceptance - 15 November 2026

Source base: current shared main 08d1cb1, branch m3-real-prediction-study.
No optimizer, metric or shared-owner code/defaults were changed. PR #9 has
landed; this record covers the next steps of assigned issue #16.

### Fixed six-tuning / four-held-out partition

Agreed to M1's exact proposal after independent checks. The approval unblocks
M1's predictions run:
https://github.com/harshjha15335/ACTIVA/issues/15#issuecomment-6893650165

| Split | Video | Participant | Duration (s) | GT segments | Distinct class pairs | Distinct text pairs |
|---|---|---|---:|---:|---:|---:|
| tune | P01_06 | P01 | 492.225 | 119 | 55 | 78 |
| tune | P02_04 | P02 | 422.389 | 168 | 67 | 83 |
| tune | P08_06 | P08 | 486.636 | 122 | 66 | 85 |
| tune | P21_03 | P21 | 585.969 | 75 | 58 | 62 |
| tune | P24_04 | P24 | 493.810 | 122 | 50 | 75 |
| tune | P26_21 | P26 | 465.465 | 89 | 49 | 63 |
| held-out | P03_19 | P03 | 550.350 | 168 | 86 | 92 |
| held-out | P11_15 | P11 | 490.573 | 109 | 64 | 73 |
| held-out | P22_13 | P22 | 500.417 | 246 | 88 | 123 |
| held-out | P30_06 | P30 | 401.351 | 144 | 84 | 90 |

All ten are in the official train CSV, 3-10 minutes, 1920x1080 and approximately
59.94 fps. Total 4,889.1843 seconds. Zero IDs overlap the actual 187-row kit
manifest; zero duplicates and zero tuning/held-out participant overlap.
The local manifest SHA-256 is
`6414e13f5a0c2d7a0d2bc07edf776125cd03a133342a91eb38ad3d729dd150fd`.

Verified against the official epic-kitchens-100-annotations repository pinned
to `ea8b40457a400c3fffa1c7f406ef3dc169cc2522`:

- [EPIC_100_train.csv](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/EPIC_100_train.csv), SHA-256 `a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87`.
- [EPIC_100_video_info.csv](https://github.com/epic-kitchens/epic-kitchens-100-annotations/blob/ea8b40457a400c3fffa1c7f406ef3dc169cc2522/EPIC_100_video_info.csv), SHA-256 `75fd040f6662cb407b4ca2eed4811ca94df280a1aba00b6bf8a5cd286deff382`.

The proposed selection rule reproduces 98 eligible videos across 25 participants
and the exact top ten when diversity uses raw verb/noun text. The class-pair
ranking is different. For coverage and mAP, request canonical GT labels using
the same class-ID-to-vocabulary mapping as M1 predictions. Narration synonyms
must not inflate the scoring denominator. These counts are annotation metadata,
not measured model performance. Raw CSVs and action timestamp rows stay local.
Machine-readable audit: docs/evidence/M3_train_split_audit_2026-11-15.json.

Important limits: ActionFormer trained on these same EPIC train sources; the
held-out partition is for selector decisions relative to the tuning six, not
unseen detector generalization. Diversity-based selection also biases this
small development set. Keep real-prediction and oracle-input experiments
separate, fix choices using the six tuning videos, and avoid repeated tuning
against the four held-out results. No model-quality/default change is made here.

### Received private M2 cache: independently verified

Input: vtest_handoff.zip, provided privately; archive SHA-256
`20f8cf0053a6454e1d059c37aede4f10fa31bd434b2ebea47e00c91bd267417b`.
It exactly matches M2's published checksum. Independently checked every receipt
member hash, then loaded the received manifest with perception_cache.read_cache
and ran the actual selector. The archive and arrays remain local and ignored.

- Public source: 79.5-second OpenCV vtest.avi; source SHA-256
  `45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf`.
- Model identity: `ViT-B-32/sha256:e9997f5060079ffdcb2412062673e039e9292309b46961c4bcbffe64ae1b17a1`.
- Actual features: 159x512 float32, norm range [0.99999994, 1.0].
- Importance and CLIP timestamps match exactly. The model checkpoint/video are
  not included; this validates consumption of M2's genuine supplied cache, not
  independent checkpoint inference on this machine.

```powershell
.\.venv\Scripts\python.exe scripts/check_m3_handoff.py --optimizer code/optimizer.py --manifest <extracted>/manifests/vtest.45cddc9490be.4eec0295c921.json --output tmp/m3/2026-11-15/genuine-m2-handoff.json --require-importance-sensitive
```

Observed exit 0. Selected [16,20] and [51,57], 10.0 seconds total (12.58%).
Controlled early/late peaks select [0,4] and [75.5,79.5]. Budget/interval and
importance-sensitivity flags are true. Also explicitly checked >=4-second clips
and idempotent canonical normalize_highlights/check_highlights on each mode.

| Mode | Selected seconds | Clips | Integrated M2 importance (diagnostic) | Normalization valid |
|---|---:|---:|---:|---|
| U | 11.76 | 2 | 7.363653 | yes |
| I | 10.00 | 2 | 8.723270 | yes |
| A | 11.76 | 2 | 7.363653 | yes |
| J | 10.00 | 2 | 8.723270 | yes |

All modes were called with empty actions, so A uses uniform and J uses
importance-only by design. Importance mass is the supplied model curve's
objective diagnostic, not human F1 or an official score. This is not the real
M1-prediction U/I/A/J study requested by #16. Derived acceptance summary:
docs/evidence/M3_genuine_cache_acceptance_2026-11-15.json. Recipient acceptance:
https://github.com/harshjha15335/ACTIVA/issues/14#issuecomment-6893650150

### Fresh tests and remaining study gate

Project environment, authentic kit installed throughout:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Observed **398 passed, 3 skipped, 0 failed**, 3 warnings, 105.72 seconds.
Skips: configured-team-ID scenario and two opt-in genuine-checkpoint tests
without their separate local model/media assets. The cache consumer acceptance
above was actually run and is not substituted by those skipped inference tests.

Still needed: real M1 predictions and canonical GT/checkpoint/settings/source
hashes from #15, then matching M2 cache inputs requested in #18. #16's first
split item is done; real-prediction comparisons and their quality results table
remain open. No setting is frozen or changed without those measurements.

## M3 real predicted-action U/I/A/J ablation - 17 November 2026

**Outcome:** the current joint selector beats uniform on all four selector-held-out
participants: **37.41% vs 24.89% local coverage, +12.52 percentage points**.
Coverage-only wins this EPIC coverage comparison at 40.78%. **Retain the current
joint default.** No FPVSum human F1 is available to establish a global advantage
for coverage-only, and replacing the entrypoint with A would make the no-action
path uniform rather than importance-sensitive. No production code/default changed.

This completes the measurements/report requested in #16. The verified study is
on EPIC-100 TRAIN sources that ActionFormer trained on. The held-out split is
for selector decisions relative to the six tuning videos; it does not establish
unseen detector generalization or an official benchmark score. The ten videos
were selected by annotation diversity, so they are a small nonrandom cohort.

### Inputs, split and locked selection

- Source: main **9ce7ab9**, unchanged M3 code. Split remains the 15 November
  agreed six tuning / four held-out participants, outside the 187-video manifest.
- Received M1 archive SHA-256: `9b81f036eee7cc833fbc9db099f6ffef6ac9e3b7b4bd5b2a061417db1a4edf49`.
  Canonical GT, actual scored detections and model metadata, generated by M4
  from M1's exporter on main `9145bfe5e3bfae8e34e7b6942195a7498417e57e`.
- Received M2 bundle SHA-256: `a149caa8edadafaa88821e9581630f26023d12d6e5fb7a1e399c9546fded4837`.
  All ten inner ZIP/payload checksums and readers pass. Each source SHA-256
  matches its M1 JSON. Labels, class-ID mapping, source durations and splits
  validate; no endpoint required clamping or annotation repair.
- CLIP model identity: `ViT-B-32/sha256:c2ea63325cebca85a9eb509481f37ea991cb20199cab6642e68dbc8f973382f5`.
  All cache configuration hashes recompute to
  `11ba40022bdb54b1df0d5f7b1c7b60b7bedf0a05d43dd11e042a864988ff8bd3`:
  novelty importance / mean pooling. Real PTS grids match exactly.
- Compared the four EXISTING modes at unchanged defaults: J C/I weights 0.6/0.4,
  coverage fraction threshold 0.5, min confidence 0.2, cap 500 and greedy
  exponents 0/0.5/1. No additional hyperparameter grid was searched.
- Selection criterion: mean per-video GT distinct canonical-pair coverage,
  local `frac>=0.5` UNION-overlap rule. Ties prefer current J, then A/I/U.
  On six tuning videos A was selected (41.99% vs J 37.65%) and written to an
  immutable selection-lock file before the held-out coverage stage. All four
  modes were then evaluated in the single frozen held-out stage, with oracle
  inputs separately labeled. No settings were chosen from held-out outcomes.

### Coverage results (%)

| Variant | Tune, real predictions (n=6) | Held-out, real predictions (n=4) | Tune, oracle actions | Held-out, oracle actions |
|---|---:|---:|---:|---:|
| U: uniform | 25.43 | 24.89 | 25.43 | 24.89 |
| I: importance | 31.30 | 28.27 | 31.30 | 28.27 |
| A: coverage | **41.99** | **40.78** | 60.68 | 55.91 |
| J: current joint | 37.65 | 37.41 | 55.60 | 52.30 |

Oracle runs replace selector action inputs with canonical GT and confidence 1;
importance remains the same predicted curve. These are separate diagnostics,
not production predictions or a proven optimal upper bound. U/I are invariant,
as expected because they ignore action locations. Detection AP below continues
to use real predictions even in oracle-selection reports.

| Held-out video | U (%) | I (%) | A (%) | J (%) |
|---|---:|---:|---:|---:|
| P03_19 | 25.58 | 39.53 | 46.51 | **47.67** |
| P11_15 | 21.88 | 20.31 | **34.38** | 31.25 |
| P22_13 | 30.68 | 31.82 | **47.73** | 39.77 |
| P30_06 | 21.43 | 21.43 | **34.52** | 30.95 |

J beats U on 4/4; A beats J on 3/4. A's mean advantage over J is 3.37 points.
Four nonrandom held-out participants do not support a representative population
confidence claim; per-participant outcomes are shown instead of a p-value.
No human importance labels, F1, category accuracy or competition weighted total
are inferred from these EPIC inputs. Those fields remain null in the reports.

### Constraints, runtime and detector checks

All **80 selections** (10 videos x 4 modes x 2 action-input sources) satisfy the
strict 15% budget, >=4 s minimum, sorted/disjoint bounds and idempotent canonical
inward normalization. Zero budget violations; no short-policy exceptions.

| Real-prediction held-out mode | Mean clips | Mean partially cut prediction fraction | Mean selector seconds |
|---|---:|---:|---:|
| U | 12.50 | 0.1592 | 0.0051 |
| I | 16.00 | 0.1625 | 0.0452 |
| A | 17.00 | 0.1723 | 0.0979 |
| J | 16.50 | 0.1480 | 0.0806 |

Cut fraction counts all exported M1 predictions with nonzero but incomplete
selected-union overlap; it is not a GT/action-boundary accuracy metric. Runtime
is one observed selector-only CPU measurement on this i7-10750H machine,
excluding model inference, AP scoring and input generation. No latency-percentile
or GPU end-to-end claim is made.

Recomputed pooled detection mAP over tIoU 0.1/0.2/0.3/0.4/0.5:

| Split | Pair mAP (%) | Verb mAP (%) | Noun mAP (%) |
|---|---:|---:|---:|
| tune | 70.20 | 74.16 | 68.03 |
| held-out | 66.81 | 67.85 | 64.62 |

These reproduce M4's rounded handoff numbers and stay identical across oracle
and predicted selector reports. The detector was not changed or rerun here.

Post-evaluation diagnostic: a GT pair has some same-label exported prediction
(ignoring time) for 96.03% of tune / 95.31% of held-out classes on average. At
M3's existing score >=0.2 gate this is 57.07% / 60.19%, retaining about 75.7 /
107.5 predictions per video from roughly 2,000. This is label presence, NOT
temporal recall. It shows why the oracle gap cannot be attributed solely to
missing detector labels; confidence gating and selection also contribute.
No threshold was changed after this diagnostic or the held-out evaluation.

### Reproduction and evidence

Existing entrypoint, in the project environment (no new inference path):

```powershell
python scripts/run_ablation.py --study tmp/m3/2026-11-17/tune-study.json --review tmp/m3/2026-11-17/study-review.json --official-manifest kit/manifest.csv --coverage-rule 'frac>=0.5' --output tmp/m3/2026-11-17/tune-predicted.json
# Record the tune-only variant choice and tune-report SHA-256 before proceeding.
python scripts/run_ablation.py --study tmp/m3/2026-11-17/held_out-study.json --review tmp/m3/2026-11-17/study-review.json --official-manifest kit/manifest.csv --coverage-rule 'frac>=0.5' --output tmp/m3/2026-11-17/held-out-predicted.json
# The separate oracle runs use the same two commands with --oracle.
```

Private adapter inputs use the already-documented M3 study schema. Each M1 JSON
maps source.sha256 to input_sha256, predictions to predicted actions, gt to
annotation actions, and engine.assets.files['SlowFast.pyth'].sha256 to the
primary checkpoint field; all checkpoint hashes remain in adapter provenance.
Study entries point to those two JSONs and the matching unpacked M2 manifest.
The manifest-bound review records the agreed board split/source-hash evidence
and explicit model-training-overlap limits. It does not pretend that the agent
visually re-reviewed source videos. Full private reports/configs remain under
tmp/m3/2026-11-17; no arrays, raw GT, kit or original archive is published.

Derived public artifacts:

- [Complete summaries, per-video records, source hashes and tune-only lock](evidence/M3_real_epic_ablation_2026-11-17.json).
- [Ten-cache recipient validation](evidence/M3_matched_cache_acceptance_2026-11-17.json).
- [Coverage chart](evidence/M3_real_epic_ablation_2026-11-17.png).

Fresh full suite on unchanged code with the real kit installed:
`.\.venv\Scripts\python.exe -m pytest -q` -> **409 passed, 3 skipped, 0 failed**,
3 warnings, 144.59 s. Skips are the configured-team-ID case and two opt-in local
checkpoint inference tests. Actual received-cache and real-action comparisons
above ran successfully; they are not replaced by the skipped inference tests.

## Fixed action-density sensitivity study - 17 November 2026 (#28)

**Decision: retain J 0.6/0.4.** M4's rule was fixed before these runs: change
only if the same alternative strictly beats J0.6 at every density on both
partitions. No alternative passes even the tune-only condition. Production
selector settings and inference code are unchanged.

The inputs and participant-disjoint 6/4 partition are the same verified private
archives described above. Per video, filter confidence >=0.2, sort by decreasing
score then start/end/verb/noun, and keep `floor(density * duration_sec / 60)`.
`all` keeps every confident prediction. These explicit floor/tie conventions
were recorded on #28 before the experiment; sparse values differ slightly from
M4's illustrative table, whose rounding/tie convention was unspecified. The
method was not adjusted to reproduce that table. No GT chooses retained actions.
All selector parameters except the two specified J weight pairs remain fixed.

Local distinct canonical GT-pair coverage at union overlap >=50%, macro mean
over videos, in percent:

| Partition | Confident actions/min cap | U | I | A | J0.6/0.4 | J0.8/0.2 |
|---|---:|---:|---:|---:|---:|---:|
| Tune | 2 | 25.43 | 31.30 | 23.10 | 32.80 | 32.49 |
| Tune | 3.5 | 25.43 | 31.30 | 30.62 | 36.14 | 32.02 |
| Tune | 6 | 25.43 | 31.30 | 38.38 | 38.31 | 39.25 |
| Tune | all | 25.43 | 31.30 | 41.99 | 37.65 | 40.83 |
| Held-out | 2 | 24.89 | 28.27 | 18.92 | 29.71 | 28.83 |
| Held-out | 3.5 | 24.89 | 28.27 | 27.58 | 34.07 | 32.53 |
| Held-out | 6 | 24.89 | 28.27 | 33.26 | 33.83 | 35.00 |
| Held-out | all | 24.89 | 28.27 | 40.78 | 37.41 | 39.75 |

The six tune sources were evaluated first. An exclusively created decision-lock
file recorded zero eligible alternatives and the fixed rule before the density
held-out command. Sparse held-out conditions and J0.8 then ran in one frozen
stage. **Full-density U/I/A/J0.6 held-out results were already seen and published
in the preceding study before #28 arrived.** This is reuse of the existing
partition, not a fresh independent holdout. No post-held-out parameter search
or rule change followed. The lock, source hashes, exact counts, per-video
outcomes, settings and raw-report digests are in the
[density evidence JSON](evidence/M3_density_study_2026-11-17.json).
The [density chart](evidence/M3_density_study_2026-11-17.png) shows both partitions.

All **200** density selections (120 tune, 80 held-out) pass strict budget,
>=4-second clips, sorted/disjoint/in-range intervals and unchanged canonical
inward normalization. U/I are invariant to action density, as expected. J0.6
leads at 2 and 3.5 on both partitions; higher-density wins for A or J0.8 do not
satisfy the fixed decision rule. This supports retaining the existing trade-off
on these inputs; it does not establish FPVSum human F1 or an official total.
The source-training overlap and small, nonrandom cohort limitations above still
apply. Score-based thinning simulates fewer retained detections; it does not
reproduce the error distribution of a different action model.

Reproduction with the private matched study adapters:

```powershell
python scripts/run_ablation.py --study <tune-study.json> --review <study-review.json> --official-manifest kit/manifest.csv --coverage-rule 'frac>=0.5' --densities 2,3.5,6,all --output tmp/m3/density-tune.json
# Record the fixed decision rule, tune report digest and eligibility before proceeding.
python scripts/run_ablation.py --study <held_out-study.json> --review <study-review.json> --official-manifest kit/manifest.csv --coverage-rule 'frac>=0.5' --densities 2,3.5,6,all --output tmp/m3/density-held-out.json
```

Manifest annotations have not been scored. D-a remains a single report-only
measurement after the final outputs are committed and hashed; it cannot be used
to revise this selector. Card-mask no-harm and human FPVSum GO are separate gates.

Fresh full-suite verification after merging current `main` (D20 vocabulary
fallback): `.\.venv\Scripts\python.exe -m pytest -q` -> **417 passed, 4 skipped,
0 failed**, 3 warnings, 150.43 s. The additional skip is vocabulary/kit equality
with absent local M1 public class CSVs. Focused density, source-identity and
JSON-roundtrip checks pass; the synthetic density CLI also exits zero.
