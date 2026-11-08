# Ready-to-send M2 messages

These are drafts, not posted messages. Task labels below are local identifiers;
replace/link the real board issue when available. No ACK or acceptance is assumed.

## Member 1

```text
[FROM: M2] [TO: M1] [TYPE: REQUEST] [TASK: M2-TIMELINE-RUNTIME]
Current status: Real pretrained M2 CPU inference and development cache export work.
Completed: extract_clip_features(info)->ClipFeats, predict_importance(info,clip)
->Importance, classify_video(info,clip,imp)->CategoryResult; 74 M2 tests pass.
Timeline: actual decoded PTS minus video stream start_time (or first PTS), seconds,
2 fps target; rotation applied once before CLIP preprocessing. Dimension 512.
Need: your timestamp origin, duration/rotation conventions, verb/noun vocabulary,
Action.score semantics, preprocessing rate, Torch/CUDA/FFmpeg dependencies and
measured runtime. Please provide a real development action cache with start/end
seconds and confidence plus the exact source MP4 hash/metadata.
Expected decision: confirm the same input produces aligned M1/M2 timestamps;
identify whether a CLIP fallback is requested and any shared device/memory limits.
Artifact: m2-perception; docs/M2_HANDOFF.md; docs/evidence/M2_validation.json.
Acceptance: common-video timeline comparison passes, dependencies coexist and
cache/confidence conventions are explicitly acknowledged.
Open blockers: canonical M4 timeline/runtime approval and actual M1 artifacts.
Status: OPEN
```

## Member 3

```text
[FROM: M2] [TO: M3] [TYPE: REQUEST] [TASK: M2-REAL-FEATURE-HANDOFF]
Current status: Actual pretrained CLIP features and all four importance baseline
paths work offline on CPU. No held-out importance F1 or safe dataset split exists.
Completed: 74 M2 tests; your accessible cc6f36a suite 49 tests; actual optimizer
consumed genuine 20x512 float32 unit features with empty actions and matching
importance timestamps. Real-image semantic sanity and 2/5/10-minute runtime pass.
Model: explicitly selected ViT-B-32 QuickGELU LAION400M_e32, not OpenAI weights.
Model ID: ViT-B-32/sha256:9bd6bea2e0b7f343b17347af1854d7fd2667a227b0997801e5d297d46808854b
Grid: target 2 fps, actual increasing seconds, importance on exactly that grid,
finite scores [0,1], average-rank ties. Final model approval remains pending.
Artifact: docs/M2_HANDOFF.md; perception_cache.read_cache(); development manifest
walking-smoke.2cc842cf92e3.948165cb5395.json under dev_cache/manifests, with hashed
clip/importance NPZs. It is a public smoke clip, not cleared FPVSum benchmark data.
Need: acknowledge cache v1 fields/reader, model/dimension/grid/normalization;
provide B0 importance-only and B2 coverage-plus-importance implementation status,
CLIP use, F1 grid/tie/budget conventions and actual acceptance/ablation tests.
Controlled result: first-vs-last importance peaks left clip positions unchanged
in your explicit uniform stub. This is not a claim that B0/B2 was fixed.
Acceptance: consumer loads genuine artifacts, implements/tests the intended
importance objective, and agrees scorer parity/safe split with M4 before metrics.
Open blockers: your importance-aware objective, cache ACK, official scorer rules
and leakage-cleared development data. Production must never read these caches.
Status: OPEN
```

## Member 4

```text
[FROM: M2] [TO: M4] [TYPE: REVIEW] [TASK: M2-CANONICAL-INTEGRATION]
Current status: CONDITIONAL GO for development handover; NO-GO for submission.
Completed: CLI > env > shared-default precedence; real pretrained local CLIP,
canonical-vocabulary adapter, four baselines, three poolings, data/split guards,
hashed development caches, benchmark/evaluation tools and handover documentation.
Fresh evidence: 74 M2 tests pass; actual M3 consumes genuine 20x512 features.
CPU extraction for looped 2/5/10-minute fixtures: 13.257/40.846/94.276 seconds.
Need: official student kit, exact supported categories, evaluation manifest,
validator, canonical vocab.load/config/contracts/video_io/main/run.sh and runtime
rules. Please confirm whether your canonical repository/foundation now exists.
Shared config.py/contracts.py/video_io.py/requirements.txt were not changed by
remediation. M2 and accessible M3 roots remain unrelated: agree reviewed patch or
cherry-pick transfer onto the approved base; do not overwrite owner files.
Weights: official OpenCLIP GitHub LAION400M_e32 candidate was explicitly selected
because original OpenAI host remains blocked. Published SHA prefix verified;
full source and converted digests are in weights/README_weights.md. Approve the
checkpoint/preprocessing and asset distribution before freezing.
Expected artifacts: approved exact vocab and adapter conventions; official
manifest/scorer/validator; agreed timeline/device policy and integration plan.
Acceptance: canonical import/vocabulary tests pass, leakage decisions are reviewed,
real local weights work offline in your runner, and unseen MP4 output passes the
actual official validator. No synthetic/mock run is counted as that acceptance.
Open blockers: official artifacts, safe annotations/splits, held-out F1/accuracy,
M1/M3 integration, model approval and full final validation. TCN is untrained.
Artifact: docs/M2_HANDOFF.md; docs/M2_INTEGRATION_STATUS.md; docs/results.md.
Status: OPEN
```
