# M2 remediation and integration status

## REPORT 1 — Before vs After

The original audit rated M2 **44/100, partially implemented, NO-GO**. This work
implements the independent fixes and validates a genuine pretrained development
candidate. **CONDITIONAL GO for development handover; final submission remains
NO-GO.** No protected data was trained on, no paid service was used, and shared
config/contracts/decoder or teammates' implementations were not changed.

| Audit finding | Before | After / evidence |
|---|---|---|
| F01: pretrained weights | Missing | Explicit OpenCLIP LAION-400M e32 candidate acquired; real semantic/offline/video checks pass. Original OpenAI host remains blocked. |
| F02: vocabulary | Development JSON only | Canonical M4 adapter and strict rejection tested; actual official vocabulary still missing. |
| F03: leakage | Preliminary exact-match tool | Strict CSV/MAT parsers, reviewed aliases, explicit PTS resampling, fingerprint candidates and manifest/split/hash guards implemented. No clearance granted. |
| F04: model quality | No evaluation tools/results | Guarded baseline/pooling evaluation and tested proxy metrics implemented; no held-out quality run. |
| F05: full integration | Missing | M2 handover prepared; actual final M1/M4 runner/schema/validator remains unavailable. |
| F06: CLI precedence | Confirmed bug | Fixed; exact reproduction passes with real CLIP. |
| F07: cache provenance | Generic model name and basename collisions | SHA-based model identity, atomic versioned artifacts and validated reader; actual M3 consumed genuine features. |
| F08: CWD/RNG/cache state | Caller-relative defaults and global reseeding | Project-relative implicit defaults, scoped RNG and file-aware caching tested. |
| F09: unrelated roots | No common foundation | Preserved; team-approved patch/cherry-pick integration still required. |
| F10: real reliability | Tiny random model only | 74 fresh M2 passes, including real 512-D inference, all baselines/pooling and runtime measurements. |
| F11: handover | No genuine cache/package | Real smoke cache, handover, messages and consumer check prepared; recipient acceptance pending. |
| F12: documentation | Stale status/layout | Updated results/leakage/decisions and canonical weight documentation. |
| F13: optional improvements | No measured basis | Runtime now measured; functioning modules were not broadly rewritten or tuned speculatively. |

## REPORT 2 — Files Changed


Each row states previous issue, implemented change/purpose, verification and result.
New files are M2 adapters/development tools, not replacements for M4's production
files. Dataset/evaluation tools are not imported by production perception inference.

| File | Previous issue | Change and purpose | Tests/evidence | Result |
|---|---|---|---|---|
| `.gitignore` | Canonical weight documentation was ignored. | Allow the small README_weights.md; keep model assets ignored. | Packaging check | PASS |
| `README.md` | Only the initial standalone workflow was documented. | Document explicit candidate, precedence, strict vocabulary, tools and fresh evidence. | Cross-checked commands/results | PASS |
| `code/category.py` | Development vocabulary only; no image/text identity guard. | Use canonical adapter; reject conflicting categories/checkpoint identities. | Vocabulary and real pooling tests | PASS |
| `code/category_vocab.py` | No canonical vocabulary bridge. | Prefer exact M4 vocab.load; validate labels and support strict blocked state. | Canonical fixture/conflict tests | PASS |
| `code/clip_backend.py` | Path-only cache, global RNG mutation, weak model identity. | File-aware cache; scoped RNG; SHA identity/manifest verification; stable normalization; optional CUDA autocast. | Checkpoint, RNG and real offline tests | PASS |
| `code/clip_features.py` | Generic local model name; no optional CUDA autocast. | Record actual weight SHA and use validated M2 settings/autocast. | Real 512-D norm/timeline/determinism tests | PASS |
| `code/fpvsum_data.py` | No actual dataset parser/alignment/split gate. | Strict loaders, explicit aliases, preserve fractional labels, explicit PTS integration and reviewed split/hash guards. | Dataset and unsafe-study tests | PASS |
| `code/importance.py` | Configuration could accept invalid selections. | Route settings through M2 validator without changing baseline mathematics. | Baseline and invalid-configuration tests | PASS |
| `code/leakage_fingerprints.py` | No decoded-content candidate checks. | First-seconds dHash candidates with duration/aspect/time checks; never clearance. | Synthetic re-encoding and no-clearance tests | PASS |
| `code/perception_cache.py` | No hashed consumer contract. | Atomic versioned cache/manifest; checksum/path/shape/grid validation. | Round-trip, tamper and collision tests; actual M3 consumption | PASS |
| `code/perception_cli.py` | Defaults overwrote environment; no canonical/provenance labels. | Fix precedence, expose explicit overrides, label vocabulary, export versioned caches before JSON. | Exact bug regression and real CLI reproduction | PASS |
| `code/perception_metrics.py` | No independent baseline/category metric helper. | Tested labeled ranking-F1 proxy and category counts/matrices. | Known-answer metric tests | PASS |
| `code/perception_options.py` | No isolated validation adapter. | Validate choices/flags and resolve implicit default assets independently of CWD. | Configuration tests | PASS |
| `docs/M2_HANDOFF.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/M2_INTEGRATION_STATUS.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/M2_TEAM_MESSAGES.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/decisions.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/evaluation_formats.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/evidence/M2_validation.json` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/leakage.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `docs/results.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `requirements-dev.txt` | Dataset test dependency missing. | Include isolated M2 development requirements. | Reinstallation and pip check | PASS |
| `requirements-m2.txt` | SciPy parser dependency undeclared. | Pin SciPy 1.17.0 for dataset tools/tests. | Installed dependency verification | PASS |
| `scripts/audit_ids.py` | Only exact ID/hash comparisons. | Add optional unresolved perceptual-content candidates without granting training safety. | Existing audit test and candidate helpers | PASS / documented external blocker |
| `scripts/benchmark_perception.py` | No actual stage/memory profiler. | Measure real model load/extraction/importance/category, imports and RSS offline. | Real 2/5/10-minute runs | PASS / documented external blocker |
| `scripts/check_m3_handoff.py` | Only synthetic consumer evidence. | Exercise actual M3 with genuine cache and controlled importance peaks. | Actual M3 numerical consumption passes; sensitivity missing | PASS / documented external blocker |
| `scripts/evaluate_perception.py` | No leakage-gated held-out comparison tool. | Review/hash gate; processed-score alignment; four baselines/three poolings; explicit proxy outputs. | Unsafe evaluation rejects before model use; metrics tests; real study BLOCKED | PASS / documented external blocker |
| `scripts/fingerprint_videos.py` | No fingerprint producer. | Generate documented development review inventories. | Re-encoding helper tests | PASS / documented external blocker |
| `scripts/inspect_fpvsum.py` | Upstream parser state undocumented. | Report strict CSV/MAT validation outcomes without automatic repair. | Actual upstream run reports blocked discrepancies | PASS / documented external blocker |
| `scripts/prepare_weights.py` | Original host blocked; setup lacked alternate/manual/manifest guards. | Explicit official-release preset; preserve published identity; complete-download, tensor-format and manifest checks. | Setup/download/checksum tests; repeated real installation | PASS / documented external blocker |
| `scripts/validate_pretrained.py` | No real semantic sanity evidence. | Explicit image/text check with local weights and blocked sockets. | Dog/bus real-image sanity | PASS / documented external blocker |
| `tests/test_checkpoint_setup.py` | Missing focused regression/real-asset coverage. | Add focused behavior tests without mocking real-pretrained success. | Fresh full suite | PASS |
| `tests/test_configuration.py` | Missing focused regression/real-asset coverage. | Add focused behavior tests without mocking real-pretrained success. | Fresh full suite | PASS |
| `tests/test_data_and_cache.py` | Missing focused regression/real-asset coverage. | Add focused behavior tests without mocking real-pretrained success. | Fresh full suite | PASS |
| `tests/test_real_pretrained.py` | Missing focused regression/real-asset coverage. | Add focused behavior tests without mocking real-pretrained success. | Fresh full suite | PASS |
| `tests/test_vocab_and_metrics.py` | Missing focused regression/real-asset coverage. | Add focused behavior tests without mocking real-pretrained success. | Fresh full suite | PASS |
| `weights/README.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |
| `weights/README_weights.md` | Handover/evidence incomplete or stale. | Record actual behavior, results, source/hash limits and explicit blockers. | Checked against executed JSON/log artifacts | PASS |


## REPORT 3 — Real CLIP Status

**REAL PRETRAINED INFERENCE PASS** for the explicit development candidate:
OpenCLIP 4.2.1, ViT-B-32 with QuickGELU, **LAION-400M e32**, 512 dimensions,
CPU FP32. This is different from OpenAI's trained checkpoint and awaits team
approval before freezing. The original shared default is not silently replaced.

The official OpenCLIP GitHub release filename's published SHA-256 prefix
`46683a32` was verified. The full observed source digest is:

```text
46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa
```

The converted local safetensors digest is:

```text
9bd6bea2e0b7f343b17347af1854d7fd2667a227b0997801e5d297d46808854b
```

An independently published full source digest was unavailable; the full observed
hash is not misrepresented as one. Explicit full hashes cannot override the
preset's published identity. TLS/checksum verification was never disabled.
Setup checks incomplete downloads and formats, supports trusted manual archives,
and passed repeat installation without another download. Inference verifies the
adjacent local manifest and never invokes the downloader.

Real dog/bus photographs ranked their matching descriptions highest. Features
are float32, unit-normalized and repeat exactly in the measured CPU test (maximum
error 0.0; tolerance 1e-6). Model evaluation mode, image/text execution and offline
socket denial passed. A new surveillance MP4 and its renamed copy also passed.
These smoke checks do not establish FPVSum accuracy or egocentric generalization.
Windows, GPU and final model approval remain outstanding.

## REPORT 4 — Importance Status

All four baselines run with genuine CLIP features: constant, novelty, decoded
motion and text guidance. They return finite values in [0,1] on exactly the CLIP
grid. Smoothing, normalization, ties, black/synthetic footage and all pooling
paths are tested.

No human-importance F1 comparison has been run. The new evaluation tool is gated
by reviewed data and explicit annotation alignment. Its independently tested
ranking-F1 proxy is clearly distinguished from original-frame and final-highlight
F1. Novelty remains a configurable development default, not a measured winner.
No final shipping model is selected. The TCN is still an untrained architecture
outside predict_importance; conditional training prerequisites remain unmet.

## REPORT 5 — Category Status

Genuine image/text classification works offline. Mean, epsilon-0.1 weighted and
top-30% pooling support zero/constant importance and deterministic ties. The M4
vocab.load adapter and conflict/strict-mode tests pass. Missing canonical data
cannot silently become official vocabulary; development output records its source.

**Official vocabulary compatibility is BLOCKED.** No held-out category accuracy,
per-class result, confusion matrix or pooling winner is claimed. The guarded tool
can produce those reports after a cleared study and reviewed label mapping exist.
Two photograph sanity cases are not a substitute for that evaluation.

## REPORT 6 — Leakage Status

**BLOCKED BY MANIFEST — NOT CLEARED. No training/tuning IDs are approved.**

Actual upstream files confirm 98 videos: 56 labeled, 42 unlabeled. The processed
and raw video-ID discrepancy remains unresolved, as does BikePolo/Bikepolo casing.
Aliases require explicit review; none was invented or applied. Some raw Label
vectors are fractional, so the parser preserves them rather than binarizing or
assuming a raw-average reconstruction of processed Score.

Resampling requires original score-aligned PTS and uses documented weighted bin
means. Exact IDs, hashes and first-seconds fingerprints can produce exclusion or
review candidates, never automatic safe status. Reviewed decisions are bound to
the supplied official manifest and actual video hashes. Source IDs/groups and
identical hashes cannot cross train/validation splits. That human attestation
format does not independently authenticate an arbitrary file as the official kit.
M4 must supply and interpret the real artifact. No EPIC validation tuning occurred.

## REPORT 7 — Testing Results

Fresh full M2 command:

```sh
ACTIVA_REAL_CLIP_WEIGHTS=/workspace/ACTIVA/weights/ViT-B-32-laion400m_e32.safetensors ACTIVA_REAL_MEDIA=/workspace/remediation-results/activa-m2/media /workspace/activa-venv/bin/python -m pytest -q -p no:cacheprovider --junitxml=/workspace/remediation-results/activa-m2/pytest.xml
```

**74 passed, 0 failed, 0 skipped, 4 warnings, 36.57 seconds.**
The accessible M3 suite separately passed all 49 tests. Dependencies reinstalled
successfully and pip check passed. Unit, synthetic, actual consumer integration
and genuine pretrained evidence are identified separately in results.md.

The upstream inspector intentionally exits 2 on unresolved ID/category differences.
The M3 controlled-peak probe records missing importance sensitivity. These are
retained blockers, not hidden passing claims. Held-out quality is BLOCKED; GPU,
Windows, official end-to-end validation and fresh-task restoration were not run.
Real-asset tests visibly skip as BLOCKED when their opt-in assets are absent.

## REPORT 8 — Performance

Actual final-checkpoint measurements, CPU, batch 16, looped public OpenCV footage
at 768 x 576 and 10 source fps:

| Length | Vectors | Load s | Extract s | Importance s | Category s | Total incl. imports s | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 120 s | 240 | 1.975 | 13.257 | 0.0042 | 0.492 | 18.696 | 1,605,904 |
| 300 s | 600 | 3.108 | 40.846 | 0.0101 | 0.506 | 47.536 | 1,606,700 |
| 600 s | 1200 | 1.914 | 94.276 | 0.0688 | 0.475 | 99.870 | 1,603,760 |

Linux x86_64, Python 3.12.14, Torch 2.15.0+cpu, Intel Xeon Platinum 8573C, five
visible CPU threads. These fresh processes used a shared development environment;
the later timing round overlapped regression work and had no exclusive CPU
reservation. Earlier extraction measurements were 14.408/32.886/65.420 seconds.
Both rounds are retained in the evidence bundle. They show observed development
runtime, not guaranteed throughput. No motion-only timings were relabeled as CLIP.

The inputs are looped surveillance footage, not held-out first-person quality
data. GPU utilization/memory and Windows runtime remain unmeasured.

## REPORT 9 — Integration Matrix

| Member | What works | What is missing | Required action |
|---|---|---|---|
| M1 | M2 publishes actual-PTS/seconds/grid/device conventions and preserves shared Torch policy | Actual action source/cache and common-video timeline test | Confirm origin, units, duration, rotation, confidence, vocabulary, dependencies and runtime. |
| M3 | Actual optimizer consumes genuine 20 x 512 features with empty actions, matching importance and valid local budget | Uniform stub does not select locations using importance; no format ACK or approved scorer | Provide B0/B2, accept cache/model/grid, agree F1 conventions and run real-prediction ablations. |
| M4 | Guide signatures unchanged; canonical adapter fixtures, local assets and handover ready | Kit, vocabulary, manifest, validator, canonical approval and runner/foundation agreement | Approve adapters/checkpoint/package; integrate and validate actual offline final JSON. |

No M3/M4 files were changed. No unrelated histories were merged. A reviewed patch
or cherry-pick transfer onto an approved common base remains a team decision.
Three complete A9.3 messages are ready in M2_TEAM_MESSAGES.md; they were not posted
and no recipient acceptance was assumed.

## REPORT 10 — Outstanding Issues

| Severity | Issue | Acceptance condition / owner |
|---|---|---|
| P0, release | No official runner/schema/validator result | M4 integrates real modules and an unseen MP4 passes the actual validator offline. |
| P1 | Actual official vocabulary and canonical approval | M4 supplies exact strings and approves import/settings/timeline adapters; strict mode passes. |
| P1 | Data identity, timeline and manifest clearance | M4/M2 review source/video/annotation mappings and safe folds with evidence. |
| P1 | Held-out quality and model selection absent | M2/M3 measure baseline F1, category accuracy, pooling and scorer parity; only then consider TCN. |
| P1 | M3 importance-aware objective absent | Owner demonstrates B0/B2 behavior and ablations on actual predictions. |
| P2 | Checkpoint/cache/foundation approval | Team approves the explicit candidate, asset distribution and reviewed shared-base integration. |
| P2 | GPU, Windows and new-task restoration unverified | Run relevant platform/resource checks after hardware and official limits are known. |
| P3 | Prototype caching/shared decoding/style polish | Implement only where profiling supports benefit and owner agreement permits it. |

No known P0 prevents the tested standalone CPU development workflow. The separate
release blockers remain; it would be incorrect to call the whole submission ready.

## REPORT 11 — Completion Score

Provisional internal engineering rubric, not a Samsung score or a measured
percentage of all work:

| Area | Max | Before | Now | Evidence/remaining deductions |
|---|---:|---:|---:|---|
| CLIP extraction | 20 | 8 | 18 | Real 512-D semantic/offline/video/runtime checks; model approval/domain/platform limits remain. |
| Importance baselines | 10 | 7 | 8 | Real execution and tested metrics; no human F1. |
| TCN and valid training | 15 | 3 | 3 | Architecture only; training correctly gated. |
| Category classification | 10 | 4 | 6 | Real encoders/pooling and canonical adapter; no official labels or held-out accuracy. |
| Leakage safeguards | 10 | 3 | 5 | Parsers, fingerprints and review/split/hash guards; no clearance. |
| Shared interfaces | 10 | 5 | 7 | Actual M3 consumer and adapter fixtures; no full canonical integration. |
| Tests and reliability | 10 | 7 | 9 | 74 fresh passes including real weights; external platform limits remain. |
| Offline/runtime | 5 | 2 | 4 | Actual CPU offline and long-video profiles; no new-task/GPU check. |
| Documentation | 5 | 4 | 5 | Sources, verification limits, commands, evidence, formats and handover. |
| Team handover | 5 | 1 | 3 | Genuine cache and documents; no accepted handoff/merge. |
| **TOTAL** | **100** | **44** | **68** | Provisional evidence-based judgment. |

A validated baseline may satisfy importance needs. The TCN deduction reflects
absent training evidence, not a recommendation to train prematurely or retain an
inferior model merely to increase a checklist score.

## REPORT 12 — Final Handover Readiness

**CONDITIONAL GO for development handover. NO-GO for final submission.**

1. Pretrained CLIP works: yes, the explicit LAION-400M e32 candidate.
2. Offline inference works: yes, local CPU image/text/MP4 paths with connections blocked.
3. Importance scoring works: yes, all four numerical baseline paths.
4. Human-importance quality measured: no; held-out F1 is blocked.
5. Category classification valid: configured-label execution works; dataset accuracy unknown.
6. Official vocabulary integrated: adapter tested; actual artifact unavailable.
7. Leakage complete: no; no approved training IDs.
8. TCN trained/worthwhile: no training or superiority claim.
9. M3 consumes actual output: yes numerically; its uniform locations ignore importance.
10. M4 can integrate: source/assets/adapters are ready for review, not already approved.
11. Tests pass: 74 M2 and 49 M3 fresh passes; external blockers are separate.
12. Missing: official artifacts, safe labels/splits, quality results and accepted final integration.
13. Team dependencies: M1 timeline/actions/runtime; M3 B0/B2/cache/scorer; M4 kit/vocab/manifest/validator/base.
14. Before submission: clear those gates, freeze based on evidence and obtain human review.
15. Handover ready: yes as a conditional development artifact, not a completed benchmark-validated submission.

**What was fixed:** configuration, verified setup/loading, model/cache identity,
canonical adapter, dataset guards and evaluation/benchmark tools.

**What now works:** genuine pretrained offline CPU inference, all baseline/pooling
paths, versioned cache export, real M3 type consumption and the regression suite.

**What still fails or blocks:** original OpenAI host access, unresolved upstream
identity/category checks, canonical strict mode without M4 data, M3 importance
sensitivity, and final official/data-quality gates.

**What is needed next:** use the precise message drafts to obtain owner decisions,
approve the candidate/cache/adapters, supply the kit/manifest/vocabulary, review
annotation/split safety, run held-out comparisons, then validate final integration.
