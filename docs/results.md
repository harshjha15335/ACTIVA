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
