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
