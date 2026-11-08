# M2 measured results — remediation

Full evidence: docs/evidence/M2_validation.json. The original independent audit is
retained outside the checkout under /workspace/audit-reports/activa-m2. These are
fresh results, not inherited counts. Source hashes and shared-owner files were
checked; no shared config/contracts/video_io or M3/M4 implementation was changed.

## Software and real-model tests

Command from /workspace/ACTIVA:

```sh
ACTIVA_REAL_CLIP_WEIGHTS=/workspace/ACTIVA/weights/ViT-B-32-laion400m_e32.safetensors ACTIVA_REAL_MEDIA=/workspace/remediation-results/activa-m2/media /workspace/activa-venv/bin/python -m pytest -q -p no:cacheprovider --junitxml=/workspace/remediation-results/activa-m2/pytest.xml
```

Fresh outcome: **74 passed,0 failed,0 skipped**,4 warnings,36.57 seconds.
Covers original behavior,exactCLIregression,paths/configflags,download/checksum/
manifest guards,canonical-vocab fixtures,processed/rawdataset parsing/fractional
labels,resampling,unsafe evaluation rejection,source-group splits,re-encoding
fingerprint candidates,cache integrity,actual pretrained semantic/video/baseline/
pooling/renamed-input repeatability and scoped Torch RNG.

Accessible M3 cc6f36a suite:49 passed,0 failed,0 skipped,1 warning,0.64seconds.
Actual source integration check consumed real cached20 x 512 features with empty
actions,matching importance grid and valid<=15percentintervals. Controlled first/
last peaks still produced the same interval[4.2500000005,5.7499999995] in10 seconds,
consistent with its explicit uniform stub. B0/B2 selection remains M3 work.

pip install -r requirements-dev.txt repeated successfully; pip check passed.
The real CLI reproduction also passed with ACTIVA_IMPORTANCE=constant and
ACTIVA_POOLING=top: output preserved those settings and all20scores were0.5.

## Real pretrained semantic and MP4 smoke

Checkpoint: explicit official OpenCLIP LAION400M_e32 B32 QuickGELU,not original
OpenAI. Published SHA prefix matched; full raw/converted hashes recorded in weight
documentation. Real photo similarity matrix(dog,bus):

```
[[0.239946,0.057791],
 [0.046162,0.246549]]
```

Each image ranked its corresponding description highest.512-D,float32,unitnorm;
repeat maximum difference0.0; normerror0.0; model.eval; socket connections denied.
This two-photo sanity test is **not category accuracy on FPVSum**.

Public smoke assets retained outside Git:
- https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
- https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
- https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi

A10-second MP4 derived from vtest.avi produced20x512real features and a development
label walking among explicitly supplied labels. Offline socket guard passed;
load1.542s,extract1.995s,importance0.0014s,category0.499s,total7.054s. It is a
surveillance sample,not a first-person benchmark or independently labeled study.
Renaming the input did not change extracted features in the real-model test.

## Actual pretrained CPU runtime

Linux x86_64,Python 3.12.14,Torch2.15.0+cpu,OpenCLIP4.2.1,IntelXeonPlatinum8573C,
5visibleCPU threads,batch16,FP32,noGPU. MP4 fixtures loop the public768x576,
10fpsOpenCVsurveillance sample. They are newly generated runtime fixtures,not
held-out first-person quality tests. All use the **actual pretrained checkpoint**
and socket-connect denial. PeakRSS is process resource.getrusage,notGPUmemory.

| Length | Embeddings | Load | Extract | Importance | Category | Total including imports | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 120 s | 240 | 1.975 s | 13.257 s | 0.0042 s | 0.492 s | 18.696 s | 1,605,904 |
| 300 s | 600 | 3.108 s | 40.846 s | 0.0101 s | 0.506 s | 47.536 s | 1,606,700 |
| 600 s | 1200 | 1.914 s | 94.276 s | 0.0688 s | 0.475 s | 99.870 s | 1,603,760 |

These timings replace no motion-only historical timings and do not demonstrate
GPU speed. Benchmarks ran in fresh processes in a shared development environment without exclusive CPU reservation; no paid resources were used.

## Honest unresolved checks

HumanimportanceF1,categoryaccuracy,poolingwinner,TCNtraining/CV: **not run**;
blocked by safe split/official manifest,annotation identity/alignment and scorer
agreement. Official vocabulary,validator,fullM1/M4run.sh,Windows,GPU and fresh-task
restoration remain unverified. No final model or baseline is frozen.

The actual FPVSum inspector exits2as intended: processedID mismatch and raw
categorycase mismatch remain unresolved. Fractional Label values are now parsed
without destructive binarization. No safe training IDs are emitted.
