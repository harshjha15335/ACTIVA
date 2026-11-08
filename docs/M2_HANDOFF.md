# M2 handover — conditional development integration

## 1. Overview and status

Pretrained CLIP extraction, four importance baselines, and zero-shot category
classification work on CPU. **CONDITIONAL GO for development handover; NO-GO for
final submission.** Official compatibility, safe data clearance, held-out quality,
final model approval, and M1/M4 orchestration remain outstanding.

## 2. Public interfaces

```python
extract_clip_features(info) -> ClipFeats
predict_importance(info, clip) -> Importance
classify_video(info, clip, imp) -> CategoryResult
```

The A5 dataclasses and function signatures are unchanged. M2 imports sibling
modules with `code/` on `sys.path`, as a `code/main.py` runner would. M4 must
review this import/configuration adapter convention.

## 3. Dependencies

Python 3.12; NumPy 2.3.5; Pillow 13.0.0; Torch 2.15.0+cpu;
torchvision 0.29.2+cpu; OpenCLIP 4.2.1; PyAV 19.0.2; FFmpeg/ffprobe on PATH.
Development dataset tools use SciPy 1.17.0 through requirements-m2.txt, included
by requirements-dev.txt. See README.md for installation commands.

Use the existing isolated checkout; no additional worktree is needed. Linux/CPU
was tested. Windows can invoke the Python CLI directly, but Windows and CUDA
execution were not tested. Select CPU explicitly as the fallback.

## 4. Checkpoint and offline setup

```sh
python scripts/prepare_weights.py --preset laion400m_e32
```

This explicitly installs OpenCLIP ViT-B-32 QuickGELU, **LAION-400M e32**. It is a
different pretrained checkpoint from OpenAI and requires team approval before
freezing. The existing OpenAI default is not silently replaced. Pass the
alternative weights explicitly in CLI/environment configuration.

See weights/README_weights.md for the upstream URL, published checksum prefix,
full observed hashes, verification limits, license reference, and manual route.
Inference loads local weights only and verifies the adjacent manifest if present.
Do not commit large assets; agree on distribution with M4.

## 5. Example invocation and outputs

```sh
python code/perception_cli.py input.mp4 outputs/perception.json --weights weights/ViT-B-32-laion400m_e32.safetensors --categories /absolute/path/development-categories.json --cache-dir dev_cache
```

The wrapper `perceive.sh` is also available on Linux. Output is M2 diagnostic
JSON, not organizer output. M4 retains ownership of run.sh/main/schema. Internal
CategoryResult.scores must remain internal unless the official schema permits them.

```python
info = inspect_video(path)
clip = extract_clip_features(info)
imp = predict_importance(info, clip)
cat = classify_video(info, clip, imp)
```

Configure local weights and canonical vocabulary or explicit development labels
before calling these functions. No ground truth or development cache is read by
production inference.

## 6. Timestamp conventions

Target CLIP rate is 2 fps. The decoder uses actual decoded PTS, subtracting video
stream start_time or the first decoded timestamp when start_time is absent.
Frames are selected at or after each target slot; returned times are actual PTS,
not fabricated uniform timestamps. RGB rotation precedes library preprocessing.
Times strictly increase and lie below duration. Low source rates can yield fewer
unique samples. Missing PTS fails clearly.

M1/M4 must confirm the same time origin, duration and rotation ownership. Shared
video_io.py was not modified during remediation.

## 7. Features, model identity and precision

The validated candidate returns `[N,512]` normalized float32 embeddings and `[N]`
float64 timestamps. The model identifier is:

```text
ViT-B-32/sha256:9bd6bea2e0b7f343b17347af1854d7fd2667a227b0997801e5d297d46808854b
```

The digest identifies the installed converted artifact. Checkpoint replacement
invalidates backend caching; classification rejects mismatched hashed image/text
identities. Initialization preserves the caller's Torch RNG state and does not
change its global deterministic-algorithm policy. Optional CUDA FP16 is implemented
but untested; CPU stays FP32. Center cropping is unchanged; no crop/pad winner was
selected without approved domain data.

## 8. Development cache v1

`perception_cache.write_cache` writes atomic NPZ payloads and a JSON manifest:

- `clip/*.npz`: times, feats, model_name.
- `importance/*.npz`: times, values.
- `manifests/*.json`: video duration/metadata, input hash, checkpoint identity,
  dimension, target fps, array shapes, configuration, and payload hashes.

Keys include input and configuration/model identities to avoid basename collisions.
`read_cache(path, expected_model=...)` returns info, clip, imp, manifest and validates
hashes, bounded paths, shapes, float32 norms, matching grids and score ranges.
It uses allow_pickle=False. It is a development consumer helper, never an inference
input. The format remains proposed until M3 acknowledges it.

The genuine smoke artifact currently available is:

```text
dev_cache/manifests/walking-smoke.2cc842cf92e3.948165cb5395.json
```

It contains 20 x 512 features from a public 10-second OpenCV surveillance clip.
It is not FPVSum training/evaluation data. Legacy motion-only NPZ diagnostics are
separate and are not genuine CLIP handover artifacts.

## 9. Importance semantics

Constant returns 0.5. Novelty uses one minus cosine similarity to the normalized
mean in a +/-10-second neighborhood. Motion uses average absolute RGB changes
at 32 x 32, aligned to the CLIP grid. Text guidance uses genuine CLIP positive
minus negative prompt similarities. Nonconstant modes use light five-sample
smoothing and deterministic average midranks. Ties remain equal; a constant curve
stays 0.5. Values are finite in [0,1] on exactly the CLIP timestamps.

No baseline has been selected by held-out F1. Novelty is a development default,
not a measured winner. The TCN remains untrained and is not deployed.

## 10. Category vocabulary and pooling

The M2 adapter prefers the exact four-item M4 vocab.load() contract when present.
Conflicting development labels fail. Strict mode
`--require-canonical-vocabulary` rejects an unavailable canonical vocabulary.
Development JSON labels are explicitly identified in diagnostic output; they do
not certify official compatibility.

Five text templates produce normalized class prototypes. Mean pooling, weighted
pooling with epsilon 0.1, and top-30% pooling are implemented. All-zero importance
is valid. Real FPVSum category accuracy and the best pooling method are unknown.

## 11. Tests

The fresh full run has **74 passed, 0 failed, 0 skipped**, with four warnings.
It includes opt-in genuine pretrained image/video tests, renamed-input determinism,
all baselines/pooling methods, cache integrity, canonical vocabulary fixtures,
configuration regression and data guards. Without explicitly supplied real assets,
real-model tests visibly skip as BLOCKED rather than substituting random weights.

The accessible M3 suite separately passed all 49 tests. Exact commands and evidence
are in docs/results.md and docs/evidence/M2_validation.json.

## 12. Runtime

Actual pretrained CPU extraction on looped 768 x 576 public surveillance fixtures:

| Video duration | Vectors | Extraction | Total including imports |
|---|---:|---:|---:|
| 120 s | 240 | 13.257 s | 18.696 s |
| 300 s | 600 | 40.846 s | 47.536 s |
| 600 s | 1200 | 94.276 s | 99.870 s |

Batch 16, five visible CPU threads, peak RSS about 1.53 GiB. These are real-model
runtime checks, not held-out egocentric quality results. GPU/Windows unmeasured.

## 13. M3 acceptance

Actual M3 commit cc6f36a consumed the genuine cache with empty actions and emitted
valid local-budget intervals. Early versus late importance peaks still produced
identical locations, consistent with its documented uniform stub. B0/B2 selection
is not claimed fixed. Run scripts/check_m3_handoff.py against M3's actual optimizer;
--require-importance-sensitive reports failure for the inspected stub.

M3 must acknowledge the cache/model/grid, normalization and scorer conventions,
and provide importance-aware selection plus ablation evidence.

## 14. M4 acceptance

M4 must supply/approve the official kit, categories, evaluation manifest, validator,
canonical shared files, asset packaging and runtime rules. Existing config.py,
contracts.py, video_io.py and requirements.txt were not changed by remediation.
M2/M3 have unrelated roots; reviewed patch/cherry-pick integration onto an approved
foundation requires team agreement. No histories were force-merged.

## 15. Dataset/evaluation gates

Status: BLOCKED BY MANIFEST; no approved training IDs. Parsers preserve processed
scores and fractional raw values, report unresolved ID/category mappings and
require original score-aligned PTS. Reviewed manifest-bound decisions and exact
video hashes gate held-out evaluation. Sources/groups/hashes cannot cross splits.
Fingerprints create review candidates, never automatic clearance.

The evaluation tool can report sample-grid ranking-F1 proxies, category accuracy,
sample counts and confusion matrices after clearance. No quality run occurred.
These proxies are not original-frame or final-highlight Samsung metrics.

## 16. Remaining blockers

Official vocabulary/manifest/validator and canonical approval; annotation identity
and timeline review; held-out F1/accuracy and model selection; conditional TCN
training; M1 action/timeline/runtime compatibility; M3 importance-aware selection;
full offline run.sh/schema validation; final weight approval and human review.

## 17. Recipient actions

M1: confirm time origin, units, duration, confidence, dependencies and action caches.
M3: load the real manifest, acknowledge the format, provide B0/B2 and scorer status.
M4: provide official artifacts, approve adapters/assets/foundation, run the integrated
validator. Ready-to-send A9 messages are in docs/M2_TEAM_MESSAGES.md. None were
posted; requests stay OPEN until recipients confirm acceptance.
