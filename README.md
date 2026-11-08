# ACTIVA — Member 2 Perception

Real pretrained CLIP extraction, four importance baselines and zero-shot category
classification are implemented. **Development handover is conditional; this is
not a complete or officially validated ACTIVA submission.** Read
[docs/M2_HANDOFF.md](docs/M2_HANDOFF.md) and
[docs/M2_INTEGRATION_STATUS.md](docs/M2_INTEGRATION_STATUS.md).

## Install

Python 3.12 and FFmpeg/ffprobe on PATH are required. Use the existing isolated
checkout; do not create another worktree unless explicitly requested.

```sh
python3 -m venv --system-site-packages /workspace/activa-venv
/workspace/activa-venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
/workspace/activa-venv/bin/python -m pip install -r requirements-dev.txt
```

CPU/Linux is validated. For Windows use a Python 3.12 venv, install the appropriate
official PyTorch wheels, requirements-dev.txt, and FFmpeg. Invoke the Python CLI
directly instead of the Bash wrapper. Windows execution has not been tested here.
No CUDA device was available; CPU is the explicit supported fallback.

## Explicit checkpoint choice

OpenAI's original host remains denied in this environment. A compatible **different**
pretrained checkpoint from OpenCLIP's official GitHub release was acquired and
validated: ViT-B-32, QuickGELU, LAION-400M e32,512 dimensions.
This is an explicit development candidate; team model approval is pending.
It is not silently substituted for OpenAI defaults.

```sh
/workspace/activa-venv/bin/python scripts/prepare_weights.py --preset laion400m_e32
```

See [weights/README_weights.md](weights/README_weights.md) for the source, exact
hashes, verification level, license reference and original-OpenAI/manual routes.
Setup downloads only on explicit request; inference never downloads. Weights and
videos stay out of Git. Checkpoint manifests are verified on local model loading.

## Run

```sh
ACTIVA_PYTHON=/workspace/activa-venv/bin/python ./perceive.sh input.mp4 outputs/perception.json --weights weights/ViT-B-32-laion400m_e32.safetensors --categories /absolute/path/development-categories.json --importance novelty --pooling mean --cache-dir dev_cache
```

Equivalent cross-platform invocation:

```sh
python code/perception_cli.py input.mp4 outputs/perception.json --weights weights/ViT-B-32-laion400m_e32.safetensors --categories development-categories.json
```

Category JSON is a unique nonempty exact-label array (or object with categories).
When M4's vocab.load() is available its canonical labels take precedence and
conflicting development labels fail. `--require-canonical-vocabulary` rejects the
missing canonical adapter. Development-label outputs explicitly record their
source; they are never claimed officially compatible.

Motion-only diagnostics require no weights and emit a null category:

```sh
ACTIVA_PYTHON=/workspace/activa-venv/bin/python ./perceive.sh input.mp4 outputs/motion.json --mode motion-only
```

Output is `activa-m2-diagnostic-v1`, not the official final schema. M4 owns the
actual main/run.sh/schema integration; no competing submission runner is created.

## Configuration

Explicit CLI > environment > shared defaults. Omitted flags preserve environment
values. M4-owned config.py is unchanged; perception_options.py validates and
resolves its settings. Explicit relative paths are caller-relative; implicit
weight defaults resolve against this checkout.

- ACTIVA_IMPORTANCE / --importance: constant,novelty,motion,text.
- ACTIVA_POOLING / --pooling: mean,weighted,top(top30percent).
- ACTIVA_CLIP_WEIGHTS / --weights; ACTIVA_CLIP_MODEL / --model.
- ACTIVA_DEVICE / --device: cpu,cuda,cuda:index; ACTIVA_BATCH_SIZE / --batch-size.
- ACTIVA_CATEGORIES / --categories: explicit development JSON.
- ACTIVA_REQUIRE_CANONICAL_VOCAB=1 / --require-canonical-vocabulary.
- ACTIVA_CLIP_FP16=1 / --fp16: optional CUDA autocast (untestedGPU); CPU staysFP32.

No global Torch RNG reseed or deterministic-algorithm policy change escapes model
initialization. Replacing checkpoint files invalidates the backend cache. Features
identify the actual checkpoint SHA256, not only an architecture name.

## Tests and tools

```sh
/workspace/activa-venv/bin/python -m pytest -q
```

Real-model tests explicitly require opt-in local assets; without them those tests
report BLOCKED skips, never random pretrained substitutes. The full executed run:

```sh
ACTIVA_REAL_CLIP_WEIGHTS=/workspace/ACTIVA/weights/ViT-B-32-laion400m_e32.safetensors ACTIVA_REAL_MEDIA=/workspace/remediation-results/activa-m2/media /workspace/activa-venv/bin/python -m pytest -q
```

Observed:74 passed,0 failed,0 skipped,4 warnings. Media are small public dog/bus photos
and an MP4 derived from OpenCV vtest.avi, retained locally, not committed.
Setup/real tests do not fetch assets implicitly. Source URLs are in docs/results.md.

Other tools: scripts/validate_pretrained.py, benchmark_perception.py,
inspect_fpvsum.py, fingerprint_videos.py, audit_ids.py, evaluate_perception.py and
check_m3_handoff.py. See docs/evaluation_formats.md for explicit development input
schemas and safeguards. No training, tuning, held-out quality result or official
score exists yet. Do not train until docs/leakage.md clearance is reviewed.
