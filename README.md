# ACTIVA — M2 Perception

Standalone perception implementation from the uploaded M2 guide. This is not a
complete ACTIVA submission: M1 action detection, M3 selection and official M4
schema/validator integration are not implemented here. Guide A5 contracts and a
local decoder have been bootstrapped so M2 can be developed independently.
M4 should review these adapters when integrating.

## Install (Python 3.12, FFmpeg/ffprobe)

Use the existing isolated checkout; no additional worktree is needed.

```sh
python3 -m venv --system-site-packages /workspace/activa-venv
/workspace/activa-venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
/workspace/activa-venv/bin/python -m pip install -r requirements-dev.txt
```

For a CUDA machine install the matching official PyTorch build instead. CPU is
the tested local configuration. No real GPU runtime or performance is claimed.

## Run without weights

```sh
ACTIVA_PYTHON=/workspace/activa-venv/bin/python ./perceive.sh input.mp4 outputs/motion.json --mode motion-only
```

This decodes real frames at 2 fps and scores smoothed, rank-normalized image
change. Category is explicitly null; no CLIP features or action predictions are
fabricated. Output is `activa-m2-diagnostic-v1`, not organizer JSON.

## Run CLIP perception

```sh
/workspace/activa-venv/bin/python scripts/prepare_weights.py
ACTIVA_PYTHON=/workspace/activa-venv/bin/python ./perceive.sh input.mp4 outputs/perception.json --categories /absolute/path/categories.json --weights weights/ViT-B-32.safetensors --importance novelty --cache-dir dev_cache
```

Category input is a JSON array of unique exact label strings, or an object with
that array under `categories`. Supply official labels when available. Test labels
are only for smoke checks; they do not establish Samsung label compatibility.

CLIP uses the library's resize/center-crop/normalization, real decoded timestamps,
2 fps and L2-normalized float32 features. Rotation uses ffprobe's display-matrix
angle; arbitrary rotations fail clearly. Output includes the model identifier,
importance timestamps/values and cosine category scores. These scores are not
calibrated probabilities. Mean pooling is the default; `--pooling weighted` and
`--pooling top` are available for later held-out comparison. Importance choices:
constant, novelty in a +/-10-second window, decoded image motion, and text-guided.
No baseline has been established as best on human labels.

Inference loads local weights only. Downloading is confined to the explicit setup
script and verifies OpenAI's published SHA-256. No dev cache is ever read by the
inference path. Optional cache exports for M3 contain `times`, `feats`,
`model_name` in `clip/*.npz`, and `times`, `values` in `importance/*.npz`.

Optional environment variables: ACTIVA_DEVICE (cpu default), ACTIVA_BATCH_SIZE
(16), ACTIVA_CLIP_MODEL (ViT-B-32), ACTIVA_CLIP_WEIGHTS, ACTIVA_CATEGORIES,
ACTIVA_IMPORTANCE and ACTIVA_POOLING. The public functions retain guide A5
signatures. `code/tcn.py` provides the untrained TCN architecture and a loss for
future experiments; no training or model selection has occurred.

## Verify

```sh
/workspace/activa-venv/bin/python -m pytest -q
```

See docs/results.md for executed checks and docs/leakage.md for the outstanding
audit. scripts/audit_ids.py accepts our own CSV inventory format `video_id,sha256`;
it is not an assumed organizer manifest parser. Exact matches are excluded and
all unmatched clips remain unresolved until content review. It never declares
training data safe based only on missing filename/hash matches.

## Remaining external work

Obtain the official kit/vocabulary and evaluation manifest, review content overlap,
and verify annotation formats. Run real pretrained CLIP video checks once weights
are available. Compare baselines and any TCN using reviewed-safe video-level folds.
Record actual F1/accuracy, runtime, and human review; no benchmark numbers exist yet.
