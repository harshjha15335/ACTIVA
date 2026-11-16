# ACTIVA

**Team LensLog** (VIT Vellore) · Samsung PRISM XR Hackathon 2026 · Track 04

ACTIVA turns a first-person (egocentric) MP4 into one JSON in the student kit's format:
- the video's **label**, e.g. `cooking` or `ski`;
- every **action** it finds, as verb + noun + time segment + confidence;
- **highlight** segments of at least 4 s, together at most 15% of the video.

The highlight selector picks moments that cover many different actions and that people are likely to find important.

```sh
./run.sh input.mp4 output.json
```

## Quick start

Requirements:
- the student kit's `supported_classes.json` (the label/verb/noun list), which you copy in; it isn't in this repo
- Python 3.12+
- FFmpeg 5.1+ (`ffmpeg` and `ffprobe` on PATH)
- about 2 GB of disk for the model files
- recommended: an NVIDIA GPU with 4 GB+ VRAM (see [Hardware and runtime](#hardware-and-runtime))

```sh
git clone https://github.com/harshjha15335/ACTIVA.git && cd ACTIVA
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# NVIDIA GPU (CUDA 13.x driver): swap in the CUDA build of the same torch version
pip install --no-deps --force-reinstall torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132
python scripts/setup_weights.py                      # one-time, ~1.8 GB, every file SHA-256 checked
cp /path/to/student-kit/supported_classes.json kit/  # from the organisers' kit (not redistributed here)
./run.sh input.mp4 outputs/input.json
```

- `setup_weights.py` downloads the model files from this repo's [`weights-v1` release](https://github.com/harshjha15335/ACTIVA/releases/tag/weights-v1).
  - Those are unchanged mirrors of the upstream files (see [Credits](#credits-and-licences)).
  - `--check` verifies an existing install. `--upstream` rebuilds it from the original authors instead.
- **`run.sh` itself never touches the network.**
- It picks the GPU automatically when CUDA is available. To force a device, set `ACTIVA_DEVICE=cpu` or `ACTIVA_DEVICE=cuda`.
- On Windows, run `run.sh` from Git Bash. A `.venv` is not picked up automatically: put it on PATH, or set `ACTIVA_PYTHON`.

Output (`code/schema.py`). The team ID is the same in every file (`code/config.py`, override with `ACTIVA_TEAM_ID`):

```json
{"team_id": "LensLog", "video_id": "P01_11", "duration_sec": 561.53,
 "label": {"name": "cooking", "score": 0.93},
 "actions": [{"segment": [12.4, 18.9], "verb": "open", "noun": "fridge", "score": 0.82}],
 "highlights": [{"segment": [12.0, 23.0]}]}
```

- Logs go to stderr; stdout stays empty.
- Exit codes:
  - 0 ok · 1 internal error · 2 usage · 3 input not found
  - 4 invalid video · 5 configuration (missing model files or vocabulary)
  - 6 a stage failed in strict mode
- **Fallbacks:** if a stage fails, the default is to log it and write a valid file using a safe fallback for that stage, because a missing file scores 0. `ACTIVA_STRICT=1` turns any fallback into exit 6 instead; we use it to verify releases.

## How it works

```
MP4 ─┬─ PyAV, 2 fps ─ CLIP ViT-B-32 ─┬─ importance (novelty) ─────┐
     │                               └─ label (zero-shot, pooled) ─┼─ highlight selector ─ JSON
     └─ ffmpeg, 30 fps ─ SlowFast (EPIC-100) ─ ActionFormer verb + noun ─ pairs + NMS ─┘
```

| Stage | Files | Owner |
|---|---|---|
| Entry point, stage orchestration with fallbacks, output schema | `run.sh`, `code/main.py`, `code/schema.py`, `code/fallbacks.py` | M4 |
| Video decoding (real timestamps, rotation, damaged packets skipped) | `code/video_io.py` | M4 |
| Actions: EPIC-100 SlowFast features → ActionFormer verb and noun detection → pairs → NMS, up to 2000 scored actions | `code/actions.py`, `code/slowfast_features.py`, `code/action_config.py` | M1 |
| Perception: CLIP features at 2 fps, frame importance, zero-shot label | `code/clip_features.py`, `code/importance.py`, `code/category.py` | M2 |
| Highlights: bounded greedy packing that rewards distinct action coverage and importance, within the 15% budget | `code/optimizer.py`, `code/candidates.py` | M3 |
| Shared types and settings | `code/contracts.py`, `code/config.py`, `code/vocab.py` | M4 |

- Labels, verbs and nouns come only from the kit's `kit/supported_classes.json`; nothing is hardcoded.
  - We don't publish kit files until the organisers say we may, so copy that file in from your kit.
  - Without it, `run.sh` exits 5 and says so. `ACTIVA_VOCAB` can point to it elsewhere.
- Inference never reads video IDs, annotations or development caches.
- Design decisions: [`docs/decisions.md`](docs/decisions.md).

## Hardware and runtime

Measured on one laptop: RTX 4060 Laptop (8 GB), i7-14650HX, 16 GB RAM, strict mode, all 187 manifest videos.

| | |
|---|---|
| Whole manifest (56,094 s of video) | ~6 h, about **0.39 s per second of video** |
| A 9.4 min EPIC video | 217 s |
| A 2 min FPVSum video | 32 s |
| Peak GPU memory | 3.2 GB |

- The run is limited by video decoding on the CPU, so the GPU is mostly idle.
- **Without a GPU, ACTIVA still runs**, but the action stage is about 13× slower.

## Outputs and checks

`outputs/` holds our 187 output files for the kit manifest (`kit/manifest.csv`, which stays local; the videos aren't included). To regenerate them:

```sh
pip install -r requirements-download.txt
python scripts/download_videos.py fpvsum && python scripts/download_videos.py epic   # EPIC is ~184 GB
ACTIVA_STRICT=1 python scripts/run_manifest.py --videos data/videos --out outputs      # resumable
python scripts/validate_outputs.py --dir outputs
```

- `run_manifest.py` writes a per-video report to `tmp/run_report.json`.
- `validate_outputs.py` is our local implementation of the kit README's rules. The official `validate.py` is listed in the kit but was missing from our copy, and we've asked the organisers for it.
- Latest full run, 14 Nov: **187/187 valid**, no fallbacks.
- **Two published EPIC files contain damaged H.264 packets: P29_05 and P30_08.** Their MD5s match EPIC's official list, so everyone's copies are the same. ACTIVA skips those packets, as ffmpeg does (D18).

Provisional choices, pending the organisers' answers (`docs/kit_compliance.md`):
- **Team ID:** `team_id` is `LensLog`.
- **Short videos:** for videos where 15% is under 4 s (3 in the manifest), we output one 4.0 s highlight.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

- The tests build synthetic videos with FFmpeg: short, long, black, VFR, rotated, corrupt, damaged-packet, empty and audio-only.
- They use a fixture vocabulary and need no model weights.
- CI runs them on every push and pull request.
- Opt-in tests with the real model weights skip unless their environment variables are set.

## Team

| Role | Member |
|---|---|
| M1 Action engine | Arham Jain ([@Arhamj1](https://github.com/Arhamj1)) |
| M2 Perception | Harsh Jha ([@harshjha15335](https://github.com/harshjha15335)) |
| M3 Highlight selector | Anuj Deshpande ([@ANUJ-DESHPANDE](https://github.com/ANUJ-DESHPANDE)) |
| M4 Integration, demo, submission | Aarush Shetty ([@AarushShetty346](https://github.com/AarushShetty346)) |

**Contributor account clarification:** [@Aqualil234](https://github.com/Aqualil234) is an alternate GitHub account used by Harsh Jha (M2), not a fifth team member. ACTIVA has four participating students, as listed above.

How we work:
- Each member works on their own branch and merges into `main` through PRs that pass CI.
- Coordination happens on the [message board](https://github.com/harshjha15335/ACTIVA/issues/4); rules are in [`docs/agent_board_protocol.md`](docs/agent_board_protocol.md).
- Coding agents start from [`AGENTS.md`](AGENTS.md).

## Credits and licences

ACTIVA's own code is by the team above. It builds on these projects, whose licences still apply to their parts:

| Component | Used for | Licence |
|---|---|---|
| [OpenCLIP](https://github.com/mlfoundations/open_clip), ViT-B-32 QuickGELU LAION-400M e32 weights | Frame features, zero-shot label | MIT |
| [epic-kitchens-slowfast](https://github.com/epic-kitchens/epic-kitchens-slowfast) code and EPIC-100 SlowFast checkpoint | Action features | CC BY-NC-SA 4.0 (non-commercial) |
| [ActionFormer](https://github.com/happyharrycn/actionformer_release) code and EPIC-100 checkpoints | Verb and noun detection | MIT |
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/) class lists | Class ID → name mapping | per the EPIC-KITCHENS terms |
| EPIC-KITCHENS-100, FPVSum | The manifest's videos (not redistributed here) | per their owners |
