# ACTIVA

**Action Intelligence and Video Analysis**

**Team LensLog** (VIT Vellore) · Samsung PRISM XR Hackathon 2026 · Track 04, Action-Aware Video Summarization

**[Demo video (2:49)](https://youtu.be/X3hYju0MdcA)** · **[Presentation (PDF)](docs/VITVellore_LensLog.pdf)** · [AI disclosure](AI_DISCLOSURE.md) · [Results](#results)

ACTIVA turns a first-person (egocentric) MP4 into one JSON in the student kit's format:
- the video's **label**, e.g. `cooking` or `ski`;
- every **action** it finds, as verb + noun + time segment + confidence;
- **highlight** segments of at least 4 s, together at most 15% of the video.

The highlight selector picks moments that cover many different actions and that look unusual compared with the rest of the video. That novelty score is training-free: no human importance labels were used, so how well it matches what people find important is unmeasured.

```sh
./run.sh input.mp4 output.json
```

This folder is the submission, in the student kit's `TEAM_ID/` layout (`run.sh`, `code/`, `weights/`, `requirements.txt`, `outputs/`, `README.md`), named `LensLog/` as the organisers asked. All paths below are relative to it.

## Quick start

Requirements:
- Python 3.12+
- FFmpeg 4.4+ (`ffmpeg` and `ffprobe` on PATH). Tested with 4.4.2 (Ubuntu 22.04 package) and 9.0.2: the action stage gets byte-identical decoded frames on both (#42). Our final run's exact version is recorded in `outputs/README.md`.
- about 4 GB of disk: the clone is ~1.8 GB (the model weights are committed), plus ~1.8 GB once they are joined
- recommended: an NVIDIA GPU with 4 GB+ VRAM (see [Hardware and runtime](#hardware-and-runtime))

```sh
git clone https://github.com/harshjha15335/ACTIVA.git && cd ACTIVA/LensLog   # every command runs from here
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# NVIDIA GPU (CUDA 13.x driver): swap in the CUDA build of the same torch version
pip install --no-deps --force-reinstall torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132
python scripts/setup_weights.py                      # one-time, no network: joins weights/parts/, checks every SHA-256
./run.sh input.mp4 outputs/input.json
```

- **The kit's `manifest.csv` and `supported_classes.json` are in `kit/`**, as the organisers asked (#46); the label/verb/noun list comes from `supported_classes.json`. If that file is removed, `run.sh` rebuilds the identical list (EPIC-100 class keys from the installed model files, FPVSum categories + `cooking` + `other`) and logs a warning. A test checks the two match exactly (D20).

- **The model weights are in the repo**, as the organisers advised (#46). GitHub refuses files over 100 MiB, so each file is cut into 95 MiB pieces in `weights/parts/` (`<file>.000`, `.001`, ...).
  - `setup_weights.py` joins them, with no network access, and checks every SHA-256. `--offline` forbids any download.
  - If a piece is missing or damaged, it downloads that file from this repo's [`weights-v1` release](https://github.com/harshjha15335/ACTIVA/releases/tag/weights-v1) instead (same bytes).
  - The files are unchanged copies of the upstream ones (see [Credits](#credits-and-licences)).
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

### Docker

The `Dockerfile` builds the same pipeline on Debian 12 (Python 3.13, FFmpeg 5.1), CPU by default. The build joins the committed weight pieces and SHA-256-checks them (`setup_weights.py --offline`), so the container runs with no network:

```sh
docker build -t activa .
docker run --rm --network none -v "$PWD/videos:/in:ro" -v "$PWD/out:/out"     activa /in/<video_id>.mp4 /out/<video_id>.json
```

- **GPU (optional):** needs the NVIDIA Container Toolkit. Build with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu132`, then add `--gpus all` to `docker run`.
- **Strict mode:** add `-e ACTIVA_STRICT=1` to `docker run` to get exit 6 instead of a fallback.
- **Tested 19 Nov** (CPU image, `--network none`, strict mode):
  - On a 25 s EPIC video: exit 0, valid JSON, and the same label and highlights as `outputs/`. 1,721 of 2,000 actions are identical; the rest differ by floating-point rounding (see [Reproducibility](#reproducibility)).
  - On a sports video: byte-identical to `outputs/`.

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
| Actions: EPIC-100 SlowFast features → ActionFormer verb and noun detection → pairs → NMS, up to 2000 scored actions. Skipped (`actions: []`) when the label is one of the 14 FPVSum sports, since the model only knows kitchen verbs and nouns (D21) | `code/actions.py`, `code/slowfast_features.py`, `code/action_config.py` | M1 |
| Perception: CLIP features at 2 fps, frame importance, zero-shot label | `code/clip_features.py`, `code/importance.py`, `code/category.py` | M2 |
| Highlights: bounded greedy packing that rewards distinct action coverage and importance, within the 15% budget | `code/optimizer.py`, `code/candidates.py` | M3 |
| Shared types and settings | `code/contracts.py`, `code/config.py`, `code/vocab.py` | M4 |

- Labels, verbs and nouns come from the kit's `kit/supported_classes.json` when it is present (`ACTIVA_VOCAB` can point to it elsewhere).
  - Without it, the same list is rebuilt from public sources: the EPIC-100 class keys in the installed model files, and the 14 FPVSum category names (`config.FPVSUM_CATEGORIES`) + `cooking` + `other`. A test checks the result equals the kit file exactly (D20).
- Inference never reads video IDs, annotations or development caches.
- Design decisions: [`docs/decisions.md`](docs/decisions.md).

## Hardware and runtime

Measured on one laptop: RTX 4060 Laptop (8 GB), i7-14650HX, 16 GB RAM, strict mode, all 187 manifest videos (56,094 s of video).

| | |
|---|---|
| Final run, 21 Nov, frozen code | 6.9 h, **0.44 s per second of video** |
| Earlier full run, 17 Nov | 7.2 h, 0.46 s per second of video (about 2 h of it lost to thermal throttling, see below) |
| A 9 to 10 min EPIC video (17 Nov) | 194 to 247 s |
| A 2 min FPVSum video (17 Nov) | 4 to 14 s (the kitchen action model is skipped on sports labels, D21) |
| Peak GPU memory (17 Nov) | 3.2 GB |

- The run is limited by video decoding on the CPU, so the GPU is mostly idle.
- **Keep the machine cool.** During the 17 Nov run the laptop's CPU reached 95 °C and throughput halved for about 2.5 h, until the fans were set to maximum. Outside that window the EPIC videos ran at 0.42 s per second of video. The outputs are unaffected; only the time changes.
- **Without a GPU, ACTIVA still runs**, about 9× slower. Measured 17 Nov on the same laptop with `ACTIVA_DEVICE=cpu` (i7-14650HX, 16 cores), on P26_41 (116.75 s of video):

  | Stage | CPU | RTX 4060 |
  |---|---:|---:|
  | CLIP (label + importance) | 84.8 s | 16.7 s |
  | Actions (SlowFast + ActionFormer) | 343.0 s | 29.3 s |
  | Total | 428.9 s (3.7 s per video-second) | 46.2 s (0.40 s per video-second) |

  - So a 33-minute video takes about 2 h on this CPU, and the whole manifest about 57 h.
  - Same label and highlights as the GPU run. Of the GPU run's 1,927 actions (1,928 on CPU), 1,562 are identical and 1,696 have the same segment, verb and noun; the rest differ slightly in score or boundaries from floating-point rounding.
  - The CLIP part alone (label and importance) runs at about 0.7 s per video-second on this CPU.

## Results

These are our own measurements, each run once. None of them is an official score, and nothing was changed because of them.

| Check | Result | Notes |
|---|---|---|
| Valid output files, final run | **187 / 187** | Strict mode, 0 fallbacks, 0 errors ([`outputs/README.md`](outputs/README.md)) |
| Action mAP, EPIC-100 validation (138 videos, mean over tIoU 0.1 to 0.5) | **verb 18.11, noun 14.73, action 13.74** | Features from the raw MP4, verbs and nouns paired by us ([#26](https://github.com/harshjha15335/ACTIVA/issues/26)). The published model on its published features, which we reproduced, gets verb 23.49 (published 23.51) and noun 21.88 (published 21.88) ([`docs/m1_status.md`](docs/m1_status.md)) |
| Distinct actions covered by the highlights (138 EPIC videos) | **32.56%** at half overlap, against 19.91% for evenly spaced clips | Any overlap 42.52% vs 33.70%, full action 21.95% vs 11.13%. A local proxy, not the organisers' score ([`docs/results.md`](docs/results.md)) |
| Label, 31 sports videos outside the manifest | **26 / 31** | A small set, not evaluation videos. Three prompt variants were tested once and none was adopted ([#33](https://github.com/harshjha15335/ACTIVA/issues/33)) |
| Highlight F1, FPVSum | **Not measured** | We have no independent human-labelled videos |

## Outputs and checks

`outputs/` holds our 187 output files for the kit manifest (`kit/manifest.csv`; the videos aren't included). To regenerate them:

```sh
pip install -r requirements-download.txt
python scripts/download_videos.py fpvsum && python scripts/download_videos.py epic   # EPIC is ~184 GB
ACTIVA_STRICT=1 python scripts/run_manifest.py --videos data/videos --out outputs      # resumable
python scripts/validate_outputs.py --dir outputs
```

- `run_manifest.py` writes a per-video report to `tmp/run_report.json`.
- `validate_outputs.py` is our local implementation of the kit README's rules. The organisers don't provide an official `validate.py` and asked teams to write their own (#46).
- **Final run, 21 Nov, on the frozen code (`v1.0-freeze`): 187/187 valid**, no fallbacks, strict mode. How it was made, and the versions used, are in [`outputs/README.md`](outputs/README.md).
- **Two published EPIC files contain damaged H.264 packets: P29_05 and P30_08.** Their MD5s match EPIC's official list, so everyone's copies are the same. ACTIVA skips those packets, as ffmpeg does (D18).

Choices about the kit's rules (`docs/kit_compliance.md`):
- **Team ID:** `team_id` is `LensLog`, as the organisers confirmed.
- **Short videos (still unanswered):** for videos where 15% is under 4 s (3 in the manifest), we output one 4.0 s highlight.

## Reproducibility

What a re-run depends on, so another machine can check our `outputs/`.

**Inputs.** [`docs/evidence/input_sha256_2026-11-17.csv`](docs/evidence/input_sha256_2026-11-17.csv) lists the SHA-256 and size of all 187 manifest videos as we ran them. Compare with `sha256sum <video>` (Linux) or `certutil -hashfile <video> SHA256` (Windows).

**Weights.** `scripts/setup_weights.py` refuses any file whose SHA-256 differs from these:

```text
c2ea63325cebca85a9eb509481f37ea991cb20199cab6642e68dbc8f973382f5  ViT-B-32-laion400m_e32.safetensors
0fc9e83d544544617471835bd345aa71a6b75d979124a492b793b786f0e90cf7  SlowFast.pyth
88486434a71e1b17ac38dd4a93d9612612026be62d7a93a032a0953a30ac687b  epic_slowfast_verb_reproduce/epoch_020.pth.tar
8d313a2dd69562bfb063cb1e3372ad056ee646283d1e32af95ff49318bf6bbf9  epic_slowfast_noun_reproduce/epoch_020.pth.tar
```

The SlowFast and ActionFormer code is in the repo at the pinned commits (see Credits).

**Software used for `outputs/`.** Direct dependencies are pinned in `requirements.txt`; every installed package, including transitive ones, is in `requirements-lock.txt` (`pip freeze` of the run's venv):

| | |
|---|---|
| OS, GPU | Windows 11, RTX 4060 Laptop, NVIDIA driver 610.91 (592.82 for the 17 Nov run) |
| Python | 3.13.5 |
| torch | 2.15.0+cu132 (CUDA 13.2, cuDNN 9.26) |
| torchvision, open_clip_torch | 0.29.2+cpu, 4.2.1 |
| av (PyAV), numpy, Pillow | 19.0.2, 2.3.5, 13.0.0 |
| FFmpeg | 9.0.2 (gyan.dev full build) |

**What stays the same, and what may not.**
- **Same machine, same inputs:** byte-identical JSON. Seeds are fixed (`config.SEED`), and repeated runs and a fresh clone gave identical files.
- **The label and the CLIP importance** use frames decoded by PyAV, which ships its own FFmpeg libraries, so the system FFmpeg doesn't affect them.
- **Actions** decode through the system `ffmpeg`, and the highlight selector uses them too. FFmpeg 4.4.2 (Ubuntu 22.04's package) and 9.0.2 gave byte-identical frames on our test videos (#42), so actions and highlights match. Decoders conceal *damaged* packets differently, so a damaged file may differ slightly between versions.
- **CPU instead of GPU:** same label and highlights. Some action scores and boundaries differ by floating-point rounding (1,562 of 1,927 actions identical on P26_41); see Hardware and runtime.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

- Full suite on a complete checkout: 461 passed, 3 skipped.
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
- Coding agents start from [`AGENTS.md`](AGENTS.md). Which AI tools we used, and how we checked their work, is in [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).

## Credits and licences

ACTIVA's own code is by the team above. It builds on these projects, whose licences still apply to their parts:

| Component | Used for | Licence |
|---|---|---|
| [OpenCLIP](https://github.com/mlfoundations/open_clip), ViT-B-32 QuickGELU LAION-400M e32 weights | Frame features, zero-shot label | MIT |
| [epic-kitchens-slowfast](https://github.com/epic-kitchens/epic-kitchens-slowfast) code and EPIC-100 SlowFast checkpoint | Action features | CC BY-NC-SA 4.0 (non-commercial) |
| [ActionFormer](https://github.com/happyharrycn/actionformer_release) code and EPIC-100 checkpoints | Verb and noun detection | MIT |
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/) class lists | Class ID → name mapping | per the EPIC-KITCHENS terms |
| EPIC-KITCHENS-100, FPVSum | The manifest's videos (not redistributed here) | per their owners |

**Third-party source in this repo.** `weights/m1/actionformer_release/` and `weights/m1/epic-kitchens-slowfast/` are unmodified copies of the upstream code at the commits pinned in `scripts/setup_m1_assets.py`, without their `.git` folders:
- ActionFormer @ `61ea7eb`, MIT.
- epic-kitchens-slowfast @ `bf50519`, CC BY-NC-SA 4.0 (non-commercial).

Each copy keeps its upstream LICENSE file. They are loaded only from disk at inference time; nothing is downloaded.
