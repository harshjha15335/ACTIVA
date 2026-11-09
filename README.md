# ACTIVA

Samsung PRISM XR Hackathon 2026, Track 04.

ACTIVA takes a 2-10 minute first-person MP4 and writes one JSON with the
video's category, every action (verb + noun + start + end), and highlight clips
totalling at most 15% of the video. The highlight selector favours covering
many different actions plus the moments people find important.

**Status: integration skeleton (gate G1 pending the official kit).** `run.sh`
works end to end, but every model is still a placeholder stub: no actions are
detected, highlights are evenly spaced, and the label is the first category.
No benchmark numbers have been measured.

## Run

Requirements: Python 3.11+, FFmpeg (`ffmpeg` and `ffprobe` on PATH).

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt        # Windows: .venv\Scripts\pip
export ACTIVA_VOCAB=/path/to/supported_classes.json  # or put it in kit/
ACTIVA_PYTHON=.venv/bin/python ./run.sh input.mp4 outputs/input.json
```

Logs go to stderr; stdout stays empty. Exit codes: 0 ok, 2 usage, 3 input not
found, 4 invalid video, 5 configuration (e.g. no vocabulary), 1 internal error.
On Windows, run `run.sh` from Git Bash.

## Test

```sh
python -m pytest -q
```

The tests generate synthetic videos with FFmpeg (2 and 10 minute, black, VFR,
rotated, corrupt, empty, audio-only) and use a placeholder vocabulary in
`tests/fixtures/`. CI runs the same suite on every push and pull request.

## Layout

| Path | What | Owner |
|------|------|-------|
| `run.sh`, `code/main.py` | Entry point and stage orchestration with fallbacks | M4 |
| `code/contracts.py` | Shared types every module uses | M4 |
| `code/config.py`, `code/vocab.py` | Settings, official vocabulary loader | M4 |
| `code/video_io.py` | ffprobe metadata, PyAV frames with real timestamps | M4 |
| `code/schema.py`, `code/fallbacks.py` | Output JSON build/validate, safe fallbacks | M4 |
| `code/actions.py` | Action detection (stub until M1 lands) | M1 |
| `code/clip_features.py`, `importance.py`, `category.py` | Perception (stubs until M2 lands) | M2 |
| `code/optimizer.py` | Highlight selection (stub until M3 lands) | M3 |
| `kit/` | Official student kit (git-ignored) | M4 |
| `docs/` | Decisions, kit checklist, integration plan, agent rules | all |

## Team workflow

Each member works on their role branch (`m1-actions`, `m2-perception`,
`m3-selector`, `m4-integration`) and opens a pull request into `main`.
Read `AGENTS.md` and `docs/agent_board_protocol.md` first; landing an existing
branch on `main` is described in `docs/integration_plan.md`.

## Credits

ActionFormer, OpenCLIP, EPIC-KITCHENS-100 and FPVSum; licences are listed with
each module as it lands.
