# ACTIVA

Samsung PRISM XR Hackathon 2026, Track 04.

ACTIVA takes a first-person MP4 and writes one JSON in the student kit's format:
the video's label, every action (verb + noun + segment + score), and highlight
segments of at least 4 s totalling at most 15% of the video. The highlight
selector favours covering many different actions plus the moments people find
important.

**Status: integration skeleton on the official output format.** `run.sh` works
end to end and its output passes our local spec validator, but the models are
not ready: no actions are detected yet, highlights fall back to evenly spaced
clips, and CLIP needs local weights. No benchmark numbers have been measured.
Kit rules and open questions: `docs/kit_compliance.md`.

## Run

Requirements: Python 3.11+, FFmpeg (`ffmpeg` and `ffprobe` on PATH), the student
kit's files in `kit/` (see `kit/README.md`).

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt        # Windows: .venv\Scripts\pip
export ACTIVA_TEAM_ID=<team id>                      # until config.TEAM_ID is set
ACTIVA_PYTHON=.venv/bin/python ./run.sh input.mp4 outputs/input.json
```

Output (`code/schema.py`):

```json
{"team_id": "...", "video_id": "P01_11", "duration_sec": 561.53,
 "label": {"name": "cooking", "score": 0.93},
 "actions": [{"segment": [12.4, 18.9], "verb": "open", "noun": "fridge", "score": 0.82}],
 "highlights": [{"segment": [12.0, 23.0]}]}
```

Logs go to stderr; stdout stays empty. Exit codes: 0 ok, 2 usage, 3 input not
found, 4 invalid video, 5 configuration (no vocabulary or team_id), 6 a stage
failed in strict mode (`ACTIVA_STRICT=1`), 1 internal error.
On Windows, run `run.sh` from Git Bash.

All manifest videos, resumable, with a per-video report in `tmp/run_report.json`:

```sh
python scripts/run_manifest.py --videos /data/videos --out outputs
python scripts/validate_outputs.py --dir outputs      # local spec validator, not the official validate.py
```

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
| `code/schema.py`, `code/fallbacks.py` | Official output JSON build/validate, safe fallbacks | M4 |
| `scripts/run_manifest.py`, `scripts/validate_outputs.py` | Batch run over the manifest, local spec validator | M4 |
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
