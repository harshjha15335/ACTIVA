# Official student kit (local only)

Unzip `Student package_track04 (1).zip` (SHA256
`ddf7ee8d2b8b9fb1363c1e72ae95da9144f64573cbca5a1b7a2378f41bbd0c09`; ask M4 or M2
for it) and copy its files here:

- `supported_classes.json`: read by `code/vocab.py` (override with `ACTIVA_VOCAB`)
- `manifest.csv`: read by `scripts/run_manifest.py` and `scripts/validate_outputs.py`
- `TEAM_README.md`, `P02_109.json`, `skiing_03.json`

Everything here except this README is git-ignored until the organisers confirm
the kit may be published (`docs/kit_checklist.md` #9). Tests that need the real
kit skip when it is absent (CI never has it).

## Downloading the videos

The kit has no videos. `scripts/download_videos.py` fetches the 187 manifest
videos into `data/videos/` (git-ignored) and checks each one: it must open, and
its duration must be within 0.5 s of the manifest.

```bash
pip install -r requirements-download.txt        # yt-dlp, for FPVSum
python scripts/download_videos.py fpvsum         # 49 YouTube videos, a few GB
python scripts/download_videos.py epic           # 138 videos, ~184 GB, resumable
python scripts/download_videos.py check          # report: data/videos/download_report.json
python scripts/run_manifest.py --videos data/videos
```

- FPVSum: names map to YouTube IDs through the upstream FPVSum video list
  (pinned commit). The `_1`, `_3` parts are separate YouTube videos.
- EPIC: uses the official downloader's URL, split and MD5 files (pinned commit),
  but splits each file across many connections. The data.bris server gives
  ~50-70 KB/s per connection, so one connection per file would take days.
  Stop it with Ctrl+C any time; re-running resumes. Each file is MD5-checked.
- Re-running either command skips videos that already pass the check.
- Measured on 2026-11-11 (M4's laptop): ~1.1 MB/s total with 16-32 connections,
  so all of EPIC is ~44 h. To split it, each person runs one share, balanced
  by duration (~61 GB of EPIC each): `python scripts/download_videos.py epic --part 1/3`
  (then `2/3`, `3/3` on the others' machines).
