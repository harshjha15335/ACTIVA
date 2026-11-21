# Final outputs

187 files, `<video_id>.json`, one per video in `kit/manifest.csv` (138 EPIC-KITCHENS-100, 49 FPVSum).
Each has `team_id` `LensLog` and passes `scripts/validate_outputs.py` (187/187 valid).

## How they were made

| | |
|---|---|
| Code | tag `v1.0-freeze` = `3968ea7`, clean tree |
| Command | `ACTIVA_STRICT=1 python scripts/run_manifest.py --videos data/videos --out outputs` |
| Run | 21 Nov 2026, 02:53 to 09:49 IST, 24,941 s of processing in total |
| Result | 187 ok, 0 fallbacks, 0 errors |
| OS, GPU | Windows 11, RTX 4060 Laptop GPU, NVIDIA driver 610.91 |
| Python, torch | 3.13.5, 2.15.0+cu132 |
| FFmpeg | 9.0.2-full_build-www.gyan.dev |

Other package versions are in `requirements-lock.txt`.

## Notes

- **Short videos (D13):** P03_26, P26_30 and P28_23 are short enough that 15% of the video is under the 4 s minimum
  highlight. They get one 4.0 s highlight, so they exceed the 15% budget. The organisers haven't answered this
  question yet (`docs/kit_compliance.md`).
- **Sports videos** have no actions (`"actions": []`): the action model only knows kitchen actions.
- **Same as the 17 Nov run:** 186 of 187 files are byte-identical. In P30_08, 69 of 2,000 actions differ by 0.01 s
  in a segment end or 0.0001 in a score, which is GPU floating-point rounding. Labels and highlights are identical in
  all 187.
- P29_05 and P30_08 contain damaged H.264 packets in the published EPIC files; ACTIVA skips them, as ffmpeg does
  (D18).
