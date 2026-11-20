# Demo video: script and how to build it

Owner: M4. Target: about 3 minutes, 1080p or 720p, with narration recorded separately.
Our copy of the kit doesn't give a length or format for the demo; check with the organisers'
latest message before the final cut.

## What goes in it

| # | Time | On screen | Narration (draft) |
|---|---|---|---|
| 1 | 0:00–0:15 | Reel of P26_41: title card, then the first two highlights | “This is ACTIVA. Give it a first-person video, and it tells you what kind of video it is, every action in it, and the few moments worth watching.” |
| 2 | 0:15–0:35 | Deck slides 2 and 3 (task, scoring) | “Head-mounted cameras record hours nobody rewatches. The task: one MP4 in, one JSON out: a label, verb-noun actions with times, and highlights of at least 4 seconds that add up to at most 15% of the video.” |
| 3 | 0:35–1:05 | Terminal: `./run.sh` on P26_41 (sped up), then the JSON | “One command, fully offline. On a laptop GPU this 2-minute video takes under a minute. The output follows the kit’s format exactly.” |
| 4 | 1:05–1:40 | Deck slide 4 (pipeline), then slides 5–7 briefly | “Two tracks read the video. CLIP at 2 fps scores how novel each moment is and picks the label zero-shot. SlowFast and ActionFormer, trained on EPIC-KITCHENS, find verbs and nouns, which we pair into actions. The selector then packs highlights that cover as many different actions as the budget allows.” |
| 5 | 1:40–2:20 | Full P26_41 reel with action overlays, then the ski reel without them | “Here are the highlights it chose, with the actions it detected as they happen. The bar at the bottom shows where each clip sits in the whole video. The same pipeline labels a ski run as ski.” |
| 6 | 2:20–2:45 | Deck slides 8 and 9 (engineering, results) | “Every one of the 187 manifest videos gets a valid file: about 6 hours for 15.6 hours of video on one laptop, never more than 3.2 GB of GPU memory, including two published videos with damaged packets.” |
| 7 | 2:45–3:00 | Deck slides 11 and 12 (team, thanks) | “ACTIVA is by Team LensLog from VIT Vellore. Code, weights and docs are on GitHub.” |

Only say numbers that are in `README.md` or measured for this video. Measured on 16 Nov:
`ACTIVA_STRICT=1 ./run.sh` on P26_41 (116.75 s of video) took 49.9 s wall-clock on the RTX 4060
laptop (46.2 s inside the pipeline, the rest is start-up), and its JSON was identical to the
full-run output. One run only, with the test suite running at the same time.

## How to build the pieces

1. **Final outputs.** Use the outputs from the frozen `main` (the 21 Nov run), not earlier ones,
   so the reels match what we submit.
2. **Live run (scene 3).** Record the terminal with OBS or the Windows Game Bar:
   ```sh
   ./run.sh data/videos/EPIC-KITCHENS/P26/videos/P26_41.MP4 outputs/P26_41.json
   python -m json.tool outputs/P26_41.json | head -40
   ```
3. **Reels (scenes 1 and 5).** No audio, 720p by default (`--size 1920x1080` for 1080p):
   ```sh
   python scripts/make_demo_reel.py data/videos/EPIC-KITCHENS/P26/videos/P26_41.MP4 outputs/P26_41.json demo/P26_41_reel.mp4
   python scripts/make_demo_reel.py data/videos/fpvsum/Ski_-Z5a4UQmZLs.mp4 outputs/Ski_-Z5a4UQmZLs.json demo/ski_reel.mp4 --actions 0
   ```
   - `--actions 0` hides action labels: the action model knows EPIC kitchen verbs and nouns,
     which mean nothing on sports footage.
   - Each reel takes a few seconds to render.
4. **Deck slides.** Export the deck as PDF or images and use them as stills.
5. **Edit.** Put the pieces in order in any video editor, lay the narration over them and export an MP4.
   Keep the final file under the size the submission allows.

## Choosing other videos

Any manifest video works if its video file and output JSON are both present. Good demo
videos have several highlights spread out and a clear label. Avoid P29_05 and P30_08,
which have damaged packets.
