# ACTIVA: Team LensLog

**Action Intelligence and Video Analysis**

Samsung PRISM XR Hackathon 2026 · Track 04, Action-Aware Video Summarization · VIT Vellore

**[Demo video (2:49)](https://youtu.be/X3hYju0MdcA)** · **[Presentation (PDF)](LensLog/docs/VITVellore_LensLog.pdf)** · [AI disclosure](LensLog/AI_DISCLOSURE.md)

The submission is the **[`LensLog/`](LensLog/)** folder, in the student kit's `TEAM_ID/` layout
(the organisers asked for `LensLog/`, 21 Nov):

```text
LensLog/
├── run.sh             # ./run.sh <video.mp4> <out.json>
├── code/              # inference source
├── weights/           # model weights (as <100 MiB pieces in weights/parts/)
├── requirements.txt
├── outputs/           # 187 <video_id>.json files, one per manifest video
└── README.md          # setup, architecture, results, reproduction
```

Start with [`LensLog/README.md`](LensLog/README.md). Every command there runs from inside `LensLog/`:

```sh
git clone https://github.com/harshjha15335/ACTIVA.git && cd ACTIVA/LensLog
```
