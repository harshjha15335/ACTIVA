"""Turn a video and its ACTIVA output JSON into an annotated highlight reel (MP4). Owned by M4.

    python scripts/make_demo_reel.py VIDEO OUTPUT_JSON REEL_MP4 [--actions 3] [--title-sec 4]

The reel is a title card (video ID, label, counts, and a timeline of every action and
highlight), followed by each highlight in order with:
  - "Highlight i/N" and its source times at the top;
  - the highest-scoring predicted actions that overlap it, while they happen (--actions 0 hides them);
  - a bar along the bottom showing where in the full video the clip comes from.

For the demo video only: it reads a finished output file and never runs in the inference path.
The reel has no audio, so narration can be laid over it. Needs ffmpeg with libx264 and libass.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

FPS = 30
INK, PAPER, SOFT, ACCENT = "12151C", "F4F1EA", "B9C0CC", "F26B3A"   # the deck's palette (RGB)


def log(message):
    print(message, file=sys.stderr, flush=True)


def ass_colour(rgb, alpha=0):
    """RGB hex -> ASS &HAABBGGRR."""
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def ass_time(seconds):
    cs = max(0, round(seconds * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def plan(pred, title_sec, fps=FPS):
    """Where each highlight lands in the reel: [(src_start, src_end, reel_start, frames)].

    Every clip is cut to a whole number of frames, so overlay times match the video exactly.
    """
    out, t = [], round(title_sec * fps) / fps
    for h in pred["highlights"]:
        start, end = h["segment"]
        frames = max(1, round((end - start) * fps))
        out.append((start, end, t, frames))
        t += frames / fps
    return out


def overlay_actions(pred, start, end, k):
    """The k highest-scoring actions overlapping [start, end], clipped to it, in time order."""
    hits = [a for a in pred["actions"] if a["segment"][0] < end and a["segment"][1] > start]
    hits = sorted(hits, key=lambda a: -a["score"])[:k]
    return sorted(((max(a["segment"][0], start), min(a["segment"][1], end), a) for a in hits),
                  key=lambda x: x[0])


def rect(x, y, w, h):
    w, h = max(1, round(w)), max(1, round(h))
    x, y = round(x), round(y)
    return f"m {x} {y} l {x + w} {y} {x + w} {y + h} {x} {y + h}"


def build_ass(pred, width, height, title_sec, k_actions, fps=FPS, title_actions=60):
    s = height / 720                      # styles are designed at 1280x720
    duration = pred["duration_sec"]
    clips = plan(pred, title_sec, fps)
    total = clips[-1][2] + clips[-1][3] / fps if clips else title_sec
    covered = sum(e - b for b, e, _, _ in clips)

    def style(name, size, colour, bold=0, border_style=1, outline=0, back=INK, back_alpha=0):
        return (f"Style: {name},Arial,{round(size * s)},{ass_colour(colour)},{ass_colour(colour)},"
                f"{ass_colour(INK)},{ass_colour(back, back_alpha)},{bold},0,0,0,100,100,0,0,"
                f"{border_style},{outline},0,7,0,0,0,1")

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        style("Tag", 24, ACCENT, bold=1),
        style("TagBox", 24, ACCENT, bold=1, border_style=3, outline=round(8 * s), back_alpha=0x30),
        style("Title", 84, PAPER, bold=1),
        style("Body", 30, SOFT),
        style("Draw", 10, SOFT),
        style("Action", 28, PAPER, bold=1, border_style=3, outline=round(10 * s), back_alpha=0x30),
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def event(start, end, style_name, text, layer=0):
        lines.append(f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style_name},,0,0,0,,{text}")

    def draw(start, end, colour, shapes, layer=0, alpha=0):
        if shapes:
            event(start, end, "Draw", "{\\an7\\pos(0,0)\\bord0\\shad0\\1c" + ass_colour(colour)
                  + "\\1a&H" + f"{alpha:02X}" + "&\\p1}" + " ".join(shapes) + "{\\p0}", layer)

    left, span = 80 * s, width - 160 * s
    x_of = lambda t: left + span * min(max(t / duration, 0), 1)

    # Title card
    T = title_sec
    label = pred["label"]
    event(0, T, "Tag", f"{{\\an7\\pos({left:.0f},{90 * s:.0f})}}ACTIVA  ·  {pred['video_id']}")
    event(0, T, "Title", f"{{\\an7\\pos({left:.0f},{130 * s:.0f})}}{label['name']}")
    event(0, T, "Body", f"{{\\an7\\pos({left:.0f},{240 * s:.0f})}}label score {label['score']:.2f}  ·  "
          f"{clock(duration)} video  ·  {len(pred['actions'])} actions  ·  {len(clips)} highlights "
          f"({100 * covered / duration:.1f}% of the video)")
    y_act, y_hl, bar_w = 420 * s, 540 * s, 2 * s
    top = sorted(pred["actions"], key=lambda a: -a["score"])[:title_actions]
    event(0, T, "Tag", f"{{\\an1\\pos({left:.0f},{y_act - 12 * s:.0f})}}TOP {len(top)} PREDICTED ACTIONS")
    draw(0, T, "2E3442", [rect(left, y_act, span, 70 * s)])
    draw(0, T, SOFT, [rect(x_of(a["segment"][0]), y_act, max(bar_w, x_of(a["segment"][1]) - x_of(a["segment"][0])),
                           70 * s) for a in top], layer=1, alpha=0x60)
    event(0, T, "Tag", f"{{\\an1\\pos({left:.0f},{y_hl - 12 * s:.0f})}}HIGHLIGHTS")
    draw(0, T, "2E3442", [rect(left, y_hl, span, 70 * s)])
    draw(0, T, ACCENT, [rect(x_of(b), y_hl, max(bar_w, x_of(e) - x_of(b)), 70 * s) for b, e, _, _ in clips], layer=1)
    event(0, T, "Body", f"{{\\an7\\pos({left:.0f},{y_hl + 84 * s:.0f})}}0:00")
    event(0, T, "Body", f"{{\\an9\\pos({left + span:.0f},{y_hl + 84 * s:.0f})}}{clock(duration)}")

    # One overlay per highlight
    y_bar, h_bar = height - 34 * s, 10 * s
    for i, (b, e, r0, frames) in enumerate(clips, 1):
        r1 = r0 + frames / fps
        event(r0, r1, "TagBox", f"{{\\an7\\pos({40 * s:.0f},{32 * s:.0f})}}"
              f"HIGHLIGHT {i}/{len(clips)}   {clock(b)}–{clock(e)}", layer=2)
        draw(r0, r1, INK, [rect(left, y_bar, span, h_bar)], alpha=0x40)
        draw(r0, r1, ACCENT, [rect(x_of(hb), y_bar, max(bar_w, x_of(he) - x_of(hb)), h_bar)
                              for hb, he, _, _ in clips], layer=1, alpha=0x90)
        draw(r0, r1, PAPER, [rect(x_of(b), y_bar - 4 * s, max(bar_w, x_of(e) - x_of(b)), h_bar + 8 * s)], layer=2)
        for a0, a1, a in overlay_actions(pred, b, e, k_actions):
            event(r0 + (a0 - b), r0 + (a1 - b), "Action",
                  f"{{\\an1\\pos({40 * s:.0f},{height - 70 * s:.0f})}}{a['verb']} {a['noun']}  {a['score']:.2f}", layer=3)
    return "\n".join(lines) + "\n", total


def run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{proc.stderr[-2000:]}")


def make_reel(video, pred, out, width=1280, height=720, title_sec=4.0, k_actions=3, crf=20):
    if not pred.get("highlights"):
        raise SystemExit("the output file has no highlights")
    ass, total = build_ass(pred, width, height, title_sec, k_actions)
    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x{INK},setsar=1,fps={FPS},format=yuv420p")
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-an"]
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parts = [tmp / "title.mp4"]
        title_frames = round(title_sec * FPS)
        run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{INK}:s={width}x{height}:r={FPS}",
             "-frames:v", str(title_frames), "-vf", "format=yuv420p", *enc, str(parts[0])])
        for i, (b, e, _, frames) in enumerate(plan(pred, title_sec), 1):
            part = tmp / f"clip{i:03d}.mp4"
            fade = min(0.25, frames / FPS / 4)
            run(["ffmpeg", "-v", "error", "-y", "-ss", f"{b:.3f}", "-i", str(Path(video).resolve()),
                 "-vf", f"{fit},trim=end_frame={frames},fade=t=in:d={fade:.3f},"
                        f"fade=t=out:st={frames / FPS - fade:.3f}:d={fade:.3f}",
                 "-frames:v", str(frames), *enc, str(part)])
            parts.append(part)
            log(f"clip {i}: {b:.2f}-{e:.2f} s")
        (tmp / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
        (tmp / "reel.ass").write_text(ass, encoding="utf-8")
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
             "-vf", "subtitles=reel.ass", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], cwd=tmp)
    log(f"wrote {out} ({total:.2f} s)")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("prediction", help="the output JSON ACTIVA wrote for this video")
    ap.add_argument("out", help="where to write the reel (.mp4)")
    ap.add_argument("--actions", type=int, default=3, help="actions shown per highlight (0 hides them)")
    ap.add_argument("--title-sec", type=float, default=4.0)
    ap.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT")
    args = ap.parse_args(argv)
    width, height = (int(x) for x in args.size.lower().split("x"))
    pred = json.loads(Path(args.prediction).read_text(encoding="utf-8"))
    make_reel(args.video, pred, args.out, width, height, args.title_sec, args.actions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
