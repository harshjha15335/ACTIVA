"""ffprobe metadata and PyAV decoding with real presentation timestamps."""
import json
import math
from pathlib import Path
import subprocess
import av
import numpy as np
from contracts import VideoInfo

def inspect_video(path):
    path = Path(path).resolve(strict=True)
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    streams = [s for s in data["streams"] if s["codec_type"] == "video"]
    if not streams:
        raise ValueError("Input contains no video stream.")
    stream = streams[0]
    duration = float(stream.get("duration", data.get("format", {}).get("duration", 0)))
    num, den = map(int, stream.get("avg_frame_rate", "0/1").split("/"))
    fps = num / den if den else 0
    if not math.isfinite(duration) or duration <= 0 or fps <= 0:
        raise ValueError("Video requires positive duration and frame rate.")
    rotation = int(stream.get("tags", {}).get("rotate", 0))
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    if rotation % 90:
        raise ValueError("Only multiples of 90 degree rotation are supported.")
    return VideoInfo(path.stem, str(path), duration, fps, int(stream["width"]), int(stream["height"]), rotation)

def iter_frames(path, target_fps):
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite.")
    info = inspect_video(path)
    next_time, previous = 0.0, -1.0
    with av.open(info.path) as container:
        stream = container.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Decoded frame lacks a presentation timestamp.")
            timestamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = timestamp
            timestamp -= origin
            if timestamp < next_time - 1e-8 or timestamp <= previous:
                continue
            if timestamp >= info.duration:
                break
            rgb = frame.to_ndarray(format="rgb24")
            if info.rotation:
                rgb = np.rot90(rgb, k=info.rotation // 90).copy()
            yield timestamp, rgb
            previous = timestamp
            next_time = (math.floor(timestamp * target_fps + 1e-8) + 1) / target_fps
    if previous < 0:
        raise ValueError("No timestamped frames could be decoded.")
