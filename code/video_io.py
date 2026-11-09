"""Video metadata (ffprobe) and frame decoding (PyAV). Owned by M4.

Adopted from M2's m2-perception version (same behaviour, same timestamps) with
clearer errors, a format-duration fallback and explicit rotation handling.

Time convention, shared by every module:
  t = frame PTS - stream start_time (or the first decoded PTS if start_time is
  missing), in seconds. iter_frames yields the first frame at or after each
  slot 0, 1/f, 2/f, ...; the time it yields is that frame's real PTS, not the
  slot time, so variable-frame-rate video stays correct.

Rotation convention: VideoInfo.rotation is the display-matrix rotation,
normalised to 0/90/180/270 degrees counter-clockwise. iter_frames applies it,
so frames come out the way a player shows them.
"""
import json
import math
from pathlib import Path
import subprocess

import av
import numpy as np

from contracts import VideoInfo


class VideoError(ValueError):
    """The input is missing, corrupt or has no usable video stream."""


def _probe(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found on PATH; install FFmpeg.") from None
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1].replace(f"{path}: ", "") if lines else "no details"
        raise VideoError(f"ffprobe cannot read {path}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise VideoError(f"ffprobe returned unreadable metadata for {path}") from None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _rate(text):
    try:
        num, den = (int(x) for x in str(text).split("/"))
    except ValueError:
        return None
    return num / den if num > 0 and den > 0 else None


def _rotation(stream):
    rotation = None
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = float(side["rotation"])      # display matrix: counter-clockwise
    if rotation is None and "rotate" in stream.get("tags", {}):
        rotation = -float(stream["tags"]["rotate"])  # legacy tag: clockwise
    if rotation is None:
        return 0
    if rotation % 90:
        raise VideoError(f"Unsupported rotation {rotation} degrees (only multiples of 90).")
    return int(rotation) % 360


def inspect_video(path) -> VideoInfo:
    try:
        path = Path(path).resolve(strict=True)
    except FileNotFoundError:
        raise VideoError(f"Input not found: {path}") from None
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoError(f"Input is not a non-empty file: {path}")
    data = _probe(path)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not streams:
        raise VideoError(f"No video stream in {path}")
    stream = streams[0]
    duration = _number(stream.get("duration")) or _number(data.get("format", {}).get("duration"))
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    if duration is None or fps is None:
        raise VideoError(f"{path} needs a positive duration and frame rate.")
    try:
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, TypeError, ValueError):
        raise VideoError(f"{path} has no frame size.") from None
    return VideoInfo(path.stem, str(path), duration, fps, width, height, _rotation(stream))


def iter_frames(path, target_fps):
    """Yield (t_seconds, rgb_uint8_HxWx3) at most target_fps times per second."""
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite.")
    info = inspect_video(path)
    next_time, previous = 0.0, -1.0
    try:
        with av.open(info.path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise VideoError("Decoded frame lacks a presentation timestamp.")
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
                    rgb = np.ascontiguousarray(np.rot90(rgb, k=info.rotation // 90))
                yield timestamp, rgb
                previous = timestamp
                next_time = (math.floor(timestamp * target_fps + 1e-8) + 1) / target_fps
    except av.error.FFmpegError as error:
        raise VideoError(f"Decoding failed for {info.path}: {error}") from None
    if previous < 0:
        raise VideoError(f"No timestamped frames could be decoded from {info.path}")
