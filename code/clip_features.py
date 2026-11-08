"""Offline 2 fps CLIP extraction with library preprocessing."""
import numpy as np
from PIL import Image
import torch
from contracts import CLIP_FPS, ClipFeats
from config import settings
from clip_backend import load_backend, normalize
from video_io import iter_frames

def extract_clip_features(info):
    cfg = settings()
    model, preprocess, _ = load_backend(cfg.model, cfg.weights, cfg.device)
    times, batches, images = [], [], []
    def flush():
        with torch.inference_mode():
            vectors = model.encode_image(torch.stack(images).to(cfg.device)).float().cpu().numpy()
        batches.append(normalize(vectors))
        images.clear()
    for timestamp, rgb in iter_frames(info.path, CLIP_FPS):
        times.append(timestamp)
        images.append(preprocess(Image.fromarray(rgb)))
        if len(images) >= cfg.batch_size:
            flush()
    if images:
        flush()
    if not batches:
        raise ValueError("Video produced no CLIP features.")
    return ClipFeats(np.asarray(times, dtype=np.float64), np.concatenate(batches), f"{cfg.model}/local")
