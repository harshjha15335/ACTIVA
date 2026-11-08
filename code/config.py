"""Runtime configuration: categories are provided explicitly, never guessed."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
@dataclass(frozen=True)
class Settings:
    weights: str
    categories: tuple[str, ...]
    model: str = "ViT-B-32"
    device: str = "cpu"
    batch_size: int = 16
    baseline: str = "novelty"
    pooling: str = "mean"
def settings():
    categories = []
    if os.environ.get("ACTIVA_CATEGORIES"):
        data = json.loads(Path(os.environ["ACTIVA_CATEGORIES"]).read_text())
        categories = data.get("categories", []) if isinstance(data, dict) else data
        if not isinstance(categories, list) or not categories or any(not isinstance(x, str) or not x.strip() for x in categories) or len(set(categories)) != len(categories):
            raise ValueError("Category JSON requires unique nonempty strings.")
    batch = int(os.environ.get("ACTIVA_BATCH_SIZE", "16"))
    if batch < 1:
        raise ValueError("Batch size must be positive.")
    return Settings(os.environ.get("ACTIVA_CLIP_WEIGHTS", "weights/ViT-B-32.safetensors"), tuple(categories), os.environ.get("ACTIVA_CLIP_MODEL", "ViT-B-32"), os.environ.get("ACTIVA_DEVICE", "cpu"), batch, os.environ.get("ACTIVA_IMPORTANCE", "novelty"), os.environ.get("ACTIVA_POOLING", "mean"))
