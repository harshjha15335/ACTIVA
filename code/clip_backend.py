"""Cached local-only CLIP inference backend."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import open_clip

@lru_cache(maxsize=2)
def load_backend(model_name, weights, device):
    if model_name.startswith(("hf-hub:", "local-dir:")):
        raise ValueError("Use a registered CLIP architecture, not a remote model identifier.")
    configuration = open_clip.get_model_config(model_name)
    if configuration is None:
        raise ValueError(f"Unknown CLIP architecture: {model_name}")
    text_config = configuration.get("text_cfg", {})
    if text_config.get("hf_tokenizer_name") or text_config.get("hf_model_name"):
        raise ValueError("This offline backend supports the bundled CLIP tokenizer only.")
    checkpoint = Path(weights).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local CLIP weights missing: {checkpoint}; run scripts/prepare_weights.py.")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=str(checkpoint), device=device, force_quick_gelu=(model_name == "ViT-B-32"))
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(model_name)

def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norms <= 1e-12):
        raise ValueError("CLIP embeddings must be finite and nonzero.")
    return (values / norms).astype(np.float32)

def text_features(prompts, cfg):
    model, _, tokenizer = load_backend(cfg.model, cfg.weights, cfg.device)
    with torch.inference_mode():
        vectors = model.encode_text(tokenizer(prompts).to(cfg.device)).float().cpu().numpy()
    return normalize(vectors)
