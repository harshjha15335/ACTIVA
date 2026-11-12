"""Zero-shot CLIP category classification with configurable exact labels."""
import numpy as np
from contracts import CategoryResult
from perception_options import settings
from importance import validate_clip
from clip_backend import normalize, text_features, model_identifier
from category_vocab import load_categories
PROMPTS = ("a first-person video of {}", "egocentric footage of someone {}", "a GoPro view of {}", "a photo taken from a head-mounted camera while {}", "a person is {}")

def pool_features(feats, values, method):
    values = np.asarray(values)
    if values.shape != (len(feats),) or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Importance must match features and be finite in [0, 1].")
    if method == "mean":
        pooled = feats.mean(axis=0)
    elif method == "weighted":
        pooled = np.average(feats, axis=0, weights=0.1 + values)
    elif method == "top":
        order = np.argsort(-values, kind="stable")[:max(1, int(np.ceil(len(feats) * 0.3)))]
        pooled = feats[order].mean(axis=0)
    else:
        raise ValueError(f"Unknown category pooling: {method}")
    if np.linalg.norm(pooled) < 1e-12:
        pooled = feats[0]
    return normalize(pooled)

def softmax(logits, temperature=0.07):
    """Numerically stable temperature-scaled softmax for probability normalization."""
    t = float(temperature) if temperature and float(temperature) > 0 else 0.07
    scaled = np.asarray(logits, dtype=np.float64) / t
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return probs.astype(np.float64)

def classify_video(info, clip, imp):
    cfg = settings()
    vocabulary = load_categories(cfg)
    labels = vocabulary.labels
    if "/sha256:" in clip.model_name and clip.model_name != model_identifier(cfg.model, cfg.weights):
        raise ValueError("Image features and configured text checkpoint identities differ.")
    times, feats = validate_clip(info, clip)
    if not np.array_equal(times, imp.times):
        raise ValueError("Importance timestamps must exactly match CLIP grid.")
    video = pool_features(feats, imp.values, cfg.pooling)
    vectors = text_features([template.format(label) for label in labels for template in PROMPTS], cfg)
    prototypes = normalize(vectors.reshape(len(labels), len(PROMPTS), -1).mean(axis=1))
    cos_sims = prototypes @ video
    probabilities = softmax(cos_sims, temperature=0.07)
    best_idx = int(np.argmax(cos_sims))
    return CategoryResult(labels[best_idx], {label: float(prob) for label, prob in zip(labels, probabilities)})
