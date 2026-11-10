"""Local-only CLIP backend, file-aware identity and scoped initialization RNG."""
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import open_clip


def _signature(path):
    stat=path.stat()
    return (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


@lru_cache(maxsize=16)
def _digest(path, signature):
    with open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def checkpoint_identity(weights):
    path=Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Local CLIP weights missing: {path}; run scripts/prepare_weights.py.')
    return _digest(str(path),_signature(path))


def model_identifier(model_name, weights):
    return f'{model_name}/sha256:{checkpoint_identity(weights)}'


@lru_cache(maxsize=2)
def _load_cached(model_name, weights, device, signature):
    if model_name.startswith(('hf-hub:','local-dir:')):
        raise ValueError('Use a registered local CLIP architecture, not a remote model identifier.')
    configuration=open_clip.get_model_config(model_name)
    if configuration is None:
        raise ValueError(f'Unknown CLIP architecture: {model_name}')
    text=configuration.get('text_cfg',{})
    if text.get('hf_tokenizer_name') or text.get('hf_model_name'):
        raise ValueError('This offline backend supports the bundled CLIP tokenizer only.')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly select ACTIVA_DEVICE=cpu.')
    devices=[] if device=='cpu' else [torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()]
    # Do not reseed M1/M4 or change their global deterministic-algorithm policy.
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(0)
        model,_,preprocess=open_clip.create_model_and_transforms(
            model_name,pretrained=weights,device=device,
            force_quick_gelu=(model_name=='ViT-B-32'),
        )
    model.eval()
    return model,preprocess,open_clip.get_tokenizer(model_name)


def load_backend(model_name, weights, device):
    path=Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Local CLIP weights missing: {path}; run scripts/prepare_weights.py.')
    manifest_path=path.with_suffix('.manifest.json')
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('architecture')!=model_name or manifest.get('sha256')!=checkpoint_identity(path):
            raise ValueError('Local checkpoint manifest architecture/checksum mismatch.')
    return _load_cached(model_name,str(path),device,_signature(path))

# Preserve the documented/testing cache reset hook without caching caller-relative paths.
load_backend.cache_clear=_load_cached.cache_clear


def autocast_context(device):
    flag=os.environ.get('ACTIVA_CLIP_FP16','0')
    if flag not in ('0','1'):
        raise ValueError('ACTIVA_CLIP_FP16 must be 0 or 1.')
    return torch.autocast(device_type='cuda',dtype=torch.float16) if flag=='1' and device.startswith('cuda') else nullcontext()


def normalize(values):
    values=np.asarray(values,dtype=np.float32)
    norms=np.linalg.norm(values.astype(np.float64),axis=-1,keepdims=True)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(norms)) or np.any(norms<=1e-12):
        raise ValueError('CLIP embeddings must be finite and nonzero.')
    return (values/norms).astype(np.float32)


def text_features(prompts,cfg):
    model,_,tokenizer=load_backend(cfg.model,cfg.weights,cfg.device)
    with torch.inference_mode(),autocast_context(cfg.device):
        vectors=model.encode_text(tokenizer(prompts).to(cfg.device)).float().cpu().numpy()
    return normalize(vectors)
