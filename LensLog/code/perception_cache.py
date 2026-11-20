"""Versioned DEVELOPMENT-only artifacts. Production inference never reads these."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
import numpy as np
from contracts import CLIP_FPS,ClipFeats,Importance,VideoInfo
from importance import validate_clip


def file_sha256(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);np.savez_compressed(stream,**arrays)
        temporary.replace(path)
    finally:
        if temporary:temporary.unlink(missing_ok=True)


def write_cache(directory,info,clip,imp,configuration):
    times,feats=validate_clip(info,clip)
    if not np.array_equal(times,imp.times) or np.asarray(imp.values).shape!=times.shape or not np.isfinite(imp.values).all() or np.any((imp.values<0)|(imp.values>1)):
        raise ValueError('Importance must exactly match the feature grid and lie in [0,1].')
    if clip.feats.dtype!=np.float32 or not np.allclose(np.linalg.norm(clip.feats,axis=1),1,atol=1e-5):
        raise ValueError('Cache requires normalized float32 features.')
    source_hash=file_sha256(info.path)
    configuration=dict(configuration)
    version_hash=hashlib.sha256(json.dumps({'model':clip.model_name,'configuration':configuration},sort_keys=True).encode()).hexdigest()
    slug=re.sub(r'[^A-Za-z0-9_.-]','_',info.video_id)[:80] or 'video'
    key=f'{slug}.{source_hash[:12]}.{version_hash[:12]}'
    root=Path(directory).resolve();clip_path=root/'clip'/f'{key}.npz';imp_path=root/'importance'/f'{key}.npz'
    _atomic_npz(clip_path,times=clip.times,feats=clip.feats,model_name=np.asarray(clip.model_name))
    _atomic_npz(imp_path,times=imp.times,values=imp.values)
    manifest={'format':'activa-m2-cache-v1','development_only':True,'video':asdict(info),'input_sha256':source_hash,'model_name':clip.model_name,'dimension':feats.shape[1],'target_fps':CLIP_FPS,'times_shape':list(times.shape),'feature_shape':list(feats.shape),'importance_shape':list(imp.values.shape),'configuration':configuration,'configuration_sha256':version_hash,'clip':{'path':str(clip_path.relative_to(root)),'sha256':file_sha256(clip_path)},'importance':{'path':str(imp_path.relative_to(root)),'sha256':file_sha256(imp_path)}}
    path=root/'manifests'/f'{key}.json';path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary:temporary.unlink(missing_ok=True)
    return path


def read_cache(path,expected_model=None):
    """Consumer helper for experiments only; validates hashes before loading arrays."""
    path=Path(path).resolve();root=path.parent.parent
    manifest=json.loads(path.read_text())
    if manifest.get('format')!='activa-m2-cache-v1' or manifest.get('development_only') is not True:
        raise ValueError('Unsupported development cache format.')
    if expected_model is not None and manifest['model_name']!=expected_model:
        raise ValueError('Cache checkpoint identity mismatch.')
    paths={}
    for name in ['clip','importance']:
        target=(root/manifest[name]['path']).resolve()
        if not target.is_relative_to(root):raise ValueError('Cache path escapes artifact root.')
        if file_sha256(target)!=manifest[name]['sha256']:raise ValueError('Cache checksum mismatch.')
        paths[name]=target
    info=VideoInfo(**manifest['video'])
    if Path(info.path).exists() and file_sha256(info.path)!=manifest['input_sha256']:
        raise ValueError('Input video changed since cache generation.')
    with np.load(paths['clip'],allow_pickle=False) as data:
        clip=ClipFeats(data['times'].copy(),data['feats'].copy(),str(data['model_name'].item()))
    with np.load(paths['importance'],allow_pickle=False) as data:
        imp=Importance(data['times'].copy(),data['values'].copy())
    validate_clip(info,clip)
    if clip.model_name!=manifest['model_name'] or clip.feats.shape!=tuple(manifest['feature_shape']) or clip.feats.shape[1]!=manifest['dimension'] or clip.feats.dtype!=np.float32 or not np.allclose(np.linalg.norm(clip.feats,axis=1),1,atol=1e-5):
        raise ValueError('Cache feature metadata mismatch.')
    if not np.array_equal(clip.times,imp.times) or imp.values.shape!=clip.times.shape or not np.isfinite(imp.values).all() or np.any((imp.values<0)|(imp.values>1)):
        raise ValueError('Cache importance grid/values invalid.')
    return info,clip,imp,manifest
