"""Opt-in real pretrained evidence. Never silently substitutes a random model."""
import json
import os
from pathlib import Path
import socket
import shutil
import sys
import numpy as np
import pytest
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from clip_backend import load_backend,model_identifier,text_features,normalize
from clip_features import extract_clip_features
from importance import predict_importance
from category import classify_video
from perception_options import settings
from perception_cache import read_cache
from video_io import inspect_video

@pytest.fixture(scope='module')
def real_inputs():
    weights=os.environ.get('ACTIVA_REAL_CLIP_WEIGHTS')
    media=os.environ.get('ACTIVA_REAL_MEDIA')
    if not weights or not media:
        pytest.skip('BLOCKED: set ACTIVA_REAL_CLIP_WEIGHTS and ACTIVA_REAL_MEDIA for real pretrained validation.')
    if not Path(weights).is_file():pytest.fail('Explicit real checkpoint is missing; no random fallback.')
    return str(Path(weights).resolve()),Path(media)


def test_real_pretrained_semantics_offline_and_rng_scope(real_inputs,monkeypatch):
    weights,media=real_inputs
    monkeypatch.setenv('ACTIVA_CLIP_WEIGHTS',weights)
    def offline(*_,**__):raise AssertionError('Inference attempted a network connection')
    monkeypatch.setattr(socket.socket,'connect',offline)
    load_backend.cache_clear();torch.manual_seed(174)
    before=torch.get_rng_state().clone();deterministic=torch.are_deterministic_algorithms_enabled()
    cfg=settings();model,preprocess,_=load_backend(cfg.model,cfg.weights,cfg.device)
    assert torch.equal(before,torch.get_rng_state())
    assert deterministic==torch.are_deterministic_algorithms_enabled()
    images=torch.stack([preprocess(Image.open(media/name).convert('RGB')) for name in ['dog.jpg','bus.jpg']])
    with torch.inference_mode():features=normalize(model.encode_image(images).numpy())
    text=text_features(['a photo of a dog','a photo of a bus'],cfg)
    assert features.shape==(2,512)
    np.testing.assert_array_equal(np.argmax(features@text.T,axis=1),[0,1])
    assert not model.training


def test_real_video_all_baselines_pooling_and_determinism(real_inputs,tmp_path,monkeypatch):
    weights,media=real_inputs
    monkeypatch.setenv('ACTIVA_CLIP_WEIGHTS',weights)
    labels=tmp_path/'labels.json';labels.write_text('["walking", "riding a bicycle", "cooking"]')
    monkeypatch.setenv('ACTIVA_CATEGORIES',str(labels))
    monkeypatch.setattr(socket.socket,'connect',lambda *_args,**_kwargs: (_ for _ in ()).throw(AssertionError('network')))
    info=inspect_video(media/'walking-smoke.mp4');clip=extract_clip_features(info)
    renamed=tmp_path/'unrelated-name.mp4';shutil.copyfile(info.path,renamed)
    second=extract_clip_features(inspect_video(renamed))
    assert clip.feats.shape==(20,512) and clip.feats.dtype==np.float32
    np.testing.assert_allclose(np.linalg.norm(clip.feats,axis=1),1,atol=1e-6)
    np.testing.assert_allclose(clip.feats,second.feats,rtol=0,atol=1e-6)
    np.testing.assert_array_equal(clip.times,second.times)
    for mode in ['constant','novelty','motion','text']:
        monkeypatch.setenv('ACTIVA_IMPORTANCE',mode);imp=predict_importance(info,clip)
        assert np.isfinite(imp.values).all() and np.all((imp.values>=0)&(imp.values<=1))
        np.testing.assert_array_equal(imp.times,clip.times)
        for pooling in ['mean','weighted','top']:
            monkeypatch.setenv('ACTIVA_POOLING',pooling)
            result=classify_video(info,clip,imp)
            assert result.label in json.loads(labels.read_text())
    from perception_cache import write_cache
    path=write_cache(tmp_path/'cache',info,clip,imp,{'importance':'text'})
    _,loaded,importance,manifest=read_cache(path,clip.model_name)
    np.testing.assert_array_equal(loaded.feats,clip.feats)
    assert manifest['dimension']==512
