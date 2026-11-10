"""CLI > environment > shared defaults; no shared config edits needed."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import perception_cli
from perception_options import settings, PROJECT_ROOT

@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in list(os.environ):
        if name.startswith('ACTIVA_'):
            monkeypatch.delenv(name)

@pytest.mark.parametrize('environment,flags,expected', [
    ({}, [], ('novelty','mean')),
    ({'ACTIVA_IMPORTANCE':'constant','ACTIVA_POOLING':'top'}, [], ('constant','top')),
    ({}, ['--importance','motion','--pooling','weighted'], ('motion','weighted')),
    ({'ACTIVA_IMPORTANCE':'constant','ACTIVA_POOLING':'top'}, ['--importance','text','--pooling','mean'], ('text','mean')),
])
def test_cli_precedence(monkeypatch,tmp_path,environment,flags,expected):
    for key,value in environment.items():monkeypatch.setenv(key,value)
    monkeypatch.setattr(sys,'argv',['perception',str(tmp_path/'input.mp4'),str(tmp_path/'output.json'),'--mode','motion-only',*flags])
    monkeypatch.setattr(perception_cli,'inspect_video',lambda _: SimpleNamespace())
    monkeypatch.setattr(perception_cli,'asdict',lambda _: {})
    monkeypatch.setattr(perception_cli,'motion_importance',lambda _: SimpleNamespace(times=np.array([0.]),values=np.array([.5])))
    captured=[]
    monkeypatch.setattr(perception_cli,'atomic_json',lambda *_:captured.append((settings().baseline,settings().pooling)))
    perception_cli.main()
    assert captured==[expected]
    assert (settings().baseline,settings().pooling)==expected

@pytest.mark.parametrize('key,value',[('ACTIVA_IMPORTANCE','bad'),('ACTIVA_POOLING','bad'),('ACTIVA_BATCH_SIZE','0'),('ACTIVA_DEVICE','unknown')])
def test_invalid_environment_fails(key,value,monkeypatch):
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError):settings()

def test_default_weight_path_independent_of_cwd(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(settings().weights)==PROJECT_ROOT/'weights/ViT-B-32.safetensors'

def test_explicit_relative_path_is_caller_relative(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('ACTIVA_CLIP_WEIGHTS','my.pt')
    assert Path(settings().weights)==tmp_path/'my.pt'

def test_repeated_resolution_deterministic(monkeypatch):
    monkeypatch.setenv('ACTIVA_IMPORTANCE','constant');monkeypatch.setenv('ACTIVA_POOLING','top')
    assert settings()==settings()

@pytest.mark.parametrize('key,value',[('ACTIVA_DEVICE','cuda:bad'),('ACTIVA_CLIP_FP16','yes'),('ACTIVA_REQUIRE_CANONICAL_VOCAB','maybe')])
def test_invalid_runtime_flags_rejected(monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError):settings()
