import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from clip_backend import checkpoint_identity,load_backend

@pytest.fixture
def setup_module():
    spec=importlib.util.spec_from_file_location('checkpoint_setup_tests',Path(__file__).resolve().parents[1]/'scripts/prepare_weights.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_partial_download_rejected_and_cleaned(tmp_path,setup_module,monkeypatch):
    class Response(io.BytesIO):
        headers={'Content-Length':'100'}
    monkeypatch.setattr(setup_module.urllib.request,'urlopen',lambda *_args,**_kwargs:Response(b'partial'))
    output=tmp_path/'model.pt'
    with pytest.raises(ValueError,match='Incomplete'):setup_module.install('openai',output)
    assert not output.exists() and not list(tmp_path.glob('*.part'))


def test_local_manual_source_still_requires_checksum(tmp_path,setup_module):
    source=tmp_path/'untrusted.pt';source.write_bytes(b'not a valid checkpoint')
    with pytest.raises(ValueError,match='checksum'):setup_module.install('openai',tmp_path/'model.pt',source)
    assert not (tmp_path/'model.pt').exists()


def test_invalid_explicit_checksum_rejected(tmp_path,setup_module):
    with pytest.raises(ValueError,match='64 lowercase'):setup_module.install('openai',tmp_path/'model.pt',expected_sha256='short')


def test_checkpoint_identity_changes_with_same_filename(tmp_path):
    path=tmp_path/'model.pt';path.write_bytes(b'old data');old=checkpoint_identity(path)
    path.write_bytes(b'new data');assert checkpoint_identity(path)!=old


def test_runtime_manifest_tampering_rejected_before_loading(tmp_path):
    path=tmp_path/'model.safetensors';path.write_bytes(b'not a model')
    path.with_suffix('.manifest.json').write_text(json.dumps({'architecture':'ViT-B-32','sha256':'0'*64}))
    with pytest.raises(ValueError,match='manifest'):load_backend('ViT-B-32',str(path),'cpu')


def test_updated_file_signature_reaches_backend_cache(tmp_path,monkeypatch):
    import clip_backend
    path=tmp_path/'model.pt';path.write_bytes(b'old');calls=[]
    monkeypatch.setattr(clip_backend,'_load_cached',lambda model,path,device,signature:calls.append((path,signature)) or ('model','preprocess','tokenizer'))
    clip_backend.load_backend('ViT-B-32',str(path),'cpu');path.write_bytes(b'new-size')
    clip_backend.load_backend('ViT-B-32',str(path),'cpu')
    assert calls[0]!=calls[1] and Path(calls[0][0]).is_absolute()

def test_supplied_checksum_cannot_override_published_identity(tmp_path,setup_module):
    with pytest.raises(ValueError,match='conflicts with the published'):
        setup_module.install('openai',tmp_path/'model.pt',expected_sha256='0'*64)
