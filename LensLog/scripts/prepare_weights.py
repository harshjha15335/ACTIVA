#!/usr/bin/env python3
"""Explicit, verified CLIP setup. Inference never invokes this downloader."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.error
import urllib.request

SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
URL = f'https://openaipublic.azureedge.net/clip/models/{SHA256}/ViT-B-32.pt'
PRESETS = {
    'openai': {'url': URL, 'hash_prefix': SHA256, 'filename': 'ViT-B-32.pt', 'format': 'torchscript'},
    'laion400m_e32': {
        'url': 'https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt',
        'hash_prefix': '46683a32',
        'filename': 'ViT-B-32-laion400m_e32.pt', 'format': 'state_dict',
    },
}


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _temporary(directory, suffix):
    with tempfile.NamedTemporaryFile(dir=directory, suffix=suffix, delete=False) as stream:
        return Path(stream.name)


def convert_checkpoint(output, preset='openai', expected_sha256=None):
    import torch
    from safetensors.torch import save_file
    output = Path(output)
    specification = PRESETS[preset]
    published = SHA256 if preset == 'openai' else specification['hash_prefix']
    if expected_sha256 and not expected_sha256.startswith(published):
        raise ValueError('Supplied SHA-256 conflicts with the published checkpoint checksum.')
    expected = expected_sha256 or published
    original_hash = digest(output)
    if not original_hash.startswith(expected):
        raise ValueError('Source checkpoint checksum mismatch before conversion.')
    if specification['format'] == 'torchscript':
        archive = torch.jit.load(str(output), map_location='cpu').eval()
        state = archive.state_dict()
    else:
        state = torch.load(output, map_location='cpu', weights_only=True)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
    if not isinstance(state, dict) or not state or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Expected a nonempty tensor state dictionary.')
    state = {key.removeprefix('module.'): value.detach().cpu().contiguous()
             for key,value in state.items() if key not in {'input_resolution','context_length','vocab_size'}}
    converted = output.with_suffix('.safetensors')
    manifest_path = converted.with_suffix('.manifest.json')
    if converted.exists():
        if not manifest_path.exists():
            raise ValueError('Existing converted checkpoint has no provenance manifest; refusing to overwrite.')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('source_sha256') != original_hash or manifest.get('sha256') != digest(converted):
            raise ValueError('Existing converted checkpoint checksum/provenance mismatch.')
        return manifest
    temporary = _temporary(converted.parent, '.safetensors.part')
    try:
        save_file(state, str(temporary), metadata={'source_sha256':original_hash,'preset':preset,'architecture':'ViT-B-32','quick_gelu':'true'})
        temporary.replace(converted)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        'format':'activa-clip-checkpoint-v1','architecture':'ViT-B-32','preset':preset,
        'embedding_dimension':512,'quick_gelu':True,'preprocess':'openai-clip-center-crop-224',
        'source_url':specification['url'],'published_hash_prefix':published,
        'supplied_expected_sha256':expected_sha256,
        'checksum_status':('published-full-sha256-verified' if len(published)==64 else 'published-prefix-and-supplied-full-sha256-verified' if expected_sha256 else 'published-sha256-prefix-verified; full hash computed locally'),
        'source_sha256':original_hash,'filename':converted.name,'sha256':digest(converted),
        'bytes':converted.stat().st_size,
        'license_reference':'https://github.com/mlfoundations/open_clip/blob/main/LICENSE' if preset!='openai' else 'https://github.com/openai/CLIP/blob/main/LICENSE',
        'team_model_approval':'pending',
    }
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def install(preset, output, local_source=None, expected_sha256=None):
    specification=PRESETS[preset]
    expected=expected_sha256 or specification['hash_prefix']
    if expected_sha256 and (len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('An explicit SHA-256 must be 64 lowercase hexadecimal characters.')
    if expected_sha256 and not expected_sha256.startswith(specification['hash_prefix']):
        raise ValueError('Supplied SHA-256 conflicts with the published checkpoint checksum.')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if not digest(output).startswith(expected):
            raise ValueError('Existing checkpoint checksum mismatch; refusing to overwrite.')
    else:
        temporary=_temporary(output.parent,'.download.part')
        try:
            if local_source:
                with Path(local_source).open('rb') as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source,target)
            else:
                request=urllib.request.Request(specification['url'],headers={'User-Agent':'ACTIVA-checkpoint-setup/1'})
                with urllib.request.urlopen(request,timeout=60) as source, temporary.open('wb') as target:
                    length=source.headers.get('Content-Length')
                    received=0
                    while chunk:=source.read(1024*1024):
                        received+=len(chunk)
                        if received>800*1024*1024:
                            raise ValueError('Unexpected checkpoint size exceeds 800 MiB.')
                        target.write(chunk)
                    if length is not None and received!=int(length):
                        raise ValueError('Incomplete download: Content-Length mismatch.')
            if not digest(temporary).startswith(expected):
                raise ValueError('Downloaded checkpoint checksum mismatch.')
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return convert_checkpoint(output,preset,expected_sha256)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=PRESETS,default='openai')
    parser.add_argument('--output')
    parser.add_argument('--local-source',help='Already obtained original upstream archive; checksum is still verified')
    parser.add_argument('--expected-sha256',help='Optional independently trusted full SHA-256')
    args=parser.parse_args()
    output=args.output or str(Path(__file__).resolve().parents[1]/'weights'/PRESETS[args.preset]['filename'])
    try:
        manifest=install(args.preset,output,args.local_source,args.expected_sha256)
    except (OSError,ValueError,RuntimeError,urllib.error.URLError) as error:
        parser.exit(1,f'Checkpoint setup failed: {error}\nUse --local-source with the original trusted archive; verification remains required.\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
