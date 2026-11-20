#!/usr/bin/env python3
"""Explicit small public smoke assets. Never used by production inference/training.

SHA values pin previously observed source content; no upstream digest or dataset
quality authority is implied. Nothing is fetched implicitly by the test suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

ASSETS = {'dog.jpg': {'url': 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg', 'sha256': 'f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a'}, 'bus.jpg': {'url': 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg', 'sha256': 'c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63'}, 'vtest.avi': {'url': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi', 'sha256': '45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf'}}


def digest(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not shutil.which('ffmpeg'):parser.error('FFmpeg must be installed on PATH.')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    for name,source in ASSETS.items():
        target=root/name
        if target.exists():
            if digest(target)!=source['sha256']:raise ValueError(f'Existing smoke asset differs: {name}; refusing to overwrite.')
            continue
        with tempfile.NamedTemporaryFile(dir=root,suffix='.download',delete=False) as stream:
            temporary=Path(stream.name)
        try:
            with urllib.request.urlopen(source['url'],timeout=30) as response,temporary.open('wb') as stream:
                received=0
                while chunk:=response.read(1024*1024):
                    received+=len(chunk)
                    if received>20*1024*1024:raise ValueError('Unexpected smoke asset size.')
                    stream.write(chunk)
            if digest(temporary)!=source['sha256']:raise ValueError(f'Smoke asset hash mismatch: {name}')
            temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    video=root/'walking-smoke.mp4';manifest=root/'validation-assets.json'
    if video.exists():
        previous=json.loads(manifest.read_text()) if manifest.exists() else {}
        if previous.get('video_sha256')!=digest(video):
            raise ValueError('Existing smoke MP4 has no matching manifest; refusing to overwrite.')
    else:
        with tempfile.NamedTemporaryFile(dir=root,suffix='.mp4',delete=False) as stream:
            temporary=Path(stream.name)
        try:
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(root/'vtest.avi'),'-t','10','-c:v','libx264','-pix_fmt','yuv420p',str(temporary)],check=True)
            temporary.replace(video)
        finally:temporary.unlink(missing_ok=True)
    result={'purpose':'public-smoke-assets-NOT-heldout-quality-or-training','sources':ASSETS,
            'video_sha256':digest(video),'video_recipe':'First 10 seconds of vtest.avi; libx264,yuv420p',
            'note':'Original source content is pinned; generated MP4 bytes may depend on FFmpeg version. Review upstream asset licenses before redistribution.'}
    manifest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
