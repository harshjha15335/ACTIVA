#!/usr/bin/env python3
"""Explicit real-image semantic sanity check, not FPVSum category accuracy."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from PIL import Image
import torch
from clip_backend import load_backend,model_identifier,normalize,text_features
from perception_options import settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--case',action='append',nargs=2,metavar=('IMAGE','TEXT'),required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=replace(settings(),weights=str(Path(args.weights).resolve()))
    def offline(*_,**__):raise RuntimeError('Network attempted during offline validation')
    socket.socket.connect=offline
    start=time.perf_counter()
    model,preprocess,_=load_backend(cfg.model,cfg.weights,cfg.device)
    load_seconds=time.perf_counter()-start
    images=torch.stack([preprocess(Image.open(path).convert('RGB')) for path,_ in args.case]).to(cfg.device)
    with torch.inference_mode():
        a=normalize(model.encode_image(images).float().cpu().numpy())
        b=normalize(model.encode_image(images).float().cpu().numpy())
    prompts=[text for _,text in args.case]
    text=text_features(prompts,cfg)
    similarities=a@text.T
    passed=bool(np.array_equal(np.argmax(similarities,axis=1),np.arange(len(args.case))))
    result={'kind':'real-image-semantic-sanity-not-dataset-accuracy','model':model_identifier(cfg.model,cfg.weights),'device':cfg.device,'dimension':a.shape[1],'offline_socket_connections_blocked':True,'eval_mode':not model.training,'float32':str(a.dtype)=='float32','maximum_norm_error':float(np.max(np.abs(np.linalg.norm(a,axis=1)-1))),'repeat_max_absolute_error':float(np.max(np.abs(a-b))),'repeat_tolerance':1e-6,'prompts':prompts,'cosine_similarities':similarities.tolist(),'semantic_sanity_pass':passed,'model_load_seconds':load_seconds,'total_seconds':time.perf_counter()-start}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not passed or result['maximum_norm_error']>1e-6 or result['repeat_max_absolute_error']>1e-6:raise SystemExit(1)

if __name__=='__main__':main()
