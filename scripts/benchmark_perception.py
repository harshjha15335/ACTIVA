#!/usr/bin/env python3
"""Stage-level actual CLIP benchmark; reports fixture identity, never model quality."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('output')
    parser.add_argument('--weights',required=True);parser.add_argument('--categories',required=True)
    parser.add_argument('--offline',action='store_true');parser.add_argument('--cache-dir')
    args=parser.parse_args()
    os.environ['ACTIVA_CLIP_WEIGHTS']=str(Path(args.weights).resolve())
    os.environ['ACTIVA_CATEGORIES']=str(Path(args.categories).resolve())
    started=time.perf_counter()
    from clip_backend import load_backend,model_identifier
    from clip_features import extract_clip_features
    from importance import predict_importance
    from category import classify_video
    from perception_options import settings
    from video_io import inspect_video
    imported=time.perf_counter()
    if args.offline:
        def offline(*_,**__):raise RuntimeError('Network attempted during offline benchmark')
        socket.socket.connect=offline
    cfg=settings();info=inspect_video(args.input)
    start=time.perf_counter();load_backend(cfg.model,cfg.weights,cfg.device);load_seconds=time.perf_counter()-start
    start=time.perf_counter();clip=extract_clip_features(info);extract_seconds=time.perf_counter()-start
    start=time.perf_counter();imp=predict_importance(info,clip);importance_seconds=time.perf_counter()-start
    start=time.perf_counter();cat=classify_video(info,clip,imp);category_seconds=time.perf_counter()-start
    try:
        import resource
        peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform=='darwin':peak_kib/=1024
    except ImportError:peak_kib=None
    result={'kind':'real-pretrained-CLIP-runtime-not-quality-evaluation','video':asdict(info),'samples':len(clip.times),'embedding_dimension':clip.feats.shape[1],'model':model_identifier(cfg.model,cfg.weights),'device':cfg.device,'batch_size':cfg.batch_size,'offline_socket_connections_blocked':args.offline,'importance_method':cfg.baseline,'category':asdict(cat),'timing_seconds':{'import':imported-started,'load':load_seconds,'extract':extract_seconds,'importance':importance_seconds,'category':category_seconds,'total':time.perf_counter()-started},'peak_cpu_rss_kib':peak_kib,'peak_gpu_allocated_bytes':__import__('torch').cuda.max_memory_allocated() if cfg.device.startswith('cuda') else None}
    if args.cache_dir:
        from perception_cache import write_cache
        result['cache_manifest']=str(write_cache(args.cache_dir,info,clip,imp,{'importance':cfg.baseline,'pooling':cfg.pooling}))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
