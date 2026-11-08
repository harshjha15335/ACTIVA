#!/usr/bin/env python3
"""Leakage-gated held-out baseline/pooling evaluation. No training or tuning.

Inputs use documented ACTIVA development schemas, not guessed organizer formats.
Ranking F1 is a sample-grid proxy and cannot be claimed as final highlight F1.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from fpvsum_data import load_inventory,load_processed,load_aliases,load_review,assert_safe_split,assert_video_hash,resample_processed
from perception_metrics import ranking_f1,category_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['study','inventory','annotations','review','official-manifest','weights','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--categories');parser.add_argument('--aliases')
    args=parser.parse_args()
    study=json.loads(Path(args.study).read_text())
    if study.get('format')!='activa-m2-study-v1' or not isinstance(study.get('validation'),list) or not study['validation']:
        raise ValueError('Expected an explicit nonempty activa-m2-study-v1 validation list.')
    inventory=load_inventory(args.inventory)
    review=load_review(args.review,args.official_manifest)
    ids=[entry['video_id'] for entry in study['validation']]
    assert_safe_split(study.get('training_ids',[]),ids,review,inventory)
    annotations=load_processed(args.annotations,inventory,load_aliases(args.aliases))
    os.environ['ACTIVA_CLIP_WEIGHTS']=str(Path(args.weights).resolve())
    if args.categories:os.environ['ACTIVA_CATEGORIES']=str(Path(args.categories).resolve())
    from perception_options import settings,IMPORTANCE_METHODS,POOLING_METHODS
    from category_vocab import load_categories
    from clip_features import extract_clip_features
    from importance import predict_importance
    from category import classify_video
    from video_io import inspect_video
    cfg=settings();vocabulary=load_categories(cfg)
    truths=[];predictions={(m,p):[] for m in IMPORTANCE_METHODS for p in POOLING_METHODS}
    f1={m:[] for m in IMPORTANCE_METHODS};per_video=[]
    for entry in study['validation']:
        name=entry['video_id'];assert_video_hash(name,entry['path'],review)
        info=inspect_video(entry['path']);clip=extract_clip_features(info)
        with np.load(entry['source_times_npz'],allow_pickle=False) as data:source_times=data['times']
        targets=resample_processed(annotations[name],source_times,clip.times,info.duration)
        label=inventory[name].category
        if label not in vocabulary.labels:
            if not study.get('category_map_reviewed_by'):raise ValueError('Exact category mapping requires explicit human review.')
            label=study.get('category_map',{}).get(label)
        if label not in vocabulary.labels:raise ValueError(f'Unmapped evaluation category for {name}.')
        truths.append(label);record={'video_id':name,'samples':len(clip.times),'baselines':{}}
        for mode in IMPORTANCE_METHODS:
            os.environ['ACTIVA_IMPORTANCE']=mode;importance=predict_importance(info,clip)
            result=ranking_f1(targets,importance.values);f1[mode].append(result['f1']);record['baselines'][mode]=result
            for pooling in POOLING_METHODS:
                os.environ['ACTIVA_POOLING']=pooling
                predictions[(mode,pooling)].append(classify_video(info,clip,importance).label)
        per_video.append(record)
    result={'format':'activa-m2-heldout-development-v1','official_metric':False,'metric_warning':'Sample-grid ranking F1, not original-frame F1 or final M3 highlights. No interval/budget metric is inferred.','validation_ids':ids,'sample_count':len(ids),'canonical_vocabulary':vocabulary.canonical,'baseline_macro_ranking_f1':{m:float(np.mean(values)) for m,values in f1.items()},'category_pooling':{f'{m}/{p}':category_metrics(vocabulary.labels,truths,values) for (m,p),values in predictions.items()},'per_video':per_video,'model':clip.model_name,'selection_frozen':False}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Wrote held-out development report for {len(ids)} reviewed videos; no official score claimed.')

if __name__=='__main__':main()
