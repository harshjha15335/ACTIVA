#!/usr/bin/env python3
"""Exercise an actual M3 selector with genuine cached M2 outputs and controlled peaks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from contracts import Importance
from perception_cache import read_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--require-importance-sensitive',action='store_true')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('m3_selector_handoff',args.optimizer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    info,clip,importance,manifest=read_cache(args.manifest)
    actual=module.select_highlights(info,[],importance,clip)
    left=np.zeros(len(clip.times),np.float32);right=left.copy();left[0]=1;right[-1]=1
    first=module.select_highlights(info,[],Importance(clip.times,left),clip)
    second=module.select_highlights(info,[],Importance(clip.times,right),clip)
    intervals=lambda items:[(float(item.start),float(item.end)) for item in items]
    positions=intervals(actual)
    valid=all(0<=a<b<=info.duration for a,b in positions) and all(a>=b for (_,b),(a,_) in zip(positions,positions[1:])) and sum(b-a for a,b in positions)<=.15*info.duration and all(0<=x.importance<=1 for x in actual)
    responsive=intervals(first)!=intervals(second)
    result={'actual_M3_source':str(Path(args.optimizer).resolve()),'M2_cache_manifest':str(Path(args.manifest).resolve()),'model':clip.model_name,'feature_shape':list(clip.feats.shape),'importance_grid_equal':bool(np.array_equal(clip.times,importance.times)),'empty_actions_supported':True,'local_15_percent_budget_and_intervals_valid':valid,'intervals':positions,'controlled_early_peak_intervals':intervals(first),'controlled_late_peak_intervals':intervals(second),'importance_sensitive':responsive,'interpretation':'Owner follow-up required: inspected selector explicitly implements uniform locations, not B0/B2 importance selection.' if not responsive else 'Locations respond to the controlled importance curves; objective correctness still requires owner tests.','official_integration_verified':False}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not valid or args.require_importance_sensitive and not responsive:raise SystemExit(2)

if __name__=='__main__':main()
