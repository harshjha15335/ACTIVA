#!/usr/bin/env python3
"""Validate upstream FPVSum inventory/annotation files without training or clearance."""
import argparse,json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from fpvsum_data import load_inventory,load_processed,load_raw,load_aliases,load_category_aliases
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inventory',required=True);parser.add_argument('--processed',required=True)
parser.add_argument('--raw',required=True);parser.add_argument('--aliases');parser.add_argument('--output',required=True)
args=parser.parse_args();inventory=load_inventory(args.inventory);aliases=load_aliases(args.aliases)
result={'inventory_videos':len(inventory),'split_counts':dict(Counter(x.split for x in inventory.values())),'categories':dict(Counter(x.category for x in inventory.values())),'leakage_status':'BLOCKED BY MANIFEST','training_allowed':False}
blocked=False
for name,loader,path in [('processed',load_processed,args.processed),('raw',load_raw,args.raw)]:
    try:
        records=loader(path,inventory,aliases,load_category_aliases(args.aliases)) if name=="raw" else loader(path,inventory,aliases);result[name]={'status':'PASS','records':len(records)}
    except ValueError as error:
        result[name]={'status':'BLOCKED','reason':str(error)};blocked=True
Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(2 if blocked else 0)
