#!/usr/bin/env python3
"""Create development fingerprints for human overlap investigation, not clearance."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from leakage_fingerprints import fingerprint_video
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('videos',nargs='+');parser.add_argument('--output',required=True)
args=parser.parse_args()
rows=[fingerprint_video(path) for path in args.videos]
Path(args.output).write_text(json.dumps({'format':'activa-fingerprint-inventory-v1','videos':rows},indent=2)+'\n')
print(f'Wrote {len(rows)} review fingerprints; no training clearance granted.')
