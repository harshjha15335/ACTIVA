"""FPVSum development parsing. Never imported by production perception inference.

Processed Score is authoritative input here; raw annotators are preserved, never
silently averaged to recreate it. Original frame PTS must be supplied explicitly.
"""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    source_id: str
    category: str
    split: str

@dataclass
class Annotation:
    video_id: str
    scores: np.ndarray
    original_name: str

@dataclass
class RawAnnotation:
    video_id: str
    annotator_id: int
    category: str
    labels: np.ndarray


def load_inventory(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not {'video_name','youtube_id','category','split'}<=set(reader.fieldnames or []):
            raise ValueError('FPVSum CSV requires video_name,youtube_id,category,split.')
        rows=list(reader)
    if not rows:raise ValueError('Empty inventory.')
    records={}
    for row in rows:
        if any(not row.get(key,'').strip() for key in ['video_name','youtube_id','category','split']):
            raise ValueError('Inventory fields cannot be empty.')
        if row['split'] not in ('labeled','unlabeled'):raise ValueError('Unknown label split.')
        name=row['video_name']
        if name in records:raise ValueError(f'Duplicate video ID: {name}')
        records[name]=VideoRecord(name,row['youtube_id'],row['category'],row['split'])
    return records


def load_aliases(path):
    if path is None:return {}
    data=json.loads(Path(path).read_text())
    if data.get('format')!='activa-reviewed-id-aliases-v1' or not data.get('reviewed_by') or not data.get('evidence'):
        raise ValueError('ID aliases require an explicit reviewed mapping with evidence.')
    aliases=data.get('aliases')
    if not isinstance(aliases,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in aliases.items()) or len(set(aliases.values()))!=len(aliases):
        raise ValueError('ID aliases must be one-to-one exact string mappings.')
    return aliases


def load_category_aliases(path):
    if path is None:return {}
    # Reuse the evidence/reviewer validation, but do not infer any case conversion.
    load_aliases(path)
    aliases=json.loads(Path(path).read_text()).get('category_aliases',{})
    if not isinstance(aliases,dict) or any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in aliases.items()):
        raise ValueError('Category aliases require explicit exact string mappings.')
    return aliases


def _identity(name,inventory,aliases):
    mapped=aliases.get(name,name)
    if mapped not in inventory:
        raise ValueError(f'Unresolved annotation video ID {name!r}; no automatic spelling/underscore repair is allowed.')
    if inventory[mapped].split!='labeled':raise ValueError(f'Annotation supplied for unlabeled video: {mapped}')
    return mapped


def _mat_records(path,key):
    from scipy.io import loadmat
    data=loadmat(path,squeeze_me=True,struct_as_record=False)
    if key not in data:raise ValueError(f'MAT file missing {key}.')
    return np.asarray(data[key],dtype=object).reshape(-1)


def load_processed(path,inventory,aliases=None,require_all=True):
    aliases=aliases or {};annotations={}
    for item in _mat_records(path,'gt_data'):
        if not hasattr(item,'Name') or not hasattr(item,'Score'):raise ValueError('Processed record requires Name and Score.')
        original=str(item.Name);name=_identity(original,inventory,aliases)
        if name in annotations:raise ValueError(f'Duplicate processed annotation: {name}')
        scores=np.atleast_1d(np.asarray(item.Score,dtype=np.float64))
        if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
            raise ValueError('Processed Score requires finite 1D values in [0,1].')
        annotations[name]=Annotation(name,scores,original)
    missing={name for name,row in inventory.items() if row.split=='labeled'}-annotations.keys()
    if require_all and missing:raise ValueError(f'Missing processed annotations: {sorted(missing)}')
    return annotations


def load_raw(path,inventory,aliases=None,category_aliases=None):
    aliases=aliases or {};category_aliases=category_aliases or {};result=[];seen=set();lengths={}
    for item in _mat_records(path,'raw_data'):
        if any(not hasattr(item,field) for field in ['Name','Category','Annotator_id','Label']):
            raise ValueError('Raw record requires Name,Category,Annotator_id,Label.')
        name=_identity(str(item.Name),inventory,aliases)
        annotator=float(item.Annotator_id)
        if not np.isfinite(annotator) or not annotator.is_integer():raise ValueError('Annotator ID must be an integer.')
        identity=(name,int(annotator))
        if identity in seen:raise ValueError('Duplicate annotator/video record.')
        seen.add(identity)
        if category_aliases.get(str(item.Category),str(item.Category))!=inventory[name].category:
            raise ValueError('Raw annotation category mismatch; an evidence-backed category alias is required.')
        labels=np.atleast_1d(np.asarray(item.Label,dtype=np.float64))
        # Upstream raw_data contains both binary and fractional [0,1] Label vectors.
        # Preserve values exactly; do not infer or undo undocumented processing.
        if labels.ndim!=1 or not len(labels) or not np.isfinite(labels).all() or np.any((labels<0)|(labels>1)):
            raise ValueError('Raw Label requires a finite frame vector in [0,1].')
        if name in lengths and lengths[name]!=len(labels):raise ValueError('Annotators disagree on frame count.')
        lengths[name]=len(labels)
        result.append(RawAnnotation(name,identity[1],str(item.Category),labels.astype(np.float64)))
    return result


def resample_processed(annotation,source_times,target_times,duration):
    """Duration-weighted bin means of original-frame scores on explicit PTS.

    Score[i] describes [source_pts[i],source_pts[i+1]); final frame ends at duration.
    This is our development alignment convention, not an assumed organizer rule.
    """
    source=np.asarray(source_times,dtype=np.float64);target=np.asarray(target_times,dtype=np.float64)
    scores=np.asarray(annotation.scores,dtype=np.float64)
    if source.ndim!=1 or source.shape!=scores.shape or target.ndim!=1 or not len(target) or not len(source):
        raise ValueError('Explicit source PTS must match every annotation frame; target must be nonempty.')
    if not np.isfinite(source).all() or not np.isfinite(target).all() or not np.isfinite(duration) or duration<=0 or np.any(np.diff(source)<=0) or np.any(np.diff(target)<=0) or source[0]<0 or source[-1]>=duration or target[0]<source[0] or target[-1]>=duration:
        raise ValueError('Invalid or uncovered annotation/target timeline.')
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):raise ValueError('Invalid processed scores.')
    edges=np.r_[source,duration]
    area=np.r_[0,np.cumsum(scores*np.diff(edges))]
    def integral(points):
        indices=np.minimum(np.searchsorted(edges,points,side='right')-1,len(scores)-1)
        return area[indices]+scores[indices]*(points-edges[indices])
    bins=np.r_[target,duration]
    return np.clip(np.diff(integral(bins))/np.diff(bins),0,1).astype(np.float32)


def load_review(path,official_manifest):
    data=json.loads(Path(path).read_text())
    with Path(official_manifest).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if data.get('format')!='activa-leakage-review-v1' or data.get('state')!='reviewed' or not data.get('reviewed_by') or not data.get('reviewed_at') or data.get('official_manifest_sha256')!=actual:
        raise ValueError('A human-reviewed leakage decision bound to the actual official manifest is required.')
    rows=data.get('videos')
    if not isinstance(rows,list) or not rows:raise ValueError('Leakage review requires explicit video decisions.')
    records={}
    for row in rows:
        name=row.get('video_id')
        if not isinstance(name,str) or not name or name in records or row.get('status') not in ['reviewed-safe','evaluation-overlap','unresolved']:
            raise ValueError('Invalid or duplicate leakage decision.')
        if row['status']=='reviewed-safe':
            sha=row.get('sha256','')
            if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha) or not row.get('evidence'):
                raise ValueError('Reviewed-safe videos require an exact file hash and content-review evidence.')
        records[name]=row
    return records


def assert_safe_split(train_ids,validation_ids,review,inventory):
    train=list(train_ids);validation=list(validation_ids)
    if len(set(train))!=len(train) or len(set(validation))!=len(validation) or set(train)&set(validation):
        raise ValueError('Duplicate/overlapping video IDs across splits.')
    for name in train+validation:
        if name not in inventory or inventory[name].split!='labeled' or name not in review or review[name].get('status')!='reviewed-safe':
            raise ValueError(f'Video is not reviewed-safe and labeled: {name}')
    def sources(names):return {inventory[name].source_id for name in names}
    def hashes(names):return {review[name]['sha256'] for name in names}
    def reviewed_groups(names):return {review[name].get("source_group",inventory[name].source_id) for name in names}
    if sources(train)&sources(validation) or hashes(train)&hashes(validation) or reviewed_groups(train)&reviewed_groups(validation):
        raise ValueError('Duplicate source video/content crosses training and validation.')


def assert_video_hash(video_id,path,review):
    if review.get(video_id,{}).get('status')!='reviewed-safe':raise ValueError('Video is not cleared.')
    with Path(path).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=review[video_id]['sha256']:raise ValueError('Video file differs from reviewed hash.')
