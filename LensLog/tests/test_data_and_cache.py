import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.io import savemat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from fpvsum_data import (VideoRecord,Annotation,load_inventory,load_processed,load_raw,load_aliases,resample_processed,load_review,assert_safe_split,assert_video_hash)
from contracts import VideoInfo,ClipFeats,Importance
from perception_cache import write_cache,read_cache,file_sha256
from leakage_fingerprints import possible_content_match

@pytest.fixture
def inventory():
    return {'A':VideoRecord('A','sourceA','Cat','labeled'),'B':VideoRecord('B','sourceB','Cat','labeled'),'U':VideoRecord('U','sourceU','Cat','unlabeled')}

def test_inventory_labeled_unlabeled_and_duplicates(tmp_path):
    path=tmp_path/'inventory.csv';path.write_text('video_name,youtube_id,category,split\nA,id,Cat,labeled\nU,id2,Cat,unlabeled\n')
    rows=load_inventory(path);assert rows['U'].split=='unlabeled'
    path.write_text(path.read_text()+'A,id3,Cat,labeled\n')
    with pytest.raises(ValueError,match='Duplicate'):load_inventory(path)

def test_processed_strict_ids_missing_and_reviewed_alias(tmp_path,inventory):
    path=tmp_path/'processed.mat'
    savemat(path,{'gt_data':[{'Name':'typoA','Score':np.array([0.,.5,1.])}]})
    with pytest.raises(ValueError,match='Unresolved'):load_processed(path,inventory)
    with pytest.raises(ValueError,match='Missing'):load_processed(path,inventory,{'typoA':'A'})
    rows=load_processed(path,inventory,{'typoA':'A'},require_all=False)
    np.testing.assert_array_equal(rows['A'].scores,[0,.5,1]);assert rows['A'].original_name=='typoA'
    aliases=tmp_path/'aliases.json';aliases.write_text('{"aliases":{"typoA":"A"}}')
    with pytest.raises(ValueError,match='reviewed'):load_aliases(aliases)
    aliases.write_text(json.dumps({'format':'activa-reviewed-id-aliases-v1','reviewed_by':'fixture-reviewer','evidence':'fixture proof','aliases':{'typoA':'A'}}))
    assert load_aliases(aliases)=={'typoA':'A'}

@pytest.mark.parametrize('score',[[0,np.nan],[0,1.1],[-.1,1]])
def test_processed_bad_scores_rejected(tmp_path,inventory,score):
    path=tmp_path/'bad.mat';savemat(path,{'gt_data':[{'Name':'A','Score':np.array(score)}]})
    with pytest.raises(ValueError,match='Score'):load_processed(path,inventory,require_all=False)

def test_raw_annotators_preserved_without_unsupported_averaging(tmp_path,inventory):
    path=tmp_path/'raw.mat';savemat(path,{'raw_data':[{'Name':'A','Category':'Cat','Annotator_id':1,'Label':[0,1,1]},{'Name':'A','Category':'Cat','Annotator_id':2,'Label':[1,0,1]}]})
    rows=load_raw(path,inventory);assert len(rows)==2
    np.testing.assert_array_equal(rows[0].labels,[0,1,1])
    savemat(path,{'raw_data':[{'Name':'A','Category':'Cat','Annotator_id':1,'Label':[0,1]},{'Name':'A','Category':'Cat','Annotator_id':2,'Label':[1]}]})
    with pytest.raises(ValueError,match='frame count'):load_raw(path,inventory)

def test_duration_weighted_annotation_resampling():
    annotation=Annotation('A',np.array([0.,1.,0.,1.]),'A')
    result=resample_processed(annotation,[0,.25,.5,.75],[0,.5],1.)
    np.testing.assert_allclose(result,[.5,.5])
    with pytest.raises(ValueError,match='PTS'):resample_processed(annotation,[0,.5],[0,.5],1.)
    with pytest.raises(ValueError):resample_processed(annotation,[0,.25,.5,.75],[0,1.],1.)

def test_review_bound_to_manifest_and_excludes_unresolved(tmp_path,inventory):
    manifest=tmp_path/'official-fixture.json';manifest.write_text('{}')
    path=tmp_path/'review.json';path.write_text('{}')
    with pytest.raises(ValueError,match='human-reviewed'):load_review(path,manifest)
    value={'format':'activa-leakage-review-v1','state':'reviewed','reviewed_by':'fixture-human','reviewed_at':'fixture-date','official_manifest_sha256':file_sha256(manifest),'videos':[{'video_id':'A','status':'reviewed-safe','sha256':'a'*64,'evidence':'fixture'}, {'video_id':'B','status':'unresolved'}]}
    path.write_text(json.dumps(value));review=load_review(path,manifest)
    assert_safe_split([],['A'],review,inventory)
    with pytest.raises(ValueError,match='not reviewed-safe'):assert_safe_split([],['B'],review,inventory)
    with pytest.raises(ValueError):assert_video_hash('A',manifest,review)
    manifest.write_text('changed')
    with pytest.raises(ValueError):load_review(path,manifest)

def test_source_hash_and_reviewed_content_duplicates_do_not_cross_splits(inventory):
    review={name:{'status':'reviewed-safe','sha256':name.lower()*64} for name in ['A','B']}
    assert_safe_split(['A'],['B'],review,inventory)
    review['B']['source_group']=review['A']['source_group']='same-reencoded-content'
    with pytest.raises(ValueError,match='source'):assert_safe_split(['A'],['B'],review,inventory)
    with pytest.raises(ValueError,match='overlapping'):assert_safe_split(['A'],['A'],review,inventory)

def sample_cache(tmp_path,name='input.mp4',payload=b'fixture'):
    source=tmp_path/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(payload)
    info=VideoInfo('same-name',str(source),1.,2.,32,32)
    clip=ClipFeats(np.array([0.,.5]),np.array([[1.,0],[0,1]],dtype=np.float32),'fixture-checkpoint')
    imp=Importance(clip.times.copy(),np.array([.25,.75],np.float32))
    return info,clip,imp

def test_cache_roundtrip_hash_identity_and_tamper(tmp_path):
    info,clip,imp=sample_cache(tmp_path)
    path=write_cache(tmp_path/'cache',info,clip,imp,{'importance':'fixture'})
    other,loaded,importance,manifest=read_cache(path,expected_model='fixture-checkpoint')
    np.testing.assert_array_equal(loaded.feats,clip.feats);np.testing.assert_array_equal(importance.times,clip.times)
    assert manifest['dimension']==2 and other.duration==1.
    with pytest.raises(ValueError,match='identity'):read_cache(path,expected_model='different')
    root=path.parent.parent;(root/manifest['clip']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):read_cache(path)

def test_cache_same_basename_different_input_never_collides(tmp_path):
    first=sample_cache(tmp_path,'a/input.mp4',b'first');second=sample_cache(tmp_path,'b/input.mp4',b'second')
    a=write_cache(tmp_path/'cache',*first,{'importance':'fixture'})
    b=write_cache(tmp_path/'cache',*second,{'importance':'fixture'})
    assert a!=b
    Path(first[0].path).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):read_cache(a)

def test_perceptual_candidates_do_not_mean_clearance():
    first={'format':'activa-video-fingerprint-v1','duration':10.,'width':640,'height':480,'times':[0,1,2],'dhash256':['0'*64]*3}
    second={**first,'video_id':'renamed'}
    assert possible_content_match(first,second)
    second={**second,'duration':50}
    assert not possible_content_match(first,second)

def test_upstream_fractional_raw_values_are_preserved(tmp_path,inventory):
    path=tmp_path/'fractional.mat';savemat(path,{'raw_data':[{'Name':'A','Category':'Cat','Annotator_id':1,'Label':[0.,.0359,.999,1.]}]})
    rows=load_raw(path,inventory)
    np.testing.assert_array_equal(rows[0].labels,[0.,.0359,.999,1.])
    assert rows[0].labels.dtype==np.float64

def test_raw_category_alias_requires_explicit_mapping(tmp_path,inventory):
    path=tmp_path/'category.mat';savemat(path,{'raw_data':[{'Name':'A','Category':'CAT','Annotator_id':1,'Label':[0.,1.]}]})
    with pytest.raises(ValueError,match='category mismatch'):load_raw(path,inventory)
    rows=load_raw(path,inventory,category_aliases={'CAT':'Cat'})
    assert rows[0].category=='CAT'  # Original category is retained, not silently normalized.

def test_evaluation_cli_blocks_before_model_when_review_missing(tmp_path):
    import subprocess
    study=tmp_path/'study.json';study.write_text(json.dumps({'format':'activa-m2-study-v1','validation':[{'video_id':'A'}]}))
    inventory=tmp_path/'inventory.csv';inventory.write_text('video_name,youtube_id,category,split\nA,sourceA,Cat,labeled\n')
    official=tmp_path/'official-fixture.json';official.write_text('{}')
    review=tmp_path/'review.json';review.write_text('{"state":"pending"}')
    output=tmp_path/'metrics.json'
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/evaluate_perception.py'),'--study',str(study),'--inventory',str(inventory),'--annotations',str(tmp_path/'missing.mat'),'--review',str(review),'--official-manifest',str(official),'--weights',str(tmp_path/'missing.safetensors'),'--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0 and 'human-reviewed' in result.stderr
    assert not output.exists()

def test_reencoded_video_fingerprint_only_creates_review_candidate(tmp_path):
    import subprocess
    from leakage_fingerprints import fingerprint_video
    original=tmp_path/'original.mp4';renamed=tmp_path/'renamed.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=96x64:r=10:d=3','-c:v','libx264',str(original)],check=True)
    subprocess.run(['ffmpeg','-v','error','-i',str(original),'-vf','scale=192:128','-c:v','libx264','-crf','28',str(renamed)],check=True)
    first,second=fingerprint_video(original),fingerprint_video(renamed)
    assert first['sha256']!=second['sha256']
    assert possible_content_match(first,second)
    # A candidate still has no reviewed-safe decision; it must never authorize a split.
    with pytest.raises(ValueError):assert_safe_split([],['original'],{}, {'original':VideoRecord('original','source','C','labeled')})
