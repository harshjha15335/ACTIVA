"""Positive toy orchestration only, never presented as model-quality evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from contracts import VideoInfo, ClipFeats, Importance, CategoryResult
from perception_cache import file_sha256


def test_full_evaluation_workflow_on_explicit_toy_data(tmp_path, monkeypatch):
    for key in ['ACTIVA_CLIP_WEIGHTS', 'ACTIVA_CATEGORIES', 'ACTIVA_IMPORTANCE', 'ACTIVA_POOLING']:
        monkeypatch.setenv(key, '')
    monkeypatch.setenv('ACTIVA_IMPORTANCE', 'novelty')
    monkeypatch.setenv('ACTIVA_POOLING', 'mean')
    # Fake input bytes/model outputs are intentional unit fixtures. They are not
    # real pretrained or protected evaluation data; real inference is tested elsewhere.
    video = tmp_path / 'toy.bin'; video.write_bytes(b'toy fixture, not an actual video')
    manifest = tmp_path / 'toy-exclusions.json'; manifest.write_text('{"purpose":"unit-fixture"}')
    inventory = tmp_path / 'inventory.csv'
    inventory.write_text('video_name,youtube_id,category,split\nToy,source-toy,ToyCategory,labeled\n')
    annotations = tmp_path / 'annotations.mat'
    savemat(annotations, {'gt_data': [{'Name': 'Toy', 'Score': [0., 1., 0., 1.]}]})
    times = tmp_path / 'source-times.npz'; np.savez(times, times=np.array([0, .25, .5, .75]))
    review = tmp_path / 'toy-review.json'
    review.write_text(json.dumps({'format': 'activa-leakage-review-v1', 'state': 'reviewed',
        'reviewed_by': 'unit-test-fixture-only', 'reviewed_at': 'fixture-date',
        'official_manifest_sha256': file_sha256(manifest),
        'videos': [{'video_id': 'Toy', 'status': 'reviewed-safe', 'sha256': file_sha256(video), 'evidence': 'synthetic unit data'}]}))
    study = tmp_path / 'study.json'
    study.write_text(json.dumps({'format': 'activa-m2-study-v1', 'purpose': 'TOY-INFRASTRUCTURE-TEST-NOT-QUALITY',
        'dataset_version': 'unit-fixture-v1', 'training_ids': [],
        'validation': [{'video_id': 'Toy', 'path': str(video), 'source_times_npz': str(times)}]}))
    labels = tmp_path / 'labels.json'; labels.write_text('["ToyCategory"]')
    import video_io, clip_features, importance, category, category_vocab
    monkeypatch.setattr(video_io, 'inspect_video', lambda p: VideoInfo('Toy', p, 1., 4., 8, 8))
    monkeypatch.setattr(clip_features, 'extract_clip_features', lambda _: ClipFeats(np.array([0., .5]), np.eye(2, dtype=np.float32), 'toy-model-not-pretrained'))
    monkeypatch.setattr(importance, 'predict_importance', lambda _info, c: Importance(c.times.copy(), np.array([.5, .5])))
    monkeypatch.setattr(category, 'classify_video', lambda *_: CategoryResult('ToyCategory', {'ToyCategory': 0.}))
    from types import SimpleNamespace
    monkeypatch.setattr(category_vocab, 'load_categories', lambda _: SimpleNamespace(labels=('ToyCategory',), canonical=False, source='toy-fixture'))
    output = tmp_path / 'result.json'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_perception.py'
    spec = importlib.util.spec_from_file_location('evaluation_workflow_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    arguments = ['evaluate_perception.py']
    for key, value in [('study', study), ('inventory', inventory), ('annotations', annotations),
                       ('review', review), ('official-manifest', manifest), ('weights', tmp_path / 'toy-unused.pt'),
                       ('categories', labels), ('output', output)]:
        arguments.extend(['--' + key, str(value)])
    monkeypatch.setattr(sys, 'argv', arguments)
    module.main()
    result = json.loads(output.read_text())
    assert result['purpose'] == 'TOY-INFRASTRUCTURE-TEST-NOT-QUALITY'
    assert result['sample_count'] == 1 and not result['official_metric']
    assert len(result['baseline_summary']) == 4 and len(result['category_pooling']) == 12
    assert result['dataset_version'] == 'unit-fixture-v1'
    assert result['input_artifact_sha256']['inventory'] == file_sha256(inventory)
    assert result['model'] == 'toy-model-not-pretrained'
    assert result['total_runtime_seconds'] >= 0 and 'evaluate_perception.py' in result['reproduction_command']
    for baseline in result['baseline_summary'].values():
        assert {'precision', 'recall', 'f1', 'total_runtime_seconds', 'video_count'} <= baseline.keys()
