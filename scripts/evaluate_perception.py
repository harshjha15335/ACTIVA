#!/usr/bin/env python3
"""Leakage-gated held-out development evaluation; no training or tuning.

The sample-grid importance-ranking F1 proxy is not original-frame or final
highlight F1. Inputs use ACTIVA adapter schemas, not guessed organizer formats.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import numpy as np
from fpvsum_data import (load_inventory, load_processed, load_aliases, load_review,
                        assert_safe_split, assert_video_hash, resample_processed)
from perception_cache import file_sha256
from perception_metrics import ranking_f1, category_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['study', 'inventory', 'annotations', 'review', 'official-manifest', 'weights', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--categories')
    parser.add_argument('--aliases')
    args = parser.parse_args()
    started = time.perf_counter()
    study = json.loads(Path(args.study).read_text())
    if (study.get('format') != 'activa-m2-study-v1'
            or not isinstance(study.get('validation'), list) or not study['validation']):
        raise ValueError('Expected an explicit nonempty activa-m2-study-v1 validation list.')
    inventory = load_inventory(args.inventory)
    review = load_review(args.review, args.official_manifest)
    ids = [entry['video_id'] for entry in study['validation']]
    assert_safe_split(study.get('training_ids', []), ids, review, inventory)
    annotations = load_processed(args.annotations, inventory, load_aliases(args.aliases))
    os.environ['ACTIVA_CLIP_WEIGHTS'] = str(Path(args.weights).resolve())
    if args.categories:
        os.environ['ACTIVA_CATEGORIES'] = str(Path(args.categories).resolve())

    from perception_options import settings, IMPORTANCE_METHODS, POOLING_METHODS
    from category_vocab import load_categories
    from clip_features import extract_clip_features
    from importance import predict_importance
    from category import classify_video, PROMPTS
    from video_io import inspect_video
    cfg = settings()
    vocabulary = load_categories(cfg)
    truths = []
    predictions = {(mode, pool): [] for mode in IMPORTANCE_METHODS for pool in POOLING_METHODS}
    baseline_results = {mode: [] for mode in IMPORTANCE_METHODS}
    per_video = []
    model_ids = set()
    for entry in study['validation']:
        name = entry['video_id']
        assert_video_hash(name, entry['path'], review)
        feature_started = time.perf_counter()
        info = inspect_video(entry['path'])
        clip = extract_clip_features(info)
        model_ids.add(clip.model_name)
        if len(model_ids) != 1:
            raise ValueError('All evaluated videos must use the same checkpoint identity.')
        extraction_seconds = time.perf_counter() - feature_started
        with np.load(entry['source_times_npz'], allow_pickle=False) as data:
            source_times = data['times']
        targets = resample_processed(annotations[name], source_times, clip.times, info.duration)
        label = inventory[name].category
        if label not in vocabulary.labels:
            if not study.get('category_map_reviewed_by'):
                raise ValueError('Exact category mapping requires explicit human review.')
            label = study.get('category_map', {}).get(label)
        if label not in vocabulary.labels:
            raise ValueError(f'Unmapped evaluation category for {name}.')
        truths.append(label)
        record = {'video_id': name, 'samples': len(clip.times), 'model': clip.model_name,
                  'extraction_seconds': extraction_seconds, 'baselines': {}, 'pooling_seconds': {}}
        for mode in IMPORTANCE_METHODS:
            os.environ['ACTIVA_IMPORTANCE'] = mode
            baseline_started = time.perf_counter()
            importance = predict_importance(info, clip)
            inference_seconds = time.perf_counter() - baseline_started
            result = {**ranking_f1(targets, importance.values), 'runtime_seconds': inference_seconds}
            baseline_results[mode].append(result)
            record['baselines'][mode] = result
            for pool in POOLING_METHODS:
                os.environ['ACTIVA_POOLING'] = pool
                pool_started = time.perf_counter()
                predictions[(mode, pool)].append(classify_video(info, clip, importance).label)
                record['pooling_seconds'][f'{mode}/{pool}'] = time.perf_counter() - pool_started
        per_video.append(record)
    summary = {
        mode: {**{key: float(np.mean([row[key] for row in values]))
                   for key in ['precision', 'recall', 'f1']},
               'total_runtime_seconds': sum(row['runtime_seconds'] for row in values),
               'video_count': len(values)}
        for mode, values in baseline_results.items()
    }
    result = {
        'format': 'activa-m2-heldout-development-v1', 'official_metric': False,
        'purpose': study.get('purpose', 'reviewed-development-study'),
        'metric_warning': 'Sample-grid ranking F1, not original-frame F1 or final M3 highlights; no interval budget enforcement.',
        'scorer_version': 'm2-sample-grid-ranking-v1',
        'protocol': {'top_ratio': .15, 'ties': 'earlier sample first', 'averaging': 'macro over identical validation videos',
                     'timestamp_alignment': 'duration-weighted original-frame bins', 'budget_enforced': False},
        'dataset_version': study.get('dataset_version', 'See exact input artifact hashes'),
        'input_artifact_sha256': {key: file_sha256(getattr(args, key))
                                 for key in ['study', 'inventory', 'annotations', 'review', 'official_manifest']},
        'safe_split': {'training_ids': study.get('training_ids', []), 'validation_ids': ids},
        'validation_ids': ids, 'sample_count': len(ids),
        'canonical_vocabulary': vocabulary.canonical, 'vocabulary_source': vocabulary.source,
        'prompt_templates': list(PROMPTS),
        'prompt_templates_sha256': hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest(),
        'baseline_summary': summary,
        'baseline_macro_ranking_f1': {mode: values['f1'] for mode, values in summary.items()},
        'category_pooling': {f'{mode}/{pool}': category_metrics(vocabulary.labels, truths, values)
                             for (mode, pool), values in predictions.items()},
        'per_video': per_video, 'model': clip.model_name,
        'total_runtime_seconds': time.perf_counter() - started,
        'reproduction_command': shlex.join([sys.executable, *sys.argv]), 'selection_frozen': False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote development report for {len(ids)} reviewed videos; no official score claimed.', file=sys.stderr)


if __name__ == '__main__':
    main()
