#!/usr/bin/env python3
"""Validate real M2 through immutable M4/M1/M3 review copies, not live owner files."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vocab', help='Explicit vocabulary file; otherwise use M4 TEST fixture, never official labels')
    args = parser.parse_args()
    root = Path(args.review_root).resolve()
    sys.path.insert(0, str(root / 'code'))
    os.environ['ACTIVA_CLIP_WEIGHTS'] = str(Path(args.weights).resolve())
    os.environ['ACTIVA_VOCAB'] = str(Path(args.vocab).resolve() if args.vocab else root / 'tests/fixtures/dev_vocab.json')
    os.environ.pop('ACTIVA_CATEGORIES', None)
    os.environ['ACTIVA_REQUIRE_CANONICAL_VOCAB'] = '1'
    os.environ['ACTIVA_IMPORTANCE'] = 'novelty'
    os.environ['ACTIVA_POOLING'] = 'mean'
    os.environ['ACTIVA_DEVICE'] = 'cpu'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    def offline(*_, **__):
        raise RuntimeError('Network attempted during integration validation')
    socket.socket.connect = offline
    import numpy as np
    import main as pipeline
    import vocab
    import actions
    import optimizer
    from contracts import Importance
    from perception_cache import file_sha256
    start = time.perf_counter()
    result = pipeline.run(args.video)
    # A valid JSON alone could hide failed perception. Require real M2 stages.
    assert not set(result.fallbacks) & {'clip', 'importance', 'category'}, result.fallbacks
    assert result.clip.feats.shape[1] == 512 and result.clip.feats.dtype == np.float32
    assert np.allclose(np.linalg.norm(result.clip.feats, axis=1), 1, atol=1e-6)
    assert np.array_equal(result.clip.times, result.importance.times)
    labels = vocab.load()
    assert result.category.label in labels.categories
    pipeline.schema.validate_output(result.output, result.info.duration, labels)

    centres = actions.window_centres(actions.expected_num_windows(result.info.duration))
    np.testing.assert_allclose(actions.seconds_to_grid(centres), np.arange(len(centres)), atol=1e-12)
    detections = actions.PairDetections(np.array([0]), np.array([0]),
        np.array([[float(centres[0]), min(result.info.duration, float(centres[0]) + 1)]]), np.array([.8]))
    converted = actions.finalize(detections, result.info.duration, labels.verbs, labels.nouns)
    accepted = pipeline.sanitize_actions(converted, result.info, labels)
    assert len(accepted) == 1 and 0 <= accepted[0].start < accepted[0].end <= result.info.duration

    early = np.zeros(len(result.clip.times), dtype=np.float32)
    late = early.copy()
    early[:max(1, len(early) // 4)] = 1
    late[-max(1, len(late) // 4):] = 1
    uniform = np.full(len(early), .5, dtype=np.float32)
    intervals = {}
    for name, values in [('early', early), ('late', late), ('uniform', uniform)]:
        selected = optimizer.select_highlights(result.info, [], Importance(result.clip.times, values), result.clip)
        problem = pipeline.check_highlights(selected, result.info)
        assert problem is None, problem
        intervals[name] = [[float(h.start), float(h.end)] for h in selected]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fixture = output.with_name('m3-importance-controlled.npz')
    np.savez_compressed(fixture, times=result.clip.times, feats=result.clip.feats,
                       model_name=result.clip.model_name, early=early, late=late, uniform=uniform)
    fixture_metadata = output.with_name('m3-importance-controlled.json')
    fixture_metadata.write_text(json.dumps({
        'purpose': 'M3 importance-only acceptance fixture, not dataset quality',
        'fixture': fixture.name, 'sha256': file_sha256(fixture), 'video': asdict(result.info),
        'input_sha256': file_sha256(args.video), 'model': result.clip.model_name,
        'actions': [], 'observed_intervals': intervals,
        'expected_B0_behavior': 'With no action-coverage term and a real importance-only objective, selections should favor the early/late high-importance region respectively.',
        'current_limitation': 'Inspected selector is explicitly a uniform stub; no importance-only configuration exists.',
    }, indent=2) + '\n')
    report = {
        'kind': 'canonical-rehearsal-not-official-submission', 'review_root': str(root),
        'vocabulary_authority': 'supplied-unverified' if args.vocab else 'M4-TEST-FIXTURE-NOT-OFFICIAL',
        'offline_socket_connections_blocked': True, 'model': result.clip.model_name,
        'feature_shape': list(result.clip.feats.shape), 'feature_dtype': str(result.clip.feats.dtype),
        'fallback_stages': result.fallbacks, 'category': asdict(result.category),
        'canonical_schema_proxy_valid': True, 'official_validator_verified': False,
        'M1_actual_action_count': len(result.actions),
        'M1_helper_action_accepted_by_M4': asdict(accepted[0]),
        'M1_grid_roundtrip_pass': True,
        'M1_real_raw_video_action_alignment': 'UNVERIFIED: detect_actions is still an empty G0 stub; conversion/finalization helpers only were tested.',
        'M3_importance_sensitive': intervals['early'] != intervals['late'],
        'M3_intervals': intervals, 'M3_budget_checks_pass': True,
        'M3_fixture': str(fixture_metadata), 'stage_seconds': result.timings,
        'total_seconds': time.perf_counter() - start,
        'serialized_output': result.output,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
