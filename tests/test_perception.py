"""Algorithm, real-decoder and offline randomly initialized CLIP plumbing tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from contracts import ClipFeats, Importance, VideoInfo
from importance import predict_importance, rank_normalize, validate_clip
from category import classify_video, pool_features
from video_io import inspect_video, iter_frames
from tcn import ImportanceTCN, ranking_mse_loss


@pytest.fixture(autouse=True)
def clean_config(monkeypatch):
    for name in list(os.environ):
        if name.startswith('ACTIVA_'):
            monkeypatch.delenv(name)
    torch.set_num_threads(2)


@pytest.fixture
def sample():
    info = VideoInfo('test', '', 120.0, 10, 64, 48)
    times = np.arange(0, 120, 0.5)
    feats = np.zeros((240, 512), np.float32)
    feats[:, 0] = 1
    return info, ClipFeats(times, feats, 'test-only')


@pytest.mark.parametrize('mode', ['constant', 'novelty'])
def test_black_equivalent_embeddings_preserve_ties(sample, monkeypatch, mode):
    info, clip = sample
    monkeypatch.setenv('ACTIVA_IMPORTANCE', mode)
    first, second = predict_importance(info, clip), predict_importance(info, clip)
    np.testing.assert_array_equal(first.times, clip.times)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.values, np.full(240, 0.5))
    assert first.values.dtype == np.float32
    first.times[0] = 10
    assert clip.times[0] == 0


def test_novel_moment_scores_above_constant_context(sample):
    info, clip = sample
    clip.feats[120] = 0
    clip.feats[120, 1] = 1
    values = predict_importance(info, clip).values
    assert values[120] > values[20]
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()


@pytest.mark.parametrize('times,features', [([0, 0], [[1], [1]]), ([1, 0], [[1], [1]]), ([np.nan], [[1]]), ([-1], [[1]]), ([120], [[1]]), ([0], [[np.nan]]), ([0], [[0]]), ([0], [[1], [1]]), ([], [])])
def test_invalid_clip_rejected(sample, times, features):
    info, _ = sample
    with pytest.raises(ValueError):
        validate_clip(info, ClipFeats(np.asarray(times), np.asarray(features), 'bad'))


def test_average_ranks_and_singleton():
    np.testing.assert_allclose(rank_normalize([1, 1, 2, 3]), [0.25, 0.25, 0.625, 0.875])
    assert rank_normalize([5])[0] == 0.5


@pytest.mark.parametrize('method', ['mean', 'weighted', 'top'])
def test_pooling_zero_importance_is_finite(sample, method):
    _, clip = sample
    result = pool_features(clip.feats, np.zeros(240), method)
    assert np.isfinite(result).all()
    assert np.linalg.norm(result) == pytest.approx(1)


def test_category_uses_exact_configured_labels(sample, monkeypatch, tmp_path):
    import category
    info, clip = sample
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(['test activity A', 'test activity B']))
    monkeypatch.setenv('ACTIVA_CATEGORIES', str(path))
    def text(prompts, cfg):
        vectors = np.zeros((len(prompts), 512), np.float32)
        vectors[:5, 0], vectors[5:, 1] = 1, 1
        return vectors
    monkeypatch.setattr(category, 'text_features', text)
    imp = Importance(clip.times, np.zeros(240))
    assert classify_video(info, clip, imp).label == 'test activity A'
    with pytest.raises(ValueError, match='timestamps'):
        classify_video(info, clip, Importance(clip.times + 0.1, imp.values))


def test_missing_categories_and_weights_fail(sample):
    import clip_backend
    info, clip = sample
    with pytest.raises(ValueError, match='ACTIVA_CATEGORIES'):
        classify_video(info, clip, Importance(clip.times, np.zeros(240)))
    with pytest.raises(FileNotFoundError, match='Local CLIP weights'):
        clip_backend.load_backend('ViT-B-32', '/nonexistent/weights.pt', 'cpu')


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / 'sample.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=64x48:r=10:d=2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


def test_real_frame_grid_and_rgb(short_video):
    info = inspect_video(short_video)
    frames = list(iter_frames(short_video, 2))
    assert info.duration == pytest.approx(2)
    np.testing.assert_allclose([t for t, _ in frames], [0, 0.5, 1, 1.5])
    assert frames[0][1].shape == (48, 64, 3)
    assert frames[0][1].dtype == np.uint8


def test_variable_frame_rate_uses_decoded_pts(tmp_path):
    path = tmp_path / 'vfr.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=64x48:r=10:d=3', '-vf', 'select=not(mod(n\\,3))', '-fps_mode', 'vfr', '-c:v', 'libx264', str(path)], check=True)
    times = np.array([t for t, _ in iter_frames(path, 2)])
    assert np.all(np.diff(times) > 0)
    assert np.allclose(times * 10, np.round(times * 10))
    assert np.any(~np.isclose(times * 2, np.round(times * 2)))


def test_rotation_changes_dimensions(short_video, tmp_path):
    rotated = tmp_path / 'rotated.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-display_rotation:v:0', '90', '-i', str(short_video), '-c', 'copy', str(rotated)], check=True)
    original = list(iter_frames(short_video, 2))[0][1]
    info = inspect_video(rotated)
    frame = list(iter_frames(rotated, 2))[0][1]
    assert abs(info.rotation) == 90
    np.testing.assert_array_equal(frame, np.rot90(original, k=info.rotation // 90))
    assert frame.shape == (64, 48, 3)


def test_two_minute_black_motion_cli_and_invalid_input(tmp_path):
    path = tmp_path / 'black.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=32x32:r=2:d=120', '-c:v', 'libx264', str(path)], check=True)
    output = tmp_path / 'out.json'
    script = Path(__file__).resolve().parents[1] / 'code' / 'perception_cli.py'
    command = [sys.executable, str(script), str(path), str(output), '--mode', 'motion-only', '--cache-dir', str(tmp_path / 'cache')]
    environment = {**os.environ, 'ACTIVA_PYTHON': sys.executable}
    subprocess.run(command, env=environment, check=True)
    data = json.loads(output.read_text())
    assert len(data['times']) == 240
    assert data['category'] is None
    np.testing.assert_array_equal(data['importance'], np.full(240, 0.5))
    cached = np.load(tmp_path / 'cache' / 'importance' / 'black.npz')
    np.testing.assert_array_equal(cached['values'], data['importance'])
    command[2] = str(tmp_path / 'missing.mp4')
    assert subprocess.run(command, env=environment, capture_output=True).returncode != 0


def test_tcn_forward_backward_without_training_data():
    torch.manual_seed(0)
    model = ImportanceTCN(dimension=16)
    x = torch.randn(2, 12, 16)
    predicted = model(x)
    loss = ranking_mse_loss(predicted, torch.rand(2, 12))
    loss.backward()
    assert predicted.shape == (2, 12)
    assert torch.isfinite(loss)
    assert model.output.weight.grad is not None
    model.eval()
    torch.testing.assert_close(model(x), model(x), rtol=0, atol=0)


def test_real_openclip_offline_with_tiny_random_checkpoint(short_video, tmp_path, monkeypatch):
    """Tests library execution only; random weights establish no semantic quality."""
    import open_clip
    import socket
    from clip_backend import load_backend
    from clip_features import extract_clip_features
    configuration = {
        'embed_dim': 16,
        'vision_cfg': {'image_size': 32, 'layers': 1, 'width': 32, 'patch_size': 16, 'head_width': 16},
        'text_cfg': {'context_length': 77, 'vocab_size': 49408, 'width': 32, 'heads': 2, 'layers': 1},
    }
    config_path = tmp_path / 'Tiny-Test-CLIP.json'
    config_path.write_text(json.dumps(configuration))
    open_clip.add_model_config(config_path)
    model = open_clip.create_model('Tiny-Test-CLIP', pretrained=None)
    weights = tmp_path / 'random-test-only.pt'
    torch.save(model.state_dict(), weights)
    labels = tmp_path / 'labels.json'
    labels.write_text('["test A", "test B"]')
    monkeypatch.setenv('ACTIVA_CLIP_MODEL', 'Tiny-Test-CLIP')
    monkeypatch.setenv('ACTIVA_CLIP_WEIGHTS', str(weights))
    monkeypatch.setenv('ACTIVA_CATEGORIES', str(labels))
    monkeypatch.setenv('ACTIVA_BATCH_SIZE', '2')
    def deny_network(*args, **kwargs):
        raise AssertionError('Inference attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', deny_network)
    load_backend.cache_clear()
    info = inspect_video(short_video)
    first = extract_clip_features(info)
    second = extract_clip_features(info)
    assert first.feats.shape == (4, 16)
    assert first.feats.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(first.feats, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(first.feats, second.feats)
    for mode in ['constant', 'novelty', 'motion', 'text']:
        monkeypatch.setenv('ACTIVA_IMPORTANCE', mode)
        imp = predict_importance(info, first)
        assert np.isfinite(imp.values).all()
        for pooling in ['mean', 'weighted', 'top']:
            monkeypatch.setenv('ACTIVA_POOLING', pooling)
            result = classify_video(info, first, imp)
            assert result.label in ['test A', 'test B']
    load_backend.cache_clear()


def test_overlap_audit_never_marks_unknown_videos_safe(tmp_path):
    root = Path(__file__).resolve().parents[1]
    public, evaluation, output = (tmp_path / name for name in ['public.csv', 'evaluation.csv', 'audit.json'])
    public.write_text('video_id,sha256\nmatched,' + 'a' * 64 + '\nrenamed,' + 'b' * 64 + '\nunknown,' + 'c' * 64 + '\n')
    evaluation.write_text('video_id,sha256\nmatched,' + 'd' * 64 + '\noriginal,' + 'b' * 64 + '\n')
    subprocess.run([sys.executable, str(root / 'scripts/audit_ids.py'), str(public), str(evaluation), str(output)], check=True)
    data = json.loads(output.read_text())
    assert [v['status'] for v in data['videos']] == ['evaluation-overlap', 'evaluation-overlap', 'unresolved']
    assert not any(v['training_allowed'] for v in data['videos'])


def test_corrupt_weights_fail_checksum_without_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    weights = tmp_path / 'corrupt.pt'
    weights.write_bytes(b'not-a-model')
    result = subprocess.run([sys.executable, str(root / 'scripts/prepare_weights.py'), '--output', str(weights)], capture_output=True)
    assert result.returncode != 0
    assert b'checksum mismatch' in result.stderr
    assert weights.read_bytes() == b'not-a-model'


def test_verified_torchscript_conversion(tmp_path, monkeypatch):
    import importlib.util
    from safetensors.torch import load_file
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_weights.py'
    spec = importlib.util.spec_from_file_location('prepare_weights_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = torch.nn.Linear(3, 2).eval()
    traced = torch.jit.trace(original, torch.zeros(1, 3))
    archive = tmp_path / 'test-fixture.pt'
    traced.save(str(archive))
    # A trusted synthetic fixture, not a substitute for the real OpenAI checksum.
    monkeypatch.setattr(module, 'SHA256', module.digest(archive))
    module.convert_checkpoint(archive)
    converted = load_file(str(archive.with_suffix('.safetensors')))
    torch.testing.assert_close(converted['weight'], original.weight)
    torch.testing.assert_close(converted['bias'], original.bias)
