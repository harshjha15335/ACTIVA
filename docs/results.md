# Executed development checks

Current machine: Linux x86-64, Python 3.12.14, CPU-only PyTorch 2.15.0+cpu,
OpenCLIP 4.2.1, PyAV 19.0.2 and FFmpeg 7.1.5. No GPU timings were measured.

From `/workspace/ACTIVA`:

```sh
/workspace/activa-venv/bin/python -m pytest -q
```

Result: **27 passed**, 4 dependency/deprecation warnings, no skipped tests.
The run took 5.45 seconds. Coverage includes:

- CLIP time-grid input validation, rank ties and novelty behavior.
- Exact configured category labels, zero-importance pooling and grid mismatch.
- Real MP4 decoding, presentation timestamps for variable-rate video and rotation.
- A 120-second black MP4: 240 motion samples, constant 0.5 importance, null category.
- Local offline OpenCLIP image/text execution using a tiny **randomly initialized**
  checkpoint: normalized float32 features, repeatability, all four baselines and
  all three pooling methods. Socket connections were blocked during inference.
  This tests execution, not semantic accuracy of pretrained CLIP.
- TCN forward/backward computation on synthetic tensors, not supervised training.
- Preliminary exact ID/hash overlap detection; unmatched clips remain unresolved.
- Corrupt-checkpoint rejection and TorchScript-to-safetensors conversion using a
  trusted synthetic fixture, not the unavailable full OpenAI archive.

A separate functional check generated a 5-second, 160x96, 12 fps test-pattern MP4:

```sh
ffmpeg -v error -f lavfi -i testsrc2=s=160x96:r=12:d=5 -c:v libx264 -y outputs/smoke.mp4
ACTIVA_PYTHON=/workspace/activa-venv/bin/python ./perceive.sh outputs/smoke.mp4 outputs/smoke-motion.json --mode motion-only --cache-dir dev_cache
```

Observed: 10 timestamped motion scores, range approximately 0.05–0.95; elapsed
application time 0.17 seconds. Category was null. Generated files are Git-ignored.
This tiny synthetic run is not a real-video performance benchmark.

`pip check` passed and reinstalling `requirements-dev.txt` found all requirements
satisfied. The official OpenAI checkpoint HEAD request returned proxy HTTP 403
for `openaipublic.azureedge.net`. The domain addition is saved in the environment
draft; saving it has not established runtime access. No pretrained checkpoint was
downloaded, so full ViT-B/32 extraction and category semantics remain unvalidated.

Unrun: official validator, pretrained full-size video tests, human importance F1,
category accuracy/confusion matrix, cross-validation, 2/5/10-minute pretrained
runtime benchmarks, GPU execution, clean-task restoration and full ACTIVA pipeline.
