# M2 local CLIP weights

## Explicit validated development candidate

Architecture: OpenCLIP ViT-B-32 with QuickGELU;512 dimensions; image224center crop,
OpenAI CLIP normalization. Checkpoint: **LAION400M_e32**, not OpenAI's trained
checkpoint. Source: OpenCLIP official release v0.2-weights:

https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt

Published filename SHA256 prefix: `46683a32` (verified).
Full raw-file SHA256 computed after that verification:
`46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa`.
Converted safetensors SHA256 for the installed local artifact:
`9bd6bea2e0b7f343b17347af1854d7fd2667a227b0997801e5d297d46808854b`.

The release/installed OpenCLIP metadata provides an8hexprefix, not an independently
published full SHA256. Verification status is therefore published-prefix verified
plus full local hashes recorded; do not overstate it as published-full-hash parity.
The official source was fetched over verified TLS; checkpoint was loaded with
weights_only=True and exported to safetensors. No random fallback is present.

```sh
python scripts/prepare_weights.py --preset laion400m_e32
```

Writes weights/ViT-B-32-laion400m_e32.pt,.safetensors and .manifest.json. Existing
archives are verified; incomplete downloads,wrong hash/format and conflicting
converted assets are rejected. Repeating setup on this machine passed manifest/
checksum verification without a second download. The manifest is checked against
architecture/full converted hash at inference. File changes invalidate the model
cache. Every ClipFeats.model_name records architecture/sha256:<actualweightfile>.

OpenCLIP's repository LICENSE is MIT:
https://github.com/mlfoundations/open_clip/blob/main/LICENSE
Review model-specific distribution and organizer restrictions with M4 before
submission. Team approval/final backbone/checkpoint freeze is pending. Models and
videos are ignored by Git; distribute through approved environment/asset storage,
not commits. The license/reference does not establish dataset or organizer terms.

## Original OpenAI route and manual installation

Original ViT-B/32 SHA256:
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
The OpenAI preset retains its original URL and full checksum verification:

```sh
python scripts/prepare_weights.py --preset openai
python scripts/prepare_weights.py --preset openai --local-source /path/to/original/ViT-B-32.pt
```

Its host openaipublic.azureedge.net remains denied by current runtime policy;
a proposed allow rule is saved but unapplied. The manual route requires the
original trusted archive and verifies the exact full hash. Do not rename an
unrelated checkpoint to bypass verification. --expected-sha256 accepts an optional
independently trusted64hex digest, never a guessed replacement.

Default shared config still references original weights/ViT-B-32.safetensors.
Select the validated alternative explicitly with --weights or ACTIVA_CLIP_WEIGHTS.
Inference never downloads and cannot infer that arbitrary supplied weights are
pretrained. This script is Python-based for Linux/Windows; FFmpeg and runtime
compatibility must be installed/tested separately. Windows/GPU were not tested.
