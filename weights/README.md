# CLIP weights

Setup command from the project root:

```sh
python scripts/prepare_weights.py
```

Model: OpenAI CLIP ViT-B/32, 512 image dimensions. Original source:
https://github.com/openai/CLIP/blob/main/clip/clip.py

Pinned download:
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

Expected SHA-256:
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`

OpenAI CLIP repository license: MIT; review the upstream model card for model
limitations and permitted use. Weight and dataset packaging must be checked
against organizer rules. Weights are ignored by Git. The setup command verifies
checksums and TLS; inference only accepts a pre-existing local checkpoint.

The model identifier reports the intended OpenAI configuration. Do not point
ACTIVA_CLIP_WEIGHTS to an unrelated checkpoint and expect compatible results.

After verifying the original TorchScript archive, setup exports its state dict to
`weights/ViT-B-32.safetensors`. This is the default inference checkpoint. The
converted file records the original hash as metadata, and its own hash is printed
by setup. Inference enables QuickGELU for the OpenAI ViT-B-32 architecture. The
conversion was tested with a synthetic TorchScript fixture; conversion/loading
of the full official checkpoint remains pending network access.
