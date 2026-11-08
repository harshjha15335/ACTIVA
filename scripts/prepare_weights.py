#!/usr/bin/env python3
"""Explicit setup-time download, pinned to OpenAI's published SHA-256."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import tempfile

SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
URL = f"https://openaipublic.azureedge.net/clip/models/{SHA256}/ViT-B-32.pt"

def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def convert_checkpoint(output):
    # Only the checksum-verified upstream TorchScript archive reaches this step.
    # Current OpenCLIP uses weights-only loading; export safe tensor weights.
    import torch
    from safetensors.torch import save_file
    if digest(output) != SHA256:
        raise ValueError("Source checkpoint checksum mismatch before conversion.")
    archive = torch.jit.load(str(output), map_location="cpu").eval()
    state = {key: value.detach().cpu().contiguous() for key, value in archive.state_dict().items()
             if key not in {"input_resolution", "context_length", "vocab_size"}}
    converted = output.with_suffix(".safetensors")
    with tempfile.NamedTemporaryFile(dir=converted.parent, suffix=".safetensors.part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        save_file(state, str(temporary), metadata={"source_sha256": SHA256})
        temporary.replace(converted)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Converted {converted}: {digest(converted)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="weights/ViT-B-32.pt")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        if digest(output) != SHA256:
            raise ValueError("Existing checkpoint checksum mismatch; refusing to overwrite.")
        print(f"Verified {output}: {SHA256}")
        convert_checkpoint(output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".download.part", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with urllib.request.urlopen(URL, timeout=60) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if digest(temporary) != SHA256:
            raise ValueError("Downloaded checkpoint checksum mismatch.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified {output}: {SHA256}")
    convert_checkpoint(output)

if __name__ == "__main__":
    main()
