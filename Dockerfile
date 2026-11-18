# ACTIVA in a container: ./run.sh <input.mp4> <output.json>, CPU by default.
#
#   docker build -t activa .
#   docker run --rm --network none -v "$PWD/videos:/in:ro" -v "$PWD/out:/out" \
#       activa /in/<video_id>.mp4 /out/<video_id>.json
#
# The build downloads the model weights once from the repo's `weights-v1` release
# (scripts/setup_weights.py checks every SHA-256). After that the container needs no
# network: the run above uses --network none.
#
# GPU (optional, needs the NVIDIA Container Toolkit): build with the CUDA torch wheel,
#   docker build -t activa-gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu132 .
# then add `--gpus all` to `docker run`. Device selection is automatic (ACTIVA_DEVICE=auto).
#
# FFmpeg here is Debian 12's 5.1.x; ACTIVA supports FFmpeg 4.4+ (README).

FROM python:3.13-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch/torchvision first from the chosen index (CPU by default), then the rest
# of the pinned requirements, so pip doesn't pull the much larger default CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX" torch==2.15.0 torchvision==0.29.2 \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# One-time weight download (~1.8 GB, SHA-256 checked); the image then runs offline.
RUN python scripts/setup_weights.py \
 && python scripts/setup_weights.py --check

# Default mode: a failing stage falls back and a valid JSON is still written. Add
# `-e ACTIVA_STRICT=1` to fail instead (exit 6), as we did for our own outputs/.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["./run.sh"]
