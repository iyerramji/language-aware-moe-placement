#!/usr/bin/env bash
set -euo pipefail
cd ~/sarvam-cap
python3 -m venv .venv || {
 sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
 sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
 python3 -m venv .venv
}
source .venv/bin/activate
python -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q numpy==2.2.6 transformers==4.57.2 pyarrow==19.0.1 sentencepiece==0.2.1 safetensors==0.6.2 huggingface-hub==0.36.0
python -c 'import torch; assert torch.cuda.device_count() == 4; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
export PYTHONUNBUFFERED=1
exec timeout 52m python massed_sarvam_cap.py
