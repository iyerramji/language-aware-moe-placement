"""Run the full-model Sarvam pilot on one four-L40S Modal worker."""
import json
import sys
from pathlib import Path

import modal

CACHE = "sarvam-placement-pilot-checkpoint"
HERE = Path(__file__).resolve().parent
MAPS = HERE.parent / "results/gpu-pilot/2026-11-20-cap/cap-maps.json"
app = modal.App("sarvam-cap-ablation")
volume = modal.Volume.from_name(CACHE)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.2.6", "transformers==4.57.2", "pyarrow==19.0.1",
                 "sentencepiece==0.2.1", "safetensors==0.6.2")
    .add_local_file(HERE / "sarvam_layer.py", "/opt/pilot/sarvam_layer.py")
    .add_local_file(HERE / "sarvam_decode.py", "/opt/pilot/sarvam_decode.py")
    .add_local_file(HERE / "sarvam_cap_runner.py", "/opt/pilot/sarvam_cap_runner.py")
    .add_local_file(HERE / "sarvam_cap_decode.py", "/opt/pilot/sarvam_cap_decode.py")
    .add_local_file(MAPS, "/opt/pilot/frozen-maps.json")
)


@app.function(image=image, gpu="L40S:4", cpu=6, memory=16384, timeout=1800, retries=0,
              volumes={"/checkpoint": volume})
def benchmark() -> str:
    sys.path.insert(0, "/opt/pilot")
    from sarvam_cap_runner import benchmark as run_benchmark

    result = run_benchmark(Path("/checkpoint/sarvam"), Path("/opt/pilot/frozen-maps.json"))
    return json.dumps(dict(result, platform="modal"))
