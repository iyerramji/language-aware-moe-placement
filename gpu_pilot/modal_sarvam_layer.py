"""Run the controlled Sarvam layer test on a four-GPU PCIe worker."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import modal

app = modal.App("sarvam-layer-l40s")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy==2.2.6")
    .add_local_file(Path(__file__).with_name("sarvam_layer.py"), "/opt/pilot/sarvam_layer.py")
)


@app.function(gpu="L40S:4", image=image, timeout=480, retries=0)
def validate_layer() -> str:
    return run_layer("L40S")


@app.function(gpu="A10:4", image=image, timeout=480, retries=0)
def validate_layer_a10() -> str:
    return run_layer("A10")


def run_layer(gpu_name) -> str:
    import sys
    import torch
    sys.path.insert(0, "/opt/pilot")
    from sarvam_layer import REVISION, stage_weights

    assert torch.cuda.device_count() == 4
    devices = [torch.cuda.get_device_name(i) for i in range(4)]
    assert all(gpu_name in device for device in devices)
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10,
    )
    peer_access = [
        [i != j and torch.cuda.can_device_access_peer(i, j) for j in range(4)]
        for i in range(4)
    ]
    print(f"{devices}; topology status={topology.returncode}; P2P={peer_access}", flush=True)
    with TemporaryDirectory() as directory:
        hashes = stage_weights(Path(directory))
        print(f"Verified byte ranges for {len(hashes)} checkpoint tensors", flush=True)
        env = dict(os.environ, SARVAM_LAYER_WEIGHTS=directory, NCCL_DEBUG="INFO",
                   NCCL_DEBUG_SUBSYS="INIT,GRAPH")
        result = subprocess.run(
            ["torchrun", "--standalone", "--nproc-per-node=4", "/opt/pilot/sarvam_layer.py"],
            capture_output=True, text=True, timeout=160, env=env,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
    return json.dumps({
        "checkpoint_revision": REVISION, "decoder_layer_zero_indexed": 1,
        "devices": devices,
        "topology": {"exit_code": topology.returncode, "stdout": topology.stdout, "stderr": topology.stderr},
        "cuda_peer_access": peer_access, "torch": torch.__version__,
        "cuda": torch.version.cuda, "tensor_sha256": hashes,
        "byte_scope": "logical NCCL payload, with split-size metadata reported separately",
        "nccl_log": result.stdout + result.stderr,
        "input_scope": "fixed top-6 selections, real Sarvam expert and shared expert weights",
        "results": json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"primaries_only":'))),
    }, indent=2)
