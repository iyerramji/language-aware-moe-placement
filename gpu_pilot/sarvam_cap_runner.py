"""Shared benchmark harness for four L40S GPUs."""
import hashlib
import io
import json
import os
import random
import subprocess
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"

DATA_REV = "1a45e707ea4b5ea3d6c71341f18bfca6a6e356a3"
LANG = dict(en="eng", bn="ben", gu="guj", hi="hin", kn="kan", ml="mal",
            mr="mar", pa="pan", ta="tam", te="tel")


def fetch_prompts(destination):
    import pyarrow.parquet as pq

    batches = [[], [], []]
    base = f"https://huggingface.co/datasets/gsarti/flores_101/resolve/{DATA_REV}/"
    for language, code in LANG.items():
        with urllib.request.urlopen(f"{base}{code}/dev/0000.parquet", timeout=25) as response:
            text = pq.read_table(io.BytesIO(response.read()), columns=["sentence"])["sentence"]
        for repeat, batch in enumerate(batches):
            for row in (250 + 2 * repeat, 251 + 2 * repeat):
                batch.append({"language": language, "id": f"flores-{language}-{row:04}",
                              "text": text[row].as_py()})
    for i, batch in enumerate(batches):
        random.Random(20261119 + i).shuffle(batch)
    destination.write_text(json.dumps(batches))
    return [[row["id"] for row in batch] for batch in batches]


def benchmark(folder: Path, inputs: Path) -> dict:
    import torch

    marker = json.loads((folder / "stage-complete.json").read_text())
    assert marker["checkpoint_revision"] == REVISION
    assert torch.cuda.device_count() == 4
    devices = [torch.cuda.get_device_name(rank) for rank in range(4)]
    assert all("L40S" in name for name in devices)
    peer = [[i != j and torch.cuda.can_device_access_peer(i, j) for j in range(4)] for i in range(4)]
    topology = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10)
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        prompt_file = scratch / "prompts.json"
        picks = fetch_prompts(prompt_file)
        print(f"Four L40S devices ready. Loaded {sum(map(len, picks))} held-out FLORES prompts.", flush=True)
        output = scratch / "partial.json"
        env = dict(os.environ, OMP_NUM_THREADS="1", NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH",
                   NCCL_DEBUG_FILE=str(scratch / "nccl-%p.log"))
        exit_code = None
        error = None
        try:
            process = subprocess.run([
                "torchrun", "--nnodes=1", "--nproc-per-node=4", "--master-addr=127.0.0.1",
                "--master-port=29650", str(Path(__file__).with_name("sarvam_cap_decode.py")),
                str(folder), str(prompt_file), str(inputs), str(output),
            ], env=env, timeout=1590)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            error = "Decode process exceeded the execution limit"
        transport = [line for file in scratch.glob("nccl-*.log") for line in file.read_text().splitlines()
                     if "via SHM" in line or "via P2P" in line or "via NET" in line]
        results = json.loads(output.read_text()) if output.exists() else []
    return {
        "status": "completed" if exit_code == 0 and len(results) == 27 else "incomplete",
        "exit_code": exit_code, "error": error,
        "checkpoint": marker, "dataset_revision": DATA_REV, "prompt_ids": picks,
        "placement_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "devices": devices, "cuda_peer_access": peer, "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
        "nvidia_smi_topology": {"status": topology.returncode, "stdout": topology.stdout, "stderr": topology.stderr},
        "nccl_routes": transport, "runs": results,
    }
