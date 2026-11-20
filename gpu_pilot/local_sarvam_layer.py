"""Run the same controlled layer fixture locally with Gloo on CPU."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist

from sarvam_layer import REVISION, measure, stage_weights


def worker():
    rank = int(os.environ["RANK"])
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    report = {}
    for use_copies in (False, True):
        result = measure(Path(os.environ["SARVAM_LAYER_WEIGHTS"]), rank, torch.device("cpu"), use_copies)
        rows = [None] * 4
        dist.all_gather_object(rows, result)
        remote_calls = 10 if use_copies else 18
        dispatch = sum(row["dispatch_logical_bytes"] for row in rows)
        combine = sum(row["combine_logical_bytes"] for row in rows)
        assert dispatch == remote_calls * 8196 and combine == remote_calls * 8192
        report["with_copies" if use_copies else "primaries_only"] = {
            "ranks": rows, "remote_expert_calls": remote_calls,
            "dispatch_logical_bytes": dispatch, "combine_logical_bytes": combine,
            "split_exchange_logical_bytes": 96,
        }
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(report))


def run(output):
    with TemporaryDirectory() as directory:
        hashes = stage_weights(Path(directory))
        print(f"Staged {len(hashes)} checkpoint tensors.", flush=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = str(sock.getsockname()[1])
        env = dict(os.environ, SARVAM_LAYER_WEIGHTS=directory,
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=port, WORLD_SIZE="4",
                   GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo")
        processes = [
            subprocess.Popen([sys.executable, __file__, "--worker"],
                             env=dict(env, RANK=str(rank)), stdout=subprocess.PIPE, text=True)
            for rank in range(4)
        ]
        try:
            outputs = [process.communicate(timeout=60)[0] for process in processes]
            if any(process.returncode for process in processes):
                raise RuntimeError("CPU layer validation failed")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
    output.write_text(json.dumps({
        "scope": "Local CPU/Gloo verification",
        "checkpoint_revision": REVISION,
        "tensor_sha256": hashes,
        "results": json.loads(outputs[0].strip().splitlines()[-1]),
    }, indent=2) + "\n")
    print("CPU/Gloo output parity and logical-byte checks passed.")


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker()
    else:
        run(Path(sys.argv[1]))
