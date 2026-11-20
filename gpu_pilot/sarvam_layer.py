"""Controlled layer-1 Sarvam dispatch using real weights and fixed top-6 inputs."""
import hashlib
import json
import os
import struct
import urllib.request
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"
BASE = f"https://huggingface.co/sarvamai/sarvam-30b/resolve/{REVISION}/"
PRIMARY = (1, 2, 3, 0, 1, 2)
COPIES = ((0, 1), (1, 2), (2, 3), (3, 4))
HIDDEN = 4096


def read_range(shard, start, stop):
    request = urllib.request.Request(
        BASE + shard, headers={"Range": f"bytes={start}-{stop - 1}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        prefix = f"bytes {start}-{stop - 1}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(prefix):
            raise RuntimeError(f"Unexpected range response for {shard}")
        raw = response.read(stop - start + 1)
    if len(raw) != stop - start:
        raise RuntimeError(f"Truncated range for {shard}")
    return raw


def stage_weights(directory):
    with urllib.request.urlopen(BASE + "model.safetensors.index.json", timeout=20) as response:
        index = json.load(response)["weight_map"]
    headers = {}
    hashes = {}
    for expert in [str(i) for i in range(6)] + ["shared"]:
        stem = f"experts.{expert}" if expert != "shared" else "shared_experts"
        for projection in ("gate", "up", "down"):
            key = f"model.layers.1.mlp.{stem}.{projection}_proj.weight"
            shard = index[key]
            if shard not in headers:
                size = struct.unpack("<Q", read_range(shard, 0, 8))[0]
                if size > 10_000_000:
                    raise RuntimeError("Unexpected safetensors header size")
                headers[shard] = size + 8, json.loads(read_range(shard, 8, 8 + size))
            header_end, header = headers[shard]
            metadata = header[key]
            storage_dtype = {"F32": torch.float32, "BF16": torch.bfloat16}[metadata["dtype"]]
            start, stop = metadata["data_offsets"]
            raw = read_range(shard, header_end + start, header_end + stop)
            hashes[key] = hashlib.sha256(raw).hexdigest()
            tensor = torch.frombuffer(bytearray(raw), dtype=storage_dtype).reshape(metadata["shape"])
            torch.save(tensor, directory / f"{expert}-{projection}.pt")
    return hashes


def load_expert(directory, expert, device):
    return tuple(
        torch.load(directory / f"{expert}-{name}.pt", weights_only=True).to(device=device, dtype=torch.bfloat16)
        for name in ("gate", "up", "down")
    )


def mlp(x, weights):
    gate, up, down = weights
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def split_sizes(sends, device):
    send = torch.tensor(sends, dtype=torch.int64, device=device)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    return recv.tolist()


def exchange(rows, sends, receives):
    result = rows.new_empty((sum(receives), rows.shape[1]))
    dist.all_to_all_single(result, rows, receives, sends)
    return result


def measure(directory, rank, device, use_copies):
    # One identical-size workload per rank; values are deterministic and distinct.
    token = torch.randn((1, HIDDEN), generator=torch.Generator().manual_seed(100 + rank))
    token = (token * 0.1).to(device=device, dtype=torch.bfloat16)
    experts = [i for i, home in enumerate(PRIMARY) if home == rank]
    if use_copies:
        experts += list(COPIES[rank])
    resident = {i: load_expert(directory, i, device) for i in experts}
    choices = [rank if i in resident else PRIMARY[i] for i in range(6)]
    # Payload packing follows destination order. Expert id is transmitted separately.
    orders = [[i for i, dst in enumerate(choices) if dst == peer] for peer in range(4)]
    packed_ids = [i for peer in range(4) if peer != rank for i in orders[peer]]
    sends = [len(ids) if peer != rank else 0 for peer, ids in enumerate(orders)]
    receives = split_sizes(sends, device)
    expected_receives = [
        0 if src == rank else sum(
            home == rank and (not use_copies or i not in COPIES[src])
            for i, home in enumerate(PRIMARY)
        )
        for src in range(4)
    ]
    assert receives == expected_receives
    ids = torch.tensor(packed_ids, dtype=torch.int32, device=device).reshape(-1, 1)
    activations = token.repeat(len(packed_ids), 1)
    recv_ids = exchange(ids, sends, receives)
    recv_activations = exchange(activations, sends, receives)
    expert_outputs = torch.empty_like(recv_activations)
    executed = recv_ids.flatten().tolist()
    for row, expert in enumerate(executed):
        expert_outputs[row:row + 1] = mlp(recv_activations[row:row + 1], resident[expert])
    returned = exchange(expert_outputs, receives, sends)

    results = {expert: returned[row:row + 1] for row, expert in enumerate(packed_ids)}
    for expert in orders[rank]:
        results[expert] = mlp(token, resident[expert])
    shared = mlp(token, load_expert(directory, "shared", device)).float()
    weights = torch.tensor([1, 2, 3, 4, 5, 6], device=device) * (2.5 / 21)
    expected = [
        mlp(token, load_expert(directory, expert, device))
        for expert in range(6)
    ]
    for expert in range(6):
        torch.testing.assert_close(results[expert], expected[expert], rtol=0, atol=0)
    actual_output = shared + sum(results[i].float() * weights[i] for i in range(6))
    reference = shared + sum(expected[i].float() * weights[i] for i in range(6))
    torch.testing.assert_close(actual_output, reference, rtol=0, atol=0)
    expected_choices = [rank if use_copies and i in COPIES[rank] else PRIMARY[i] for i in range(6)]
    assert choices == expected_choices
    dispatch = activations.numel() * activations.element_size() + ids.numel() * ids.element_size()
    combine = expert_outputs.numel() * expert_outputs.element_size()
    assert dispatch == sum(dst != rank for dst in expected_choices) * 8196
    assert combine == sum(receives) * 8192
    return {
        "rank": rank, "destinations_by_expert": choices, "resident_experts": experts,
        "local_experts_executed": orders[rank], "remote_experts_executed": executed,
        "dispatch_logical_bytes": dispatch, "combine_logical_bytes": combine,
        "split_exchange_logical_bytes": 3 * 8,
        "max_output_abs_error": float((actual_output - reference).abs().max()),
    }


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    directory = Path(os.environ["SARVAM_LAYER_WEIGHTS"])
    report = {}
    for use_copies in (False, True):
        result = measure(directory, rank, device, use_copies)
        by_rank = [None] * 4
        dist.all_gather_object(by_rank, result)
        remotes = 10 if use_copies else 18
        dispatch = sum(row["dispatch_logical_bytes"] for row in by_rank)
        combine = sum(row["combine_logical_bytes"] for row in by_rank)
        assert dispatch == remotes * 8196 and combine == remotes * 8192
        report["with_copies" if use_copies else "primaries_only"] = {
            "ranks": by_rank, "remote_expert_calls": remotes, "input_tokens": 4,
            "dispatch_logical_bytes": dispatch, "combine_logical_bytes": combine,
            "split_exchange_logical_bytes": 96,
        }
    if rank == 0:
        print(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
