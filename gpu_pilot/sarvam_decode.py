"""Shared PyTorch/NCCL research runtime for a three-placement Sarvam decode pilot."""

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from tokenizers import Tokenizer
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sarvam_layer import REVISION, exchange, split_sizes

ARMS = ("research_default", "language_blind", "language_aware")
MIXES = ("full", "indic_only")
STEPS = 64
REPEATS = 3


def expert_layout(plan, layer):
    if plan is None:
        return [i // 32 for i in range(128)], [[], [], [], []]
    return (plan["primary_rank_by_layer_expert"][layer],
            plan["replicas_by_layer_rank"][layer])


class DistributedExperts(nn.Module):
    def __init__(self, original, primary, replicas, rank, device):
        super().__init__()
        self.rank = rank
        self.experts = nn.ModuleDict({
            str(i): original[i] for i in range(128)
            if primary[i] == rank or i in replicas[rank]
        })
        assert len(self.experts) == (34 if replicas[rank] else 32)
        self.local_ids = frozenset(map(int, self.experts))
        self.local_copy_ids = frozenset(replicas[rank])
        self.destination = torch.tensor([
            rank if i in self.local_ids else home for i, home in enumerate(primary)
        ], device=device)
        self.stats = [0] * 7

    def forward(self, hidden, expert_ids, expert_weights):
        flat_ids = expert_ids.flatten()
        width = expert_ids.shape[1]
        dest = self.destination[flat_ids]
        positions = [
            torch.where(dest == peer)[0] if peer != self.rank else dest.new_empty(0)
            for peer in range(4)
        ]
        send_rows = torch.cat(positions)
        sends = [row.numel() for row in positions]
        receives = split_sizes(sends, hidden.device)
        send_ids = flat_ids[send_rows].to(torch.int32).reshape(-1, 1)
        send_hidden = hidden[send_rows // width]
        remote_ids = exchange(send_ids, sends, receives).flatten().to(torch.long)
        remote_hidden = exchange(send_hidden, sends, receives)

        local_rows = torch.where(dest == self.rank)[0]
        local_ids = flat_ids[local_rows]
        execution_ids = torch.cat((local_ids, remote_ids))
        work = torch.cat((hidden[local_rows // width], remote_hidden))
        outputs = torch.empty_like(work)
        for expert in execution_ids.unique().tolist():
            mask = execution_ids == expert
            outputs[mask] = self.experts[str(expert)](work[mask])

        local_count = local_rows.numel()
        returned = exchange(outputs[local_count:].contiguous(), receives, sends)
        combined = hidden.new_empty((flat_ids.numel(), hidden.shape[1]))
        combined[send_rows] = returned
        combined[local_rows] = outputs[:local_count]
        replica_hits = sum(local_ids.tolist().count(i) for i in self.local_copy_ids)
        outbound = send_hidden.numel() * send_hidden.element_size() + send_ids.numel() * send_ids.element_size()
        inbound_return = outputs[local_count:].numel() * outputs.element_size()
        assert outbound == sum(sends) * (hidden.shape[1] * hidden.element_size() + 4)
        assert inbound_return == sum(receives) * hidden.shape[1] * hidden.element_size()
        delta = [outbound, inbound_return, 3 * 8, sum(sends), local_count,
                 replica_hits, execution_ids.numel()]
        self.stats = [a + b for a, b in zip(self.stats, delta)]
        return (combined.view(*expert_ids.shape, -1).float() *
                expert_weights.float().unsqueeze(-1)).sum(1).to(hidden.dtype)


def make_model(folder, plan, rank, device):
    config = AutoConfig.from_pretrained(folder, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], folder)
    assert config.first_k_dense_replace == 1 and config.num_hidden_layers == 19
    assert config.num_experts == 128 and config.num_experts_per_tok == 6
    with torch.device("meta"):
        model = model_class(config)
    layers = []
    for index, decoder in enumerate(model.model.layers[1:]):
        primary, replicas = expert_layout(plan, index)
        block = decoder.mlp
        # Preserve checkpoint names while swapping the expert container.
        block.experts = DistributedExperts(block.experts, primary, replicas, rank, device)
        layers.append(block.experts)

    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    pending = defaultdict(list)
    for name, _ in model.named_parameters():
        # ModuleDict adds a namespace around the original numbered MLPs.
        key = name.replace(".mlp.experts.experts.", ".mlp.experts.")
        pending[index[key]].append((name, key))
    for shard, parameters in pending.items():
        with safe_open(folder / shard, framework="pt", device="cpu") as weights:
            for name, key in parameters:
                owner, leaf = name.rsplit(".", 1)
                tensor = weights.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                model.get_submodule(owner).register_parameter(leaf, nn.Parameter(tensor, requires_grad=False))
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config, device=device)
    assert all(p.is_cuda for p in model.parameters())
    assert all(b.is_cuda for b in model.buffers())
    return model.eval(), layers


def make_batch(tokenizer, rows, local_rows, device):
    tokens = [tokenizer.encode(row["text"], add_special_tokens=True).ids for row in rows]
    width = max(map(len, tokens))
    ids = [tokens[i] for i in local_rows]
    framed = [[0] * (width - len(seq)) + seq for seq in ids]
    masks = [[0] * (width - len(seq)) + [1] * len(seq) for seq in ids]
    return torch.tensor(framed, device=device), torch.tensor(masks, device=device)


@torch.inference_mode()
def decode(model, layers, tokenizer, rows, homes, rank, device, steps):
    local_rows = [i for i, home in enumerate(homes) if home == rank]
    assert local_rows, homes
    ids, mask = make_batch(tokenizer, rows, local_rows, device)
    position = mask.cumsum(-1).sub(1).clamp_min(0)
    prefill = model.model(input_ids=ids, attention_mask=mask, position_ids=position, use_cache=True)
    token = model.lm_head(prefill.last_hidden_state[:, -1:]).argmax(-1)
    cache = prefill.past_key_values
    del prefill
    for layer in layers:
        layer.stats = [0] * 7
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    last_ready = time.perf_counter()
    intervals = []
    continuation = []
    for step in range(steps):
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), -1)
        position = mask.sum(-1, keepdim=True) - 1
        state = model.model(input_ids=token, attention_mask=mask, position_ids=position,
                            past_key_values=cache, use_cache=True)
        token = model.lm_head(state.last_hidden_state[:, -1:]).argmax(-1)
        cache = state.past_key_values
        torch.cuda.synchronize()
        ready = time.perf_counter()
        # Exclude first measured interval, which starts at the prefill barrier.
        if step:
            intervals.append((ready - last_ready) * 1000)
        last_ready = ready
        continuation.append(token)
    generated = torch.cat(continuation, dim=1).tolist()
    return {
        "rank": rank, "request_ids": [rows[i]["id"] for i in local_rows],
        "generated_tokens": steps * len(local_rows), "step_itl_ms": intervals,
        "continuation_sha256": {
            rows[i]["id"]: hashlib.sha256(json.dumps(seq).encode()).hexdigest()
            for i, seq in zip(local_rows, generated)
        },
        "layer_stats": [layer.stats for layer in layers],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }


def run(folder, manifest_file, map_file, output_file):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    maps = json.loads(map_file.read_text())
    prompts = json.loads(manifest_file.read_text())
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    assert tokenizer.token_to_id("<pad>") == 0
    results = []
    for mix in MIXES:
        languages = maps[mix]["language_aware"]["home_rank_by_language"]
        for arm in ARMS:
            plan = maps[mix].get(arm)
            torch.cuda.reset_peak_memory_stats(device)
            model, layers = make_model(folder, plan, rank, device)
            for repeat in range(REPEATS):
                rows = [r for r in prompts[repeat] if r["language"] in languages]
                # Same rank assignment for both language-agnostic arms.
                homes = ([languages[r["language"]] for r in rows]
                         if arm == "language_aware"
                         else [i % 4 for i in range(len(rows))])
                if repeat == 0:
                    decode(model, layers, tokenizer, rows, homes, rank, device, steps=3)
                local = decode(model, layers, tokenizer, rows, homes, rank, device, steps=STEPS)
                records = [None] * 4
                dist.all_gather_object(records, local)
                if rank == 0:
                    assert {r["id"]: h for r, h in zip(rows, homes)} == {
                        rid: worker["rank"] for worker in records for rid in worker["request_ids"]
                    }
                    assert sum(w["generated_tokens"] for w in records) == len(rows) * STEPS
                    for layer in range(18):
                        total = [sum(w["layer_stats"][layer][i] for w in records) for i in range(7)]
                        assert total[3] + total[4] == total[6] == len(rows) * STEPS * 6
                        assert total[0] == total[3] * 8196 and total[1] == total[3] * 8192
                    if arm != "research_default":
                        assert sum(row[5] for w in records for row in w["layer_stats"]) > 0
                    results.append({"mix": mix, "arm": arm, "repeat": repeat, "ranks": records})
                    output_file.write_text(json.dumps(results))
                    print(f"Complete {mix} {arm} repeat={repeat+1}", flush=True)
                dist.barrier()
            del model, layers
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    run(*(Path(a) for a in sys.argv[1:]))
