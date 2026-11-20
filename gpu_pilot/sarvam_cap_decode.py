"""Matched fixed-continuation Sarvam pilot with isolated NCCL timing passes."""
import hashlib
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from tokenizers import Tokenizer

from sarvam_decode import make_batch, make_model

STEPS = 64
PROFILE_STEPS = 12
LAYOUTS = {
    "indic_only": ["language_blind", "language_aware", "aware_cap_125", "aware_cap_115", "blind_aa"],
    "full": ["language_blind", "language_aware", "aware_cap_115", "blind_aa"],
}


def layout_name(arm):
    return "language_blind" if arm == "blind_aa" else arm


@torch.inference_mode()
def run_decode(model, layers, tokenizer, rows, homes, rank, device, steps,
               reference=None, profile=False):
    local = [i for i, home in enumerate(homes) if home == rank]
    names = [rows[i]["id"] for i in local]
    assert names
    ids, mask = make_batch(tokenizer, rows, local, device)
    pos = mask.cumsum(-1).sub(1).clamp_min(0)
    out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    token = model.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
    del out
    forced = None if reference is None else torch.tensor(
        [reference[name][:steps] for name in names], device=device, dtype=torch.long
    )
    for layer in layers:
        layer.stats = [0] * 7
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    intervals, collection, timing, tokens = [], [], [], []
    previous_stats = [[0] * 7 for _ in layers]
    original = dist.all_to_all_single

    def instrument(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original(*args, **kwargs)
        end.record()
        collection.append((start, end))
        return result

    def loop():
        nonlocal mask, token, cache, previous_stats
        for step in range(steps):
            collection.clear()
            start = time.perf_counter()
            if forced is not None:
                token = forced[:, step:step + 1]
            tokens.append(token)
            mask = torch.cat((mask, torch.ones_like(mask[:, :1])), -1)
            pos = mask.sum(-1, keepdim=True) - 1
            out = model.model(input_ids=token, attention_mask=mask, position_ids=pos,
                              past_key_values=cache, use_cache=True)
            token = model.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
            cache = out.past_key_values
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            intervals.append(elapsed)
            if profile:
                assert len(collection) == len(layers) * 4, len(collection)
                calls = [a.elapsed_time(b) for a, b in collection]
                timing.append({
                    "step_ms": elapsed,
                    "collective_stream_ms": sum(calls),
                    "size_exchange_ms": sum(calls[0::4]),
                    "dispatch_ms": sum(calls[1::4]) + sum(calls[2::4]),
                    "combine_ms": sum(calls[3::4]),
                    "executed_work_by_layer": [x.stats[6] - old[6]
                                               for x, old in zip(layers, previous_stats)],
                })
                previous_stats = [x.stats.copy() for x in layers]

    if profile:
        with patch.object(dist, "all_to_all_single", instrument):
            loop()
    else:
        loop()
    token_rows = dict(zip(names, torch.cat(tokens, dim=1).tolist()))
    if reference is None:
        return token_rows
    assert token_rows == {name: reference[name][:steps] for name in names}
    return {
        "rank": rank,
        "request_ids": names,
        "fixed_input_hashes": {name: hashlib.sha256(json.dumps(seq).encode()).hexdigest()
                               for name, seq in token_rows.items()},
        "generated_token_positions": steps * len(names),
        "decode_step_ms": intervals,
        "layer_stats": [x.stats for x in layers],
        "profile": timing,
    }


def all_ranks(value):
    result = [None] * 4
    dist.all_gather_object(result, value)
    return result


def run(folder, prompt_file, map_file, output):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    maps = json.loads(map_file.read_text())
    prompts = json.loads(prompt_file.read_text())
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    assert tokenizer.token_to_id("<pad>") == 0
    results = []

    for mix, arms in LAYOUTS.items():
        languages = maps[mix]["language_aware"]["home_rank_by_language"]
        batches = [[p for p in batch if p["language"] in languages] for batch in prompts]
        model, layers = make_model(folder, maps[mix]["language_blind"], rank, device)
        references = []
        for rows in batches:
            homes = [i % 4 for i in range(len(rows))]
            local = run_decode(model, layers, tokenizer, rows, homes, rank, device, STEPS)
            references.append({k: v for part in all_ranks(local) for k, v in part.items()})
        del model, layers
        torch.cuda.empty_cache()
        # Rotate scheduling order across held-out repeat blocks.
        for repeat, (rows, reference) in enumerate(zip(batches, references)):
            offset = repeat * 2 % len(arms)
            order = arms[offset:] + arms[:offset]
            for arm in order:
                policy = layout_name(arm)
                plan = maps[mix][policy]
                model, layers = make_model(folder, plan, rank, device)
                homes = ([languages[r["language"]] for r in rows] if policy != "language_blind"
                         else [i % 4 for i in range(len(rows))])
                run_decode(model, layers, tokenizer, rows, homes, rank, device, 3, reference)
                main = all_ranks(run_decode(model, layers, tokenizer, rows, homes, rank, device,
                                            STEPS, reference))
                diagnostic = all_ranks(run_decode(model, layers, tokenizer, rows, homes, rank,
                                                  device, PROFILE_STEPS, reference, profile=True))
                if rank == 0:
                    for ranks, length in ((main, STEPS), (diagnostic, PROFILE_STEPS)):
                        assert {p["id"]: h for p, h in zip(rows, homes)} == {
                            name: r["rank"] for r in ranks for name in r["request_ids"]
                        }
                        for li in range(18):
                            a = [sum(r["layer_stats"][li][i] for r in ranks) for i in range(7)]
                            assert a[0] == a[3] * 8196 and a[1] == a[3] * 8192
                            assert a[3] + a[4] == a[6] == len(rows) * length * 6
                        assert sum(layer[5] for r in ranks for layer in r["layer_stats"]) > 0
                    results.append({"mix": mix, "arm": arm, "repeat": repeat,
                                    "order": order, "ranks": main, "diagnostic": diagnostic})
                    temp = output.with_suffix(".tmp")
                    temp.write_text(json.dumps(results))
                    temp.replace(output)
                    print(f"Complete {mix}, {arm}, repeat={repeat+1}", flush=True)
                dist.barrier()
                del model, layers
                torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    run(*(Path(arg) for arg in sys.argv[1:]))
