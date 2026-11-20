"""Summarize verified GPU pilot records for Raj."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

folder = Path(sys.argv[1])
artifact = json.loads((folder / "decode-result.json").read_text())
groups = defaultdict(list)
for trial in artifact["runs"]:
    groups[trial["mix"], trial["arm"]].append(trial)
fields = ("dispatch_bytes", "combine_bytes", "size_exchange_bytes", "remote_calls",
          "local_calls", "replica_calls", "executed_calls")
rows = []
for (mix, arm), trials in groups.items():
    workers = [rank for trial in trials for rank in trial["ranks"]]
    totals = np.array([layer for worker in workers for layer in worker["layer_stats"]]).sum(0)
    counts = dict(zip(fields, map(int, totals)))
    tokens = sum(w["generated_tokens"] for w in workers)
    samples = np.concatenate([
        np.tile(w["step_itl_ms"], len(w["request_ids"])) for w in workers
    ])
    q50, q95, q99 = np.percentile(samples, [50, 95, 99])
    timing_by_repeat = [float(np.percentile(np.concatenate([
        np.tile(w["step_itl_ms"], len(w["request_ids"])) for w in trial["ranks"]
    ]), 50)) for trial in trials]
    balance = np.concatenate([
        (lambda load: load.max(axis=0) / load.mean(axis=0))(
            np.array([[row[6] for row in w["layer_stats"]] for w in trial["ranks"]])
        ) for trial in trials
    ])
    rows.append({
        "mix": mix, "placement": arm, "repeats": len(trials),
        "generated_tokens": tokens, "itl_samples": len(samples), **counts,
        "bytes_per_token": (totals[0] + totals[1]) / tokens,
        "bytes_per_token_including_sizes": sum(totals[:3]) / tokens,
        "itl_p50_ms": q50, "itl_p95_ms": q95, "itl_p99_ms": q99,
        "repeat_p50_min_ms": min(timing_by_repeat), "repeat_p50_max_ms": max(timing_by_repeat),
        "mean_layer_max_over_mean_expert_load": balance.mean(),
        "peak_gpu_gib": max(w["peak_memory_bytes"] for w in workers) / (1024**3),
    })
if not rows:
    raise SystemExit("The GPU job returned no completed trials; inspect its log.")
with (folder / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
table = [
    "| Traffic mix | Placement | Repeats | Logical bytes/token | ITL p50 ms | ITL p95 ms | ITL p99 ms | Local replica hits |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    table.append("| {mix} | {placement} | {repeats} | {bytes_per_token:,.1f} | {itl_p50_ms:.2f} | {itl_p95_ms:.2f} | {itl_p99_ms:.2f} | {replica_calls:,} |".format(**row))

mapping = {(trial["mix"], trial["arm"], trial["repeat"]):
           {k: digest for w in trial["ranks"] for k, digest in w["continuation_sha256"].items()}
           for trial in artifact["runs"]}
deltas = []
for row in rows:
    mix, arm = row["mix"], row["placement"]
    if arm == "research_default":
        continue
    pairs = [(mapping[(mix, "research_default", repeat)], hashed)
             for (m, a, repeat), hashed in mapping.items()
             if m == mix and a == arm and (mix, "research_default", repeat) in mapping]
    matched = sum(sum(default[k] == digest for k, digest in other.items()) for default, other in pairs)
    total = sum(len(other) for _, other in pairs)
    if total:
        deltas.append(f"- {mix}, {arm} versus default: {matched}/{total} identical continuation hashes.")

comparison = []
for mix in ("full", "indic_only"):
    stats = {row["placement"]: row for row in rows if row["mix"] == mix and row["repeats"] == 3}
    if len(stats) != 3:
        continue
    target = stats["language_aware"]
    for baseline in ("research_default", "language_blind"):
        base = stats[baseline]
        byte_change = 100 * (target["bytes_per_token"] / base["bytes_per_token"] - 1)
        itl_change = 100 * (target["itl_p50_ms"] / base["itl_p50_ms"] - 1)
        tail_change = 100 * (target["itl_p95_ms"] / base["itl_p95_ms"] - 1)
        comparison.append(f"- {mix}, language-aware versus {baseline}: bytes/token {byte_change:+.2f}%, median ITL {itl_change:+.2f}%, p95 ITL {tail_change:+.2f}%.")

transports = [x for x in ("SHM", "P2P", "NET") if any(f"via {x}" in line for line in artifact["nccl_routes"])]
report = [
    "# Sarvam-30B decode placement pilot",
    f"Modal run status **{artifact['status']}**. {len(artifact['runs'])} completed arm/repeat records. GPU allocation {', '.join(artifact['devices'])}.",
    *table,
    *comparison,
    "The default arm uses contiguous expert IDs and zero replicas in our PyTorch/NCCL research runtime. Blind and aware each use two extra expert slots per GPU per routed layer. They share one dispatch implementation; the aware policy also homes requests by language. Default and blind use the same round-robin request homes.",
    "Each matched repeat uses a distinct pair of held-out FLORES parallel sentences per language. For each placement, the runtime prefills the prompts, then generates 64 timed decode tokens per request. ITL uses consecutive token-ready times inside each CUDA-synchronized rank, repeated over its active requests. The first measured token has no in-window predecessor and is excluded from ITL. These are batch decode intervals from a research loop with synchronous instrumentation. Per-repeat median ranges and sample counts are in the CSV. The arm order was fixed, so repeat variability does not capture possible drift between arms. Requests execute in rank-local batches and their inter-token intervals share batch-step timing.",
    f"NCCL logged transport paths {', '.join(transports) or 'unavailable'} on this worker. CUDA peer-access matrix and raw transport output are in the evidence file.",
    "Byte counts include outgoing off-rank BF16 activations and int32 expert IDs on dispatch and BF16 outputs on return, counted once at each sender. The size exchange is captured separately. Each step checks bytes against the number of routed expert calls. These are logical payload bytes. NCCL transport metadata, physical link counters and prefill are outside that measurement.",
    "The runner verifies all request homes against observed rank records and checks that replicas execute work in both replicated arms. BF16 execution at different batch sizes or expert groupings may produce different continuations; the checks below flag that source of variation.",
    *deltas,
    "Reproduce with `modal run --timestamps --write-result <output>/decode-result.json gpu_pilot/modal_sarvam_decode.py::benchmark` after staging the pinned checkpoint. Source: [Modal runner](../../../gpu_pilot/modal_sarvam_decode.py), [decode and counters](../../../gpu_pilot/sarvam_decode.py), [report builder](../../../gpu_pilot/report_decode.py).",
    "[Run evidence](decode-result.json) · [Metrics CSV](summary.csv) · [One-layer counter validation](../modal-sarvam-layer/REPORT.md)",
]
(folder / "REPORT.md").write_text("\n\n".join(["\n".join(report[0:2]), "\n".join(table), *report[2+len(table):]]) + "\n")
