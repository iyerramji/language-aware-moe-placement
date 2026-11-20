"""Validate and summarize the fixed-continuation load-cap ablations."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
data = json.loads((root / "gpu-result.json").read_text())
audit = json.loads((root / "fit-audit.json").read_text())
assert data["placement_sha256"] == hashlib.sha256((root / "cap-maps.json").read_bytes()).hexdigest()
assert len(data["devices"]) == 4 and all("L40S" in name for name in data["devices"])
groups, references = defaultdict(list), {}
repeats = []

for trial in data["runs"]:
    mix, arm, repeat = (trial[k] for k in ("mix", "arm", "repeat"))
    ranks, diag = (trial[k] for k in ("ranks", "diagnostic"))
    assert sorted(w["rank"] for w in ranks) == sorted(w["rank"] for w in diag) == list(range(4))
    keys = {name: h for rank in ranks for name, h in rank["fixed_input_hashes"].items()}
    profiles = {name: h for rank in diag for name, h in rank["fixed_input_hashes"].items()}
    pair = mix, repeat
    expected = references.setdefault(pair, (keys, profiles))
    assert expected == (keys, profiles)
    ids = data["prompt_ids"][repeat]
    ids = {name for name in ids if mix == "full" or not name.startswith("flores-en-")}
    assert keys.keys() == profiles.keys() == ids
    assert all(len(w["decode_step_ms"]) == 64 for w in ranks)
    assert all(len(w["profile"]) == 12 for w in diag)
    bytes_stats = np.array([w["layer_stats"] for w in ranks], dtype=np.int64)
    total = bytes_stats.sum(axis=(0, 1))
    tokens = sum(w["generated_token_positions"] for w in ranks)
    assert tokens == 64 * len(keys)
    assert total[0] == total[3] * 8196 and total[1] == total[3] * 8192
    assert total[3] + total[4] == total[6] == tokens * 18 * 6
    intervals = np.concatenate([np.tile(w["decode_step_ms"][1:], len(w["request_ids"])) for w in ranks])
    profile_ms = np.array([[s["step_ms"] for s in w["profile"]] for w in diag])
    collective = np.array([[s["collective_stream_ms"] for s in w["profile"]] for w in diag])
    assert (collective >= 0).all() and (profile_ms > 0).all()
    slowest = profile_ms.argmax(axis=0)
    shares = (collective / profile_ms)[slowest, np.arange(12)]
    assert (shares <= 1.01).all()
    selected = [diag[int(r)]["profile"][step] for step, r in enumerate(slowest)]
    work = np.array([[s["executed_work_by_layer"] for s in w["profile"]] for w in diag])
    imbalance = work.max(0) / work.mean(0)
    assert np.isfinite(imbalance).all()
    item = {
        "mix": mix, "arm": arm, "repeat": repeat + 1, "decode_tokens": tokens,
        "logical_MB_per_token": (total[0] + total[1]) / tokens / 1e6,
        "logical_MB_per_token_with_size_exchange": sum(total[:3]) / tokens / 1e6,
        "itl_p50_ms": float(np.median(intervals)), "itl_p95_ms": float(np.percentile(intervals, 95)),
        "collective_pct_slowest_rank": float(100 * np.median(shares)),
        "nccl_sizes_pct_slowest_rank": float(np.median(
            [100 * s["size_exchange_ms"] / s["step_ms"] for s in selected])),
        "nccl_dispatch_pct_slowest_rank": float(np.median(
            [100 * s["dispatch_ms"] / s["step_ms"] for s in selected])),
        "nccl_combine_pct_slowest_rank": float(np.median(
            [100 * s["combine_ms"] / s["step_ms"] for s in selected])),
        "profile_slowest_step_p50_ms": float(np.median(profile_ms.max(0))),
        "unprofiled_prefix_slowest_p50_ms": float(np.median(
            np.array([w["decode_step_ms"][:12] for w in ranks]).max(0))),
        "mean_layer_step_imbalance": float(imbalance.mean()),
        "p95_layer_step_imbalance": float(np.percentile(imbalance, 95)),
        "max_layer_step_imbalance": float(imbalance.max()),
        "local_replica_calls": int(total[5]),
    }
    repeats.append(item)
    groups[mix, arm].append(item)
assert repeats, "No completed GPU trials"
if data["status"] == "completed":
    expected = {"language_blind", "language_aware", "aware_cap_115", "blind_aa"}
    assert set(groups) == ({(mix, arm) for mix in ("full", "indic_only") for arm in expected}
                           | {("indic_only", "aware_cap_125")})
    assert len(repeats) == 27 and all({r["repeat"] for r in items} == {1, 2, 3}
                                     and len(items) == 3 for items in groups.values())
    assert len(references) == 6
with (root / "per-repeat.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=repeats[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(repeats)

keys = ("logical_MB_per_token", "itl_p50_ms", "itl_p95_ms", "collective_pct_slowest_rank",
        "mean_layer_step_imbalance", "p95_layer_step_imbalance",
        "nccl_sizes_pct_slowest_rank", "nccl_dispatch_pct_slowest_rank",
        "nccl_combine_pct_slowest_rank", "profile_slowest_step_p50_ms",
        "unprofiled_prefix_slowest_p50_ms")
rows = []
for (mix, arm), items in sorted(groups.items()):
    rows.append({"mix": mix, "arm": arm, "repeats": len(items),
                 **{k: float(np.median([r[k] for r in items])) for k in keys}})
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

report = ["# Sarvam-30B fixed-continuation placement follow-up\n",
          f"**{data['status'].title()}** on four L40S GPUs in Modal. {len(repeats)} trials, "
          f"{sum(r['decode_tokens'] for r in repeats):,} measured decode token positions.\n",
          "All runs use Sarvam-30B, BF16, PyTorch/NCCL, a fixed two-copy budget, "
          "and decode-only measurements on the same worker. The "
          "[prior research-default pilot](../modal-sarvam-decode/REPORT.md) used free generation.\n",
          "## Local cap audit\n",
          "| Mix | Cap | Changed layers | Replica slots replaced | Old fit peak | New fit peak |",
          "|---|---:|---:|---:|---:|---:|"]
for name, item in audit.items():
    report.append(f"| {name.split('/')[0]} | {item['cap']:.2f}× | "
                  f"{len(item['changed_decoder_layers'])} | {item['replica_slots_replaced']} | "
                  f"{item['fit_old_max']:.3f}× | {item['fit_new_max']:.3f}× |")
report += [
    "",
    "The local search retains the primary assignments and request homes. An integer program "
    "minimizes changes to existing replica slots in each violating layer, then favors local "
    "hits within that minimum. Each GPU retains exactly two copies per layer. The cap is "
    "on predicted executed expert work after local replica selection, per layer on fit "
    "data. The full-mix 1.25× layout is identical to the prior layout and is not rerun as "
    "a separate arm. The 1.15× runs are sensitivity tests selected before GPU timings.",
    "",
    "### Held-out routing replay\n",
    "This local replay uses counts from all held-out FLORES sequences. It estimates the share "
    "of remote expert calls and rank load. Blind demand is split equally among the four ranks.",
    "",
    "| Mix | Layout | Sequences | Remote calls | Mean layer imbalance | Worst layer imbalance |",
    "|---|---|---:|---:|---:|---:|",
]
with (root / "local-replay.csv").open() as stream:
    for r in csv.DictReader(stream):
        report.append(f"| {r['mix']} | {r['arm']} | {r['evaluation_sequences']} | "
                      f"{float(r['remote_call_share_mean']):.2%} | "
                      f"{float(r['mean_layer_max_over_mean']):.3f}× | "
                      f"{float(r['max_layer_max_over_mean']):.3f}× |")
report += [
    "",
    "## GPU measurements\n",
    "Metrics are medians of the three repeat summaries. Timing uses consecutive decode "
    "steps on each rank, weighted by active requests, with prefill excluded.",
    "",
    "| Mix | Layout | Logical MB/token | p50 ITL ms | p95 ITL ms | NCCL share | Mean imbalance | p95 imbalance |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    report.append("| {mix} | {arm} | {logical_MB_per_token:.3f} | {itl_p50_ms:.2f} | "
                  "{itl_p95_ms:.2f} | {collective_pct_slowest_rank:.1f}% | "
                  "{mean_layer_step_imbalance:.3f}× | {p95_layer_step_imbalance:.3f}× |".format(**r))
report += ["", "## Paired comparisons\n"]
for mix in ("indic_only", "full"):
    for arm, comparator in (("blind_aa", "language_blind"),
                            ("aware_cap_125", "language_aware"),
                            ("aware_cap_125", "language_blind"),
                            ("aware_cap_115", "language_aware"),
                            ("aware_cap_115", "language_blind")):
        a = {v["repeat"]: v for v in groups[mix, arm]}
        b = {v["repeat"]: v for v in groups[mix, comparator]}
        if len(a.keys() & b.keys()) != 3:
            continue
        ratios = lambda k: [100 * (a[i][k] / b[i][k] - 1) for i in sorted(a.keys() & b.keys())]
        mb, itl = ratios("logical_MB_per_token"), ratios("itl_p50_ms")
        report.append(f"- {mix}, {arm} versus {comparator}. Logical MB/token {np.median(mb):+.2f}%. "
                      f"Median ITL change {np.median(itl):+.2f}% across paired repeats "
                      f"(range {min(itl):+.2f}% to {max(itl):+.2f}%).")
report += [
    "",
    "## Where decode time went",
    "",
    "The phase columns are separate medians and need not sum to the median total NCCL share. "
    "The last two columns compare the first 12 steps with and without event instrumentation.",
    "",
    "| Mix | Layout | Size exchange | Dispatch | Combine | Profiled step ms | Unprofiled step ms |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for r in rows:
    report.append("| {mix} | {arm} | {nccl_sizes_pct_slowest_rank:.1f}% | "
                  "{nccl_dispatch_pct_slowest_rank:.1f}% | {nccl_combine_pct_slowest_rank:.1f}% | "
                  "{profile_slowest_step_p50_ms:.2f} | {unprofiled_prefix_slowest_p50_ms:.2f} |".format(**r))
report += [
    "",
    "## Measurement limits\n",
    "The runner uses the same pinned checkpoint, held-out FLORES batches, and 64 fixed "
    "decode inputs per request (generated once with the blind layout). Input hashes match "
    "across arms. Floating-point execution and expert selections can still vary. The "
    "language-blind duplicate provides a timing repeatability check. All comparisons use "
    "the same research runtime and the two-copy memory budget. We rotate arm order.",
    "",
    "NCCL share is the median, across a separate 12-step diagnostic replay, of the sum "
    "of elapsed CUDA stream intervals around the size exchange, dispatch and combine, "
    "divided by wall time on the rank with the longest measured step. Stream intervals "
    "include collective synchronization and may include enqueue/wait effects. This is "
    "not physical link time or an isolated critical-path contribution. The residual step "
    "time includes expert compute, attention, and host work. The measurement does not "
    "assign that residual to Python alone. The separate "
    "unprofiled 64-step runs supply the primary latency numbers. Diagnostic overhead is "
    "checkable using the matching first-12-step wall times in the repeat CSV. The JSON "
    "records per-phase measurements and per-layer executed work.",
    "",
    "Logical MB/token counts sender-side dispatch and combine, with expert IDs included. "
    "Split-size bytes are tracked separately. 1 MB is 1,000,000 bytes. NCCL's runtime "
    "transport and CUDA peer-access checks are in the result. Physical link counters "
    "were not measured.",
    "",
    "Reproduce by running `cap_plans.py` and `replay_cap.py` with the output folder as "
    "their argument, then `modal run --timestamps --write-result "
    "results/gpu-pilot/2026-11-20-cap/gpu-result.json gpu_pilot/modal_sarvam_cap.py::benchmark`. "
    "Run `report_cap.py` with the same folder to validate and summarize the payload.",
    "",
    "[Fit audit](fit-audit.json) · [Local replay CSV](local-replay.csv) · [Frozen maps](cap-maps.json) · [GPU records](gpu-result.json) · "
    "[Summary CSV](summary.csv) · [Per-repeat CSV](per-repeat.csv)",
]
(root / "REPORT.md").write_text("\n".join(report) + "\n")
