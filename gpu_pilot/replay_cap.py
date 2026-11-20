"""Replay held-out Sarvam FLORES routing counts with frozen placement maps."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from cap_plans import MIXES, fit_load, load, normalize

directory = Path(sys.argv[1])
maps = json.loads((directory / "cap-maps.json").read_text())
rows = []
for mix in MIXES:
    eval_data = load("sarvam", "flores", indic=mix == "indic_only")[0]["eval"]
    counts, languages = eval_data["counts"], eval_data["language"]
    pooled = normalize(counts.sum(0, keepdims=True)) / 4
    for arm, plan in maps[mix].items():
        homes = plan.get("home_rank_by_language")
        demand = (np.array([counts[[homes[l] == i for l in languages]].sum(0)
                            for i in range(4)]) if homes else np.repeat(pooled, 4, axis=0))
        demand = normalize(demand)
        load_ratios, remotes = [], []
        for i, (primary, copies) in enumerate(zip(
            plan["primary_rank_by_layer_expert"], plan["replicas_by_layer_rank"]
        )):
            work = fit_load(demand[:, i], primary, copies)
            assert abs(work.sum() - 1) < 1e-9
            load_ratios.append(4 * work.max())
            on_home = sum(demand[g, i, np.asarray(primary) == g].sum() +
                          demand[g, i, copies[g]].sum() for g in range(4))
            remotes.append(1 - on_home)
        rows.append({"mix": mix, "arm": arm, "evaluation_sequences": len(counts),
                     "remote_call_share_mean": float(np.mean(remotes)),
                     "mean_layer_max_over_mean": float(np.mean(load_ratios)),
                     "max_layer_max_over_mean": float(max(load_ratios))})
with (directory / "local-replay.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
