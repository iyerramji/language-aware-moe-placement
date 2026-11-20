"""Change as few replica slots as possible to cap final fit work."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_followup"))
from common import by_language, load, normalize

MIXES = ("full", "indic_only")
CAPS = (1.25, 1.15)


def fit_demand(mix, homes):
    data, languages, _ = load("sarvam", "flores", indic=mix == "indic_only")
    fit = data["fit"]
    profiles = normalize(by_language(fit["counts"], fit["language"], languages))
    return np.array([profiles[[i for i, name in enumerate(languages) if homes[name] == rank]].sum(0)
                     for rank in range(4)])


def fit_load(demand, primary, replicas):
    destination = np.tile(primary, (4, 1))
    for rank, local in enumerate(replicas):
        destination[rank, local] = rank
    return np.bincount(destination.ravel(), weights=demand.ravel(), minlength=4)


def choose_copies(demand, primary, old, cap):
    current = fit_load(demand, primary, old)
    limit = cap * demand.sum() / 4
    if current.max() <= limit + 1e-9:
        return old
    slots = [(g, e) for g in range(4) for e, p in enumerate(primary) if g != p]
    base = fit_load(demand, primary, [[], [], [], []])
    capacity = np.zeros((4, len(slots)))
    load_delta = np.zeros_like(capacity)
    objective = []
    for i, (g, e) in enumerate(slots):
        capacity[g, i] = 1
        load_delta[g, i] = demand[g, e]
        load_delta[primary[e], i] = -demand[g, e]
        # Total normalized locality gain is <=1; a changed slot costs 2.
        objective.append(2 * (e not in old[g]) - demand[g, e])
    result = milp(
        objective, integrality=np.ones(len(slots)), bounds=Bounds(0, 1),
        constraints=[LinearConstraint(capacity, 2, 2),
                     LinearConstraint(load_delta, -np.inf, limit - base)],
        options={"time_limit": 20},
    )
    if not result.success:
        raise RuntimeError(f"Replica-only cap {cap} has no certified solution: {result.message}")
    replicas = [[] for _ in range(4)]
    for value, (g, e) in zip(result.x, slots):
        if value > 0.5:
            replicas[g].append(e)
    assert [len(row) for row in replicas] == [2] * 4
    assert fit_load(demand, primary, replicas).max() <= limit + 1e-8
    return replicas


def main(dest):
    source = ROOT / "results/gpu-pilot/modal-l40s-readiness/frozen-maps.json"
    maps = json.loads(source.read_text())
    evidence = {}
    for mix in MIXES:
        plan = maps[mix]["language_aware"]
        d = fit_demand(mix, plan["home_rank_by_language"])
        old = plan["replicas_by_layer_rank"]
        primary = plan["primary_rank_by_layer_expert"]
        for cap in CAPS:
            new = []
            for layer in range(18):
                new.append(choose_copies(d[:, layer], primary[layer], old[layer], cap))
            name = "aware_cap_" + str(cap).replace(".", "")
            maps[mix][name] = dict(plan, replicas_by_layer_rank=new)
            old_ratios = [float(4 * fit_load(d[:, l], primary[l], old[l]).max()) for l in range(18)]
            new_ratios = [float(4 * fit_load(d[:, l], primary[l], new[l]).max()) for l in range(18)]
            changed = [l + 1 for l in range(18) if old[l] != new[l]]
            evidence[f"{mix}/{name}"] = {
                "cap": cap, "changed_decoder_layers": changed,
                "replica_slots_replaced": sum(len(set(new[l][g]) - set(old[l][g]))
                                              for l in range(18) for g in range(4)),
                "fit_old_max_over_mean_by_layer": old_ratios,
                "fit_new_max_over_mean_by_layer": new_ratios,
                "fit_old_max": max(old_ratios), "fit_new_max": max(new_ratios),
            }
            print(mix, name, changed, max(old_ratios), max(new_ratios), flush=True)
    (dest / "cap-maps.json").write_text(json.dumps(maps, separators=(",", ":")) + "\n")
    (dest / "fit-audit.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
