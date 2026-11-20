"""Prepare fixed four-GPU maps using the existing Sarvam FLORES fit counts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "exact_followup"))
from common import load, by_language, groups_fit, normalize, placement, copies


def make_plans(indic_only):
    data, languages, _ = load("sarvam", "flores", indic=indic_only)
    fit = data["fit"]
    counts = by_language(fit["counts"], fit["language"], languages)
    groups, _ = groups_fit(counts, G=4)
    profiles = normalize(counts)
    aware = np.array([profiles[group].sum(0) for group in groups])
    blind = np.repeat(aware.sum(0, keepdims=True) / 4, 4, axis=0)
    plans = {}
    for name, demand in (("language_blind", blind), ("language_aware", aware)):
        primary = placement(demand, cap=1.1)
        replicas = copies(demand, primary, 2, blind=name == "language_blind")
        assert primary.shape == (18, 128)
        for layer in range(18):
            assert np.bincount(primary[layer], minlength=4).tolist() == [32] * 4
            assert replicas[layer].sum(1).tolist() == [2] * 4
            assert not replicas[layer, primary[layer], np.arange(128)].any()
        plans[name] = {
            "primary_rank_by_layer_expert": primary.tolist(),
            "replicas_by_layer_rank": [
                [np.flatnonzero(rank).tolist() for rank in layer]
                for layer in replicas
            ],
        }
    plans["language_aware"]["home_rank_by_language"] = {
        languages[i]: rank for rank, group in enumerate(groups) for i in group
    }
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = PROJECT / "data/per_sequence/sarvam_per_sequence.npz"
    result = {
        "source": "Sarvam FLORES fit only",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "primary_load_cap": 1.1,
        "replicas_per_rank_per_layer": 2,
        "full": make_plans(False),
        "indic_only": make_plans(True),
    }
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
