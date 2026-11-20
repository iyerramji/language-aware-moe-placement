"""Stage the pinned Sarvam checkpoint and run the same benchmark on EC2."""
import argparse
import json
import os
from pathlib import Path

from sarvam_cap_runner import REVISION, benchmark


def stage(folder):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(
        "sarvamai/sarvam-30b", revision=REVISION, local_dir=folder,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.py",
                        "*.json", "*.model", "*.jinja", "*.txt"],
        max_workers=12,
    ))
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    shards = {name: (folder / name).stat().st_size for name in set(index["weight_map"].values())}
    assert len(shards) == 26 and sum(shards.values()) >= index["metadata"]["total_size"]
    (folder / "stage-complete.json").write_text(json.dumps({
        "checkpoint_revision": REVISION, "shards": dict(sorted(shards.items()))
    }))
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("placement", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--instance-type", choices=("g6e.12xlarge", "g6e.24xlarge"),
                        required=True)
    args = parser.parse_args()
    folder = stage(args.checkpoint)
    result = benchmark(folder, args.placement)
    output = args.output.with_suffix(".tmp")
    output.write_text(json.dumps(dict(result, platform=f"AWS EC2 {args.instance_type}")))
    output.replace(args.output)
    raise SystemExit(result["status"] != "completed")
