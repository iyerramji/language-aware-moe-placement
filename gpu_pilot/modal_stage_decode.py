"""Stage the pinned public Sarvam checkpoint on CPU for the decode pilot."""
import json
from pathlib import Path

import modal

REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"
CACHE = "sarvam-placement-pilot-checkpoint"
app = modal.App("sarvam-placement-stage")
volume = modal.Volume.from_name(CACHE, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub==0.36.0")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HUB_DOWNLOAD_TIMEOUT": "90"})
)


@app.function(image=image, cpu=4, memory=4096, timeout=1200,
              retries=0, volumes={"/checkpoint": volume})
def stage() -> str:
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(
        "sarvamai/sarvam-30b", revision=REVISION, local_dir="/checkpoint/sarvam",
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.py", "*.json",
                        "*.model", "*.jinja", "*.txt"], max_workers=6,
    ))
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    sizes = {name: (folder / name).stat().st_size for name in shards}
    assert len(sizes) == 26 and sum(sizes.values()) >= index["metadata"]["total_size"]
    result = {"checkpoint_revision": REVISION, "shards": sizes}
    (folder / "stage-complete.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return json.dumps(result, indent=2)
