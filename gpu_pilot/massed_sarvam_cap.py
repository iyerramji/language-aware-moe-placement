import json
from pathlib import Path
from aws_sarvam_cap import stage
from sarvam_cap_runner import benchmark

folder = Path.cwd()
checkpoint = stage(folder / "checkpoint")
payload = benchmark(checkpoint, folder / "cap-maps.json")
payload["platform"] = "Massed Compute gpu_4x_l40s, Kansas City"
output = folder / "gpu-result.json"
temp = output.with_suffix(".tmp")
temp.write_text(json.dumps(payload))
temp.replace(output)
raise SystemExit(payload["status"] != "completed")
