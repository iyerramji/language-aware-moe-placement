# SGLang feasibility for Sarvam-30B on four L40S GPUs

**Decision:** Hold the Modal benchmark. The upstream SGLang configuration inspected has an explicit expert-map loader, but the Sarvam-30B forward path has no demonstrated NVIDIA dispatch backend that provides local replica selection and selective token transfers on L40S. This was a source audit. Calibration and the three decode comparisons remain unmeasured.

SGLang source revision: `c80444d5510797c6a1b0daefb816ab8ab3ebd3d3`. Sarvam checkpoint revision: `071ae95e933605ca1104a6b4524a36a98488efa4`.

| Requirement | Evidence |
|---|---|
| Load the custom expert placement | SGLang reads a physical-to-logical expert map via its initial expert-location setting. The fused MoE weight loader can resolve multiple physical copies for a checkpoint expert. Sarvam-30B inherits Bailing, whose map metadata uses the model's 19 decoder-layer indices. Converting our maps needs a row for the dense first layer so the 18 routed layers stay aligned. |
| Use a local replica when available | SGLang's map builder can prefer a same-GPU expert for an all-to-all backend. This rank-specific preference is skipped for the default `none` backend because ranks must agree on the expert choice. |
| Execute selective dispatch on four L40S GPUs | At the inspected revision, Sarvam-30B inherits Bailing's MoE forward path. The special dispatch branch checks `is_deepep()` or Ascend. The predicate matches the specific DeepEP backend, not every backend present in SGLang. Upstream DeepEP lists Hopper or newer and NVLink for intra-node operation. NVIDIA lists L40S as Ada with PCIe and no NVLink. The generic SGLang backend list does not establish Sarvam-30B compatibility for this path. |
| Produce a fair three-way comparison | Pending a common compatible backend. Changing backends between arms would mix placement effects with transport and kernel changes. |

A viable next engineering step is to wire a PCIe-capable selective dispatch/combine backend into the Sarvam-30B path. First verify one deterministic MoE layer with local and remote expert selections, including weight identity, output parity and sender-side byte counts. Before model timing, verify request homes, replica destinations and the full expert map on each rank. Then compare engine default, language-blind and balanced language-aware using the same serving backend and workload. Match replica capacity for the last two and collect decode all-to-all bytes per generated token and inter-token latency. If the backend itself requires a patch for all three, identify A as the default expert layout on that shared backend and disclose the patch.

## Sources

- [Sarvam model class](https://github.com/sgl-project/sglang/blob/c80444d5510797c6a1b0daefb816ab8ab3ebd3d3/python/sglang/srt/models/sarvam_moe.py#L1366)
- [Bailing MoE dispatch](https://github.com/sgl-project/sglang/blob/c80444d5510797c6a1b0daefb816ab8ab3ebd3d3/python/sglang/srt/models/bailing_moe.py#L346-L372)
- [SGLang expert mapping and local preference](https://github.com/sgl-project/sglang/blob/c80444d5510797c6a1b0daefb816ab8ab3ebd3d3/python/sglang/srt/eplb/expert_location.py#L585-L639)
- [SGLang expert weight loading](https://github.com/sgl-project/sglang/blob/c80444d5510797c6a1b0daefb816ab8ab3ebd3d3/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1260-L1300)
- [DeepEP hardware requirements](https://github.com/deepseek-ai/DeepEP#requirements) and [NVIDIA L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/)
