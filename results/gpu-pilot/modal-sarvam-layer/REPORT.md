# Sarvam-30B PCIe dispatch validation

## Test we submitted

One Modal worker with four L40S GPUs and PyTorch 2.8.0/CUDA 12.8. The standalone dispatcher uses variable-size NCCL all-to-all to send the inputs to remote experts and return their outputs. The probe checks the GPU type and records the topology before executing.

The fixture downloads selected tensors directly from the pinned [Sarvam-30B checkpoint](https://huggingface.co/sarvamai/sarvam-30b/tree/071ae95e933605ca1104a6b4524a36a98488efa4). It uses experts 0–5 and the shared expert from decoder layer index 1. The checkpoint stores these weights in FP32. The probe uses BF16 for expert computation and transport. Each of four ranks originates one synthetic hidden state with six prescribed expert selections and fixed combination weights. This isolates placement and dispatch from router choices and the rest of decoding.

Both runs use the same primaries and inputs. The second adds two copies per rank and dispatches to a local expert whenever one exists. The test checks per-rank destinations against the fixture, compares the actual outputs with recomputation using the checkpoint weights, and checks the transmitted byte counts from buffer sizes against independently calculated counts. The counters measure logical outgoing bytes. They count the BF16 activations and the expert-ID metadata in dispatch, BF16 outputs in combine, and report the size exchange separately. Physical link traffic, NCCL protocol overhead and model latency are outside this counter.

## Expected values and pass criteria

The numbers below were computed locally from the fixture and are the expected values for the GPU run.

| Fixture | Remote expert calls | Dispatch bytes | Combine bytes | Split metadata bytes |
|---|---:|---:|---:|---:|
| Primaries only | 18 | 147,528 | 147,456 | 96 |
| Two local copies per rank | 10 | 81,960 | 81,920 | 96 |

A passing cloud result must show four L40S GPUs, the expected destinations and bytes, and agreement with reference outputs. These are deliberately forced cases for validation. The expected drop in bytes is a property of the fixture.

## Reproduction

- [Modal entry point](../../../gpu_pilot/modal_sarvam_layer.py)
- [NCCL and layer assertions](../../../gpu_pilot/sarvam_layer.py)
- [Earlier feasibility review](../modal-l40s-readiness/SGLANG-FEASIBILITY.md)

The three-arm benchmark will need a full-model adapter with verified request homes, frozen maps and per-token decode timing on the same backend in all arms. The standalone layer probe is the prerequisite for that integration and its measurements must be reported separately.

## Local CPU verification

The same expert computation and dispatch fixture passed in four CPU processes using Gloo over loopback. We staged all 21 checkpoint tensors and preserved their SHA-256 hashes. All four outputs exactly matched the single-rank recomputation in both cases. The destinations and logical send counts matched every expected value above. Eight of the 24 prescribed expert calls moved from a remote primary to a local replica.

The dispatch-plus-combine payload fell from 294,984 to 163,880 logical bytes for four synthetic input tokens. The size exchange added 96 bytes in both cases. This result verifies the protocol and counting logic on CPU. PCIe transport and end-to-end decode remain a separate check.

- [Local CPU result and tensor hashes](local-cpu-result.json)
- [Local CPU runner](../../../gpu_pilot/local_sarvam_layer.py)

## Modal result: four L40S GPUs

**The standalone layer test passed.** Modal app `ap-Sp7T4mBG6ctXzoS7EiPIyL` ran the first Sarvam routed layer fixture on four NVIDIA L40S GPUs. All 21 tensor hashes match the local reference. Each rank obtained exactly matching BF16 expert outputs in both cases. The measured software buffer counts match the expected table.

| Configuration | Remote expert calls | Dispatch + combine logical bytes | Logical bytes including size exchange |
|---|---:|---:|---:|
| Primaries only | 18 | 294,984 | 295,080 |
| Two local copies per rank | 10 | 163,880 | 163,976 |

CUDA reported peer access unavailable between these devices. The topology utility exited with status 255. NCCL logged SHM transport, consistent with host staging. These figures describe logical payload on the measured GPU run. They cannot be used as physical PCIe-link byte counts. The model router and decode loop were outside the layer fixture. There is no measured inter-token latency or three-arm result.

The earlier L40S attempts failed on topology diagnostics and a missing NumPy dependency in diagnostic result collection. The successful run recovered both issues. The A10 fallback was stopped before execution once four-L40S evidence was available.

- [Successful Modal result, tensor hashes and NCCL transport logs](layer-result.json)
- [Successful run progress](run-progress.log)

## Serving-engine alternatives

1. **Sarvam model code plus the validated PyTorch/NCCL dispatcher.** This is the most direct route to the three policies on L40S. Extend the dispatcher to all MoE layers, preserve the checkpoint router and KV cache, and use the same transport for all arms. Define A as the default expert layout on this research runtime. A separate stock-engine comparison would be needed to answer a stock-engine performance question. Host staging is part of the observed Modal setup.
2. **Patch an existing serving engine.** SGLang already has Sarvam model support and explicit expert maps. Integrating the validated dispatcher would retain its serving machinery and require verifying scheduling, weight placement and replica selection. See the [SGLang feasibility notes](../modal-l40s-readiness/SGLANG-FEASIBILITY.md).
3. **TensorRT-LLM.** NVIDIA documents L40S hardware support and frozen per-layer slot assignments with `initial_global_assignments` and `layer_updates_per_iter: 0`. The supported-model list and source tree at `b963a7ab75650e961b7eb4e7abb6a968e298117c` do not show native Sarvam-30B or Bailing support. The specialized NCCL-EP backend lists Hopper and Blackwell as tested configurations. TensorRT-LLM would still require a model adapter and a verified selective transport and local replica policy. [Model list](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html) · [Static expert assignment](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer) · [NCCL-EP requirements](https://github.com/NVIDIA/nccl-extensions/tree/main/nccl_ep)
4. **ThunderEP.** A research backend specifically evaluated on PCIe-connected RTX 4090/5090. It is a useful lead for host-staged communication, with no verified drop-in Sarvam/L40S deployment in this review. [ThunderEP paper](https://arxiv.org/abs/2609.40093)
