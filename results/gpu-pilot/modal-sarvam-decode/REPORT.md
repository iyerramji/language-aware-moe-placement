# Sarvam-30B decode placement pilot
Modal run status **completed**. 18 completed arm/repeat records. GPU allocation 4 × NVIDIA L40S. 21,888 measured decode tokens across both mixes and all layouts. [Modal run](https://modal.com/apps/iyer-ramji-c/main/ap-dGQnv42ICYfkuklpdY8vgP).

**Main result.** The language-aware plan reduced logical bytes per token in both mixes, while median and p95 inter-token latency rose against both baselines. This pilot does not support scaling the current plan to eight GPUs on the expectation of a latency win. The timing includes the research runtime’s Python scheduling, synchronization and dispatch overhead. Different generated continuations weaken causal attribution; a fixed-continuation repeat would separate that effect from the placement.

| Traffic mix | Placement | Repeats | Logical bytes/token | ITL p50 ms | ITL p95 ms | ITL p99 ms | Local replica hits |
|---|---|---:|---:|---:|---:|---:|---:|
| full | research_default | 3 | 1,323,092.0 | 96.02 | 103.48 | 105.16 | 0 |
| full | language_blind | 3 | 1,268,738.5 | 97.25 | 110.71 | 113.71 | 14,554 |
| full | language_aware | 3 | 1,198,227.4 | 99.44 | 113.32 | 117.41 | 17,254 |
| indic_only | research_default | 3 | 1,332,146.2 | 88.86 | 98.16 | 99.80 | 0 |
| indic_only | language_blind | 3 | 1,259,372.3 | 95.24 | 103.66 | 105.65 | 13,026 |
| indic_only | language_aware | 3 | 1,237,426.8 | 98.46 | 105.02 | 108.84 | 12,200 |

- full, language-aware versus research_default: bytes/token -9.44%, median ITL +3.56%, p95 ITL +9.51%.

- full, language-aware versus language_blind: bytes/token -5.56%, median ITL +2.26%, p95 ITL +2.36%.

- indic_only, language-aware versus research_default: bytes/token -7.11%, median ITL +10.80%, p95 ITL +6.99%.

- indic_only, language-aware versus language_blind: bytes/token -1.74%, median ITL +3.38%, p95 ITL +1.32%.

The default arm uses contiguous expert IDs and zero replicas in our PyTorch/NCCL research runtime. Blind and aware each use two extra expert slots per GPU per routed layer. They share one dispatch implementation; the aware policy also homes requests by language. Default and blind use the same round-robin request homes.

Each matched repeat uses a distinct pair of held-out FLORES parallel sentences per language. For each placement, the runtime prefills the prompts, then generates 64 timed decode tokens per request. ITL uses consecutive token-ready times inside each CUDA-synchronized rank, repeated over its active requests. The first measured token has no in-window predecessor and is excluded from ITL. These are batch decode intervals from a research loop with synchronous instrumentation. Per-repeat median ranges and sample counts are in the CSV. The arm order was fixed, so repeat variability does not capture possible drift between arms. Requests execute in rank-local batches and their inter-token intervals share batch-step timing.

NCCL logged transport paths SHM on this worker. CUDA peer-access matrix and raw transport output are in the evidence file.

Byte counts include outgoing off-rank BF16 activations and int32 expert IDs on dispatch and BF16 outputs on return, counted once at each sender. The size exchange is captured separately. Each step checks bytes against the number of routed expert calls. These are logical payload bytes. NCCL transport metadata, physical link counters and prefill are outside that measurement.

The runner verifies all request homes against observed rank records and checks that replicas execute work in both replicated arms. BF16 execution at different batch sizes or expert groupings may produce different continuations; the checks below flag that source of variation.

- full, language_blind versus default: 32/60 identical continuation hashes.

- full, language_aware versus default: 30/60 identical continuation hashes.

- indic_only, language_blind versus default: 24/54 identical continuation hashes.

- indic_only, language_aware versus default: 22/54 identical continuation hashes.

Reproduce with `modal run --timestamps --write-result <output>/decode-result.json gpu_pilot/modal_sarvam_decode.py::benchmark` after staging the pinned checkpoint. Source: [Modal runner](../../../gpu_pilot/modal_sarvam_decode.py), [decode and counters](../../../gpu_pilot/sarvam_decode.py), [report builder](../../../gpu_pilot/report_decode.py).

[Run evidence](decode-result.json) · [Metrics CSV](summary.csv) · [Modal run progress](run-progress.log) · [One-layer counter validation](../modal-sarvam-layer/REPORT.md)
