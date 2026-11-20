# Sarvam-30B four-L40S load-cap follow-up

The local placement audit and held-out replay are complete. The Modal benchmark was pending GPU allocation at this snapshot. The table below comes from recorded routing counts.

## Local results

The planner holds primary assignments and request home GPUs fixed. It replaces the fewest replica slots needed to meet a per-layer load cap on fit traces with local replicas enabled. Ties favor locality. All plans retain two replicas per GPU per MoE layer. Load is executed expert calls relative to the mean across the four ranks.

The 1.25× cap changes three Indic-only layers. It leaves the full-mix layout intact because that plan already peaks at 1.201× on the fit traces. The stronger 1.15× sensitivity changes nine Indic-only layers and seven full-mix layers.

| Traffic | Layout | Held-out remote expert calls | Worst held-out layer imbalance |
|---|---|---:|---:|
| Indic only | Blind | 60.77% | 1.256× |
| Indic only | Aware | 56.71% | 1.270× |
| Indic only | Aware, 1.25× cap | 56.75% | 1.248× |
| Indic only | Aware, 1.15× cap | 57.62% | 1.220× |
| Full mix | Blind | 61.73% | 1.284× |
| Full mix | Aware, unchanged at 1.25× | 56.08% | 1.290× |
| Full mix | Aware, 1.15× cap | 56.32% | 1.244× |

The replay covers 2,500 held-out Sarvam FLORES sequences for the full mix and 2,250 for Indic-only. Blind assigns aggregate demand evenly across ranks. The cap applies to fit traces. Held-out and live load can exceed it.

## GPU measurements planned

The four-L40S PyTorch/NCCL benchmark compares the blind, aware and capped layouts at equal replica memory. Each trial replays 64 fixed decode inputs per request. Three matched repeat blocks use rotated trial order and include a blind A/A repeat. A separate 12-step diagnostic pass times NCCL size exchange, dispatch and combine as a share of slowest-rank step time. The primary run records logical MB per token, median and p95 inter-token latency, and observed rank work.

[Fit audit](fit-audit.json) · [Local replay CSV](local-replay.csv) · [Frozen maps](cap-maps.json) · [Modal benchmark source](../../../gpu_pilot/modal_sarvam_cap.py) · [Prior GPU pilot](../modal-sarvam-decode/REPORT.md)
