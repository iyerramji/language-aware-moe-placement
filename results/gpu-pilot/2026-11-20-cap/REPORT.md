# Sarvam-30B fixed-continuation placement follow-up

**Completed** on four L40S GPUs via Massed Compute (Kansas City). 27 trials, 32,640 measured decode token positions.

All runs use Sarvam-30B, BF16, PyTorch/NCCL, a fixed two-copy budget, and decode-only measurements on the same worker. The [prior research-default pilot](../modal-sarvam-decode/REPORT.md) used free generation.

## Findings for Raj

- **Capped-aware used fewer logical bytes than blind at the same replica memory.** The paired median reduction was **2.02%** for the Indic-only 1.25× plan and **4.99%** for the full-mix 1.15× sensitivity. The uncapped aware plan achieved similar byte reductions.
- **Adding the cap did not establish a decode speedup.** Versus blind, median paired p50 inter-token latency moved **+0.37%** for Indic-only at 1.25× and **−1.12%** for the full mix at 1.15×. The blind A/A repeats spanned −1.65% to +2.76% and −0.43% to +1.79%, respectively. Compare the paired changes below when choosing between layouts. Ratios of the individual arm medians can give a different impression.
- **The static cap improved fit balance and carried over weakly to live decode.** Indic-only mean layer-step imbalance was 1.595× without the cap and 1.594× at 1.25×, with the same 2.194× p95. Full-mix mean imbalance was 1.526× without the cap and 1.530× at 1.15×. These were fit-trace caps, so they did not enforce a maximum on realized decode steps.
- **NCCL intervals accounted for about 22–29% of the diagnostic decode step in the main blind and capped comparisons.** The remaining time includes expert compute, attention, and host work. The instrumented prefix had 2.8% higher median elapsed time across trials, so infer latency from the separate unprofiled pass.

The four L40S devices were in one Kansas City VM. NVIDIA reported a PCIe host-bridge path (PHB) for all six GPU pairs. CUDA peer access was unavailable and the logged NCCL channels used SHM. The byte totals describe logical payload, and physical PCIe link traffic was not measured. Inputs, maps and checkpoint revision were verified against the raw records. The VM and temporary cloud SSH key were removed after retrieving the results.

## Local cap audit

| Mix | Cap | Changed layers | Replica slots replaced | Old fit peak | New fit peak |
|---|---:|---:|---:|---:|---:|
| full | 1.25× | 0 | 0 | 1.201× | 1.201× |
| full | 1.15× | 7 | 7 | 1.201× | 1.149× |
| indic_only | 1.25× | 3 | 3 | 1.282× | 1.243× |
| indic_only | 1.15× | 9 | 10 | 1.282× | 1.150× |

The local search retains the primary assignments and request homes. An integer program minimizes changes to existing replica slots in each violating layer, then favors local hits within that minimum. Each GPU retains exactly two copies per layer. The cap is on predicted executed expert work after local replica selection, per layer on fit data. The full-mix 1.25× layout is identical to the prior layout and is not rerun as a separate arm. The 1.15× runs are sensitivity tests selected before GPU timings.

### Held-out routing replay

This local replay uses counts from all held-out FLORES sequences. It estimates the share of remote expert calls and rank load. Blind demand is split equally among the four ranks.

| Mix | Layout | Sequences | Remote calls | Mean layer imbalance | Worst layer imbalance |
|---|---|---:|---:|---:|---:|
| full | language_blind | 2500 | 61.73% | 1.157× | 1.284× |
| full | language_aware | 2500 | 56.08% | 1.159× | 1.290× |
| full | aware_cap_125 | 2500 | 56.08% | 1.159× | 1.290× |
| full | aware_cap_115 | 2500 | 56.32% | 1.146× | 1.244× |
| indic_only | language_blind | 2250 | 60.77% | 1.132× | 1.256× |
| indic_only | language_aware | 2250 | 56.71% | 1.173× | 1.270× |
| indic_only | aware_cap_125 | 2250 | 56.75% | 1.163× | 1.248× |
| indic_only | aware_cap_115 | 2250 | 57.62% | 1.134× | 1.220× |

## GPU measurements

Metrics are medians of the three repeat summaries. Timing uses consecutive decode steps on each rank, weighted by active requests, with prefill excluded.

| Mix | Layout | Logical MB/token | p50 ITL ms | p95 ITL ms | NCCL share | Mean imbalance | p95 imbalance |
|---|---|---:|---:|---:|---:|---:|---:|
| full | aware_cap_115 | 1.196 | 104.16 | 109.59 | 22.0% | 1.530× | 1.942× |
| full | blind_aa | 1.259 | 107.30 | 111.66 | 27.2% | 1.509× | 2.100× |
| full | language_aware | 1.195 | 103.66 | 108.80 | 21.5% | 1.526× | 1.933× |
| full | language_blind | 1.259 | 107.42 | 114.22 | 29.0% | 1.509× | 2.100× |
| indic_only | aware_cap_115 | 1.247 | 102.15 | 106.60 | 23.8% | 1.603× | 2.231× |
| indic_only | aware_cap_125 | 1.245 | 97.98 | 102.87 | 23.5% | 1.594× | 2.194× |
| indic_only | blind_aa | 1.268 | 102.83 | 107.98 | 22.6% | 1.555× | 2.056× |
| indic_only | language_aware | 1.244 | 103.09 | 108.51 | 23.2% | 1.595× | 2.194× |
| indic_only | language_blind | 1.268 | 104.56 | 108.05 | 27.1% | 1.555× | 2.056× |

## Paired comparisons

- indic_only, blind_aa versus language_blind. Logical MB/token +0.00%. Median ITL change +0.64% across paired repeats (range -1.65% to +2.76%).
- indic_only, language_aware versus language_blind. Logical MB/token -2.06%. Median ITL change -0.28% across paired repeats (range -1.41% to -0.25%).
- indic_only, aware_cap_125 versus language_aware. Logical MB/token +0.05%. Median ITL change +0.62% across paired repeats (range -4.96% to +0.84%).
- indic_only, aware_cap_125 versus language_blind. Logical MB/token -2.02%. Median ITL change +0.37% across paired repeats (range -6.29% to +0.55%).
- indic_only, aware_cap_115 versus language_aware. Logical MB/token +0.34%. Median ITL change +0.21% across paired repeats (range -0.91% to +4.44%).
- indic_only, aware_cap_115 versus language_blind. Logical MB/token -1.88%. Median ITL change -0.07% across paired repeats (range -2.31% to +4.19%).
- full, blind_aa versus language_blind. Logical MB/token +0.00%. Median ITL change -0.12% across paired repeats (range -0.43% to +1.79%).
- full, language_aware versus language_blind. Logical MB/token -5.09%. Median ITL change -1.56% across paired repeats (range -3.50% to +2.11%).
- full, aware_cap_115 versus language_aware. Logical MB/token +0.06%. Median ITL change +0.47% across paired repeats (range -3.17% to +1.47%).
- full, aware_cap_115 versus language_blind. Logical MB/token -4.99%. Median ITL change -1.12% across paired repeats (range -3.04% to -0.12%).

## Where decode time went

The phase columns are separate medians and need not sum to the median total NCCL share. The last two columns compare the first 12 steps with and without event instrumentation.

| Mix | Layout | Size exchange | Dispatch | Combine | Profiled step ms | Unprofiled step ms |
|---|---|---:|---:|---:|---:|---:|
| full | aware_cap_115 | 4.2% | 5.4% | 9.7% | 111.68 | 107.70 |
| full | blind_aa | 5.0% | 6.8% | 16.0% | 114.69 | 110.73 |
| full | language_aware | 4.1% | 6.1% | 12.6% | 111.35 | 108.04 |
| full | language_blind | 4.8% | 6.4% | 17.0% | 114.33 | 110.70 |
| indic_only | aware_cap_115 | 4.4% | 6.2% | 13.6% | 104.20 | 103.23 |
| indic_only | aware_cap_125 | 3.9% | 5.7% | 12.6% | 102.56 | 99.44 |
| indic_only | blind_aa | 2.9% | 5.8% | 13.9% | 105.80 | 105.27 |
| indic_only | language_aware | 2.2% | 5.1% | 12.6% | 103.20 | 100.56 |
| indic_only | language_blind | 3.9% | 5.9% | 17.3% | 107.37 | 102.39 |

## Measurement limits

The runner uses the same pinned checkpoint, held-out FLORES batches, and 64 fixed decode inputs per request (generated once with the blind layout). Input hashes match across arms. Floating-point execution and expert selections can still vary. The language-blind duplicate provides a timing repeatability check. All comparisons use the same research runtime and the two-copy memory budget. We rotate arm order.

NCCL share is the median, across a separate 12-step diagnostic replay, of the sum of elapsed CUDA stream intervals around the size exchange, dispatch and combine, divided by wall time on the rank with the longest measured step. Stream intervals include collective synchronization and may include enqueue/wait effects. This is not physical link time or an isolated critical-path contribution. The residual step time includes expert compute, attention, and host work. The measurement does not assign that residual to Python alone. The separate unprofiled 64-step runs supply the primary latency numbers. Diagnostic overhead is checkable using the matching first-12-step wall times in the repeat CSV. The JSON records per-phase measurements and per-layer executed work.

Logical MB/token counts sender-side dispatch and combine, with expert IDs included. Split-size bytes are tracked separately. 1 MB is 1,000,000 bytes. NCCL's runtime transport and CUDA peer-access checks are in the result. Physical link counters were not measured.

Rebuild the local maps and replay with `cap_plans.py` and `replay_cap.py`, each with this results directory as its argument. To replay on a four-L40S Ubuntu VM, copy [the Massed Compute entry point](../../../gpu_pilot/massed_sarvam_cap.py), `aws_sarvam_cap.py`, `sarvam_cap_runner.py`, `sarvam_cap_decode.py`, `sarvam_decode.py`, `sarvam_layer.py` and `cap-maps.json` to `~/sarvam-cap`. Run [the bootstrap script](massed-bootstrap.sh) to install the pinned environment and execute the benchmark. Run `report_cap.py` on the retrieved records to validate them and regenerate the tables.

[Fit audit](fit-audit.json) · [Local replay CSV](local-replay.csv) · [Frozen maps](cap-maps.json) · [GPU records](gpu-result.json) · [Summary CSV](summary.csv) · [Per-repeat CSV](per-repeat.csv) · [Trial completion log](run-progress.log)
