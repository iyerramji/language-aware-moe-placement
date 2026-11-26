# Results

Updated 26 November 2026. This complete section supersedes the earlier paragraph and table at this link.

## Language structure in routing

English and the nine Indian languages used different expert distributions in both Qwen3-30B-A3B and Sarvam-30B. On Wikipedia, mean English–Indic Jensen–Shannon divergence was 0.318 bits for Qwen3 and 0.274 bits for Sarvam, against within-language split-half floors of 0.013 and 0.018 bits. Every MoE layer exceeded three times its model's floor. On parallel FLORES text, the divergences were 0.294 and 0.212 bits, retaining approximately 92% and 77% of the Wikipedia values. Thus the split persisted when the text content was matched across languages.

Differences among the Indian languages were smaller and more sensitive to the text. Their mean pairwise divergences fell from 0.035 to 0.017 bits on Qwen3 and from 0.071 to 0.038 bits on Sarvam between Wikipedia and FLORES. The routing signal is therefore primarily an English–Indic split. The code-mixed prediction was not supported: romanised Hinglish was closer to Hindi than English in early layers under the original reference, and its position changed with the reference text. We treat code-mixed traffic as a separate class rather than assigning it an English-like routing profile. The padding control also failed, so FLORES routing was collected one sentence at a time; a recorded route is not guaranteed to remain identical under a different batch shape.

## Simulated locality and balance

With four simulated GPUs and two extra expert copies per GPU per layer, fully language-aware placement improved the simulated traffic-reduction percentage by 9.67–11.18 percentage points over language-blind copies across the two models and the Wikipedia/FLORES datasets. Both reductions used the default layout as their reference. Nearly all of the additional benefit came from the layout and request home assignment: blind copies on the aware layout came within 0.2–0.8 percentage points of fully aware copies. The gain came with greater imbalance. On Wikipedia, busiest-rank expert work divided by mean rank work was 1.99 for aware versus 1.25 for blind placement on Qwen3, and 1.51 versus 1.23 on Sarvam.

The Qwen3/Wikipedia aggregate-count ablation showed that a balance-aware planner could retain most of the simulated locality benefit. At the same two-copy budget, the capped integer-program plan reduced remote expert hits by 32.01%, compared with 33.05% for unconstrained aware placement, while mean layer imbalance fell from 1.987 to 1.261. These percentages use a no-copy, uniform-origin reference. The fit constraint applies to primary load before replica dispatch and does not guarantee a bound on live executed work.

Exact recorded-request shuffles supported an additional benefit from the original language/home association. On four simulated GPUs with two copies and token-work weighting, original groups exceeded the mean of refitted shuffled groups by 6.80 percentage points of remote-hit reduction on Indic-only Sarvam/Wikipedia, 6.95 on Qwen3/FLORES, and 5.68 on Sarvam/FLORES. Equal-sequence weighting and sentence-matched FLORES shuffles retained positive differences. The control preserves group request counts and uses independently shuffled fit and evaluation records; it is conditional on the selected home-group capacities, not a pipeline-wide significance test. These simulations measure remote expert-hit proxies and load, not network traffic or latency.

## H200 replay: fewer bytes, provisional timing directions

In the four-H200 NVLink replay experiments, uncapped language-aware placement reduced communication payload by 25.7–32.2% in decode and 26.2–33.0% in prefill relative to the default layout. The capped plan reduced prefill payload by 22.0–27.1%, and language-blind copies reduced it by 17.3–24.4%, also relative to the default. These baselines differ from the blind-copy baseline used for the L40S comparisons below.

The H200 repeat check failed in 3 of 12 cells, almost all in decode; under the run's decision rule, every latency verdict from that run is not trusted, including prefill. We therefore claim no change in decode and report the prefill timings as consistent observed directions, not established effects. Uncapped aware placement showed a slower layer in five of six prefill cells, by up to 49.7%; the remaining cell's interval included zero. The capped plan showed 8–15% lower layer time than the default, while blind copies showed 13–19% lower layer time and were faster than the capped plan in all six cells. Thus blind copies saved fewer bytes but had the lowest observed layer times, subject to the run-wide timing qualification.

## L40S fixed-continuation decode and cap audit

On four L40S GPUs, with fixed continuations and equal replica memory, uncapped and load-capped language-aware placement reduced logical all-to-all bytes per generated token by 5.09% and 4.99%, respectively, on the full traffic mix and by 2.06% and 2.02% on Indic-only traffic, relative to language-blind copies. The capped plan's paired median inter-token latency was 1.12% lower on the full mix and 0.37% higher on Indic-only. These measurements did not establish a decode speedup.

The static cap lowered planned peak load but barely changed live mean layer-step imbalance. On Indic-only, the 1.25× cap reduced the fit peak from 1.282 to 1.243, while live imbalance changed from 1.595 to 1.594. On the full mix, a 1.25× cap left the plan unchanged, so the measured 1.15× plan was a sensitivity test; it reduced the fit peak from 1.201 to 1.149, while live imbalance changed from 1.526 to 1.530. The cap constrained predicted executed expert work on fit data after replica selection, not the realized load of every decode step. Its failure to improve realized balance helps explain why it did not establish a latency benefit.

Instrumented NCCL intervals occupied about 22–29% of the L40S diagnostic decode step in the blind and capped comparisons, including waiting for the slowest GPU. Their share was noisy: it fell from 29.0% to 22.0% between blind and capped placement on the full mix, while identical blind layouts measured 27.1% and 22.6% on Indic-only. These intervals do not justify a proportional conversion from byte savings to step-time savings. They include synchronization and waiting and are not isolated physical link time. The remainder of the step includes model computation and host work, which were not timed separately. Primary latency results come from the separate unprofiled runs.

## Combined hardware results

Changes are relative to the baseline named in each row. Negative time changes mean faster; positive changes mean slower. L40S inter-token latency (ITL) changes are medians of paired repeat changes, not ratios of arm medians.

| Hardware and measurement | Traffic | Placement | Baseline | Communication reduction | Time result |
|---|---|---|---|---|---|
| 4 × L40S, full-model decode | Full mix | Uncapped aware | Blind copies, equal replica memory | 5.09% logical bytes/token | Median ITL −1.56%; no established speedup |
| 4 × L40S, full-model decode | Full mix | Capped aware, 1.15× sensitivity | Blind copies, equal replica memory | 4.99% logical bytes/token | Median ITL −1.12%; no established speedup |
| 4 × L40S, full-model decode | Indic-only | Uncapped aware | Blind copies, equal replica memory | 2.06% logical bytes/token | Median ITL −0.28%; no established speedup |
| 4 × L40S, full-model decode | Indic-only | Capped aware, 1.25× | Blind copies, equal replica memory | 2.02% logical bytes/token | Median ITL +0.37%; no established speedup |
| 4 × H200, NVLink decode replay | Six model/text/traffic cells | Uncapped aware | Default layout | 25.7–32.2% payload | We claim no change in decode [^h200] |
| 4 × H200, NVLink prefill layer replay | Six model/text/traffic cells | Uncapped aware | Default layout | 26.2–33.0% payload | Observed slower in five of six cells, by up to 49.7% [^h200] |
| 4 × H200, NVLink prefill layer replay | Same six cells | Capped aware, 1.25× | Default layout | 22.0–27.1% payload | Observed layer time 8–15% lower; slower than blind copies in all six [^h200] |
| 4 × H200, NVLink prefill layer replay | Same six cells | Blind copies | Default layout | 17.3–24.4% payload | Observed layer time 13–19% lower; faster than capped aware in all six [^h200] |

[^h200]: The four-H200 repeat check failed in 3 of 12 cells, almost all in decode. Under the run's decision rule, **every latency verdict from that run is not trusted, prefill included**. The prefill timings show consistent observed directions, not established effects. The payload reductions remain reported separately from these timing verdicts.

The H200 cells cover Qwen3/FLORES, Sarvam/Wikipedia, and Sarvam/FLORES, each with Indic-only and mixed traffic. H200 layer timings include replayed communication and a FLOP-matched expert FFN; L40S timings measure Sarvam-30B full-model decode in the research runtime. These are complementary experiments, not a controlled hardware comparison. L40S byte totals describe logical payload; physical link traffic was not measured. Across these experiments, lower communication volume alone did not establish faster decode, and a cap that improved planned balance did not necessarily improve live balance.

## Sources for review

- [Routing and parallel-text results](exp02-results.md).
- [Aggregate-count ablations](../results/organized-experiments/raj-requested/FINDINGS.md) and [exact recorded-request follow-up](../results/raj-exact-followup/2026-11-09/REPORT.md).
- [L40S fixed-continuation report, paired comparisons, cap audit, and measurement limits](../results/gpu-pilot/2026-11-20-cap/REPORT.md).
- H200 replay results: supplied paper draft, Results and Limitations (pp. 3–5), with Raj's exp03 verification and corrections supplied on 26 November 2026. The capped and blind prefill byte ranges and run-wide timing interpretation follow that correction; the underlying exp03 raw records were not independently reanalysed for this revision.

No new experiments were run for this revision.
