# Revised results: L40S and H200

Hi Raj,



Thanks for the corrections. Here is the revised paragraph and combined table.



On four L40S GPUs, with fixed continuations and equal replica memory, uncapped and load-capped language-aware placement reduced logical all-to-all bytes per generated token by 5.09% and 4.99%, respectively, on the full traffic mix and by 2.06% and 2.02% on Indic-only traffic, relative to language-blind copies. The capped plan’s paired median inter-token latency was 1.12% lower on the full mix and 0.37% higher on Indic-only; these measurements did not establish a decode speedup. In the separate four-H200 NVLink replay experiments, uncapped language-aware placement reduced decode communication by approximately 26–32% relative to the default layout, a different baseline from the L40S comparison. Because the H200 repeat check failed in several decode cells, we claim no change in decode. The H200 prefill results also show the cost of imbalance: uncapped language-aware placement made the layer slower in five of six cells, by up to 50%. The load cap reduced layer time by 8–15% relative to the default, but language-blind copies reduced it by 13–19% and beat the capped plan in all six cells. On L40S, the cap lowered the planned peak load but barely changed live mean layer-step imbalance: 1.595 to 1.594 on Indic-only and 1.526 to 1.530 on the full mix. This failure to improve realized balance helps explain why the static cap did not establish a latency benefit. Instrumented NCCL intervals occupied about 22–29% of the L40S diagnostic decode step in the blind and capped comparisons, including waiting for the slowest GPU. Their share was noisy: it fell from 29.0% to 22.0% between blind and capped placement on the full mix, while identical blind layouts measured 27.1% and 22.6% on Indic-only. These intervals therefore do not justify a proportional conversion from byte savings to step-time savings. The remainder of the step includes model computation and host work, which were not timed separately.

| Hardware and measurement              | Traffic                      | Placement                       | Baseline                           | Communication reduction   | Time result                                                        |
| ------------------------------------- | ---------------------------- | ------------------------------- | ---------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| 4 × L40S, full-model decode           | Full mix                     | Uncapped aware                  | Blind copies, equal replica memory | 5.09% logical bytes/token | Median ITL −1.56%; no established speedup                          |
| 4 × L40S, full-model decode           | Full mix                     | Capped aware, 1.15× sensitivity | Blind copies, equal replica memory | 4.99% logical bytes/token | Median ITL −1.12%; no established speedup                          |
| 4 × L40S, full-model decode           | Indic-only                   | Uncapped aware                  | Blind copies, equal replica memory | 2.06% logical bytes/token | Median ITL −0.28%; no established speedup                          |
| 4 × L40S, full-model decode           | Indic-only                   | Capped aware, 1.25×             | Blind copies, equal replica memory | 2.02% logical bytes/token | Median ITL +0.37%; no established speedup                          |
| 4 × H200, NVLink decode replay        | Six model/text/traffic cells | Uncapped aware                  | Default layout                     | 25.7–32.2% payload        | We claim no change in decode; repeat check failed in several cells |
| 4 × H200, NVLink prefill layer replay | Six model/text/traffic cells | Uncapped aware                  | Default layout                     | 26.2–33.0% payload        | Slower in five of six cells, by up to 49.7%                        |
| 4 × H200, NVLink prefill layer replay | Same six cells               | Capped aware, 1.25×             | Default layout                     | —                         | Layer time 8–15% lower; slower than blind copies in all six        |
| 4 × H200, NVLink prefill layer replay | Same six cells               | Blind copies                    | Default layout                     | —                         | Layer time 13–19% lower; beats capped aware in all six             |

Negative time changes mean faster; positive changes mean slower. L40S ITL changes are medians of paired repeat changes. The L40S cap reduced the fit peak from 1.282 to 1.243 on Indic-only and from 1.201 to 1.149 on the full mix, while live imbalance changed as reported above.



The H200 cells cover Qwen3/FLORES, Sarvam/Wikipedia, and Sarvam/FLORES, each with Indic-only and mixed traffic. H200 layer timings include replayed communication and a FLOP-matched expert FFN; L40S timings measure Sarvam-30B full-model decode in the research runtime. These are complementary experiments, not a controlled hardware comparison. A dash means the communication reduction is not specified in this table.



Thanks,\
Ramji
