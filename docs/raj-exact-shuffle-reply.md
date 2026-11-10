Hi Raj,

Thanks for adding the per-sequence counts. I pulled them and ran the shuffle locally using Codex. I kept these results separate from the earlier runs.

**Exact shuffle**

I ran Sarvam Wikipedia and FLORES for both models, on the full mix and Indic-only traffic. I shuffled fit and evaluation sequences independently, kept each group's request count fixed, and refitted placement and copies on the shuffled fit data. There were 100 draws per condition, with the fit-selected group capacities held fixed.

The original language groups beat the shuffled groups on all three datasets. For Indic-only traffic, with four GPUs and two copies per GPU, the extra remote-hit reduction was:

- Sarvam Wikipedia: **6.80 percentage points**.
- Qwen3 FLORES: **6.95 points**.
- Sarvam FLORES: **5.68 points**.

The Indic-only result surprised me most. The languages had looked similar in the earlier routing analysis, but keeping the real language groups still helped compared with random groups. Some saving survives the shuffle, so I wouldn't say all the benefit comes from language.

FLORES sentence lengths affected the size of the result. I also ran equal-sequence weighting and shuffles within matched parallel sentences; the original groups still had an advantage. I left Qwen3 Wikipedia out of the exact shuffle because only English has per-sequence counts.

[Exact-shuffle results and controls](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/exact-shuffle-results/results/raj-exact-followup/2026-11-09/REPORT.md)

**Locality versus imbalance**

I plotted the four- and eight-GPU curves over the load-cap sweep. One thing that came up: on eight-GPU Sarvam FLORES, caps of 1.1 and 1.25 are impossible under the primary-placement constraint. In one layer, the hottest expert plus the other 15 experts required on its GPU forces imbalance to at least 1.269. Those points are marked infeasible rather than treated as solver failures.

[Locality–imbalance curves](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/exact-shuffle-results/results/raj-exact-followup/2026-11-09/curves/locality_imbalance_R2.png)

**Rebuild intervals**

I expressed the earlier synthetic scenarios in input tokens. With the chosen workload scale, the gradual case retained at least 90% of fresh-plan saving at up to 524,288 tokens between rebuilds; the abrupt case needed 131,072 among the intervals tested. These are scenario-specific numbers, not a measured production interval, and they don't include migration cost.

[Token-based rebuild results](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/exact-shuffle-results/results/raj-exact-followup/2026-11-09/rebuild_tokens/table.csv)

These are still locality and expert-load results from recorded counts, not measured network bytes or latency. Do these controls look sufficient to move to the one-page plan, or is there another check you would suggest first?

Thanks,
Ramji
