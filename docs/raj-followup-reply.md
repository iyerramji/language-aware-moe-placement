Hi Raj,

I ran the follow-ups locally using Codex and kept them separate from the earlier results.

**Shuffle control**

I couldn’t run the exact request-level shuffle from the file we have. It contains counts aggregated by language, rather than counts for each request. I ran an expected random-homing control instead, keeping the group sizes. I also kept the whole-language grouping shuffles separate, since those aren’t the same test.

With two copies per GPU on four GPUs, the Indic-only remote-hit reduction went from 36.21% to 33.99% after refitting to the randomized-homing expectation. The language-blind comparator was 25.10%. On the full mix it went from 33.05% to 28.00%, against 23.07% for the language-blind comparator.

So quite a bit of the Indic-only benefit survives without the language/home association in this approximation. But I don’t think we should treat this as the exact shuffle result yet.

[Shuffle results and limitations](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/raj-followup-results/results/raj-followup/2026-11-09/shuffle/NOTES.md)

**B on eight GPUs**

I completed the greedy version across all 48 layers, with the same caps and replica budgets as before. With two copies per GPU and a fit-load cap of 1.1, it gave 26.11% remote-hit reduction and mean held-out imbalance of 1.27. The unconstrained assignment gave 31.55% and 3.15. So balance improved a lot, but the trade-off was bigger than on four GPUs. This is a greedy result, not a proven optimum.

[Eight-GPU B results](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/raj-followup-results/results/raj-followup/2026-11-09/balance8/NOTES.md)

**Rebuilding the plan**

I tested gradual, abrupt and alternating changes in synthetic language mixes. With two copies per GPU and one-block-old mix information, rebuilding every eight blocks retained at least 90% of fresh-plan saving in the gradual case; the abrupt case needed every two blocks among the intervals tested. None passed that threshold in the rapidly alternating case. These are scenario results, not a refresh interval in seconds, and they don’t include migration cost.

[Rebuild tests](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/raj-followup-results/results/raj-followup/2026-11-09/rebuild/NOTES.md)

Could you share the per-request or per-sequence routing histograms if they were retained? For each request I’d need an ID, its original language, fit/evaluation split, valid token count, and expert-hit counts for each layer. The original home-group assignment would also help. Per-token selected-expert logs would work too. We don’t need the raw text for this control.

That would let us shuffle individual requests while keeping the group sizes, refit on the shuffled fit split, and repeat the held-out test with different seeds. If those counts weren’t saved, I think we’d need to collect them before completing that control.

I’ve left family grouping out of these follow-ups. These are still count-based simulations, not measured bytes or latency. Which of these would you suggest we follow up on while we get the exact shuffle data?

[Full follow-up report, tables and figures](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/raj-followup-results/results/raj-followup/2026-11-09/REPORT.md)

Thanks,
Ramji
