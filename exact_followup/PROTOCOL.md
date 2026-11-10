# Exact-request follow-up protocol

Data imported locally from upstream commit d2a1f96427d769e817921bdcb5aa7652647fb2c8. No merge overwrites earlier work. Output: results/raj-exact-followup/2026-11-09. This is a new local analysis, not a GPU/model run.

## Exact recorded-request shuffle

- Sarvam Wikipedia and FLORES for Qwen3/Sarvam; full ten-language mix and Indic-only nine-language mix. Qwen3 Wikipedia has only English per sequence and is excluded from exact shuffle. Code-mixed records have no fit/evaluation design and are not included here.
- Validate every sequence's expert total against top-k × valid tokens and index alignment. No text or token IDs are read.
- Four simulated GPUs; fit home groups chosen from 300 seeded balanced-language-count candidates on fit data only, separately per dataset, mix and weighting. Hold the resulting group capacities fixed during controls. Shuffle each real sequence's group label, preserving the exact number of requests in every group, independently within fit and evaluation splits. Refit expert placement and copies to shuffled fit counts; evaluate on shuffled held-out records. Save all assignments and seeds. Original frozen-plan replay is also recorded. This is a conditional home-group randomization; it does not rerun the 300-candidate language-group search for each randomized labeling.
- 100 draws, seed91026; R2/R4 at identical primary/replica capacities. Whole sequence expert histograms are unchanged. Each logical expert remains accessible.
- Primary weighting sums actual expert work across equally sampled requests, so longer requests contribute more. Equal-sequence sensitivity divides each histogram by its valid token length, giving equal routing-distribution weight per request. Compare each shuffled condition with its own unshuffled weighting baseline; do not compare these to prior equal-language-token headlines as if the estimands were identical.
- FLORES global shuffles are accompanied by within-parallel-sentence-index shuffles. Both preserve request-group cardinalities. Group token totals can change under shuffling; they are not constrained. Sentence-blocking retains the parallel-content structure. Null draw quantiles are descriptive permutation ranges, not population confidence intervals; no formal pipeline-wide significance claim is made.

## Locality–imbalance curves

- New per-sequence datasets: full mix, 4/8 simulated GPUs, all MoE layers; fit groups fixed across cap sweep. Greedy caps1.1/1.25/1.5/2/none, R0/2/4; primary-load cap before copies as in original B. Show held-out local-hit share against mean/worst-layer expert imbalance. No optimum or held-out cap guarantee is claimed.
- Also replot the prior Qwen3 Wikipedia aggregate-count sweep separately, preserving its earlier equal-language-token weighting and data provenance. This dataset supports curves but not the requested exact shuffle.

## Rebuild intervals in tokens

Re-express the previous synthetic equal-work scenarios with a declared window of128requests ×512inputtokens =65,536inputtokens. Periods1/2/4/8/16/32 become65,536through2,097,152inputtokens. One-window observation lag equals65,536inputtokens. This is an explicit scenario scale, not inferred production traffic. No new temporal observations, output-token counts, migration costs or seconds-based claim are introduced. All previous relative-saving values are retained exactly.

All results are recorded expert-hit locality/load proxies. BF16 batch-sensitive routing is held frozen as recorded; this replay does not show routing after changing batch composition. Sarvam's shared expert is excluded as in the source counts. No measured network bytes, all-to-all calls, latency, model quality or novelty claim follows from this analysis.
