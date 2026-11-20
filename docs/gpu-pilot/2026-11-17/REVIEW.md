# Sarvam four-GPU decode pilot — review before execution

Prepared 17 November 2026. **PROPOSED ONLY. No cloud resources, downloads, implementation or experiments started.** This narrows the earlier plan according to Raj's latest message. Approval of this document is not assumed. Eight GPUs and migration/rebuilding are deferred.

## What we want to learn

Does our balanced language-aware placement actually make Sarvam's decode faster, and is any improvement accompanied by lower expert-communication bytes? The earlier counts establish neither. Three placements, one engine/backend, four GPUs, identical model and workload.

| Stage | Question | What changes | What we capture and how |
|---|---|---|---|
| 0: readiness and counter calibration | Can this engine implement the plans, and what does our byte counter mean? | Small known dispatch patterns, not a performance comparison | Rank topology; expert mapping and replicas; known send sizes versus instrumented communication-buffer bytes; padding/metadata and send/receive conventions. Correctness and instrumentation overhead checks. |
| 1: engine default (A) | What happens without our placement policy? | Engine's documented default expert mapping and request assignment | Decode-only bytes/token and per-request token timestamps, with final expert counts/rank, actual batch sizes and GPU memory. This is the baseline. |
| 2: language-blind (B) | Does language add value beyond extra replicas and aggregate popularity? | Plan from pooled fit routing; same replica slots and memory as C, no language-based request assignment | Same measurements. B versus C is the equal-memory comparison. Freeze both maps during timing. |
| 3: balanced language-aware (C) | Does the proposed complete policy help? | Fit-selected language homes plus balanced placement and replicas | Same measurements, plus observed remote selections and replica destinations. Compare against BOTH A and B. |

Stage 0 is a prerequisite, not evidence of a speedup. A may use less replica memory than B/C; report that explicitly, do not call A versus C an equal-memory experiment. B versus C tests the whole language-aware policy including homing, not placement alone.

## Engine and hardware decision

**Provisional choice: vLLM on one Modal allocation with four L40S GPUs**, subject to availability, actual topology and compatibility. Do not substitute NVLink hardware and label it a PCIe result. Modal documentation supports multi-GPU allocations; existing account credentials/quota and profiler access have NOT been inspected. No Modal calls are needed to approve this protocol.

vLLM documents expert parallelism, load balancing and redundant experts; the Sarvam model page exposes a vLLM serving recipe. That does not validate Sarvam custom placement with replication in a particular release. Before implementation, pin the checkpoint/tokenizer revision, vLLM commit/container, CUDA/driver and backend. Proposed layout is TP=1, DP=4 with EP across those ranks, only if Sarvam and the chosen engine path support it. Confirm attention/KV stay on the request's home rank and MoE experts are distributed across all four. This layout must match the simulator's home assumption.

Required adapter: load our per-layer logical-to-physical expert map, enforce request homes, and select a local replica when available under the declared policy. Verify that weights follow remapped experts and routing still chooses the same logical experts. Disable automatic map changes during fixed-plan runs; record default settings and any deviations. Stock EPLB is not automatically our optimizer. If the engine cannot realize the policy, stop and report the gap rather than benchmark a different method.

For R2, proposed capacity is 32 primary + 2 replica slots per GPU per MoE layer: 8 redundant slots globally per layer, not 2 globally. Verify version-specific config semantics and actual loaded slots. Sarvam's shared expert and other compute remain in timing/memory, even though earlier routed-expert counts excluded them.

Backend selection is a gate: a dense all-gather/reduce-scatter path may not reduce communication with locality. Inspect actual dispatch/combine behavior; do not infer fewer bytes from fewer remote hits. Use the same compatible backend for A/B/C. No speculative decoding or dynamic migration.

## Workload proposal for review

Start with held-out Sarvam FLORES: full ten-language mix, then Indic-only, equal request sampling. Use permitted source text and exact split/IDs; current histograms contain no prompts. Preserve fit/evaluation separation. Select the balanced cap and replica plan from fit data before timing, never choose the fastest point on evaluation results. If existing fit data is prefill-derived, label the plan accordingly and measure whether it transfers to decode; do not quietly refit on evaluated decode tokens.

Proposed small matrix: A/B/C × full/Indic × total concurrent requests 4/16 × five paired repeats = 60 measured batches. Each batch prefills the same matched prompts, then generates 128 tokens per request with a fixed deterministic decoding configuration. Prefill completes before the measured decode window; admit no new prefill work into it. Use matched prompt-length distributions and identical batches across plans, rotate plan order, warm up kernels first. Fixed output length/ignore-EOS is a benchmark choice to disclose. No prompt truncation or source substitution without recording it. Report divergent outputs rather than silently removing them.

This is an isolated decode microbenchmark, not mixed prefill/decode production serving. Budget and durations are not established yet: before any launch, provide a maximum cost/runtime and failure stop policy for approval. Start with readiness/calibration, then one small paired batch before the full matrix.

## Precisely what we measure

1. **Logical remote communication payload:** instrument the selected backend's dispatch and combine send buffers at all ranks, tagged by MoE layer and decode step. Sum off-rank bytes once on the sending side; do not add receives again. Record activation payload, routing metadata and padded/allocated bytes separately where distinguishable. Account for token-to-destination deduplication rather than multiplying expert hits by hidden size. Divide total measured decode bytes by total generated tokens across all requests, not by steps or ranks. Collective buffer sizes need backend-specific accounting; do not invent per-peer sends from a collective API's tensor size.
2. **Physical link traffic:** only report if accessible, calibrated PCIe/link counters or suitable profiler evidence can attribute it. Link totals can include unrelated traffic and protocol overhead. Label units, counter direction, scope, sampling and subtraction method. If unavailable, explicitly report logical payload only; never rename it measured PCIe bytes. Calibration records which interpretation is actually supported before performance testing.
3. **Inter-token latency (ITL):** timestamp each token becoming ready per request at the engine, excluding first-token/prefill latency. Compute consecutive-token intervals and report p50/p95/p99 plus per-repeat summaries. Streaming chunk gaps are not necessarily token gaps; any client timestamps are a separate end-to-end metric. Use device events on diagnostic passes for GPU phase durations; do not synchronize every step in the timed run.
4. **Explanation metrics:** remote logical expert hits, final replica execution counts, max/mean work per layer/rank, achieved tokens/sec, batch composition, peak memory/KV capacity, errors and output checks. Tokens per expert are not GPU time. Record backend communication timing separately where possible.

Use minimally instrumented timing runs plus paired diagnostic captures. Quantify counter overhead with counter-on/off repeats; expensive tracing is not the latency headline. Compare paired run differences and variability across repeats; do not treat every token as an independent trial. A small pilot's p99 is exploratory, with sample counts reported.

## Files and decision

New output directory only: `results/gpu-pilot/<approved-run-id>/`. Save configuration/provenance, fit plan and hash, GPU topology, counter calibration, per-request token events, per-layer/rank byte records, replica counts, repeat summaries, errors and a short report. Raw dataset text stays private. No current results folder is overwritten.

Deliver one comparison table (A/B/C bytes/generated-token, ITL p50/p95/p99, throughput, imbalance, memory) and paired plots. If C reduces remote hits but not bytes, inspect backend aggregation/padding. If bytes fall without ITL improvement, inspect compute/load/synchronization. If neither improves or timings are noisy, report that; do not proceed automatically to eight GPUs. Only a credible four-GPU latency improvement motivates proposing expansion to Raj. Rebuild plus migration remains a later separately approved experiment.

## Sources checked for planning

- [vLLM expert parallel deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/): EP, communication backends and redundant experts. Exact custom policy compatibility remains unverified.
- [Sarvam model page](https://huggingface.co/sarvamai/sarvam-30b): model identity and advertised serving path; not an EP validation.
- [Modal GPU guide](https://modal.com/docs/guide/gpu): allocation capabilities; actual account availability/topology remain unverified.
- Local previous evidence: `results/raj-exact-followup/2026-11-09/REPORT.md`.

Review decisions before execution: approve stage order and three arms; confirm permitted prompts; agree fit-selected cap/R2 plan, batch sizes/output length, byte-counter scope, engine compatibility and hard spending/runtime ceiling. No execution authorization has been given.
