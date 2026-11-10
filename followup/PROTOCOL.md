# Raj follow-up: local count-based checks

New output folder: results/raj-followup/2026-11-09. Earlier A–F and published reports remain unchanged. No family extension, joint planner, shared FFN, cloud/GPU work or new model inference is part of this run.

## Controls and limitations set before reporting

- Data: original Qwen3/Wikipedia NPZ, separate fit/evaluation language count halves; fit-only co-selection pair counts also exist, but are not per-request traces and are not used by these marginal-demand checks.
- The literal request-label shuffle cannot be recreated from aggregate counts. Run the analytic expected randomized-homing control with preserved group masses, with fixed original plans and fit-expectation-refitted plans. Retain a separate weaker whole-language grouping permutation (100 seeded draws per mixture, two replica budgets). Label the expectation and packet shuffle distinctly; do not report either as an observed request-by-request shuffle or give a request-level permutation p-value.
- Main shuffle comparisons: four GPUs, R2/R4, Indic-only and equal full mix. English absent only in Indic-only. Fit and held-out aggregates are kept separate. Identical randomized profiles remove language/home association, not model language capabilities.
- B: greedy primary-load-constrained placement on all48layers,4/8GPUs,caps1.1/1.25/1.5/2/none,R0/2/4. Greedy is explicitly non-optimal. Same initial grouping and pre-replica cap semantics as B; report final held-out load and replicate earlier4GPU greedy results.
- Rebuild:32 equal-work blocks per synthetic trajectory: gradual, abrupt, alternating. Test periods1/2/4/8/16/32, R2/R4,4GPUs. Current-mixture oracle and previous-block observed-mixture controls distinguish lookahead. Bounded six-candidate grouping planner common to fresh/stale comparisons. Define “most” as≥90%of fresh oracle average remote-hit saving; also report gain over no rebuilding. No time-based frequency recommendation without actual temporal request arrivals and migration costs.
- Metrics: remote expert-hit fraction and per-layer expert-load imbalance only. No measured all-to-all bytes, latency, p99, communication call count, or model-quality result.
