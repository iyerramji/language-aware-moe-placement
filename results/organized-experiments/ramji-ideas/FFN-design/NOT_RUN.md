# Shared FFN before routing — proposed architecture experiment, not executed

Ramji's idea is a shared FFN transformation of hidden states before expert routing. It is not text translation and is not equivalent to replicating existing experts.

The supplied NPZ contains aggregate expert counts. It contains neither hidden states nor the learned router/FFN weights, so it cannot establish the behavior of an inserted FFN. A hypothetical change to expert counts would be invented evidence. No such result is claimed.

A future trained experiment could compare insertion near the input, before an early MoE router, before a middle MoE router, before a late MoE router, and paired boundary adapters. Exact layer indices must follow the chosen checkpoint's architecture. Hold adapter parameter budget, training data, updates, seed sets, residual scaling and evaluation prompts fixed. Include identity and parameter-matched shared-adapter controls. Define whether the transformation changes only router inputs or both routing and expert inputs; these are different treatments. Hold router and original expert weights frozen in an initial adapter-only comparison, then label any joint-training variant separately.

Measure quality, per-token expert choices and co-selection, language-conditioned overlap, expert load and actual dispatch bytes. Account for added FFN compute and memory before any latency claim. More similar routing can create hotspots and need not lower communication. If changing home GPUs between layers, measure token handoff and attention/KV implications explicitly.

The F placement experiment varies WHERE EXISTING EXPERT COPIES are stored, at equal replica memory. It is related to layer budgeting but provides no evidence for or against the new FFN architecture. No weights were downloaded, no models trained, and no GPU/cloud resources launched.
