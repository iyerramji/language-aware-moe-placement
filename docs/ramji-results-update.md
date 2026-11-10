Hi Raj,

Thanks for the update. I ran the setup and A–D locally using Codex and the Qwen3 routing data from your repo. The setup matched your numbers.

**What you asked me to run**

- **A — Replication:** Most of the benefit came from placement and assigning requests to home GPUs. Choosing the copies by language added very little—about 0.32 percentage points with two copies per GPU on four GPUs.
- **B — Balance:** We could reduce overload without losing much of the simulated saving. With two copies per GPU on four GPUs, the remote-hit reduction went from 33.05% to 32.01%, while mean held-out layer imbalance dropped from 1.99 to 1.26.
- **C — Layer budgets:** Putting copies where they saved the most remote hits worked better than spreading them evenly. Concentrating copies in the most language-specific layers did worse.
- **D — Traffic mixes:** Indic-only traffic still benefited, but a plan could lose substantial savings when the language mix changed.

[Results, tables and figures for A–D](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/raj-requested/FINDINGS.md)

**What surprised me**

The Indic-only result surprised me most. Since the Indian languages looked similar in the routing analysis, I thought removing English might leave much less benefit. But with two copies per GPU on four GPUs, the Indic-only plan still had an 11.10 percentage-point advantage over the language-blind comparator. I would like to understand how much comes from request homing and layout rather than differences between the languages.

I was also surprised that the balance constraint kept almost all the saving. I expected a bigger trade-off between keeping expert work local and spreading the load.

**What I wanted to try separately**

I was thinking of trying grouping by language origins—Indo-Aryan and Dravidian—to see whether that might help placement. I also tried keeping English separate and pooling the Indic languages, then varying which layers received the copies.

The family grouping improved four-GPU locality but increased overload. On eight GPUs it was essentially tied with the English-separated, same-size control, so it did not give a consistent advantage. For the English/Indic plan, spreading copies across layers beat concentrating them in early, middle, late or boundary layers at the same total replica memory.

[My additional experiments: results, tables and figures](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/ramji-ideas/USER_IDEAS.md)

These are count-based simulations, not measured network bytes or latency, and they do not reproduce the newer Sarvam/FLORES results. A and C include four and eight simulated GPUs. B completed all four-GPU cases; I stopped the optional eight-GPU solver pilot because it was slow. D uses four GPUs. My additional tests include four and eight GPUs.

Could we go through the Indic-only result and the balance trade-off before deciding what to take to real GPUs?

Thanks,
Ramji
