Hi Raj,

I ran the setup and A–D locally using Codex and the Qwen3 routing data from your repo. The setup matched your numbers.

**What you asked me to run**

- [A — Replication](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/raj-requested/A/NOTES.md): most of the benefit came from placement and request assignment. Choosing copies by language added only about 0.32 percentage points with two copies per GPU on four GPUs.
- [B — Balance](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/raj-requested/B/NOTES.md): the fit-load cap kept nearly all the simulated saving, while mean held-out imbalance dropped from 1.99 to 1.26.
- [C — Layer budgets](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/raj-requested/C/NOTES.md): putting copies where they saved the most remote hits worked better than spreading them evenly or concentrating them in the most language-specific layers.
- [D — Traffic mixes](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/raj-requested/D/NOTES.md): Indic-only traffic still benefited, but a plan could lose substantial savings when the traffic mix changed.

**What I wanted to try separately**

- [Language-family grouping and English/Indic pooling](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/ramji-ideas/E/NOTES.md): I wanted to see whether keeping English separate and grouping Indo-Aryan and Dravidian languages helped. The family plan improved four-GPU locality but increased overload. On eight GPUs it was essentially tied with the English-separated, same-size control. So this did not give a consistent family-based advantage.
- [Where to put the copies](https://github.com/iyerramji/language-aware-moe-placement/blob/codex/ablation-results/results/organized-experiments/ramji-ideas/F/NOTES.md): with English separate and Indic languages pooled, I compared early, middle, late and boundary layers at the same total replica memory. Uniform copying across layers beat the concentrated variants we tested.

The tables and figures are linked in the reports. These are remote expert-hit estimates, not measured network bytes or latency. A and C include four and eight simulated GPUs; B completed four GPUs, and I stopped the optional eight-GPU solver pilot because it was slow. D uses four GPUs. My additional tests include four and eight GPUs.

The Indic-only result is the one I would like to discuss next—particularly how much comes from request homing rather than differences between the Indic languages.

Thanks,
Ramji
