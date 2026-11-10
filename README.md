# Language-aware expert placement for multilingual MoE serving

A research project in the Vizuara Inference Research Bootcamp (November 2026 to January 2027). The target is a
workshop paper; the first choice is EuroMLSys 2027.

## The question

A mixture-of-experts model sends each token to a few experts out of many. When the experts are spread over several
GPUs, every layer pays for an all-to-all exchange, and over PCIe, which most smaller clouds rent, that exchange often
dominates. Experts specialise, and multilingual traffic is the norm for models built for India. We ask whether placing
experts by language keeps more of each token's work on its own GPU, how much traffic and latency that saves, and where
it beats the simpler practice of placing and copying experts by load alone.

## What we know so far

Our first experiment (`results/exp01/`, notes in `docs/exp01-results.md`) recorded which experts Qwen3-30B-A3B chooses
for text in ten languages, layer by layer, and simulated expert parallelism on 4 and 8 GPUs.

- English routes very differently from the nine Indian languages at every layer, most of all in the last layers; the
  Indian languages route much like each other.
- A language-aware layout on its own keeps 34% of expert hits on the token's own GPU, against 26% for the default
  layout on 4 GPUs. Copying four extra experts onto each GPU raises that to 60%.
- A layout built only for locality overloads some GPUs; the busiest carries about 2.3 times the mean load.

The paper has to answer one question honestly: how much does language awareness add over a language-blind baseline
that copies the busiest experts onto every GPU? The four ablations below are built to answer exactly that.

## How we work: agentic-first

Coding is done by an agent, Claude Code or Codex. Your job is to understand each experiment, run it through the
agent, check what comes back and think about what it means.

1. Clone this repository and open a terminal in it.
2. Start the agent here: `claude` for Claude Code, or `codex` for Codex.
3. Paste `prompts/00-setup.md` first. It sets up the environment and checks that our numbers reproduce.
4. Then paste the ablation prompts one at a time, in order: A, B, C, D. Before you paste each one, read its
   explanation below and make sure you could explain it to a colleague.
5. After each one, read the agent's `NOTES.md` and table. If a number looks wrong, ask the agent to check it.
6. Send us the four tables, the figures and a few lines per ablation on what surprised you.

## The ablations, and why each matters for the paper

**A. Replication budget, against a language-blind baseline.** Copying a few experts onto more GPUs keeps more traffic
local but costs memory. At equal memory we compare three policies: copy the busiest experts everywhere (language-blind),
a language-aware layout with those same copies, and a fully language-aware plan. This becomes the paper's central
figure: it shows where language awareness adds to plain load-based replication, and where it does not.

**B. Balance-aware placement.** A layout built only for locality overloads some GPUs, and the slowest GPU sets the
latency. We add a load cap, solve the placement exactly as an integer program, and measure how much locality survives
as the cap tightens. This is the method contribution.

**C. Spending memory across layers.** Some layers are more language-specific than others. With a fixed memory budget,
is it better to spread copies evenly or to put them where they save the most traffic? This decides how the planner
allocates memory.

**D. Concentrated and shifting traffic.** Real traffic is rarely an equal mix. We test Hindi-heavy, English-heavy and
Indic-only mixes, and plans built for one mix but served on another. This shows when the method pays most and whether
it survives change.

After these four, we move to real GPUs: a placement plugin for vLLM and latency measurements on 4 and 8 L40S over PCIe.

## Repository

- `analyze.py`: the simulation core and the Exp01 analysis (runs on a laptop).
- `exp01_routing.py`: the GPU run that produced the routing counts (Modal, one H100); you do not need to run it.
- `results/routing_qwen3_30b_a3b.npz`: routing counts for ten languages, 48 layers and 128 experts.
- `results/exp01/`: our Exp01 figures and summary.
- `prompts/`: the prompts for the agent.
- `docs/`: notes, and the literature review once it is final.

## Data and licence

The routing counts come from Qwen3-30B-A3B (Apache 2.0) run on Wikipedia text (CC BY-SA 3.0 and GFDL). The code is released
under the MIT licence.

## Ramji’s local ablation results

Executed CPU simulations using the supplied Qwen3/Wikipedia counts, 9 November 2026. No measured GPU latency or network-byte reduction is claimed.

- [Raj-requested setup and A–D](results/organized-experiments/raj-requested/FINDINGS.md): tables, figures, checks and original submission bundle. A/C cover 4/8 simulated GPUs; B covers 4 with certified layer optima; D covers 4. The optional 8-GPU B pilot was stopped.
- [Ramji’s additional ideas](results/organized-experiments/ramji-ideas/USER_IDEAS.md): language-family controls, English/Indic pooling and replica-position sensitivity.

Runnable additions are in `ablations/`. Original `analyze.py` and `results/exp01/` are unchanged. The aborted joint planner is not included.

## Raj follow-up results

The follow-up is separate from the first A–F runs: [report and limitations](results/raj-followup/2026-11-09/REPORT.md), [tables and figures](results/raj-followup/2026-11-09/index.html), and [reply draft](docs/raj-followup-reply.md). It includes the eight-GPU greedy B sweep, expected randomized-homing controls and synthetic rebuild sensitivity. The exact request-level shuffle remains unrun because per-request routing counts are missing. Reproduction scripts are in `followup/`.

## Exact per-sequence shuffle follow-up

[Exact-shuffle report](results/raj-exact-followup/2026-11-09/REPORT.md) and [reply draft](docs/raj-exact-shuffle-reply.md), kept separately from earlier A–F and approximation runs. Actual recorded sequences cover Sarvam Wikipedia and both FLORES models. Includes length-weighting and parallel-content controls, locality–imbalance curves and a token-normalized synthetic rebuild analysis. Inputs in data/per_sequence are from upstream commit d2a1f96427d769e817921bdcb5aa7652647fb2c8, with provenance and licences documented there. No text or model weights are included.
