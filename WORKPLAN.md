# Mentor ablation workflow

Completed locally on 9 November 2026. Research automation remains paused. No cloud/GPU work, training or messages to the mentor were performed.

- [x] Clone official repository, retain source revision in run manifest.
- [x] Create Python 3.11 environment and install pinned requirements.
- [x] Reproduce setup targets and explain the simulation in five sentences.
- [x] A: all replica budgets and policies on 4/8 simulated GPUs.
- [x] B: all caps, replica budgets, exact IP and greedy comparison on 4 simulated GPUs.
- [x] C: all five layer-allocation variants on 4/8 simulated GPUs.
- [x] D: six fitted mixtures by six served mixtures, R2/R4, on 4 simulated GPUs.
- [x] Validate accounting and inspect all figures.
- [x] Prepare tables, figures, notes and shareable archive.

The optional eight-GPU B pilot was stopped when tight-cap solves reached per-layer time limits without completing the first sweep in the short pilot. It is not reported as an exact eight-GPU result. A and C do include eight-GPU results.

Read [the findings](results/organized-experiments/raj-requested/FINDINGS.md), then each study's NOTES.md. Review the caveats before sharing [the prepared archive](results/organized-experiments/raj-requested/submission.zip) with Raj. Sending it remains your decision.

To rerun locally, activate `.venv`, then run `python analyze.py`, `python ablations/a_replication.py`, `python ablations/b_balance.py`, `python ablations/c_layer_budget.py`, `python ablations/d_traffic.py`, `python ablations/validate.py`, in that order. The optional eight-GPU B command is `python ablations/b_balance.py --gpus 8`.

The simulator and `results/exp01` were preserved. Only the supplied Qwen3/Wikipedia counts are available here; this is not a reproduction of the newer two-model/FLORES email results.

For Ramji’s separate experiments, run E, then F, then validate_family from the repository-root ablations directory. Published reports are in results/organized-experiments; reruns generate results/ablations.
