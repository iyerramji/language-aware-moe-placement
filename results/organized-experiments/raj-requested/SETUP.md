# Setup reproduction — completed locally

1. A home GPU is the simulated origin assigned to a language’s requests.
2. Local share is the fraction of selected expert hits that can be served on that origin GPU.
3. Replicas are additional copies of existing logical experts on other GPUs, not new experts or changed router choices.
4. Imbalance is the busiest GPU’s expert-hit load divided by the average, reported here as a mean across layers and also a worst-layer value.
5. The first half of the supplied text fits the plan and the second half evaluates it; grouping, expert placement and copy selection must not use the evaluation half.

Reproduced four-GPU aware local share **0.3442** (target0.344) and home-conditioned baseline **0.2569** (target0.257), both within0.001.
