# B: balance-aware placement

Executed four-GPU CPU simulations. Cap constrains fit-half PRIMARY expert load before replicas, not held-out final load. Each GPU holds exactly E/G primaries. Integer-program proof flags and timings are recorded per layer; greedy uses heavy-first placement followed by feasible pair swaps. No request latency measured.

- G=4, cap=1.1, R=2: IP traffic cut 32.014%, mean imbalance 1.261; optimal 48/48 layers; IP/greedy time 38.11/5.84s; greedy fit-local-share gap 0.800 percentage points.

- G=4, cap=1.25, R=2: IP traffic cut 32.875%, mean imbalance 1.363; optimal 48/48 layers; IP/greedy time 21.44/5.94s; greedy fit-local-share gap 0.575 percentage points.

- G=4, cap=1.5, R=2: IP traffic cut 33.932%, mean imbalance 1.552; optimal 48/48 layers; IP/greedy time 15.51/5.60s; greedy fit-local-share gap 0.364 percentage points.

- G=4, cap=2.0, R=2: IP traffic cut 34.243%, mean imbalance 1.815; optimal 48/48 layers; IP/greedy time 1.88/5.51s; greedy fit-local-share gap 0.025 percentage points.

- G=4, cap=None, R=2: IP traffic cut 33.046%, mean imbalance 1.987; optimal 48/48 layers; IP/greedy time 0.87/5.41s; greedy fit-local-share gap 0.003 percentage points.

For four GPUs and R2, cap1.1 is a useful conservative candidate:32.014% traffic cut at1.261 mean held-out imbalance, versus33.046% and1.987 without cap. This is an exploratory trade-off recommendation, not a predeclared winner or a latency guarantee. The greedy cap1.1 alternative gives31.061% at1.129 mean imbalance. More permissive caps are not guaranteed to improve held-out traffic: the fit objective optimizes primary locality before adding copies, and the data halves differ.
