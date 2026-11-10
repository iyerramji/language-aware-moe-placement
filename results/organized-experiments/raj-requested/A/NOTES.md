# A: replication budget

Executed CPU simulation on supplied Qwen3/Wikipedia counts. Fit first half, evaluate second. Same replica count per GPU/layer. Traffic reduction is relative to uniform language-blind default (local share 1/G), not the original home-conditioned default. Expert-hit traffic proxy is not measured bytes or latency.

- G=4, R=0: aware vs blind traffic-cut advantage 12.565 percentage points; language-specific copies add 0.000 points. Aware mean imbalance 2.323.

- G=4, R=2: aware vs blind traffic-cut advantage 9.975 percentage points; language-specific copies add 0.317 points. Aware mean imbalance 1.987.

- G=4, R=4: aware vs blind traffic-cut advantage 8.025 percentage points; language-specific copies add 0.409 points. Aware mean imbalance 1.754.

- G=4, R=16: aware vs blind traffic-cut advantage 4.734 percentage points; language-specific copies add 0.467 points. Aware mean imbalance 1.333.

- G=8, R=0: aware vs blind traffic-cut advantage 11.494 percentage points; language-specific copies add 0.000 points. Aware mean imbalance 3.769.

- G=8, R=2: aware vs blind traffic-cut advantage 10.921 percentage points; language-specific copies add 0.801 points. Aware mean imbalance 3.147.

- G=8, R=4: aware vs blind traffic-cut advantage 9.475 percentage points; language-specific copies add 1.089 points. Aware mean imbalance 2.672.

- G=8, R=16: aware vs blind traffic-cut advantage 5.040 percentage points; language-specific copies add 1.018 points. Aware mean imbalance 1.753.
