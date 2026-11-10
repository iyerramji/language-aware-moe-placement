# C: fixed layer budget

Executed fixed-layout CPU simulation, 96 copies per GPU. Specificity measured on fit half; largest-deficit rounding preserves exact budgets. Greedy uses fit-half marginal local-hit gains, with no held-out tuning. Equal-sized experts are assumed.

- G=4: best held-out policy greedy_gain at 34.291% traffic cut, 1.244 points over uniform; imbalance 1.958. Winner is descriptive, not selected using evaluation data for deployment.

- G=8: best held-out policy greedy_gain at 33.004% traffic cut, 1.454 points over uniform; imbalance 3.015. Winner is descriptive, not selected using evaluation data for deployment.
