# D: concentrated and shifted traffic

Fit group search, layouts and copies exclusively on the fit half and fit mixture. Evaluate each frozen plan on the held-out half under every served mixture. Blind reference is re-fitted for each served mixture (a strong oracle-mixture comparator), using only fit-half counts. Groups have balanced LANGUAGE COUNTS as in analyze.py, not balanced request or expert load. Zero-weight English remains in group search but contributes no traffic.

- R=2, equal: aware traffic cut 33.046%, advantage over blind 9.975 points; mean imbalance 1.987.

- R=2, hindi_heavy: aware traffic cut 57.387%, advantage over blind 38.384 points; mean imbalance 2.943.

- R=2, english_heavy: aware traffic cut 45.859%, advantage over blind 31.552 points; mean imbalance 2.250.

- R=2, indic_only: aware traffic cut 36.207%, advantage over blind 11.103 points; mean imbalance 2.107.

- R=2: worst mismatch loss versus matched-fit plan 51.555 percentage points (random_1 → hindi_heavy). These are static mixture changes, not temporal adaptation or migration-cost measurements.

- R=4, equal: aware traffic cut 46.061%, advantage over blind 8.025 points; mean imbalance 1.754.

- R=4, hindi_heavy: aware traffic cut 62.410%, advantage over blind 30.868 points; mean imbalance 2.894.

- R=4, english_heavy: aware traffic cut 52.759%, advantage over blind 28.234 points; mean imbalance 2.257.

- R=4, indic_only: aware traffic cut 49.886%, advantage over blind 8.799 points; mean imbalance 1.882.

- R=4: worst mismatch loss versus matched-fit plan 46.426 percentage points (random_1 → hindi_heavy). These are static mixture changes, not temporal adaptation or migration-cost measurements.
