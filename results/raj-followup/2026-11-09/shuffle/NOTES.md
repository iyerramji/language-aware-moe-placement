# Language-information controls

The supplied counts are aggregated by language/half, with 48 sequences of512tokens per fit/evaluation half and language. Request-level expert counts are absent. Therefore the exact random-label realization and finite-request shuffle variance cannot be reconstructed.

The expectation control replaces every pseudo-language profile by the same weighted aggregate profile, separately on fit and evaluation halves, preserving total expert demand and each original home group’s mass. This is the expected demand under independently randomized request-to-group assignments with fixed group sizes for these equal-length sequences, not an observed shuffled dataset. Report both frozen-plan and expectation-refitted-plan variants. Max-load metrics are computed on expected loads, not expected finite-shuffle maxima.

The separate100whole-language-profile permutations keep language packets intact and refit placement. This is a weaker random-grouping control, NOT request-label randomization and NOT evidence that language information is unnecessary. No permutation p-value is claimed for the expectation control.

No group is selected on evaluation data. Full mix shuffles include English; Indic-only excludes English requests. Blind comparator spreads every language uniformly across GPUs, whereas the random-homing expectation preserves original unequal home-group masses. This distinction can itself provide locality benefits unrelated to language.

- indic_only,R2: original 36.207%, shuffled expectation/refit 33.989%, blind 25.103%; original-minus-null 2.218 percentage points. Do not translate directly to network bytes.

- indic_only,R4: original 49.886%, shuffled expectation/refit 48.231%, blind 41.087%; original-minus-null 1.656 percentage points. Do not translate directly to network bytes.

- equal,R2: original 33.046%, shuffled expectation/refit 27.996%, blind 23.071%; original-minus-null 5.050 percentage points. Do not translate directly to network bytes.

- equal,R4: original 46.061%, shuffled expectation/refit 41.757%, blind 38.036%; original-minus-null 4.304 percentage points. Do not translate directly to network bytes.
