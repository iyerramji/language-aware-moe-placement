# Rebuild frequency under synthetic mixture changes

Three PREDEFINED synthetic32-block trajectories: gradual transitions among equal/Hindi-heavy/English-heavy/Indic-only; abrupt8-block regimes; alternating equal/Hindi-heavy each block. These are not recorded production arrivals or observed temporal routing drift. Each block has equal total work. Within-language fit/evaluation profiles remain fixed.

The bounded planner searches only six pre-existing candidate home groupings from D, choosing on fit-half primary locality and then refitting layout/copies. All refreshed/frozen comparisons use the SAME candidate planner. It is not the original300-candidate grouping search.

Compare every1/2/4/8/16/32blocks, with either current-mixture oracle information or a one-block-lagged observed mixture. The latter avoids advance knowledge. Averages weight blocks equally. “Keep most” is defined BEFORE reading results as at least90%of fresh current-mixture oracle average remote-hit savings; also report fraction of adaptation gain recovered relative to never rebuilding.

These units are mixture blocks, not seconds. Actual request arrival timing, collection delay, rebuild/transfer disruption and bytes are absent. Do not prescribe a wall-clock rebuild interval from this test; profiling-only fit times do not include migration. Periodic phase can alias alternating trajectories, so apparent nonmonotonicity must not be read as a universal recommendation.

- gradual,R2,current_mix_oracle: tested intervals retaining≥90%fresh saving: [1, 2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.

- gradual,R2,previous_block_mix: tested intervals retaining≥90%fresh saving: [1, 2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.

- gradual,R4,current_mix_oracle: tested intervals retaining≥90%fresh saving: [1, 2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.

- gradual,R4,previous_block_mix: tested intervals retaining≥90%fresh saving: [1, 2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.

- abrupt,R2,current_mix_oracle: tested intervals retaining≥90%fresh saving: [1, 2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.

- abrupt,R2,previous_block_mix: tested intervals retaining≥90%fresh saving: [1, 2]; longest tested qualifying interval 2. Not a general time recommendation.

- abrupt,R4,current_mix_oracle: tested intervals retaining≥90%fresh saving: [1, 2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.

- abrupt,R4,previous_block_mix: tested intervals retaining≥90%fresh saving: [1, 2, 4]; longest tested qualifying interval 4. Not a general time recommendation.

- alternating,R2,current_mix_oracle: tested intervals retaining≥90%fresh saving: [1]; longest tested qualifying interval 1. Not a general time recommendation.

- alternating,R2,previous_block_mix: tested intervals retaining≥90%fresh saving: []; longest tested qualifying interval none. Not a general time recommendation.

- alternating,R4,current_mix_oracle: tested intervals retaining≥90%fresh saving: [1]; longest tested qualifying interval 1. Not a general time recommendation.

- alternating,R4,previous_block_mix: tested intervals retaining≥90%fresh saving: [2, 4, 8]; longest tested qualifying interval 8. Not a general time recommendation.
