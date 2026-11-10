# B on eight GPUs, greedy implementation

All48layers, caps1.1/1.25/1.5/2/none, R0/2/4. Four-GPU same-method rerun provides a paired reference. Greedy is not a certified global optimum. Original B semantics retained: cap constrains FIT PRIMARY expert load before adding replicas. Held-out final expert load can exceed the cap. Every GPU has E/G primaries and exactly R extra experts. Groups fixed to original baseline. No GPU/cloud execution.

- G4,cap1.1,R2: reduction 31.061%, mean imbalance 1.129, worst layer 1.359; fit time 5.55s.

- G4,cap1.25,R2: reduction 32.631%, mean imbalance 1.237, worst layer 1.382; fit time 5.55s.

- G8,cap1.1,R2: reduction 26.107%, mean imbalance 1.269, worst layer 1.505; fit time 8.47s.

- G8,cap1.25,R2: reduction 26.795%, mean imbalance 1.355, worst layer 1.685; fit time 8.62s.
