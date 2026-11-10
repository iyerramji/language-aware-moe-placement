# E: language-family and English/Indic grouping

Executed on Qwen3/Wikipedia marginal routing counts only. Indo-Aryan:Hindi,Bengali,Marathi,Gujarati,Punjabi. Dravidian:Tamil,Telugu,Kannada,Malayalam. English held separate as an experimental control, not classified as Indo-Aryan. Family classification from Census of India, Statement9.

Four GPUs:1English home,2Indo-Aryan homes,1Dravidian home; family subgroup sizes1/3/2/4. Eight GPUs:1English,4Indo-Aryan,3Dravidian homes. English/Indic-only pooling keeps English on one GPU and fit-optimizes Indic partitions among the other GPUs. Every GPU retains E/G primary expert slots; same R copies per GPU/layer. All experts remain accessible; no router is changed. This is home-assignment grouping, not an architectural language-specific expert restriction.

The different grouping sizes and home workload can confound raw comparisons. The same-sizes control fixes English isolation and group sizes but removes family restrictions. Twenty seeded shuffled-family labelings preserve all family sizes and within-family optimization, excluding the true mapping. They are an exploratory control, not exhaustive statistical proof. Grouping uses fit counts; metrics use held-out counts. No measured bytes, latency or learned FFN result.

- G=4, R2 equal: family remote-hit reduction 39.776%, imbalance 2.539; advantage versus English-separated same-sizes control 1.582 percentage points. English/Indic pooling 36.456% reduction, imbalance 1.643. Family beats 19/20 shuffled controls; shuffle range 37.378–39.932%.

- G=8, R2 equal: family remote-hit reduction 31.489%, imbalance 3.345; advantage versus English-separated same-sizes control -0.012 percentage points. English/Indic pooling 31.107% reduction, imbalance 3.311. Family beats 9/20 shuffled controls; shuffle range 30.920–32.182%.

- Held-out Indic pairwise JS: within-family 0.032233 bits, between-family 0.039607 bits. Descriptive pair means; overlapping language pairs are not independent samples.
