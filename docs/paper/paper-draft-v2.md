# Fewer Bytes, Not Less Time

## Language-Aware Expert Placement for Multilingual MoE Serving

Ramji and the Vizuara Research Team · Vizuara AI Labs

26 November 2026

### Abstract

Multilingual mixture-of-experts models route tokens differently by language, suggesting that expert placement could reduce communication without changing the model. We study Qwen3-30B-A3B and Sarvam-30B with English and nine Indian languages. Parallel text retains 92% and 77% of the English–Indic routing divergence observed on Wikipedia. In simulation, language-aware placement improves traffic reduction by 9.67–11.18 percentage points over blind copies at equal replica memory. Four-H200 NVLink replay reduces payload by 25.7–32.2% in decode and 26.2–33.0% in prefill relative to the default layout, but the run-wide repeat-check rule prevents trusted latency verdicts. In a separate four-L40S full-model decode experiment, capped placement saves 2.02% of logical bytes per token on Indic-only traffic and 4.99% on the full mix relative to blind copies, without establishing a decode speedup. The cap improves planned peak load but barely changes live imbalance. These results establish communication savings while showing that lower aggregate payload alone is insufficient to establish faster decode. Realized rank balance, synchronization, and the communication contribution to the critical path remain central constraints.

### 1 Introduction

Expert parallelism distributes the experts of a mixture-of-experts model across GPUs. A token assigned to a remote expert must send its activation to that GPU and receive the expert output. Placement methods exploit expert co-selection or workload structure to reduce this traffic [1, 6, 9–12]. Studies of multilingual routing separately identify language-correlated expert use [2, 3, 14]. A request's language can therefore provide a signal for placing experts and assigning request homes, although it is not a complete description of the workload.

For a model serving English and nine Indian languages, the practical question is whether this locality signal improves execution time. The benefit depends on which GPU performs the work and when the communication completes. Concentrating requests on a language's home GPU may reduce aggregate transfers while increasing the work or waiting on the rank that finishes last.

We fit placements on one half of the recorded text and evaluate them on the other half. The study connects routing measurements, held-out placement simulations, H200 communication and layer replay, and L40S full-model decode. It separates default-layout comparisons from equal-memory blind-copy comparisons. Language awareness saves real bytes, but neither the H200 timing evidence under its decision rule nor the L40S measurements establishes faster decode. The observed balance and timing patterns motivate optimizing the slowest rank rather than treating total bytes as a sufficient performance objective.


<!-- PAGEBREAK -->

### 2 Setup

**Models and text.** Qwen3-30B-A3B has 48 MoE layers, 128 experts per layer, and selects eight experts per token. Sarvam-30B has 18 MoE layers, selects six routed experts, and includes a shared expert excluded from the routed-expert counts. Routing is recorded in BF16 as per-sequence expert counts. Wikipedia covers English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam and Punjabi, with 96 sequences of 512 tokens per language. FLORES supplies the same 500 sentences per language. Code-mixed evaluations use romanised Hinglish with its English side and Tanglish. The first half of each language's text fits the placements and the second half evaluates them. FLORES is processed one sentence at a time following the failed padding control.

**Routing and placement metrics.** We compare expert distributions using Jensen–Shannon divergence in bits, averaged over layers. The split-half floor compares two halves of the same language. Simulated communication is a remote expert-hit proxy. Imbalance is busiest-rank expert work divided by mean rank work. It excludes attention, KV-cache pressure, network topology, and other runtime costs.

**Placement arms.** DEF assigns experts in contiguous blocks and distributes sequences by token count. BLIND2 adds two copies per GPU per layer of commonly used remote experts without language conditioning. AWARE2 chooses the layout, request home GPU, and copies using language-conditioned demand. The H200 AWARE2 CAP variant uses a 1.25× load cap. The L40S follow-up separately constrains predicted executed work after replica selection on fit data: 1.25× on Indic-only and a 1.15× full-mix sensitivity. These policies share a two-copy budget within each comparison, but the experiments do not establish identical implementations of their caps.

**H200 replay.** The four-H200 NVLink experiment replays held-out routing for Qwen3/FLORES and Sarvam/Wikipedia and FLORES, each under Indic-only and mixed traffic. Each remote token-expert hit sends a BF16 vector to the expert and returns its output through NCCL all-to-all with exact split sizes. One pass measures communication; a second adds a FLOP-matched SwiGLU expert FFN to measure layer time. Prefill uses eight whole-sequence steps, up to 2,048 tokens per GPU. Decode uses 16 steps of 64 sequences per GPU, one token each. Step time is the maximum over GPUs from a common barrier. The analysis uses six repeats, paired ratios against DEF, and bootstrap intervals over steps. A second DEF arm is the A/A repeat control. This manuscript's combined table covers the four-GPU experiment.

**L40S decode.** Sarvam-30B runs in BF16 with PyTorch/NCCL on four L40S GPUs in one VM. The topology reports PCIe host-bridge paths; CUDA peer access is unavailable and the logged NCCL transport uses shared memory. All arms use the same fixed continuations, replica budget, and held-out FLORES inputs, with 64 decode positions per request, three repeat blocks, and rotated arm order. The run comprises 27 trials and 32,640 measured decode token positions. Logical payload counts sender-side dispatch and combine, including expert IDs; split-size bytes are tracked separately. A duplicate blind layout supplies the repeatability control. Separate 12-step diagnostic replays measure NCCL stream intervals; unprofiled runs supply the headline ITL values.


<!-- PAGEBREAK -->

### 3 Results

#### 3.1 Language structure in routing

English and the nine Indian languages used different expert distributions in both Qwen3-30B-A3B and Sarvam-30B. On Wikipedia, mean English–Indic Jensen–Shannon divergence was 0.318 bits for Qwen3 and 0.274 bits for Sarvam, against within-language split-half floors of 0.013 and 0.018 bits. Every MoE layer exceeded three times its model's floor. On parallel FLORES text, the divergences were 0.294 and 0.212 bits, retaining approximately 92% and 77% of the Wikipedia values. Thus the split persisted when the text content was matched across languages.

Differences among the Indian languages were smaller and more sensitive to the text. Their mean pairwise divergences fell from 0.035 to 0.017 bits on Qwen3 and from 0.071 to 0.038 bits on Sarvam between Wikipedia and FLORES. The routing signal is therefore primarily an English–Indic split. The code-mixed prediction was not supported: romanised Hinglish was closer to Hindi than English in early layers under the original reference, and its position changed with the reference text. We treat code-mixed traffic as a separate class rather than assigning it an English-like routing profile. The padding control also failed, so FLORES routing was collected one sentence at a time; a recorded route is not guaranteed to remain identical under a different batch shape.

#### 3.2 Simulated locality and balance

With four simulated GPUs and two extra expert copies per GPU per layer, fully language-aware placement improved the simulated traffic-reduction percentage by 9.67–11.18 percentage points over language-blind copies across the two models and the Wikipedia/FLORES datasets. Both reductions used the default layout as their reference. Nearly all of the additional benefit came from the layout and request home assignment: blind copies on the aware layout came within 0.2–0.8 percentage points of fully aware copies. The gain came with greater imbalance. On Wikipedia, busiest-rank expert work divided by mean rank work was 1.99 for aware versus 1.25 for blind placement on Qwen3, and 1.51 versus 1.23 on Sarvam.

The Qwen3/Wikipedia aggregate-count ablation showed that a balance-aware planner could retain most of the simulated locality benefit. At the same two-copy budget, the capped integer-program plan reduced remote expert hits by 32.01%, compared with 33.05% for unconstrained aware placement, while mean layer imbalance fell from 1.987 to 1.261. These percentages use a no-copy, uniform-origin reference. The fit constraint applies to primary load before replica dispatch and does not guarantee a bound on live executed work.

Exact recorded-request shuffles supported an additional benefit from the original language/home association. On four simulated GPUs with two copies and token-work weighting, original groups exceeded the mean of refitted shuffled groups by 6.80 percentage points of remote-hit reduction on Indic-only Sarvam/Wikipedia, 6.95 on Qwen3/FLORES, and 5.68 on Sarvam/FLORES. Equal-sequence weighting and sentence-matched FLORES shuffles retained positive differences. The control preserves group request counts and uses independently shuffled fit and evaluation records; it is conditional on the selected home-group capacities, not a pipeline-wide significance test. These simulations measure remote expert-hit proxies and load, not network traffic or latency.



<!-- PAGEBREAK -->

#### 3.3 H200 replay: fewer bytes, provisional timing directions

In the four-H200 NVLink replay experiments, uncapped language-aware placement reduced communication payload by 25.7–32.2% in decode and 26.2–33.0% in prefill relative to the default layout. The capped plan reduced prefill payload by 22.0–27.1%, and language-blind copies reduced it by 17.3–24.4%, also relative to the default. These baselines differ from the blind-copy baseline used for the L40S comparisons below.

The H200 repeat check failed in 3 of 12 cells, almost all in decode; under the run's decision rule, every latency verdict from that run is not trusted, including prefill. We therefore claim no change in decode and report the prefill timings as consistent observed directions, not established effects. Uncapped aware placement showed a slower layer in five of six prefill cells, by up to 49.7%; the remaining cell's layer-time interval included zero. The capped plan showed 8–15% lower layer time than the default, while blind copies showed 13–19% lower layer time and were faster than the capped plan in all six cells. Thus blind copies saved fewer bytes but had the lowest observed layer times, subject to the run-wide timing qualification.

#### 3.4 L40S fixed-continuation decode and cap audit

On four L40S GPUs, with fixed continuations and equal replica memory, uncapped and load-capped language-aware placement reduced logical all-to-all bytes per generated token by 5.09% and 4.99%, respectively, on the full traffic mix and by 2.06% and 2.02% on Indic-only traffic, relative to language-blind copies. The capped plan's paired median inter-token latency was 1.12% lower on the full mix and 0.37% higher on Indic-only. These measurements did not establish a decode speedup.

The static cap lowered planned peak load but barely changed live mean layer-step imbalance. On Indic-only, the 1.25× cap reduced the fit peak from 1.282 to 1.243, while live imbalance changed from 1.595 to 1.594. On the full mix, a 1.25× cap left the plan unchanged, so the measured 1.15× plan was a sensitivity test; it reduced the fit peak from 1.201 to 1.149, while live imbalance changed from 1.526 to 1.530. The cap constrained predicted executed expert work on fit data after replica selection, not the realized load of every decode step. Its failure to improve realized balance helps explain why it did not establish a latency benefit.

Instrumented NCCL intervals occupied about 22–29% of the L40S diagnostic decode step in the blind and capped comparisons, including waiting for the slowest GPU. Their share was noisy: it fell from 29.0% to 22.0% between blind and capped placement on the full mix, while identical blind layouts measured 27.1% and 22.6% on Indic-only. These intervals do not justify a proportional conversion from byte savings to step-time savings. They include synchronization and waiting and are not isolated physical link time. The remainder of the step includes model computation and host work, which were not timed separately. Primary latency results come from the separate unprofiled runs.



<!-- PAGEBREAK -->

#### 3.5 Combined hardware results

Changes are relative to the baseline named in each row. Negative time changes mean faster; positive changes mean slower. L40S inter-token latency (ITL) changes are medians of paired repeat changes, not ratios of arm medians.

| Hardware and measurement | Traffic | Placement | Baseline | Communication reduction | Time result |
|---|---|---|---|---|---|
| 4 × L40S, full-model decode | Full mix | Uncapped aware | Blind copies, equal replica memory | 5.09% logical bytes/token | Median ITL −1.56%; no established speedup |
| 4 × L40S, full-model decode | Full mix | Capped aware, 1.15× sensitivity | Blind copies, equal replica memory | 4.99% logical bytes/token | Median ITL −1.12%; no established speedup |
| 4 × L40S, full-model decode | Indic-only | Uncapped aware | Blind copies, equal replica memory | 2.06% logical bytes/token | Median ITL −0.28%; no established speedup |
| 4 × L40S, full-model decode | Indic-only | Capped aware, 1.25× | Blind copies, equal replica memory | 2.02% logical bytes/token | Median ITL +0.37%; no established speedup |
| 4 × H200, NVLink decode replay | Six model/text/traffic cells | Uncapped aware | Default layout | 25.7–32.2% payload | We claim no change in decode [^h200] |
| 4 × H200, NVLink prefill layer replay | Six model/text/traffic cells | Uncapped aware | Default layout | 26.2–33.0% payload | Observed slower in five of six cells, by up to 49.7% [^h200] |
| 4 × H200, NVLink prefill layer replay | Same six cells | Capped aware, 1.25× | Default layout | 22.0–27.1% payload | Observed layer time 8–15% lower; slower than blind copies in all six [^h200] |
| 4 × H200, NVLink prefill layer replay | Same six cells | Blind copies | Default layout | 17.3–24.4% payload | Observed layer time 13–19% lower; faster than capped aware in all six [^h200] |

[^h200]: The four-H200 repeat check failed in 3 of 12 cells, almost all in decode. Under the run's decision rule, **every latency verdict from that run is not trusted, prefill included**. The prefill timings show consistent observed directions, not established effects. The payload reductions remain reported separately from these timing verdicts.

The H200 cells cover Qwen3/FLORES, Sarvam/Wikipedia, and Sarvam/FLORES, each with Indic-only and mixed traffic. H200 layer timings include replayed communication and a FLOP-matched expert FFN; L40S timings measure Sarvam-30B full-model decode in the research runtime. These are complementary experiments, not a controlled hardware comparison. L40S byte totals describe logical payload; physical link traffic was not measured. Across these experiments, lower communication volume alone did not establish faster decode, and a cap that improved planned balance did not necessarily improve live balance.



<!-- PAGEBREAK -->

### 4 Discussion

For saved bytes to save time, the reduction must reach the work that determines when the step finishes. In an expert-parallel collective, a rank can wait even after its own transfers or expert computation are complete. Reducing aggregate payload while leaving the slowest rank's work unchanged need not shorten the step. The observed H200 prefill ordering and the L40S cap audit are consistent with this explanation, but they do not isolate its causal contribution. In particular, the H200 timing rule prevents treating the observed ordering as an established performance effect.

A useful planner would therefore need to control realized load after requests are assigned and replicas are selected, rather than only the fit-time peak. Possible changes include spreading a language's requests over multiple home GPUs, selecting replicas according to current demand, or adapting placement as the traffic mix shifts. Each change may increase remote transfers, scheduling work, or migration costs. The relevant question is whether it reduces the slowest rank's completion time at the same memory budget after those costs are included. These are candidate mechanisms, not tested improvements in this study.

The runtime would also need enough exposed communication on the critical path for a byte reduction to matter. The L40S NCCL intervals include synchronization and waiting, so their 22–29% share cannot be multiplied by a payload reduction to predict a speedup. A timeline that separates transfer, expert work, host work, and waiting would help identify whether a better dispatcher, different overlap, or a different transport can convert locality into shorter steps. The shared-memory transport observed on L40S and the NVLink replay on H200 are different execution conditions, not a controlled comparison of interconnects.

Any claim of improvement would require repeat checks that pass, matched inputs and replica memory, and paired latency measurements against blind copies. Until then, fewer bytes are a demonstrated benefit, while faster decode remains unestablished.

### 5 Related work

Related expert-placement approaches include ExFlow, Occult and MoETuner [6, 10, 12], semantic scheduling in Sem-MoE [9], and workload-conditioned approaches in TACG and Bambhaniya et al. [1, 11]. Load-based replication is represented by DeepSeek's expert-parallel load balancing [4, 5]. Lina and METRO address communication or load balance in MoE execution [8, 13]. These lines of work motivate evaluating locality together with rank load and runtime behavior rather than using aggregate communication as the sole objective.

Multilingual routing studies [2, 3, 14] and language-specific pruning for translation models [7] provide a separate motivation for using language as a placement signal. Here, the contribution is an evaluation of that signal from recorded routing through simulations and measured communication, with explicit limits on the latency conclusions. We do not infer a general serving speedup from the routing structure, or claim exhaustive novelty coverage from the references retained in this draft.


<!-- PAGEBREAK -->

### 6 Limitations

The study covers two models and a limited collection of text and serving shapes. FLORES is short parallel text, and Wikipedia is not a production request stream. Equal request or token weighting changes the simulated effect size. Language labels correlate with script, tokenizer behavior, domain, and training data; the shuffle controls do not identify which of these causes the locality benefit. Code-mixed reference sensitivity further limits a simple language-only interpretation.

Recorded marginal counts do not preserve full expert co-selection or exact temporal arrival patterns. Reconstructed H200 decode expert sets and frozen routing replay are not equivalent to a complete serving engine. The padding diagnostic shows that routing can change with batch shape. The cap on predicted work cannot guarantee balance on live steps, and replica migration and dynamic adaptation costs were not measured.

H200 replay measures communication plus a matched expert FFN without attention, KV-cache management, or full-model serving. The run-wide timing qualification in Results applies to its prefill as well as decode measurements. L40S adds full-model decode in a research runtime, but does not measure physical link traffic and does not establish an improvement in production tail latency or throughput. Its diagnostic NCCL intervals include waiting and synchronization; they are not isolated network-transfer or critical-path time. The comparison does not isolate hardware, transport, model, or workload effects across platforms.

**Source and verification scope.** Routing, simulation, and L40S results were checked against the project reports. H200 replay results follow the supplied paper draft and Raj's exp03 verification and corrections. The underlying exp03 raw records were not independently reanalysed for this revision. References from the original paper are retained as supplied; their bibliographic details have not been independently reverified for this revision. No new experiments were run to prepare this draft.

### 7 Conclusion

Language-aware placement exploits a reproducible routing signal and reduces measured communication payload. The current evidence does not establish faster decode on the tested systems. Blind replication and the cap audit show why locality must be assessed alongside realized rank balance and runtime waiting. A practical placement objective must account for the completion time of the slowest rank, with its benefit demonstrated under passing repeat checks rather than inferred from byte savings alone.


<!-- PAGEBREAK -->

### References

[1] Abhimanyu Bambhaniya et al. Scaling multi-node mixture-of-experts inference using expert activation patterns. arXiv:2604.23150, 2026.

[2] Lucas Bandarkar et al. Multilingual routing in mixture-of-experts. ICLR, 2026.

[3] Yuxin Chen et al. Understanding multilingualism in mixture-of-experts LLMs: Routing mechanism, expert specialization, and layerwise steering. arXiv:2601.14050, 2026.

[4] DeepSeek-AI. DeepSeek-V3 technical report. arXiv:2412.19437, 2024.

[5] DeepSeek-AI. EPLB: Expert parallelism load balancer. GitHub repository, 2025.

[6] Seokjin Go and Divya Mahajan. MoETuner: Optimized mixture-of-experts serving with balanced expert placement and token routing. arXiv:2502.06643, 2025.

[7] Yeskendir Koishekenov, Alexandre Bérard, and Vassilina Nikoulina. Memory-efficient NLLB-200: Language-specific expert pruning of a massively multilingual machine translation model. arXiv:2212.09811, 2022.

[8] Jiamin Li et al. Accelerating distributed MoE training and inference with Lina. arXiv:2210.17223, 2022.

[9] Yan Li et al. Semantic parallelism: Redefining efficient MoE inference via model-data co-scheduling. ICLR, 2026.

[10] Shuqing Luo et al. Occult: Optimizing collaborative communication across experts for accelerated MoE training and inference. ICML, 2025.

[11] Zhiyao Xu et al. Beyond task-agnostic: Task-aware grouping for communication-efficient multi-task MoE inference. arXiv:2606.01007, 2026.

[12] Jinghan Yao et al. Exploiting inter-layer expert affinity for accelerating mixture-of-experts model inference. arXiv:2401.08383, 2024.

[13] Yanpeng Yu et al. Efficient MoE serving in the memory-bound regime: Balance activated experts, not tokens. arXiv:2512.09277, 2025.

[14] Kening Zheng et al. Unveiling language routing isolation in multilingual MoE models for interpretable subnetwork adaptation. Findings of EMNLP, 2026.
