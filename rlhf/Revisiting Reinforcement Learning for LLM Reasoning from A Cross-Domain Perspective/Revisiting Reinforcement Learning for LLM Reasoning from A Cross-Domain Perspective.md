# Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective

Zhoujun Cheng<sup>1,\*</sup>, Shibo Hao<sup>1,\*</sup>, Tianyang Liu<sup>1,\*</sup>
Fan Zhou<sup>2</sup>, Yutao Xie<sup>1</sup>, Feng Yao<sup>1</sup>, Yuexin Bian<sup>1</sup>, Yonghao Zhuang<sup>3</sup>, Nilabjo Dey<sup>4</sup>
Yuheng Zha<sup>1</sup>, Yi Gu<sup>1</sup>, Kun Zhou<sup>1</sup>, Yuqi Wang<sup>2</sup>, Yuan Li<sup>3</sup>, Richard Fan<sup>2</sup>, Jianshu She<sup>2</sup>
Chengqian Gao<sup>2</sup>, Abulhair Saparov<sup>4</sup>, Haonan Li<sup>2</sup>, Taylor W. Killian<sup>2</sup>, Mikhail Yurochkin<sup>2</sup>
Zhengzhong Liu<sup>2</sup>, Eric P. Xing<sup>2,3</sup>, Zhiting Hu<sup>1</sup>

<sup>1</sup>UC San Diego, <sup>2</sup>MBZUAI, <sup>3</sup>Carnegie Mellon University, <sup>4</sup>Purdue University \*Equal Contribution

#### **Abstract**

Reinforcement learning (RL) has emerged as a promising approach to improve large language model (LLM) reasoning, yet most open efforts focus narrowly on math and code, limiting our understanding of its broader applicability to general reasoning. A key challenge lies in the lack of reliable, scalable RL reward signals across diverse reasoning domains. We introduce GURU, a curated RL reasoning corpus of 92K verifiable examples spanning six reasoning domains—Math, Code, Science, Logic, Simulation, and Tabular—each built through domain-specific reward design, deduplication, and filtering to ensure reliability and effectiveness for RL training. Based on GURU, we systematically revisit established findings in RL for LLM reasoning and observe significant variation across domains. For example, while prior work suggests that RL primarily elicits existing knowledge from pretrained models, our results reveal a more nuanced pattern: domains frequently seen during pretraining (Math, Code, Science) easily benefit from cross-domain RL training, while domains with limited pretraining exposure (Logic, Simulation, Tabular) require in-domain training to achieve meaningful performance gains, suggesting that RL is likely to facilitate genuine skill acquisition. Finally, we present GURU-7B/32B, two models that achieve state-of-the-art performance among open models RL-trained with publicly available data, outperforming best baselines by 7.9% and 6.7% on our 17-task evaluation suite across six reasoning domains. We also show that our models effectively improve the Pass@k performance of their base models, particularly on complex tasks less likely to appear in pretraining data. We release data, models, training and evaluation code to facilitate general-purpose reasoning research at https://github.com/LLM360/Reasoning360.

<span id="page-0-0"></span>![](_page_0_Figure_6.jpeg)

![](_page_0_Figure_7.jpeg)

(a): Cross-domain transfer performance.

(b): Model performance across reasoning domains.

**Figure 1: Left:** Absolute accuracy gain (%) over the base model when RL-trained on various domains. Pretrained-heavy domains (Math, Code, Science) benefit from cross-domain training, while others require in-domain data, indicating RL aids skill acquisition. **Right:** Our GURU-7B/32B models consistently outperform strong open baselines across 17 reasoning tasks when RL-trained with mixed-domain data.

## **1 Introduction**

Recent frontier reasoning models, such as OpenAI-O3 [\(OpenAI,](#page-14-0) [2025\)](#page-14-0) and DeepSeek-R1 [\(Guo et al.,](#page-13-0) [2025\)](#page-13-0), demonstrate impressive performance and generalizability across diverse domains. These advances heavily leverage reinforcement learning (RL) as a core post-training technique to enhance large language model (LLM) reasoning capabilities. Motivated by these results, the open-source community has developed competitive reasoning models using RL [\(Yu et al.,](#page-15-0) [2025;](#page-15-0) [Hu et al.,](#page-13-1) [2025a;](#page-13-1) [Wang et al.,](#page-15-1) [2025b\)](#page-15-1) and systematically studied the principles behind RL's effectiveness for reasoning [\(Zeng et al.,](#page-15-2) [2025;](#page-15-2) [Yue et al.,](#page-15-3) [2025;](#page-15-3) [Shao et al.,](#page-14-1) [2025a;](#page-14-1) [Agarwal et al.,](#page-12-0) [2025;](#page-12-0) [Wang et al.,](#page-15-4) [2025a\)](#page-15-4).

However, the vast majority of RL works for LLM reasoning train and evaluate RL models exclusively on math and code domains. The domain-specific concentration raises two critical limitations in understanding and improving general-purpose reasoning. First, the current understanding of RL's role in reasoning is predominantly derived from studies focused on math and coding [\(Shao et al.,](#page-14-2) [2025b;](#page-14-2) [Zeng et al.,](#page-15-2) [2025;](#page-15-2) [Yue](#page-15-3) [et al.,](#page-15-3) [2025;](#page-15-3) [Wang et al.,](#page-15-4) [2025a;](#page-15-4) [Zhao et al.,](#page-15-5) [2025\)](#page-15-5), potentially limiting generalizability. For instance, recent observations about RL's fundamental mechanisms—whether it teaches new reasoning skills or merely elicits existing knowledge—may not generalize across other domains. Second, this approach produces models that excel only within their specialized training domains [\(Hu et al.,](#page-13-1) [2025a;](#page-13-1) [Zeng et al.,](#page-15-2) [2025;](#page-15-2) [Yu et al.,](#page-15-0) [2025;](#page-15-0) [Luo](#page-14-3) [et al.,](#page-14-3) [2025b\)](#page-14-3), limiting both our understanding of how RL enables reasoning and practical performance gains across diverse reasoning tasks. Consequently, studying RL across multiple domains is essential for advancing general reasoning research and deepening our understanding of RL's contributions.

A key bottleneck in extending RL-based reasoning to diverse domains is the scarcity of reliable reward signals and curated training data. Math and code problems are more widely sourced and studied, and offer easily verifiable answers [\(Li et al.,](#page-13-2) [2024a;](#page-13-2) [Li,](#page-13-3) [2024\)](#page-13-3). However, reasoning domains beyond them usually require more careful and creative data sourcing and domain-specific reward design. Even for math and coding domains, where queries and reward signals are more readily available, publicly available datasets require extensive curation to address common issues: duplicates, noisy samples, and unbalanced difficulty distributions [\(Bercovich et al.,](#page-12-1) [2025;](#page-12-1) [Hu et al.,](#page-13-4) [2025b\)](#page-13-4).

We introduce GURU, a curated RL corpus spanning six diverse reasoning domains: Math, Code, Science, Logic, Simulation, and Tabular. Each domain within GURU is constructed via a meticulous data pipeline designed to ensure reliability and utility for RL. The pipeline involves: data sourcing and synthesis, deduplication, domain-specific reward function design (with rule, execution, or model-based verification) for precise feedback, and both heuristic and model-based filtering to remove noisy or trivial examples and to ensure an appropriate difficulty distribution. The final corpus offers 92k examples, each paired with a verifiable reward signal, establishing a foundational and openly available benchmark to systematically conduct research on general-purpose reasoning with RL.

By performing RL on six domains in GURU with Qwen2.5-7B and 32B base models, we revisit established insights in RL-based reasoning and observe that these conclusions are domain-dependent. Most critically, prior work suggests RL primarily elicits existing knowledge from pretrained models rather than learning new capabilities, even with indirect or minimal elicitation like wrong or random labels or even performing RL from a single example [\(Zhao et al.,](#page-15-5) [2025;](#page-15-5) [Yue et al.,](#page-15-3) [2025;](#page-15-3) [Shao et al.,](#page-14-2) [2025b;](#page-14-2) [Wang et al.,](#page-15-4) [2025a\)](#page-15-4). As shown in Figure [1\(](#page-0-0)Left), domains frequently seen during continual pretraining [\(Ke et al.,](#page-13-5) [2023;](#page-13-5) [Yang et al.,](#page-15-6) [2024\)](#page-15-6) (Math, Code, Science) readily show performance gains from cross-domain RL training (warm colors on non-diagonal cells), indicating reasoning capabilities are more easily elicited in these domains even with cross-domain data. In contrast, relatively less familiar domains (Logic, Simulation, Tabular) require in-domain RL training to achieve meaningful improvements (cold colors on all non-diagonal cells) suggesting RL is more likely to learn new skills in these cases [\(Section 3.1\)](#page-5-0). Additionally, we find that response length increases correlate with performance gains only in certain domains [\(Section 3.2\)](#page-6-0), and difficulty filtering improves within-domain performance while harming cross-domain transfer [\(Section 3.3\)](#page-7-0). These findings reveal that RL reasoning mechanisms are more domain-specific than previously recognized, emphasizing the need for multi-domain training and evaluation in RL for LLM reasoning research.

Finally, we deliver GURU-7B and GURU-32B, two general reasoning models trained with RL on the GURU

dataset. On our unified evaluation suite with 17 reasoning tasks across six domains, our models establish a new state-of-the-art among open models trained with publicly available data, outperforming the best baselines by 7.9% and 6.7% respectively. Furthermore, we observe that Pass@k patterns of RL-trained and base models, which is defined as the LLM's reasoning boundary in [Yue et al.](#page-15-3) [\(2025\)](#page-15-3), vary by tasks, model scales, and generation parameters (e.g., temperature and top-p) [\(Section 4.3\)](#page-9-0). For example, on a synthetic Zebra Puzzle task, a complex constraint satisfaction problem that is less likely to appear in pretraining data, both GURU-7B and GURU-32B effectively expand the reasoning boundary of their base models. Overall, these substantial performance gains and cross-domain findings demonstrate the importance of multi-domain RL research for advancing general reasoning capabilities. We release the GURU dataset, models, and code at <https://github.com/LLM360/Reasoning360> for the community's use and continued study.

## **2 Data Construction**

Datasets used for RL reasoning predominantly focus on narrow domains such as Math [\(Luo et al.,](#page-14-3) [2025b;](#page-14-3) [Hu et al.,](#page-13-1) [2025a;](#page-13-1) [He et al.,](#page-13-6) [2025;](#page-13-6) [Zeng et al.,](#page-15-2) [2025\)](#page-15-2) and Code [\(Liu and Zhang,](#page-14-4) [2025;](#page-14-4) [Luo et al.,](#page-14-5) [2025a\)](#page-14-5). However, applying RL on such single-domain datasets often leads to overfitting to domain-specific structures and heuristics. As illustrated in [Figure 1,](#page-0-0) models trained to excel on advanced math benchmarks suffer sharp performance drops when evaluated on other domains. Even datasets that claim to span general reasoning capabilities tend to remain confined within the STEM boundary and exhibit similar generalization failures. Furthermore, existing RL reasoning datasets frequently contain redundant samples, noisy queries, and poorly calibrated difficulty levels. These limitations highlight the need for principled data curation, including deduplication, domain-specific filtering, and difficulty-aware sampling. To address these challenges, we propose a comprehensive data curation pipeline for constructing GURU, a multi-domain RL dataset designed to promote both do-

<span id="page-2-0"></span>![](_page_2_Figure_3.jpeg)

**Figure 2:** Overview of the data curation pipeline of GURU dataset.

main diversity and reward verifiability. An overview of our pipeline is shown in [Figure 2,](#page-2-0) consisting of five stages detailed below.

## **2.1 Data Sourcing**

We curate data across six reasoning-intensive domains, ensuring both diversity and verifiability:

- **Math:** We adopt recent math reasoning collections including OR1 [\(He et al.,](#page-13-6) [2025\)](#page-13-6), DAPO [\(Yu et al.,](#page-15-0) [2025\)](#page-15-0), and DeepScaler [\(Luo et al.,](#page-14-3) [2025b\)](#page-14-3), which compile numerous math datasets, notably previous competition problems such as AIME and AMC.
- **Code:** We include programming problems sourced from online coding platforms, programming competitions, and synthetic code generation tasks. Our collection features real-world problems from LeetCode [\(Xia](#page-15-7) [et al.,](#page-15-7) [2025\)](#page-15-7), curated and verified problems from TACO-Verified [\(Li,](#page-13-3) [2024\)](#page-13-3), synthetic tasks from PrimeIntellect [\(Mattern et al.,](#page-14-6) [2025\)](#page-14-6), and historical problems from LiveCodeBench [\(Jain et al.,](#page-13-7) [2024\)](#page-13-7). Notably, we reuse filtered subsets of PrimeIntellect and LiveCodeBench as processed by DeepCoder [\(Luo et al.,](#page-14-5) [2025a\)](#page-14-5).
- **Science:** We adopt WebInstruct-Verified [\(Ma et al.,](#page-14-7) [2025\)](#page-14-7), a dataset sourced from web content and refined using LLMs, as the primary source for science questions.
- **Logic:** We include symbolic reasoning tasks sourced from existing datasets and newly synthesized tasks. From existing datasets, we include ARC-AGI (1 and 2) [\(Chollet et al.,](#page-12-2) [2024\)](#page-12-2), which focus on abstract rule induction over symbolic grids, and BARC [\(Li et al.,](#page-13-8) [2024b\)](#page-13-8), which extends these tasks for inductive and transductive generalization. In addition, we synthesize three tasks for RL training: Zebra Puzzle [\(Lin](#page-13-9) [et al.,](#page-13-9) [2025\)](#page-13-9), a classic logic grid puzzle requiring models to chain positional and equality constraints; Ordering Puzzles, which require recovering a unique linear order of up to 50 objects from diverse relational

<span id="page-3-0"></span>Table 1: GURU dataset statistics across all 6 domains, including raw sizes and filtered subsets.

| Domain     | Dataset                                     | Raw Size | w/ Dedup.<br>& Domain Filt.  | w/ Difficulty<br>Filt. |
|------------|---------------------------------------------|----------|------------------------------|------------------------|
|            | OR1 (He et al., 2025)                       | 105k     | 118.2k                       | 54.4k                  |
| Math       | DAPO (Yu et al., 2025)                      | 17k      | 110,11                       | 0                      |
|            | DeepScaler (Luo et al., 2025b)              | 40.3k    | (-27.2%)                     | (-54.0%)               |
|            | LeetCode (Xia et al., 2025)                 | 2k       |                              |                        |
| C. J.      | TACO-Verified (Li, 2024)                    | 12.9k    | 23.7k                        | 18.1k                  |
| Code       | PrimeIntellect (Mattern et al., 2025)       | 16.3k    | (-26.2%)                     | (-23.6%)               |
|            | LiveCodeBench (history) (Jain et al., 2024) | 0.9k     |                              |                        |
| Calamas    | WebInstruct-Verifed (Ma et al., 2025)       | 0.001-   | 31.7k                        | 3.6k                   |
| Science    |                                             | 232k     | 31.7k 3.6k (.86.3%) (.88.6%) | (-88.6%)               |
|            | Zebra Puzzle                                | 5.7k     |                              |                        |
|            | Ordering Puzzle                             | 2.9k     |                              |                        |
|            | Graph Puzzle                                | 2.8k     | 13.6k                        | 6.3k                   |
| Logic      | ARC-AGI (Chollet et al., 2024)              | 0.4k     | (-12.8%)                     | (-53.7%)               |
|            | ARC-AGI-2 (Chollet et al., 2025)            | 1k       |                              |                        |
|            | BARC (Li et al., 2024b)                     | 3.4k     |                              |                        |
| 6' · 1.1'  | Code I/O (PyEdu) (Li et al., 2025)          | 0071     | 12.1k                        | 3.7k                   |
| Simulation |                                             | 227k     | (-94.7%)                     | (-69.4%)               |
| Tabadaa    | HiTab (Cheng et al., 2021)                  | 7.5k     | 10.3k                        | 6.1k                   |
| Tabular    | MultiHierTT (Zhao et al., 2022)             | 7.8k     | (-36.7%)                     | (-32.7%)               |
| Total      | -                                           | 684.9k   | 209.6k                       | 91.9k                  |

constraints; and *Graph Search*, where models must construct a predicate graph from natural language facts and reason over it to identify entity paths.

- **Simulation:** We include simulation-style reasoning tasks from *Code I/O (PyEdu)* (Li et al., 2025), where models must predict outputs or infer inputs based on code snippets without execution.
- **Tabular:** We repurpose existing table-based QA datasets involving both single- and multi-table reasoning. Specifically, we adopt *HiTab* (Cheng et al., 2021) and *MultiHierTT* (Zhao et al., 2022).

Table 1 summarizes the concrete data statistics each domain. Further details, including specific data links, collection or synthesis methods, and curation procedures for all datasets, are provided in Appendix B.

#### 2.2 Data Deduplication

Significant content overlap exists within the Math and Code domains, as many datasets originate from shared upstream sources. For example, numerous Math datasets reuse problems from past AIME exams or curated collections such as *NuminaMath* (Li et al., 2024a). Likewise, Code datasets are often derived from online coding platforms (e.g., *LeetCode*) and curated benchmarks (e.g., *APPS*, *CodeContests*), leading to repeated or near-duplicate problems across datasets. In our preliminary analysis, we observe that similarity-based metrics—such as embedding distance, Jaccard similarity, and n-gram overlap—are prone to false positives. For instance, n-gram matching may erroneously flag distinct samples that merely share a common prefix or structural template. To mitigate this, we apply a conservative deduplication strategy: if one question is a strict substring of another, the shorter sample is removed. This approach eliminates 27.2% of Math samples and 7.5% of Code samples. To further support transparency and data quality assessment, we construct metadata-level mappings of dataset provenance and cross-source dependencies. Visualizations of these relationships, including Sankey diagrams for Math and Code datasets, are provided in Appendix B.

#### 2.3 Reward Design

A key challenge in RL for reasoning tasks is designing reward functions that are fully automatic, low-noise, and domain-appropriate. Across all domains in our dataset, we adopt binary rewards: a sample receives a reward of 1 if the output is judged as correct under domain-specific verification rules, and 0 otherwise. We

categorize reward design into three types, described below. Additional task-specific details about reward design are further provided in [Appendix B.](#page-16-0)

- 1. **Rule-Based Matching:** This is the most common strategy, employed in domains such as Math, Logic, Simulation, and Tabular reasoning. Despite surface-level differences, these tasks share a common structure: the model is prompted to produce its final answer in a designated format—typically enclosed in \boxed{}, wrapped in a special tag (e.g., <answer>), or contained within a JSON block. The verifier extracts this region, normalizes the output, and applies strict matching. For Math tasks, we integrate a symbolic program [\(Cui et al.,](#page-13-12) [2025\)](#page-13-12) to handle algebraic and notational variations.
- 2. **Execution-Based Verification:** This approach is used in Code domains, where correctness is determined by program behavior. The model generates a function or script, which is executed against a suite of test cases. A reward of 1 is given only if all test cases pass. For tasks using stdin/stdout formats, we incorporate fuzzy comparison routines to handle formatting inconsistencies and numerical tolerances.
- 3. **Model-Based Verification:** This method is used for Science tasks, where answers are often open-ended and difficult to evaluate using deterministic rules. We employ a 1.5B verifier model [\(Ma et al.,](#page-14-7) [2025\)](#page-14-7) to assess whether the model's response semantically entails the reference answer, enabling reward computation for tasks that require greater flexibility in interpretation.

## **2.4 Heuristic Filtering**

Reliable reinforcement learning requires accurate and stable reward signals, which can be undermined by noisy, ambiguous, or poorly constructed samples. To address this, we implement a heuristic filtering stage across all domains, aimed at improving reward verifiability, stability, and coverage diversity. Concretely, we remove samples with overly long prompts that risk truncation, invalid reference outputs (e.g., code that fails its own test cases), or excessively large inputs that may hinder stable reward computation in multi-process environments. For structurally complex tasks (e.g., graph reasoning), we control instance difficulty to retain only those with sufficient symbolic or compositional depth. Additional domain-specific filtering heuristics, including thresholds for size, precision, and redundancy, are detailed in [Appendix B.](#page-16-0) After heuristic filtering, some categories may still exceed 10k examples. In such cases, we apply uniform random sampling to cap each subset at approximately 10k examples, ensuring a manageable dataset size and balanced task coverage.

## **2.5 Difficulty Filtering**

To further enhance RL data quality and ensure that RL focuses on challenging and learnable samples, we introduce a difficulty-aware filtering stage. This stage is designed to (i) prioritize examples that present sufficient reasoning difficulty, and (ii) aggressively remove samples that possibly exhibit annotation noise or unstable reward signals.

We implement this by measuring the empirical pass rates of a weak model (Mweak, Qwen2.5-7B-Instruct) and a strong model (Mstrong, Qwen3-30B-A8B) over N = 16 runs per sample, denoted as Pweak and Pstrong, respectively. Based on the statistics, we discard samples that satisfy any of the following conditions:

- 1. **Overly easy samples**: Pweak ≥ 15 <sup>16</sup> . Such examples are consistently solved by the weak model and thus provide limited signal for further improvement through reinforcement learning.
- 2. **Potentially noisy samples**: Pstrong = 0. These samples are consistently failed by the strong model, which often indicates that the task is either ambiguous, incorrectly labeled, or inherently unsuitable for RL training (e.g., questions requiring extreme numeric precision or external tools to solve accurately).
- 3. **Anomalous samples**: Pweak > Pstrong. These samples typically occur when the weak model consistently produces an answer that matches the provided label. Our empirical analysis suggests that such behavior often stems from incorrect ground-truth labels in the dataset. We suspect this arises because the weak model has previously encountered these flawed question–answer pairs during pretraining or supervised finetuning, leading it to memorize the incorrect response. These samples can mislead the RL process by reinforcing erroneous behaviors.

To further refine the dataset in specific domains, we analyze the difficulty gap, defined as Pstrong − Pweak. This metric serves as a proxy for the learnability of a sample, where higher values indicate greater potential

for RL to improve model reasoning capabilities. In the Math domain, given its large scale and generally high baseline performance, we adopt a more aggressive filtering strategy. We remove samples where the difficulty gap is marginal ( $P_{\rm strong} - P_{\rm weak} \leq \frac{6}{16}$ ) and the strong model already demonstrates high competence ( $P_{\rm strong} \geq 0.75$ ). This ensures that RL focuses on sufficiently challenging math problems where meaningful improvements remain achievable. In the Science domain, where data quality and model-based verifier are inherently noisier, we apply a stricter criterion by discarding samples with  $P_{\rm strong} - P_{\rm weak} < 0.5$ , prioritizing samples with clearer reasoning gaps and minimizing the inclusion of ambiguous or low-signal examples.

## <span id="page-5-1"></span>3 Analysis of Cross-Domain RL for Reasoning

To gain a deeper understanding of how reasoning capabilities generalize across different domains with RL, we conducted a controlled experimental analysis with our GURU dataset. Specifically, we investigated the impact of RL on single reasoning domains versus a mixed-domain corpus on their subsequent performance across a suite of domain-specific benchmarks. These experiments aim to inform the design of additional large-scale RL reasoning experiments while also providing a broadened perspective on the utility of data diversity when developing reasoning models.

To this end, we constructed an experimental dataset by randomly sampling a subset of 3K training samples from each of the six domains covered by our GURU dataset. These subsets form a mixed training dataset totaling 18K samples called Guru-18K. We train reasoning models directly from the Qwen2.5-7B-Base and Qwen2.5-32B-Base models using RL on each of these single-domain subsets and the combined mixed dataset. The detailed RL training setting is described in Section 4.1. Figure 3 illustrates the cross-domain generalization performance trained on every single domain and mixed (rows) and evaluated on various target domain tasks (columns). For each training, we perform online test on 13 tasks for every 10 steps and select the checkpoint with the best average performance. Qwen2.5-32B exhibits heatmap patterns highly similar to those of Qwen2.5-7B; thus, its results are presented separately in the appendix Figure 8 for reference. To mitigate potential confounding effects from the instruction-following gap (Zeng et al., 2025), we confirm the following conditions: (1) each training run continues until validation accuracy plateaus, ensuring the initial instruction alignment phase is completed; (2) experiments are conducted on both 7B and 32B models, where the 32B model has stronger zero-shot instruction-following capabilities, and consistent results are observed across both; and (3) all runs use the same system prompt, with reward computation explicitly designed to be independent of special tokens such as <think> included in the prompt. Our analysis yields several key findings below.

#### <span id="page-5-0"></span>3.1 Differential Transferability

**Domain identity matters.** Math, Code, and Science consistently achieve stronger performance gains than other domains from cross-domain RL training (Figure 3). We hypothesize that this asymmetry arises from the base model's extensive exposure to these domains during (continual-)pretraining (Liu et al., 2024; Yang et al., 2024). The model thus contains deep knowledge in these domains which is effectively elicited and refined through RL, even when the training data originates from other domains. Our observation is consistent with the recent finding from Shao et al. (2025a) that shows math reasoning can be easily triggered even with indirect signals like spurious rewards. In contrast, domains less represented in pretraining, such as Logic, Simulation, and Tabular, show limited improvement under cross-domain training. These results underscore the need to include underrepresented domains when curating RL training data, in order to build models with robust, general reasoning capabilities.

**Task difficulties matter.** As shown in Figure 3, easier tasks within Math (e.g., MATH500 (Hendrycks et al., 2021), AMC) and Code (e.g., HumanEval (Chen et al., 2021a), MBPP (Austin et al., 2021)) readily exhibit positive transfer from other domains. In contrast, performance on more challenging benchmarks in those same domains, namely AIME24 and LiveCodeBench (Jain et al., 2024), show considerably less improvement from cross-domain training. For the most challenging tasks, those with the lowest baseline absolute scores (e.g., ARC-AGI (Chollet et al., 2024), CodeI/O (Li et al., 2025)), we observe marginal to negligible gains from training on non-native domain data. This variance in reasoning transferability suggests that cross-domain RL alone is not sufficient to effectively develop models with general, complex reasoning capability. Thus, advanced competence in multiple domains is dependent on training with domain-specific examples or with

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 3: Cross-Domain RL Transfer Performance per Task. The heatmap shows the performance gains (accuracy) from RL training on different domains (rows) when evaluated on the test sets on different domains (columns). Warmer colors indicate higher performance gains, computed by applying min-max normalization to the validation accuracies within each column. Accuracy is reported using the checkpoint with the highest average score across tasks. Khaki-colored rectangles mark in-domain evaluations (diagonal); others reflect cross-domain generalization. This highlights differential transferability: Math, Code, and Science benefit significantly from cross-domain transfer, while Logic, Simulation, and Tabular tasks see limited gains, with improvements primarily driven by within-domain training. Notably, naive mixed-domain by combining data from all domains performs on par or better than single-domain RL.

substantially more diverse data, perhaps drawing from multiple difficult domains.

Mixed-domain training matches or exceeds single-domain performance. RL training on a uniformly mixed dataset of reasoning tasks proves remarkably effective as shown in the last row of Figure 3. Across individual downstream benchmarks, models trained on this combined data achieve performance levels consistently comparable to, and sometimes surpassing, those attained by models trained exclusively on data from the target domain. This demonstrates that even a simple uniformly mixed dataset across multiple domains can significantly enhance general reasoning capabilities with minimal apparent interference between the six domains involved. However, future research should investigate whether interference remains negligible as the number and diversity of included domains is increased. Scaling in this manner might necessitate more refined domain balancing strategies.

#### <span id="page-6-0"></span>3.2 Reward and Response-Length Dynamics

Figure 4 shows the reward and average response length across six reasoning domains during RL fine-tuning. The upper row indicates RL training with 3k single-domain examples, while the lower row indicates joint training with 18k examples. The x-axes are aligned so that each horizontal position represents an equal number of target domain samples.

In single-domain training, contrary to the common belief that RL drives models to produce longer responses, we observe strong domain effects: on Code, Logic, and Tabular tasks, the policy actually contracts its outputs, while Science and Math become more verbose, and Simulation remains largely unchanged. When we switch from single-domain training to joint training on the full Guru -18k mixture, rewards climb steeply within the first few hundred gradient steps across all domains. This demonstrates positive cross-domain transfer. Joint training may also reshape length dynamics: for Logic, the single-domain run monotonically shortens answers, but the multi-domain run first lengthens and then shortens them, suggesting that shared representations learned from other domains modulate brevity preferences. In contrast, domains such as Tabular preserve their "short-answer" tendency even under joint training, indicating that some length priors

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

**Figure 4:** The reward and response length of each domain during RL training with: **(top row)** single domain data (3k examples each) from GURU-18k; and **(bottom row)**: using the full GURU-18k mixture dataset. The x-axis is the number of gradient update steps.

<span id="page-7-2"></span>**Table 2:** Performance (best validation accuracy within 200 RL steps) comparison trained on difficulty-filtered (harder) v.s. unfiltered (easier) math data. Training on harder data improves in-domain performance but leads to degradation on easier cross-domain tasks (e.g., HumanEval and HiTab).

|                          | Math (in-domain) |      |        | Co        | Code & Tabular (cross-domain) |       |             |  |
|--------------------------|------------------|------|--------|-----------|-------------------------------|-------|-------------|--|
|                          | MATH500          | AMC  | AIME24 | HumanEval | LiveCodeBench                 | HiTab | Multihiertt |  |
| Unfiltered math          | 75.8             | 52.1 | 15.8   | 82.3      | 11.1                          | 56.5  | 32.0        |  |
| Difficulty-filtered math | 78.6             | 58.4 | 21.7   | 73.1      | 10.7                          | 53.5  | 35.5        |  |
| $\Delta$ (+/-)           | +2.8             | +6.3 | +5.9   | -9.2      | -0.4                          | -3.0  | +3.5        |  |

<span id="page-7-0"></span>remain robust to cross-domain influence.

## 3.3 Effects of Training Data Difficulty

We further conducted an ablation to investigate the impact of training data difficulty on both in-domain performance and cross-domain transferability. Specifically, we trained the base Qwen2.5-7B-Base model using RL for 200 steps on either the complete, unfiltered Math domain data (representing a mix of difficulties) or a difficulty-filtered subset containing primarily harder Math problems from GURU. Other training configurations followed Section 3.1. We then evaluated performance on the Math tasks (in-domain) and selected Code and Tabular analysis tasks (cross-domain).

As shown in Table 2, training on difficulty-filtered math data consistently improves in-domain performance compared to unfiltered data, particularly on harder Math tasks like AMC (+6.3) and AIME24 (+5.9). However, the effects on cross-domain tasks are difficulty-dependent. Easier tasks such as HumanEval (-9.2) and HiTab (-3.0) suffer notable degradation. This decline was corroborated by observed accuracy collapse (large drops witnessed after approximately 100 steps and 150 steps respectively) during training on the filtered data. Conversely, performance on harder cross-domain tasks (LiveCodeBench, Multihiertt) shows minimal negative impact or positive change.

In summary, increasing training data difficulty within a domain can consistently enhance in-domain performance when using Qwen2.5 as the base model. It is reasonable to perform aggressive difficulty filtering when the sole objective is maximizing performance within that specific domain on challenging tasks, e.g., AIME and LiveCodeBench. While for cross-domain transfer, it introduces a risk of negative transfer to easier tasks in other domains. These results suggest that for beneficial cross-domain transfer, a more balanced distribution of training data difficulties, or the explicit inclusion of cross-domain data, may be more effective than solely increasing the difficulty of the source domain data.

<span id="page-8-1"></span>**Table 3:** Full 17 benchmark performance on 7B and 32B models. All baselines and our models are trained from Qwen2.5- Base-7B/32B. GURU shows significant improvements over baselines on general reasoning skills under two model sizes. ⋄ : ORZ represents Open-Reasoner-Zero [\(Hu et al.,](#page-13-1) [2025a\)](#page-13-1). ♡: Zebra Puzzle and LiveBench employ non-binary scoring that awards scores proportional to output correctness.

| Models<br>→<br>Benchmarks<br>↓ |                      | GURU<br>7B | General<br>Reasoner<br>7B | ORZ<br>7B⋄ | SimpleRL<br>7B | GURU<br>32B | ORZ<br>32B⋄ | SimpleRL<br>32B |
|--------------------------------|----------------------|------------|---------------------------|------------|----------------|-------------|-------------|-----------------|
| Math                           | AIME24(avg@32)       | 17.50      | 17.08                     | 16.25      | 15.60          | 34.89       | 47.50       | 27.20           |
|                                | MATH500              | 77.25      | 70.40                     | 80.80      | 87.00          | 86.00       | 89.80       | 89.60           |
| Code                           | LiveCodeBench(avg@4) | 16.49      | 8.51                      | 5.47       | 6.72           | 29.30       | 22.04       | 19.80           |
|                                | HumanEval(avg@4)     | 82.62      | 61.12                     | 67.38      | 58.08          | 90.85       | 84.30       | 81.25           |
|                                | MBPP                 | 70.00      | 39.80                     | 48.40      | 49.60          | 78.80       | 74.20       | 76.75           |
| Science                        | GPQA-diamond(avg@4)  | 40.78      | 38.64                     | 37.63      | 35.98          | 50.63       | 55.67       | 46.46           |
|                                | SuperGPQA            | 31.80      | 30.64                     | 29.75      | 27.29          | 43.60       | 46.05       | 37.73           |
| Logic                          | ARC-AGI(avg@4)       | 3.31       | 0.75                      | 0.00       | 0.50           | 7.63        | 2.31        | 5.25            |
|                                | Zebra Puzzle♡(avg@4) | 39.40      | 0.07                      | 1.00       | 0.62           | 45.21       | 0.54        | 1.16            |
| Simulation                     | CodeI/O(avg@4)       | 15.63      | 7.13                      | 5.13       | 6.63           | 12.63       | 3.75        | 9.75            |
|                                | CruxEval-I           | 61.72      | 63.63                     | 69.38      | 56.25          | 80.63       | 71.13       | 72.63           |
|                                | CruxEval-O           | 71.28      | 56.50                     | 65.88      | 58.31          | 88.75       | 82.38       | 67.75           |
| Tabular                        | FinQA                | 34.70      | 34.33                     | 37.60      | 35.10          | 46.14       | 45.20       | 45.41           |
|                                | HiTab                | 74.20      | 54.40                     | 54.10      | 50.40          | 82.00       | 63.30       | 69.00           |
|                                | MultiHiertt(avg@4)   | 44.94      | 31.62                     | 38.10      | 37.57          | 55.28       | 52.83       | 52.83           |
| Others                         | IFEval               | 35.81      | 39.56                     | 32.72      | 36.69          | 55.45       | 38.26       | 55.27           |
|                                | LiveBench♡           | 18.57      | 19.76                     | 12.64      | 15.20          | 34.30       | 28.78       | 28.33           |
| Average Score                  |                      | 43.29      | 33.76                     | 35.42      | 33.97          | 54.24       | 47.53       | 46.25           |

# **4 Main Experiment**

Having established that RL behaviors vary significantly across domains [\(Section 3\)](#page-5-1), we now demonstrate the practical impact of multi-domain RL data through large-scale training experiments. We train 7B and 32B models on the complete GURU dataset to empirically validate that mixed-domain RL training enhances general reasoning capabilities at scale. In this paper, we prioritize showing the core effectiveness of mixeddomain training, and we further study advanced data mixing and RL training strategies for future work.

## <span id="page-8-0"></span>**4.1 Experimental Setup**

**Training Configurations** We use verl [\(Sheng et al.,](#page-15-9) [2024\)](#page-15-9) as the RL training framework and GRPO [\(Shao](#page-15-10) [et al.,](#page-15-10) [2024\)](#page-15-10) as the RL training algorithm. For general training hyper-parameters, we utilize the AdamW optimizer [\(Loshchilov and Hutter,](#page-14-9) [2017\)](#page-14-9) with learning rate 1 × 10<sup>−</sup><sup>6</sup> , incorporating a linear warm-up of 10 RL steps. The prompt batch size is 512 for one RL step, and we sample 16 responses for each prompt with a temperature of 1.0. The mini-batch size is set to 64, i.e., 8 gradient updates for each RL step. The maximum number of tokens is 4k for input prompt and 8k for generation. We do not apply any KL or entropy loss in the training. The clipping parameter ϵ for GRPO is set to 0.2. Experiments are conducted on 20 GPU nodes, each equipped with 8 Hopper GPUs, for both RL training and evaluation. The 7B model training (3 epochs) and 32B model training (2 epochs) each required approximately 3 days of wall-clock time.

**Baselines** For baseline comparisons, we compare the most performant RL-trained reasoning models with open data in general and math reasoning at 7B and 32B scale: (1) General Reasoner [\(Ma et al.,](#page-14-7) [2025\)](#page-14-7), (2) Open-Reasoner-Zero (ORZ) [\(Hu et al.,](#page-13-1) [2025a\)](#page-13-1), and (3) SimpleRL-Zoo [\(Zeng et al.,](#page-15-2) [2025\)](#page-15-2) To ensure a fair comparison, we exclude baselines that use SFT-trained models or RL training on top of SFT models. All baselines and our models are directly RL-trained from Qwen2.5 [\(Yang et al.,](#page-15-6) [2024\)](#page-15-6) series base models. It also

allows us to directly evaluate the effectiveness of our GURU dataset without confounding factors introduced in SFT data. All benchmark scores reported in the following sections are evaluated using identical evaluation code and generation parameters for both our models and baselines.

**Evaluation Suite** We build a comprehensive evaluation suite using both offline benchmarking for holistic assessment and online evaluation for monitoring training progress. The comprehensive evaluation dataset statistics are provided in [Appendix C.](#page-20-0)

Offline Evaluation. Our offline suite encompasses 17 benchmarks across six domains and one unseen generalization, designed to rigorously assess reasoning capability through diverse and challenging tasks:

- **Math**: AIME24 [\(MAA,](#page-14-10) [2024\)](#page-14-10) and MATH500 [\(Hendrycks et al.,](#page-13-13) [2021\)](#page-13-13) for mathematical reasoning problems
- **Code**: HumanEval [\(Chen et al.,](#page-12-4) [2021a\)](#page-12-4), MBPP [\(Austin et al.,](#page-12-5) [2021\)](#page-12-5), and LiveCodeBench [\(Jain et al.,](#page-13-7) [2024\)](#page-13-7) for programming capabilities spanning basic generation to competitive programming
- **Science**: GPQA [\(Rein et al.,](#page-14-11) [2023\)](#page-14-11) and SuperGPQA [\(Team et al.,](#page-15-11) [2025\)](#page-15-11) covering graduate-level questions
- **Logic**: ARC-AGI [\(Chollet et al.,](#page-12-2) [2024\)](#page-12-2) for abstract reasoning and Zebra Puzzle [\(Lin et al.,](#page-13-9) [2025\)](#page-13-9) for structured logical deduction
- **Simulation**: CodeI/O [\(Li et al.,](#page-13-11) [2025\)](#page-13-11) and CRUXEval [\(Gu et al.,](#page-13-14) [2024\)](#page-13-14) for execution tracing abilities
- **Tabular**: HiTab [\(Cheng et al.,](#page-12-3) [2021\)](#page-12-3), MultiHiertt [\(Zhao et al.,](#page-15-8) [2022\)](#page-15-8), and FinQA [\(Chen et al.,](#page-12-6) [2022a\)](#page-12-6) for heterogeneous and complex structured data reasoning
- **Others**: IFEval [\(Zhou et al.,](#page-15-12) [2023\)](#page-15-12) and LiveBench [\(White et al.,](#page-15-13) [2024\)](#page-15-13) for unseen domain adaptation

For offline evaluation inference, we use vLLM [\(Kwon et al.,](#page-13-15) [2023\)](#page-13-15) framework with temperature=1.0 and top-p=0.7 for all the tasks and models. To ensure statistical reliability, we generate 4 samples per question for datasets with less than 400 samples and report averaged results. For AIME24, we generate 32 samples per question. Large datasets (more than 1000 samples) are subsampled to 1000 examples for efficiency.

Online Evaluation. During RL training, we monitor progress using 13 signal tasks with up to 200 samples per task for computational efficiency. We select final checkpoints based on average performance across all online tasks. Online inference uses temperature=1.0 and top-p=1.0, with 8 samples per AIME24 problem and 1 sample for other tasks.

#### **4.2 Results**

As shown in [Table 3,](#page-8-1) GURU models demonstrate superior and well-balanced performance across all six evaluated domains. GURU-7B and GURU-32B achieve 43.29% and 54.24% respectively, outperforming the best baselines by 9.0% and 6.7%.

Domain-specific baselines exhibit significant weaknesses outside their specialization: math-focused models (Open-Reasoner-Zero, SimpleRL) and STEM-focused models (General-Reasoner) perform poorly on coding and logic tasks. Notably, Zebra Puzzle—a challenging synthetic constraint satisfaction task—shows substantial improvement from RL. GURU also maintains competitive results on unseen tasks (IFEval, LiveBench), demonstrating that multi-domain RL training effectively builds robust general reasoning capabilities.

## <span id="page-9-0"></span>**4.3 Analysis of Pass@k Curves**

Pass@k measures the chance that at least one of k independent samples from a model is correct. Unlike accuracy, Pass@k probes the reachable reasoning space of a model rather than its average-case behavior. Recent work [\(Yue et al.,](#page-15-3) [2025\)](#page-15-3) reports that RL fails to improve Pass@k at large k, concluding that the current RL recipe does not uncover fundamentally new reasoning capabilities beyond the base model. We revisit this claim using our Guru-7B and Guru-32B models on a diverse suite of tasks. The results reveal that the effect of RLVR on Pass@k depends on several factors, with three key observations below.

**Task-dependent pattern.** As shown in [Figure 5a,](#page-10-0) Pass@k curves exhibit different patterns across tasks rather than adhering to a uniform pattern. For example, on AIME, the curves of GURU-7B and the base model cross

<span id="page-9-1"></span><sup>1</sup>We generate a new test set for evaluation rather than using the original test set from [Lin et al.](#page-13-9) [\(2025\)](#page-13-9).

at k = 64, reproducing the pattern reported in [Yue et al.](#page-15-3) [\(2025\)](#page-15-3). Combined with our domain transfer study in [Section 3.1,](#page-5-0) where math reasoning gains can be elicited via out-of-domain RL training, these results suggest that the observed improvements in math largely exploit the base model rather than introduce fundamentally new capabilities. However, this conclusion does not hold for every task. For example, on the synthetic Zebra Puzzle task, which is likely less exposed in the pretraining stage compared to math and coding, GURU-32B and GURU-7B both effectively expand the reasoning boundary of the base model. We also report the Pass@k for checkpoints after different training steps in [subsection D.2.](#page-21-0)

**Effect of model scale.** Even within a single task, Pass@k behaviour changes with model scale. On AIME, the 32B curves never intersect, and RL consistently outperforms the base model. On the contrary, the 7B curves cross at k=64. On Zebra Puzzle, GURU-7B plateaus around k=32, while GURU-32B continues to rise across the entire sampled range. These patterns suggest that a stronger base model might more easily uncover new reasoning trajectories during RL finetuning, consistent with the DeepSeek-R1 study [\(Guo et al.,](#page-13-0) [2025\)](#page-13-0), which claims RLVR works better for larger models.

**Influence of sampling hyperparameters.** Our analysis of training logs reveals that the collapse in output entropy during RL training likely underlies the narrowing of Pass@k at larger k. We find that adjusting decoding parameters can counteract this narrowing to a certain degree. Specifically, we explored raising the sampling temperature and tuning the nucleus-sampling threshold (top-p). As shown in [Figure 5b,](#page-10-0) compared to the standard decoding hyperparameter we applied, these settings broaden the model's exploration, thereby improving Pass@k at larger k. Given its sensitivity to decoding hyperparameters, we suggest treating Pass@k as an informative but not exhaustive indicator of a model's reasoning boundary and interpreting it with caution.

<span id="page-10-0"></span>![](_page_10_Figure_3.jpeg)

**Figure 5:** Pass@k analysis of GURU. **(a)** Pass@k on AIME24 (Math) and Zebra Puzzle (Logic) for base Qwen2.5-7B/32B vs. RL-tuned GURU-7B/32B. **(b)** Pass@k under different decoding settings: higher sampling temperature or larger top-p broadens exploration and offsets RL-induced entropy collapse.

## **5 Related Work**

**Improving LLM Reasoning with RL** Enhancing LLM reasoning with RL has gained significant interest [\(Shao](#page-15-10) [et al.,](#page-15-10) [2024;](#page-15-10) [Yuan et al.,](#page-15-14) [2024;](#page-15-14) [Pang et al.,](#page-14-12) [2024;](#page-14-12) [Wang et al.,](#page-15-15) [2024;](#page-15-15) [Havrilla et al.,](#page-13-16) [2024\)](#page-13-16) due to its potential for self-improvement without relying on human-labeled solutions. Recent milestones—most notably OpenAI o1 [\(OpenAI,](#page-14-13) [2024\)](#page-14-13) and DeepSeek-R1 [\(Guo et al.,](#page-13-0) [2025\)](#page-13-0)—demonstrate that RL can be scaled to industry-level models, markedly advancing state-of-the-art reasoning benchmarks and revealing emergent capabilities such as long reasoning chains. Following initial successes, a significant body of work has explored open recipes to build strong reasoning models. Efforts such as Open-Reasoner-Zero [\(Hu et al.,](#page-13-1) [2025a\)](#page-13-1), Skywork-OR1 [\(He](#page-13-6) [et al.,](#page-13-6) [2025\)](#page-13-6), DeepScaler [\(Luo et al.,](#page-14-3) [2025b\)](#page-14-3), and SimpleRL [\(Zeng et al.,](#page-15-2) [2025\)](#page-15-2) have notably leveraged extensive mathematical data to achieve state-of-the-art performance on complex math benchmarks. Similarly, Deep-Coder [\(Luo et al.,](#page-14-5) [2025a\)](#page-14-5) focused on RL for code generation tasks. While powerful within their specific areas, this domain-specific focus inherently limits the generalizability of the resulting models across the broader landscape of reasoning tasks. Concurrent works like General-Reasoner [\(Ma et al.,](#page-14-7) [2025\)](#page-14-7) and Nemotron-CrossThinker [\(Akter et al.,](#page-12-7) [2025\)](#page-12-7) have begun to explore broader domains for RL training. However, their scope remains constrained to STEM problems, leaving many crucial scenarios, including comprehensive coding, logic, and tabular analysis unexplored by large-scale open RL efforts. Addressing this critical gap in open resources and models for general reasoning via RLVR, we introduce GURU, a novel multi-domain dataset spanning six reasoning domains: Math, Code, Science, Logic, Simulation, and Tabular. Utilizing GURU, we train GURU-7B/32B, general reasoning models optimized via RL exclusively on open data using our GURU dataset and see state-of-the-art results on general reasoning among open models trained with publicly available data using RL.

**Understanding RL for Reasoning** Alongside the development of specialized models, a parallel line of research has critically examined the underlying mechanisms of how reinforcement learning enhances reasoning. This inquiry has revealed that RL for reasoning may not fundamentally lead to the acquisition of new reasoning capabilities, but rather serves as an elicitor of latent abilities already present in the base model. For instance, [Yue et al.](#page-15-3) [\(2025\)](#page-15-3) find that RL primarily improves the sampling efficiency of correct reasoning paths that already exist in the base model's distribution, rather than expanding its core reasoning capacity. This amplifier hypothesis is further substantiated by several surprising findings. Research has shown that significant performance gains can be triggered by seemingly uninformative signals, such as rewards that are random or even known to be incorrect [\(Shao et al.,](#page-14-1) [2025a\)](#page-14-1), or through training on just a single example [\(Wang](#page-15-4) [et al.,](#page-15-4) [2025a\)](#page-15-4). In an extreme case, [Agarwal et al.](#page-12-0) [\(2025\)](#page-12-0) demonstrate that strong reasoning performance can be unlocked without any labeled data at all, using only entropy minimization as the training objective. However, this body of work also highlights that these effects are highly model-dependent. The success of such techniques is often pronounced in models like Qwen, while being significantly less effective on other model families [\(Shao et al.,](#page-14-1) [2025a;](#page-14-1) [Zeng et al.,](#page-15-2) [2025;](#page-15-2) [Agarwal et al.,](#page-12-0) [2025\)](#page-12-0). Recent work has begun to challenge this amplifier-only perspective: [Liu et al.](#page-14-14) [\(2025\)](#page-14-14) demonstrate that prolonged RL training can discover novel reasoning strategies inaccessible to base models even under extensive sampling, particularly in domains where the base model initially struggles. However, our work systematically revisits these underlying mechanisms from a novel cross-domain perspective and at multiple scales, attempting to explore whether the role of RL exhibits high domain-dependency and whether synergistic multi-domain training can yield mutual benefits for general reasoning.

## **6 Conclusion**

In this work, we introduce GURU, a curated reinforcement learning (RL) dataset of 92K high-quality examples spanning six diverse domains—Math, Code, Science, Logic, Simulation, and Tabular reasoning—designed to bridge a critical gap in RL-based reasoning research. Our extensive empirical analysis reveals that the effects of RL on reasoning behaviors are highly domain-sensitive. For example, while RL can effectively elicit pretrained knowledge in well-covered domains like Math, Code, and Science, it is more likely to foster genuine reasoning capabilities in underrepresented domains in pretraining. We develop GURU-7B and GURU-32B, two general-purpose reasoning models trained via RL on the full GURU dataset. Across 17 benchmark tasks across six reasoning domains, these models achieve state-of-the-art performance among open models trained with RL on open data, marking a substantial advance in general reasoning capability. We release the full GURU dataset, models, evaluation suite, and training code to encourage further research into general-purpose RL for reasoning.

## **References**

- <span id="page-12-0"></span>Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in llm reasoning, 2025. URL <https://arxiv.org/abs/2505.15134>.
- <span id="page-12-7"></span>Syeda Nahida Akter, Shrimai Prabhumoye, Matvei Novikov, Seungju Han, Ying Lin, Evelina Bakhturi, Eric Nyberg, Yejin Choi, Mostofa Patwary, Mohammad Shoeybi, et al. Nemotron-crossthink: Scaling self-learning beyond math reasoning. arXiv preprint arXiv:2504.13941, 2025.
- <span id="page-12-8"></span>Alon Albalak, Duy Phung, Nathan Lile, Rafael Rafailov, Kanishk Gandhi, Louis Castricato, Anikait Singh, Chase Blagden, Violet Xiang, Dakota Mahan, and Nick Haber. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models, 2025. URL <https://arxiv.org/abs/2502.17387>.
- <span id="page-12-5"></span>Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.
- <span id="page-12-1"></span>Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosallanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi, Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin, Matvei Novikov, Daria Gitman, Evelina Bakhturina, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi Kong, Markus Kliegl, Rabeeh Karimi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha, Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Abhinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor-Yiing Su, Guyue Huang, Terry Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru, Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf, Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya, Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam, Smita Ithape, Karthik Ramamoorthy, Yuting Wu, Suguna Varshini Velury, Omri Almog, Joyjit Daw, Denys Fridman, Erick Galinkin, Michael Evans, Shaona Ghosh, Katherine Luna, Leon Derczynski, Nikki Pope, Eileen Long, Seth Schneider, Guillermo Siman, Tomasz Grzegorzek, Pablo Ribalta, Monika Katariya, Chris Alexiuk, Joey Conway, Trisha Saar, Ann Guan, Krzysztof Pawelec, Shyamala Prayaga, Oleksii Kuchaiev, Boris Ginsburg, Oluwatobi Olabiyi, Kari Briski, Jonathan Cohen, Bryan Catanzaro, Jonah Alben, Yonatan Geifman, and Eric Chung. Llama-nemotron: Efficient reasoning models, 2025. URL <https://arxiv.org/abs/2505.00949>.
- <span id="page-12-4"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021a.
- <span id="page-12-9"></span>Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. Hybridqa: A dataset of multi-hop question answering over tabular and textual data. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1026–1036, 2020.
- <span id="page-12-10"></span>Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, et al. Finqa: A dataset of numerical reasoning over financial data. arXiv preprint arXiv:2109.00122, 2021b.
- <span id="page-12-6"></span>Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. Finqa: A dataset of numerical reasoning over financial data, 2022a. URL <https://arxiv.org/abs/2109.00122>.
- <span id="page-12-11"></span>Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. Convfinqa: Exploring the chain of numerical reasoning in conversational finance question answering. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6279–6292, 2022b.
- <span id="page-12-3"></span>Zhoujun Cheng, Haoyu Dong, Zhiruo Wang, Ran Jia, Jiaqi Guo, Yan Gao, Shi Han, Jian-Guang Lou, and Dongmei Zhang. Hitab: A hierarchical table dataset for question answering and natural language generation. arXiv preprint arXiv:2108.06712, 2021.
- <span id="page-12-2"></span>Francois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2024: Technical report. arXiv preprint arXiv:2412.04604, 2024.

- <span id="page-13-10"></span>Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard. Arc-agi-2: A new challenge for frontier ai reasoning systems, 2025. URL <https://arxiv.org/abs/2505.11831>.
- <span id="page-13-12"></span>Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.
- <span id="page-13-17"></span>Yangruibo Ding, Jinjun Peng, Marcus Min, Gail Kaiser, Junfeng Yang, and Baishakhi Ray. Semcoder: Training code language models with comprehensive semantics reasoning. Advances in Neural Information Processing Systems, 37: 60275–60308, 2024.
- <span id="page-13-14"></span>Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I Wang. Cruxeval: A benchmark for code reasoning, understanding and execution. arXiv preprint arXiv:2401.03065, 2024.
- <span id="page-13-0"></span>Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- <span id="page-13-16"></span>Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large language models to reason with reinforcement learning. arXiv preprint arXiv:2403.04642, 2024.
- <span id="page-13-6"></span>Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner series. [https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reaonser-Serie](https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680) [s-1d0bc9ae823a80459b46c149e4f51680](https://capricious-hydrogen-41c.notion.site/Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680), 2025. Notion Blog.
- <span id="page-13-13"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- <span id="page-13-1"></span>Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025a.
- <span id="page-13-4"></span>Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025b. URL [https://arxiv.org/](https://arxiv.org/abs/2503.24290) [abs/2503.24290](https://arxiv.org/abs/2503.24290).
- <span id="page-13-18"></span>HuggingFaceTB and PyEdu Developers. SmolLM Corpus Dataset. [https://huggingface.co/datasets/Huggin](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) [gFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), 2024.
- <span id="page-13-7"></span>Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.
- <span id="page-13-5"></span>Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bing Liu. Continual pre-training of language models. In Proceedings of The Eleventh International Conference on Learning Representations (ICLR-2023), 2023.
- <span id="page-13-15"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023. URL <https://arxiv.org/abs/2309.06180>.
- <span id="page-13-2"></span>Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13:9, 2024a.
- <span id="page-13-11"></span>Junlong Li, Daya Guo, Dejian Yang, Runxin Xu, Yu Wu, and Junxian He. Codei/o: Condensing reasoning patterns via code input-output prediction. arXiv preprint arXiv:2502.07316, 2025.
- <span id="page-13-3"></span>Kaixin Li. Verified taco problems. <https://huggingface.co/datasets/likaixin/TACO-verified>, 2024. URL <https://huggingface.co/datasets/likaixin/TACO-verified>.
- <span id="page-13-8"></span>Wen-Ding Li, Keya Hu, Carter Larsen, Yuqing Wu, Simon Alford, Caleb Woo, Spencer M. Dunn, Hao Tang, Michelangelo Naim, Dat Nguyen, Wei-Long Zheng, Zenna Tavares, Yewen Pu, and Kevin Ellis. Combining induction and transduction for abstract reasoning, 2024b. URL <https://arxiv.org/abs/2411.02272>.
- <span id="page-13-9"></span>Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. Zebralogic: On the scaling limits of llms for logical reasoning, 2025. URL <https://arxiv.org/abs/2502.01100>.

- <span id="page-14-8"></span>Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- <span id="page-14-4"></span>Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. 2025.
- <span id="page-14-14"></span>Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models, 2025. URL [https://arxiv.org/abs/25](https://arxiv.org/abs/2505.24864) [05.24864](https://arxiv.org/abs/2505.24864).
- <span id="page-14-9"></span>Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- <span id="page-14-5"></span>Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level, 2025a. URL [https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Sourc](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) [e-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51). Notion Blog.
- <span id="page-14-3"></span>Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025b. URL [https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-M](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) [odel-by-Scaling-RL-19681902c1468005bed8ca303013a4e2](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2). Notion Blog.
- <span id="page-14-7"></span>Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. [https://github.com/TIGER-AI-Lab/General-Reasoner/blob/main/Gene](https://github.com/TIGER-AI-Lab/General-Reasoner/blob/main/General_Reasoner.pdf) [ral\\_Reasoner.pdf](https://github.com/TIGER-AI-Lab/General-Reasoner/blob/main/General_Reasoner.pdf), 2025.
- <span id="page-14-10"></span>MAA. American invitational mathematics examination - aime. In American Invitational Mathematics Examination - AIME 2024, February 2024. URL [https://maa.org/math-competitions/american-invitational-mathemati](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime) [cs-examination-aime](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime).
- <span id="page-14-6"></span>Justus Mattern, Sami Jaghouar, Manveer Basra, Jannik Straube, Matthew Di Ferrante, Felix Gabriel, Jack Min Ong, Vincent Weisser, and Johannes Hagemann. Synthetic-1: Two million collaboratively generated reasoning traces from deepseek-r1, 2025. URL <https://www.primeintellect.ai/blog/synthetic-1-release>.
- <span id="page-14-17"></span>Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. 2021.
- <span id="page-14-13"></span>OpenAI. OpenAI o1 System Card. <https://openai.com/index/openai-o1-system-card/>, 2024.
- <span id="page-14-0"></span>OpenAI. Introducing openai o3 and o4-mini, 2025. URL [https://openai.com/index/introducing-o3-and-o](https://openai.com/index/introducing-o3-and-o4-mini/) [4-mini/](https://openai.com/index/introducing-o3-and-o4-mini/). Accessed: 2025-06-12.
- <span id="page-14-12"></span>Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems, 37:116617–116637, 2024.
- <span id="page-14-18"></span>Panupong Pasupat and Percy Liang. Compositional semantic parsing on semi-structured tables. arXiv preprint arXiv:1508.00305, 2015.
- <span id="page-14-11"></span>David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2311.12022) [2311.12022](https://arxiv.org/abs/2311.12022).
- <span id="page-14-15"></span>Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought, 2023. URL <https://arxiv.org/abs/2210.01240>.
- <span id="page-14-16"></span>Abulhair Saparov, Srushti Pawar, Shreyas Pimpalgaonkar, Nitish Joshi, Richard Yuanzhe Pang, Vishakh Padmakumar, Seyed Mehran Kazemi, Najoung Kim, and He He. Transformers struggle to learn to search, 2025. URL [https:](https://arxiv.org/abs/2412.04703) [//arxiv.org/abs/2412.04703](https://arxiv.org/abs/2412.04703).
- <span id="page-14-1"></span>Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards: Rethinking training signals in rlvr. [https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinkin](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) [g-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f), 2025a. Notion Blog.
- <span id="page-14-2"></span>Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards:

- Rethinking training signals in rlvr. [https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinkin](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f) [g-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f), 2025b. Notion Blog.
- <span id="page-15-10"></span>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- <span id="page-15-9"></span>Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.
- <span id="page-15-11"></span>P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, King Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, Kaixin Deng, Shawn Gavin, Shian Jia, Sichao Jiang, Yiyan Liao, Rui Li, Qinrui Li, Sirun Li, Yizhi Li, Yunwen Li, David Ma, Yuansheng Ni, Haoran Que, Qiyao Wang, Zhoufutu Wen, Siwei Wu, Tyshawn Hsing, Ming Xu, Zhenzhu Yang, Zekun Moore Wang, Junting Zhou, Yuelin Bai, Xingyuan Bu, Chenglin Cai, Liang Chen, Yifan Chen, Chengtuo Cheng, Tianhao Cheng, Keyi Ding, Siming Huang, Yun Huang, Yaoru Li, Yizhe Li, Zhaoqun Li, Tianhao Liang, Chengdong Lin, Hongquan Lin, Yinghao Ma, Tianyang Pang, Zhongyuan Peng, Zifan Peng, Qige Qi, Shi Qiu, Xingwei Qu, Shanghaoran Quan, Yizhou Tan, Zili Wang, Chenqing Wang, Hao Wang, Yiya Wang, Yubo Wang, Jiajun Xu, Kexin Yang, Ruibin Yuan, Yuanhao Yue, Tianyang Zhan, Chun Zhang, Jinyang Zhang, Xiyue Zhang, Xingjian Zhang, Yue Zhang, Yongchi Zhao, Xiangyu Zheng, Chenghua Zhong, Yang Gao, Zhoujun Li, Dayiheng Liu, Qian Liu, Tianyu Liu, Shiwen Ni, Junran Peng, Yujia Qin, Wenbo Su, Guoyin Wang, Shi Wang, Jian Yang, Min Yang, Meng Cao, Xiang Yue, Zhaoxiang Zhang, Wangchunshu Zhou, Jiaheng Liu, Qunshu Lin, Wenhao Huang, and Ge Zhang. Supergpqa: Scaling llm evaluation across 285 graduate disciplines, 2025. URL <https://arxiv.org/abs/2502.14739>.
- <span id="page-15-15"></span>Huaijie Wang, Shibo Hao, Hanze Dong, Shenao Zhang, Yilin Bao, Ziran Yang, and Yi Wu. Offline reinforcement learning for llm multi-step reasoning. arXiv preprint arXiv:2412.16145, 2024.
- <span id="page-15-4"></span>Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Reinforcement learning for reasoning in large language models with one training example, 2025a. URL <https://arxiv.org/abs/2504.20571>.
- <span id="page-15-1"></span>Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Revisiting mid-training in the era of rl scaling. <https://tinyurl.com/OctoThinker>, 2025b. Notion Blog.
- <span id="page-15-13"></span>Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Ben Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Siddartha Naidu, et al. Livebench: A challenging, contamination-free llm benchmark. arXiv preprint arXiv:2406.19314, 2024.
- <span id="page-15-7"></span>Yunhui Xia, Wei Shen, Yan Wang, Jason Klein Liu, Huifeng Sun, Siyue Wu, Jian Hu, and Xiaolong Xu. Leetcodedataset: A temporal dataset for robust evaluation and efficient training of code llms, 2025. URL [https://arxiv.org/abs/25](https://arxiv.org/abs/2504.14655) [04.14655](https://arxiv.org/abs/2504.14655).
- <span id="page-15-6"></span>An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- <span id="page-15-0"></span>Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- <span id="page-15-14"></span>Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al. Advancing llm reasoning generalists with preference trees. arXiv preprint arXiv:2404.02078, 2024.
- <span id="page-15-3"></span>Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.
- <span id="page-15-16"></span>Wojciech Zaremba and Ilya Sutskever. Learning to execute. arXiv preprint arXiv:1410.4615, 2014.
- <span id="page-15-2"></span>Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.
- <span id="page-15-5"></span>Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo chamber: Rl post-training amplifies behaviors learned in pretraining. arXiv preprint arXiv:2504.07912, 2025.
- <span id="page-15-8"></span>Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. Multihiertt: Numerical reasoning over multi hierarchical tabular and textual data. arXiv preprint arXiv:2206.01347, 2022.
- <span id="page-15-12"></span>Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instructionfollowing evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

# **Appendix**

# **A Limitations**

We are encouraged by the promising improvements observed on general reasoning datasets through the application of RL with our 92K sample data. However, for 7B and 32B models, our current training only lasts three and two epochs due to computing resource constraints. It is conceivable that extended RL training might yield even more pronounced results according to [Liu et al.](#page-14-14) [\(2025\)](#page-14-14) and [Yu et al.](#page-15-0) [\(2025\)](#page-15-0).

Our approach initiated training directly from a base model using RL, bypassing an initial SFT phase. This decision was made to ensure that our conclusions regarding cross-domain generalization remained independent of specific SFT data dependencies. However, we expect that integrating an SFT then RL pipeline could further enhance performance largely, and we plan to explore this in future work.

# <span id="page-16-0"></span>**B Training Data Curation Details**

In this section, we elaborate the data sourcing and curation details of all used dataset in GURU.

**New Assets**: We are committed to full transparency and open science. All the new assets we've created, as well as those repurposed from existing datasets, are fully open-source and publicly available. We believe this approach fosters collaboration and accelerates progress within the research community.

<span id="page-16-1"></span>

| Domain     | Dataset                                                                         | Data Source Link                                                                                                                        |
|------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Math       | OR1<br>DAPO<br>DeepScaler                                                       | Skywork/Skywork-OR1-RL-Data<br>BytedTsinghua-SIA/DAPO-Math-17k<br>agentica-org/DeepScaleR-Preview-Dataset                               |
| Code       | LeetCode<br>TACO-Verified<br>PrimeIntellect<br>LiveCodeBench                    | newfacade/LeetCodeDataset<br>likaixin/TACO-verified<br>agentica-org/DeepCoder-Preview-Dataset<br>agentica-org/DeepCoder-Preview-Dataset |
| Science    | WebInstruct-Verified                                                            | TIGER-Lab/WebInstruct-verified                                                                                                          |
| Logic      | Zebra Puzzle<br>Ordering Puzzle<br>Graph Puzzle<br>ARC-AGI<br>ARC-AGI-2<br>BARC | –<br>–<br>–<br>fchollet/ARC-AGI<br>arcprize/ARC-AGI-2<br>barc0/200k_HEAVY_gpt4oproblems                                                 |
| Simulation | Code I/O (PyEdu)                                                                | hkust-nlp/CodeIO-PyEdu-Reasoning-Raw                                                                                                    |
| Tabular    | HiTab<br>MultiHierTT                                                            | microsoft/HiTab<br>psunlpgroup/MultiHiertt                                                                                              |

#### **B.1 Math**

For our math subset, we focus on gathering some of the most effective datasets recently released. We prioritize OR1 [\(He et al.,](#page-13-6) [2025\)](#page-13-6), DAPO [\(Yu et al.,](#page-15-0) [2025\)](#page-15-0), and DeepScaler [\(Luo et al.,](#page-14-3) [2025b\)](#page-14-3) due to their strong performance in advancing mathematical reasoning in large language models. For other math reasoning datasets, such as BigMath [\(Albalak et al.,](#page-12-8) [2025\)](#page-12-8), we observe that their quality is inconsistent, and they contain more duplicate queries, which led us to exclude them from our selection. We identify the original sources of these datasets and, as shown in [Figure 6,](#page-17-0) meticulously trace their origins to understand their provenance and potential overlaps. We further standardize the data formats across all selected datasets, ensuring consistency and ease of integration into our training pipeline. A critical step is identifying and eliminating overlapping or duplicate problems within and across datasets. This ensures that our model is exposed to a diverse set of unique challenges, preventing overfitting to specific problem types and promoting broader generalization.

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

**Figure 6:** The sankey diagram tracing the provenance of all math data included in GURU.

By carefully curating these datasets, we aim to provide a robust and varied foundation for training and evaluating advanced mathematical reasoning capabilities in LLMs.

We instruct the model to output its final answer in \boxed{}, and extract the content in the box as the model's predicted answer. Following PRIME [\(Cui et al.,](#page-13-12) [2025\)](#page-13-12), the reward is granted if the LLM's generated answer either normalizes to the same string or is numerically equal (convertible to floats and equivalent) to the ground truth, or if Sympy[2](#page-17-1) can simplify the difference between the predicted and ground truth expressions to zero.

#### **B.2 Code**

Our code subset consists of programming problems collected from a variety of sources, including online coding platforms and programming competitions. Specifically, we include real-world challenges from LeetCode [\(Xia et al.,](#page-15-7) [2025\)](#page-15-7), curated and validated tasks from TACO-Verified [\(Li,](#page-13-3) [2024\)](#page-13-3), problems from PrimeIntellect [\(Mattern et al.,](#page-14-6) [2025\)](#page-14-6), and historical contest problems from LiveCodeBench [\(Jain et al.,](#page-13-7) [2024\)](#page-13-7). For PrimeIntellect and LiveCodeBench, we adopt the pre-filtered subsets provided by the DeepCoder dataset [\(Luo](#page-14-5) [et al.,](#page-14-5) [2025a\)](#page-14-5), which applies rigorous cleaning and verification protocols. A comprehensive breakdown of data source and corresponding Hugging Face links is provided in [Table 4.](#page-16-1) We also construct a Sankey diagram [Figure 7](#page-18-0) that traces the provenance of all code data included in our benchmark.

Each problem is presented as a program synthesis task requiring the model to generate a Python solution that satisfies a given problem specification to pass certain test cases. Specifically, we have two problem formats: (1) pure function format, where a function is implemented and validated via inline assert statements; and (2) stdin/stdout format, where the model must write a full script that reads from standard input and writes to standard output. All programs are executed with a uniform 30-second timeout and 10 GB memory limit to ensure stability and computational efficiency. Programs that successfully pass all test cases receive a reward of 1. We discard any example whose reference solution fails its own unit tests. To ensure stable reward computation in high-concurrency training environments, we further filter out samples with large inputs (e.g., stdin exceeding 1MB) and randomly subsample at most 8 test cases per problem. Currently, our setup

<span id="page-17-1"></span><sup>2</sup><https://www.sympy.org/en/index.html>

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

**Figure 7:** The sankey diagram tracing the provenance of all code data included in GURU.

only focuses on Python, we leave the inclusion of additional programming languages (e.g., Java, C++, Go, etc.) as future work, to explore both cross-lingual transfer and the necessity of language-level diversity in RL training for code.

#### **B.3 Science**

Our science data leverages WebInstruct-Verified [\(Ma et al.,](#page-14-7) [2025\)](#page-14-7). This dataset comprises over 230K verifiable question-answer pairs across diverse scientific disciplines, including physics, chemistry, biology, and finance. To ensure the dataset's suitability for reinforcement learning, we retain only university- and PhD-level questions in physics, chemistry, and biology, removing multiple-choice and boolean formats to avoid answer shortcuts. We also discard samples with extremely high numeric precision, which often lead to brittle or noisy reward evaluations.

We instruct the model to output its final answer in \boxed{}, and extract the content in the box as the model's predicted answer. To calculate the reward, we apply a 1.5B model verifier[3](#page-18-1) provided by [Ma et al.](#page-14-7) [\(2025\)](#page-14-7) to compare the reference and model prediction.

#### **B.4 Logic**

To evaluate and improve the logical reasoning capabilities of language models, we design three synthetic tasks and collect three existing datasets, all of which require multi-step deduction under structured constraints. For all logical tasks, we instruct the model to wrap the prediction into special tags <answer> and </answer> and extract the prediction from there. The model prediction will be parsed into a list, and the reward is granted if every element matches the reference answer.

**Zebra Puzzles** Zebra puzzles are a class of logic grid problems that require deducing a complete set of attribute assignments (e.g., who owns the zebra) based on a set of relational constraints. The task becomes exponentially more difficult as the number of entities and constraints increases, making it a challenging benchmark for multi-step reasoning. We construct puzzles by first sampling a ground-truth table over

<span id="page-18-1"></span><sup>3</sup><https://huggingface.co/TIGER-Lab/general-verifier>

randomly chosen attribute categories and values, then backsolving to generate constraints. These include positional relations (e.g., left/right, middle), equality/inequality constraints, disjunctive logic, relative distances, and parity-based rules. We define 20 difficulty levels based on constraint complexity, support puzzles up to 15 × 10 in size, and optionally include redundant constraints to increase distractor robustness. To ensure a meaningful symbolic reasoning challenge, we retain only puzzles with grid sizes of at least 10 objects and 5 attributes.

**Ordering Puzzles** Ordering puzzles require models to recover a linear order over n objects (up to 50) based on a set of relational constraints. We first sample a ground-truth permutation and then generate constraints from it. These include absolute positions (e.g., "A is at position 3"), relative positions (e.g., "A is to the left of B"), adjacency relations (e.g., "A is immediately to the right of B"), and distance-based constraints (e.g., "There are 3 items between A and B"). To ensure uniqueness, we apply constraint propagation to select a minimal satisfiable subset. Our dataset spans diverse entity types, such as animals, people, and vehicles. To maintain task complexity, we filter out samples with fewer than 20 objects.

**Graph Puzzles** Graph puzzles task models with identifying a valid path through a directed graph defined by natural language implications. We build upon the structure of PrOntoQA [\(Saparov and He,](#page-14-15) [2023\)](#page-14-15), extending it to free-form language using the setup of Graph Search Tasks [\(Saparov et al.,](#page-14-16) [2025\)](#page-14-16). Each instance defines a set of predicates that implicitly specify the graph structure, along with a query that requires multi-hop traversal. We vary the number of nodes, edge density, and path length to control task difficulty, with longer and denser paths posing greater challenges for compositional reasoning. At heuristc filtering stage, we retain only instances with more than 10 nodes and at least 3-hop queries to ensure the graph structure supports multi-step traversal reasoning.

**ARC-AGI** The original Abstraction and Reasoning Corpus for Artificial General Intelligence [\(Chollet et al.,](#page-12-2) [2024\)](#page-12-2) (ARC-AGI-1) comprises 400 public training tasks and an equally sized public evaluation split, all expressed as grid-based input–output transformations. Tasks are deliberately designed to be straightforward for humans yet challenging for machines, probing fluid, compositional intelligence over a space of 30 × 30 coloured grids. Each instance supplies 2–5 demonstration pairs (as input examples) and one unseen test grid. We include the full training split of ARC-AGI-1 when constructing GURU.

**ARC-AGI-2** ARC-AGI-2 [\(Chollet et al.,](#page-13-10) [2025\)](#page-13-10) expands to ARC-AGI to 1,000 public training tasks and 120 public evaluation tasks while introducing two hidden private test sets for competition use. The new version rebalances difficulty so that average human solvers achieve roughly 66% accuracy, sharpens the trial budget to two attempts per test grid, and retains the canonical grid-world format. By fusing novel tasks with the ARC-AGI-1 corpus, ARC-AGI-2 covers a broader concept space (symmetry, physics-like dynamics, object counting, etc.) while preserving the "few demonstrations, zero prior training" ethos. We include the public training portion in GURU.

**BARC** BARC [\(Li et al.,](#page-13-8) [2024b\)](#page-13-8) is a large-scale synthetic extension containing approximately 200,000 ARC-style problems generated from 160 manually curated seed programs via an LLM-guided remix pipeline. Each synthetic task is accompanied by an executable Python reference code and an input generator, enabling controllable difficulty and unlimited fresh samples for augmentation or test-time fine-tuning. We sample 3.4k examples whose input lengths do not exceed the token limit of our training.

#### **B.5 Simulation**

Simulation is the process of creating a model of a real or hypothetical system and executing it to observe its dynamic behavior over time. This process is inherently linked to reasoning because it requires the systematic application of logical inference to predict the consequences of actions or changes within a system. In this domain, our work focuses on the specific subproblem of simulating Python program execution. Prior research has explored various facets of this problem. [Zaremba and Sutskever](#page-15-16) [\(2014\)](#page-15-16) trained LSTM to evaluate short computer programs. [Nye et al.](#page-14-17) [\(2021\)](#page-14-17) demonstrated the capability of LLMs to predict code execution traces. CruxEval [\(Gu et al.,](#page-13-14) [2024\)](#page-13-14) further advanced this by introducing a benchmark to evaluate LLM performance

in both inferring inputs from given function outputs and predicting outputs from given inputs. Recent works, such as SemCoder [\(Ding et al.,](#page-13-17) [2024\)](#page-13-17) and Code I/O [\(Li et al.,](#page-13-11) [2025\)](#page-13-11), have used SFT with code execution prediction tasks to enhance LLM capabilities in code generation and general reasoning.

In GURU, we repurpose the raw data from Code I/O for our RL training. The Code I/O dataset, sourced from PyEdu [\(HuggingFaceTB and PyEdu Developers,](#page-13-18) [2024\)](#page-13-18), comprises 4.5 million verified programs with corresponding input-to-output and output-to-input samples. Inputs for this dataset were generated by DeepSeek-v2.5, and outputs were obtained through actual function execution. Since multiple input-output variants may exist for each program, to ensure diversity within our training data, we first applied a heuristic filter to retain only a single pair per problem, reducing the dataset to 227K programs. Subsequently, we randomly sampled 12K I/O pairs. A final difficulty-based filtering step was then applied, resulting in a refined dataset of 3.7K I/O samples used for our RL training.

We instruct the model to wrap the answer in a json code block with Markdown grammar, and extract the content in the code block to compare with the reference answer. The reward will be granted if they are strictly the same.

#### **B.6 Tabular**

Tabular data are ubiquitous and serve as vital media for data organization and analysis. Table Question Answering (TQA), which involves deriving an answer from one or more tables given a natural language query, is a fundamental and critical problem in data understanding. Early work in TQA, such as WikiTQ [\(Pasupat](#page-14-18) [and Liang,](#page-14-18) [2015\)](#page-14-18) and HybridQA [\(Chen et al.,](#page-12-9) [2020\)](#page-12-9), explored answering queries on WikiTables, sometimes incorporating accompanying text. The complexity of TQA increased with datasets like HiTab [\(Cheng et al.,](#page-12-3) [2021\)](#page-12-3) and MultiHiertt [\(Zhao et al.,](#page-15-8) [2022\)](#page-15-8), which focused on hierarchical tables featuring multiple levels of headers, demanding more sophisticated reasoning. More specialized applications emerged with FinQA [\(Chen](#page-12-10) [et al.,](#page-12-10) [2021b\)](#page-12-10) and ConvFinQA [\(Chen et al.,](#page-12-11) [2022b\)](#page-12-11), addressing TQA within the financial domain.

In GURU, we adapt the training sets from HiTab (7.5K samples) and MultiHiertt (7.8K samples) for our RL training. We chose these datasets because they necessitate complex multi-hop reasoning over tabular structures. For table representation, we linearize the tables into a markdown format. If multiple tables are present for a single query, we concatenate them, separated by a line break.

We instruct the model to output its final answer in \boxed{}, and extract the content in the box as the model's predicted answer. To calculate the reward, we apply the same program for math reasoning to compare the model prediction and the reference answer.

## <span id="page-20-0"></span>**C Evaluation Benchmark Details**

We provide detailed benchmark statistics in [Table 5.](#page-21-1) For online evaluations, we randomly sample up to 200 examples from each dataset to ensure efficiency. An exception is ARC-AGI, where we select only those examples with input prompt lengths shorter than 4,096 tokens. This introduces a distributional discrepancy compared to the full offline set (400 examples), which partially explains the performance gap between online and offline settings. Another exception is MultiHiertt, where we specifically include only examples with table inputs and no accompanying textual context, to deliberately evaluate the model's multi-table reasoning capabilities. For offline evaluations, large datasets (i.e., those with more than 1,000 examples) are subsampled to 1,000 for computational efficiency (SuperGPQA, HiTab). To ensure statistical reliability, we generate four completions per example for datasets with fewer than 400 samples and report the average results. For AIME24, we generate 32 completions per example. For LiveBench, we evaluate the language, reasoning, and data analysis subsets, following [Li et al.](#page-13-11) [\(2025\)](#page-13-11).

<span id="page-21-1"></span>**Table 5:** Evaluation benchmark statistics across all domains. The left table presents offline evaluations, while the right table presents online evaluations. Benchmarks marked with \* indicate that a subset of the official evaluation set was used, either for computational efficiency or to focus on specific reasoning capabilities.

#### **Offline Evaluation**

| Domain     | Benchmark                                             | Size                |
|------------|-------------------------------------------------------|---------------------|
| Math       | AIME24 (avg@32)<br>MATH500                            | 30<br>500           |
| Code       | LiveCodeBench v5 (avg@4)<br>MBPP<br>HumanEval (avg@4) | 279<br>500<br>164   |
| Science    | GPQA-diamond (avg@4)<br>SuperGPQA*                    | 198<br>1000         |
| Logic      | ARC-AGI (avg@4)<br>Zebra Puzzle (avg@4)               | 400<br>200          |
| Simulation | Code I/O<br>CruxEval-I<br>CruxEval-O                  | 200<br>800<br>800   |
| Tabular    | FinQA<br>HiTab* (avg@4)<br>MultiHiertt* (avg@4)       | 1100<br>1000<br>336 |
| Others     | IFEval<br>LiveBench* (avg@4)                          | 541<br>440          |

#### **Online Evaluation**

| Domain     | Benchmark        | Size |
|------------|------------------|------|
|            | AIME24 (avg@8)   | 30   |
| Math       | AMC (avg@4)      | 332  |
|            | MATH500          | 500  |
|            | LiveCodeBench v5 | 279  |
| Code       | MBPP             | 500  |
|            | HumanEval        | 164  |
| Science    | SuperGPQA        | 200  |
|            | ARC-AGI          | 200  |
| Logic      | Zebra Puzzle     | 200  |
|            | Ordering Puzzle  | 100  |
| Simulation | Code I/O         | 200  |
|            | HiTab            | 200  |
| Tabular    | MultiHiertt      | 200  |

## **D More Results**

## **D.1 Cross-Domain Transferability**

To supplement our main findings on differential transferability (see [Section 3.1\)](#page-5-0), we also conduct the same set of cross-domain RL experiments using a larger backbone, Qwen-32B-Base. Results are presented in [Figure 8,](#page-22-0) following the same experimental setup and visualization protocol as described in [Figure 3.](#page-6-1)

<span id="page-21-0"></span>We observe that the overall pattern of cross-domain transferability in Qwen-32B-Base closely mirrors that of the 7B variant, reaffirming our core conclusions regarding the role of pretraining priors, task difficulty, and training composition. Specifically, we find that (1) domains with strong pretraining coverage—such as Math, Code, and Science—continue to exhibit substantial gains under cross-domain RL, suggesting that reinforcement learning primarily serves to elicit capabilities already encoded in the base model. In the 32B setting, these benefits extend modestly to additional domains, including Tabular, likely due to the model's larger capacity to retain and utilize relevant prior knowledge, and stronger instruction following capability. However, the asymmetry in transferability persists, as domains with weaker representation during pretraining remain difficult to improve through out-of-domain supervision alone. (2) Task difficulty remains a key factor in determining transfer effectiveness: easier tasks within each domain (e.g., MATH500, AMC, HumanEval, MBPP) consistently benefit from cross-domain training, while more challenging benchmarks (e.g., AIME24, LiveCodeBench, ARC-AGI, CodeI/O) show limited or negligible improvement. This reinforces the limitation of relying solely on cross-domain RL to develop general reasoning capabilities on complex tasks. (3) Mixed-domain training continues to be highly effective in the larger model: a uniformly sampled multi-domain corpus achieves performance comparable to—or in many cases exceeding—single-domain RL across most benchmarks. We observe no strong signs of domain interference under this setup, indicating that multi-domain RL remains a robust approach for generalization. Nonetheless, as the number and diversity of domains scale further, more refined balancing strategies may become necessary to sustain this advantage.

<span id="page-22-0"></span>![](_page_22_Figure_0.jpeg)

**Figure 8: Cross-Domain RL Transfer Performance per Task with Qwen-32B-Base.** This figure replicates the heatmap analysis from Figure 3 using Qwen-32B-Base. Warmer colors indicate higher performance gains from RL training on each source domain (rows) when evaluated on target domains (columns). Each cell shows the min-max normalized validation accuracy, with the best per-domain accuracy (%) labeled. In-domain training results are outlined.

### D.2 Pass@k Dynamics during Training

We visualize Pass@k curves of different GURU-7B training steps in Figure 9.

<span id="page-22-1"></span>![](_page_22_Figure_4.jpeg)

**Figure 9:** Pass@k performance curves of GURU model at different training steps on the AIME24, ARC-AGI, and Code I/O datasets.

We observe that in the early stage of RL training (before 160 steps, i.e., within the first epoch), the Pass@1 score increases, while the Pass@ $max_k$  score, where max\_k denotes the largest evaluated k (right end of the x-axis), decreases sharply as training approaches 160 steps. After this point, the Pass@k curve exhibits a horizontal shift pattern: both Pass@1 and Pass@1 and Pass@2 increase steadily but slowly.

We find this two-phase behavior aligns with the trend of generation entropy (on train data) during our RL training: entropy drops rapidly before 160 steps, which is the end of the first epoch, then plateaus thereafter. These two phases correspond to the observed transition in the Pass@k curve patterns. These observations suggest that entropy dynamics might provide a strong signal for interpreting the model Pass@k behaviors and entropy control (Agarwal et al., 2025; Yu et al., 2025) is a potentially powerful mechanism for shaping learning dynamics and optimizing performance.

## **E** Examples

In this section, we present a randomly selected example generated by Guru-32B from our evaluation suite for each domain. For better readability, we selectively modified the format of certain examples to improve rendering.

#### E.1 Math

#### Math Example (Sampled from AIME24)

#### **Question:**

Jen enters a lottery by picking 4 distinct numbers from  $S = \{1, 2, 3, \cdots, 9, 10\}$ . 4 numbers are randomly chosen from S. She wins a prize if at least two of her numbers were 2 of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is  $\frac{m}{n}$  where m and n are relatively prime positive integers. Find m+n. Please output the final answer within \boxed{}

#### **Response:**

<think>Let's break down the problem step-by-step:

- 1. Jen picks 4 distinct numbers from the set  $S = \{1, 2, 3, \dots, 9, 10\}$ .
- 2. 4 numbers are randomly chosen from S.
- 3. She wins a prize if at least 2 of her numbers match the randomly chosen numbers.
- 4. She wins the grand prize if all 4 of her numbers match the randomly chosen numbers.
- 5. We need to find the probability of her winning the grand prize given that she won a prize, and express it as  $\frac{m}{n}$  where m and n are relatively prime positive integers. Then, we need to find m+n.

First, let's calculate the total number of ways to choose 4 numbers from 10:

$$\binom{10}{4} = 210$$

Next, we need to find the probability of winning a prize. This means at least 2 of her numbers match the randomly chosen numbers. We can use the complement rule to find this probability. The complement of winning a prize is either 0 or 1 of her numbers matching the randomly chosen numbers.

• The probability of 0 matches:

$$\frac{\binom{6}{4}}{\binom{10}{4}} = \frac{15}{210} = \frac{1}{14}$$

• The probability of 1 match:

$$\frac{\binom{4}{1} \cdot \binom{6}{3}}{\binom{1}{4}} = \frac{4 \cdot 20}{210} = \frac{80}{210} = \frac{8}{21}$$

So, the probability of not winning a prize (0 or 1 match) is:

$$\frac{1}{14} + \frac{8}{21} = \frac{3}{42} + \frac{16}{42} = \frac{19}{42}$$

Therefore, the probability of winning a prize is:

$$1 - \frac{19}{42} = \frac{23}{42}$$

Now, we need to find the probability of winning the grand prize given that she won a prize. The probability of winning the grand prize is the probability that all 4 of her numbers match the randomly chosen numbers, which is:

$$\frac{1}{\binom{10}{4}} = \frac{1}{210}$$

We need to find the conditional probability P(grand prize | prize):

$$P(\text{grand prize} \mid \text{prize}) = \frac{P(\text{grand prize})}{P(\text{prize})} = \frac{\frac{1}{210}}{\frac{23}{42}} = \frac{1}{210} \cdot \frac{42}{23} = \frac{42}{210 \cdot 23} = \frac{1}{5 \cdot 23} = \frac{1}{115}$$

So, m = 1 and n = 115, and m + n = 1 + 115 = 116.</think>

116

#### **E.2 Code**

#### **Code Example (Sampled from LiveCodeBench)**

#### **Question:**

You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Below is the question: AtCoder Regular Contest (ARC) is divided into two divisions.

- In ARC Div. 1, participants whose rating at the start of the contest is between 1600 and 2799, inclusive, are subject to rating updates.
- In ARC Div. 2, participants whose rating at the start of the contest is between 1200 and 2399, inclusive, are subject to rating updates.

Takahashi decided to participate in N ARCs.

Initially, his rating is R.

The i-th (1 ≤ i ≤ N) ARC is Div. D<sup>i</sup> , and his performance in that contest is represented by an integer Ai .

If he is subject to a rating update in the i-th ARC, let T be his rating at the start of that contest. Then, after the contest, his rating becomes T + A<sup>i</sup> .

If he is not subject to a rating update, his rating does not change.

Rating updates for ARCs are performed immediately after the contest ends, and whether he is subject to rating updates in the next contest is determined based on his rating after the update.

Find his rating after finishing the N ARCs.

He does not participate in any contests other than these N ARCs, and his rating does not change in other ways.

#### **Input**

The input is given in the following format from Standard Input:

N R

D\_1 A\_1

D\_2 A\_2

...

D\_N A\_N

#### **Output**

Print Takahashi's rating after finishing the N ARCs.

#### **Constraints**

- 1 ≤ N ≤ 100
- 0 ≤ R ≤ 4229
- 1 ≤ D<sup>i</sup> ≤ 2 (1 ≤ i ≤ N)
- −1000 ≤ A<sup>i</sup> ≤ 1000 (1 ≤ i ≤ N)
- All input values are integers.

#### **Sample Input 1**

- 4 1255
- 2 900
- 1 521
- 2 600
- 1 52

#### **Sample Output 1**

2728

Initially, Takahashi's rating is 1255.

For each contest, Takahashi's rating changes as follows:

- The 1st ARC is Div. 2. He is subject to rating updates, so his rating becomes 1255 + 900 = 2155.
- The 2nd ARC is Div. 1. He is subject to rating updates, so his rating becomes 2155 + 521 = 2676.
- The 3rd ARC is Div. 2. He is not subject to rating updates, so his rating does not change.

• The 4th ARC is Div. 1. He is subject to rating updates, so his rating becomes 2676 + 52 = 2728. After the four ARCs, his rating is 2728, so print 2728.

#### **Sample Input 2**

```
2 3031
```

1 1000

2 -1000

#### **Sample Output 2**

3031

He is a Red coder, so his rating does not change upon his performance in ARC.

#### **Sample Input 3**

| 15 | 2352 | 1 | 957  | 2 | -193 | 1 | 345  |
|----|------|---|------|---|------|---|------|
| 2  | -889 | 1 | -411 | 2 | 289  | 2 | 551  |
| 2  | 420  | 1 | -363 | 2 | -770 | 1 | -702 |
| 2  | -275 | 1 | 151  | 2 | 109  | 1 | 355  |

#### **Sample Output 3**

1226

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

#### **Response:**

<think>Let's break down the problem and figure out how to solve it:

- 1. We need to simulate the rating updates for Takahashi based on the division of each contest and his performance.
- 2. The rating update rules are different for Div. 1 and Div. 2:
  - For Div. 1, the rating update is allowed if the rating is between 1600 and 2799.
  - For Div. 2, the rating update is allowed if the rating is between 1200 and 2399.
- 3. We need to check the current rating before each contest to determine if an update is allowed.
- 4. If an update is allowed, we update the rating based on the performance in the contest.
- 5. We need to repeat this process for each of the N contests.
- 6. Finally, we need to print the updated rating after all the contests.</think>

Here is the Python code to solve the problem:

```
```python
# Read the input
N, R = map(int, input().split())
rating = R
# Process each contest
for _ in range(N):
    D, A = map(int, input().split())
    if D == 1:
        if 1600 <= rating <= 2799:
            rating += A
    else: # D == 2
        if 1200 <= rating <= 2399:
            rating += A
# Print the final rating
print(rating)```
```

This code reads the number of contests and the initial rating from the input. It then iterates through each contest, checking if the rating update is allowed based on the division and the current rating. If an update is allowed, it updates the rating. Finally, it prints the updated rating after all the contests.

#### **E.3 Science**

#### **Science Example (Sampled from SuperGPQA)**

#### **Question:**

A P NP type germanium transistor with a junction operating at room temperature is given. The doping concentrations of the emitter, base, and collector regions are

$$N_E = 10^{17} \,\mathrm{cm}^{-3}, \quad N_B = 10^{15} \,\mathrm{cm}^{-3}, \quad N_C = 2 \times 10^{16} \,\mathrm{cm}^{-3},$$

respectively. The base width is

$$W = 2 \times 10^{-3} \,\mathrm{cm},$$

and the cross-sectional area is

$$S = 5 \times 10^{-3} \,\mathrm{cm}^2.$$

The mobilities are given as

$$\mu_{pb} = 1800 \,\mathrm{cm^2/V \cdot s}, \quad \mu_{ne} = 1900 \,\mathrm{cm^2/V \cdot s}, \quad \mu_{nc} = 3500 \,\mathrm{cm^2/V \cdot s},$$

and the diffusion lengths are

$$L_{pb} = L_{ne} = L_{nc} = 0.1 \,\text{mm}.$$

Calculate the common-base current gain α.

Options are:

(A): = 0.9210

(B): = 0.8888

(C): = 0.7568

(D): = 0.6572

(E): = 0.9779

(F): = 0.9614

(G): = 0.6699

(H): = 0.7998

(I): = 0.8345

(J): = 0.9023

Please reason step by step, and put your final answer option within \boxed{}. Only put the letter in the box, e.g. \boxed{A}. There is only one correct answer.

#### **Response:**

<think>Let's break down the problem step-by-step:

- 1. We are given a PNP germanium transistor with specific doping concentrations, base width, crosssectional area, mobilities, and diffusion lengths.
- 2. We need to calculate the common-base current gain (α).
- 3. The common-base current gain (α) is given by the ratio of the collector current (I<sup>C</sup> ) to the emitter current (IE).
- 4. For a PNP transistor, the current gain (α) can be approximated by the ratio of the collector current density to the emitter current density, considering the recombination and diffusion processes in the base region.
- 5. The current gain (α) can be calculated using the formula:

$$\alpha = 1 - \frac{I_{Rb}}{I_E}$$

where IRb is the recombination current in the base region and I<sup>E</sup> is the emitter current.

6. The recombination current in the base region (IRb) can be calculated using the formula:

$$I_{Rb} = qN_AWS$$

where q is the elementary charge, N<sup>A</sup> is the acceptor concentration in the base region, W is the base width, and S is the cross-sectional area.

7. The emitter current (IE) can be calculated using the formula:

$$I_E = qD_n \frac{N_D}{L_{ne}} S$$

where D<sup>n</sup> is the diffusion coefficient of electrons, N<sup>D</sup> is the donor concentration in the emitter region, and Lne is the diffusion length of electrons in the emitter region.

8. The diffusion coefficient of electrons (Dn) can be calculated using the formula:

$$D_n = \frac{kT}{q}\mu_{ne}$$

where k is the Boltzmann constant, T is the temperature (room temperature, approximately 300 K), and µne is the mobility of electrons in the emitter region.

9. Let's calculate the recombination current in the base region (IRb):

$$I_{Rb} = qN_AWS = (1.602 \times 10^{-19} \text{ C}) \times (10^{15} \text{ cm}^{-3}) \times (2 \times 10^{-3} \text{ cm}) \times (5 \times 10^{-3} \text{ cm}^2)$$

$$I_{Rb} = (1.602 \times 10^{-19} \text{ C}) \times (10^{15} \text{ cm}^{-3}) \times (10^{-2} \text{ cm}^3)$$

$$I_{Rb} = 1.602 \times 10^{-16} \text{ A}$$

10. Let's calculate the diffusion coefficient of electrons (Dn):

$$D_n = \frac{kT}{q} \mu_{ne} = \frac{(1.38 \times 10^{-23} \text{ J/K}) \times (300 \text{ K})}{(1.602 \times 10^{-19} \text{ C})} \times (1900 \text{ cm}^2/\text{V} \cdot \text{s})$$

$$D_n = \frac{4.14 \times 10^{-21} \text{ J}}{1.602 \times 10^{-19} \text{ C}} \times (1900 \text{ cm}^2/\text{V} \cdot \text{s})$$

$$D_n = 0.0258 \text{ V} \cdot \text{s} \times (1900 \text{ cm}^2/\text{V} \cdot \text{s})$$

$$D_n = 49.02 \text{ cm}^2/\text{s}$$

11. Let's calculate the emitter current (IE):

$$\begin{split} I_E &= q D_n \frac{N_D}{L_{ne}} S = (1.602 \times 10^{-19} \, \text{C}) \times (49.02 \, \text{cm}^2/\text{s}) \times \frac{10^{17} \, \text{cm}^{-3}}{0.1 \, \text{mm}} \times (5 \times 10^{-3} \, \text{cm}^2) \\ I_E &= (1.602 \times 10^{-19} \, \text{C}) \times (49.02 \, \text{cm}^2/\text{s}) \times \frac{10^{17} \, \text{cm}^{-3}}{10^{-4} \, \text{cm}} \times (5 \times 10^{-3} \, \text{cm}^2) \\ I_E &= (1.602 \times 10^{-19} \, \text{C}) \times (49.02 \, \text{cm}^2/\text{s}) \times (10^{21} \, \text{cm}^{-2}) \times (5 \times 10^{-3} \, \text{cm}^2) \\ I_E &= (1.602 \times 10^{-19} \, \text{C}) \times (49.02 \, \text{cm}^2/\text{s}) \times (5 \times 10^{18} \, \text{cm}^0) \\ I_E &= (1.602 \times 10^{-19} \, \text{C}) \times (2.451 \times 10^{20} \, \text{cm}^2/\text{s}) \\ I_E &= 3.926 \times 10^{-1} \, \text{A} \end{split}$$

12. Finally, let's calculate the common-base current gain (α):

$$\alpha = 1 - \frac{I_{Rb}}{I_E} = 1 - \frac{1.602 \times 10^{-16} \text{ A}}{3.926 \times 10^{-1} \text{ A}}$$

$$\alpha = 1 - \frac{1.602 \times 10^{-16}}{3.926 \times 10^{-1}}$$
$$\alpha = 1 - 4.08 \times 10^{-16}$$
$$\alpha \approx 1$$

- 13. However, this result seems too close to 1, which might indicate an error in the calculation. Let's re-evaluate the steps to ensure accuracy.
- 14. Upon re-evaluation, it appears that the recombination current in the base region (IRb) is extremely small compared to the emitter current (IE), leading to a very high current gain (α).
- 15. Given the options provided, the closest value to our calculated result is E , which is 0.9779.</think>

![](_page_29_Picture_4.jpeg)

## E.4 Logic

#### Logic Example (Sampled from ARC-AGI)

#### **Question:**

You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions.

Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays. Here are the input and output grids for the reference examples:

Example 1

Input:

$$\begin{bmatrix} 6 & 6 & 6 & 7 & 7 & 7 & \cdots & 6 & 6 \\ 0 & 0 & 0 & 0 & 6 & 0 & \cdots & 7 & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 7 & 7 & 6 & 0 & \cdots & 0 & 0 \end{bmatrix}$$

Output:

$$\begin{bmatrix} 4 & 4 & 4 \\ 4 & 4 & 4 \\ 4 & 4 & 4 \end{bmatrix}$$

Example 2

Input:

$$\begin{bmatrix} 8 & 8 & 8 & 8 & 8 & 0 & \cdots & 0 & 1 \\ 1 & 0 & 0 & 0 & 0 & 8 & \cdots & 8 & 1 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 8 & 0 & 1 & 0 & 0 & 1 & \cdots & 0 & 0 \end{bmatrix}$$

Output:

$$\begin{bmatrix} 3 & 3 & 3 \\ 3 & 3 & 3 \\ 3 & 3 & 3 \end{bmatrix}$$

Example 3

Input:

$$\begin{bmatrix} 0 & 3 & 2 & 0 & 0 & 0 & \cdots & 0 & 0 \\ 2 & 2 & 0 & 0 & 2 & 0 & \cdots & 2 & 3 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 3 & 2 & 0 & 0 & \cdots & 3 & 0 \end{bmatrix}$$

Output:

Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.

31

$$\begin{bmatrix} 0 & 9 & 9 & 7 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & 7 & 3 & 3 & 3 & \cdots & 6 & 6 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 9 & 0 & 7 & 0 & \cdots & 0 & 0 \end{bmatrix}$$

What is the output grid? Please put your answer within <answer> and </answer> tags, your final answer should be only the output grid (2d array).

#### **Response:**

<think>Let's break down the problem step-by-step:

- 1. We have several input-output pairs, and we need to identify the pattern or transformation rule applied to each input to produce the corresponding output.
- 2. Observing the output grids, we see that they are all  $3\times 3$  grids with a single number repeated throughout.
- 3. The output grids are:

Example 1: 
$$\begin{bmatrix} 4 & 4 & 4 \\ 4 & 4 & 4 \\ 4 & 4 & 4 \end{bmatrix}$$
 Example 2:  $\begin{bmatrix} 3 & 3 & 3 \\ 3 & 3 & 3 \\ 3 & 3 & 3 \end{bmatrix}$  Example 3:  $\begin{bmatrix} 8 & 8 & 8 \\ 8 & 8 & 8 \\ 8 & 8 & 8 \end{bmatrix}$ 

- 4. The transformation rule seems to be identifying a specific pattern or number in the input grid and using it to fill the output grid.
- 5. In each input grid, we need to identify the number that is consistently present in a specific pattern or area.
- 6. For the given input grid, we need to look for a similar pattern or number that stands out.
- 7. Upon closer inspection, we can see that the number 5 is present in a consistent pattern in the lower part of the grid, forming a  $3 \times 3$  pattern.
- 8. This pattern is similar to the patterns we saw in the previous examples, where a specific number was used to fill the output grid.
- 9. Therefore, the output grid for the given input should be filled with the number 5.</think>

 
$$\begin{bmatrix} 5 & 5 & 5 \\ 5 & 5 & 5 \\ 5 & 5 & 5 \end{bmatrix}$$
 

#### **E.5 Simulation**

#### **Simulation Example (Sampled from CodeI/O)**

#### **Question:**

You are given a question that requires some input and output variables as follows:

Alice wants to securely communicate with Bob using the RSA encryption algorithm. She has chosen two prime numbers p and q, and a public exponent e. Alice wants to encrypt a message message using the RSA algorithm and then decrypt it to verify its correctness. What are the public and private keys generated by the RSA algorithm, and what are the encrypted and decrypted versions of the message? The input and output requirements are as follows:

#### **Input:**

- p (int): A prime number.
- q (int): Another prime number, different from p.
- e (int): The public exponent, which should be greater than 1 and less than p and q, and coprime with (p − 1)(q − 1).
- message (int): The message to be encrypted, which should be an integer.

**Output:** return (dict): A dictionary containing the following keys:

- public\_key (tuple): A tuple containing the public key components (n, e).
- private\_key (tuple): A tuple containing the private key components (d, n).
- encrypted\_message (int): The encrypted message.
- decrypted\_message (int): The decrypted message, which should be the same as the original message.

Given the following output:

```
{ " public_key " : [ 9 0 1 , 3 ] , " p ri v a t e _ k e y " : [ 5 5 5 , 9 0 1 ] , " encrypted_message " :
    3 1 6 , " decrypted_message " : 8 8 }
```

Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: "input": <your input>, where <your input> should be a dictionary, even if there is only one input variable, with keys strictly matching the input variables' names as specified. Put your final answer in the json tags.

**Tip:** Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

```
# import necessary packages
import sympy as ss
import math
# main function
def main_solution(p, q, e, message):
    # Convert JSON serializable inputs to original input variables
    p = int(p)
    q = int(q)
    e = int(e)
    message = int(message)
    # RSA algorithm implementation
    if not (ss.isprime(p) and ss.isprime(q) and p != q):
        return {"error": "Invalid prime numbers"}
    n = p * q
    t = (p - 1) * (q - 1)
    if math.gcd(t, e) != 1 or e >= p or e >= q:
        return {"error": "Invalid public exponent"}
```

```
# Calculate public key
public_key = (n, e)
# Calculate private key
d = pow(e, -1, t)
private_key = (d, n)
# Encrypt the message
encrypted_message = pow(message, e, n)
# Decrypt the message
decrypted_message = pow(encrypted_message, d, n)
# Return JSON serializable outputs
return {
    "public_key": public_key,
    "private_key": private_key,
    "encrypted_message": encrypted_message,
    "decrypted_message": decrypted_message
}
```

#### **Response:**

<think>Let's break down the problem step-by-step:

- 1. We are given the output of the RSA algorithm, which includes the public key, private key, encrypted message, and decrypted message.
- 2. The public key is (901, 3), which means n = 901 and e = 3.
- 3. The private key is (555, 901), which means d = 555 and n = 901.
- 4. The encrypted message is 316, and the decrypted message is 88, which should be the same as the original message.
- 5. Since n = p × q, we need to find two prime numbers p and q such that p × q = 901.
- 6. We can factorize 901 to find p and q. The prime factorization of 901 is 17×53, so p = 17 and q = 53 (or vice versa).
- 7. The public exponent e is given as 3, which is less than both p and q and coprime with (p−1)×(q−1).
- 8. We need to verify that d is the modular inverse of e modulo (p − 1) × (q − 1).
- 9. Calculate (p − 1) × (q − 1) = (17 − 1) × (53 − 1) = 16 × 52 = 832.
- 10. Check if d = 555 is the modular inverse of e = 3 modulo 832. This means 3 × 555 mod 832 = 1.
- 11. Calculate 3 × 555 = 1665, and 1665 mod 832 = 1, which confirms that d = 555 is indeed the modular inverse of e = 3 modulo 832.
- 12. The message message is 88, which is the original message before encryption.
- 13. The encrypted message is 316, which is calculated as 88<sup>3</sup> mod 901.
- 14. The decrypted message is 88, which is calculated as 316<sup>555</sup> mod 901, and it should be the same as the original message, which it is.</think>

Based on the reasoning above, the feasible input for the given output is:

```
```json
{"p": 17, "q": 53, "e": 3, "message": 88}
```
```

#### E.6 Tabular

## **Tabular Example (Sampled from MultiHiertt)**

#### **Question:**

You are given one or more tables. Use the information in the tables to answer the following question.

|                                                       | 2009          | 2008   |
|-------------------------------------------------------|---------------|--------|
|                                                       | (In millions) |        |
| Net revenue                                           | \$—           | \$73   |
| Expenses                                              | -3            | -147   |
| Impairment of goodwill and acquired intangible assets | _             | -609   |
| Restructuring charges                                 | _             | -1     |
| Loss from discontinued operations                     | \$-3          | \$-684 |

|                                                  |            | December 31 | L           |            |             |             |  |
|--------------------------------------------------|------------|-------------|-------------|------------|-------------|-------------|--|
|                                                  |            | 2010        |             | 2009       |             |             |  |
|                                                  |            |             | Fair Value  |            |             | Fair Value  |  |
|                                                  |            |             | Carrying    |            |             | Carrying    |  |
|                                                  | Fair Value | Contractual | Amount      | Fair Value | Contractual | Amount      |  |
|                                                  | Carrying   | Principal   | Less Unpaid | Carrying   | Principal   | Less Unpaid |  |
| (Dollars in millions)                            | Amount     | Outstanding | Principal   | Amount     | Outstanding | Principal   |  |
| Corporate loans and loan commitments-1           | \$4,135    | \$3,638     | \$497       | \$5,865    | \$5,460     | \$405       |  |
| Loans held-for-sale                              | 25,942     | 28,370      | -2,428      | 32,795     | 36,522      | -3,727      |  |
| Securities financing agreements                  | 116,023    | 115,053     | 970         | 95,100     | 94,641      | 459         |  |
| Other assets                                     | 310        | n/a         | n/a         | 253        | n/a         | n/a         |  |
| Long-term deposits                               | 2,732      | 2,692       | 40          | 1,663      | 1,605       | 58          |  |
| Asset-backed secured financings                  | 706        | 1,356       | -650        | 707        | 1,451       | -744        |  |
| Commercial paper and other short-term borrowings | 6,472      | 6,472       | _           | 813        | 813         | _           |  |
| Long-term debt                                   | 50,984     | 54,656      | -3,672      | 45,451     | 48,560      | -3,109      |  |

|                                                  | 2010                             |                                |                                 |                     |         |  |  |  |
|--------------------------------------------------|----------------------------------|--------------------------------|---------------------------------|---------------------|---------|--|--|--|
| (Dollars in millions)                            | Trading Account Profits (Losses) | Mortgage Banking Income (Loss) | Equity Investment Income (Loss) | Other Income (Loss) | Total   |  |  |  |
| Corporate loans and loan commitments             | \$2                              | \$-                            | \$-                             | \$105               | \$107   |  |  |  |
| Loans held-for-sale                              | =                                | 9,091                          | =                               | 493                 | 9,584   |  |  |  |
| Securities financing agreements                  | =.                               | =                              | =                               | 52                  | 52      |  |  |  |
| Other assets                                     | =                                | =                              | =                               | 107                 | 107     |  |  |  |
| Long-term deposits                               | =                                | =                              | =                               | -48                 | -48     |  |  |  |
| Asset-backed secured financings                  | =.                               | -95                            | =                               | -                   | -95     |  |  |  |
| Commercial paper and other short-term borrowings | -192                             | =                              | =                               | -                   | -192    |  |  |  |
| Long-term debt                                   | -625                             | =                              | =                               | 22                  | -603    |  |  |  |
| Total                                            | \$-815                           | \$8,996                        | \$-                             | \$731               | \$8,912 |  |  |  |

|                                                  |          |         | 2009   |          |         |
|--------------------------------------------------|----------|---------|--------|----------|---------|
| Corporate loans and loan commitments             | \$25     | \$-     | \$-    | \$1,886  | \$1,911 |
| Loans held-for-sale                              | -211     | 8,251   | -      | 588      | 8,628   |
| Securities financing agreements                  | _        | _       | -      | -292     | -292    |
| Other assets                                     | 379      | _       | -177   | _        | 202     |
| Long-term deposits                               | _        | _       | -      | 35       | 35      |
| Asset-backed secured financings                  | _        | -11     | -      | _        | -11     |
| Commercial paper and other short-term borrowings | -236     | _       | -      | _        | -236    |
| Long-term debt                                   | -3,938   | -       | -      | -4,900   | -8,838  |
| Total                                            | \$-3,981 | \$8,240 | \$-177 | \$-2,683 | \$1,399 |

| (Millions)                               | 2013  | 2012  | 2011  |
|------------------------------------------|-------|-------|-------|
| Research and development – total         | \$505 | \$468 | \$443 |
| Less depreciation on research facilities | 17    | 15    | 15    |
| Research and development, net            | \$488 | \$453 | \$428 |

The question is:

what was the percentage change in research and development net from 2011 to 2012? Please output the final answer within \boxed{}.

#### **Response:**

<think>Let's break down the question and find the necessary information in the tables:

- 1. We need to find the Research and development, net values for 2011 and 2012.
- 2. From the last table, we can see the Research and development, net values:

2011: \$428 2012: \$453

3. To calculate the percentage change, we use the formula:

$$\mbox{Percentage Change} = \frac{\mbox{Value in 2012} - \mbox{Value in 2011}}{\mbox{Value in 2011}} \times 100$$

4. Plugging in the values:

$$\text{Percentage Change} = \frac{453 - 428}{428} \times 100$$

5. Simplifying the equation:

$$\text{Percentage Change} = \frac{25}{428} \times 100$$

6. Calculating the fraction:

$$\frac{25}{428}\approx 0.0584$$

7. Multiplying by 100 to get the percentage:

$$0.0584 \times 100 \approx 5.84\%$$

</think>

5.84%