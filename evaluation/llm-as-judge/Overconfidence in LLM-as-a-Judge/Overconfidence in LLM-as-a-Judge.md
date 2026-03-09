# Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution

Zailong Tian\*1, Zhuoheng Han\*2, Yanzhe Chen4, Haozhe Xu2, Xi Yang3, Richeng Xuan³†, Hongfeng Wang²‡, Lizi Liao¹§

<sup>1</sup>School of Computing and Information Systems, Singapore Management University

<sup>2</sup>State Key Laboratory for Multimedia Information Processing, Peking University

<sup>3</sup>Beijing Academy of Artificial Intelligence

<sup>4</sup>School of Computing, National University of Singapore

{zltian,lzliao}@smu.edu.sg, {2100017789,xhzgenius}@stu.pku.edu.cn, chenyanzhe@u.nus.edu,
 {rcxuan,yangxi}@baai.ac.cn, wanghf@pku.edu.cn

#### **Abstract**

Large Language Models (LLMs) are widely used as automated judges, where practical value depends on both accuracy and trustworthy, risk-aware judgments. Existing approaches predominantly focus on accuracy, overlooking the necessity of well-calibrated confidence, which is vital for adaptive and reliable evaluation pipelines. In this work, we advocate a shift from accuracy-centric evaluation to confidence-driven, risk-aware LLM-as-a-Judge systems, emphasizing the necessity of well-calibrated confidence for trustworthy and adaptive evaluation. We systematically identify the Overconfidence Phenomenon in current LLM-as-a-Judges, where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment. To quantify this phenomenon, we introduce TH-**Score**, a novel metric measuring confidence-accuracy alignment. Furthermore, we propose LLM-as-a-Fuser, an ensemble framework that transforms LLMs into reliable, risk-aware evaluators. Extensive experiments demonstrate that our approach substantially improves calibration and enables adaptive, confidence-driven evaluation pipelines, achieving superior reliability and accuracy compared to existing baselines.

#### 1 Introduction

The widespread adoption of large language models (LLMs) as automated judges—termed the LLM-as-a-Judge paradigm—has revolutionized the evaluation of AI-generated content by offering scalability and efficiency over traditional human annotation (Zheng et al. 2023). In this paradigm, LLMs act as evaluators, with one common application being pairwise comparisons where the model decides which of two text segments is better based on criteria like quality, relevance, or coherence. However, the practical value of these systems depends not only on accuracy but also on trustworthy, risk-aware judgments that can adapt to real-world deployment scenarios. Existing approaches, such

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

![](_page_0_Figure_13.jpeg)

Figure 1: Well-calibrated judgments align confidence with accuracy, qualifying correct high-confidence predictions ( $\checkmark$   $\rightarrow$  qualified) and disqualifying inaccurate low-confidence ones ( $\times$   $\rightarrow$  unqualified). In contrast, poorly calibrated models mismatch confidence and accuracy, often disqualifying even highly accurate judgments ( $\checkmark$   $\rightarrow$  unqualified), leading to unreliable outcomes overall.

as FairEval (Wang et al. 2023a) and JudgeBench (Tan et al. 2024), predominantly emphasize accuracy, often overlooking the critical role of well-calibrated confidence. This calibration, defined as the alignment between a model's predicted confidence and its actual correctness, is essential for building adaptive evaluation pipelines. For instance, well-calibrated confidence allows high-confidence outputs to be automatically accepted, minimizing manual intervention, while low-confidence cases can be flagged for human review (Li et al. 2024). In this work, we advocate a fundamental shift from accuracy-centric evaluations to confidence-driven, risk-aware LLM-as-a-Judge systems, prioritizing calibration to ensure reliable and trustworthy assessments.

Despite these potential benefits, current LLM-as-a-Judge systems suffer from a pervasive Overconfidence Phenomenon, where predicted confidence levels significantly overstate actual correctness (Mielke et al. 2022; Zhou, Jurafsky, and Hashimoto 2023), thereby undermining reliability in practical applications. Through systematic analysis, we observe that state-of-the-art LLMs exhibit this issue promi-

<sup>\*</sup>These authors contributed equally.

<sup>&</sup>lt;sup>†</sup>Corresponding author.

<sup>‡</sup>Corresponding author.

<sup>§</sup>Corresponding author.

nently, leading to inflated confidence scores that do not reflect true performance (Zhao et al. 2021). This misalignment results in substantial risks: overconfident models may propagate erroneous judgments without detection, eroding the efficiency gains of automated evaluation, while also complicating downstream decision-making in pipelines (Gu et al. 2024). Furthermore, existing benchmarks and metrics exacerbate the problem by focusing on aggregate accuracy without addressing confidence alignment, introducing biases such as response length or model familiarity that distort calibration assessments (Chen et al. 2024; Zheng et al. 2023; Wang et al. 2023a). Consequently, the lack of calibrationaware tools limits the deployment of LLMs as dependable evaluators in high-stakes environments.

To address these challenges, we introduce TH-Score, a novel metric that quantifies confidence-accuracy alignment by focusing on critical high- and low-confidence intervals, where practical decisions hinge. Unlike traditional metrics like accuracy or Expected Calibration Error (ECE)—which ignore confidence or overlook key thresholds—TH-Score balances accuracy within these intervals against their coverage, rewarding aligned successes (e.g., high-confidence correct predictions) while penalizing mismatches like overconfident errors. This makes TH-Score a principled tool for detecting the Overconfidence Phenomenon under LLM-asa-Judge scenario, highlighting cases where high confidence fails to match actual correctness.

Furthermore, we propose LLM-as-a-Fuser, an ensemble framework that leverages a dedicated "fuser" LLM to synthesize judgments and critiques from multiple models, transforming LLMs into reliable, risk-aware evaluators. By integrating diverse perspectives, LLM-as-a-Fuser significantly enhances calibration. Extensive experiments on a widelyused benchmark demonstrate that our approach achieves superior calibration, reliability, and overall accuracy compared to existing baselines, paving the way for more trustworthy LLM-as-a-Judge systems in practical settings.

In a nutshell, our contributions are threefold:

- Overconfidence Phenomenon: We systematically identify and characterize the overconfidence in LLM-as-a-Judge, where confidence overstates correctness, limiting risk-aware evaluation.
- Metric Innovation: We introduce TH-Score, a novel metric quantifying confidence-accuracy alignment for trustworthy LLM judgments.
- Framework Advancement: We propose LLM-as-a-Fuser, an ensemble approach that enhances calibration, enabling adaptive, confidence-driven pipelines with superior reliability and accuracy.

## 2 Related Work

### LLM-as-a-Judge

LLMs are increasingly used as automated evaluators for text quality. (Zheng et al. 2023) showed GPT-4 aligns with human judgments in pairwise comparisons, but proprietary APIs limit reproducibility. PandaLM (Wang et al. 2023b) introduced a 7B-parameter local evaluator with 94% agreement with ChatGPT, supporting offline use. JudgeLM (Zhu, Wang, and Wang 2025) and Agent-as-a-Judge (Zhuge et al. 2024) use modular frameworks with memory and planning, cutting DevAI evaluation costs by 97%. However, alignment between model confidence and accuracy is often ignored, causing inconsistent judgments. Meta's self-rewarding models (Wu et al. 2024) generate and evaluate outputs iteratively, but calibration needs further study.

As LLM-based judges gain traction for evaluating and enhancing LLMs, various benchmarks have emerged to gauge their effectiveness. Prior works like LLMEval (Lin and Chen 2023), MTBench, and FairEval primarily assess how well LLM-based judges align with subjective human preferences, often emphasizing stylistic differences over factual and logical accuracy. Similarly, LLMBar (Zeng et al. 2024) evaluates judges based on their ability to follow instructions, using response pairs with clear ground truth labels tied to instruction adherence. In contrast, JudgeBench offers a novel benchmark specifically designed to test LLM-based judges' reasoning capabilities. It features 350 challenging response pairs across knowledge, reasoning, math, and coding domains, each containing one objectively correct and one subtly incorrect response, prioritizing factual and logical correctness over subjective or stylistic factors.

## Calibration in LLMs

Accurate calibration, aligning a model's confidence with its accuracy, is crucial for reliable LLM applications. Traditional methods like temperature scaling (Guo et al. 2017) adjust confidence with a single scalar but are less effective for large models, while Bayesian methods are computationally infeasible. Recent approaches, such as the Thermometer method (Shen et al. 2024), train auxiliary models for recalibration, achieving top uncertainty quantification across 12 benchmarks, and SPACE (Yi et al. 2024) uses lightweight linear layers for dynamic confidence adjustment. However, most techniques focus on single models, missing multi-model aggregation benefits, and Collaborative Calibration (Yang et al. 2024) reduces overconfidence via multi-agent deliberation but requires significant resources. Current research lacks focus on calibration's impact on downstream tasks like data generation, where confidence filtering affects output quality, warranting further exploration.

# Uncertainty Quantification and Reward Modeling

Uncertainty-aware frameworks bridge calibration and practical applications. Generating with Confidence (Lin, Trivedi, and Sun 2023) combines Monte Carlo dropout and response length analysis to filter low-confidence outputs, demonstrating that well-calibrated models yield higher-quality synthetic data. Inference-Time Scaling (Liu et al. 2025) dynamically aligns reward models with human preferences, indirectly improving calibration through gradient-free optimization. However, these approaches often assume static datasets, failing to address the iterative nature of LLM-asa-Judge workflows. Benchmarking LLMs via Uncertainty Quantification (Ye et al. 2024) reveals that calibration degrades under distribution shifts (e.g., domain-specific tasks), underscoring the need for adaptive methods.

#### 3 Overconfidence in LLM-as-a-Judge

In the LLM-as-a-Judge paradigm, models are typically required to select the superior option from pairwise samples. However, the reliability of model predictions warrants careful examination, particularly regarding the Overconfidence Phenomenon—a tendency for language models to display predicted confidence levels that significantly exceed their actual accuracy, resulting in calibration gaps that undermine reliability. Underconfident models tend to underestimate their own accuracy, while overconfident ones overestimate their judgment correctness. Such biases introduce noisy signals that can adversely affect the performance of downstream tasks (e.g., reward modeling). Particularly in unsupervised or weakly-supervised scenarios, developing a well-calibrated model where judgment capability aligns with confidence becomes crucial. By acquiring confidence of model judgments, we can not only filter out low-accuracy predictions but also effectively identify high-accuracy decisions, thereby enhancing the overall system reliability.

#### How to measure confidence in LLMs?

We employed three methods for calculating confidence: Self-Confidence (SC), Multiple-Prompting (MP) confidence, and Logprob confidence.

**SC setting**: We prompt the model to output both the result and its confidence. Model's temperature is set to 0 to ensure the reproducibility of the setting.

**MP setting**: We adopt a method similar to SimpleQA (Wei et al. 2024), but reduce the number of requests from 100 to 10 for efficiency, while keeping the temperature at 0.7. The final reply is determined by majority voting, and the confidence is the count of the chosen response over 10.

**LogP Setting.** In this setting, confidence scores are derived from softmax-normalized logits for the final output tokens (e.g., 'A' or 'B'). For a binary choice task with options A and B, and corresponding logits  $l_A$  and  $l_B$ , we first compute the softmax probabilities:

$$p(A) = \frac{e^{l_A}}{e^{l_A} + e^{l_B}}, \quad p(B) = \frac{e^{l_B}}{e^{l_A} + e^{l_B}}.$$

Confidence is then defined as the maximum probability:

Confidence<sub>logp</sub> = 
$$\max(p(A), p(B))$$
.

Temperature is set to 0 to ensure deterministic outputs.

#### **Existing Calibration Evaluation Metrics**

To conduct an objective and comprehensive evaluation of the calibration of LLM-as-a-Judge, we applied five existing metrics—Expected Calibration Error (ECE), Adaptive Calibration Error (ACE), Maximum Calibration Error (MCE), Brier Score, and Negative Log Likelihood (NLL)—to the three confidence calculation methods described earlier in this section. Table 2 provides a brief introduction to the calculation methods and characteristics of these metrics.

#### **Initial Results**

We systematically evaluate 14 cutting-edge models on the JudgeBench benchmark, with complete results presented in Table1. These include open-source models such as Qwen3-235B-A22B (Qwen Team 2025), DeepSeek-R1-0528 (DeepSeek-AI et al. 2025), R1-Distill-Qwen, R1-Distill-Llama, DeepSeek-V3-0324 (DeepSeek-AI et al. 2024), Llama-3.3-70B (Dubey et al. 2024), and Mistral-Nemo (Team 2024), as well as proprietary models like OpenAI-o3-mini (OpenAI 2025b), Claude-Sonnet-4 (Anthropic 2025), GPT-4.1 (OpenAI 2025a), GPT-4.1-mini, Gemini-2.5-Flash (DeepMind 2025), GPT-40 (Ahmad et al. 2024), and GPT-4.1-nano, with special attention to scaled variants (e.g., GPT-4.1-mini/nano and o3-mini). Our analysis focuses on the impact of model scales on accuracy and confidence calibration (ECE/ACE), further illustrated by reliability plots in Figure3 showing calibration gaps in high-confidence (red) and low-confidence (green) regions for selected models.

## **Empirical Observations of Overconfidence**

Figure 3 reveals significant calibration gaps across the evaluated models, with most exhibiting the Overconfidence Phenomenon in high-confidence regions (highlighted in green). This pattern undermines the reliability of the LLM-as-a-Judge, as models like DeepSeek-R1-0528 and GPT-40 cluster predictions at high confidence levels (90-100%) but achieve accuracies well below the ideal calibration line.

This overconfidence impacts downstream tasks, such as data filtering, by retaining flawed outputs (false positives) or discarding valuable ones (false negatives), thereby degrading overall performance. For instance, high ECE values in GPT-40 (39.25 in SC, 47.09 in MP, 45.05 in LogP), Mistral-Nemo (74.22 in SC, 68.89 in MP, 64.63 in LogP), and GPT-4.1-nano (57.03 in SC, 67.43 in MP, 66.05 in LogP) necessitate increased human oversight to mitigate risks, diminishing the efficiency of automated judging processes (see Table 1 and the corresponding results in the Appendix).

# 4 TH-Score: A New Metric for LLM-as-a-Judge Calibration Evaluation

While existing calibration metrics such as ECE and Brier Score offer valuable insights into model reliability, they often overlook practical aspects like high-confidence regions essential for real-world applications in LLM-as-a-Judge scenarios. To address these limitations and better align confidence with accuracy in targeted intervals, we introduce TH-Score, a novel metric designed to improve evaluation for data filtering and quality assessment tasks.

#### **Definition**

The TH-Score focuses on two key confidence intervals relevant to practical applications:

- High-Confidence Data ( $100 \epsilon$ , 100): These predictions are highly reliable, and selecting them can enhance the overall dataset quality.  $\epsilon$  is a hyperparameter defining the high-confidence threshold.
- Low-Confidence Data (0, ε): These predictions are uncertain, and discarding them can reduce noise and enhance data quality. ε is also a hyperparameter that determines a threshold for what constitutes low confidence.

![](_page_3_Figure_0.jpeg)

Figure 2: Visualization of the three confidence calculation settings: Self-Confidence (SC), Multiple-Prompting (MP), and Logprob (Logp), using data with ID 122 from JudgeBench as an example.

![](_page_3_Figure_2.jpeg)

Figure 3: Illustration of calibration gaps in high-confidence regions (red) and low-confidence regions (green) where models show significant accuracy-confidence discrepancy.

This metric quantifies model performance by jointly considering the accuracy of predictions within specified confidence intervals and the coverage of these intervals, facilitating effective data filtering and quality evaluation. The TH-Score is formally defined as:

$$\mbox{TH-Score} = (e^{(\mbox{accuracy} - 0.5)} - 1) \times \mbox{percentage},$$
 where:

• e denotes the base of the natural logarithm, serving as a scaling hyperparameter

- accuracy represents the prediction accuracy specifically for samples falling within the target confidence intervals
- percentage indicates the proportion of total samples that fall within these confidence intervals

This formulation ensures that the TH-Score increases with both higher accuracy and a larger proportion of highconfidence or low-confidence data, providing a balanced measure of model reliability in practical usage scenarios.

| Model              | Acc ↑              | ECE ↓ | ACE ↓ | Brier Score ↓ | MCE ↓ | NLL ↓ | TH Score ↑ |  |  |  |
|--------------------|--------------------|-------|-------|---------------|-------|-------|------------|--|--|--|
| Open Source Models |                    |       |       |               |       |       |            |  |  |  |
| Qwen3-235B-A22B    | 77.43              | 11.78 | 12.16 | 0.16          | 63.50 | 0.52  | 17.52      |  |  |  |
| DeepSeek-R1-0528   | 76.86              | 12.07 | 11.39 | 0.13          | 40.00 | 0.42  | 14.59      |  |  |  |
| R1-Distill-Qwen    | 65.71              | 27.26 | 27.10 | 0.29          | 69.00 | 0.91  | 8.16       |  |  |  |
| R1-Distill-Llama   | 59.71              | 31.02 | 30.89 | 0.31          | 65.00 | 1.31  | 7.01       |  |  |  |
| DeepSeek-V3-0324   | 49.71              | 36.21 | 36.35 | 0.37          | 50.24 | 1.03  | 2.46       |  |  |  |
| Llama-3.3-70B      | 42.00              | 47.37 | 46.78 | 0.45          | 63.78 | 2.75  | 0.80       |  |  |  |
| Mistral-Nemo       | 20.29              | 74.22 | 74.21 | 0.71          | 80.00 | 3.01  | -11.64     |  |  |  |
|                    | Proprietary Models |       |       |               |       |       |            |  |  |  |
| OpenAI-o3-mini     | 74.29              | 15.97 | 17.20 | 0.20          | 60.00 | 0.62  | 12.83      |  |  |  |
| Claude-Sonnet-4    | 64.29              | 17.98 | 18.00 | 0.24          | 45.00 | 0.69  | 9.89       |  |  |  |
| GPT-4.1            | 63.14              | 26.39 | 26.86 | 0.29          | 55.00 | 0.85  | 7.55       |  |  |  |
| GPT-4.1-mini       | 55.71              | 32.70 | 32.79 | 0.35          | 44.21 | 1.00  | 3.29       |  |  |  |
| Gemini-2.5-Flash   | 39.43              | 30.49 | 30.41 | 0.26          | 56.11 | 0.78  | 2.71       |  |  |  |
| GPT-4o             | 49.71              | 39.25 | 39.28 | 0.40          | 57.50 | 1.15  | 1.57       |  |  |  |
| GPT-4.1-nano       | 26.86              | 57.03 | 57.08 | 0.52          | 72.50 | 1.38  | -0.07      |  |  |  |

Table 1: Model performance under Self-Confidence (SC) setting, grouped by model type (Open Source vs. Proprietary). Arrows indicate optimization direction: ↑ higher is better, ↓ lower is better. Best results are bolded.

# Impact of ϵ on TH-Score Performance

Table 3 presents the TH-Score results for various models under different values of ϵ. The table also includes accuracy rates within specified intervals and the proportion of interval data relative to the total dataset. When ϵ = 0.05, most models, except the most powerful ones, exhibit limited calibration capability. Consequently, most models either have minimal data within this interval or demonstrate significantly reduced accuracy, highlighting the stringent calibration demands of such a small ϵ and underscoring the challenges in achieving reliable confidence alignment at fine-grained thresholds. At ϵ = 0.1, the value used in our primary experiments, most models align well with this calibration threshold, resulting in strong discriminative power. With the exception of weaker models like Mistral-Nemo, the majority of models have substantial data within this interval, enabling effective comparison of their calibration performance. This observation suggests that an effective approach for selecting ϵ is to choose a value where most models contribute significant data to the interval.

However, when ϵ is increased to 0.15, while data coverage improves,the discriminative power diminishes. The advantages of high-performing models, such as DeepSeek-R1- 0528, become less pronounced due to the relaxed performance requirements associated with a larger ϵ. Thus, selecting an appropriate ϵ requires balancing data coverage with discriminative power, avoiding excessively large values that dilute model differentiation.

# 5 LLM-as-a-Fuser

As shown in the section on overconfidence in LLM-as-ajudge, while *LLM-as-a-judge* offers a promising approach to evaluating model outputs, its calibration issues—such as overconfidence in unreliable judgments—limit its reliability. Traditional aggregation methods (e.g., majority voting) compound this problem by ignoring nuanced critiques from individual models and focusing only on final decisions. To address these limitations, we propose LLM-as-a-Fuser framework, which redefines the LLM's role from a passive judge to an active *fuser*. By synthesizing model decisions and their rationales, the fuser enables evidence-aware aggregation, improving both calibration and robustness.

## Methodology

The fuser LLM ingests decisions and critiques from an ensemble of models, analyzing their reasoning. Unlike traditional methods, this approach grounds the final decision in comprehensive evidence, as illustrated in Figure 4.

#### Baseline Methods

To evaluate the performance of LLM-as-a-Fuser, we compare it against several baseline aggregation methods that combine predictions from multiple models. These methods vary in how they weight or process model predictions and confidences but do not incorporate model critiques, relying solely on final decisions and, where applicable, associated confidence scores. The baseline methods are:

- Majority Voting: Selects the most frequent label across models, with equal votes. Ties are broken by the highest confidence score.
- Confidence-Weighted Voting: Weights votes by model confidence scores, selecting the label with the highest total. Ties use the maximum confidence.
- Square-Root Confidence-Weighted Voting: Applies square-root transformation to confidences, summing them to select the label with the highest total.

| Metric          | Formula                                                                         | Key Characteristics                                                                                                                                                                          |
|-----------------|---------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ECE             | $\sum_{i=1}^{M} \frac{n_i}{N}  \operatorname{acc}(i) - \operatorname{conf}(i) $ | <ul><li>Uses fixed-width bins.</li><li>Ignores critical high-confidence regions.</li></ul>                                                                                                   |
| ACE             | Variant of ECE with adaptive binning                                            | <ul><li>Computationally more expensive.</li><li>Lacks focus on specific confidence intervals.</li></ul>                                                                                      |
| Brier Score     | $\frac{1}{N}\sum_{i=1}^{N}(p_i-o_i)^2$                                          | <ul> <li>Less interpretable; conflates calibration and refinement.</li> <li>Does not isolate miscalibration in specific regions.</li> </ul>                                                  |
| MCE             | $\max_{i \in \{1,\dots,M\}}  \operatorname{acc}(i) - \operatorname{conf}(i) $   | <ul><li>Overly sensitive to outliers and single bins.</li><li>Not representative of overall calibration.</li></ul>                                                                           |
| NLL             | $-\frac{1}{N}\sum[y_i\log(p_i)+(1-y_i)\log(1-p_i)]$                             | <ul><li>Unbounded range makes it hard to compare.</li><li>Very sensitive to overconfident errors.</li></ul>                                                                                  |
| TH-Score (Ours) | $(e^{(acc-0.5)}-1)\times\%$                                                     | ✓ Focuses on high-confidence regions.<br>✓ Uses an adaptive evaluation threshold $\epsilon$ .<br>✓ Explicitly balances accuracy and coverage.<br>✓ Provides an interpretable, bounded score. |

Table 2: A comparison of calibration metrics. Our proposed TH-Score is designed to evaluate practical reliability in high-confidence regions, addressing the limitations of standard approaches. Notation: % = interval coverage;  $\epsilon$  = adjustable threshold (default=0.1); acc = accuracy within  $\epsilon$  ranges;  $o_i$  = ground truth;  $p_i$  = predicted probability.

|                  | $\epsilon = 0.05$ |              |       | $\epsilon = 0.10$ |                         |        | $\epsilon = 0.15$ |                         |        |
|------------------|-------------------|--------------|-------|-------------------|-------------------------|--------|-------------------|-------------------------|--------|
| Model            | Acc ↑             | Percentage ↑ | TH↑   | Acc ↑             | Percentage <sup>†</sup> | TH↑    | Acc ↑             | Percentage <sup>†</sup> | TH↑    |
| DeepSeek-R1-0528 | 1.0000            | 3.71         | 12.14 | 0.9075            | 64.86                   | 17.52  | 0.8672            | 77.43                   | 18.38  |
| GPT-4.1          | 1.0000            | 1.14         | 0.37  | 0.8346            | 38.00                   | 7.55   | 0.6923            | 74.29                   | 7.88   |
| GPT-4.1-mini     | 0.0000            | 0.00         | 0.00  | 0.8333            | 13.71                   | 2.71   | 0.6250            | 68.57                   | 4.57   |
| Qwen3-235B-A22B  | 0.8846            | 7.43         | 1.93  | 0.8773            | 62.86                   | 14.59  | 0.8376            | 77.43                   | 15.73  |
| GPT-4o           | 0.0000            | 0.00         | 0.00  | 0.7067            | 21.43                   | 2.46   | 0.5375            | 72.29                   | 1.38   |
| DeepSeek-V3-0324 | 1.0000            | 0.57         | 0.19  | 0.7879            | 9.43                    | 1.57   | 0.6516            | 44.29                   | 3.63   |
| R1-Distill-Llama | 0.8605            | 24.57        | 5.42  | 0.7101            | 59.14                   | 7.01   | 0.6514            | 81.14                   | 6.72   |
| Llama-3.3-70B    | 0.5195            | 22.00        | 0.22  | 0.5364            | 43.14                   | 0.80   | 0.4846            | 74.29                   | -0.57  |
| GPT-4.1-nano     | 0.0000            | 0.00         | 0.00  | 0.4286            | 2.00                    | -0.07  | 0.4583            | 6.86                    | -0.14  |
| Mistral-Nemo     | 0.3556            | 12.86        | -0.86 | 0.1961            | 88.86                   | -11.64 | 0.2048            | 94.86                   | -12.12 |

Table 3: Model performance under different  $\epsilon$  values under SC setting.

• Entropy-Weighted Voting: Weights confidences by inverse entropy, selecting the label with the highest weighted confidence sum.

These baseline methods serve as standard approaches for aggregating model predictions and provide a robust comparison for evaluating the effectiveness of LLM-as-a-Fuser, which leverages model critiques in addition to final decisions. Each method was implemented with careful consideration of model calibration and tie-breaking mechanisms to ensure fair and consistent comparisons.

#### **Experimental Results**

Table 4 presents the performance of LLM-as-a-Fuser and baseline methods on JudgeBench, compared to individual model results under the Self-Confidence (SC) setting in Table 1. LLM-as-a-Fuser with Qwen3-235B-A22B achieves the highest accuracy (86.29%) and best calibration (ECE of 6.42%), outperforming baselines like Entropy Weighted

Voting (81.71% Acc, 8.48% ECE) and showing substantial gains over SC models (e.g., +8.86% Acc and -5.36% ECE relative to its SC counterpart at 77.43% Acc, 11.78% ECE). Notably, models like Mistral-Nemo exhibit the most dramatic improvements (+47.14% Acc, -53.73% ECE), followed by Gemini-2.5-Flash (+38.57% Acc) and GPT-4.1nano (+30.85% Acc), indicating that weaker SC performers benefit significantly from critique integration in the fuser framework. Baseline aggregation methods also surpass individual SC performances; for instance, Entropy Weighted Voting exceeds the top SC model by ~4.28% in accuracy and 3.3% in ECE, while other baselines (~80% Acc) outperform most SC models. Other fusers vary, with GPT-40 the weakest (49.71% Acc, 44.07% ECE). Critique integration drives LLM-as-a-Fuser's superior accuracy and calibration, and ensemble methods generally yield better results than isolated self-confidence evaluations.

| Method/Fuser Model   | Acc ↑          | ECE ↓          | ACE ↓          | Brier Score ↓ | NLL ↓        | TH ↑           |
|----------------------|----------------|----------------|----------------|---------------|--------------|----------------|
| Entropy W. Voting    | 81.71          | 8.48           | 9.4            | 0.15          | 0.53         | 13.08          |
| Conf. W. Voting      | 80.00          | 10.43          | 13.0           | 0.16          | 0.50         | 12.64          |
| Majority Voting      | 80.00          | 10.77          | 12.9           | 0.16          | 0.50         | 12.58          |
| Sqrt Conf. W. Voting | 80.00          | 10.43          | 13.0           | 0.16          | 0.50         | 12.64          |
|                      |                |                | LLM-as-a-Fuser |               |              |                |
| Qwen3-235B-A22B      | 86.29 (+8.86)  | 6.42 (-5.36)   | 8.9 (-3.3)     | 0.12 (-0.04)  | 0.39 (-0.13) | 17.38 (-0.14)  |
| OpenAI-o3-mini       | 84.86 (+10.57) | 8.16 (-7.81)   | 9.1 (-8.1)     | 0.13 (-0.07)  | 0.48 (-0.14) | 16.39 (+3.56)  |
| GPT-4.1-mini         | 83.14 (+27.43) | 10.24 (-22.46) | 11.8 (-21.0)   | 0.14 (-0.21)  | 0.47 (-0.53) | 16.37 (+13.08) |
| Claude-Sonnet-4      | 81.71 (+17.42) | 9.06 (-8.92)   | 10.3 (-7.7)    | 0.15 (-0.09)  | 0.54 (-0.15) | 12.31 (+2.42)  |
| GPT-4.1              | 80.00 (+16.86) | 14.92 (-11.47) | 15.6 (-11.2)   | 0.18 (-0.11)  | 0.69 (-0.16) | 16.04 (+8.49)  |
| DeepSeek-V3-0324     | 78.86 (+29.15) | 12.71 (-23.50) | 13.5 (-22.9)   | 0.17 (-0.20)  | 0.54 (-0.49) | 11.96 (+9.50)  |
| Gemini-2.5-Flash     | 78.00 (+38.57) | 15.72 (-14.77) | 16.0 (-14.4)   | 0.19 (-0.07)  | 0.67 (-0.11) | 13.49 (+10.78) |
| Deepseek-R1-0528     | 68.57 (-8.29)  | 21.44 (+9.37)  | 22.3 (+11.0)   | 0.24 (+0.11)  | 1.65 (+1.23) | 10.34 (-4.25)  |
| Mistral-Nemo         | 67.43 (+47.14) | 20.49 (-53.73) | 20.5 (-53.7)   | 0.22 (-0.49)  | 0.95 (-2.06) | 13.53 (+25.17) |
| Llama-3.3-70B        | 62.86 (+20.86) | 24.38 (-22.99) | 24.8 (-22.0)   | 0.27 (-0.18)  | 1.39 (-1.36) | 9.80 (+9.00)   |
| GPT-4.1-nano         | 57.71 (+30.85) | 37.25 (-19.78) | 37.4 (-19.7)   | 0.38 (-0.14)  | 2.36 (+0.98) | 5.48 (+5.55)   |
| GPT-4o               | 49.71 (+0.00)  | 44.07 (+4.82)  | 44.3 (+5.0)    | 0.44 (+0.04)  | 2.06 (+0.91) | 0.72 (-0.85)   |

Table 4: Performance comparison of baseline aggregation methods and LLM-as-a-Fuser models. Values in parentheses represent changes compared to the original Self-Confidence (SC) setting (Table 1).

![](_page_6_Figure_2.jpeg)

Figure 4: Illustration of the LLM-as-a-Fuser framework, aggregating decisions and critiques via the fuser model.

## Disagreement with Majority Voting

We analyzed cases where LLM-as-a-Fuser's decisions diverged from majority voting, as visualized in Figure 5. Qwen3-235B-A22B, the top-performing fuser (Table 4), has the most correct disagreements (34) and few incorrect ones (12), reflecting its effective use of model critiques. In contrast, GPT-4o has the most incorrect disagreements (112) and fewest correct ones (6), indicating poor integration. DeepSeek-V3-0324 shows the fewest total disagreements (30), suggesting conservative decision-making, while Llama-3.3-70B has few correct disagreements (11), aligning with its lower accuracy (62.86%). These results highlight the fuser's ability to leverage critiques for accurate decisions, with Qwen3-235B-A22B's performance underscoring the framework's strength.

![](_page_6_Figure_7.jpeg)

Figure 5: Number of correct (positive bars) and incorrect (negative bars) disagreements between majority voting and the LLM-as-a-Fuser across different models.

# 6 Conclusion

This work diagnoses the Overconfidence Phenomenon in LLM-as-a-Judge, where confidence exceeds accuracy, undermining reliability in tasks like data filtering. We introduce TH-Score to quantify calibration in key intervals, offering a practical alternative to metrics like ECE, and propose LLM-as-a-Fuser, an ensemble framework that integrates critiques for enhanced calibration—yielding up to +47.14% accuracy and -53.73% ECE improvements on JudgeBench.

These innovations enable confidence-driven, risk-aware evaluations, thereby reducing human oversight while boosting trustworthiness in practical applications. Future directions include investigating the root causes of the overconfidence phenomenon and developing more scalable solutions.

# References

Ahmad, L.; et al. 2024. GPT-4o System Card. arXiv:2410.21276.

Anthropic. 2025. Claude Sonnet 4.

Chen, G. H.; Chen, S.; Liu, Z.; Jiang, F.; and Wang, B. 2024. Humans or LLMs as the Judge? A Study on Judgement Bias. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 8301–8327.

DeepMind, G. 2025. Gemini Flash.

DeepSeek-AI; Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; Zhang, X.; Yu, X.; Wu, Y.; Wu, Z.; Gou, Z.; Shao, Z.; Li, Z.; Gao, Z.; Liu, A.; Xue, B.; Wang, B.; Wu, B.; Feng, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li, G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Ding, H.; Xin, H.; Gao, H.; Qu, H.; Li, H.; Guo, J.; Li, J.; Wang, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Cai, J.; Ni, J.; Liang, J.; Chen, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Zhao, L.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Wang, M.; Li, M.; Tian, N.; Huang, P.; Zhang, P.; Wang, Q.; Chen, Q.; Du, Q.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Chen, R.; Jin, R.; Chen, R.; Lu, S.; Zhou, S.; Chen, S.; Ye, S.; Wang, S.; Yu, S.; Zhou, S.; Pan, S.; Li, S.; and et al. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

DeepSeek-AI; Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; Dai, D.; Guo, D.; Yang, D.; Chen, D.; Ji, D.; Li, E.; Lin, F.; Dai, F.; Luo, F.; Hao, G.; Chen, G.; Li, G.; Zhang, H.; Bao, H.; Xu, H.; Wang, H.; Zhang, H.; Ding, H.; Xin, H.; Gao, H.; Li, H.; Qu, H.; Cai, J.; Liang, J.; Guo, J.; Ni, J.; Li, J.; Wang, J.; Chen, J.; Chen, J.; Yuan, J.; Qiu, J.; Li, J.; Song, J.; Dong, K.; Hu, K.; Gao, K.; Guan, K.; Huang, K.; Yu, K.; Wang, L.; Zhang, L.; Xu, L.; Xia, L.; Zhao, L.; Wang, L.; Zhang, L.; Li, M.; Wang, M.; Zhang, M.; Zhang, M.; Tang, M.; Li, M.; Tian, N.; Huang, P.; Wang, P.; Zhang, P.; Wang, Q.; Zhu, Q.; Chen, Q.; Du, Q.; Chen, R.; Jin, R.; Ge, R.; Zhang, R.; Pan, R.; Wang, R.; Xu, R.; Zhang, R.; Chen, R.; Li, S.; Lu, S.; Zhou, S.; Chen, S.; Wu, S.; Ye, S.; Ye, S.; Ma, S.; Wang, S.; Zhou, S.; Yu, S.; Zhou, S.; Pan, S.; Wang, T.; Yun, T.; Pei, T.; Sun, T.; Xiao, W.; Zeng, W.; and et al. 2024. DeepSeek-V3 Technical Report. arXiv:2412.19437.

Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; et al. 2024. The llama 3 herd of models. *arXiv e-prints*, arXiv–2407.

- Gu, J.; Jiang, X.; Shi, Z.; Tan, H.; Zhai, X.; Xu, C.; Li, W.; Shen, Y.; Ma, S.; Liu, H.; et al. 2024. A Survey on LLM-asa-Judge. *arXiv e-prints*, arXiv–2411.
- Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On calibration of modern neural networks. In *International conference on machine learning*, 1321–1330. PMLR.
- Li, H.; Dong, Q.; Chen, J.; Su, H.; Zhou, Y.; Ai, Q.; Ye, Z.; and Liu, Y. 2024. Llms-as-judges: a comprehensive survey on llm-based evaluation methods. *arXiv preprint arXiv:2412.05579*.
- Lin, Y.-T.; and Chen, Y.-N. 2023. LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models. arXiv:2305.13711.
- Lin, Z.; Trivedi, S.; and Sun, J. 2023. Generating with confidence: Uncertainty quantification for black-box large language models. *arXiv preprint arXiv:2305.19187*.
- Liu, Z.; Wang, P.; Xu, R.; Ma, S.; Ruan, C.; Li, P.; Liu, Y.; and Wu, Y. 2025. Inference-Time Scaling for Generalist Reward Modeling. *arXiv preprint arXiv:2504.02495*.
- Mielke, S. J.; Szlam, A.; Dinan, E.; and Boureau, Y.- L. 2022. Reducing conversational agents' overconfidence through linguistic calibration. *Transactions of the Association for Computational Linguistics*, 10: 857–872.
- OpenAI. 2025a. Introducing GPT-4.1 in the API.
- OpenAI. 2025b. Introducing OpenAI o3 and o4-mini.
- Qwen Team, A. 2025. Qwen3: Think Deeper, Act Faster. https://qwenlm.github.io/blog/qwen3/.
- Shen, M.; Das, S.; Greenewald, K.; Sattigeri, P.; Wornell, G.; and Ghosh, S. 2024. Thermometer: Towards universal calibration for large language models. *arXiv preprint arXiv:2403.08819*.
- Tan, S.; Zhuang, S.; Montgomery, K.; Tang, W. Y.; Cuadron, A.; Wang, C.; Popa, R. A.; and Stoica, I. 2024. Judgebench: A benchmark for evaluating llm-based judges. *arXiv preprint arXiv:2410.12784*.
- Team, M. A. 2024. Mistral NeMo.
- Wang, P.; Li, L.; Chen, L.; Cai, Z.; Zhu, D.; Lin, B.; Cao, Y.; Liu, Q.; Liu, T.; and Sui, Z. 2023a. Large language models are not fair evaluators. *arXiv preprint arXiv:2305.17926*.
- Wang, Y.; Yu, Z.; Zeng, Z.; Yang, L.; Wang, C.; Chen, H.; Jiang, C.; Xie, R.; Wang, J.; Xie, X.; et al. 2023b. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. *arXiv preprint arXiv:2306.05087*.
- Wei, J.; Karina, N.; Chung, H. W.; Jiao, Y. J.; Papay, S.; Glaese, A.; Schulman, J.; and Fedus, W. 2024. Measuring short-form factuality in large language models. *arXiv preprint arXiv:2411.04368*.
- Wu, T.; Yuan, W.; Golovneva, O.; Xu, J.; Tian, Y.; Jiao, J.; Weston, J.; and Sukhbaatar, S. 2024. Meta-rewarding language models: Self-improving alignment with llm-as-ameta-judge. *arXiv preprint arXiv:2407.19594*.
- Yang, R.; Rajagopal, D.; Hayati, S. A.; Hu, B.; and Kang, D. 2024. Confidence calibration and rationalization for LLMs via multi-agent deliberation. *arXiv preprint arXiv:2404.09127*.

- Ye, F.; Yang, M.; Pang, J.; Wang, L.; Wong, D.; Yilmaz, E.; Shi, S.; and Tu, Z. 2024. Benchmarking llms via uncertainty quantification. *Advances in Neural Information Processing Systems*, 37: 15356–15385.
- Yi, H.; Lin, F.; Li, H.; Ning, P.; Yu, X.; and Xiao, R. 2024. Generation meets verification: Accelerating large language model inference with smart parallel auto-correct decoding. *arXiv preprint arXiv:2402.11809*.
- Zeng, Z.; Yu, J.; Gao, T.; Meng, Y.; Goyal, T.; and Chen, D. 2024. Evaluating Large Language Models at Evaluating Instruction Following. arXiv:2310.07641.
- Zhao, Z.; Wallace, E.; Feng, S.; Klein, D.; and Singh, S. 2021. Calibrate before use: Improving few-shot performance of language models. In *International conference on machine learning*, 12697–12706. PMLR.
- Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.; et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36: 46595–46623.
- Zhou, K.; Jurafsky, D.; and Hashimoto, T. 2023. Navigating the Grey Area: How Expressions of Uncertainty and Overconfidence Affect Language Models. arXiv:2302.13439.
- Zhu, L.; Wang, X.; and Wang, X. 2025. JudgeLM: Finetuned Large Language Models are Scalable Judges. In *The Thirteenth International Conference on Learning Representations*.
- Zhuge, M.; Zhao, C.; Ashley, D.; Wang, W.; Khizbullin, D.; Xiong, Y.; Liu, Z.; Chang, E.; Krishnamoorthi, R.; Tian, Y.; et al. 2024. Agent-as-a-judge: Evaluate agents with agents. *arXiv preprint arXiv:2410.10934*.

# **A** Appendices

# **Self-Confidence (SC) Setting Prompt**

The following is the prompt template used in the Self-Confidence (SC) setting to elicit both the judgment result and confidence score from the LLM. Placeholders such as {{question}}, {{answer\_a}}, and {{answer\_b}} are replaced with the actual instruction and output pairs during evaluation.

```
1 You are a helpful assistant in
       evaluating the quality of the outputs
        for a given instruction. Your goal
       is to select the best output for the
       given instruction and provide a
       confidence score (0-100) for your
       selection.
2 Select the Output (a) or Output (b) that
        is better for the given instruction.
        The two outputs are generated by two
        different AI chatbots respectively.
3 Evaluate the following outputs, and
       provide your best guess along with a
       confidence score in the following
       JSON format:
4
5
     "selected_output": "Output (a)" or "
         Output (b)",
6
     "confidence_score": number,
     "explanation": "Your detailed
7
         explanation here"
8
  }
9 # Instruction:
10 {{question}}
11 # Output (a):
12 {{answer_a}}
13 # Output (b):
14
   {{answer_b}}
15 Your response must be in the JSON format
        as shown above. Do not output
       ANYTHING else. Do not provide the %
       symbol.
```

## **Multiple-Prompting (MP) Setting Prompt**

The following is the prompt template used in the Multiple-Prompting (MP) setting to elicit the judgment result from the LLM. Placeholders such as {{question}}, {{answer\_a}}, and {{answer\_b}} are replaced with the actual instruction and output pairs during evaluation.

```
1 You are a helpful assistant in
      evaluating the quality of the outputs
       for a given instruction. Your goal
      is to select the best output for the
      given instruction.
2 Select the Output (a) or Output (b) that
       is better for the given instruction.
       The two outputs are generated by two
       different AI chatbots respectively.
3 Evaluate the following outputs, and
      provide your best guess in the
      following JSON format:
    "selected_output": "Output (a)" or "
        Output (b)",
```

```
"explanation": "Your detailed
         explanation here"
7
8 # Instruction:
9
   {{question}}
10
   # Output (a):
11
   {{answer_a}}
12 # Output (b):
13 {{answer_b}}
14 Your response must be in the JSON format
        as shown above. Do not output
       ANYTHING else.
```

# **LLM-as-a-Fuser Prompt**

The following is the prompt template used in the LLM-as-a-Fuser framework to synthesize judgments from multiple models. Placeholders such as {{question}}, {{answer\_a}}, {{answer\_b}}, and the Jinja loop for JSON outputs are replaced with actual data during evalua-

```
1 You are a helpful assistant tasked with
       combining multiple model responses to
        select the best output for the
       following instruction: Evaluate the
       quality of multiple outputs for a
       given instruction and select the best
        one based on specific rules.
3
   **Task:**
4 You will receive:
5\, 1. The instruction describing the task.
   2. Multiple outputs (e.g., Output (a),
       Output (b)) generated by different
       models.
7 3. A list of JSON outputs, each
       containing:
 8
      - selected_output: The chosen output
          (e.g., "Output (a)").
      - confidence_score: A score showing
          the model's confidence (e.g., 85).
10
      - explanation: Why the model chose
          that output.
11
12 Your goal is to:
13 - Review the JSON outputs and evaluate
       the original outputs (Output (a),
       Output (b), etc.) using the
       evaluation rules.
14 - Pick the best output or create a new
       one by combining the best parts of
       multiple outputs.
   - Return a JSON response with the
       selected output, confidence_score,
       and an explanation.
16
17
   **Input:**
18
   - **Instruction**: {{ question }}
19
   - **Outputs**:
20
     - Output (a): {{ answer_a }}
21
     - Output (b): {{ answer_b }}
22 - **JSON Outputs**:
```

23 {% for output in json\_outputs %}

output }}

- JSON Output {{ loop.index }}: {{

```
{% endfor %}
25
26
27
   **Steps:**
28
   1. **Check JSON Outputs**:
29

    Look at each selected_output,

          confidence_score, and explanation.
30
       - Use the explanation to understand
          why the model picked that output.
31
       - Note the confidence_score, but
          focus on explanation quality and
          rule compliance.
32
   2. **Evaluate Original Outputs**:
33
       - Judge Output (a), Output (b), etc.,
           against the evaluation rules.
34
        Use JSON explanations to guide your
           evaluation.
35
      **Pick or Combine**:
36
       - Choose the best output if one
          clearly meets the rules.
37
      - If no output is perfect, combine
          the best parts of multiple outputs
           to create a better response.
38
   4. **Explain Your Choice**:
39
        Say why you picked the output or
          created a new one.
40
       - Mention the JSON outputs'
          explanations and scores, noting
          agreements or differences.
41
       - Show how your choice follows the
          rules better than others.
42
43
   **Output Format: **
   '''json
44
45
46
      "selected_output": "Output (a)" or "
         Output (b)",
47
      "confidence score": number(0-100),
48
      "explanation": "Why you chose this
         output or how you combined outputs,
          referencing JSON explanations,
         confidence scores, and evaluation
         rules."
49
```

### **Supplementary Figures and Tables**

![](_page_10_Figure_2.jpeg)

Figure 6: R1-Distill-Llama (SC setting)

![](_page_10_Figure_4.jpeg)

Figure 7: R1-Distill-Qwen (SC setting)

![](_page_10_Figure_6.jpeg)

Figure 8: Qwen3-235B-A22B (SC setting)

![](_page_10_Figure_8.jpeg)

Figure 9: GPT-4.1-nano (SC setting)

![](_page_10_Figure_10.jpeg)

Figure 10: OpenAI-o3-mini (SC setting)

| Model            | Acc ↑ | ECE ↓ | ACE ↓ | Brier Score ↓ | MCE ↓ | NLL ↓ | TH Score ↑ |
|------------------|-------|-------|-------|---------------|-------|-------|------------|
| DeepSeek-R1-0528 | 85.43 | 7.17  | 6.69  | 0.108         | 70.00 | 0.83  | 17.90      |
| Qwen3-235B-A22B  | 78.86 | 13.00 | 12.04 | 0.151         | 70.00 | 0.98  | 16.85      |
| OpenAI-o3-mini   | 76.00 | 18.49 | 18.77 | 0.184         | 43.91 | 1.84  | 17.08      |
| R1-Distill-Llama | 71.71 | 15.14 | 15.83 | 0.206         | 60.00 | 1.46  | 7.84       |
| R1-Distill-Qwen  | 67.71 | 18.06 | 17.72 | 0.215         | 37.42 | 1.17  | 8.10       |
| Claude-Sonnet-4  | 64.29 | 34.51 | 34.48 | 0.340         | 65.00 | 6.12  | 9.17       |
| GPT-4.1          | 63.14 | 34.91 | 34.96 | 0.346         | 70.00 | 6.04  | 8.27       |
| Gemini-2.5-Flash | 52.57 | 14.43 | 14.77 | 0.220         | 43.33 | 1.44  | 7.40       |
| DeepSeek-V3-0324 | 50.57 | 47.89 | 47.96 | 0.479         | 70.00 | 9.02  | 0.79       |
| GPT-4o           | 49.71 | 47.09 | 46.97 | 0.463         | 58.13 | 7.64  | 1.68       |
| GPT-4.1-mini     | 56.00 | 42.31 | 42.18 | 0.419         | 65.00 | 7.52  | 4.35       |
| Llama-3.3-70B    | 42.86 | 54.31 | 54.28 | 0.537         | 70.00 | 9.49  | -1.82      |
| GPT-4.1-nano     | 28.29 | 67.43 | 67.41 | 0.663         | 71.26 | 10.86 | -6.76      |
| Mistral-Nemo     | 19.43 | 68.89 | 68.93 | 0.643         | 78.69 | 6.69  | -4.35      |

Table 5: Performance Comparison of Different LLMs under MP Setting

| Model            | Acc ↑ | ECE ↓ | ACE ↓ | Brier Score ↓ | MCE ↓ | NLL ↓  | TH Score ↑ |
|------------------|-------|-------|-------|---------------|-------|--------|------------|
| DeepSeek-R1-0528 | 78.29 | 6.84  | 6.62  | 0.1298        | 46.15 | 0.4211 | 2.96       |
| GPT-4.1          | 63.43 | 34.46 | 34.43 | 0.3462        | 63.80 | 1.7287 | 7.36       |
| GPT-4.1-mini     | 55.14 | 42.56 | 42.53 | 0.4253        | 58.32 | 1.7946 | 2.56       |
| GPT-4o           | 50.86 | 45.05 | 45.06 | 0.4493        | 61.19 | 1.6238 | 0.79       |
| DeepSeek-V3-0324 | 48.29 | 50.76 | 50.68 | 0.5044        | 50.76 | 2.4714 | -0.85      |
| Llama-3.3-70B    | 43.43 | 53.53 | 53.55 | 0.5318        | 54.04 | 2.3400 | -3.25      |
| GPT-4.1-nano     | 28.00 | 66.05 | 66.16 | 0.6349        | 82.98 | 2.1206 | -8.70      |
| Mistral-Nemo     | 23.43 | 64.63 | 64.60 | 0.6051        | 79.59 | 1.7887 | -5.89      |

Table 6: Performance Comparison of Different LLMs under LogP Setting

![](_page_12_Figure_0.jpeg)

Figure 11: Llama-3.3-70B (SC setting)

![](_page_12_Figure_2.jpeg)

Figure 12: GPT-4.1 (SC setting)

![](_page_12_Figure_4.jpeg)

Figure 13: GPT-4.1-mini (SC setting)

![](_page_12_Figure_6.jpeg)

Figure 14: Claude-Sonnet-4 (MP setting)

![](_page_12_Figure_8.jpeg)

Figure 15: DeepSeek-V3-0324 (MP setting)

![](_page_12_Figure_10.jpeg)

Figure 16: DeepSeek-R1-0528 (MP setting)

![](_page_12_Figure_12.jpeg)

Figure 17: R1-Distill-Llama (MP setting)

![](_page_12_Figure_14.jpeg)

Figure 18: R1-Distill-Qwen (MP setting)

![](_page_13_Figure_0.jpeg)

Figure 19: Gemini-2.5-Flash (MP setting)

![](_page_13_Figure_2.jpeg)

Figure 20: Llama-3.3-70B (MP setting)

![](_page_13_Figure_4.jpeg)

Figure 21: Mistral-Nemo (MP setting)

![](_page_13_Figure_6.jpeg)

Figure 22: GPT-4.1 (MP setting)

![](_page_13_Figure_8.jpeg)

Figure 23: GPT-4.1-mini (MP setting)

![](_page_13_Figure_10.jpeg)

Figure 24: GPT-4.1-nano (MP setting)

![](_page_13_Figure_12.jpeg)

Figure 25: GPT-4o (MP setting)

![](_page_13_Figure_14.jpeg)

Figure 26: OpenAI-o3-mini (MP setting)

![](_page_14_Figure_0.jpeg)

Figure 27: Qwen3-235B-A22B (MP setting)

![](_page_14_Figure_2.jpeg)

Figure 28: DeepSeek-V3-0324 (Logp setting)

![](_page_14_Figure_4.jpeg)

Figure 29: R1-Distill-Llama (Logp setting)

![](_page_14_Figure_6.jpeg)

Figure 30: Llama-3.3-70B (Logp setting)

![](_page_14_Figure_8.jpeg)

Figure 31: Mistral-Nemo (Logp setting)

![](_page_14_Figure_10.jpeg)

Figure 32: GPT-4.1 (Logp setting)

![](_page_14_Figure_12.jpeg)

Figure 33: GPT-4.1-mini (Logp setting)

![](_page_14_Figure_14.jpeg)

Figure 34: GPT-4.1-nano (Logp setting)

![](_page_15_Figure_0.jpeg)

Figure 35: GPT-4o (Logp setting)

![](_page_15_Figure_2.jpeg)

Figure 36: Qwen3-235B-A22B (Logp setting)

![](_page_15_Figure_4.jpeg)

Figure 37: DeepSeek-R1-0528 (Logp setting)