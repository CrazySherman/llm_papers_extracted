# KNOWLEDGE or REASONING ? A Close Look at How LLMs Think Across Domains

Juncheng Wu1<sup>⋆</sup> Sheng Liu2<sup>⋆</sup> Haoqin Tu1<sup>⋆</sup> Hang Yu3<sup>⋆</sup> Xiaoke Huang<sup>1</sup> James Zou<sup>2</sup> Cihang Xie<sup>1</sup> Yuyin Zhou<sup>1</sup>

> ⋆ equal technical contribution

<sup>1</sup>UC Santa Cruz <sup>2</sup>Stanford University <sup>3</sup>Tongji University

Project Page: <https://ucsc-vlaa.github.io/ReasoningEval> Code: <https://github.com/UCSC-VLAA/ReasoningEval>

# Abstract

Recent advances in reasoning-enhanced Large Language Models such as OpenAIo1/3 and DeepSeek-R1 have significantly improved performance on complex tasks. However, the quality and transparency of their internal reasoning processes remain underexplored. This work moves beyond the final-answer accuracy and investigates step-by-step reasoning in the medical and mathematical domains by explicitly decomposing the thinking trajectories into two parts: *knowledge* and *reasoning*. Specifically, we introduce a fine-grained evaluation framework that judges: (1) the correctness of *knowledge* used (measured by Knowledge Index (KI)) and (2) the quality of *reasoning* (measured by Information Gain (InfoGain)). Using this framework, we study R1-distilled and base Qwen models trained with supervised fine-tuning (SFT) and/or reinforcement learning (RL) in the medical and math domains. Three intriguing findings emerge: (1) The general reasoning abilities in R1-distilled models do not transfer effectively to the medical domain through either SFT or RL. (2) SFT raises final-answer accuracy in both domains, but often at the cost of reasoning quality: InfoGain drops by 38.9% on average compared with untrained models; In the medical domain, however, SFT remains crucial because domain knowledge is indispensable. (3) RL enhances medical reasoning by pruning inaccurate or irrelevant knowledge from reasoning paths, thereby improving both reasoning accuracy and knowledge correctness.

# 1 Introduction

Recent proprietary and open-source Large Language Models (LLMs) [\[23,](#page-10-0) [18,](#page-10-1) [21\]](#page-10-2) have demonstrated remarkable progress in reasoning-intensive benchmarks, particularly in mathematics and general knowledge [\[43,](#page-11-0) [33,](#page-11-1) [36,](#page-11-2) [6\]](#page-9-0). Despite this rapid progress, the evaluation of LLM reasoning remains limited in scope—largely centered on final-answer accuracy or aggregate performance metrics. Such evaluations obscure the step-by-step process by which models reason and offer little insight into the interplay between factual knowledge and logical inference that underlies these capabilities.

Earlier work [\[14\]](#page-10-3) evaluates reasoning based on its embedding similarity to the original question, assuming higher similarity implies greater informativeness and faithfulness. However, LLMs often rely on internal knowledge or previous deductions, making question alignment an unreliable measure of knowledge accuracy or reasoning quality. As shown in Tab. [2](#page-12-0) in the Appendix, existing reasoning metrics yield similar scores across models but with differing capacities, suggesting their unreliability.

![](_page_1_Figure_0.jpeg)

<span id="page-1-0"></span>Figure 1: Reasoning and Knowledge are Different Evaluation Aspects for LLMs. A reasoning step may effectively reduce uncertainty toward the final answer despite relying on incorrect knowledge (*e.g.*, Step 3), or it may present factually correct but irrelevant/redundant knowledge that hinders reasoning efficiency (*e.g.*, Step 4). Accuracy alone fails to capture these nuances. We introduce two complementary metrics that separately evaluate knowledge correctness and reasoning informativeness.

Such gaps in reasoning evaluation are particularly salient when comparing domains with different knowledge and reasoning demands. For instance, mathematical problems often emphasize symbolic manipulation and internal consistency [\[5\]](#page-9-1), whereas medical tasks typically require the integration of domain-specific knowledge grounded in external facts [\[13\]](#page-10-4). Both domains involve multi-step reasoning, but they differ in how much they depend on knowledge versus the reasoning steps required during generation. Understanding these differences is critical not only for building domain-adaptive models but also for advancing interpretability and reliability in high-stakes applications.

In this work, we pose a fundamental question: *What are the respective roles of knowledge and reasoning in the thinking process of LLMs, and how do they interact across different domains?* To answer this, we introduce an evaluation framework (see Fig. [2\)](#page-2-0) that decomposes each reasoning step into two components: the factual knowledge it invokes and the logical reasoning operation it performs. We define two novel metrics to quantify reasoning and knowledge: (1) Information Gain (Info Gain) how much a reasoning step reduces uncertainty toward the final answer, calculated as the probability gap between adjacent response steps. A higher Info Gain indicates a more informative reasoning path towards the final answer. (2) Knowledge Index (KI), on the contrary, evaluates the factual correctness of each step by verifying extracted knowledge against external ground truth sources. We identify the knowledge point in each reasoning step and access external factual data to verify if the knowledge aligns with the retrieved facts. Finally, models with stronger knowledge grounding yield higher KI scores. A running example is provided in Figure [1](#page-1-0) to explain our motivation intuitively. This fine-grained evaluation allows us to characterize not just the model's final performance, but also the trajectory it takes to get there.

Building upon this framework, we analyze models trained via supervised fine-tuning (SFT) and reinforcement learning (RL) across both mathematical and medical domains. Our findings reveal several key insights: (1) mathematical reasoning does not naturally transfer to the medical domain via SFT, largely due to domain-specific knowledge gaps, as evidenced by the consistently lower performance of the DeepSeek-distilled model; (2) tasks across domains demand distinct model competencies—medical problems require richer domain knowledge, with knowledge–accuracy correlations exceeding those of reasoning–accuracy in four of five benchmarks; (3) while SFT improves final accuracy and raises knowledge levels (*e.g.*, a 6.2% average KI increase on medical tasks), it often introduces verbose or suboptimal reasoning, reducing Info. Gain by an average of 38.9%; and (4) RL mitigates such inefficiencies by reinforcing correct knowledge trajectories, boosting medical knowledge with an average KI gain of 12.4. We hope our study can foster a more comprehensive understanding of individual LLM reasoning steps and how they collectively influence the reliability of model outcomes. This, in turn, could be helpful in guiding the way towards more effective training strategies to build reliable LLMs.

### 2 Related Works

Reasoning-enhanced LLMs. Recent reasoning large language models (LLMs) have shown remarkable performance in the fields of mathematics [\[11,](#page-9-2) [36\]](#page-11-2) and medicine [\[41,](#page-11-3) [9\]](#page-9-3) Subsequent studies

![](_page_2_Figure_0.jpeg)

<span id="page-2-0"></span>Figure 2: Evaluation Pipeline. (a) We decompose the model's reasoning into reasoning steps using gpt4o[\[27\]](#page-10-5) , then evaluate the (b) information gain and (c) knowledge index of each reasoning step.

have focused on enhancing the reasoning abilities of LLMs by producing high-quality training datasets [\[40,](#page-11-4) [17,](#page-10-6) [39\]](#page-11-5) or crafting comprehensive reward mechanisms [\[8,](#page-9-4) [4\]](#page-9-5). However, these efforts are directed exclusively towards boosting the answering accuracy of reasoning LLMs but overlook the comprehension of internal reasoning processes. This study seeks to investigate deeper into the reasoning process by introducing a step-by-step pipeline for reasoning evaluation, aiming to offer empirical findings that aid in the advancement of stronger reasoning models.

Evaluating LLMs Reasoning beyond Accuracy. Examining the quality of LLMs' reasoning cannot be restricted to just evaluating the accuracy of the final answer. Prior studies concentrated on identifying factual inaccuracies throughout the whole reasoning process [\[48,](#page-11-6) [42,](#page-11-7) [30\]](#page-10-7), rather than assessing each reasoning step separately. Research such as [\[14\]](#page-10-3) proposed to evaluate an individual reasoning step by measuring its embedding similarity with the source information provided in the question. However, similarity to the source content cannot fully reveal the knowledge correctness or logical effectiveness of a reasoning step, as LLMs often generate reasoning based on its internal knowledge and or previous reasoning steps. Nonetheless, LLMs frequently produce reasoning grounded in their internal knowledge or prior reasoning steps. Merely resembling the original content doesn't entirely indicate the knowledge correctness or logical soundness of a reasoning step. In this paper, we propose to step-by-step evaluate LLMs' across two dimensions to gain a thorough comprehension of the quality of reasoning.

## <span id="page-2-1"></span>3 A Closer Look at Reasoning Evaluation

#### 3.1 Evaluation Pipeline

All experiments in both the medical and math domains are initialized from the universal 7B-parameter base models Qwen2.5-7B [\[37\]](#page-11-8) and DeepSeek-R1-Distill-Qwen 7B [\[11\]](#page-9-2). We choose these two models as baselines because: (1) Qwen2.5-7B and its DeepSeek-distilled variant show strong generalization across domains, with the DeepSeek-distilled one often matching or outperforming larger or private models like Claude-3.5 in math and coding [\[11\]](#page-9-2); (2) their open-source nature enables in-depth study of training, evaluation, and architecture; and (3) their shared backbone ensures fair comparison of post-training thinking patterns.

We employ both the SFT and RL training to the model. In detail, for medical domain, we take *huatuoGPT-o1* [\[9\]](#page-9-3) with SFT and RL data splits for respective SFT and RL training, while in the math domain, we employ SFT and RL splits from *RLHFlow* [\[12\]](#page-9-6) for the corresponding training strategy.

With regard to the evaluation across both domains, we consider [MedMCQA](https://medmcqa.github.io/) [\[28\]](#page-10-8), [MedQA-USMLE](https://paperswithcode.com/dataset/medqa-usmle) [\[19\]](#page-10-9), [PubMedQA](https://pubmedqa.github.io/) [\[20\]](#page-10-10), [MMLU-Pro \(Medical\)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) [\[38\]](#page-11-9), [MedXpertQA](https://arxiv.org/abs/2501.18362) [\[49\]](#page-11-10) for medical and [AIME 2024](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) [\[1\]](#page-9-7), [MATH500](https://www.vals.ai/benchmarks/math500-03-24-2025) [\[2\]](#page-9-8), [AMC \(10 & 12\)](https://huggingface.co/datasets/hendrycks/competition_math) [\[16\]](#page-10-11), [Minerva-Math](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/minerva_math) [\[22\]](#page-10-12), [USAMO 2025 /](https://www.maa.org/math-competitions) [Olympiad](https://www.maa.org/math-competitions) [\[3\]](#page-9-9) for math. More details about model training configurations, evaluation and training datasets are in Appendix [B.](#page-12-1)

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Figure 3: Example of reasoning decomposition. Every stage of reasoning corresponds to a logical step  $s_i$ , accompanied by the specific knowledge point it contains  $(k_i)$ .

In this section, we investigate the roles of reasoning and knowledge in model responses by first decomposing them into separate steps, and then introducing two novel metrics to evaluate the knowledge and reasoning abilities embedded in those responses.

### 3.2 Response Decomposition

In the first stage, we aim at decomposing the model responses into explicit and successive steps. Given a question Q, the LLM produces the reasoning process  $\mathcal{R}$  and subsequently delivers the answer A. As depicted in Fig. 2 (a), our evaluation pipeline initially breaks down  $\mathcal{R}$  into a sequence of logical steps using gpt4o, denoted as  $\mathcal{S} = [s_1, s_2, \cdots, s_t]$ . We employ different prompts fed in gpt4o for different domains with more details presented in Fig. 10 and Fig. 11.

For instance, as shown in Fig. 3, to answer a question regarding cubitus varus, the model engages in reasoning involving six logical steps. The initial step introduces *the characteristic of cubitus varus*, followed by *describing the appearance of elbow* in the second step.

#### 3.3 Information Gain

To quantify the uncertainty reduced by each logical step towards the final answer, we track the change in the model's confidence over each step, using the probability assigned to the true answer by leveraging step-wise perplexity (PPL) [7]. As shown in Fig. 2 (b), let Q be the question, and  $S = [s_1, s_2, \ldots, s_t]$  be the sequence of reasoning steps. Let  $A^*$  be the correct solution to the query. After each step  $s_i$ , we concatenate the steps  $s_{1:i}$  and compute the probability assigned to the correct solution. To avoid potential biases introduced by self-evaluation [29], we deploy another language model (i.e., the untrained Qwen2.5-7B) for PPL calculation:

$$P_i = \prod_{j=1}^{N} P(A_j^*|Q, s_{1:i}),$$

where  $A_j^*$  is the j-th token in the correct answer, and N is the total number of tokens in  $A^*$ . This measures how likely the model is to generate the correct answer given the current plan. We then convert the probability into perplexity (PPL) and calculate the PPL gain between adjacent steps:

$$[\Delta PPL_1, \Delta PPL_2, \dots, \Delta PPL_t]$$
, where  $\Delta PPL_i = PPL_{i-1} - PPL_i$ ,

where  $PPL_i$  represents the PPL of the i-th step. Each  $\Delta PPL_i$  is designed to measure how effective a step  $s_i$  is in reducing the model's uncertainty towards the final answer. In the end, we average the  $\Delta PPLs$  across all steps to obtain the final information gain:

$$\Delta I = \frac{1}{t} \sum_{i=1}^{t} \Delta \text{PPL}_i.$$

While a higher  $\Delta I$  indicates that more information emerges during reasoning, reflecting stronger reasoning capabilities, a lower  $\Delta I$  suggests weaker reasoning, likely characterized by redundant or less informative model responses.

![](_page_4_Figure_0.jpeg)

<span id="page-4-0"></span>Figure 4: **Comparison of SFT Qwen-Base and Qwen-R1.** While the medical knowledge of all reasoning steps in Qwen-R1 + SFT is correct, the ignorance of considering more appropriate treatment for the specified disease results in an incorrect answer.

#### 3.4 Knowledge Index

As for accessing the knowledge presented in model responses, we propose the metric Knowledge Index (KI) (show in Fig. 2 (c)). For each step, we retrieve ground truth answers for the extracted knowledge from medical textbooks [19] using gpt-40, and then use these references to assess whether the step aligns with the factual knowledge.

Specifically, we decompose the process of obtaining the knowledge correctness into three stages:

- **Knowledge Extraction.** In each step  $s_i$ , we employ gpt4o [27] to extract knowledge in the model response step  $s_i$ , denoted as  $k_i$ . We then reformat  $k_i$  as a question of querying about specific knowledge for the next step retrieval from database.
- Knowledge Retrieval. Using the generated knowledge query, we employ a widely-employed and well-structured medical database to retrieve relevant external knowledge  $k_i^*$  as the ground truth answer to the knowledge query  $k_i$ .
- **Knowledge Judgement.** Finally, to determine whether the extracted knowledge aligns with the ground truth, we use gpt-40 to assess whether each reasoning step  $s_i$  is consistent with the retrieved facts provided in the prompt. The outcome of this consistency check, consistency, is recorded as a Boolean value: True if consistent, False otherwise.

The overall knowledge index across all steps for each query is computed as

$$KI = \frac{1}{t} \sum_{i=1}^{t} consistency_i.$$

A higher KI metric consistently reflects more accurate knowledge incorporated during generation, thus stronger knowledge capacity in the model. We present more details about gpt4o input prompts and the calculation of the proposed metrics in Appendix B.2.

### 4 Experiment Findings

### 4.1 Main Results

We first focus on probing the performance of different models on the general accuracy or the knowledge and reasoning aspects in the medical domain. We select two types of base models for training: Qwen-Base (Qwen-2.5-7B-base) and Qwen-R1 (DeepSeek-R1 distilled Qwen2.5-7B).

| Table 1: Comparison between base model and R1      | <b>distilled model.</b> After finetuning in the medical |
|----------------------------------------------------|---------------------------------------------------------|
| domain, base model demonstrates consistently super | rior performance across all metrics                     |

<span id="page-5-0"></span>

| Base Model              | SFT | RL | MedMCQA | MedQA | PubMedQA | MMLU-Pro | MedXpert | AVG   |
|-------------------------|-----|----|---------|-------|----------|----------|----------|-------|
| Metric: Accuracy (%)    |     |    |         |       |          | Low      | High     |       |
| Qwen-R1                 | 1   | Х  | 36.29   | 35.69 | 66.07    | 41.54    | 10.74    | 38.07 |
|                         | ✓   | ✓  | 33.20   | 32.60 | 50.70    | 30.49    | 10.40    | 31.48 |
| Qwen-Base               | ✓   | X  | 55.48   | 62.61 | 71.17    | 59.11    | 15.57    | 52.79 |
|                         | 1   | ✓  | 64.04   | 71.56 | 78.40    | 68.27    | 17.81    | 60.02 |
| Metric: Info Gain       |     |    |         |       |          |          |          |       |
| 0 01                    | /   | X  | 8.876   | 0.139 | 0.205    | 1.729    | 0.298    | 2.249 |
| Qwen-R1                 | ✓   | ✓  | 9.202   | 0.113 | 0.183    | 1.750    | 0.298    | 2.309 |
| Qwen-Base               | ✓   | X  | 9.291   | 0.157 | 0.192    | 1.785    | 0.312    | 2.347 |
|                         | ✓   | ✓  | 9.314   | 0.161 | 0.190    | 1.762    | 0.315    | 2.348 |
| Metric: Knowledge Index |     |    |         |       |          |          | High     |       |
|                         | 1   | Х  | 44.41   | 61.97 | 54.51    | 69.32    | 52.24    | 56.49 |
| Qwen-R1                 | ✓   | ✓  | 41.35   | 59.41 | 47.81    | 66.38    | 56.48    | 54.29 |
| 0 0                     | ✓   | X  | 57.92   | 67.65 | 59.69    | 68.48    | 63.24    | 63.40 |
| Qwen-Base               | 1   | ✓  | 59.23   | 69.17 | 54.95    | 71.93    | 65.87    | 64.23 |

Subsequently, we follow the conventional paradigm of deploying SFT and SFT + RL [11] on LLMs under the same setting and present results in Table 1.

Trained Qwen-Base Outperforms its R1-distilled **Counterpart.** From the overall accuracy results, we observe that Qwen-Base consistently outperforms the R1-distilled variant across the evaluated benchmarks, whether using SFT alone or in combination with subsequent RL. For instance, Owen-Base with only SFT witnesses a 14.7% average accuracy improvement over the distilled model (52.79% vs. 38.07%). With the addition of RL, the gap widens further with a 22.6% increase in average accuracy (54.06% vs. 31.48%). This interesting performance difference may stem from a domain shift across training stages. Since Owen-R1 was primarily trained on R1-generated texts focused on math and code [11], the medical domain knowledge introduced during our post-training could conflict with its prior representations, thereby undermining its performance in medical tasks.

To illustrate the advantage of the trained Qwen-Base over Qwen-R1 more intuitively, we include a case in Fig. 4, where Qwen-R1 + SFT fails to produce

![](_page_5_Figure_5.jpeg)

<span id="page-5-1"></span>Figure 5: Correlations between the proposed two metrics and accuracy. Different tasks require different levels of knowledge and/or reasoning capabilities in LLMs.

the correct answer. In this example, although the model demonstrates sound reasoning, it selects Fluorometholone without evaluating safer alternatives. Ketotifen, with fewer side effects, would have been the better choice with less side effect—highlighting the importance of nuanced decision-making in medicine, unlike the single-solution nature of math problems.

Knowledge and Reasoning Should be Two Distinct Evaluation Aspects for LLMs. When turning our focus on reasoning and knowledge as separate evaluation dimensions, we find that these aspects should be treated independently. While the reasoning abilities of the two Qwen-Base variants remain comparable, the RL-ed Qwen-Base demonstrates slightly better medical knowledge according to our KI metric (64.2 vs. 63.4). Similarly, although the RL-enhanced Qwen-R1 shows a minor improvement over its SFT-only version in general performance, it underperforms in knowledge

![](_page_6_Figure_0.jpeg)

<span id="page-6-0"></span>Figure 6: **Comparison between medical and math domain.** (a) In mathematics, RL enhances accuracy most, whereas in the medical field, SFT provides a greater enhancement in overall accuracy; (b) Across both fields, RL is more adept at boosting information gain, whereas SFT results in a reduction of information gain; (c) Within the medical field, SFT achieves the pinnacle of the knowledge index.

evaluation, trailing by 2.2 points in the KI metric (54.3). This pattern is also observed in specific datasets like MMLU-Pro and MedMCQA, hence supporting our argument for a clear separation between reasoning and knowledge evaluation.

The Challenging Nature of Different Benchmarks may Inherit from Different Aspects. Separately evaluating knowledge and reasoning raises the question of which contributes more to the final accuracy. To explore this, we compute the correlation scores between each capability and the accuracy of the task in Figure 5 with the values on the x axis and on the y axis representing accuracy and KI, accuracy and Info. Gain correlations, respectively. Our findings suggest that benchmark difficulty often hinges more on specific aspects. In PubMedQA, knowledge dominates, with KI correlating more strongly with accuracy than InfoGain (0.897 vs. 0.465). By contrast, tasks like MedQA and MedXpert depend on both aspects, as shown by correlations above 0.9 for both metrics. The figure also implies that most medical benchmarks are more knowledge-intensive than reasoning-driven, as all correlation points except one fall below the x=y line. This line represents parity between the correlation of accuracy with knowledge and with reasoning. This trend is reinforced by the average scores across all tasks, where KI shows a striking 0.998 correlation with the final accuracy—0.3 higher than that of reasoning.

### 4.2 SFT vs. RL in Medical and Math Domains

Motivated by recent researches focusing on exploring the roles of SFT and RL in training reasoning models, we take a step forward on the popular training paradigm of "SFT then RL" and leverage SFT or RL separately on medical and math tasks. In detail, for the medical capability we employ Qwen2.5-7B-Base [37] on the medical-o1 data [9], while in the math domain, we take Qwen2.5-7B-math [44] as the base model trained on RLHFlow data [47, 12]. We only present the knowledge index metric for medical benchmarks, as the knowledge base of math.

**SFT Usually Provides Knowledge with Reasoning Compromised, RL Boosts Both.** Figure(c) 6 shows that in the medical domain, both RL and SFT encourage the model to incorporate more domain-specific knowledge. SFT consistently yields a greater boost than RL, with average absolute gains in the KI metric of 13.69 and 10.53 over the base model, respectively (base 49.71 < RL-ed 60.24 < SFT-ed 63.40). By contrast, the reasoning ability of models follows an opposite trend in

![](_page_7_Figure_0.jpeg)

Figure 7: Comparison of Models w and wo SFT We compare both factual grounding and reasoning efficiency. SFT adds redundant reasoning steps, reducing per-step information gain and inference efficiency.

![](_page_7_Figure_2.jpeg)

<span id="page-7-0"></span>Figure 8: RL improves knowledge Index. In the medical domain, applying RL to base model before or post SFT consistently improves the knowlegde index.

Figure [6\(](#page-6-0)b): RL consistently enhances reasoning across both medical and math domains (0.39 *vs.* 0.42 in medical, 0.13 *vs.* 0.15 in math). SFT, however, appears to hinder reasoning, leading to notable drops of 0.17 and 0.04 in the medical and math domains, respectively (0.22 *vs.* 0.39 in medical, 0.09 *vs.* 0.13 in math). This drop accounts for an average reduction of 37.1% in the original Info Gain scores.

These findings suggest that while RL promotes a more balanced improvement in both knowledge and reasoning, SFT tends to favor domain knowledge (*e.g.*, medical facts) at the cost of undermining the model's inherent reasoning ability, consistent with previous observations [\[10,](#page-9-11) [8\]](#page-9-4).

Medical Problems are Likely to Require More Knowledge While Math Needs Enhanced Reasoning. Final accuracy on medical tasks, shown in Figure [6\(](#page-6-0)a), indicates that the SFT-ed model outperforms both its RL-ed and base counterparts, with average gains of 4.6% and 6.2%, respectively (SFT-ed 49.8% > RL-ed 49.8% > base 48.2%). However, in the math domain, the RL-ed variant leads to the highest accuracy, surpassing both the base and SFT-ed models (average RL-ed 61.7% > SFT-ed 51.9% > base 40.5%). Given the earlier evidence that SFT enhances domain knowledge while RL improves reasoning, we conclude that, unlike reasoning-intensive tasks such as math problems, challenging medical tasks benefit more from additional domain knowledge, resulting in higher task accuracy.

RL can Improve Model's Medical Knowledge Correctness. While RL is widely recognized for enhancing medical reasoning abilities [\[9\]](#page-9-3), it can also improve the knowledge presented within reasoning trajectories, as discussed earlier. To explore this effect, we analyze Qwen-Base with either RL alone or SFT followed by RL, as shown in Figure [8.](#page-7-0) The results indicate that RL boosts the knowledge index metric across different medical tasks, with notable gains of 12.4 points (RL alone) and 2.2 points (SFT+RL) over their respective RL-free baselines. This is particularly interesting given that RL introduces little new knowledge to the model [\[10,](#page-9-11) [8\]](#page-9-4).

![](_page_8_Figure_0.jpeg)

<span id="page-8-0"></span>Figure 9: **Comparison of models w/ and w/o RL.** We compare the knowledge correctness of the models' reasoning steps. RL improves the knowledge correctness by guiding the model to select reasoning paths that have fewer knowledge errors.

To further investigate, we conduct a case study revealing that RL enhances knowledge metrics by guiding the model to discard reasoning paths containing incorrect knowledge, rather than adding new facts. We refer to this effect as enhanced knowledge correctness. As shown in Fig. 9, both the SFT-only and SFT+RL models begin similarly, correctly linking symptoms to chemotherapy-induced hearing loss. The key challenge lies in identifying which chemotherapy drugs are ototoxic. While both cisplatin and carboplatin fit this criterion and share a DNA cross-linking mechanism, the SFT-only model incorrectly selects carboplatin and misattributes its mechanism to free radical generation, leading to an incorrect answer. In contrast, the SFT+RL model correctly selects cisplatin, avoiding the knowledge error.

### 5 Discussion and Conclusion

**Limitations.** Our evaluation framework explicitly separates knowledge from reasoning steps, though this initial study has its limitations. In particular, all experiments were conducted on the Qwen family of 7B-parameter models. Nevertheless, the two tested variants (Qwen-R1 and Qwen-Base) reflect strong generalization, robustness, and high-capacity. We therefore consider the findings reasonably convincing within this scope. We acknowledge that evaluating a broader range of models is necessary to assess the generality of our conclusions. Although our benchmark focuses on mathematics and medicine, these domains were chosen deliberately: mathematics serves as a conventional testbed for reasoning in LLMs [35], while medicine represents a high-stakes domain where factual accuracy and domain knowledge are essential for ensuring safety and trust [26].

Broader Impact. Looking ahead, we believe our evaluation framework can be extended to other structured reasoning domains. In legal tasks, for instance, the IRAC[34] (Issue, Rule, Application, Conclusion) structure offers a natural alignment with stepwise evaluation for knowledge and reasoning decomposition [34]. A model could be prompted to identify the issue, retrieve relevant laws (knowledge), apply them to the case (reasoning), and conclude—allowing our framework to assess performance at each stage. Early work on legal reasoning with LLMs (e.g., LegalBench [15], Chain of Logic prompting [34]) highlights the value of explicit multi-step reasoning. Similarly, financial and economic tasks[46] involve structured reasoning, often combining domain knowledge with time-series data. Recent studies show LLMs can integrate textual knowledge with historical data to make forecasts [46]. A typical financial problem may involve summarizing trends (knowledge), applying economic models (reasoning), and predicting outcomes. Although processing raw numerical data remains challenging, promising results suggest that LLMs can perform sequential reasoning over such inputs. Adapting our knowledge-vs.-reasoning framework to these domains, despite their structural differences, offers a promising path to better understand and improve LLM reasoning in complex, high-stakes settings.

Conclusion. In this paper, we put our primary focus on decomposing the reasoning paths of LLMs into knowledge and reasoning components to better understand their contributions to final performance. Through our proposed evaluation framework and analysis of Qwen models trained with SFT and RL, we demonstrate that these two capacities are not only distinct but also unequally demanded across domains. Our findings reveal that SFT boosts factual knowledge, especially in knowledge-intensive domains like medicine, albeit at the cost of reasoning clarity, while RL improves reasoning quality by pruning incorrect knowledge. This decomposition offers a more transparent lens into LLM decision-making and paves the way for targeted improvements in domain-specific reasoning tasks.

# 6 Acknowledgement

This work was partially funded by an unrestricted gift from Google. We thank the Microsoft Accelerate Foundation Models Research Program for supporting our computing needs.

# References

- <span id="page-9-7"></span>[1] American invitational mathematics examination (aime) 2024 problems. [https://](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) [artofproblemsolving.com/wiki/index.php/AIME\\_Problems\\_and\\_Solutions](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions), 2024. Accessed 7 May 2025.
- <span id="page-9-8"></span>[2] Math500 benchmark (2025-03-24 release). [https://www.vals.ai/benchmarks/](https://www.vals.ai/benchmarks/math500-03-24-2025) [math500-03-24-2025](https://www.vals.ai/benchmarks/math500-03-24-2025), 2025. Accessed 7 May 2025.
- <span id="page-9-9"></span>[3] Usamo 2025 problems and solutions. <https://www.maa.org/math-competitions>, 2025. Accessed 7 May 2025.
- <span id="page-9-5"></span>[4] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. *arXiv preprint arXiv:2503.04697*, 2025.
- <span id="page-9-1"></span>[5] Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. Large language models for mathematical reasoning: Progresses and challenges. *arXiv preprint arXiv:2402.00157*, 2024.
- <span id="page-9-0"></span>[6] Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. Llama-nemotron: Efficient reasoning models. *arXiv preprint arXiv:2505.00949*, 2025.
- <span id="page-9-10"></span>[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, and *et al.* Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020.
- <span id="page-9-4"></span>[8] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. *arXiv preprint arXiv:2504.11468*, 2025.
- <span id="page-9-3"></span>[9] Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. Huatuogpt-o1, towards medical complex reasoning with llms. *arXiv preprint arXiv:2412.18925*, 2024.
- <span id="page-9-11"></span>[10] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. *arXiv preprint arXiv:2501.17161*, 2025.
- <span id="page-9-2"></span>[11] DeepSeek AI. Deepseek-r1-distill-qwen-7b. [https://huggingface.co/deepseek-ai/](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), 2025. HuggingFace model card, accessed 7 May 2025.
- <span id="page-9-6"></span>[12] Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. *Transactions on Machine Learning Research*, 2024. Technical Report.

- <span id="page-10-4"></span>[13] Ethan Goh, Robert Gallo, Jason Hom, Eric Strong, Yingjie Weng, Hannah Kerman, Joséphine A Cool, Zahir Kanjee, Andrew S Parsons, Neera Ahuja, et al. Large language model influence on diagnostic reasoning: a randomized clinical trial. *JAMA Network Open*, 7(10):e2440969– e2440969, 2024.
- <span id="page-10-3"></span>[14] Olga Golovneva, Moya Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. Roscoe: A suite of metrics for scoring step-by-step reasoning. *arXiv preprint arXiv:2212.07919*, 2022.
- <span id="page-10-15"></span>[15] Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, et al. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. *Advances in Neural Information Processing Systems*, 36:44123–44279, 2023.
- <span id="page-10-11"></span>[16] Dan Hendrycks, Steven Basart, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.
- <span id="page-10-6"></span>[17] Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, and Yuyin Zhou. m1: Unleash the potential of test-time scaling for medical reasoning with large language models. *arXiv preprint arXiv:2504.00869*, 2025.
- <span id="page-10-1"></span>[18] Yunjie Ji, Xiaoyu Tian, Sitong Zhao, et al. Am-thinking-v1: Advancing the frontier of reasoning at 32b scale. *arXiv preprint*, 2025.
- <span id="page-10-9"></span>[19] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. *arXiv preprint arXiv:2009.13081*, 2020.
- <span id="page-10-10"></span>[20] Qiao Jin, Bhavdeep Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. In *EMNLP-IJCNLP*, pages 2567–2577. Association for Computational Linguistics, 2019.
- <span id="page-10-2"></span>[21] William Jurayj, Jeffrey Cheng, and Benjamin Van Durme. Is that your final answer? test-time scaling improves selective question answering. *arXiv preprint*, 2025.
- <span id="page-10-12"></span>[22] Aitor Lewkowycz, Ethan Du, Klaus Siniscalchi, Jason Wei, Xuezhi Wang, et al. Solving quantitative reasoning problems with language models. *arXiv preprint arXiv:2206.14858*, 2022.
- <span id="page-10-0"></span>[23] Wenfeng Liang, DeepSeek-AI, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint*, 2025.
- <span id="page-10-16"></span>[24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *International Conference on Learning Representations*, 2019.
- <span id="page-10-17"></span>[25] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. *arXiv preprint arXiv:2501.19393*, 2025.
- <span id="page-10-14"></span>[26] Harsha Nori, Nicholas King, Scott Mayer McKinney, Dean Carignan, and Eric Horvitz. Capabilities of gpt-4 on medical challenge problems. *arXiv preprint arXiv:2303.13375*, 2023.
- <span id="page-10-5"></span>[27] OpenAI. Gpt-4o system card. *arXiv preprint arXiv:2410.21276*, 2024.
- <span id="page-10-8"></span>[28] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. In *Proceedings of the Conference on Health, Inference, and Learning*, pages 248–260. PMLR, 2022.
- <span id="page-10-13"></span>[29] Arjun Panickssery, Samuel Bowman, and Shi Feng. Llm evaluators recognize and favor their own generations. *Advances in Neural Information Processing Systems*, 37:68772–68802, 2024.
- <span id="page-10-7"></span>[30] Archiki Prasad, Swarnadeep Saha, Xiang Zhou, and Mohit Bansal. Receval: Evaluating reasoning chains via correctness and informativeness. *arXiv preprint arXiv:2304.10703*, 2023.

- <span id="page-11-17"></span>[31] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. ZeRO: Memory optimizations toward training trillion parameter models. In *Proceedings of SC20: The International Conference for High Performance Computing, Networking, Storage and Analysis*, 2020.
- <span id="page-11-16"></span>[32] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. In *arXiv preprint arXiv:1707.06347*, 2017.
- <span id="page-11-1"></span>[33] ByteDance Seed. Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning. *arXiv preprint arXiv:2504.13914*, 2025.
- <span id="page-11-14"></span>[34] Sergio Servantez, Joe Barrow, Kristian Hammond, and Rajiv Jain. Chain of logic: Rule-based reasoning with large language models. *arXiv preprint arXiv:2402.10400*, 2024.
- <span id="page-11-13"></span>[35] Kathrin Seßler, Yao Rong, Emek Gözlüklü, and Enkelejda Kasneci. Benchmarking large language models for math reasoning tasks. *arXiv preprint arXiv:2408.10839*, 2024.
- <span id="page-11-2"></span>[36] Kimi Team. Kimi k1.5: Scaling Reinforcement Learning with LLMs. *arXiv preprint arXiv:2501.12599*, 2025.
- <span id="page-11-8"></span>[37] Qwen Team. Qwen2.5: A party of foundation models, September 2024.
- <span id="page-11-9"></span>[38] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. *arXiv preprint arXiv:2406.01574*, 2024.
- <span id="page-11-5"></span>[39] Zijun Wang, Haoqin Tu, Yuhan Wang, Juncheng Wu, Jieru Mei, Brian R Bartoldson, Bhavya Kailkhura, and Cihang Xie. Star-1: Safer alignment of reasoning llms with 1k data. *arXiv preprint arXiv:2504.01903*, 2025.
- <span id="page-11-4"></span>[40] Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, et al. Medreason: Eliciting factual medical reasoning steps in llms via knowledge graphs. *arXiv preprint arXiv:2504.00993*, 2025.
- <span id="page-11-3"></span>[41] Yunfei Xie, Juncheng Wu, Haoqin Tu, Siwei Yang, Bingchen Zhao, Yongshuo Zong, Qiao Jin, Cihang Xie, and Yuyin Zhou. A preliminary study of o1 in medicine: Are we closer to an ai doctor? *arXiv preprint arXiv:2409.15277*, 2024.
- <span id="page-11-7"></span>[42] Tianci Xue, Ziqi Wang, Zhenhailong Wang, Chi Han, Pengfei Yu, and Heng Ji. Rcot: Detecting and rectifying factual inconsistency in reasoning by reversing chain-of-thought. *arXiv preprint arXiv:2305.11499*, 2023.
- <span id="page-11-0"></span>[43] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. Qwen3 technical report, 2025.
- <span id="page-11-11"></span>[44] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. *arXiv preprint arXiv:2409.12122*, 2024.
- <span id="page-11-18"></span>[45] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. *arXiv preprint arXiv:2502.03387*, 2025.
- <span id="page-11-15"></span>[46] Xinli Yu, Zheng Chen, Yuan Ling, Shujing Dong, Zongyi Liu, and Yanbin Lu. Temporal data meets llm–explainable financial time series forecasting. *arXiv preprint arXiv:2306.11025*, 2023.
- <span id="page-11-12"></span>[47] Hanning Zhang, Jiarui Yao, Chenlu Ye, Wei Xiong, and Tong Zhang. Online-dpo-r1: Unlocking effective reasoning without the ppo overhead. [https://www.notion.so/](https://www.notion.so/Online-DPO-R1-1908b9a70e7b80c3bc83f4cf04b2f175) [Online-DPO-R1-1908b9a70e7b80c3bc83f4cf04b2f175](https://www.notion.so/Online-DPO-R1-1908b9a70e7b80c3bc83f4cf04b2f175). Project blog post, Feb. 2025.
- <span id="page-11-6"></span>[48] Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei Qin, and Lidong Bing. Verify-and-edit: A knowledge-enhanced chain-of-thought framework. *arXiv preprint arXiv:2305.03268*, 2023.
- <span id="page-11-10"></span>[49] Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, et al. Medxpertqa: Benchmarking expert-level medical reasoning and understanding. *arXiv preprint arXiv:2501.18362*, 2025.

### **Technical Appendices and Supplementary Material**

Technical appendices with additional results, figures, graphs and proofs may be submitted with the paper submission before the full submission deadline (see above), or as a separate PDF in the ZIP file below before the supplementary material deadline. There is no page limit for the technical appendices.

#### A Result of ROSCOE Metric

<span id="page-12-0"></span>Table 2: **Evaluation using ROSCOE-SA metrics.** We employ ROSCOE semantic alignment metrics (ROSCOE-SA) including Faithfulness-Step and Informativeness-Step. Despite the knowledge index and information gain differing among the base model, SFT-ed model, and RL-ed model, these metrics yield comparable evaluations for each.

| Base Model                             | SFT     | RL          | MedMCQA                 | MedQA                   | PubMedQA                | MMLU-Pro                | AVG                     |
|----------------------------------------|---------|-------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| Metric: ROSCOE-SA Faithfulness-Step    |         |             |                         |                         |                         |                         |                         |
| Qwen-Base                              | X<br>✓  | X<br>X<br>✓ | 0.744<br>0.742<br>0.742 | 0.776<br>0.777<br>0.778 | 0.842<br>0.843<br>0.842 | 0.762<br>0.760<br>0.760 | 0.781<br>0.781<br>0.781 |
| Metric: ROSCOE-SA Informativeness-Step |         |             |                         |                         |                         |                         |                         |
| Qwen-Base                              | X<br>./ | X<br>X<br>✓ | 0.776<br>0.774<br>0.775 | 0.772<br>0.773<br>0.774 | 0.801<br>0.802<br>0.802 | 0.776<br>0.775<br>0.775 | 0.781<br>0.781<br>0.782 |

### <span id="page-12-1"></span>**B** Experiment Details

#### **B.1** Experimental Settings

**Models.** All experiments in both the medical and math domains are initialized from the universal 7B-parameter base models:

- **Qwen2.5-7B** [37] is an open-weigh language model that takes leading positions in a wide range of tasks. It is pretrained on a large-scale multilingual, multi-domain corpus.
- DeepSeek-R1-Distill-Qwen-7B [11] is a distilled variant of Qwen2.5-7B model. Being trained with supervised fine-tuning (SFT) on DeepSeek-R1 distilled data, this model develops inherent thinking abilities and presents superior performance especially on math and coding tasks.

We select these two models as baselines due to: (1) the strong generalization and robustness of Qwen-7B across domains; (2) their open-source nature, enabling in-depth exploration of training, evaluation, and architecture; and (3) their compatibility for fair comparison on the impact of post-training thinking patterns.

**Training Datasets and Details.** For the medical domain, we fine-tune two LLMs using either SFT or RL on the corresponding data split from *medical-o1* [9]. For models in the math domain, we employ the SFT or RL pipeline from RLHFlow [47] to train the Qwen models, resulting in four model variants on two Qwen models with SFT or RL. We provide detailed medical training data below:

- Supervised Fine-Tuning (SFT): medical-o1-reasoning-SFT contains ~ 40,000 physician-level question-answer pairs. Each record comprises (i) an instruction field (clinical prompt), (ii) a complex\_cot sequence with comprehensive chain-of-thought generated by a multi-round data pipeline, and (iii) a concise response.
- Reinforcement Learning (RL): medical-o1-verifiable-problem offers the same  $\sim 40,000$  clinical cases but retains only the question and a single ground\_truth answer.

The absence of chain-of-thought makes each item a *verifiable prompt*: model outputs are scored by an external medical verifier model with a sparse, sign-based reward (+1 for correct, 0 for incorrect).

The training processes involved reinforcement learning are conducted on 4 NVIDIA H100 GPUs, costing 12 hours for training each model. SFT are conducted on 8 NVIDIA A5000 GPUs, costing 28 hours for training each model.

**Evaluation Datasets** We evaluate our models on two representative domains—*medicine* and *mathematics*—using the public benchmarks listed below.

#### Medical

- MedMCQA [28]  $-\sim$ 194,000 four-option MCQs drawn from India's AIIMS / NEET-PG entrance exams, covering 21 subjects and 2 400 + topics.
- MedQA-USMLE [19]  $\sim$ 12,000 multiple choice questions patterned after USMLE Steps 1–3 in both English and Chinese.
- PubMedQA [20]  $\sim$ 61,000 yes / no / uncertain questions, each paired with a PubMed abstract for context.
- MMLU-Pro (Medical) [38]  $-\sim$ 1,050 ten-option items from the professional-medicine slice of the MMLU Pro suite.
- MedXpertQA [49] 4,460 expert-level free response questions spanning 17 specialties and 11 body systems.

### Mathematics

- AIME 2024[1] 15 open-response problems from the 2024 American Invitational Mathematics Examination.
- MATH500[2] 500 tasks sampled from the original MATH benchmark, covering algebra, geometry, number theory, and combinatorics.
- AMC (10 & 12)[16] multi-year collections of 25-item multiple choice tests widely used in high-school math studies.
- Minerva-Math[22] quantitative reasoning subset introduced with Google's Minerva model, focusing on chain-of-thought solutions.
- USAMO 2025 / Olympiad[3] six proof-based questions from the 2025 USA Mathematical Olympiad for rigorous derivation evaluation.

| Stage         | Key Parameter                  | Value                                                                             |  |  |  |
|---------------|--------------------------------|-----------------------------------------------------------------------------------|--|--|--|
|               | Epochs                         | 3                                                                                 |  |  |  |
| Supervised FT | Effective batch size Optimizer | 128 sequences<br>AdamW[24] ( $\beta_1$ =0.9, $\beta_2$ =0.999, wd 0.01)           |  |  |  |
|               | LR schedule                    | linear warm-up $\rightarrow$ cosine decay, LR <sub>max</sub> = $3 \times 10^{-5}$ |  |  |  |
|               | Max sequence length            | 1 024 tokens                                                                      |  |  |  |
|               | Precision                      | bf16                                                                              |  |  |  |
|               | Parallel engine                | DeepSpeed ZeRO-3 (param&opt off-load CPU)                                         |  |  |  |
| RL (PPO)      | Total episodes                 | 20 000                                                                            |  |  |  |
|               | PPO epochs / update            | 3                                                                                 |  |  |  |
|               | Mini-batches / epoch           | 1                                                                                 |  |  |  |
|               | Rollout batch size             | 4 prompts (per forward pass)                                                      |  |  |  |
|               | Clip range $\epsilon$          | 0.2 (implicit default)                                                            |  |  |  |
|               | KL penalty $\lambda_{\rm KL}$  | 0.03                                                                              |  |  |  |
|               | Target KL                      | 6.0 (early LR reduce)                                                             |  |  |  |
|               | Actor LR                       | $5 \times 10^{-7}$                                                                |  |  |  |
|               | Warm-up ratio                  | 0.05                                                                              |  |  |  |
|               | Grad accumulation              | 16 (effective $bsz = 64$ )                                                        |  |  |  |

Table 3: Principal hyper-parameters for supervised fine-tuning (SFT) and PPO[32] reinforcement learning on the *medical-o1* corpus. Both stages employ bf16 mixed precision and DeepSpeed ZeRO-3[31] on a single 8-GPU node.

#### <span id="page-13-0"></span>**B.2** Prompts for Metric Calculation.

This section provides the detailed prompts employed in the evaluation pipeline (Sec. 3).

```
Prompt for Medical Reasoning Decomposition
messages = [ """ You are a helpful, pattern-following medical assistant. Given a paragraph of medical reasoning, you need to decompose the reasoning into individual
steps. Each step should contain four parts:
1. "id" : a unique number for the step
2. "planning" : a brief description of the main idea of the step."planning" parts from all steps
should form a coherent and logical sequence. 3. "action" : a detailed description of the actions taken in this step. "action" is taken based on
the "planning" part of the step, and should contain specific medical knowledge or procedures. 4. "step_text" : sentence(s) from the input reasoning paragraph that corresponds to this step. ### Output Format:
Strictly follow the JSON structure below.
```json
{{"Steps: [
 {{"id" : 1, "planning" : "Planning for step 1", "action" : "Action for step 1", "step_text" : "Corresponding sentence(s) from the input reasoning paragraph"}}, {{"id" : 2, "planning" : "Planning for step 2", "action" : "Action for step 2", "step_text" : "Corresponding sentence(s) from the input reasoning paragraph"}},
 ...]}}
### Input Reasoning paragraph: {…………}
]
```

<span id="page-14-0"></span>Figure 10: Medical Reasoning Decomposition Prompt Full prompt employed to decompose the model's reasoning into reasoning steps using gpt4o.

```
Prompt for Math Reasoning Decomposition
messages = [ """ You are a helpful, pattern-following medical assistant. Given a paragraph of reaasoning addressing a math question,you need to decompose the
reasoning into individual steps. Each step should contain two parts:
1. "id" : a unique number for the step
2. "step" : coresponding sentence(s) from the input reasoning paragraph, each reasoning step
should describe a logical step in the question solution process. ### Output Format:
Strictly follow the structurebelow, and do not add any other content.
```text
###step: Corresponding sentence(s) 1streasoning step, from the input reasoning paragraph, ###step: Corresponding sentence(s) 2nd reasoning step, from the input reasoning paragraph, ###step: Corresponding sentence(s) 3rd reasoning step, from the input reasoning paragraph, ###step: Corresponding sentence(s) 4th reasoning step, from the input reasoning paragraph,
....
### Input Reasoning paragraph: {…………}
]
```

<span id="page-14-1"></span>Figure 11: Math Reasoning Decomposition Prompt Full prompt employed to decompose the model's reasoning into reasoning steps using gpt4o.

### Prompt for Medical Knowledge Retrieve messages = [ """ You are a helpful, pattern-following medical assistant. Given a medical content. You need to generate a concise query for provided content. If there isno specific medical knowledge in the input content (such as a recognize the symptoms, analysis the known information, description of the patient condition, etc.), DO NOT GENERATE QUESTION, set the query as "None". Else, if there isspecific medicalknowledge in the input content (such as inference and diagnosis based on some medical knowledge), you need to generate a question that is concise and specific to the medical knowledge in the content. ### Output Format: You should strictly follow the JSON structure below. ```json {{"query" : "query for the medical",}} ### Example: …… ### Input Knowledge point: {…………} Output: """]

Figure 12: Medical Knowledge Retrieve Prompt Full prompt employed to identify the knowledge pertinent to the reasoning step using gpt4o.

```
Prompt for Planning Consistent
messages = [ """ You are a helpful, pattern-following medical assistant. Given two listsof planning lists, you need to check if the two lists are consistent with each
other. Two lists are considered consistent if they form a similar and coherent sequence of steps, and
the steps in the two listsare in the same order. Your response must be one word: 'True' or 'False'.
If the two listsare consistent, please respond with 'True'.
If the two listsare not consistent,please respond with 'False'. Planning list 1: {…………}
Planning list 2: {…………}
]
```

Figure 13: Planning Consistency Evaluation Prompt Full prompt employed to evaluate whether the reasoning step is consistent with the retrieved facts using gpt4o.

# C More experiment results

### C.1 Data Filtering for Medical Reasoning

Previous studies in general reasoning domains, such as mathematics, have demonstrated that filtering training data based on reasoning quality or problem difficulty can achieve comparable or superior performance relative to training on complete datasets, while significantly reducing computational cost [\[25,](#page-10-17) [45\]](#page-11-18).

We extend this investigation to the medical domain by applying two filtering regimes to the medical-o1 SFT corpus. As presented in Table [4:](#page-16-0) Quality-based filtering retains only those examples whose CoT reasoning steps have been verified for resulting in the correct answer. This strategy preserves nearly all performance (Accuracy: 52.62% vs. 52.79%; ∆I: 2.341 vs. 2.347) while discarding training tokens. Difficulty-based filtering selects examples based on question difficulty. In contrast to the results in general reasoning, this approach degrades medical performance (Accuracy: 49.28%; ∆I: 2.271), indicating that factual correctness is paramount in knowledge-intensive settings.

<span id="page-16-0"></span>Table 4: Comparison between different SFT data filtering strategies. Quality filtering achieves comparable performance in terms of accuracy and information gain as full data, while reducing the training cost.

| Data Filtering MedMCQA MedQA PubMedQA MMLU-Pro MedXpert |                         |                         |                         |                         |                         | Avg                     |
|---------------------------------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| Metric: Accuracy                                        |                         |                         |                         |                         |                         |                         |
| None<br>Difficulty<br>Quality                           | 55.48<br>52.43<br>55.33 | 62.61<br>57.63<br>63.34 | 71.17<br>65.10<br>71.03 | 59.11<br>57.39<br>58.70 | 15.57<br>13.83<br>14.70 | 52.79<br>49.28<br>52.62 |
| Metric: Info Gain                                       |                         |                         |                         |                         |                         |                         |
| None<br>Difficulty<br>Quality                           | 9.291<br>9.034<br>9.289 | 0.157<br>0.155<br>0.154 | 0.192<br>0.185<br>0.191 | 1.785<br>1.680<br>1.759 | 0.312<br>0.302<br>0.315 | 2.347<br>2.271<br>2.341 |

In both the general and medical domains, quality-based filtering delivers the best cost-to-performance ratio. But unlike mathematics, where difficulty-based filtering can be neutral or beneficial, medical reasoning degrades when filtered purely by difficulty, underscoring the primacy of knowledge correctness in medical contexts.