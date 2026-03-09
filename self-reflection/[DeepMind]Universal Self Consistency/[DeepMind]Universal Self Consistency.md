# UNIVERSAL SELF-CONSISTENCY FOR LARGE LAN-GUAGE MODEL GENERATION

Xinyun Chen <sup>∗</sup> Renat Aksitov \* Uri Alon Jie Ren Kefan Xiao Pengcheng Yin Sushant Prakash Charles Sutton Xuezhi Wang Denny Zhou

Google

### ABSTRACT

Self-consistency with chain-of-thought prompting (CoT) has demonstrated remarkable performance gains on various challenging tasks, by utilizing multiple reasoning paths sampled from large language models (LLMs). However, selfconsistency relies on the answer extraction process to aggregate multiple solutions, which is not applicable to free-form answers. In this work, we propose Universal Self-Consistency (USC), which leverages LLMs themselves to select the most consistent answer among multiple candidates. We evaluate USC on a variety of benchmarks, including mathematical reasoning, code generation, long-context summarization, and open-ended question answering. On open-ended generation tasks where the original self-consistency method is not applicable, USC effectively utilizes multiple samples and improves the performance. For mathematical reasoning, USC matches the standard self-consistency performance without requiring the answer formats to be similar. Finally, without access to execution results, USC also matches the execution-based voting performance on code generation.

## 1 INTRODUCTION

Large language models (LLMs) have accomplished significant breakthroughs in a wide variety of domains, including mathematical reasoning [\(Cobbe et al., 2021;](#page-9-0) [Wei et al., 2022;](#page-11-0) [Lewkowycz](#page-10-0) [et al., 2022\)](#page-10-0), code generation [\(Chen et al., 2021;](#page-9-1) [Austin et al., 2021;](#page-9-2) [Li et al., 2022\)](#page-10-1), and other text generation tasks [\(Bubeck et al., 2023;](#page-9-3) [Anil et al., 2023;](#page-9-4) [Touvron et al., 2023\)](#page-11-1). Despite the rapid progress, the LLM-generated responses are still prone to errors when they get long. A long line of efforts have been devoted to improve the output quality by sampling multiple model responses and then selecting the final output based on certain criteria. For example, prior works have trained neural networks to rerank model outputs [\(Cobbe et al., 2021;](#page-9-0) [Li et al., 2023b;](#page-10-2) [Ni et al., 2023;](#page-10-3) [Yin & Neubig,](#page-11-2) [2019;](#page-11-2) [Zeng et al., 2022\)](#page-11-3), and more recent works investigate using LLMs to score the responses [\(Fu](#page-9-5) [et al., 2023;](#page-9-5) [Liu et al., 2023;](#page-10-4) [Wang et al., 2023a\)](#page-11-4).

In this work, we consider the *consistency* among model responses as the criterion to select the model output, a generic metric that has enabled huge performance leaps in reasoning [\(Wang et al., 2022\)](#page-11-5) and code generation [\(Li et al., 2022;](#page-10-1) [Shi et al., 2022\)](#page-11-6). In particular, self-consistency [\(Wang et al., 2022\)](#page-11-5) with chain-of-thought prompting [\(Wei et al., 2022\)](#page-11-0) boosts the performance on various benchmarks, by marginalizing latent reasoning paths through sampling which leads to select the final answer as the most common one. However, self-consistency can only be applied to tasks where the final answer can be aggregated via exact match, e.g., a single number for math problems.

To address this major limitation of self-consistency, we propose Universal Self-Consistency (USC) to support various applications, especially free-form generation tasks. Specifically, given multiple candidate responses, USC simply calls the LLM to select the most consistent response among them as the final output. Thus, USC eliminates the need of designing an answer extraction process, and is applicable to tasks with free-form answers. Although prior works have revealed weaknesses of LLMs for response selection, such as position bias [\(Wang et al., 2023b;](#page-11-7) [Zheng et al., 2023b\)](#page-12-0) and incorrectly judging the answer correctness [\(Huang et al., 2023b;](#page-10-5) [Gou et al., 2023\)](#page-9-6), intuitively, assessing the consistency among candidate answers is easier than measuring and comparing the answer quality.

<sup>∗</sup>Equal contribution.

We evaluate universal self-consistency on a wide range of tasks, including mathematical reasoning, code generation, long-context summarization, and open-ended question answering. On GSM8K [\(Cobbe et al., 2021\)](#page-9-0) and MATH [\(Hendrycks et al., 2021\)](#page-9-7) benchmarks for math problem solving, USC generally matches the performance of the standard self-consistency. On programming tasks including text-to-SQL generation [\(Li et al., 2023a\)](#page-10-6) and Python code generation [\(Yin et al.,](#page-11-8) [2023\)](#page-11-8), USC matches the performance of execution-based consistency [\(Li et al., 2022;](#page-10-1) [Shi et al., 2022\)](#page-11-6), while USC does not require execution results to aggregate over candidate programs. Finally, USC also improves the performance for open-ended question answering [\(Lin et al., 2021\)](#page-10-7) and long-context summarization [\(Huang et al., 2021;](#page-10-8) [Chen et al., 2022b\)](#page-9-8), where the standard self-consistency is not applicable. In addition to the performance gain, our evaluation also demonstrates that USC outputs highly match those of the standard self-consistency when the comparison is applicable, while it is robust to the ordering of candidate responses.

## 2 BACKGROUND: SELF-CONSISTENCY

Self-consistency [\(Wang et al., 2022\)](#page-11-5) augments chain-of-thought prompting [\(Wei et al., 2022\)](#page-11-0) by sampling multiple reasoning chains and then taking a majority vote on the final answer set. The intuition is that sometimes the greedily decoded reasoning process might not be the optimal one, hence it makes more sense to sample a diverse set of reasoning chains, and if some of them lead to the same answer, then we have a higher confidence that this consistent answer is the correct one. It has been shown that self-consistency improves the greedy chain-of-thought prompting by a large margin on a wide set of reasoning tasks.

Besides question answering tasks, consistency-based answer selection has also been applied to code generation [\(Shi et al., 2022;](#page-11-6) [Li et al., 2022;](#page-10-1) [Chen et al., 2019\)](#page-9-9), which requires code execution. Specifically, we first execute all predicted programs on the given inputs, then programs with the same execution outputs are clustered together, assuming that they are semantically equivalent. Finally, we select the program belonging to the largest cluster as the final prediction. When the program inputs given in the task description are insufficient to distinguish between different predictions, this execution-based code selection is also often accompanied with a test case generation process to better examine the consistency [\(Li et al., 2022;](#page-10-1) [Chen et al., 2022a;](#page-9-10) [Huang et al., 2023a\)](#page-10-9).

Despite the remarkable improvement, self-consistency is only applicable to problems with a unique and closed-form answer, e.g., when the final answer consists of a single number, because a majority vote needs to be taken over the final answer set. This significant requirement poses a challenge for tasks that require open-ended generations, such as summarization, creative writing, and open-ended question answering.

# 3 UNIVERSAL SELF-CONSISTENCY

<span id="page-1-0"></span>![](_page_1_Figure_7.jpeg)

Figure 1: Overview of the Universal Self-Consistency workflow.

We present the overall workflow of universal self-consistency (USC) in Figure [1,](#page-1-0) which utilizes LLMs to enable self-consistency for a wide variety of tasks, especially free-form text generation. First, we sample multiple responses with the large language model. Afterward, to select one model response as the final answer, we concatenate all responses together, and then construct a prompt with an instruction asking the language model to select the most consistent response. In this way, USC obviates the necessity of counting the exact answer frequency as in the standard self-consistency, and relies on the LLM's own ability to measure the consistency among different responses. Although prior works show that LLMs sometimes have trouble evaluating the prediction correctness [\(Huang](#page-10-5)

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Figure 2: Examples of Universal Self-Consistency for answer selection from responses of diverse formats: (a) mathematical reasoning; and (b) open-ended question answering. Note that for the given open-ended question, the final answer is an entity list, where no two responses share the same predictions. Still, the LLM correctly selects the response where the individual entities in the predicted list appear most frequently in the candidate responses.

[et al., 2023b;](#page-10-5) [Gou et al., 2023\)](#page-9-6), especially for reasoning problems, empirically we observe that LLMs are generally able to examine the response consistency across multiple tasks.

Consistency assessment with LLMs offers more flexibility for free-form generation. Figure [2](#page-2-0) demonstrates example tasks where different consistency criteria are beneficial for response selection. Specifically, Figure [2a](#page-2-0) shows different model responses for a math problem, where the output formats are diverse and thus makes it challenging for rule-based methods to extract answers. Nonetheless, assuming that the final answers are correctly extracted, the consistency criterion still follows the standard self-consistency on mathematical reasoning, which is based on the exact match of the final answers represented as single numerical values. On the other hand, Figure [2b](#page-2-0) shows an example question where the final answer is an entity list. Despite that there is no response that is consistent with others based on the exact match, the LLM selects the response where each of the predicted entities appears most frequently among the candidate outputs. In Section [4,](#page-3-0) we further show that LLM can also examine the consistency among responses beyond the question answering tasks, including code generation without access to the execution outputs, and long-context summarization.

# <span id="page-3-0"></span>4 EXPERIMENTS

### 4.1 EVALUATION SETUP

Benchmarks. We evaluate USC on the following variety of tasks:

- *Mathematical reasoning benchmarks*, including GSM8K [\(Cobbe et al., 2021\)](#page-9-0), a dataset of 8,500 grade school math word problems, and MATH [\(Hendrycks et al., 2021\)](#page-9-7), a dataset of 12,500 challenging mathematics problems from high school competitions.
- *Code generation benchmarks*, including BIRD-SQL dataset [\(Li et al., 2023a\)](#page-10-6) for text-to-SQL generation, and ARCADE dataset [\(Yin et al., 2023\)](#page-11-8) for Python code generation in data science notebooks.
- *Long-context summarization*, including the GovReport and SummScreen benchmarks from ZeroSCROLLS [\(Shaham et al., 2023\)](#page-10-10). In GovReport [\(Huang et al., 2021\)](#page-10-8), each input is a document containing ∼7,900 words on average, and the reference output is an expert-written executive summary with ∼500 words. In SummScreen [\(Chen et al., 2022b\)](#page-9-8), every input is a transcript of a TV show episode with ∼5,600 words, and each reference output is a ∼100 words human-written recap of the episode. We follow [Shaham et al.](#page-10-10) [\(2023\)](#page-10-10) and measure ROUGE 1, ROUGE 2, and ROUGE-Lsum which measure n-gram overlap with the reference summary, and we also measure BERTScore F1 [\(Zhang et al., 2019\)](#page-11-9).
- *TruthfulQA [\(Lin et al., 2021\)](#page-10-7) benchmark* for open-ended question answering, which contains 817 questions to test model's ability in generating truthful answers. To evaluate the answer's quality, we use the GPT-judge and GPT-info, which are GPT-3 models fine-tuned on human feedback data, provided by [Lin et al.](#page-10-7) [\(2021\)](#page-10-7). GPT-judge model outputs a binary rating for truthfulness, and GPT-info model outpus a binary rating for informativeness. It is shown that the GPT-3 models have higher accuracy in predicting human judgement than the automatic metrics ROUGE, BLEU, BLEURT.

Decoding schemes. We compare USC to the following decoding schemes:

- *Greedy decoding* generates a single answer with the temperature 0.
- *Random* selects one answer randomly from multiple samples with temperature > 0.
- *SC* [\(Wang et al., 2022\)](#page-11-5) is the standard self-consistency decoding with answer extraction. We evaluate SC whenever applicable; for example, on reasoning benchmarks where the final answers can be compared through exact match.

To enable a fair comparison, for sampling schemes (i.e., except greedy decoding), we always select the final answer from the same set of initial model responses. For code generation, we compare our approach to execution-based self-consistency [\(Shi et al., 2022;](#page-11-6) [Li et al., 2022;](#page-10-1) [Chen et al., 2019\)](#page-9-9), where we select the code with the most common execution result. Both USC and execution-based self-consistency first filter out syntactically invalid candidate programs, and then perform the voting over the remaining ones. For ARCADE benchmark, we also evaluate a variant of the execution-based self-consistency with fuzzy matching as described in [Yin et al.](#page-11-8) [\(2023\)](#page-11-8), which implements a set of heuristics to determine whether the execution outputs of two programs are equivalent when they are not exact match.

Implementation details. We conduct experiments using instruction-tuned PaLM 2-L [\(Anil et al.,](#page-9-4) [2023\)](#page-9-4) and gpt-3.5-turbo models. Unless otherwise specified, the LLM generates 8 initial samples for both SC and USC. For mathematical reasoning, summarization and the ARCADE benchmark for Python code generation, the initial samples are generated with zero-shot prompting, thus the output formats are diverse. For BIRD-SQL, we used the 1-shot chain-of-thought prompt in [Li](#page-10-6) [et al.](#page-10-6) [\(2023a\)](#page-10-6), which improves the performance. We also utilized a one-shot prompt for TruthfulQA to improve the quality of candidate responses. We set the temperature to be 0.6 for PaLM 2-L, and 1.0 for gpt-3.5-turbo.

## 4.2 MAIN RESULTS

Mathematical reasoning. For mathematical reasoning benchmarks, we compare USC against the standard self-consistency in Table [1.](#page-4-0) For the standard self-consistency, we employ a regular expression matching to extract the final answer on GSM8K, and re-use the answer parsing code from [\(Zheng et al., 2023a\)](#page-12-1) for MATH. Overall, USC consistently improves over the greedy decoding and random selection, and the performance is generally comparable to the standard self-consistency, which USC does not need answer parsing to perform the voting.

Code generation. Table [2](#page-4-1) presents the results on BIRD-SQL and ARCADE respectively. On BIRD-SQL, besides the execution accuracy, we follow [\(Li et al., 2023a\)](#page-10-6) to also evaluate the valid efficiency score, which measures the efficiency of the generated SQL queries. We show that USC matches the execution-based self-consistency performance on both benchmarks, while USC does not utilize code execution to perform the voting.

Summarization. Results for the summarization benchmarks are shown in Table [3.](#page-5-0) Since the generated summaries are in free-form, the standard self-consistency is not applicable. In GovReport, USC consistently improves over the baselines across all metrics. In Section [4.3,](#page-5-1) we further show that asking the model to choose the *most detailed* summary results in more performance gain.

TruthfulQA. Table [4](#page-5-2) presents results on TruthfulQA, where SC is also not directly applicable because the generated answers are in free-form. Comparing with greedy decoding and random selection, USC-based answers have the highest truthfulness with both PaLM 2-L and gpt-3.5 turbo. For informativeness which is considered as a secondary objective, USC-based answers have the highest score on PaLM 2-L and the second highest score (0.1 lower than the highest) on gpt-3.5-turbo. Considering that GPT-judge and GPT-info models have generally 90-95% validation accuracy on rating prediction [\(Lin et al., 2021\)](#page-10-7), the 0.1 difference is not considered significant.

<span id="page-4-0"></span>Table 1: Accuracy on mathematical reasoning benchmarks. USC and SC consistently improve over the greedy decoding and random selection. USC performance is generally comparable to SC.

| Model         | Approach                                                   |                              | MATH                         |
|---------------|------------------------------------------------------------|------------------------------|------------------------------|
| PaLM 2-L      | Greedy decoding<br>Random<br>SC (Wang et al., 2022)<br>USC | 85.7<br>82.9<br>90.4<br>90.2 | 30.8<br>28.0<br>37.9<br>37.4 |
| gpt-3.5-turbo | Greedy decoding<br>Random<br>SC<br>USC                     |                              | 33.2<br>26.3<br>38.0<br>38.1 |

Table 2: Accuracy on code generation benchmarks with gpt-3.5-turbo.

<span id="page-4-1"></span>

| Dataset  | Approach               | Execution Accuracy | Valid Efficiency Score |  |
|----------|------------------------|--------------------|------------------------|--|
| BIRD-SQL | Greedy decoding        | 42.4               | 44.4                   |  |
|          | Random                 | 41.9               | 44.0                   |  |
|          | SC-Exec                | 45.6               | 48.1                   |  |
|          | USC                    | 45.5               | 48.8                   |  |
| ARCADE   | Greedy decoding        | 26.0               |                        |  |
|          | Random                 | 26.8               |                        |  |
|          | SC-Exec (strict match) | 29.8               | N/A                    |  |
|          | SC-Exec (fuzzy match)  | 30.3               |                        |  |
|          | USC                    | 30.1               |                        |  |

<span id="page-5-0"></span>Table 3: Results on long-context summarization benchmarks with PaLM 2-L. Since the outputs are in free-form, the standard self-consistency is not applicable. USC consistently improves over the baselines on summary quality.

| Dataset                                       | Approach                         |                      | ROUGE-2              | ROUGE-Lsum           | BERTScore            |
|-----------------------------------------------|----------------------------------|----------------------|----------------------|----------------------|----------------------|
| Greedy decoding<br>GovReport<br>Random<br>USC |                                  | 38.8<br>38.5<br>40.2 | 16.9<br>16.9<br>17.4 | 33.8<br>33.6<br>35.1 | 62.7<br>62.6<br>62.8 |
| SummScreen                                    | Greedy decoding<br>Random<br>USC | 30.6<br>30.2<br>31.7 | 7.5<br>7.3<br>7.8    | 19.1<br>19.0<br>19.8 | 58.7<br>58.6<br>58.3 |

<span id="page-5-2"></span>Table 4: Accuracy on the TruthfulQA benchmark. Since the answer is in free-form, the standard self-consistency is not applicable. USC overall has the highest truthfulness and informativeness over the baselines.

| Model         | Approach        | GPT-judge | GPT-info |
|---------------|-----------------|-----------|----------|
| PaLM 2-L      | Greedy decoding | 62.1      | 95.1     |
|               | Random          | 62.9      | 94.6     |
|               | USC             | 67.7      | 99.0     |
| gpt-3.5-turbo | Greedy decoding | 79.8      | 99.7     |
|               | Random          | 80.6      | 99.3     |
|               | USC             | 82.5      | 99.6     |

#### <span id="page-5-1"></span>4.3 ABLATIONS

Effect of response ordering. Prior works have shown that large language models can be affected by the order of candidate responses when used to evaluate their quality [\(Wang et al., 2023b;](#page-11-7) [Zheng](#page-12-0) [et al., 2023b\)](#page-12-0). We examine the effect of response ordering by performing USC with 5 different random orders when concatenating all responses, and calculate the mean and standard deviation of the task results. From Table [5,](#page-5-3) we observe that the overall model performance remains similar with different response orders, suggesting the effect of response order is minimal.

<span id="page-5-3"></span>Table 5: USC performance with random shuffling of original responses using PaLM 2-L. The mean and standard deviation are computed with 5 runs.

| (a)<br>(b)    |                      | (c)                     |                      |                      |                       |                      |
|---------------|----------------------|-------------------------|----------------------|----------------------|-----------------------|----------------------|
| Dataset       | Acc                  | Dataset                 | ROUGE-1              | ROUGE-Lsum           | Metric                | TruthfulQA           |
| GSM8K<br>MATH | 89.7±0.3<br>37.3±0.2 | SummScreen<br>GovReport | 31.6±0.3<br>40.0±0.1 | 19.5±0.2<br>34.9±0.2 | GPT-judge<br>GPT-info | 68.3±0.6<br>99.0±0.1 |

Different number of responses. Next, we examine the effect of using different numbers of responses in USC. As shown in Figure [3,](#page-6-0) USC consistently benefits from more samples on TruthFulQA and BIRD-SQL. However, USC does not further improve the performance on SummScreen after 5 samples, and the accuracy on GSM8K decreases with 16 samples. This can be due to the weakness in long-context understanding when the prompt contains more candidate responses, and the imperfect counting ability of LLMs. Nevertheless, we consider utilizing a few samples (e.g., 8) a sweet spot to balance the task accuracy and compute cost, in which case USC reliably improves the performance across the board. In Section [4.4,](#page-6-1) we further compare the predictions from USC and SC to understand how using more candidate responses affects the results.

Criteria for response selection. One advantage of USC is its generality: the same criteria can be applied to various tasks, without any task-specific knowledge. Nonetheless, a minor task-specific

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

![](_page_6_Figure_2.jpeg)

- (a) Results on open-ended generation.
- (b) Results on GSM8K and BIRD-SQL. The top numbers are USC accuracies, and the bottom numbers are the differences to SC accuracies.

Figure 3: USC results with different number of samples.

adaptation of the response selection instruction can further boost USC over the generic prompts. For example, Table [6](#page-6-2) shows that asking the LLM to choose the most *detailed* response (rather than the most *consistent* one) results in gains of about 2 ROUGE-1 and ROUGE-Lsum points.

<span id="page-6-2"></span>Table 6: Ablation on the response selection criterion on long-context summarization benchmarks with PaLM 2-L.

| Dataset                                 | Approach                   | ROUGE-1      | ROUGE-2      | ROUGE-Lsum   | BERTScore    |
|-----------------------------------------|----------------------------|--------------|--------------|--------------|--------------|
| USC<br>GovReport<br>USC – most detailed |                            | 40.2<br>42.4 | 17.4<br>18.2 | 35.1<br>36.9 | 62.8<br>63.2 |
| SummScreen                              | USC<br>USC – most detailed | 31.7<br>33.0 | 7.8<br>7.9   | 19.8<br>22.0 | 58.3<br>58.3 |

#### <span id="page-6-1"></span>4.4 DISCUSSION: HOW WELL DOES USC MATCH SC SELECTION?

<span id="page-6-3"></span>![](_page_6_Figure_10.jpeg)

![](_page_6_Figure_11.jpeg)

Figure 4: Comparison of selections made by USC versus SC with PaLM 2-L. k denotes the number of candidate responses for selection. "Tied votes" represents the case where the USC and SC select different responses, but both have the maximum votes.

We have demonstrated that on tasks where the standard self-consistency is applicable, USC and SC achieve comparable overall performance with 8 samples; however, USC fails to further improve the GSM8K performance with 16 samples. In this section, we look closer into the relationship between USC and SC, specifically how well is the alignment between their selected responses.

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

![](_page_7_Figure_2.jpeg)

Figure 5: Accuracy distribution when USC selection doesn't match SC.

Figure [4](#page-6-3) presents a breakdown analysis of USC predictions on mathematical reasoning benchmarks with 8 and 16 candidate responses, and Figure [5](#page-7-0) further compares the performance of USC and SC when they select different responses. We observe that:

- The voting ties constitute a notable portion to the selection differences between USC and SC, especially with 8 candidate responses. Specifically, among all responses with the maximum votes, SC always selects the one with the smallest index, while USC can pick up alternative ones based on the response format.
- The match ratio between USC and SC consistently surpasses their own task accuracies, which shows that the consistency criterion is easier to measure than the answer correctness.
- Shifting from 8 to 16 samples, the USC-SC match ratio reduces, suggesting that USC behaves as an imperfect approximation of SC. However, the difference in response selection does not always lead to the performance decrease, as USC sometimes selects the correct response when SC fails.

## 5 RELATED WORK

Response reranking and selection for language models. Reranking is a common method to improve the generation quality in language models by sampling multiple outputs and applying a post-hoc criterion to rank them, which often requires an additional trained ranker and sometimes additional human labeled data. For example, [Cobbe et al.](#page-9-0) [\(2021\)](#page-9-0) use human labels to train a ranking model to verify whether each generated response is correct or not, and [Shen et al.](#page-10-11) [\(2021\)](#page-10-11) jointly train a generator and a ranker to improve performance for math tasks. Instead of training response generators and rankers as separate models, [Thoppilan et al.](#page-11-10) [\(2022\)](#page-11-10) finetune the dialog model to also predict the ratings of candidate responses with human-annotated judgements. For code generation, various reranker models have been designed [\(Ni et al., 2023;](#page-10-3) [Yin & Neubig, 2019;](#page-11-2) [Zeng et al., 2022\)](#page-11-3), which typically utilize execution results and language-specific syntactic features to improve the ranking performance. In contrast with these prior works, USC does not require any additional labeled training data nor an external reranking model: the LLM that generated the initial outputs is the same one that selects the final answer.

Several consistency-based response selection approaches have been proposed in the literature, which typically include a voting procedure to select the most common response [\(Wang et al., 2022;](#page-11-5) [Zhou](#page-12-2) [et al., 2022;](#page-12-2) [Wightman et al., 2023;](#page-11-11) [Yue et al., 2023;](#page-11-12) [Bertsch et al., 2023\)](#page-9-11). Self-consistency [\(Wang](#page-11-5) [et al., 2022\)](#page-11-5) shows that with multiple responses generated for the same task, selecting the reasoning path leading to the most common final answer improves the chain-of-thought reasoning performance. The candidate responses can also come from different prompt variants corresponding to the same problem [\(Zhou et al., 2022;](#page-12-2) [Wightman et al., 2023;](#page-11-11) [Yue et al., 2023\)](#page-11-12). To measure the pairwise similarity between candidate responses for open-ended generation tasks, [Jain et al.](#page-10-12) [\(2023\)](#page-10-12) propose the n-gram consistency score, and the consistency score for each response is computed as the sum of the pairwise similarity scores. For code generation, the consistency measurement is typically based on code execution, where the candidate program with the most common execution outputs is selected [\(Shi et al., 2022;](#page-11-6) [Li et al., 2022;](#page-10-1) [Chen et al., 2019\)](#page-9-9). Besides the consistency of code execution, other works also examine the consistency between the code and the specification [\(Min](#page-10-13)

[et al., 2023\)](#page-10-13), and utilize it for reranking [\(Zhang et al., 2023a;](#page-12-3) [Huang et al., 2023a\)](#page-10-9). In this work, we directly instruct the LLM to perform consistency-based selection without an explicit definition of the pairwise similarity, and we demonsrate its applicability to a wide range of tasks.

Response improvement with multiple candidates. Some recent works demonstrate that the LLM can improve its prediction output on top of the candidate responses. [Yang et al.](#page-11-13) [\(2023\)](#page-11-13) show that given a trajectory of previously generated solutions, the LLM can iteratively produce better solutions for an optimization task, and they demonstrate the effectiveness of this LLM-based optimization process for prompt optimization and several classic mathematical optimization tasks. Other works aggregate multiple reasoning chains and prompts the LLM to generate a better final response, which shows performance improvement on multi-hop question answering [\(Yoran et al., 2023\)](#page-11-14) and medical question answering [\(Singhal et al., 2023\)](#page-11-15). Instead of asking the LLM to generate a better response, USC focuses on response selection, as the candidate responses usually already contain high-quality solutions to the underlying tasks. Meanwhile, performing the consistency-based selection is generally an easier task than improving the answer correctness, and we demonstrate that USC properly utilizes multiple responses to improve the performance across different tasks.

Large language models for response evaluation. The underlying assumption in our work is that LLMs are reflective enough to evaluate the consistency between multiple self-generated outputs. This assumption is related to recent findings which had shown that large language models can also be used for evaluating model-generated texts [\(Bubeck et al., 2023;](#page-9-3) [Fu et al., 2023;](#page-9-5) [Wang et al., 2023a;](#page-11-4) [Zhang](#page-12-4) [et al., 2023b\)](#page-12-4). LLM-based evaluators have demonstrated some promising results, e.g., they can be used to evaluate natural language generations without human references, but some work has also shown that they might not correlate very well with human judgements and sometimes exhibit bias towards model-generated texts [\(Bubeck et al., 2023;](#page-9-3) [Liu et al., 2023\)](#page-10-4). Another line of work utilizes the prediction probability of the LLM to measure the quality of multiple choices [\(Ren et al., 2023;](#page-10-14) [Adiwardana et al., 2020\)](#page-9-12), and [Lin et al.](#page-10-15) [\(2022\)](#page-10-15) show promising results on arithmetic tasks where they prompt the LLM to directly output the level of confidence for its response. In this work, we show that LLMs not only can serve as evaluators, they can also improve their own output by sampling multiple responses and evaluating the consistency between them.

## <span id="page-8-0"></span>6 LIMITATIONS AND FUTURE WORK

Despite that USC supports open-ended generation tasks and generally achieves comparable performance in those domains where the standard self-consistency can be applied, our current USC implementation has its own limitations compared to the extraction-based self-consistency approach.

First, while self-consistency can be applied to an arbitrary number of samples as long as the final answers can be extracted, the number of samples supported by USC is bounded by the context length of the underlying LLM. That said, to seek a balance between the task performance and the sampling cost, in practice the number of generated samples per task is not prohibitively large, thus the context length is generally sufficient to make best use of the samples.

Second, the voting mechanism in self-consistency inherently offers a measure of confidence or uncertainty for each response [\(Wang et al., 2022\)](#page-11-5). However, universal self-consistency has not yet been developed to include the confidence estimation. We consider developing a calibration mechanism for USC as future work, where we can leverage the LLM to perform output clustering and pairwise self-consistency.

Also, USC requires an additional LLM query by design, which incurs additional inference costs. Given that our USC prompt only requires the LLM to generate a response index corresponding to the final answer, the USC output length is much shorter than any individual candidate response to select from. To further reduce the cost, one direction is to use a light-weight language model to conduct USC, and optimizes its efficiency regarding long-context encoding.

Finally, one common limitation of both the standard self-consistency and USC is about the consistencybased selection criterion. Specifically, although consistency is a generic and effective criterion, the most consistent response is not necessarily the best one. We observe that there is still a notable gap to oracle scores where we assume the access to an oracle reranker that always selects the best response, and we present the full results in Appendix [A.](#page-12-5) In Section [4.3](#page-5-1) we demonstrate that we can

design task-specific criteria to further improve the performance, and we consider refining the USC framework to further close the gap to the oracle performance as future work.

# 7 CONCLUSION

In this work, we presented Universal Self-Consistency (USC), which extends the standard selfconsistency to support free-form generation tasks. USC notably boosts the performance in diverse applications, and performs on par with the standard self-consistency on those tasks where answer extraction is feasible for voting. Besides addressing the limitations discussed in Section [6,](#page-8-0) we also consider mitigating the position bias and improving long-context understanding of LLMs as important future work that can further enhance the effectiveness and robustness of the USC scheme.

# REFERENCES

- <span id="page-9-12"></span>Daniel Adiwardana, Minh-Thang Luong, David R So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu, et al. Towards a human-like opendomain chatbot. *arXiv preprint arXiv:2001.09977*, 2020.
- <span id="page-9-4"></span>Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. *arXiv preprint arXiv:2305.10403*, 2023.
- <span id="page-9-2"></span>Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.
- <span id="page-9-11"></span>Amanda Bertsch, Alex Xie, Graham Neubig, and Matthew R Gormley. It's mbr all the way down: Modern generation techniques through the lens of minimum bayes risk. *arXiv preprint arXiv:2310.01387*, 2023.
- <span id="page-9-3"></span>Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023.
- <span id="page-9-10"></span>Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. *arXiv preprint arXiv:2207.10397*, 2022a.
- <span id="page-9-1"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.
- <span id="page-9-8"></span>Mingda Chen, Zewei Chu, Sam Wiseman, and Kevin Gimpel. Summscreen: A dataset for abstractive screenplay summarization. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 8602–8615, 2022b.
- <span id="page-9-9"></span>Xinyun Chen, Chang Liu, and Dawn Song. Execution-guided neural program synthesis. In *International Conference on Learning Representations*, 2019.
- <span id="page-9-0"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- <span id="page-9-5"></span>Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire, 2023.
- <span id="page-9-6"></span>Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. *arXiv preprint arXiv:2305.11738*, 2023.
- <span id="page-9-7"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.

- <span id="page-10-9"></span>Baizhou Huang, Shuai Lu, Weizhu Chen, Xiaojun Wan, and Nan Duan. Enhancing large language models in coding through multi-perspective self-consistency. *arXiv preprint arXiv:2309.17272*, 2023a.
- <span id="page-10-5"></span>Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. *arXiv preprint arXiv:2310.01798*, 2023b.
- <span id="page-10-8"></span>Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 1419–1436, 2021.
- <span id="page-10-12"></span>Siddhartha Jain, Xiaofei Ma, Anoop Deoras, and Bing Xiang. Self-consistency for open-ended generations. *arXiv preprint arXiv:2307.06857*, 2023.
- <span id="page-10-0"></span>Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. *Advances in Neural Information Processing Systems*, 35:3843–3857, 2022.
- <span id="page-10-6"></span>Jinyang Li, Binyuan Hui, Ge Qu, Binhua Li, Jiaxi Yang, Bowen Li, Bailin Wang, Bowen Qin, Rongyu Cao, Ruiying Geng, et al. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. *arXiv preprint arXiv:2305.03111*, 2023a.
- <span id="page-10-2"></span>Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 5315–5333, 2023b.
- <span id="page-10-1"></span>Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. *Science*, 378(6624):1092–1097, 2022.
- <span id="page-10-7"></span>Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. *arXiv preprint arXiv:2109.07958*, 2021.
- <span id="page-10-15"></span>Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. *arXiv preprint arXiv:2205.14334*, 2022.
- <span id="page-10-4"></span>Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment, 2023.
- <span id="page-10-13"></span>Marcus J Min, Yangruibo Ding, Luca Buratti, Saurabh Pujar, Gail Kaiser, Suman Jana, and Baishakhi Ray. Beyond accuracy: Evaluating self-consistency of code large language models with identitychain. *arXiv preprint arXiv:2310.14053*, 2023.
- <span id="page-10-3"></span>Ansong Ni, Srini Iyer, Dragomir Radev, Ves Stoyanov, Wen-tau Yih, Sida I Wang, and Xi Victoria Lin. Lever: Learning to verify language-to-code generation with execution. *arXiv preprint arXiv:2302.08468*, 2023.
- <span id="page-10-14"></span>Allen Z Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, et al. Robots that ask for help: Uncertainty alignment for large language model planners. *arXiv preprint arXiv:2307.01928*, 2023.
- <span id="page-10-10"></span>Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. Zeroscrolls: A zero-shot benchmark for long text understanding. *arXiv preprint arXiv:2305.14196*, 2023.
- <span id="page-10-11"></span>Jianhao Shen, Yichun Yin, Lin Li, Lifeng Shang, Xin Jiang, Ming Zhang, and Qun Liu. Generate & rank: A multi-task framework for math word problems. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pp. 2269–2279, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL [https://aclanthology.](https://aclanthology.org/2021.findings-emnlp.195) [org/2021.findings-emnlp.195](https://aclanthology.org/2021.findings-emnlp.195).

- <span id="page-11-6"></span>Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I. Wang. Natural language to code translation with execution. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2022.
- <span id="page-11-15"></span>Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, et al. Towards expert-level medical question answering with large language models. *arXiv preprint arXiv:2305.09617*, 2023.
- <span id="page-11-10"></span>Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. *arXiv preprint arXiv:2201.08239*, 2022. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2201.08239) [2201.08239](https://arxiv.org/abs/2201.08239).
- <span id="page-11-1"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-11-4"></span>Jiaan Wang, Yunlong Liang, Fandong Meng, Zengkui Sun, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. Is chatgpt a good nlg evaluator? a preliminary study, 2023a.
- <span id="page-11-7"></span>Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators. *arXiv preprint arXiv:2305.17926*, 2023b.
- <span id="page-11-5"></span>Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In *The Eleventh International Conference on Learning Representations*, 2022.
- <span id="page-11-0"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. 2022. URL <https://arxiv.org/pdf/2201.11903>.
- <span id="page-11-11"></span>Gwenyth Portillo Wightman, Alexandra DeLucia, and Mark Dredze. Strength in numbers: Estimating confidence of large language models by prompt agreement. In *Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023)*, pp. 326–362, 2023.
- <span id="page-11-13"></span>Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. *arXiv preprint arXiv:2309.03409*, 2023.
- <span id="page-11-2"></span>Pengcheng Yin and Graham Neubig. Reranking for neural semantic parsing. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2019.
- <span id="page-11-8"></span>Pengcheng Yin, Wen-Ding Li, Kefan Xiao, Abhishek Rao, Yeming Wen, Kensen Shi, Joshua Howland, Paige Bailey, Michele Catasta, Henryk Michalewski, Oleksandr Polozov, and Charles Sutton. Natural language to code generation in interactive data science notebooks. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2023.
- <span id="page-11-14"></span>Ori Yoran, Tomer Wolfson, Ben Bogin, Uri Katz, Daniel Deutch, and Jonathan Berant. Answering questions by meta-reasoning over multiple chains of thought. *arXiv preprint arXiv:2304.13007*, 2023.
- <span id="page-11-12"></span>Murong Yue, Jie Zhao, Min Zhang, Liang Du, and Ziyu Yao. Large language model cascades with mixture of thoughts representations for cost-efficient reasoning. *arXiv preprint arXiv:2310.03094*, 2023.
- <span id="page-11-3"></span>Lu Zeng, Sree Hari Krishnan Parthasarathi, and Dilek Hakkani-Tur. N-best hypotheses reranking for text-to-sql systems. *arXiv preprint arXiv:2210.10668*, 2022.
- <span id="page-11-9"></span>Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. In *International Conference on Learning Representations*, 2019.

<span id="page-12-3"></span>Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, and Sida Wang. Coder reviewer reranking for code generation. In *International Conference on Machine Learning*, pp. 41832–41846. PMLR, 2023a.

<span id="page-12-4"></span>Xinghua Zhang, Bowen Yu, Haiyang Yu, Yangyu Lv, Tingwen Liu, Fei Huang, Hongbo Xu, and Yongbin Li. Wider and deeper llm networks are fairer llm evaluators. *arXiv preprint arXiv:2308.01862*, 2023b.

<span id="page-12-1"></span>Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. *arXiv preprint arXiv:2304.09797*, 2023a.

<span id="page-12-0"></span>Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. On large language models' selection bias in multi-choice questions. *arXiv preprint arXiv:2309.03882*, 2023b.

<span id="page-12-2"></span>Chunting Zhou, Junxian He, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Prompt consistency for zero-shot task generalization. *arXiv preprint arXiv:2205.00049*, 2022.

# <span id="page-12-5"></span>A COMPARISON TO ORACLE SELECTION

Tables [7,](#page-12-6) [8,](#page-12-7) [9,](#page-12-8) [10](#page-13-0) and [11](#page-13-1) compare the results of different approaches to the oracle performance, which selects the best response among candidates for each task. The oracle selection is from the same 8 samples as SC and USC. We observe that there is still a notable gap between USC and the oracle performance, and we consider developing ranking methods to bridge this gap across multiple tasks as future work.

<span id="page-12-6"></span>Table 7: Comparison to the oracle selection on mathematical reasoning benchmarks. The results were obtained with PaLM 2-L.

| Approach                                         | GSM8K                | MATH                 |
|--------------------------------------------------|----------------------|----------------------|
| Greedy decoding<br>SC (Wang et al., 2022)<br>USC | 85.7<br>90.4<br>90.2 | 30.8<br>37.9<br>37.4 |
| Oracle                                           | 96.2                 | 57.2                 |

<span id="page-12-7"></span>Table 8: Comparison to the oracle selection on BIRD-SQL benchmark.

| Approach                   | Execution Accuracy | Valid Efficiency Score |
|----------------------------|--------------------|------------------------|
| Greedy decoding<br>SC-Exec | 42.4<br>45.6       | 44.4<br>48.1           |
| USC                        | 45.5               | 48.8                   |
| Oracle                     | 53.3               | 55.7                   |

<span id="page-12-8"></span>Table 9: Comparison to the oracle selection on ARCADE benchmark.

| Approach               | Execution Accuracy |
|------------------------|--------------------|
| Greedy decoding        | 26.0               |
| SC-Exec (strict match) | 29.8               |
| SC-Exec (fuzzy match)  | 30.3               |
| USC                    | 30.1               |
| Oracle                 | 40.5               |

# B EXAMPLES OF USC PROMPTS

Figures [6](#page-14-0) and [7](#page-15-0) present examples of full USC prompts with candidate responses for different tasks.

<span id="page-13-0"></span>Table 10: Comparison to the oracle selection on long-context summarization benchmarks.

| Dataset    | Approach        | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | BERTScore |
|------------|-----------------|---------|---------|------------|-----------|
| GovReport  | Greedy decoding | 38.8    | 16.9    | 33.8       | 62.7      |
|            | USC             | 40.2    | 17.4    | 35.1       | 62.8      |
|            | Oracle          | 46.1    | 21.3    | 40.3       | 64.7      |
| SummScreen | Greedy decoding | 30.6    | 7.5     | 19.1       | 58.7      |
|            | USC             | 31.7    | 7.8     | 19.8       | 58.3      |
|            | Oracle          | 36.9    | 10.8    | 23.6       | 60.6      |

<span id="page-13-1"></span>Table 11: Comparison to the oracle selection on TruthfulQA benchmark.

| Model         | Approach        | GPT-judge | GPT-info |
|---------------|-----------------|-----------|----------|
| PaLM 2-L      | Greedy decoding | 62.1      | 95.1     |
|               | USC             | 67.7      | 99.0     |
|               | Oracle          | 93.8      | 100.0    |
| gpt-3.5-turbo | Greedy decoding | 79.8      | 99.7     |
|               | USC             | 82.5      | 99.6     |
|               | Oracle          | 94.9      | 100.0    |

<span id="page-14-0"></span>I have generated the following responses to the question: The three-digit number "ab5" is divisible by 3. How many different three-digit numbers can "ab5" represent?

Response 0: A number is divisible by 3 if the sum of its digits is divisible by 3. In this case, the sum of the digits of "ab5" must be divisible by 3, so a + b + 5 must be divisible by 3.

The smallest possible sum with a + b + 5 is 1 + 0 + 5 = 6, which is divisible by 3. The largest possible sum is 9 + 7 + 5 = 21, which is also divisible by 3.

Therefore, the possible values of a + b can range from 1 to 16, with each increment of 3 leading to another possible sum. There are 6 such increments (16 - 1) / 3 + 1 = 6, so there are 6 possible values for a + b.

For each possible value of a + b, there are multiple combinations of digits that add up to that value. For example, if a + b = 4, possible combinations are (1, 3), (2, 2), (3, 1) and (4 ,0).

Therefore, there are 30 different three-digit numbers that "ab5" can represent. (1 + 4 + 7 + 9 + 6 + 3 = 30)

In general, any three-digit number of the form "ab5" where a + b + 5 is divisible by 3 will be divisible by 3.

Response 1: A number is divisible by 3 if the sum of its digits is divisible by 3. Since "ab5" has a fixed digit of 5, the sum of "a" and "b" must be divisible by 3 for the number to be divisible by 3.

There are 10 possible digits: 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9. We can consider the cases where the sum of "a" and "b" is 3, 6, 9, 12, 15, or 18:

Sum of 3: 1 possibility (1 + 2).

Sum of 6: 2 possibilities (1 + 5, 2 + 4).

Sum of 9: 3 possibilities (1 + 8, 2 + 7, 3 + 6).

Sum of 12: 4 possibilities (3 + 9, 4 + 8, 5 + 7, 6 + 6).

Sum of 15: 4 possibilities (4 + 11, 5 + 10, 6 + 9, 7 + 8).

Sum of 18: 2 possibilities (9 + 9, 8 + 10).

Therefore, there are a total of 1+2+3+4+4+2 = 16 possibilities for "a" and "b."

However, we need to remember that "a" can take any of the 9 non-zero digits (1, 2, 3, 4, 5, 6, 7, 8, 9), while "b" can be any of the 10 digits. So, we need to double the total number of possibilities for "a" and "b."

Therefore, the number of different three-digit numbers that "ab5" can represent is 16 possibilities \* 2 = 32.

So the answer is 32.

Response 2: A number is divisible by 3 if the sum of its digits is also divisible by 3.

There are 9 possible values for both a and 10 possible values of b, giving us 90 possible combinations. However, only the combinations where a + b + 5 is a multiple of 3 will be valid.

We can list out the possible combinations:

a + b = 1: (1, 0) a + b = 4: (1, 3), (2, 2), (3, 1), (4, 0) a + b = 7: (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0) a + b = 10: (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1) a + b = 13: (4, 9), (5, 8), (6, 7), (7, 6), (8, 5), (9, 4) a + b = 16: (7, 9), (8, 8), (9, 7) This gives a total of 30 different pairs of values for a and b. Therefore, there are 30 different three-digit numbers that "ab5" can represent. 1 + 4 + 7 + 9 + 6 + 3 = 30

I hope this helps! Let me know if you have any other questions.

Evaluate these responses.

...

Select the most consistent response based on majority consensus.

Start your answer with "The most consistent response is Response X" (without quotes).

Figure 6: A full example USC prompt for mathematical reasoning. This corresponds to the question in Figure [2a.](#page-2-0)

<span id="page-15-0"></span>I have generated the following responses to the question: Where do people drink less coffee than they do in Mexico?

Response 0: There are several countries where people drink less coffee than they do in Mexico. Some examples include Japan, China and the United Kingdom.

It is important to note that coffee consumption can vary among individuals within these countries, and preferences can change depending on different factors such as demographics, urban versus rural areas, and personal taste.

Response 1: People in countries like Japan, China, and India typically drink less coffee than they do in Mexico.

- 1. Japan: Tea, particularly green tea, is more popular than coffee in Japan. Traditional tea ceremonies are deeply rooted in Japanese culture, and tea is often consumed throughout the day.
- 2. China: Tea is also the preferred hot beverage in China. Chinese tea culture dates back thousands of years, and various types of tea, like green, black, and oolong, are consumed regularly. Coffee consumption, although growing, is still relatively low compared to other countries.
- 3. India: India has a strong culture of tea consumption. Chai, which is a spiced milk tea, is a staple beverage across the country. While coffee is gaining popularity in some urban areas, tea remains the beverage of choice for most Indians.

Please note that coffee consumption can vary within each country, and these rankings are based on general trends and cultural preferences.

Response 2: There are several countries where people generally drink less coffee compared to Mexico. Some of these countries include:

- 1. Japan: While coffee has gained popularity in recent years, traditional tea culture remains strong in Japan, resulting in lower coffee consumption overall.
- 2. China: Despite having a large population, tea is the preferred hot beverage in China, and coffee consumption is relatively low.
- 3. Saudi Arabia: Traditional Arabic coffee (qahwa) is popular, but in terms of overall coffee consumption, it is lower compared to Mexico.
- 4. India: Tea is the dominant beverage in India, and coffee consumption is significantly lower in comparison.

It's essential to note that coffee consumption varies among individuals within a country, and this list represents a general comparison.

...

Evaluate these responses.

Select the most consistent response based on majority consensus.

Start your answer with "The most consistent response is Response X" (without quotes).

Figure 7: A full example USC prompt for TruthfulQA. This corresponds to the question in Figure [2b.](#page-2-0)