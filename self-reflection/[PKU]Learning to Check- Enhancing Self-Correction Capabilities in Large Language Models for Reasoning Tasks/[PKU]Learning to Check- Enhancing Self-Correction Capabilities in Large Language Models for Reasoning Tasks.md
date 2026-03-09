# Learning to Check: Enhancing Self-Correction Capabilities in Large Language Models for Reasoning Tasks

Che Zhang\* ♠ △ , Zhenyang Xiao\* ♠, Chengcheng Han ♢ , Yixin Lian♡, Yuejian Fang ♠ † ♠ School of Software & Microelectronics, Peking University ♢School of Data Science and Engineering, East China Normal University ♡Xiaobing.AI

> {mmt, kjn}@stu.pku.edu.cn, fangyj@ss.pku.edu.cn chengchenghan@stu.ecnu.edu.cn lianyixin@xiaobing.ai

# Abstract

Self-correction has achieved impressive results in enhancing the style and security of the generated output from large language models (LLMs). However, recent studies suggest that self-correction might be limited or even counterproductive in reasoning tasks due to LLMs' difficulties in identifying logical mistakes.

In this paper, we aim to enhance the selfchecking capabilities of LLMs by constructing training data for checking tasks. Specifically, we apply the Chain of Thought (CoT) methodology to self-checking tasks, utilizing fine-grained step-level analyses and explanations to assess the correctness of reasoning paths. We propose a specialized checking format called "Step CoT Check". Following this format, we construct a checking-correction dataset that includes detailed step-by-step analysis and checking. Then we fine-tune LLMs to enhance their error detection and correction abilities.

Our experiments demonstrate that fine-tuning with the "Step CoT Check" format significantly improves the self-checking and self-correction abilities of LLMs across multiple benchmarks. This approach outperforms other formats, especially in locating the incorrect position, with greater benefits observed in larger models.

For reproducibility, all the datasets and code are provided in [https://github.com/bammt/Learn](https://github.com/bammt/Learn-to-check)[to-check.](https://github.com/bammt/Learn-to-check)

# 1 Introduction

The colossal number of parameters and extensive training data of Large Language Models (LLMs) have led to remarkable performances in reasoning tasks [\(Yao et al.,](#page-11-0) [2023b;](#page-11-0) [Wei et al.,](#page-11-1) [2023\)](#page-11-1), notably in mathematical problem-solving [\(Luo et al.,](#page-9-0) [2023;](#page-9-0) [Wang et al.,](#page-11-2) [2023;](#page-11-2) [Yu et al.,](#page-11-3) [2023;](#page-11-3) [Yuan et al.,](#page-11-4) [2023\)](#page-11-4). Despite these advancements, there remains considerable potential to further enhance the mathematical reasoning capabilities of LLMs.

To further improve the mathematical reasoning capabilities of LLMs, some studies [\(Madaan et al.,](#page-9-1) [2023;](#page-9-1) [Weng et al.,](#page-11-5) [2023;](#page-11-5) [Huang et al.,](#page-9-2) [2023;](#page-9-2) [Saun](#page-10-0)[ders et al.,](#page-10-0) [2022;](#page-10-0) [Welleck et al.,](#page-11-6) [2022;](#page-11-6) [Miao et al.,](#page-9-3) [2023;](#page-9-3) [Chen et al.,](#page-9-4) [2023\)](#page-9-4) have attempted to prompt LLMs to identify and self-correct errors in their reasoning paths, thereby achieving higher accuracy. In general, self-correction involves two distinct processes: self-checking feedback gathering and feedback-driven reasoning. The former requires LLMs to review their previous output and attempt to identify errors within it, while the latter allows the model to modify and optimize its previous outputs based on the received feedback.

However, recent research [\(Huang et al.,](#page-9-2) [2023;](#page-9-2) [Stechly et al.,](#page-10-1) [2023;](#page-10-1) [Valmeekam et al.,](#page-10-2) [2023\)](#page-10-2) indicates that without external feedback, such as ground truth labels for determining the stopping point [\(Shinn et al.,](#page-10-3) [2023;](#page-10-3) [Kim et al.,](#page-9-5) [2023\)](#page-9-5), LLMs cannot improve their reasoning abilities through intrinsic self-correction. This difficulty arises because LLMs often fail to accurately identify errors in reasoning paths and provide precise feedback. [\(Stechly et al.,](#page-10-1) [2023;](#page-10-1) [An et al.,](#page-9-6) [2023;](#page-9-6) [Madaan](#page-9-1) [et al.,](#page-9-1) [2023\)](#page-9-1). This is counterintuitive because, for humans, verifying whether an answer is correct is often easier than directly solving the problem. The former is an NP-hard problem, while the latter is a P problem. Therefore, we hypothesize that by constructing specific task training, LLMs can enhance their ability to directly check for errors.

Given these challenges, our work aims to address the limitations by constructing a checkingcorrection dataset and then fine-tuning LLMs using this dateset. For generating the dataset, we first collect reasoning paths that contain errors and then we propose a specialized prompt format for

<sup>\*</sup>Equal contribution.

<sup>△</sup> Work done during an internship at Xiaobing.AI.

<sup>†</sup>Corresponding author.

checking called "Step CoT Check". This format involves evaluating the correctness of each step by sequentially checking different types of errors and providing detailed explanations. If an error is detected, the model concludes the check for that step and provides a summary. Following this format, we generated data to check erroneous reasoning paths, and subsequently used these checks to create a correction dataset. Then we fine-tune our models on a mixed dataset of CoT data and checking-correction data.

In our setup, the entire reasoning process includes a stage of direct reasoning and two stages of self-correction: (1) direct reasoning, (2) selfchecking feedback gathering and (3) feedbackdriven reasoning. Our training data encompasses three different tasks, each corresponding to one of these stages. And in the stage 2, we compared the performance of the Step CoT Check format with two other prompt formats: All Direct Check and Step Direct Check.

Our contributions are threefold:

- 1. We propose using training data specifically designed to check reasoning paths and demonstrate their effectiveness in enhancing the self-checking capabilities. This approach enables LLMs to achieve self-correction independently of external data.
- 2. We introduce a Chain-of-Thought (CoT) method that incorporates checking tasks. Specifically, we propose the "Step CoT Check" format for checking mathematical reasoning tasks, which categorizes error types and elucidates the reasoning process prior to judgment, thereby improving interpretability. This approach markedly surpasses traditional direct answer verification techniques.
- 3. We have released a checking-correction dataset to improve the performance of LLMs on reasoning tasks. To our knowledge, this dataset is the first to use fine-grained, step-bystep explanations to annotate errors in LLMs' outputs.

# 2 Related Work

Chain-of-Thought (CoT) reasoning, initially pioneered by [Wei et al.](#page-11-1) [\(2023\)](#page-11-1), has significantly advanced the reasoning capabilities of large language models (LLMs). This approach, which enhances LLMs by incorporating detailed reasoning

steps into few-shot examples, has been foundational in improving performance across various reasoning tasks. Subsequent research has expanded upon this, introducing methodologies like Self-Consistency [\(Wang et al.,](#page-11-7) [2022\)](#page-11-7), various prompting techniques [\(Zhou et al.,](#page-11-8) [2022;](#page-11-8) [Drozdov et al.,](#page-9-7) [2022\)](#page-9-7), Self-Training [\(Huang et al.,](#page-9-8) [2022\)](#page-9-8), Verifier [\(Li et al.,](#page-9-9) [2022\)](#page-9-9), Tree of Thought [\(Yao et al.,](#page-11-9) [2023a\)](#page-11-9) and DialCoT [\(Han et al.,](#page-9-10) [2023\)](#page-9-10). The effectiveness of CoT has been demonstrated in domains including logical, commonsense, and mathematical reasoning. Despite the success, these models are not without their flaws. They are still prone to reasoning confusion, generating hallucinations, and computational errors. A prevailing strategy to rectify these undesired behaviors of LLMs is learning from feedback. Extensive research [\(Huang](#page-9-8) [et al.,](#page-9-8) [2022;](#page-9-8) [Madaan et al.,](#page-9-1) [2023;](#page-9-1) [Gero et al.,](#page-9-11) [2023;](#page-9-11) [Jiang et al.,](#page-9-12) [2023\)](#page-9-12) has been undertaken to improve LLMs through the paradigm of learning from feedback. However, [Huang et al.](#page-9-2) [\(2023\)](#page-9-2) found that selfimprovement is not effective when oracle labels are not available. This may be attributed to the limited capability of LLMs to identify errors. Previous works have primarily focused on self-correction, overlooking the enhancement of the models' error identification capabilities. This paper addresses the gap by meticulously constructing training data that includes steps for detecting and correcting errors, thereby strengthening the model's capability to identify mistakes.

## 3 Methodology

In this section, we present our methodology, which comprises two components: checking-correction data generation and fine-tuning LLMs. The former component involves collecting erroneous reasoning paths, generating data that includes a detailed checking process and the revised answers. The latter involves fine-tuning the model on a combined dataset that contains three tasks: direct reasoning, checking the correctness of the reasoning paths and pointing out the error locations, and rereasoning to obtain the correct answers.

#### 3.1 Checking-Correction Data Generation

The process of generating the checking-correction dataset is detailed in Figure [1.](#page-2-0)

### 3.1.1 Collecting Reasoning Paths with Errors

We use a generator model M<sup>g</sup> to produce reasoning path P(Ans(P) ̸= a, and a is the correct an-

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1: The process of checking-correction dataset generation.

swer.) for original question Q, then we get pairs x = (Q, P). Like [An et al.](#page-9-6) [\(2023\)](#page-9-6), we use a newline character as a delimiter to decompose P into fine-grained reasoning steps s1, s2, . . . , sK, and append "Step N:" before each step.

To simplify the notation, we use Path<sup>×</sup> to represent the reasoning path with logical errors. For comparison, we use Path✓ to represent the reasoning paths that lead to a correct answer.

### 3.1.2 Error Type Definition

To construct a prompt format suitable for LLMs to check the answers to mathematical problems, we categorize and analyze the types of errors in a certain step of the generation Path<sup>×</sup> mentioned above. In mathematical reasoning tasks, the types of errors are as follows:

- 1. Goal Errors: The model outputs the process of calculating a quantity that was already mentioned in the known conditions or a quantity that could not be computed based on the existing known conditions. Alternatively, it calculates an intermediate quantity that is irrelevant to the final problem's solution, ultimately leading to a deviation in the reasoning path.
- 2. Computational Expression Errors: The model experiences a hallucination or misunderstands the given conditions of the problem, leading to an incorrect process for calculating an intermediate quantity. This often manifests as the outputted computational expression not matching the given conditions.
- 3. Calculation Errors: The model correctly identifies the intermediate quantity that needs to be calculated and outputs the correct computational expression, but the calculation result is incorrect.

### 3.1.3 Checking Data Generation

For a reasoning path P involving K steps, we prompt a feedback model M<sup>f</sup> to generate checking data for each step sk, k ∈ [1, K] one by one. For each step, M<sup>f</sup> check sequentially according to the three types of errors mentioned above and generate a detailed explanation process.

- 1. Reasoning Goal Appropriateness: Assesses whether the computational goal of s<sup>k</sup> is conducive to solving the question. If the step's objective is unreasonable or redundant, it is deemed incorrect.
- 2. Computational Expression Correctness: Identifies the known conditions related to sk's computational goal and evaluates if the computation aligns with the problem statement and whether the derived expression achieves the computational goal correctly.
- 3. Verification through Inverse Operations: The correctness of the computation is checked by performing inverse operations on the derived expression.

If the above error types are detected, the model will generate a brief summary of the error message; otherwise, it will summary that the step is correct. Since the three types of errors are hierarchical and subsequent errors may depend on the initial error, it only makes sense to check for a particular type of error if the previous types have not occurred. For the same reason, we only generate the checking data of steps up to the first wrong step.[1](#page-2-1) If step sn, n ∈ [1, K] is determined to be incorrect after these checks, a concise summary indicating the error will be generated and denoted as Fn. We refer to this method of step-by-step checking of arithmetic reasoning paths with detailed explanations as "Step CoT Check".

Before generating data on a large scale, we use GPT-4-1106-preview [\(OpenAI et al.,](#page-9-13) [2023\)](#page-9-13) as M<sup>f</sup> and manually assess the quality of 50 checking data samples of Path<sup>×</sup> to ensure data quality. Among these, 43 entries are perfect(accurately identify the erroneous step and provide precise analysis), while 5 entries are good (pinpoint the incorrect step but offer analysis that lacks sufficient accuracy). 2 entries are poor (fail to identify the location of the

<span id="page-2-1"></span><sup>1</sup> In practical experiments, redundant steps are also considered correct as long as they do not contain logical errors. Only steps that contain logical errors are used as stopping criteria.

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 2: The reasoning process combined with self-correction. The Process involves three stages. **Stage 1** involves direct reasoning for a given question. **Stage 2** examines the direct reasoning path and provides feedback at the end. If Stage 2 check identifies errors, Stage 3 is then initiated. **Stage 3** requires the model to generate a new reasoning path and attempt to correct the erroneous answer given the question, original answer, and feedback.

error). Based on this human evaluation, we suppose the overall quality of check data is sufficient for fine-tuning.

To ensure that the model can effectively learn and distinguish between correct and incorrect reasoning processes and reduce bias for the further fine-tuning stage, in addition to  $\mathbf{Path}_{\times}$ , we also filter some original answers from the CoT dataset as  $\mathbf{Path}_{\vee}$  and generate their checking data. <sup>2</sup>

#### 3.1.4 Correction Data Generation

We task a correction model  $\mathcal{M}_c$  to generate correction data. Specifically, given  $(Q,P,F_n)$ , where  $P \in \mathbf{Path}_{\times}$ , the model is required to generate a revised reasoning path  $P_r$ . We use regular expressions to compare  $P_r$  with the original answer a to question Q. If  $P_r$ 's final answer is correct, we then add the checking data and correction data to the Checking-correction dataset; otherwise, we drop the data sample. For the  $\mathbf{Path}_{\checkmark}$ 's checking data, we also add it to the Checking-correction dataset.

#### 3.2 Fine-Tuning LLMs

After generating the checking-correction data, we fine-tune the LLMs to evaluate whether these models can self correct. To this end, we mainly compare the performance of two fine-tuning settings:

- Fine-Tuning on CoT Data We fine-tune the model on GSM8K training data alone. To facilitate comparison with our method, we only use the original train set and don't generate additional reasoning paths for training. We use the model fine-tuned with the original CoT data as a baseline and compare it with our self-correction method.
- Fine-Tuning on CoT Data + Checking-Correction data In addition to CoT data, we also used the Checking-Correction dataset for training. To eliminate the effect of data size, we employed a replacement method, substituting one piece of data from the original CoT dataset with the corresponding Checking-Correction data for the same question Q.

The entire reasoning process is illustrated in Figure 2. And in the stage of **Self-Checking Feedback Gathering**, we employed three different prompt formats and compared the performance of the finetuned models:

- All Direct Check: Input the original question and the original reasoning path into the model M, instructing M to check the entire reasoning path. The model should directly output the first step that contains a logical error.
- Step Direct Check: In each inference round,

<span id="page-3-0"></span><sup>&</sup>lt;sup>2</sup>The specific prompts and examples of generated data can be found in Appendix A.

input the original question and the original reasoning path into the model M, ask M to check one step sk, k ∈ [1, K]. The complete path P is input to provide M with a holistic view. The model stop the process as soon as it finds s<sup>n</sup> containing an error, then we get the error feedback Fn.

• Step CoT Check: The input is the same with "Step Direct Check". But before outputs a checking conclusion for each step, this method requires M to analyze step s<sup>k</sup> for correctness from three dimensions as mentioned before.

The input-output format of CoT data and correction data used for fine-tuning are showed in the Appendix [B.](#page-13-0) For All Direct Check and Step Direct Check formats, we use regular expressions and rules to generate their responses.

# 4 Experiments

# 4.1 Experiment Step

CoT Dataset Our research focuses on multi-step arithmetic reasoning problems and employs the GSM8K dataset [\(Cobbe et al.,](#page-9-14) [2021\)](#page-9-14). GSM8K contains high quality linguistically diverse grade school math word problems with 7,473 examples in the training set and 1,319 test cases.

Generate the Checking-Correction Dataset We use LLaMA-7b and LLaMA-13b [\(Touvron](#page-10-4) [et al.,](#page-10-4) [2023a\)](#page-10-4) as the generator model M<sup>g</sup> to generate Path×, which is fine-tuned for one epoch on the GSM8K training set to be able to learn the CoT format. Subsequently, we increase the temperature of M<sup>g</sup> and generate five different responses for each question in the training set. For each question, we filter one reasoning path that contains an incorrect answer.

We use GPT-4-1106-preview [\(OpenAI et al.,](#page-9-13) [2023\)](#page-9-13) as M<sup>f</sup> to generate accurate Path×'s checking data. However, generating Path✓'s checking data is relatively simple, requiring only the "reading comprehension" of the problem conditions and each step in the reasoning path. To save costs, we use GPT-3.5-turbo [\(OpenAI,](#page-9-15) [2023\)](#page-9-15) as M<sup>f</sup> to generate Path✓'s checking data and as M<sup>c</sup> to generate correction data.

As a result, There are approximately 4,500 instances of Path×'s checking and correction data, and 2,000 instances of Path✓'s checking data in the checking-correction dataset.

Fine-Tuning and Evaluation We conduct experiment on LLaMA-2 [\(Touvron et al.,](#page-10-5) [2023b\)](#page-10-5) models with 7B and 13B parameters, trained on 4 Nvidia A100 80GB GPUs. The models undergo full parameter fine-tuning, employing a learning rate of 8 × 10−<sup>6</sup> , over three epochs, with a cumulative batch size of 32. For models trained on mixed data, we used mixed data and replaced the original 4,000 GSM8K entries with checking-correction data. After extensive experimentation, we determined the optimal ratio of Path✓ to Path<sup>×</sup> for each checking format. For the "All Direct Check" and "Step Direct Check" methods, we adopted a ratio of 3000:1000, while for the "Step CoT Check" method, we chose a ratio of 3700:300. (more information about the data ratio is showed in [4.4\)](#page-6-0).

The effectiveness of the checkpoints is evaluate on the GSM8K benchmark and additional three datasets: MultiArith [\(Roy and Roth,](#page-10-6) [2015\)](#page-10-6), AS-Div [\(Miao et al.,](#page-9-16) [2020\)](#page-9-16), and SVAMP [\(Patel et al.,](#page-10-7) [2021\)](#page-10-7). Each dataset contains elementary mathematical problems with distinct lexical distributions. Our evaluation strategy validates the robustness of our method across these varied data distributions. We use the full 1,319-instance GSM8K set, and randomly sample subsets from MultiArith (400), ASDiv (500), and SVAMP (500).

Checkpoint evaluations are conducted using the vLLM library [\(Kwon et al.,](#page-9-17) [2023\)](#page-9-17). For direct reasoning path generation, we set the temperature parameter to zero. The check and correction mechanism employs a temperature of 0.5, complemented by a top-p value of 0.85 and a top-k setting of 30. For all reasoning tasks, we apply a repetition penalty of 1.2 [\(Artiges et al.,](#page-9-18) [2019\)](#page-9-18), which effectively mitigates the generation of duplicate content.

## 4.2 Main Results and Analysis

Performance Of Reasoning Combined Self-Correction Table [1](#page-5-0) and Figure [3](#page-5-1) illustrate the performance of models trained using our three prompt formats on the test sets. When a portion of the CoT data is replaced with checking-correction instances, there is a decrease in the models' direct reasoning accuracy. This decrease is more pronounced in the LLaMA-2-7B model, while it is much smaller in the LLaMA-2-13B model. This suggests that models with larger parameters may be able to obtain extra information from the Checking-Correction data, thereby somewhat improving CoT performance [\(An et al.,](#page-9-6) [2023\)](#page-9-6).

<span id="page-5-0"></span>

| Method            | Model       | GSM8K |                         | ASDiv |                             | MultiArith |                             | SVAMP |                             |
|-------------------|-------------|-------|-------------------------|-------|-----------------------------|------------|-----------------------------|-------|-----------------------------|
|                   |             | D-R   | w/S-C                   | D-R   | w/S-C                       | D-R        | w/S-C                       | D-R   | w/S-C                       |
| CoT Fine-Tuning   | LLaMA-2-7B  | 32.6  | /                       | 26.8  | /                           | 73.5       | /                           | 35.0  |                             |
| All Direct Check  |             | 27.2  | $32.5  ^{\uparrow 5.3}$ | 25.2  | <b>31.6</b> <sup>↑6.4</sup> | 71.3       | $76.8  ^{\uparrow 5.5}$     | 32.0  | $35.8 \uparrow 3.8$         |
| Step Direct Check |             | 25.1  | $32.4^{\uparrow 7.3}$   | 26.2  | $29.2^{\uparrow 3.0}$       | 72.0       | $79.0^{\uparrow 7.0}$       | 30.0  | $35.6^{\uparrow 5.6}$       |
| Step CoT Check    |             | 27.9  | <b>34.8</b> ↑6.9        | 26.0  | 31.0 ↑5.0                   | 72.8       | <b>80.3</b> <sup>↑7.5</sup> | 31.6  | <b>36.4</b> <sup>↑4.8</sup> |
| CoT Fine-Tuning   |             | 45.6  | /                       | 45.2  | /                           | 85.3       | /                           | 47.2  |                             |
| All Direct Check  | LLaMA-2-13B | 43.5  | $48.1  ^{\uparrow 4.6}$ | 41.4  | $45.8  ^{\uparrow 4.4}$     | 87.0       | $89.5  ^{\uparrow 2.5}$     | 47.0  | $50.8  ^{\uparrow 3.8}$     |
| Step Direct Check |             | 43.7  | $48.8  ^{\uparrow 5.1}$ | 41.2  | $45.4^{14.2}$               | 82.5       | $86.5  ^{\uparrow 4.0}$     | 45.0  | $49.2  ^{\uparrow 4.2}$     |
| Step CoT Check    |             | 45.3  | <b>51.2</b> ↑5.9        | 44.2  | <b>49.0</b> <sup>↑4.8</sup> | 88.3       | <b>92.0</b> †3.7            | 48.2  | <b>53.0</b> <sup>↑4.8</sup> |

Table 1: Correct rates (%) on the test set combined with the self-correction mechanism. For each example in the table, the number on the left represents the correct rate of direct reasoning(D-R), and the number on the right represents the correct rate after self-correction(S-C).

<span id="page-5-1"></span>![](_page_5_Figure_2.jpeg)

Figure 3: Accuracy(%) of checking the correctness of reasoning paths obtained by itself.

For each dataset, all models show a significant increase in accuracy following self-correction. And in almost all datasets, models trained on mixed data outperform those fine-tuned solely on CoT data. Notably, compared to the COT baseline, the improvement in the LLaMA-2-13B model is more significant than in the LLaMA-2-7B model, with the former showing an average increase of 5.5% and the latter 3.8%. This discrepancy is primarily due to a larger reduction in direct reasoning performance in the LLaMA-2-7B model following data substitution. In the LLaMA-2-13B model, the "Step CoT Check" method significantly outperforms other checking methodologies, indicating that in models with larger parameters, categorizing error types and explicitly displaying the checking process can provide more precise feedback.

Comparison of Checking Abilities. To fairly evaluate the model's capability of checking CoT traces and finding errors, we sampled 200 reasoning traces from GSM8K and manually annotated the first incorrect step. This sample included 100 Path√ and 100 Pathx. In addition to our fine-

tuned models, we also conducted experiments on the models without fine-tuning, as well as on two closed-source LLMs: GPT-3.5-turbo and GPT-4-1106-preview. When using the few-shot method to prompt these models, the original model may fail to follow complex prompts and output incorrect steps. Therefore, we required the model to directly evaluate the entire reasoning path to determine its correctness. When generating arithmetic reasoning paths, we find that the probability of obtaining the correct answer after a logical error occurs is extremely low. Therefore, when checking the reasoning path step by step, if any step is deemed incorrect, we mark the entire reasoning path as incorrect. Only when all steps are correct do we mark the entire reasoning path as correct.

As shown in Table 2, the original LLaMA models performed poorly in checking, just above random (50%). Although the performance on GPT-3.5 and GPT-4 seems quite good (with an accuracy of 70% for the former and 89% for the latter), it is worth noting that in direct reasoning scenarios, the accuracies of GPT-3.5 and GPT-4 are 75.9% and 95% (Tyen et al., 2024), respectively. If the model's ability to judge correctness is comparable to its ability to directly output the correct answer (assuming the model's accuracy in judging the correctness of a reasoning path on questions beyond its capability range is 50%), then the accuracy of checking should be 89.95% and 97.5%. It is evident that in LLMs that have not undergone specialized training, the ability to verify a reasoning path is significantly lower than the ability to directly output the correct answer to a question. We believe that it is precisely this "mismatch" between direct reasoning and checking abilities that leads to ineffective intrinsic self-correction.

<span id="page-6-1"></span>

| Method            | Model               | Path√ | Path <sub>×</sub> | Average | Step |
|-------------------|---------------------|-------|-------------------|---------|------|
|                   | GPT-3.5-turbo*      | 93.0  | 47.0              | 70.0    | /    |
| Few-Shot          | GPT-4-1106-preview* | 85.0  | 93.0              | 89.0    | /    |
| rew-Shot          | LLaMA-2-7B          | 74.0  | 35.0              | 54.5    | /    |
|                   | LLaMA-2-13B         | 69.0  | 49.0              | 59.0    | /    |
| All Direct Check  |                     | 84.0  | 63.0              | 73.5    | 28.0 |
| Step Direct Check | LLaMA-2-7B          | 80.0  | 65.0              | 72.5    | 22.0 |
| Step CoT Check    |                     | 63.0  | <b>77.0</b>       | 70.0    | 30.0 |
| All Direct Check  |                     | 61.0  | 83.0              | 72.0    | 36.0 |
| Step Direct Check | LLaMA-2-13B         | 76.0  | 79.0              | 77.5    | 32.0 |
| Step CoT Check    |                     | 80.0  | 83.0              | 81.5    | 48.0 |

Table 2: The accuracy (%) of checking sampled 200 reasoning paths. Step denotes the rate of successfully identifying the first wrong steps in 100 **Path** $_{\times}$ .

In all of our fine-tuned models, the average accuracy exceeds 70%, which is significantly better than the untrained base models. This indicates that through training, the models' self-checking capabilities have been enhanced. Notably, the LLaMA-2-13B model performs exceptionally well in the "Step CoT Check" method, achieving an 81.5% accuracy in path checking and demonstrating a higher success rate in identifying the initial incorrect step, significantly outperforming other methods. However, on the LLaMA-2-7B model, although the probability of identifying the initial incorrect step is relatively high, its accuracy in checking the path does not show a significant advantage compared to the other two formats, suggesting that similar to CoT reasoning(Wei et al., 2023), CoT for checking is also more effective in models with larger parameters.

Despite the LLaMA-2-7B model's lower probability of accurately identifying specific erroneous steps, self-correction still leads to significant improvements in the previous experiments. This is because, although the model may not precisely pinpoint the error, it can still recognize incorrect answers, thus providing numerous incorrect paths with opportunities for regeneration and subsequent correction.

#### 4.3 Ablation Study

To examine the propensity of LLMs to commit specific errors and to validate the effectiveness of our "Step CoT Check" three-step checking method, we conducted an ablation study focusing on the "Step CoT Check" implemented on the LLaMA-2-13B model. The findings are catalogued in Table 3. Among the 541 error paths correctly detected, 20.1% of the errors were successfully corrected

<span id="page-6-2"></span>

|                         | Goal  | Formula | Calculate | Total |
|-------------------------|-------|---------|-----------|-------|
| Number of wrong cases   | 131   | 329     | 81        | 541   |
| Successfully correction | 31    | 61      | 17        | 109   |
| Performance gains       | 1.74% | 3.11%   | 0.99%     | 5.84% |

Table 3: The results of ablation study. In the table, Performance gains is the proportion of "wrong to correct" minus the number of "correct to wrong" cases.

to be accurate. Calculation errors were the most common, accounting for approximately 61.1% of the total errors. These errors usually stem from the LLMs' inability to correctly understand the relationships in the problem or the erroneous use of non-existent conditions, leading to the incorrect construction of mathematical expressions. These computational missteps proved challenging to rectify, with a mere 18% being corrected during the third step, signaling an avenue for future enhancement.

Goal errors, conversely, represent a category of errors more amenable to correction, with a rectification rate of 24%. Regarding computation, LLMs were found to be more susceptible to miscalculations when the equation involved multiple numerical entities or integrated unknowns.

#### <span id="page-6-0"></span>4.4 Further Analysis on Data Proportion

The experiments in this section use the Step Check prompt, and fine tune on LLama-2-13B.

The check ability has a relationship with the data ratio. To discern the influence of the proportion of  $Path_{\checkmark}$  confirmations to  $Path_{\times}$  corrections within a pool of 4,000 training data replacements, we adjusted their ratios and documented the outcomes in Table 4. On average, a  $Path_{\checkmark}$  comprised 3.33 correct steps ( $Step_{\checkmark}$ ) checking, whereas a  $Path_{\times}$  included an average of 1.24 cor-

<span id="page-7-0"></span>

| :<br>Path×<br>Path✓ | :<br>Step×<br>Step✓ | Ratio |       | Check Accuracy(%) |         |                 |
|---------------------|---------------------|-------|-------|-------------------|---------|-----------------|
|                     |                     |       | Path✓ | Path×             | Average | Correct Rate(%) |
| 2000:2000           | 2000:9260           | 0.22  | 93.6  | 36.2              | 60.8    | 45.3            |
| 3000:1000           | 3000:7225           | 0.42  | 88.0  | 47.0              | 64.5    | 47.5            |
| 3500:500            | 3500:6117           | 0.57  | 84.5  | 61.8              | 72.2    | 50.3            |
| 3700:300            | 3700:5681           | 0.65  | 78.9  | 75.0              | 76.8    | 51.2            |
| 4000:0              | 4000:4990           | 0.80  | 46.1  | 88.1              | 69.2    | 48.6            |

Table 4: Results of the data scaling experiment. The table displays how the number of correct and incorrect reasoning paths in the training data have an effect on the accuracy of the check and the correct rate of the reasoning combined with self-correction.

rect steps (Step✓) checking and 1.0 wrong steps (Step×) checking. When the ratio of Step✓ to Step<sup>×</sup> is too high, the model may tend to judge the steps as correct, thereby creating bias. Conversely, an excess of incorrect-checking data may lead the model to judge answers as incorrect more frequently, thereby increasing the instances of "correct becoming wrong," and consequently reducing performance.

From the table, it can be observed that maintaining a ratio of approximately 0.65 between Step<sup>×</sup> and Step✓ in the training set allows the model to achieve a high level of accuracy in distinguishing between correct and incorrect steps. This equilibrium facilitated the highest performance enhancement through a single iteration of the selfcorrection process.

# More checking-corection data can continuously improve the reasoning capability of LLMs. Unlike previous experiments, we use the full GSM8K training dataset and add our own checkingcorrection data. We aim to see how increasing checking-correction data affects the model's reasoning, keeping the original reasoning data the same. We adhere to the optimal ratio of Path✓ and Path<sup>×</sup> ascertained from preceding experiments. Fig. [4](#page-7-1) shows that both direct and self-corrected reasoning improve as we increase the checkingcorrection data.

Similar to the insights from LEMA [\(An et al.,](#page-9-6) [2023\)](#page-9-6), We found that LLMs can also enhance their direct CoT reasoning abilities by training on our checking-correction dataset. Additionally, an increased corpus of checking-correction data markedly bolsters the model's self-correction proficiency. Combining reasoning with self-correction improves faster than direct reasoning alone. These results endorse the inclusion of error-annotated correction data within the training regimen of an inference model, positing that an expanded dataset

<span id="page-7-1"></span>![](_page_7_Figure_6.jpeg)

Figure 4: Accuracy (%) of direct reasoning and reasoning combined with self-correction on GSM8K as checking-corection data types increase.

catalyzes improved inferential processes within the model. Consequently, the model's self-corrective capacity is significantly enhanced, culminating in substantial performance advancements through the mechanism of self-correction.

# 5 Conclusion

In this paper, we focused on improving the selfcorrection capabilities of LLMs by enhancing their self-checking abilities. By conducting an in-depth analysis of error patterns in mathematical reasoning, we developed the "Step CoT Check" prompt and constructed a checking-correction dataset. This dataset, combined with the original CoT data, has been instrumental in training models that exhibit improved self-checking and self-correction, reducing dependence on external feedback or ground truth labels for error identification and correction. Notably, models fine-tuned with our "Step CoT Check" prompt demonstrated superior performance over those trained with alternative prompts. This approach not only enhances the precision of feedback but also significantly increases correctness rates, showing the effectiveness of our method in refining LLMs for complex reasoning tasks.

# Limitations

Our paper has some limitations, which we leave for future work:

#### The computational cost of the step level Check

Compared to direct reasoning and checking the whole path at once, although step-by-step check leads to better performance, it requires more inference rounds. On the GSM8K dataset, inference using step-by-step checking combined with correction process requires 4.8 model calls compared to only one for direct reasoning and 2.6 for checking the whole path at once. In practical use, there is a trade-off between performance and resource consumption.

#### Lack of applicability to broader reasoning tasks.

We analyze the types of mistakes in mathematical reasoning and propose a set of hints for checking the reasoning process. Our approach focuses on checking and correcting errors in mathematical reasoning problems, and it remains to be investigated in future work how to extend this approach to a wider range of reasoning tasks.

Lack of experimentation on larger data sizes and scale language models. Due to constraints on computational resources and funding, we do not conduct experiments on larger scale language models(>20B), or generate additional CoT and correction data like other works. Thus the performance on larger LLMs and sufficient amount of training data remains undetermined. We will further explore the performance of our method on larger data sizes and scale language models in future research.

# Ethics Statement

The proposed method has no obvious potential risks. All the scientific artifacts used/created are properly cited/licensed, and the usage is consistent with their intended use. Also, we open up our codes and hyper-parameters to facilitate future reproduction without repeated energy cost.

# References

- <span id="page-9-6"></span>Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen. 2023. [Learn](http://arxiv.org/abs/2310.20689)[ing From Mistakes Makes LLM Better Reasoner.](http://arxiv.org/abs/2310.20689) ArXiv:2310.20689 [cs].
- <span id="page-9-18"></span>Emmanuel Artiges, Carlos Gracia-Lazaro, Luis Mario Floria, and Yamir Moreno. 2019. [Replicator pop](https://doi.org/10.1103/PhysRevE.100.052307)[ulation dynamics of group \(\\$n\\$-agent\) interactions.](https://doi.org/10.1103/PhysRevE.100.052307) [Broken symmetry, thresholds for metastability and](https://doi.org/10.1103/PhysRevE.100.052307) [macroscopic behavior.](https://doi.org/10.1103/PhysRevE.100.052307) *Phys. Rev. E*, 100(5):052307. ArXiv:1909.06858 [physics, q-bio].
- <span id="page-9-4"></span>Pinzhen Chen, Zhicheng Guo, Barry Haddow, and Kenneth Heafield. 2023. [Iterative Translation Refinement](https://doi.org/10.48550/arXiv.2306.03856) [with Large Language Models.](https://doi.org/10.48550/arXiv.2306.03856) ArXiv:2306.03856 [cs].
- <span id="page-9-14"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. [Training Verifiers to Solve Math Word Prob](https://doi.org/10.48550/arXiv.2110.14168)[lems.](https://doi.org/10.48550/arXiv.2110.14168) ArXiv:2110.14168 [cs].
- <span id="page-9-7"></span>Andrew Drozdov, Nathanael Schärli, Ekin Akyürek, Nathan Scales, Xinying Song, Xinyun Chen, Olivier Bousquet, and Denny Zhou. 2022. Compositional semantic parsing with large language models. *arXiv preprint arXiv:2209.15003*.
- <span id="page-9-11"></span>Zelalem Gero, Chandan Singh, Hao Cheng, Tristan Naumann, Michel Galley, Jianfeng Gao, and Hoifung Poon. 2023. Self-verification improves fewshot clinical information extraction. *arXiv preprint arXiv:2306.00024*.
- <span id="page-9-10"></span>Chengcheng Han, Xiaowei Du, Che Zhang, Yixin Lian, Xiang Li, Ming Gao, and Baoyuan Wang. 2023. Dialcot meets ppo: Decomposing and exploring reasoning paths in smaller language models. *arXiv preprint arXiv:2310.05074*.
- <span id="page-9-8"></span>Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve. *arXiv preprint arXiv:2210.11610*.
- <span id="page-9-2"></span>Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2023. [Large Language](http://arxiv.org/abs/2310.01798) [Models Cannot Self-Correct Reasoning Yet.](http://arxiv.org/abs/2310.01798) ArXiv:2310.01798 [cs].
- <span id="page-9-12"></span>Shuyang Jiang, Yuhao Wang, and Yu Wang. 2023. Selfevolve: A code evolution framework via large language models. *arXiv preprint arXiv:2306.02907*.
- <span id="page-9-5"></span>Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. [Language Models can Solve Computer Tasks.](https://doi.org/10.48550/arXiv.2303.17491) ArXiv:2303.17491 [cs].
- <span id="page-9-17"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. [Efficient](https://doi.org/10.48550/arXiv.2309.06180)

- [Memory Management for Large Language Model](https://doi.org/10.48550/arXiv.2309.06180) [Serving with PagedAttention.](https://doi.org/10.48550/arXiv.2309.06180) ArXiv:2309.06180 [cs].
- <span id="page-9-9"></span>Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2022. On the advance of making language models better reasoners. *arXiv preprint arXiv:2206.02336*.
- <span id="page-9-0"></span>Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. 2023. [Wiz](https://doi.org/10.48550/arXiv.2308.09583)[ardMath: Empowering Mathematical Reasoning for](https://doi.org/10.48550/arXiv.2308.09583) [Large Language Models via Reinforced Evol-Instruct.](https://doi.org/10.48550/arXiv.2308.09583) ArXiv:2308.09583 [cs].
- <span id="page-9-1"></span>Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. [Self-Refine: Iterative](http://arxiv.org/abs/2303.17651) [Refinement with Self-Feedback.](http://arxiv.org/abs/2303.17651) ArXiv:2303.17651 [cs].
- <span id="page-9-3"></span>Ning Miao, Yee Whye Teh, and Tom Rainforth. 2023. [SelfCheck: Using LLMs to Zero-Shot Check Their](https://doi.org/10.48550/arXiv.2308.00436) [Own Step-by-Step Reasoning.](https://doi.org/10.48550/arXiv.2308.00436) ArXiv:2308.00436 [cs].
- <span id="page-9-16"></span>Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. [A Diverse Corpus for Evaluating and Develop](https://doi.org/10.18653/v1/2020.acl-main.92)[ing English Math Word Problem Solvers.](https://doi.org/10.18653/v1/2020.acl-main.92) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 975–984, Online. Association for Computational Linguistics.
- <span id="page-9-15"></span>OpenAI. 2023. [Gpt-3.5 turbo fine-tuning and api up](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-\ api-updates)[dates.](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-\ api-updates)
- <span id="page-9-13"></span>OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mo Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane

Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, C. J. Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2023. [GPT-](https://doi.org/10.48550/arXiv.2303.08774)[4 Technical Report.](https://doi.org/10.48550/arXiv.2303.08774) ArXiv:2303.08774 [cs].

<span id="page-10-7"></span>Arkil Patel, Satwik Bhattamishra, and Navin Goyal.

2021. [Are NLP Models really able to Solve Simple](https://doi.org/10.48550/arXiv.2103.07191) [Math Word Problems?](https://doi.org/10.48550/arXiv.2103.07191) ArXiv:2103.07191 [cs].

<span id="page-10-6"></span>Subhro Roy and Dan Roth. 2015. [Solving General](https://doi.org/10.18653/v1/D15-1202) [Arithmetic Word Problems.](https://doi.org/10.18653/v1/D15-1202) In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 1743–1752, Lisbon, Portugal. Association for Computational Linguistics.

<span id="page-10-0"></span>William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. 2022. [Self-critiquing models for assisting human evaluators.](http://arxiv.org/abs/2206.05802) ArXiv:2206.05802 [cs].

<span id="page-10-3"></span>Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. [Reflexion: Language Agents with Verbal Re](http://arxiv.org/abs/2303.11366)[inforcement Learning.](http://arxiv.org/abs/2303.11366) ArXiv:2303.11366 [cs].

<span id="page-10-1"></span>Kaya Stechly, Matthew Marquez, and Subbarao Kambhampati. 2023. [GPT-4 Doesn't Know It's Wrong:](http://arxiv.org/abs/2310.12397) [An Analysis of Iterative Prompting for Reasoning](http://arxiv.org/abs/2310.12397) [Problems.](http://arxiv.org/abs/2310.12397) ArXiv:2310.12397 [cs].

<span id="page-10-4"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. [LLaMA:](https://doi.org/10.48550/arXiv.2302.13971) [Open and Efficient Foundation Language Models.](https://doi.org/10.48550/arXiv.2302.13971) ArXiv:2302.13971 [cs].

<span id="page-10-5"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. [Llama 2: Open Foundation and](https://doi.org/10.48550/arXiv.2307.09288) [Fine-Tuned Chat Models.](https://doi.org/10.48550/arXiv.2307.09288) ArXiv:2307.09288 [cs].

<span id="page-10-8"></span>Gladys Tyen, Hassan Mansoor, Victor Carbune, Pe- ˘ ter Chen, and Tony Mak. 2024. [LLMs can](http://arxiv.org/abs/2311.08516)[not find reasoning errors, but can correct them!](http://arxiv.org/abs/2311.08516) ArXiv:2311.08516 [cs].

<span id="page-10-2"></span>Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. 2023. [Can Large Language Models](http://arxiv.org/abs/2310.08118) [Really Improve by Self-critiquing Their Own Plans?](http://arxiv.org/abs/2310.08118) ArXiv:2310.08118 [cs].

- <span id="page-11-2"></span>Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023. [MathCoder:](https://doi.org/10.48550/arXiv.2310.03731) [Seamless Code Integration in LLMs for Enhanced](https://doi.org/10.48550/arXiv.2310.03731) [Mathematical Reasoning.](https://doi.org/10.48550/arXiv.2310.03731) ArXiv:2310.03731 [cs].
- <span id="page-11-7"></span>Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*.
- <span id="page-11-1"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. [Chain-of-Thought Prompt](https://doi.org/10.48550/arXiv.2201.11903)[ing Elicits Reasoning in Large Language Models.](https://doi.org/10.48550/arXiv.2201.11903) ArXiv:2201.11903 [cs].
- <span id="page-11-6"></span>Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. 2022. [Generating Sequences by Learning to Self-](http://arxiv.org/abs/2211.00053)[Correct.](http://arxiv.org/abs/2211.00053) ArXiv:2211.00053 [cs].
- <span id="page-11-5"></span>Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. 2023. [Large Language Models are Better Reasoners](http://arxiv.org/abs/2212.09561) [with Self-Verification.](http://arxiv.org/abs/2212.09561) ArXiv:2212.09561 [cs].
- <span id="page-11-9"></span>Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a. Tree of thoughts: Deliberate problem solving with large language models. *arXiv preprint arXiv:2305.10601*.
- <span id="page-11-0"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023b. [ReAct: Synergizing Reasoning and Acting in Lan](https://doi.org/10.48550/arXiv.2210.03629)[guage Models.](https://doi.org/10.48550/arXiv.2210.03629) ArXiv:2210.03629 [cs].
- <span id="page-11-3"></span>Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. [Meta-](https://doi.org/10.48550/arXiv.2309.12284)[Math: Bootstrap Your Own Mathematical Questions](https://doi.org/10.48550/arXiv.2309.12284) [for Large Language Models.](https://doi.org/10.48550/arXiv.2309.12284) ArXiv:2309.12284 [cs].
- <span id="page-11-4"></span>Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. 2023. [Scaling Relationship on Learn](https://doi.org/10.48550/arXiv.2308.01825)[ing Mathematical Reasoning with Large Language](https://doi.org/10.48550/arXiv.2308.01825) [Models.](https://doi.org/10.48550/arXiv.2308.01825) ArXiv:2308.01825 [cs].
- <span id="page-11-8"></span>Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. 2022. Least-to-most prompting enables complex reasoning in large language models. *arXiv preprint arXiv:2205.10625*.

# <span id="page-12-0"></span>A Prompt for Checking-corection Data Generation

#### Example 1: Prompt for Checking Data Generation

## System:

Below you will be given a [Question] and the [original answer] of it which has mistakes. You start from the first step, think carefully to check step by step. For each step, you should strictly follow three processes to check: (1) Find out the goal of this step in the [original answer], and you check if the goal is reasonable, i.e., it's not included in the known conditions, can be calculated from previous information, and it can help to solve the final question. Notice you don't need to check the calculation process here. (2) Analyze the what we know now and how to calculate the goal, then determine if the formula in [original answer] is correct. Notice you should analyze before you check! (3) Use the inverse operation to check if the calculation is correct (e.g. 17-9=8 to check 8+9=17 and 25-9-8=8 to check 8+8+9=25 ). And when you find the first wrong step, stop check and print "Stop check! ".

#### Human:

[Question]: Maila is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

[original answer]:

Step 1: Today's number of pages = 12 x 2= «12\*2=24»24

Step 2: Remaining pages after today = 120 - 24 = «120-24=96»96

Step 3: Tomorrow, she has to read 96 / 2 = «96/2=48»48 pages.

answer: 48

\*\*reference answer\*\*:

Maila read 12 x 2 = «12\*2=24»24 pages today.

So she was able to read a total of 12 + 24 = «12+24=36»36 pages since yesterday.

There are 120 - 36 = «120-36=84»84 pages left to be read.

Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = «84/2=42»42 pages.

#### 42

#### Assistant:

Step 1:

- (1) The goal is calculate the number of pages Maila read today. Because it contributes to solving the final problem, so it's reasonable.
- (2) We already know that she read twice as many pages today as she did yesterday. To get today's pages, we should multiply yesterday's reading by 2 . The formula for this step is correct.
- (3) Using the inverse operation, 24/2=12, so the calculation of 12 \* 2 = 24 is correct. So step 1 is correct.

Step 2:

- (1) The goal is calculate the remaining pages after today. Because it contributes to solving the final problem, so it's reasonable.
- (2) We already know that Maila read 12 pages yesterday and 24 pages today. To find remaining pages, we should subtract the number of pages read both in today and yesterday from the total number of pages. The formula should account for the total pages read so far, which includes both yesterday's and today's reading. The formula for this step is wrong because it does not include the pages read yesterday.
- (3) Because the formula is incorrect, no need to check the calculation.

So step 2 is wrong because the formula is not correct.

Stop check!

#### Example 2: Prompt for Correction Data Generation

## System:

You are an AI assistant who can check and correct the answers very well. Please correct the [original answer] of the [Question] to get the correct answers. You should correct the error pointed out by the [check] in the [original answer]. Please keep your reasoning process close to the [original answer] if possible.

### Human:

[Question]: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

[original\_answer]:

step 1: Today's number of pages = 12 x 2= «12\*2=24»24

step 2: Remaining pages after today = 120 - 24 = «120-24=96»96

step 3: Tomorrow, she has to read 96 / 2 = «96/2=48»48 pages.

answer: 48

[check]:

Step 2 is wrong because the formula is not correct.

## Assistant:

The original answer is wrong because it incorrectly calculated the remaining pages after today in

Step 2. The correct reasoning process is as follows:

Step 1: Today's number of pages = 12 x 2 = 24

step 2: Remaining pages after today is 120 - 12 - 24 = «120-12-24=84»84

step 3: Tomorrow, she has to read 84 / 2 = «96/2=48»42 pages.

answer: 42

# <span id="page-13-0"></span>B Prompt for Reasoning Combined with Self-correction Mechanisms

#### Example 3: Direct Reasoning

#### System:

You are an AI assistant for solving math problems that can think step by step calculate the results accurately.

## Human:

Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

## Assistant:

Step 1: Natalia sold 48/2 = «48/2=24»24 clips in May.

Step 2: Natalia sold 48+24 = «48+24=72»72 clips altogether in April and May.

answer: 72

#### Example 4: Feedback-Driven Reasoning

#### System:

You are an AI assistant who can check and correct the answers very well. Please correct the [original answer] of the [Question] to get the correct answers. You should correct the error step pointed out by the [check] in the [original answer]. Please keep your reasoning process close to the [original answer] if possible.

#### Human:

[Question]: Tina makes \$18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?

[original answer]:

Step 1: She works 8 hours a day for \$18 per hour so she makes 8\*18 = \$«8\*18=144.00»144.00 per 8-hour shift

Step 2: She works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = «10-8=2»2 hours of overtime

Step 3: Overtime is calculated as time and a half so and she makes \$18/hour so her overtime pay is 18\*.5 = \$«18\*.5=9.00»9.00

Step 4: Her overtime pay is 18+9 = \$«18+9=27.00»27.00

Step 5: With the overtime pay included it gives her base earnings of 144+27 =\$«144+27=171.00»171.00 per five day week

answer: 171 [check]:

Step 5 is wrong. (All Direck Check and Step Direct Check)

Step 5 is wrong because the formula is not correct.(Step CoT Check)

## Assistant:

Step 1: Tina works 8 hours a day for \$18 per hour, so she makes 8 \* \$18 = \$144 per 8-hour shift.

Step 2: Tina works 10 hours a day, and anything over 8 hours is eligible for overtime, so she gets 10 - 8 = 2 hours of overtime.

Step 3: Overtime is calculated as time and a half, so she makes \$18 \* 1.5 = \$27 per hour for overtime.

Step 4: Her overtime pay is \$27 \* 2 = \$54.

Step 5: With the overtime pay included, her base earnings of \$144 and her overtime pay of \$54, she makes \$144 + \$54 = \$198 per day.

Step 6: Tina works 5 days a week, so her total earnings per week are \$198 \* 5 = \$990.

Answer: \$990

#### Example 5: Feedback-Acquisition(Direct Check)

#### System:

You are an AI assistant who can check and correct the answers very well. Please check if the [solution] to the following question is correct. If the answer is not correct, point out the error step.

# Human:

[Question]: Weng earns \$12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

[solution]:

Step 1: Weng earns 12/60 = \$«12/60=0.2»0.2 per minute.

Step 2: Working 50 minutes, she earned .2\*50 =\$ «.2\*50=10»10.

answer: 10

#### Assistant:

The answer is all correct.

#### Example 6: Feedback-Acquisition(Direct Step)

### System:

You are an AI assistant who can check and correct the answers very well. Please check if the [solution] to the following question is correct. If the answer is not correct, point out the error step.

### Human:

[Question]: Weng earns \$12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

[solution]:

Step 1: Weng earns 12/60 = \$«12/60=0.2»0.2 per minute.

Step 2: Working 50 minutes, she earned .2\*50 =\$ «.2\*50=10»10.

answer: 10

Please determine if Step 1 is correct.

## Assistant:

Step 1 is all correct.

# Example 7: Feedback-Acquisition(CoT Step)

### System:

You are an AI assistant who can check and correct the answers very well. Based on the given text, please check if the mentioned step of the [solution] is correct.

## Human:

[Question]: Betty is saving money for a new wallet which costs \$100. Betty has only half of the money she needs. Her parents decided to give her \$15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet? [solution]:

Step 1: In the beginning, Betty has only 100 / 2 = \$«100/2=50»50.

Step 2: Betty's grandparents gave her 15 \* 2 = \$«15\*2=30»30.

Step 3: That leaves Betty with 50 + 30 = \$«50+30=80»80 from all sides.

Step 4: So now, Betty still lacks 100 - 80 = \$«100-80=20»20.

answer: 20

Please determine if Step 1 is correct.

# Assistant:

Step 1:

The goal is to determine how much money Betty initially has. This is reasonable because it's necessary to calculate the total amount she will have after receiving additional money from her parents and grandparents.

From the question, we know that Betty has only half of the money she needs for the wallet, which costs \$100. To find out how much she has, we divide the total cost by 2. The formula for this step is correct.

Using the inverse operation, 50 \* 2 = 100, so the calculation of 100 / 2 = 50 is correct.

Step 1 is correct.