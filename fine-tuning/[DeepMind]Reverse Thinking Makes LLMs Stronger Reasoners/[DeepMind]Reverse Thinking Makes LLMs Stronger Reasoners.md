# Reverse Thinking Makes LLMs Stronger Reasoners

Justin Chih-Yao Chen<sup>1</sup>\*, Zifeng Wang<sup>2</sup> , Hamid Palangi<sup>2</sup> , Rujun Han<sup>2</sup> , Sayna Ebrahimi<sup>3</sup> , Long Le<sup>2</sup> , Vincent Perot<sup>3</sup> , Swaroop Mishra<sup>3</sup> , Mohit Bansal<sup>1</sup> , Chen-Yu Lee<sup>2</sup> , Tomas Pfister<sup>2</sup> <sup>1</sup>UNC Chapel Hill, <sup>2</sup>Google Cloud AI Research, <sup>3</sup>Google DeepMind

## Abstract

Reverse thinking plays a crucial role in human reasoning. Humans can reason not only from a problem to a solution but also in reverse, i.e., start from the solution and reason towards the problem. This often enhances overall reasoning performance as it enables consistency checks between their forward and backward thinking. To enable Large Language Models (LLMs) to perform reverse thinking, we introduce Reverse-Enhanced Thinking (REVTHINK), a framework composed of data augmentation and learning objectives. In REV-THINK, we augment the dataset by collecting structured forward-backward reasoning from a teacher model, consisting of: (1) the original question, (2) forward reasoning, (3) backward question, and (4) backward reasoning. We then employ three objectives to train a smaller student model in a multi-task learning fashion: (a) generate forward reasoning from a question, (b) generate a backward question from a question, and (c) generate backward reasoning from the backward question. Experiments across 12 datasets covering commonsense, math, and logical reasoning show an average 13.53% improvement over the student model's zero-shot performance and a 6.84% improvement over the strongest knowledge distillation baselines. Moreover, our method demonstrates sample efficiency – using only 10% of the correct forward reasoning from the training data, it outperforms a standard fine-tuning method trained on 10× more forward reasoning. REVTHINK also exhibits strong generalization to out-of-distribution held-out datasets.

### <span id="page-0-1"></span>1 Introduction

*"Invert, always, invert."* — Carl Jacobi. Reverse thinking plays a crucial role in the human reasoning process [\(Branchini et al.,](#page-8-0) [2021\)](#page-8-0). Take a

#### <span id="page-0-0"></span>**1. Teacher LLM Samples Multiple Reasoning Chains**

![](_page_0_Picture_9.jpeg)

#### **2. Symbolic Knowledge Distillation (Unidirectional)**

![](_page_0_Picture_11.jpeg)

### **3. Reverse-Enhanced Thinking (Ours, Bidirectional)**

![](_page_0_Picture_13.jpeg)

Figure 1: Comparison between symbolic knowledge distillation (SKD) and our method. (1) the teacher model generates multiple reasoning chains for a given question, (2) SKD supervised fine-tunes on the correct reasoning chains, and (3) our method incorporates bidirectional reasoning, learning from both Q-to-A and A-to-Q using our multi-task objectives.

math test for example. An effective way to improve test scores is to reason both forward and backward. In forward reasoning, we begin with the question and work step by step to an answer. Reverse thinking, on the other hand, starts from the predicted answer and works backward to the original question. This two-way approach allows us to verify the accuracy of the solution and identify potential errors. Consider a simple math problem: *Emma has two apples, and Jack has three. How many do they have together?* Forward reasoning leads to the calculation 2 + 3 = 5. Using reverse reasoning, we start with the conclusion that they have five apples. If Emma has two, we can ask: how many does Jack have? The result is three, which matches the orig-

<sup>\*</sup>Correspondence to: cychen@cs.unc.edu

inal problem and confirms the solution is correct. However, if forward reasoning mistakenly predicts the answer as six, reverse reasoning would reveal a conflict: *They have six apples, Emma has two, so Jack must have four,* which contradicts the original problem. This discrepancy signals the need to reassess and improve the solution.

Prior work has shown that Large Language Models (LLMs) benefit from forward-backward reasoning in math [\(Jiang et al.,](#page-9-0) [2024;](#page-9-0) [Weng et al.,](#page-11-0) [2022\)](#page-11-0). This is largely due to two factors: (1) the highly structured nature of math, which facilitates a clear inverse relationship between forward and backward reasoning, and (2) the ability to create new math problems by simply replacing variables like names or numbers. These factors lead to the first research question: *Can reverse thinking be applied to broader, less structured domains?* Moreover, these methods operate at test time, serving as a verification purpose: given solution, we can ask the LLM to think backward and see whether the forward reasoning is correct or not. While they show moderate improvements over other test-time methods such as Self-Consistency [\(Wang et al.,](#page-10-0) [2022\)](#page-10-0), it prompts the second question: *Instead of using backward reasoning for verification at test time, can we train a model to inherently think backward, thereby improving its forward reasoning?*

To address these research questions, we propose REVTHINK, a framework that consists of data augmentation and novel learning objectives designed to instill reverse thinking in language models. We begin by augmenting the dataset using a larger, more capable teacher model. Reasoning benchmark data typically consists of a question and answer. We extend this by generating (1) forward reasoning, (2) a backward question, and (3) backward reasoning, all through few-shot prompting from the teacher model. Both forward and backward reasoning are Chain-of-Thought [\(Wei et al.,](#page-11-1) [2022\)](#page-11-1). We retain only those data points where the forward reasoning is accurate (verified against ground truth) and where the backward reasoning aligns with the original question (validated by prompting the teacher model). After augmenting the dataset, we propose three key objectives for training the smaller student model. Specifically, the student learns to: (1) generate correct forward reasoning from the question, (2) generate a backward question from the original question, and (3) generate backward reasoning from the backward question. The rationale for these objectives is threefold. First,

generating correct reasoning from a question is a standard method of knowledge distillation [\(Li](#page-9-1) [et al.,](#page-9-1) [2023a;](#page-9-1) [West et al.,](#page-11-2) [2022\)](#page-11-2). Second, producing a reverse question encourages the student model to "think" about how to invert a problem and determine the right question to ask. Lastly, solving the backward question reinforces the student's ability to reason backward. At test time, the student model is prompted with the question, and it generates only forward reasoning, similar to standard zero-shot inference. In essence, our pipeline internalizes the ability to reason backward during training, while keeping test-time computation as efficient as zeroshot approaches. As shown in Fig. [1,](#page-0-0) conventional supervised fine-tuning focuses on unidirectional reasoning, from the question to the answer. In contrast, REVTHINK introduces bidirectional thinking by learning to reason in both directions through our data augmentation method and proposed objectives, resulting in greater improvements.

We evaluate REVTHINK on 12 diverse datasets across commonsense reasoning, mathematical reasoning, logical reasoning, and natural language inference, using two models: Mistral-7B-Instruct [\(Jiang et al.,](#page-9-2) [2023\)](#page-9-2) and Gemma-7B-Instruct [\(Team](#page-10-1) [et al.,](#page-10-1) [2024\)](#page-10-1). Our results demonstrate that learning to think backward through our pipeline consistently improves performance, with an average gain of 13.53% over the student's zero-shot performance and 6.84% over the widely-used Symbolic Knowledge Distillation (SKD) method. REVTHINK also shows similar 4.52% − 7.99% gains compared to other data augmentation baselines. Our analysis further highlights that REVTHINK exhibits sample efficiency, where in a low-resource regime, using just 10% of the training instances (augmented by our method) surpasses the performance of SKD applied to the full training set (using forward reasoning). Moreover, REVTHINK scales positively with the student model size from 2B to 176B, achieving better results on a 7B model than a 176B model's zero-shot performance, despite the latter having 25× more parameters. Furthermore, REVTHINK generalizes well to unseen datasets and complements existing data augmentation techniques.

## 2 Related Work

Reasoning with LLMs. A large body of research has shown that LLM reasoning can be improved via advanced test-time approaches, such as prompting and aggregation. Representative methods include Chain-of-Thought (CoT) [\(Kojima et al.,](#page-9-3) [2022;](#page-9-3) [Wei](#page-11-1) [et al.,](#page-11-1) [2022\)](#page-11-1) and Self-Consistency [\(Wang et al.,](#page-10-0) [2022\)](#page-10-0), Tree-of-Thought prompting [\(Yao et al.,](#page-11-3) [2024\)](#page-11-3), Self-Reflection [\(Shinn et al.,](#page-10-2) [2024;](#page-10-2) [Madaan](#page-10-3) [et al.,](#page-10-3) [2024;](#page-10-3) [Yao et al.,](#page-11-4) [2022\)](#page-11-4), Multi-agent collaboration [\(Du et al.,](#page-9-4) [2023;](#page-9-4) [Liang et al.,](#page-10-4) [2023;](#page-10-4) [Wang](#page-11-5) [et al.,](#page-11-5) [2023;](#page-11-5) [Lu et al.,](#page-10-5) [2024;](#page-10-5) [Feng et al.,](#page-9-5) [2024;](#page-9-5) [Chen et al.,](#page-8-1) [2023\)](#page-8-1). Several works have been proposed to leverage backward reasoning to verify the chain-of-thought and improve math reasoning [\(Weng et al.,](#page-11-0) [2022;](#page-11-0) [Jiang et al.,](#page-9-0) [2024\)](#page-9-0), while effective, these methods operate on test time, showing moderate improvements compared to other testtime methods like self-consistency [\(Wang et al.,](#page-10-0) [2022\)](#page-10-0). Also, these methods have mostly been developed for mathematical tasks, limiting their generalizability. In contrast, REVTHINK trains the student model using carefully curated data, enabling it to develop backward reasoning skills in a structured manner. This approach maintains the same test-time efficiency as zero-shot prompting, while delivering greater improvements and generalizing to a broader range of tasks.

Knowledge Distillation. Knowledge distillation is an effective way to transfer knowledge from a larger teacher model to a smaller student model. Classic knowledge distillation learns from the teacher model's distribution, and the objective is to minimize the student distribution with the teacher [\(Hinton et al.,](#page-9-6) [2015;](#page-9-6) [Bucilua et al.](#page-8-2) ˇ , [2006;](#page-8-2) [Chen](#page-8-3) [et al.,](#page-8-3) [2020\)](#page-8-3). Recent advancements in LLMs have shifted the focus toward leveraging the outputs of these larger models. Teacher models provide Chain-of-Thought rationales, which can be sampled directly from the teacher [\(West et al.,](#page-11-2) [2022;](#page-11-2) [Li et al.,](#page-9-1) [2023a;](#page-9-1) [Hsieh et al.,](#page-9-7) [2023;](#page-9-7) [Fu et al.,](#page-9-8) [2023;](#page-9-8) [Magister et al.,](#page-10-6) [2022;](#page-10-6) [Mukherjee et al.,](#page-10-7) [2023;](#page-10-7) [Mi](#page-10-8)[tra et al.,](#page-10-8) [2023\)](#page-10-8), generated via bootstrapping [\(Ze](#page-11-6)[likman et al.,](#page-11-6) [2022;](#page-11-6) [Lewkowycz et al.,](#page-9-9) [2022;](#page-9-9) [Li](#page-10-9) [et al.,](#page-10-9) [2023b\)](#page-10-9), or obtained from multiple teacher models [\(Chen et al.,](#page-8-4) [2024;](#page-8-4) [You et al.,](#page-11-7) [2017\)](#page-11-7). Additionally, teacher model outputs can be used to augment ground truth data [\(Ding et al.,](#page-9-10) [2024\)](#page-9-10). Our method aligns with this recent trend, leveraging the teacher model to generate CoT reasoning, along with backward questions and backward reasoning to augment data. A line of work focuses on improving math reasoning by bootstrapping math-specific datasets [\(Yu et al.,](#page-11-8) [2024;](#page-11-8) [Li et al.,](#page-9-11) [2024;](#page-9-11) [Yuan et al.,](#page-11-9) [2023\)](#page-11-9), as we argue in Section [1,](#page-0-1) math reasoning is inherently structured, making it more suitable for bootstrapping via modifications to names and

variables. Similarly, [Guo et al.](#page-9-12) [\(2024\)](#page-9-12) reverse existing math datasets and find that even powerful LLMs struggle to solve them – implying that these models may be memorizing the problems without genuine comprehension. To the best of our knowledge, we are the first attempt to teach a smaller student model to reason backward, across a broad spectrum of reasoning tasks.

Dual Learning. Dual learning has been extensively studied in machine translation [\(Sennrich](#page-10-10) [et al.,](#page-10-10) [2016;](#page-10-10) [Xia et al.,](#page-11-10) [2018;](#page-11-10) [He et al.,](#page-9-13) [2016\)](#page-9-13), dialog generation [\(Lv et al.,](#page-10-11) [2023;](#page-10-11) [Shen et al.,](#page-10-12) [2018;](#page-10-12) [Li](#page-9-14) [et al.,](#page-9-14) [2021\)](#page-9-14), and question-answering [\(Tang et al.,](#page-10-13) [2017\)](#page-10-13). The core concept is to leverage the primaldual structure inherent to a task, such as the bidirectional relationship between English and German in translation. This duality acts as a form of regularization during training, thereby enhancing performance across both tasks. REVTHINK also incorporates backward question generation and backward reasoning as forms of regularization to improve reasoning capabilities. While dual learning is closely related to our work, the dual relationships established in prior studies—such as source-target language pairs in machine translation—are relatively straightforward. In contrast, we focus on the mutually inverse relationship between a question and its backward counterpart. In our reasoning tasks, backward questions and backward reasoning are often absent and must be generated by LLMs. Our innovation lies in establishing connections between forward questions with forward reasoning and backward questions with backward reasoning, thereby exploiting the consistency of this connection within our training objectives.

## <span id="page-2-1"></span>3 Method

REVTHINK mainly consists of two stages. In Section [3.1,](#page-2-0) we provide a formal description of the problem setup. Section [3.2](#page-3-0) then describes the details of training data creation. Lastly, in Section [3.3,](#page-3-1) we introduce the learning objectives.

## <span id="page-2-0"></span>3.1 Problem Setup

Let D = {(Q(i) , A(i) )} n <sup>i</sup>=1 denote a dataset of n samples, where each sample comprises a question Q(i) and its corresponding answer A(i) . We assume black-box access to a teacher model T , where we can get the output but not logits from the teacher, and our objective is to train a smaller student model S and enhance its reasoning capabilities. During

![](_page_3_Figure_0.jpeg)

Figure 2: REVTHINK consists of two stages: (1) Data augmentation and (2) Student model learning. First, given a dataset  $\mathcal{D}=\{(Q^{(i)},A^{(i)})\}_{i=1}^n$ , we augment it by prompting the teacher model to generate forward reasoning, backward question, and backward reasoning. We keep instances only with correct forward reasoning (validated by the ground truth) and consistent forward-backward reasoning (validated by the teacher model). This yields an augmented dataset  $\mathcal{D}_{\text{aug}}=(Q^{(i)},R_f^{(i)},Q_b^{(i)},R_b^{(i)})_{i=1}^n$ . Next, we train the student model with three objectives:  $Q\to R_f, Q\to Q_b$  and  $Q_b\to R_b$ , enabling the student to reason in both directions during training. At test time, the student model performs only forward reasoning, making test-time compute as efficient as zero-shot prompting.

the training phase, we augment  $\mathcal{D}$  with the teacher's demonstrations of backward questions and backward reasoning to produce  $\mathcal{D}_{aug}$ . A backward question is one that reverses the original question. For example, given the math word problem: John has 3 apples, and Emma has 2; how many apples do they have in total? The corresponding backward question would be: John and Emma have 5 apples in total. If Emma has 2, how many does John have? Backward reasoning refers to the process of solving this reversed question. We then use  $\mathcal{D}_{aug}$  to train the student model  $\mathcal{S}$ . At test time, the student model is prompted only with the original question, similar to zero-shot prompting.

#### <span id="page-3-0"></span>3.2 Data Augmentation

Given a reasoning dataset  $\mathcal{D}=\{(Q^{(i)},A^{(i)})\}_{i=1}^n$ , we begin with augmenting it to produce  $\mathcal{D}_{\mathrm{aug}}$ , where each data point in  $\mathcal{D}_{\mathrm{aug}}$  consists of  $(Q^{(i)},R_f^{(i)},Q_b^{(i)},R_b^{(i)})$ , representing the original question, forward reasoning, backward question, and backward reasoning, respectively. Note that  $R_f,Q_b,R_b$  are all generated by the teacher model  $\mathcal{T}$ . First, we generate forward reasoning  $R_f$  by prompting  $\mathcal{T}$ , and we only keep the samples that  $R_f$  is leading to the correct answer, i.e.,  $g(R_f)=A$ , where g is an answer extraction function. Then, we generate the backward question by conditioning on the original question Q and the ground truth answer A, using a detailed instruction  $I_{\mathrm{bq}}$  (see Appendix B):  $Q_b=\mathcal{T}(Q,A;I_{\mathrm{bq}})$ .

After obtaining the backward question, we prompt the teacher model to generate the backward reasoning by answering the backward question:  $R_b = \mathcal{T}(Q_b)$ . To filter inconsistent pairs (i.e., the backward reasoning is causing conflict with the original question), we prompt  $\mathcal{T}$  with an instruction  $I_{\rm con}$  (see Appendix C) to check for the consistency:  $c = \mathcal{T}(Q, A, Q_b, R_b; I_{con})$ , where  $c \in \{0, 1\}$  represents whether the forward-backward reasoning is consistent. We filter out the data points that are not consistent, i.e., c = 0. That is, we augment  $\mathcal{D}$  by prompting the teacher model to incorporate backward question and backward reasoning, and we only keep samples when (1) the forward reasoning is correct, and (2) the backward reasoning that is consistent with the question.

### <span id="page-3-1"></span>3.3 Learning Objectives

We train the student model S with the augmented dataset  $\mathcal{D}_{aug}$ . To internalize the backward reasoning process, we use the following objectives:

$$\begin{split} \mathcal{L} &= \frac{1}{3n} \sum_{i=1}^{n} \bigg[ \underbrace{\ell(\mathcal{S}(Q^{(i)}), R_f^{(i)})}_{\text{(a) forward reasoning}} + \underbrace{\ell(\mathcal{S}(Q^{(i)}), Q_b^{(i)})}_{\text{(b) backward question}} \\ &+ \underbrace{\ell(\mathcal{S}(Q_b^{(i)}), R_b^{(i)})}_{\text{(c) backward reasoning}} \bigg] \end{split}$$

where  $\ell$  is the cross-entropy between the predicted and target tokens. Specifically, the objective function  $\mathcal{L}$  is composed of three losses that make full

use of our augmented data: (a) learning from generating forward reasoning, (b) learning from generating backward questions, and (c) learning from generating backward reasoning, conditioned on the (b) generated backward question. Below we introduce details of each component.

- (a) Generate Forward Reasoning. The student model takes an original Q as input and generates a forward reasoning R<sup>f</sup> , similar to symbolic knowledge distillation [\(West et al.,](#page-11-2) [2022;](#page-11-2) [Li et al.,](#page-9-1) [2023a\)](#page-9-1).
- (b) Generate Backward Question. The student model still takes Q as input, but instead learns to generate the backward question Qb, i.e., a question that is inversely connected to Q.
- (c) Generate Backward Reasoning. The student model takes the backward question Q<sup>b</sup> as input, and generate backward reasoning R<sup>b</sup> to answer Qb. Our proposed objectives aim to tie all the components together in a multi-task learning way. The objectives of learning to generate backward questions and learning to generate backward reasoning (objectives (b) and (c)) are treated as auxiliary tasks – during inference, we only prompt the trained student model to answer the original question. We show in Table [1](#page-5-0) and Fig. [4](#page-6-0) that learning these two auxiliary tasks can further improve the performance at test time. Another possible way for multi-task learning is to separate the three objectives as three instances, and apply different instructions for finetuning. Empirically, we find our proposed objectives are more effective, as later shown in Table [2.](#page-6-1)

## 4 Experimental Setup

We use Gemini-1.5-Pro-001 [\(Reid et al.,](#page-10-14) [2024\)](#page-10-14) as the teacher model T , with Mistral-7B-Instruct-v0.3 and Gemma-7B-Instruct as the student model S. For training, we use LoRA fine-tuning [\(Hu et al.,](#page-9-15) [2022\)](#page-9-15) with rank 32. We use vllm [\(Kwon et al.,](#page-9-16) [2023\)](#page-9-16) and greedy decoding (with temperature = 0) for all baselines as well as our method. The student model is fine-tuned for 3 epochs on math reasoning tasks (MATH and GSM8K) and 10 epochs for all other domains. For Mistral-7B-Instructv0.3, we set the learning rate to 5e-6, while for Gemma-7B-Instruct, we use a learning rate of 2e-4. These configurations remain consistent across all baseline comparisons. We evaluate our method on a wide range of tasks: Commonsense reasoning: StrategyQA (SQA; [Geva et al.,](#page-9-17) [2021\)](#page-9-17), CommonsenseQA (CSQA; [Talmor et al.,](#page-10-15) [2019\)](#page-10-15), ARCchallenge (ARC; [Clark et al.,](#page-9-18) [2018\)](#page-9-18). Math rea-

soning: MATH [\(Hendrycks et al.,](#page-9-19) [2021\)](#page-9-19), GSM8K [\(Cobbe et al.,](#page-9-20) [2021\)](#page-9-20). Tabular data reasoning: TabMWP [\(Lu et al.,](#page-10-16) [2023\)](#page-10-16). Natural Language Inference: ANLI [\(Nie et al.,](#page-10-17) [2020\)](#page-10-17). Logical Reasoning: Date Understanding [\(bench authors,](#page-8-5) [2023\)](#page-8-5). We compare with three categories of baselines as follows. (1) Zero-shot: We compare with the student's zero-shot performance [\(Kojima et al.,](#page-9-3) [2022\)](#page-9-3) as a reference. (2) Knowledge distillation: We compare with Symbolic Knowledge Distillation (SKD; [Li et al.,](#page-9-1) [2023a;](#page-9-1) [West et al.,](#page-11-2) [2022\)](#page-11-2) that generate CoT from the teacher model, and apply nexttoken prediction loss as the objective. We also compared with Distilling Step-by-Step [\(Hsieh et al.,](#page-9-7) [2023\)](#page-9-7), which employs a loss to predict the label in addition to the CoT rationale. (3) Data augmentation: This set of baselines uses various methods to augment the dataset while applying the same next-token prediction objective. We compare with: (a) Question Rephrasing [\(Yu et al.,](#page-11-8) [2024\)](#page-11-8), which asks the teacher model to paraphrase the question to create a new one. (b) Question Augmentation [\(Li et al.,](#page-9-11) [2024\)](#page-9-11), where the teacher model generates a new question based on the original. (c) Answer Augmentation [\(Yu et al.,](#page-11-8) [2024\)](#page-11-8), which samples another correct reasoning chain from the teacher model for each question.

## 5 Results and Analysis

## 5.1 Main Results

We present our main result in Table [1.](#page-5-0) First, REV-THINK demonstrates superior average performance, outperforming all baselines across datasets and models. Compared to the zero-shot performance of the student model, REVTHINK achieves an average improvement of 12.68% with Mistral and 14.37% with Gemma. Additionally, when compared to SKD and Distill Step-by-Step, which rely on supervised fine-tuning using the correct reasoning chains from the teacher model, REVTHINK shows substantial improvements of 6.44% to 7.15%. Compared to augmentation-based baselines, REV-THINK exhibits greater performance gains, particularly in the areas of commonsense reasoning, tabular reasoning and date understanding. While some of these augmentation methods, e.g., Answer Augmentation (AnsAug) are effective for math reasoning, they tend to show less improvements in other domains, suggesting that math, being a more structured domain, scales better with additional data [\(Li et al.,](#page-9-11) [2024;](#page-9-11) [Yuan et al.,](#page-11-9) [2023\)](#page-11-9). In contrast,

<span id="page-5-0"></span>

|                                           | SQA   | CSQA  | ARC   | MATH  | GSM8K | TabMWP | ANLI  | Date  | Avg.  |
|-------------------------------------------|-------|-------|-------|-------|-------|--------|-------|-------|-------|
| Mistral-7B-Instruct                       |       |       |       |       |       |        |       |       |       |
| Zero-shot (Kojima et al., 2022)           | 53.89 | 62.57 | 73.68 | 10.42 | 54.71 | 65.59  | 43.92 | 39.64 | 50.55 |
| SKD (Li et al., 2023a; West et al., 2022) | 63.76 | 71.86 | 74.66 | 12.48 | 56.16 | 78.19  | 44.90 | 48.50 | 56.08 |
| Distill Step-by-Step (Hsieh et al., 2023) | 64.19 | 71.92 | 75.32 | 11.54 | 56.01 | 76.78  | 44.42 | 49.63 | 56.26 |
| Rephrase Question (Yu et al., 2024)       | 65.07 | 70.19 | 74.51 | 12.98 | 55.10 | 76.31  | 43.58 | 45.51 | 55.41 |
| Question Aug (Li et al., 2024)            | 65.07 | 72.23 | 73.32 | 13.64 | 58.70 | 80.11  | 42.20 | 47.21 | 56.56 |
| Answer Aug (Yu et al., 2024)              | 66.38 | 69.12 | 76.77 | 14.78 | 59.08 | 79.67  | 45.01 | 49.12 | 57.49 |
| REVTHINK (Ours)                           | 70.97 | 75.76 | 78.50 | 15.28 | 60.88 | 85.44  | 48.58 | 70.40 | 63.23 |
| Gemma-7B-Instruct                         |       |       |       |       |       |        |       |       |       |
| Zero-shot (Kojima et al., 2022)           | 56.33 | 66.26 | 68.34 | 8.58  | 41.09 | 55.67  | 37.92 | 40.24 | 46.80 |
| SKD (Li et al., 2023a; West et al., 2022) | 56.77 | 72.48 | 73.29 | 16.86 | 52.24 | 60.52  | 45.42 | 59.62 | 54.65 |
| Distill Step-by-Step (Hsieh et al., 2023) | 56.77 | 73.01 | 72.92 | 16.04 | 51.88 | 62.11  | 44.23 | 60.91 | 54.73 |
| Rephrase Question (Yu et al., 2024)       | 54.15 | 70.22 | 72.37 | 16.96 | 53.07 | 57.62  | 43.07 | 57.99 | 53.18 |
| Question Aug (Li et al., 2024)            | 55.10 | 68.11 | 72.74 | 17.76 | 56.38 | 63.16  | 41.22 | 59.83 | 54.29 |
| Answer Aug (Yu et al., 2024)              | 57.21 | 73.01 | 73.92 | 18.92 | 57.37 | 65.93  | 42.72 | 64.14 | 56.65 |
| REVTHINK (Ours)                           | 64.19 | 74.53 | 75.09 | 19.96 | 57.21 | 84.71  | 47.36 | 66.27 | 61.17 |
| Gemini-1.5-Pro-001 (Teacher Model)        |       |       |       |       |       |        |       |       |       |
| Zero-shot (Kojima et al., 2022)           | 77.39 | 76.72 | 91.51 | 55.90 | 93.73 | 94.27  | 70.12 | 80.00 | 79.76 |

Table 1: On 8 held-in datasets, RevThink outperforms different knowledge distillation and data augmentation baselines. Specifically, our method outperforms the best distillation baseline, Distill Step-by-Step, by 6.97% and 6.44% using Mistral and Gemma, respectively, and the best data augmentation baseline, Answer Augmentation, by 5.74% and 4.52%. While augmentation-based methods like Answer Augmentation work well for math datasets, REVTHINK provide consistent improvements on a wide range of tasks.

**REVTHINK consistently improves performance across a variety of reasoning tasks.** Later in Table 3, we also show that when evaluating on heldout datasets, REVTHINK shows larger gains on an out-of-domain math dataset.

#### 5.2 Additional Analysis

REVTHINK exhibits sample efficiency. Having demonstrated that RevThink outperforms all baselines with the full training set, we now explore the performance of RevThink and the SKD baseline with varying portions of the training data, denoted by  $p \in \{0.1, 0.25, 0.5, 1.0\}$ . For instance, when p=0.1, we sample 10% of the correct forward reasoning for SKD fine-tuning and apply our data augmentation approach, as described in Section 3, for our fine-tuning. Results shown in Fig. 3 demonstrate that RevThink exhibits strong sample efficiency. Across multiple reasoning tasks, RevThink consistently outperforms SKD at all levels of p, even surpassing SKD at p=1.0 with only 10% of the data on StrategyQA.

Backward question generation boosts performance, but the full use of our dataset yields the best performance. Recall that each instance in our teacher data is a tuple  $(Q^{(i)}, R_f^{(i)}, Q_b^{(i)}, R_b^{(i)})$ , consists of the original question, forward reason-

ing, backward question and backward reasoning. We analyze which combination of components enhances the student model the most. In Fig. 4, we find that learning from all components leads to the best performance. Besides this, we find: (1) only learning from answering backward questions hurts the performance. In Fig. 4, using only  $Q_b \to R_b$ results in the worst performance and can even fall below the original student model's zero-shot performance. This is likely because reverse questions are not fully in-domain, and focusing solely on them can lead to distributional shifts, which harms overall performance. (2) learning to generate backward questions can improve the student model. Here we refer to  $(Q \to R_f)\&(Q \to Q_b)$ , where in this setting, the student model has two learning objectives: one is to answer the question by generating forward reasoning, and two is to generate the reverse question given the original question. We observe that adding backward question generation to the learning objectives already improves the student model. However, we show that adding  $Q_b \to R_b$ which enables the student to also learn to answer the reverse question, is the most effective.

REVTHINK's objective is more effective than separated instances with instruction-tuning. Having demonstrated that our data augmentation

<span id="page-6-2"></span>![](_page_6_Figure_0.jpeg)

Figure 3: Comparison of REVTHINK and the SFT baseline with different sample sizes. Notably, REVTHINK shows sample efficiency by largely outperforming SFT given any portion of the training data. Furthermore, our method with only 10% of training data outperforms SFT with the full training data on StrategyQA.

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Figure 4: Comparison of different learning sources and the combination of input/output. We denote  $X \to Y$  as "given X as the input to generate Y". Also, we use & to denote simultaneous learning from different combinations. RevThink's learning to generate forward questions, backward questions and backward reasoning is the most effective, while only learning from generating backward reasoning is the least effective.

<span id="page-6-1"></span>

| Learning Variations         | StrategyQA | GSM8K | Date  |
|-----------------------------|------------|-------|-------|
| Multi-task (Instruction)    | 71.19      | 59.81 | 64.36 |
| Multi-task (Task-Prefix)    | 69.11      | 58.74 | 66.12 |
| Joint Objectives (REVTHINK) | 73.36      | 60.62 | 70.40 |

Table 2: Comparison of learning variations. Our joint objectives performs the best compared to separating each instance into three with multi-task training.

method is effective, and that learning from all of the components leads to the best performance, we further investigate other possible ways to leverage the augmented data. We compare REVTHINK with two settings: (1) Multi-task (Instruction): each instance  $(Q^{(i)}, R_f^{(i)}, Q_b^{(i)}, R_b^{(i)})$  is separated into three instances:  $(Q^{(i)}, R_f^{(i)})$ ,  $(Q^{(i)}, Q_b^{(i)})$  and  $(Q^{(i)}, R_b^{(i)})$ , and then train the student model with different instructions for each pair. (2) Multi-task (Task-Prefix): instead of using different instructions, we specify a prefix for each task. For example, a special token [FR] is appended when learning  $Q \to R_f$ . Results in Table 2 show that our joint objective performs the best.

<span id="page-6-3"></span>![](_page_6_Figure_7.jpeg)

Figure 5: The average token counts per sample *used in training* versus the *test-time* accuracy. The dashed line shows the regression over the baselines. Our method outperforms the baselines with only a slight increase in token count. Note that REVTHINK generates a comparable number of tokens across all baselines at test time.

REVTHINK obtains greater improvements with only slightly more tokens. We note that augmenting the dataset with our method produces more tokens during training. While this holds true for any data augmentation method, we compare the increased token count and the test-time performance. In Fig. 5, we compare against SKD, Ques-

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 6: REVTHINK scales effectively with student model size. Notably, Mistral-7B + REVTHINK outperforms Mistral-8x22B (the red dashed line) by 8.36%.

tion Rephrasing (QR), Question Augmentation (Q Aug) and Answer Augmentation (AnsAug). The token count and the accuracy are average across all datasets. We find that while our method produces slightly more tokens than AnsAug, it largely outperforms it, as seen by REVTHINK's deviation from the dashed regression line.

REVTHINK scales positively with model size. We apply REVTHINK on StrategyQA using models of varying sizes (all models are instruction-tuned). As shown in Fig. 6, our method demonstrates a clear upward trend in accuracy as model size increases. For each model size, applying our method leads to consistent improvements. Notably, Mistral-7B with our method surpasses Mistral-8x22B by 8.36%, despite the latter having  $25\times$  more parameters. These results underscore that REVTHINK scales effectively with the model size.

<span id="page-7-0"></span>

| Method        | BoolQ          | OpenbookQA     | e-SNLI         | GSM8K-Rev      |
|---------------|----------------|----------------|----------------|----------------|
| Zero-shot     | 53.18          | 70.20          | 52.27          | 17.37          |
| SKD<br>AnsAug | 60.82<br>61.74 | 75.40<br>76.40 | 56.62<br>55.54 | 27.54<br>28.96 |
| REVTHINK      | 63.85          | 79.60          | 61.97          | 32.05          |

Table 3: Performance comparison on four held-out datasets. REVTHINK shows better generalizability compared to the baselines, indicating reverse-enhanced thinking contributes to a better understanding of the problems while reducing the risk of overfitting.

REVTHINK generalizes to OOD datasets. We observe that REVTHINK demonstrates superior generalization compared to the baselines. Specifically, we evaluate REVTHINK against the baselines in four different settings: (1) trained on StrategyQA and tested on BoolQ (Clark et al., 2019), (2) trained on ARC-c and evaluated on OpenbookQA (Mihaylov et al., 2018), (3) trained on ANLI and tested on e-SNLI (Camburu et al., 2018), and (4) trained on GSM8K and evaluated on GSM8K-

Reversal (Guo et al., 2024). As shown in Table 3, REVTHINK exhibits superior generalizability over the baselines, achieving a 2.11% improvement on BoolQ, a 3.20% improvement on OpenbookQA, and an even larger 5.35% gain on e-SNLI. Notably, GSM8K-Reversal is a dataset generated by prompting GPT-4 (Achiam et al., 2023) to preserve the style and format of the original GSM8K questions while reversing the task by swapping the answer variable with a given variable. While all methods show reduced performance on this reversed GSM8K, REVTHINK shows larger improvements, outperforming AnsAug by 3.09%. Overall, these results suggest that learning to reason backward not only enhances in-domain reasoning but also improves the generalizability to unseen datasets.

<span id="page-7-2"></span>

| Method            | SQA          | GSM8K        |
|-------------------|--------------|--------------|
| SKD               | 63.76        | 56.16        |
| AnsAug            | 66.38        | 59.08        |
| REVTHINK          | 70.97        | 60.88        |
| REVTHINK + AnsAug | <b>73.56</b> | <b>61.58</b> |

Table 4: REVTHINK complements existing methods such as AnsAug. While AnsAug shows improvements by sampling more forward reasoning, integrating REVTHINK further improves by enabling reverse thinking.

# REVTHINK complements existing methods.

With Answer Augmentation (AnsAug) being the best-performing augmentation baseline, we show the complementary strength of REVTHINK. In Table 4, we compare the performance of AnsAug alone, REVTHINK alone and their combination. For AnsAug + REVTHINK, we follow the same augmented dataset and procedure as in Section 3 – with the only difference being sampling another correct reasoning chain for each question from the teacher model – as AnsAug does. We find that while both AnsAug and REVTHINK yield individual improvements over SKD, their combination results in even greater enhancements, indicating that REVTHINK effectively complements existing data augmentation methods like AnsAug.

REVTHINK shows greater improvements on invertible and medium-hard problems. We further explore when REVTHINK shows its strength. In Fig. 7, we break down correct predictions by problem type and difficulty, as annotated in the original dataset (Hendrycks et al., 2021). REVTHINK improves the most in prealgebra, precalculus, and counting & probability, but no significant gains in number theory. This can be attributed to the fact

<span id="page-8-9"></span>![](_page_8_Figure_0.jpeg)

Figure 7: The break down analysis of the MATH dataset. Each grouped bar shows our results on the left and the SKD baseline on the right.

that subjects like number theory are less invertible, i.e., thinking backward doesn't offer much advantage, or the problem itself cannot be inverted. In contrast, fields like algebra, calculus, and counting often exhibit an inverse relationship between the problem and its solution, making them more conducive to REVTHINK's approach. Interestingly, although REVTHINK outperforms SKD across all difficulty levels, it shows the greatest improvement on level 3 problems, indicating that medium-hard problems benefit the most from REVTHINK.

## 6 Conclusion

We introduce REVTHINK, a framework that improves LLM by enabling backward reasoning. We propose an effective data augmentation method that generates well-structured forward-backward data from a teacher model, and we also propose an effective learning objective with auxiliary tasks that make full use of this augmented data. Experimental results not only show that REVTHINK is effective across 12 datasets on a wide range of tasks, but also reveal additional benefits, including sample efficiency, generalization, and the complementary strength to existing methods.

## Limitations

Despite efforts to make state-of-the-art large language models (LLMs) safer and more trustworthy [\(Liu et al.,](#page-10-19) [2023\)](#page-10-19), the teacher model used in REVTHINK can still produce biased responses or reflect stereotypes embedded in its pre-training data. As a result, student models generated through distillation may inherit these undesirable traits, a challenge inherent to any distillation method. In other words, because student models learn from the teacher model, they remain vulnerable to producing similar biased outputs. Therefore, models created through REVTHINK distillation share the

same risks of misuse as other LLM-derived methods. Further research is needed to effectively evaluate and mitigate these biases in LLMs.

## References

<span id="page-8-8"></span>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*.

<span id="page-8-5"></span>BIG bench authors. 2023. [Beyond the imitation game:](https://openreview.net/forum?id=uyTL5Bvosj) [Quantifying and extrapolating the capabilities of lan](https://openreview.net/forum?id=uyTL5Bvosj)[guage models.](https://openreview.net/forum?id=uyTL5Bvosj) *Transactions on Machine Learning Research*.

<span id="page-8-0"></span>Erika Branchini, Elena Capitani, Roberto Burro, Ugo Savardi, and Ivana Bianchi. 2021. [Opposites in rea](https://doi.org/10.3389/fpsyg.2021.715696)[soning processes: Do we use them more than we](https://doi.org/10.3389/fpsyg.2021.715696) [think, but less than we could?](https://doi.org/10.3389/fpsyg.2021.715696) *Frontiers in Psychology*, 12:715696.

<span id="page-8-2"></span>Cristian Bucilua, Rich Caruana, and Alexandru ˇ Niculescu-Mizil. 2006. [Model compression.](https://dl.acm.org/doi/abs/10.1145/1150402.1150464) In *Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 535–541.

<span id="page-8-7"></span>Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. [e-snli: Natu](http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf)[ral language inference with natural language expla](http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf)[nations.](http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf) In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems 31*, pages 9539–9549. Curran Associates, Inc.

<span id="page-8-4"></span>Justin Chen, Swarnadeep Saha, Elias Stengel-Eskin, and Mohit Bansal. 2024. [MAGDi: Structured distillation](https://proceedings.mlr.press/v235/chen24ah.html) [of multi-agent interaction graphs improves reasoning](https://proceedings.mlr.press/v235/chen24ah.html) [in smaller language models.](https://proceedings.mlr.press/v235/chen24ah.html) In *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 7220–7235. PMLR.

<span id="page-8-1"></span>Justin Chih-Yao Chen, Swarnadeep Saha, and Mohit Bansal. 2023. Reconcile: Round-table conference improves reasoning via consensus among diverse llms. *arXiv preprint arXiv:2309.13007*.

<span id="page-8-3"></span>Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. 2020. [Big](https://arxiv.org/abs/2006.10029) [self-supervised models are strong semi-supervised](https://arxiv.org/abs/2006.10029) [learners.](https://arxiv.org/abs/2006.10029) *Advances in neural information processing systems*, 33:22243–22255.

<span id="page-8-6"></span>Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. *arXiv preprint arXiv:1905.10044*.

- <span id="page-9-18"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*.
- <span id="page-9-20"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.
- <span id="page-9-10"></span>Yangruibo Ding, Jinjun Peng, Marcus J. Min, Gail Kaiser, Junfeng Yang, and Baishakhi Ray. 2024. [Semcoder: Training code language mod](https://arxiv.org/abs/2406.01006)[els with comprehensive semantics.](https://arxiv.org/abs/2406.01006) *arXiv preprint arXiv:2406.01006*.
- <span id="page-9-4"></span>Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.
- <span id="page-9-5"></span>Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, and Yulia Tsvetkov. 2024. Don't hallucinate, abstain: Identifying llm knowledge gaps via multi-llm collaboration. *arXiv preprint arXiv:2402.00367*.
- <span id="page-9-8"></span>Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In *International Conference on Machine Learning*, pages 10421–10430. PMLR.
- <span id="page-9-17"></span>Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. [Did aristotle](https://doi.org/10.1162/tacl_a_00370) [use a laptop? a question answering benchmark with](https://doi.org/10.1162/tacl_a_00370) [implicit reasoning strategies.](https://doi.org/10.1162/tacl_a_00370) *Transactions of the Association for Computational Linguistics*, 9:346– 361.
- <span id="page-9-12"></span>Pei Guo, Wangjie You, Juntao Li, Yan Bowen, and Min Zhang. 2024. Exploring reversal mathematical reasoning ability for large language models. In *Findings of the Association for Computational Linguistics ACL 2024*, pages 13671–13685.
- <span id="page-9-13"></span>Di He, Yingce Xia, Tao Qin, Liwei Wang, Nenghai Yu, Tie-Yan Liu, and Wei-Ying Ma. 2016. Dual learning for machine translation. *Advances in neural information processing systems*, 29.
- <span id="page-9-19"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. *NeurIPS*.
- <span id="page-9-6"></span>Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. [Distilling the knowledge in a neural network.](https://arxiv.org/abs/1503.02531) *arXiv preprint arXiv:1503.02531*.
- <span id="page-9-7"></span>Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! outperforming larger language

- models with less training data and smaller model sizes. In *Findings of the Association for Computational Linguistics: ACL 2023*.
- <span id="page-9-15"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. [LoRA: Low-rank adaptation of](https://openreview.net/forum?id=nZeVKeeFYf9) [large language models.](https://openreview.net/forum?id=nZeVKeeFYf9) In *International Conference on Learning Representations*.
- <span id="page-9-2"></span>Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. *arXiv preprint arXiv:2310.06825*.
- <span id="page-9-0"></span>Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T Kwok. 2024. Forward-backward reasoning in large language models for mathematical verification. In *Findings of Annual Meeting of the Association for Computational Linguistics*.
- <span id="page-9-3"></span>Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In *Advances in Neural Information Processing Systems*, volume 35, pages 22199–22213.
- <span id="page-9-16"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*.
- <span id="page-9-9"></span>Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. 2022. Solving quantitative reasoning problems with language models. *Advances in Neural Information Processing Systems*, 35:3843– 3857.
- <span id="page-9-11"></span>Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng. 2024. Common 7b language models already possess strong math capabilities. *arXiv preprint arXiv:2403.04706*.
- <span id="page-9-14"></span>Jinpeng Li, Yingce Xia, Rui Yan, Hongda Sun, Dongyan Zhao, and Tie-Yan Liu. 2021. Stylized dialogue generation with multi-pass dual learning. *Advances in Neural Information Processing Systems*, 34:28470– 28481.
- <span id="page-9-1"></span>Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. 2023a. [Sym](https://doi.org/10.18653/v1/2023.acl-long.150)[bolic chain-of-thought distillation: Small models can](https://doi.org/10.18653/v1/2023.acl-long.150) [also "think" step-by-step.](https://doi.org/10.18653/v1/2023.acl-long.150) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2665– 2679, Toronto, Canada. Association for Computational Linguistics.

- <span id="page-10-9"></span>Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2023b. Making language models better reasoners with step-aware verifier. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5315–5333.
- <span id="page-10-4"></span>Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. 2023. Encouraging divergent thinking in large language models through multi-agent debate. *arXiv preprint arXiv:2305.19118*.
- <span id="page-10-19"></span>Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. 2023. Trustworthy llms: a survey and guideline for evaluating large language models' alignment. *arXiv preprint arXiv:2308.05374*.
- <span id="page-10-5"></span>Li-Chun Lu, Shou-Jen Chen, Tsung-Min Pai, Chan-Hung Yu, Hung-yi Lee, and Shao-Hua Sun. 2024. Llm discussion: Enhancing the creativity of large language models via discussion framework and roleplay. *arXiv preprint arXiv:2405.06373*.
- <span id="page-10-16"></span>Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2023. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In *International Conference on Learning Representations (ICLR)*.
- <span id="page-10-11"></span>Ang Lv, Jinpeng Li, Shufang Xie, and Rui Yan. 2023. [Envisioning future from the past: Hierarchical dual](https://doi.org/10.18653/v1/2023.acl-long.407)[ity learning for multi-turn dialogue generation.](https://doi.org/10.18653/v1/2023.acl-long.407) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 7382–7394, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-10-3"></span>Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2024. Self-refine: Iterative refinement with self-feedback. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-10-6"></span>Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. 2022. Teaching small language models to reason. *arXiv preprint arXiv:2212.08410*.
- <span id="page-10-18"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *EMNLP*.
- <span id="page-10-8"></span>Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agarwal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, et al. 2023. [Orca 2: Teaching small](https://arxiv.org/abs/2311.11045) [language models how to reason.](https://arxiv.org/abs/2311.11045) *arXiv preprint arXiv:2311.11045*.

- <span id="page-10-7"></span>Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. *arXiv preprint arXiv:2306.02707*.
- <span id="page-10-17"></span>Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. [Adversarial](https://doi.org/10.18653/v1/2020.acl-main.441) [NLI: A new benchmark for natural language under](https://doi.org/10.18653/v1/2020.acl-main.441)[standing.](https://doi.org/10.18653/v1/2020.acl-main.441) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 4885–4901, Online. Association for Computational Linguistics.
- <span id="page-10-14"></span>Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint arXiv:2403.05530*.
- <span id="page-10-10"></span>Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. [Improving neural machine translation models](https://doi.org/10.18653/v1/P16-1009) [with monolingual data.](https://doi.org/10.18653/v1/P16-1009) In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 86–96, Berlin, Germany. Association for Computational Linguistics.
- <span id="page-10-12"></span>Xiaoyu Shen, Hui Su, Wenjie Li, and Dietrich Klakow. 2018. [NEXUS network: Connecting the preceding](https://doi.org/10.18653/v1/D18-1463) [and the following in dialogue generation.](https://doi.org/10.18653/v1/D18-1463) In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 4316–4327, Brussels, Belgium. Association for Computational Linguistics.
- <span id="page-10-2"></span>Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2024. Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-10-15"></span>Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. [CommonsenseQA: A ques](https://doi.org/10.18653/v1/N19-1421)[tion answering challenge targeting commonsense](https://doi.org/10.18653/v1/N19-1421) [knowledge.](https://doi.org/10.18653/v1/N19-1421) In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.
- <span id="page-10-13"></span>Duyu Tang, Nan Duan, Tao Qin, Zhao Yan, and Ming Zhou. 2017. Question answering and question generation as dual tasks. *arXiv preprint arXiv:1706.02027*.
- <span id="page-10-1"></span>Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. *arXiv preprint arXiv:2403.08295*.
- <span id="page-10-0"></span>Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and

- Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*.
- <span id="page-11-5"></span>Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. 2023. Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona selfcollaboration. *arXiv preprint arXiv:2307.05300*.
- <span id="page-11-1"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837.
- <span id="page-11-0"></span>Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. 2022. Large language models are better reasoners with self-verification. *arXiv preprint arXiv:2212.09561*.
- <span id="page-11-2"></span>Peter West, Chandra Bhagavatula, Jack Hessel, Jena Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. 2022. [Symbolic](https://doi.org/10.18653/v1/2022.naacl-main.341) [knowledge distillation: from general language mod](https://doi.org/10.18653/v1/2022.naacl-main.341)[els to commonsense models.](https://doi.org/10.18653/v1/2022.naacl-main.341) In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4602–4625, Seattle, United States. Association for Computational Linguistics.
- <span id="page-11-10"></span>Yingce Xia, Xu Tan, Fei Tian, Tao Qin, Nenghai Yu, and Tie-Yan Liu. 2018. [Model-level dual learning.](https://proceedings.mlr.press/v80/xia18a.html) In *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pages 5383–5392. PMLR.
- <span id="page-11-3"></span>Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-11-4"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*.
- <span id="page-11-7"></span>Shan You, Chang Xu, Chao Xu, and Dacheng Tao. 2017. Learning from multiple teacher networks. In *Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining*, pages 1285–1294.
- <span id="page-11-8"></span>Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. [Metamath:](https://openreview.net/forum?id=N8N0hgNDRt) [Bootstrap your own mathematical questions for large](https://openreview.net/forum?id=N8N0hgNDRt) [language models.](https://openreview.net/forum?id=N8N0hgNDRt) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-11-9"></span>Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and

- Jingren Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. *arXiv preprint arXiv:2308.01825*.
- <span id="page-11-6"></span>Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. Star: Bootstrapping reasoning with reasoning. *Advances in Neural Information Processing Systems*, 35:15476–15488.

## A Additional Analysis

#### <span id="page-12-1"></span>A.1 Verification improves data quality.

|                  | StrategyQA | GSM8K | ANLI  |
|------------------|------------|-------|-------|
| REVTHINK         | 70.97      | 60.88 | 48.58 |
| w/o Verification | 68.63      | 59.75 | 47.63 |

Table 5: Teacher verification improves data quality, as the performance drops w/o verification.

In Section [3.2,](#page-3-0) we note that our data augmentation consists of a verification stage conducted by the teacher model. In this ablation, we study the difference on downstream performance. Results in Table [5](#page-12-1) show that removing verification cause a drop in performance, suggesting that verification improves data quality, even though the number of training samples might be reduced.

# <span id="page-12-0"></span>B Prompt for Backward Question Generation

## Prompt for Backward Question Generation (Multiple Choice Problem)

Your task is to generate an inverse question with {n} answer choices, based on the input question and its correct answer.

Follow these rules:

- 1. Use the correct answer from the input question to create a new, related but inverse question.
- 2. Ensure that the {n} new answer choices are inversely correlated with the {n} input question's choices.
- 3. Make sure only one answer choice in your generated question is correct and reasonable.
- 4. The correct answer in your generated question must be present in the input question.
- 5. The generated question and answer choices should be semantically different from the input question.

{in\_context\_samples}

{input\_question}

#### Prompt for Backward Question Generation (Math Reasoning Problem)

Your task is to generate an inverse question, based on the input question and its correct answer. Follow these rules:

- 1. Use the correct answer from the input question to create a new, related but inverse question.
- 2. Make sure there exists only one correct answer in your generated question.
- 3. The correct answer in your generated question must be present in the input question.
- 4. The generated question should be semantically different from the input question.

{in\_context\_samples}

{input\_question}

The prompt Ibq we use to generate backward questions. They are general templates, where we insert specific {in\_context\_samples} depending on the task. Below, we provide the in-context examples for each task.

#### In-context samples for StrategyQA

INPUT: Is shrimp scampi definitely free of plastic? The correct answer is no.

OUTPUT: If shrimp scampi does not definitely free of plastic, can shrimp scampi possibly contain plastic? (A) yes (B) no. The correct answer is (A).

INPUT: Would a retail associate envy the retailer's CEO's pay? The correct answer is yes.

OUTPUT: If a retail associate envy the retailer's CEO's pay, then the CEO's pay is \_ than the associate's pay. (A) higher (B) lower. The correct answer is (B).

INPUT: Should you be skeptical of a 21 year old claiming to have a doctorate? The correct answer is yes.

OUTPUT: If you should be skeptical of a 21 year old claiming to have a doctorate, then the average age that someone gets their doctorate at is \_ than 21. (A) higher (B) lower. The correct answer is (A). INPUT: Would a vegan eat a traditional Paella dish? The correct answer is no.

OUTPUT: If a vegan would not eat a traditional Paella dish, then Paella \_ contain animals or products derived from animals (A) must (B) must not. The correct answer is (B).

## In-context samples for ARC

INPUT: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? (A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion. The correct answer is (A).

OUTPUT: George is rubbing his dry palms. What is he most likely trying to do? (A) Warm his hands (B) Moisturize his hands (C) Care for his skin (D) Lubricate his hands. The correct answer is (A).

INPUT: Which of the following statements best explains why magnets usually stick to a refrigerator door? (A) The refrigerator door is smooth (B) The refrigerator door contains iron (C) The refrigerator door is a good conductor (D) The refrigerator door has electric wires in it. The correct answer is (B). OUTPUT: If a refrigerator door contains iron, which of the following is most likely to happen? (A) The refrigerator door will be slippery (B) Magnets will usually stick to the refrigerator door (C) A person will get shocked by touching the refrigerator door (D) The refrigerator won't work without the iron in the door. The correct answer is (B).

INPUT: A fold observed in layers of sedimentary rock most likely resulted from the (A) cooling of flowing magma (B) solution of carbonate minerals (C) deposition of river sediments (D) converging of crustal plates. The correct answer is (D).

OUTPUT: Which of the following is most likely caused by converging crustal plates? (A) Volcanic eruptions (B) The formation of caves (C) The creation of river deltas (D) Folds observed in layers of sedimentary rock. The correct answer is (D).

INPUT: The male insects in a population are treated to prevent sperm production. Would this reduce this insect population? (A) No, because the insects would still mate (B) No, because it would not change the offspring mutation rate (C) Yes, because it would sharply decrease the reproduction rate (D) Yes, because the males would die. The correct answer is (C).

OUTPUT: What is the most likely cause of sharply decreasing the reproduction rate of an insect population? (A) the insects mate (B) the offspring mutation rate is constant (C) male insects in a population are treated to prevent sperm production (D) male insects are less than female insects. The correct answer is (C).

#### **In-context samples for MATH**

INPUT: The mean of one set of five numbers is 13, and the mean of a separate set of six numbers is 24. What is the mean of the set of all eleven numbers? The correct answer is 19.

OUTPUT: The mean of one set of six numbers is 24, and the mean of the set of all eleven numbers is 19. What is the mean of the separate set of five numbers?

INPUT: What is  $\frac{5}{6}$  of 30? The correct answer is 25.

OUTPUT: If  $\frac{5}{6}$  X is 25, what is the value of unknown variable X?

INPUT: Twenty-seven increased by twice a number is 39. What is the number? The correct answer is 6.

OUTPUT: Twenty-seven increased by 6 \* 2 is a number. What is the number?

INPUT: In a particular right triangle, the two legs have lengths of 40 inches and 42 inches. What is the area of the triangle? The correct answer is 840.

OUTPUT: In a particular right triangle, the two legs have lengths of 40 inches and X inches. The area of the triangle is 840. What is the value of unknown variable X?

## In-context samples for GSM8K

INPUT: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? The correct answer is 72.0.

OUTPUT: Natalia sold clips to x of her friends in April, and then she sold half as many clips in May. If the number of clips Natalia sell altogether in April and May is 72, what is the value of x?

INPUT: Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden? The correct answer is 35.0.

OUTPUT: Mark has a garden with flowers. He planted plants of three different colors in it. x of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. If Mark have 35 flowers in his garden, what is the value of x? INPUT: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years? The correct answer is 121.0.

OUTPUT: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Ships have been built larger over time, so each new ship has twice as many people as the last ship. In the first hundred year, it consumed 121 people. How many people were on the ship the monster ate in the third hundred years?

INPUT: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of \$200 and spent \$30 on a button-up shirt, \$46 on suit pants, \$38 on a suit coat, \$11 on socks, and \$18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has \$16 left from her budget. How much did Alexis pay for the shoes? The correct answer is 41.0.

OUTPUT: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of \$200 and spent \$30 on a button-up shirt, \$38 on a suit coat, \$11 on socks, \$18 on a belt and \$41 on the shoes. She also purchased suit pants, but lost the receipt for them. She has \$16 left from her budget. How much did Alexis pay for the suit pants?

## In-context samples for Date

INPUT: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY? (A) 03/11/2021 (B) 05/01/2021 (C) 02/23/2021 (D) 04/29/2021 (E) 05/09/2021 (F) 06/12/2021. The correct answer is (B).

OUTPUT: Today is May 1, 2021. What is the date yesterday? (A) 03/11/2021 (B) 04/30/2021 (C) 02/23/2021 (D) 04/29/2021 (E) 05/09/2021 (F) 06/12/2021. The correct answer is (B).

INPUT: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date today in MM/DD/YYYY? (A) 06/20/2021 (B) 05/30/1980 (C) 05/22/2021 (D) 05/30/2021 (E) 04/30/2021 (F) 04/15/2021. The correct answer is (D).

OUTPUT: Today is May 30, 2021, and there is a deadline, which is 2 days away from now. What is the date of the deadline? (A) 06/20/2021 (B) 05/30/1980 (C) 05/22/2021 (D) 06/01/2021 (E) 04/30/2021 (F) 04/15/2021. The correct answer is (D).

INPUT: Tomorrow is 11/12/2019. What is the date a month ago in MM/DD/YYYY? (A) 10/11/1974 (B) 10/10/2019 (C) 10/12/2019 (D) 10/11/2018 (E) 10/16/2019 (F) 10/11/2019. The correct answer is (F).

OUTPUT: The date a month ago is 10/11/2019. What is the date tomorrow? (A) 10/11/1974 (B) 10/10/2019 (C) 10/12/2019 (D) 10/11/2018 (E) 10/16/2019 (F) 11/12/2019. The correct answer is (F).

INPUT: Today, 8/3/1997, is a day that we will never forget. What is the date 10 days ago in MM/DD/YYYY? (A) 08/21/1997 (B) 10/24/1997 (C) 07/24/1997 (D) 07/23/1997 (E) 06/11/1997 (F) 08/14/1997. The correct answer is (C).

OUTPUT: The date 10 days ago is 07/24/1997, what is the date today? (A) 08/21/1997 (B) 10/24/1997 (C) 08/03/1997 (D) 07/23/1997 (E) 06/11/1997 (F) 08/14/1997. The correct answer is (C).

# <span id="page-15-0"></span>C Prompt for Verification

#### Prompt for Verification (Consistency Check)

You will be given two question-answering pairs, (Q1, A1) and (Q2, A2).

Your task is to check the consistency between Q1 and A2.

If (1) A2 can be found in Q1, and (2) A2 is correct, output 'True'.

Otherwise, if Q1 and A2 is not related, or A2 is not correct, output 'False'.

{in\_context\_samples}

{input\_question}

The prompt Icon used to verify the consistency c = T (Q, A, Qb, Rb; Icon) follows the structure outlined above. Similarly, it serves as a general template to be filled with in-context examples, as shown below.

### In-context samples for Verification (Multiple Choice Problem)

INPUT: Q1: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? (A) dry palms (B) wet palms (C) palms covered with oil (D) palms covered with lotion. A1: The correct answer is (A). Q2: If George's palms are dry, what will happen when he rubs them together? (A) His hands will cool down. (B) His hands will warm up. (C) His hands will become wet. (D) His hands will become oily. A2: The correct answer is (B).

OUTPUT: A2 is selecting (B) His hands will warm up. Q1 states that George wants to warm his hands. So A2 matches Q1, they are consistent. True

INPUT: Q1: Which of the following is a trait that a dog does NOT inherit from its parents? (A) the length of its fur (B) the shape of its nose (C) the size of its appetite (D) the color of its fur. A1: The correct answer is (C). Q2: If a dog's trait is not inherited from its parents, what does this trait likely depend on? (A) The dog's breed (B) The dog's environment (C) The dog's age (D) The dog's gender. A2: The correct answer is (B).

OUTPUT: A2 is selecting (B) The dog's environment. Q1 is asking about which trait that a dog does NOT inherit from its parents. So A2 does not matches Q1, they are not consistent. False

INPUT: Q1: A fold observed in layers of sedimentary rock most likely resulted from the (A) cooling of flowing magma (B) converging of crustal plates (C) deposition of river sediments (D) solution of carbonate minerals. A1: The correct answer is (B). Q2: Which of the following is most likely to be caused by the converging of crustal plates? (A) The formation of canyons (B) A fold observed in layers of sedimentary rock (C) Increased volcanic activity in the area (D) The formation of large lakes. A2: The correct answer is (B).

OUTPUT: A2 is selecting (B) A fold observed in layers of sedimentary rock. Q1 is asking about the cause of a fold observed in layers of sedimentary rock. So A2 matches Q1, they are consistent. True INPUT: Q1: If a new moon occurred on June 2, when will the next new moon occur? (A) June 30 (B) June 28 (C) June 23 (D) June 15. A1: The correct answer is (A). Q2: What could we celebrate on or around June 30th, astronomically speaking? (A) A full moon (B) A lunar eclipse (C) A new moon (D) A meteor shower. A2: The correct answer is (C).

OUTPUT: A2 is selecting (C) A new moon Q1 is asking about when will the next new moon occur, if the last new moon occurred on June 2. So A2 does not matches Q1, they are not consistent. False

#### **In-context samples for Verification (Math Reasoning Problem)**

INPUT: Q1: If (x + 2)(x - 3) = 14, find the sum of the possible values of x. A1: The correct answer is 1. Q2: If the sum of the possible values of x in the equation (x + 2)(x - 3) = Y is 1, what is the value of Y? A2: The correct answer is 0.

OUTPUT: A2 is giving 0 as the answer, but in Q1 we see Y=14, so A2 is not correct and they are not consistent. False

INPUT: Q1: Evaluate the expression  $(a^2+b)^2-(a^2-b)^2$ , if a=4 and b=1. A1: The correct answer is 64. Q2: If  $(a^2+b)^2-(a^2-b)^2=64$  and b=1, what is the positive value of a? A2: The correct answer is 4.

OUTPUT: A2 is giving 4 as the answer, where we see in Q1 that a is indeed equal to 4, so A2 is correct and they are consistent. True

INPUT: Q1: The roots of the equation  $2x^2-5x-4=0$  can be written in the form  $x=\frac{m\pm\sqrt{n}}{p}$ , where m,n, and p are positive integers with a greatest common divisor of 1. What is the value of n? A1: The correct answer is 57. Q2: The roots of the equation  $2x^2-5x-4=0$  can be written in the form  $x=\frac{m\pm\sqrt{57}}{p}$ , where m,n, and p are positive integers with a greatest common divisor of 1. What is the value of p? A2: The correct answer is 4.

OUTPUT: A2 is giving 4 as the answer, which is correct and is consistent with Q1. True

INPUT: Q1: Simplify  $(576)^{\frac{1}{4}}(216)^{\frac{1}{2}}$ . A1: The correct answer is 72. Q2: What is the value of X, if  $(X)^{\frac{1}{4}}(216)^{\frac{1}{2}} = 72$ ? A2: The correct answer is 16.

OUTPUT: A2 is giving 16 as the answer, where we see in Q1 that the answer should be 576, so A2 is not correct and they are not consistent. False

INPUT: Q1: Baez has 25 marbles. She loses 20% of them one day. Then a friend sees her and gives her double the amount that Baez has after she lost them. How many marbles does Baez end up with? A1: The correct answer is 60. Q2: Baez has some marbles. She loses 20% of them one day. Then a friend sees her and gives her double the amount that Baez has after she lost them. If Baez ends up with 60 marbles, how many did she start with? A2: The correct answer is 38.

OUTPUT: A2 is giving 38 as the answer, where we see in Q1 that the answer should be 25, so A2 is not correct and they are not consistent. False

INPUT: Q1: Tony's dad is very strict about the washing machine and family members are only allowed to wash 50 total ounces of clothing at a time. Tony doesn't want to break the rules, so he weighs his clothes and finds that a pair of socks weighs 2 ounces, underwear weighs 4 ounces, a shirt weighs 5 ounces, shorts weigh 8 ounces, and pants weigh 10 ounces. Tony is washing a pair of pants, 2 shirts, a pair of shorts, and 3 pairs of socks. How many more pairs of underwear can he add to the wash and not break the rule? A1: The correct answer is 4. Q2: Tony's dad is very strict about the washing machine and family members are only allowed to wash 50 total ounces of clothing at a time. Tony doesn't want to break the rules, so he weighs his clothes and finds that a pair of socks weighs 2 ounces, underwear weighs 4 ounces, a shirt weighs 5 ounces, shorts weigh 8 ounces, and pants weigh 10 ounces. If Tony is washing a pair of pants, 2 shirts, a pair of shorts, x pairs of socks, and 4 pairs of underwear without breaking the rule, what is the value of x? A2: The correct answer is 3. OUTPUT: A2 is giving 3 as the answer, which is correct and is consistent with Q1. True

## In-context samples for Verification (Math Reasoning Problem, Cont'd)

INPUT: Q1: Olga has an aquarium with fish in 5 different colors. There are 12 yellow ones, half as many blue ones, and twice as many green ones as yellow ones. How many fish does Olga have in her aquarium? A1: The correct answer is 42. Q2: Olga has an aquarium with fish in 5 different colors. There are yellow, blue and green ones, as well as some other colors. There are twice as many green fish as yellow fish, and half as many blue fish as yellow fish. If Olga has 42 fish in total in her aquarium, how many yellow fish does she have? A2: The correct answer is 12.

OUTPUT: A2 is giving 12 as the answer, which is correct and is consistent with Q1. True

INPUT: Q1: Jay and Gloria were hosting a 4th of July party at their house. Jay invited 22 people and Gloria invited 36. They wanted to buy small American flags for everyone. The craft store was having a sale on small flags, 5 flags for \$1.00. If they wanted all of their guests to have a flag and they also wanted 1 flag each, how much would they spend at the craft store? A1: The correct answer is 12. Q2: Jay and Gloria were hosting a 4th of July party at their house. Jay invited 22 people and Gloria invited some more. They wanted to buy small American flags for everyone. The craft store was having a sale on small flags, 5 flags for \$1.00. If they wanted all of their guests to have a flag, and themselves a flag each, and ended up spending \$12.00 at the craft store, how many people did Gloria invite? A2: The correct answer is 35.

OUTPUT: A2 is giving 35 as the answer, where we see in Q1 that the answer should be 36, so A2 is not correct and they are not consistent. False

## D Examples of the Augmented Data

## Example on StrategyQA

Original Question (Q): Could the members of The Police perform lawful arrests? Forward Reasoning (R<sup>f</sup> ): Here is the step-by-step reasoning:

- 1. \*\*The Police were a rock band:\*\* Their members were musicians, not law enforcement officers.
- 2. \*\*Law enforcement officers have specific powers:\*\* These powers are granted by the government and include the ability to make arrests.
- 3. \*\*Being in a band called "The Police" does not grant legal authority:\*\* Band names are artistic choices and have no bearing on legal powers.
- \*\*Therefore, the members of The Police could not perform lawful arrests.\*\*

The answer is \*\*no\*\*

Backward Question (Qb): If the members of The Police could not perform lawful arrests, then The Police are likely (A) musicians (B) police officers.

Backward Reasoning (Rb): Here is the reasoning:

- 1. The question implies there are two possible meanings for "The Police".
- 2. One meaning refers to a group that can make lawful arrests. This is the definition of police officers.
- 3. The other meaning refers to a group that does not have the authority to make lawful arrests.
- 4. Since the question states this group cannot make lawful arrests, they cannot be police officers.
- 5. Therefore, "The Police" in this context likely refers to a group known for something other than law enforcement, making them likely musicians.

The answer is (A)

## Example on GSM8K

Original Question (Q): James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?

Forward Reasoning (R<sup>f</sup> ): Here's how to solve the problem:

- \* \*\*Pages per friend per week:\*\* James writes 3 pages/letter \* 2 letters/week = 6 pages/week per friend.
- \* \*\*Total pages per week:\*\* He writes 6 pages/week/friend \* 2 friends = 12 pages/week.
- \* \*\*Total pages per year:\*\* He writes 12 pages/week \* 52 weeks/year = 624 pages/year.

The answer is 624 .

Backward Question (Qb): James writes a letter to 2 different friends twice a week, and he writes 624 pages a year. How many pages long is the letter he writes to each friend?

Backward Reasoning (Rb): Here's how to solve this problem:

- \* \*\*Calculate total letters per week:\*\* James writes 2 letters/week \* 2 friends = 4 letters per week.
- \* \*\*Calculate total letters per year:\*\* Assuming a year has 52 weeks, he writes 4 letters/week \* 52 weeks/year = 208 letters per year.
- \* \*\*Calculate the length of each letter:\*\* He writes 624 pages/year / 208 letters/year = 3 pages per letter.

The answer is 3

## E Dataset Statistics

| Dataset                            | Domain         | License      | Train (Original) | Train (Filtered) | Test  |
|------------------------------------|----------------|--------------|------------------|------------------|-------|
| SQA (Geva et al., 2021)            | Commomsense    | MIT          | 2,061            | 1,544            | 229   |
| CSQA (Talmor et al., 2019)         | Commomsense    | MIT          | 9,741            | 6,478            | 1,140 |
| ARC (Clark et al., 2018)           | Commomsense    | CC BY-SA 4.0 | 1,199            | 1,035            | 1,172 |
| MATH (Hendrycks et al., 2021)      | Math           | MIT          | 7,500            | 2,511            | 5,000 |
| GSM8K (Cobbe et al., 2021)         | Math           | MIT          | 7,379            | 4,293            | 1,339 |
| TabMWP (Lu et al., 2023)           | Math (Tabular) | CC BY-SA 4.0 | 23,059           | 15,544           | 7,686 |
| ANLI (r3) (Nie et al., 2020)       | NLI            | CC BY-NC 4.0 | 100,459          | 883              | 1,200 |
| Date (bench authors, 2023)         | Logic          | Apache       | -                | 200              | 169   |
| BoolQ (Clark et al., 2019)         | Commonsense    | CC BY-SA 3.0 | 9,427            | 0                | 3,270 |
| OpenbookQA (Mihaylov et al., 2018) | Commonsense    | Apache       | 4957             | 0                | 500   |
| e-SNLI (Camburu et al., 2018)      | NLI            | CC BY-NC 4.0 | 549,367          | 0                | 9,824 |
| GSM8K-Rev (Guo et al., 2024)       | Math           | Apache       | -                | 0                | 777   |

Table 6: The datasets used in this work are listed in the order of appearance. For each dataset, we report the domain, the number of original training samples, the number of filtered training samples, and the number of testing samples. Note that the last four datasets are held out and thus contain no filtered training samples. Due to the large size of ANLI's training set, we randomly sampled 2,000 instances, of which 883 remained after filtering. For the Date Understanding dataset, given its small size, we randomly split the data into 200 training and 169 testing samples, and we keep all the training data.