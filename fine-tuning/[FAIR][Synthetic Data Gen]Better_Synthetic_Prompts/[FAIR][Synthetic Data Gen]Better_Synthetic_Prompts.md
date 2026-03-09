# COT-SELF-INSTRUCT: BUILDING HIGH QUALITY SYNTHETIC PROMPTS FOR REASONING AND NON-REASONING TASKS

### Anonymous authors

**000**

**003 004**

**006**

**024**

**028 029 030**

**052 053** Paper under double-blind review

### ABSTRACT

We systematically investigate how models can autonomously generate highquality synthetic data to significantly enhance their capabilities during posttraining stage. We propose CoT-Self-Instruct, a new synthetic data generation pipeline that instruct LLMs to first reason and plan based on the given seed tasks, then generate a new synthetic prompt of similar quality and complexity for use in LLM training, followed by filtering high quality data with various automatic metrics. In verifiable reasoning, our synthetic data significantly outperforms existing datasets, such as S1k with approximately 1k training samples and OpenMathReasoning with 10k samples, across MATH500, AMC23, AIME24, GPQA-Diamond. For non-verifiable instruction-following tasks, our synthetic data also surpasses the performance of Wildchat data on the Alpaca Eval and other benchmarks.

# 1 INTRODUCTION

The transformative rise of Large Language Models (LLMs) has initiated a substantial paradigm shift in the domain of deep learning [\(Zhang et al., 2023;](#page-11-0) [Guo et al., 2023;](#page-9-0) [Long et al., 2024\)](#page-10-0). Despite these advancements, the development of NLP models continues to rely heavily on large volumes of high-quality data [\(Gandhi et al., 2024;](#page-9-1) [Abdin et al., 2024\)](#page-9-2). However, acquiring such data from human sources can often be challenging or even impractical due to factors such as high costs, data scarcity, and privacy concerns [\(Kurakin et al., 2023\)](#page-9-3). Furthermore, several studies [\(Hosking et al.,](#page-9-4) [2023;](#page-9-4) [Singh et al., 2023;](#page-10-1) [Gilardi et al., 2023\)](#page-9-5) have pointed out that human-generated data, being inherently prone to biases and errors, may not always be ideal for model training or evaluation. In this context, synthetic data emerges as a viable alternative for obtaining high-quality datasets.

Synthetic data is artificially generated to mimic the characteristics and patterns of real-world data. This type of data is produced using algorithms [\(Saxton et al., 2019\)](#page-10-2), generative models [\(Borisov](#page-9-6) [et al., 2022;](#page-9-6) [Meng et al., 2022\)](#page-10-3), or simulations [\(Vezhnevets et al., 2023\)](#page-11-1), rather than being directly created by humans [\(Liu et al., 2024\)](#page-10-4). It presents a promising solution for training models, particularly in scenarios where real-world data is scarce, expensive, or difficult to obtain. The Self-Instruct method [\(Wang et al., 2022b\)](#page-11-2) leverages large language models (LLMs) to generate instruction-following data across diverse scenarios. This approach starts by selecting a small set of "seed instruction-following samples" and then prompts LLMs to replicate this format to produce additional demonstrations. A significant challenge with this method is ensuring the quality of the data and its effectiveness for language model training. Some related work aims to enhance the usability of model-generated synthetic data by increasing the complexity of queries [\(Liu et al., 2023;](#page-10-5) [Zeng](#page-11-3) [et al., 2024\)](#page-11-3), maintaining semantic diversity [\(Ding et al., 2023\)](#page-9-7), and scaling the synthetic dataset [\(Yuan et al., 2023\)](#page-11-4).

Although these proposed methods have enhanced the Self-Instruct recipe, the process of generating high-quality synthetic data and optimizing its use in both reasoning and non-reasoning tasks remains insufficiently understood. In this paper, we systematically investigate how models can autonomously generate high-quality synthetic data to significantly enhance their capabilities during the post-training stage of large language models (LLMs). Specifically, we examine strategies for effectively utilizing LLMs as prompt generators, filtering high-quality data from the generated synthetic data, and evaluating the final performance when employing this synthetic data.

In this paper, we present our contributions based on experimental results in both reasoning and nonreasoning tasks. For reasoning tasks, we demonstrate that synthetic reasoning instructions generated using Chain-of-Thought (CoT) techniques outperform those generated without CoT. The Answer-Matching Filtering CoT-Self-Instruct method, which incorporates answer filtering, achieves the best performance due to the availability of accurate labels. Additionally, the RIP filter emerges as the best option when accurate labels for filtering are unavailable, surpassing the performance of the self-consistency filter. For non-reasoning tasks, we initially employed the self-Instruct template, which does not utilize CoT reasoning in generating synthetic data. We then enhanced this process by integrating CoT, prompting the model to devise a plan before generating new synthetic data. This planning step significantly improves the quality of the generated data. We evaluated two filtering methods: the RIP filter [\(Yu et al., 2025\)](#page-11-5) and the reward model, Armo [\(Wang et al., 2024\)](#page-11-6). Our results indicate that the RIP filter is particularly effective for data filtering. Finally, we assessed the synthetic data using online DPO and found that high-quality synthetic data not only enhances DPO [\(Rafailov et al., 2023\)](#page-10-6) training but also performs well in online DPO scenarios.

# 2 RELATED WORK

**054 055 056**

**059**

**061**

**072 073 074**

**079**

**094**

Synthetic Data Generation There are many different ways to generate synthetic data, some of them are using more powerful model such as GPT models from OpenAI to generate such as [Xu](#page-11-7) [et al.](#page-11-7) [\(2023\)](#page-11-7) create a pipeline that can automatically generate high quality multi-turn chat corpus by leveraging ChatGPT to engage in a conversation with itself. Using such powerful model can distill their knowledge into smaller model, which can improve their ability. But in this paper, we studied that how to use model itself generate high quality synthetic data and eventually help to improve itself.

Self-Instruct [\(Wang et al., 2022b\)](#page-11-2) first propose a framework that let model prompt the language model with seed data as few shot and let model itself generate new synthetic data. However, it is challenge to maintain synthetic data quality when base model is not very powerful. Following this, Evol Instruct [\(Zeng et al., 2024\)](#page-11-3) tried to increase prompt complexity by let language model re-write prompt following their template. However, Evol instruct paper used OpenAI model to re-write their instruct and it works well, but when we test it with llama 3.1 8b instruct model, it struggle to follow their instructions.

Synthetic Data Selection Data selection is a critical component for post-training with synthetic data. Previously, LLM training was regarded as largely dependent on the size of available training data [Mishra et al.](#page-10-7) [\(2021\)](#page-10-7); [Wei et al.](#page-11-8) [\(2021\)](#page-11-8); [Wang et al.](#page-11-9) [\(2022c\)](#page-11-9). More recent work has revealed that training on a smaller yet higher-quality curated set of prompts tends to be more effective in improving models' both instruction following and reasoning capabilities [\(Zhou et al., 2024;](#page-11-10) [Chen et al.,](#page-9-8) [2024;](#page-9-8) [Muennighoff et al., 2025;](#page-10-8) [Ye et al., 2025\)](#page-11-11). In addition to preprocessing such as deduplication of similar prompts using similarity metrics such as ROUGE-L similarity score [\(Wang et al., 2022b\)](#page-11-2), clustering [\(Chen et al., 2023\)](#page-9-9), as language models become more powerful, prompt curation can also be facilitated by using LLMs as a quality judge. Recent work studies employing powerful language models to measure the complexity, diversity and quality of instructions [\(Lu et al., 2023;](#page-10-9) [Chen et al.,](#page-9-8) [2024;](#page-9-8) [Touvron et al., 2023;](#page-11-12) [Dubey et al., 2024;](#page-9-10) [Li et al., 2024\)](#page-10-10). The the success of RLHF for posttraining [\(Stiennon et al., 2020;](#page-10-11) [Rafailov et al., 2024\)](#page-10-12), has attracted more attention to collecting large scale and high quality preference data. Most work involving preference optimization employs existing methods derived from pretraining and instruction-tuning [\(Touvron et al., 2023;](#page-11-12) [Muennighoff](#page-10-8) [et al., 2025\)](#page-10-8), such as deduplication, clustering, quality classifiers or filtering heuristics. However, such methods overlook the importance of the

### 3 COT-SELF-INSTRUCT-PLANNER? META-SELF-INSTRUCTION WITH COT

Our approach first assumes access to a base language model, and a small amount of high quality human-annotated seed data. We then aim to explore models' capability in

• Self-Instruction Creation with Chain-of-Thoughts (CoT): given sample human-annotated seed instructions, the ability to reason step by step to come with instructions of similar complexity and domain.

**114 115**

- Self-Instruction Curation: the ability to curate high-quality synthetic instructions for selftraining.
- Self-Instruction as Training Targets: train LLMs on the ability to generate high quality synthetic instructions.

These skills are crucial components for scaling self-evolving language models with synthetic data. Specifically we study these skills in both reasoning and non-reasoning domains.

Self-Instruction Creation with CoT The process of Self-CoT-Instruction Creation starts with a small seed set of instructions as instruction pool. Multiple instructions are sampled at random from the instruction pool, and then used to few-shot prompt a language model to generate a series of intermediate reasoning steps, followed by a new instruction. Unlike [Wang et al.](#page-11-2) [\(2022b\)](#page-11-2) which directly prompts the model to write new instructions given a list of 8 seed instructions, each time we show LLM two few shot sample instructions, and ask it to carefully analyze both instructions (such as domain, complexity, purpose). After analyzing the seed instructions, reflecting on what makes them high quality data, LLM is prompted to reason step by step, come up with a plan to generate a new self-contained instruction that is of similar quality and complexity as the given seed instructions, and output the final synthetic instruction that satisfies such requirements.

Prompt Curation Not all synthetic instructions are well-formed and answerable, or are effective in base model self-training. We therefore apply additional prompt curation methods to select higher quality instructions for self-training. [Yu et al.](#page-11-5) [\(2025\)](#page-11-5) shows Rejection Instruction Prompt (RIP) filtering which curate prompts based on its response quality distributions measured by response rewards can effectively filter out low quality general instruction following synthetic instructions for self-training. We therefore adopt RIP metrics to curate synthetic non-reasoning instructions. For reasoning domain, due to the lack of verifiable rewards for synthetic prompts, we explore several filtering methods, including self-consistency metric [\(Wang et al., 2022a;](#page-11-13) [Prasad et al., 2024\)](#page-10-13) to filter reasoning instructions' answer-ability and complexity.

Self-Instruction as Training Targets Generating high quality instructions are essential for developing self-evolving language model. To accomplish such goals, we explore training a model to generate high quality instructions, inducing such meta-reasoning capabilities (i.e., thinking about how to a good question). More specifically, we consider seed instructions as training input x, and synthetic prompts after filtering as training output y, and supervised finetune a prompt generator model M(y|x) to generate a new synthetic prompt given seed instructions x. We then prompt the finetuned prompt generator model to generate high quality instructions for self-training.

Self-training To examine the quality of the synthetic instructions generated by our proposed method, we self-train a base language model on the set of generated instructions. We then compare the performance of the self-trained LLMs with models trained on human-annotated data and on seed instructions in reasoning and non-reasoning domains respectively.

# 4 EXPERIMENTAL SETUP

We study the effectiveness of synthetic prompt generations reasoning and non-reasoning domains along the following two axes: instruction for prompt generation, and prompt curation methods.

### 4.1 REASONING

Seed Instructions We use s1k [\(Muennighoff et al., 2025\)](#page-10-8) reasoning instructions as our seed reasoning tasks. The S1k dataset consists of 1000 high-quality, diverse and difficult reasoning prompts, with solutions from DeepSeek R1 Thinking. To conduct self-training with verifiable reward we carefully select a subset of s1k consisting of 893 verifiable reasoning instructions. We then use this subset as seed instruction pool to generate more verifiable reasoning instructions.

Prompt Generation Template To study how CoT could help on generating verifiable reasoning tasks, we explore applying the different prompt generation templates to Qwen3-4B-Base models, Qwen3-4B model with think mode and Qwen3-4B model with nothink mode.

**169**

**184**

**186 187**

**204**

- Qwen3-4B-Base as Instruction Generator + NoCoT Template: By few-shot prompting the Qwen3-4B-Base model, a non-reasoning Qwen3 pretrained model, with randomly sampled 2 reasoning questions from s1k verifiable set, it instructs the Qwen3-base model to output a new self-contained reasoning question directly, without any chain-of-thoughts or reasoning.
- Qwen3-4B(NoThink mode) as Instruction Generator + NoCoT Template: Prompting the Qwen3-4B model with No Think mode using the same few-shot template as in above, forcing it to not reason at all and directly generate a new reasoning question.
- Qwen3-4B(Think mode) as Instruction Generator + CoT Template: Prompting the Qwen3-4B model with Think mode using the few-shot self-instruction template, instruct the model to first reason step by step given 2 randomly selected seed instructions and then output a new self-contained verifiable reasoning question as final output.
- Qwen3-4B(Think mode) as Instruction Generator + CoT-Instruct-Then-Answer Template: Prompting the Qwen3-4B model with Think mode with the few-shot self-instruction template, instruct it to first reason step by step given 2 randomly selected seed instructions, then do another round of reasoning to solve its own generated prompt. The final output is the newly generated synthetic prompt along with the final answer to it.

Build Train Data for RL with Verifiable Reward (RLVR) To build labels for RLVR training initialized from Qwen3-4B-Base, we additionally sample Qwen3-4B-Base answers K times with random seeds and assign one of them as the pseudo label for RLVR training. We explore multiple ways of pseudo-labeling: (1) by selecting the majority-voted answers out of K Qwen3-4B-Base responses; (2) by selecting the highest scored answers out of K Qwen3-4B-Base responses, annotated by a reasoning reward model infly/INF-ORM-Llama3.1-70B [\(Minghao Yang, 2024\)](#page-10-14). For baseline RLVR on original S1k prompts, we use the public available DeepSeek R1 [\(Guo et al., 2025\)](#page-9-11) thinking solution from simplescaling/s1K-1.1. For self-instruction and response sampling we use temperature = 0.7 and top-p=0.8 for Qwen3-Base-4B models and temperature = 0.6 and top-p=0.95 for Qwen3-4B model.

Prompt Curation We explore multiple synthetic reasoning prompt curation methods:

- Self-Consistency Based Filter: for models trained with synthetically generated prompts and majority-voted labels, we curate prompts by calculating the ratio of the number of votes of majority-voted answers over total votes, and then filtering out instructions with lower self-consistency ratio.
- RIP Based Filter: for models trained with synthetically generated prompts and rewardmodel annotated solutions, we curate synthetic prompts by applying RIP metrics developed by [Yu et al.](#page-11-5) [\(2025\)](#page-11-5) where we annotate all K sampled responses given a single prompt, and compute the RIP rejected scores (the lowest score out of all K scores), rejected length (the response length of the lowest-scored model response), and reward score gap (the score difference between highest and lowest scores given a prompt)

RLVR Training All our reasoning experiments are GRPO training initialized from Qwen3-Base-4B model with verifiable rewards. For hyperparameter, we use a cosine learning rate scheduler with a peak value of 1e − 6 and adopt the AdamW optimizer for the policy model. We set the number of training epochs to 40 with a batch size of 128. For rollout, we sample 32 rollouts for each prompt with temperature = 0.6 and top-p=0.95, with a maximum length of 4096 tokens.

### 4.2 NON-VERIFIABLE INSTRUCTION FOLLOWING

Seed Instruction We utilized the Wildchat-RIP-Filtered-by-8b-Llama dataset[1](#page-3-0) as our seed prompt, which comprises 24.3k non-verifiable instruction-following prompts. The original data from Wildchat includes noisy prompts, so to ensure high quality, we applied the RIP filter [\(Yu et al.,](#page-11-5) [2025\)](#page-11-5). Ultimately, we retained only 4k high-quality prompts from the initial 24.3k.

<span id="page-3-0"></span><sup>1</sup><https://huggingface.co/datasets/facebook/Wildchat-RIP-Filtered-by-8b-Llama>

> **218 219**

> **221**

**224**

<span id="page-4-0"></span>Table 1: Think v.s. NoThink in Generating Reasoning Prompts on pseudo labels sampled from *Qwen3-4B* model: We conduct GRPO-training using Qwen3-4B-Base model on selected S1k verifiable prompts and Synthetic prompts from different Self-Instruct Template with pseudo labels sampled from Qwen3-4B model. We report pass@1 averaged over 16 seeds on MATH500, AMC23, AMIE24, GPQA-Diamond.

(Random = take random sampled Qwen3-4B response as pseudo-label; MajVote = take majorityvoted Qwen3-4B response as pseudo-label; RmChosen = take highest-scored Qwen3-4B response by a Reward Model (RM) as pseudo-label.)

|                                                                                    | # Train<br>examples | MATH<br>500 | AIME<br>24 | AMC<br>23 | GPQA<br>Diamond | Avg. |
|------------------------------------------------------------------------------------|---------------------|-------------|------------|-----------|-----------------|------|
| Qwen3-4B-Base (Zero-Shot)                                                          | -                   | 67.4        | 10.6       | 42.0      | 24.2            | 36.1 |
| S1k Prompts + Gold Label (by R1)                                                   | 893                 | 68.6        | 18.5       | 51.3      | 40.1            | 44.6 |
| OpenMathReasoning Prompts + Gold Label                                             | 10,000              | 79.0        | 13.3       | 62.5      | 35.4            | 47.5 |
| S1k Prompts                                                                        |                     |             |            |           |                 |      |
| + MajVote                                                                          | 893                 | 71.3        | 13.7       | 51.5      | 38.7            | 43.8 |
| Qwen3-4B (NoThink mode) as Instruction Generator + NoCoT Template                  |                     |             |            |           |                 |      |
| + Random                                                                           | 5000                | 81.1        | 16.3       | 58.1      | 42.5            | 49.5 |
| + MajVote                                                                          | 5000                | 80.8        | 15.6       | 57.2      | 43.7            | 49.3 |
| + MajVote + Consistency Filter                                                     | 3467                | 80.9        | 17.7       | 63.9      | 46.3            | 52.2 |
| + MajVote + Consistency Filter (893-subsampled)                                    | 893                 | 82.1        | 19.1       | 64.3      | 42.0            | 51.8 |
| Qwen3-4B (Think mode) as Instruction Generator + CoT-Instruct Template             |                     |             |            |           |                 |      |
| + Random                                                                           | 5000                | 84.3        | 20.2       | 65.5      | 43.7            | 53.4 |
| + MajVote                                                                          | 5000                | 82.9        | 21.9       | 65.3      | 44.4            | 53.6 |
| + MajVote + Consistency Filter                                                     | 3972                | 83.7        | 21.3       | 68.8      | 44.2            | 54.5 |
| + MajVote + Consistency Filter (893-subsampled)                                    | 893                 | 83.6        | 23.1       | 62.9      | 41.9            | 52.9 |
| Qwen3-4B (Think mode) as Instruction Generator + CoT-Instruct-Then-Answer Template |                     |             |            |           |                 |      |
| + Reference Answer                                                                 | 5000                | 84.9        | 20.4       | 62.2      | 44.4            | 53.0 |
| + MajVote + Answer-Matching Filter                                                 | 2926                | 86.5        | 24.6       | 72.3      | 45.5            | 57.2 |
| S1k Prompts                                                                        |                     |             |            |           |                 |      |
| + RmChosen                                                                         | 893                 | 68.6        | 18.5       | 51.3      | 40.1            | 44.6 |
| Qwen3-4B (Think mode) as Instruction Generator + CoT-Instruct Template             |                     |             |            |           |                 |      |
| + Random                                                                           | 5000                | 84.3        | 20.2       | 65.5      | 43.7            | 53.4 |
| + RmChosen                                                                         | 5000                | 82.9        | 22.5       | 64.8      | 42.7            | 53.2 |
| + RmChosen + RIP Filter                                                            | 3651                | 85.2        | 24.4       | 71.1      | 46.8            | 56.9 |

**249 250 251**

> **252 253 254**

> > Unlike reasoning tasks, our instruction-following seed prompts vary across different domains. For instance, combining prompts from storytelling and coding could result in unnatural synthetic prompts. To address this, we categorized all seed data into 8 distinct categories. During sampling, we select 2 seed prompts from the same category to serve as few-shot prompts. Our seed data spans 8 categories: Writing & Storytelling, Technical & Programming, Creative & Design, Data & Analysis, Education & Research, Communication & Support, Business & Marketing, and Miscellaneous.

**256 259**

Prompt Generation Template To investigate how Chain-of-Thought (CoT) reasoning can aid in generating non-verifiable instruction-following tasks, we experimented with various templates using the Llama-3.1-8B-Instruct model [\(Grattafiori et al., 2024\)](#page-9-12). By employing different templates, we controlled the length of CoT when generating synthetic prompts.

**261 262 263**

**264**

**260**

• No CoT Template: We used few-shot prompting with the Llama-3.1-8B-Instruct model, selecting 2 instruction prompts at random from a single cluster within the 4k RIPfiltered instruction data. The model generates a new synthetic prompt directly, without incorporating CoT. The template is illustrated in [Figure 2.](#page-12-0)

**269**

• Short CoT Template: Similar to the No CoT approach, we prompted the Llama-3.1-8B-Instruct model to generate a new synthetic prompt, but utilized a different template, as shown in [Figure 3.](#page-12-1)

> **294**

**309**

**321 322 323**

<span id="page-5-1"></span>Table 2: Think v.s. NoThink in Generating Reasoning Prompts on pseudo labels sampled from *Qwen3-4B-Base* model: We conduct GRPO-training using Qwen3-4B-Base model on selected S1k verifiable prompts and Synthetic prompts from different Self-Instruct Template on . We report pass@1 averaged over 16 seeds on MATH500, AMC23, AMIE24, GPQA-Diamond. (Base-Random = take random sampled Qwen3-Base-4B response as pseudo-label; Base-MajVote = take majority-voted Qwen3-Base-4B response as pseudo-label; Base-RmChosen = take highestscored Qwen3-Base-4B response by a Reward Model (RM) as pseudo-label.)

| # Train                     | MATH<br>500                                                                                                                                                   | AIME<br>24                                                                                                                                                                                                                                           | AMC<br>23                    | GPQA<br>Diamond              | Avg.                         |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|------------------------------|------------------------------|
| -<br>893                    | 67.4<br>68.6                                                                                                                                                  | 10.6<br>18.5                                                                                                                                                                                                                                         | 42.0<br>51.3                 | 24.2<br>40.1                 | 36.1<br>44.6                 |
|                             |                                                                                                                                                               |                                                                                                                                                                                                                                                      |                              |                              | 40.4                         |
|                             |                                                                                                                                                               |                                                                                                                                                                                                                                                      |                              |                              |                              |
| 5000<br>5000                | 75.7<br>76.2                                                                                                                                                  | 13.1<br>11.7                                                                                                                                                                                                                                         | 51.4<br>51.7                 | 28.0<br>30.5                 | 42.1<br>42.5<br>43.6         |
| 5000                        | 75.3                                                                                                                                                          | 10.9                                                                                                                                                                                                                                                 | 49.8                         | 30.9                         | 41.7                         |
|                             |                                                                                                                                                               |                                                                                                                                                                                                                                                      |                              |                              | 42.6                         |
| 5000<br>5000<br>1672<br>893 | 75.5<br>76.1<br>77.0<br>75.7                                                                                                                                  | 11.0<br>12.3<br>13.5<br>12.3                                                                                                                                                                                                                         | 52.2<br>54.5<br>55.3<br>53.9 | 31.4<br>31.3<br>31.4<br>31.6 | 42.5<br>43.5<br>44.3<br>43.4 |
| 893                         | 75.4                                                                                                                                                          | 10.2                                                                                                                                                                                                                                                 | 53.1                         | 31.2                         | 42.5                         |
| 5000<br>3492                | 77.8<br>76.8                                                                                                                                                  | 14.8<br>12.3                                                                                                                                                                                                                                         | 50.9<br>53.4                 | 31.7<br>34.9                 | 43.8<br>44.3                 |
| 5000<br>5000                | 75.5<br>77.2                                                                                                                                                  | 11.0<br>14.6                                                                                                                                                                                                                                         | 52.2<br>54.5                 | 31.4<br>32.9                 | 42.5<br>44.8<br>46.4         |
|                             | examples<br>893<br>Qwen3-4B-Base as Instruction Generator + NoCoT Template<br>2815<br>1757<br>Qwen3-4B-Base as Instruction Generator + NoCoT Template<br>2456 | 75.1<br>77.5<br>Qwen3-4B (NoThink mode) as Instruction Generator + NoCoT Template<br>75.9<br>Qwen3-4B (Think mode) as Instruction Generator + CoT-Instruct Template<br>Qwen3-4B (Think mode) as Instruction Generator + CoT-Instruct Template<br>8.0 | 10.4<br>13.1<br>11.9<br>13.1 | 47.3<br>54.5<br>50.9<br>58.6 | 28.7<br>29.0<br>31.6<br>35.7 |

• Long CoT Template: Again, following the No CoT method, we prompted the Llama-3.1-8B-Instruct model to create a new synthetic prompt, using yet another template, depicted in [Figure 1.](#page-6-0)

Build Training Data for DPO To verify the quality of our synthetic prompts, we employ them in Direct Preference Optimization (DPO) training to assess their effectiveness. For each prompt in the DPO training, we generate 64 responses using the Llama-3.1-8B-Instruct model. These responses are then annotated with the Athene-RM-8B model [2](#page-5-0) . Compared to human prompts, our synthetic prompts tend to be more complex, resulting in longer average response lengths, which can lead to length explosion. During DPO training, the evaluation judge often favors longer responses, potentially causing response lengths to increase over time [Yuan et al..](#page-11-14) To mitigate this issue, we adopted the approach outlined by [Wu et al.](#page-11-15) [\(2024\)](#page-11-15), which involves combining the reward score with length information to determine the preferred response. This method ensures that shorter responses are selected when scores are similar. We applied a length normalization coefficient (lc) of 0.2 for the length-normalized reward. Based on this adjusted reward, we then select the chosen and rejected responses.

DPO Training Following the Instruct setup in [Meng et al.](#page-10-15) [\(2024\)](#page-10-15), we utilize the DPO training approach with the off-the-shelf LLama 3.1-8B-Instruct, leveraging the fairseq2 library [\(Balioglu,](#page-9-13)

<span id="page-5-0"></span><sup>2</sup><https://huggingface.co/Nexusflow/Athene-RM-8B>

Figure 1: Long CoT Prompt Generation Template for Non-verifiable instruction following task.

<span id="page-6-0"></span>You are a **prompt generator assistant**. Your goal is to create diverse and creative synthetic prompts.

326 327

Please follow the steps below to create synthetic prompts.

329 330 331

328

Step 1: Carefully read #Prompt 1# and #Prompt 2#. Identify and list all the common elements between these two prompts. If no common elements are found, list the main elements from each prompt.

332333334

Step 2: Develop a comprehensive plan based on the #Common Elements List# or #Main Elements List# from Step 1. This plan will guide the generation of new synthetic prompts that are similar to the original prompts.

335 336 337

Step 3: Execute the plan step by step and provide one #Synthetic Prompt#.

338

Please reply strictly in the following format:

339 340

- Step 1 #Common Elements List# or #Main Elements List#:

340341

Step 2 #Plan#:Step 3 #Synthetic Prompt#:

342 343

#Prompt 1#:
{INSTRUCTION 1}

344345346

#Prompt 2#:
{INSTRUCTION 2}

347 348 349

350 351

353

355

357

359

360

Table 3: Synthetic Data Performance on Instruction Following Task

<span id="page-6-1"></span>AlpacaEval2 Prompting Filter Train Prompts Method Method LC Win Win Random 49.1 45.5 Wildchat RIP 57.6 58.8 52.9 49.3 Random No RIP 55.2 53.2 CoT Synthetic Random 56.4 52.5 Short Prompts RIP 59.0 57.3 CoT 58.5 59.8 Random Long RIP 63.5 66.2 CoT

361 362

364

366

367

368

369

370

371

372373

374

375

376

377

2023). We use a batch size of 64 and learning rates of 1e-6 with dropout rate of 0.0 and a  $\beta$  value of 0.1 throughout the experiments.

To ensure a fair comparison across methods, we randomly sampled 7k data points from the DPO training dataset. As shown in Table 3, our synthetic prompts outperform the original Wildchat prompts when used with DPO training. Specifically, when evaluating synthetic prompts, we observed that even without a chain of thought (CoT), the synthetic prompts surpass the original Wildchat prompts in performance. Furthermore, as the length of the CoT increases—from no CoT to short CoT and then to long CoT—the quality of the prompts improves, as evidenced by enhanced model performance.

**Filter Synthetic Prompts** To ensure the quality of our synthetic prompts, we utilize the RIP method (Yu et al., 2025) for filtering. It's important to note that we experimented with using a reward model, such as ArmoRM-Llama3-8B-v0.1 <sup>3</sup>, by inputting our template with two random seed prompts and allowing the model to rate the responses. However, since the reward model may

<span id="page-6-2"></span> $<sup>^3</sup>$ https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1

**384**

not have been specifically trained for prompt generation tasks, it struggled to provide significant improvements over random selection. In contrast, we found that the RIP method [\(Yu et al., 2025\)](#page-11-5) is effective for selecting synthetic prompts. By employing this method, we generate 32 responses for each synthetic prompt and have Athene rate the responses. We then calculate the RIP score using the reward scores of the responses.

Online DPO training For all tasks, we initialize model parameters using the Llama-3.1-8B-Instruct model. During training, we use the default sampling parameters (temperature=1.0, top-p=1.0) to generate exploration rollouts. We train models using the fairseq2 library [\(Balioglu, 2023\)](#page-9-13), where model inference is performed with the vllm library [\(Kwon et al., 2023\)](#page-9-14). We run all experiments using 32 H200 GPUs for training workers and 8 H200 GPUs for inference workers (16 for combined task training).

[Table 4](#page-7-0) shows our synthetic data with online DPO training results. Our synthetic prompts also out perform wildchat data largely.

<span id="page-7-0"></span>Table 4: Synthetic Data Performance on Instruction Following Task with Online DPO

|                               | AlpacaEval2  |              |  |
|-------------------------------|--------------|--------------|--|
| Train Prompts                 | LC Win       | Win          |  |
| Wildchat<br>Synthetic Prompts | 80.1<br>83.2 | 74.8<br>81.2 |  |

# 5 EXPERIMENT RESULTS

### 5.1 REASONING TASKS

Synthetic Reasoning Instructions generated by CoT-Self-Instruct outperforms No-CoT template In [Table 1](#page-4-0) where models are GRPO trained on Qwen3-4B generated prompts and responses, Qwen3-4B-base models GRPO trained on synthetic instructions generated by think mode with CoT-Instruct template achieves an average accuracy of 53.6%, outperforming 49.3 % from models trained on self-instructions by no think mode with NoCoT template. Similarly in table [Table 2](#page-5-1) where models are GRPO trained on Qwen3-4B & Qwen3-4B-Base generated prompts and Qwen3-4B-Base generated responses, Qwen3-4B-base models GRPO trained on synthetic instructions generated by think mode with CoT-Instruct template achieves an average accuracy of 43.4%, outperforming those trained on self-instructions by no think mode with NoCoT template and base model with NoCoT self-instruct template.

Answering-Matching Filtering outperforms Self-Consistency filtering In [Table 1,](#page-4-0) by training Qwen3-Base-4B model on self-instructions generated by prompting Qwen3 model to reason step by step to propose a new reasoning tasks given seed instructions, followed by immediately reason again to answer its own generated prompt in the same prompting, the model achieves 53.0% average accuracy on par with 53.4% by training on self-instructions with CoT-Self-Instruct template by the same Qwen3 model. Furthermore, apply Answer-Matching filter significantly improves the model performance as compared to consistency filtering, from 53.0% to 57.2 % by filtering out 40% of the CoT-Self-Instructions.

CoT-Self-Instructions curated by Self-Consistency and RIP filtering methods outperforms no filtering For self-instructions and majority-voted pseudo-labels in [Table 1,](#page-4-0) applying consistency filtering further improves the model's reasoning performances. Specifically, by filtering out prompts with self-consistency lower than 0.5 (i.e. the votes of majority-voted solution are lower than 0.5 of total votes), it can improve accuracy of Qwen3-Base-4B trained models from 49.3% to 52.2% on NoCoT self-instruct with nothink mode, and 53.6% to 54.5% with CoT-Self-Instruct think mode. Similarly for self-instructions and majority-voted pseudo-labels in [Table 2,](#page-5-1) filtering out instructions with lower consistency ( self-consistency ≤ 0.5) improves over no filtering from 43.5% to 44.3 % on CoT-self-instructions. In addition, we observed that sometimes filtering both lower consistent and higher consistent instructions further improves, in our case, NoCoT-self-instructions by no

think mode. Similarly findings are reported in [Prasad et al.](#page-10-13) [\(2024\)](#page-10-13), where filtering out reasoning instructions with lower self-consistent ratio outperforms no filtering in DPO training.

For self-instructions and pseudo-labels determined by the highest-scored model response by a reward model, RIP filtering further improves models' reasoning average accuracy from 53.2% to 56.9% by filtering out instructions with lower 25% quantile RIP rejected scores for Qwen3 generated pseudo-labels in [Table 1,](#page-4-0) improves from 44.8% to 46.4% by filtering out nearly half of the total selfinstructions with lower 50% quantile RIP rejected scores for Qwen3-Base generated pseudo-labels. We also explore other RIP metrics based filtering, including length of rejected response, and results show that filtering out prompts with shorter Qwen3-Base generated rejected response also improves performance from 43.8% to 44.3% on NoCot-Self-Instruct generated instructions in [Table 2.](#page-5-1)

CoT-Self-Instruct-Then-Answer + Answering Matching Filter achieves the best reasoning performances among all CoT and NoCoT curated self-instruct methods In [Table 1,](#page-4-0) by training Qwen3-Base-4B model on self-instructions generated by prompting the Qwen3-4B model to reason step by step to propose a new reasoning tasks given seed instructions, followed by immediately reason again to answer its own generated prompt in the same prompting, the model achieves 53.0% average accuracy on par with 53.4% by training on self-instructions with CoT-Self-Instruct template by the same Qwen3 model. Furthermore, apply Answer-Matching filter significantly improves the model performance as compared to consistency filtering, from 53.0% to 57.2 % by filtering out 40% of the CoT-self-instructions.

High Quality Synthetic Prompts generated by CoT-Instruct significantly outperform seed instructions and others publicly available reasoning prompts CoT-Self-Instructions outperforms S1k according to [Table 1,](#page-4-0) where models trained on 893 consistency-filtered CoT-Self-Instructions achieves 52.9%, much higher than 43.8 % S1k prompts, both trained with majority-voted Qwen3 model generated solutions, and higher than 47.5 % from 10K OpenMath-Reasoning instructions with gold labels. Self instructions from CoT-Self-Instruct-Then-Answer template with answermatching filtering further improves the performances, achieving the highest reasoning performances compared to synthetic and seed instructions.

### 5.2 NON-VERIFIABLE INSTRUCTION FOLLOWING TASK

Synthetic instruction following data generated by Long CoT outperforms short NoT and no-NoT template [Table 3](#page-6-1) demonstrates that allowing the model to create a plan beforehand significantly enhances the quality of synthetic data. This approach outperforms simply asking the model to provide an explanation before generating synthetic data, which in turn is more effective than the self-Instruct method without Chain of Thought (CoT). Regardless of the type of synthetic data used, the RIP method proved to be effective across all variations.

RIP filtering further improves CoT-Self-Instructions compared to random selection We applied the RIP filter to each method and found it to be effective across various types of prompts. As shown in [Table 3,](#page-6-1) the RIP filter significantly improved the original Wildchat prompts, increasing the Length Control win rate on Alpaca Eval from 49.1 to 57.6. Although the RIP filter also enhances performance across three different types of synthetic data, the improvement is less pronounced compared to its effect on the original Wildchat prompts. This is mainly because the original Wildchat data is relatively noisy, whereas the synthetic prompts are of higher quality. For comparison, we also tested ArmoRM-Llama3-8B-v0.1 as a data selection method. When applied without a chain of thought (CoT), it achieved a Length Control win rate of 54.9 on Alpaca Eval, which is higher than Random's 52.9 but not as effective as the RIP filter's 55.2.

### 6 CONCLUSION

In this paper, we propose CoT-Self-Instruct, a synthetic data creation and curation pipeline that instruct LLMs to plan and reason to come up with new synthetic prompts given seed instructions. We show that applying such pipeline improves models' reasoning and general instruction following capabitilies in both reasoning and non-reasoning domains, creating high quality synthetic instructinos for RL training, surpassing existing seed human-annoated instructions and public datasets on challenging benchmarks.

# REFERENCES

<span id="page-9-10"></span>**509**

<span id="page-9-4"></span>**529 530**

**538 539**

- <span id="page-9-2"></span>Marah Abdin, Jyoti Aneja, Harkirat Behl, Sebastien Bubeck, Ronen Eldan, Suriya Gunasekar, ´ Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. *arXiv preprint arXiv:2412.08905*, 2024.
- <span id="page-9-13"></span>Can Balioglu. fairseq2, 2023. URL [http://github.com/facebookresearch/](http://github.com/facebookresearch/fairseq2) [fairseq2](http://github.com/facebookresearch/fairseq2).
- <span id="page-9-6"></span>Vadim Borisov, Kathrin Seßler, Tobias Leemann, Martin Pawelczyk, and Gjergji Kasneci. Language models are realistic tabular data generators. *arXiv preprint arXiv:2210.06280*, 2022.
- <span id="page-9-9"></span>Lichang Chen, Jiuhai Chen, Tom Goldstein, Heng Huang, and Tianyi Zhou. Instructzero: Efficient instruction optimization for black-box large language models. *arXiv preprint arXiv:2306.03082*, 2023.
- <span id="page-9-8"></span>Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al. AlpaGasus: Training a better alpaca with fewer data. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https:](https://openreview.net/forum?id=FdVXgSJhvz) [//openreview.net/forum?id=FdVXgSJhvz](https://openreview.net/forum?id=FdVXgSJhvz).
- <span id="page-9-7"></span>Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. *arXiv preprint arXiv:2305.14233*, 2023.
- Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-9-1"></span>Saumya Gandhi, Ritu Gala, Vijay Viswanathan, Tongshuang Wu, and Graham Neubig. Better synthetic data by retrieving and transforming existing datasets. *arXiv preprint arXiv:2404.14361*, 2024.
- <span id="page-9-5"></span>Fabrizio Gilardi, Meysam Alizadeh, and Mael Kubli. Chatgpt outperforms crowd workers for text- ¨ annotation tasks. *Proceedings of the National Academy of Sciences*, 120(30):e2305016120, 2023.
- <span id="page-9-12"></span>Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-9-0"></span>Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. How close is chatgpt to human experts? comparison corpus, evaluation, and detection. *arXiv preprint arXiv:2301.07597*, 2023.
- <span id="page-9-11"></span>Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.
- Tom Hosking, Phil Blunsom, and Max Bartolo. Human feedback is not gold standard. *arXiv preprint arXiv:2309.16349*, 2023.
- <span id="page-9-3"></span>Alexey Kurakin, Natalia Ponomareva, Umar Syed, Liam MacDermed, and Andreas Terzis. Harnessing large-language models to generate private synthetic text. *arXiv preprint arXiv:2306.01684*, 2023.
- <span id="page-9-14"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.

<span id="page-10-10"></span>**540 541 542 543** Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?](https://openreview.net/forum?id=1oijHJBRsT) [id=1oijHJBRsT](https://openreview.net/forum?id=1oijHJBRsT).

<span id="page-10-4"></span>**544 545**

Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, et al. Best practices and lessons learned on synthetic data. *arXiv preprint arXiv:2404.07503*, 2024.

**546 547 548**

<span id="page-10-5"></span>**549**

Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning. *arXiv preprint arXiv:2312.15685*, 2023.

<span id="page-10-0"></span>**550**

**553**

<span id="page-10-9"></span>**556**

Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. On llms-driven synthetic data generation, curation, and evaluation: A survey. *arXiv preprint arXiv:2406.15126*, 2024.

**554 555**

Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, Chang Zhou, and Jingren Zhou. # instag: Instruction tagging for analyzing supervised fine-tuning of large language models. In *The Twelfth International Conference on Learning Representations*, 2023.

<span id="page-10-3"></span>**558 559**

Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. Generating training data with language models: Towards zero-shot language understanding. *Advances in Neural Information Processing Systems*, 35:462–477, 2022.

<span id="page-10-15"></span>

**560**

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. *arXiv preprint arXiv:2405.14734*, 2024.

<span id="page-10-14"></span>**564**

Xiaoyu Tan Minghao Yang, Chao Qu. Inf-orm-llama3.1-70b, 2024. URL [\[https://]([https://huggingface.co/infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B)) [huggingface.co/infly/INF-ORM-Llama3.1-70B\]\(https://huggingface.]([https://huggingface.co/infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B)) [co/infly/INF-ORM-Llama3.1-70B\)]([https://huggingface.co/infly/INF-ORM-Llama3.1-70B](https://huggingface.co/infly/INF-ORM-Llama3.1-70B)).

**567 568**

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. *arXiv preprint arXiv:2104.08773*, 2021.

<span id="page-10-7"></span>**569 570**

> Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-time ` scaling. *arXiv preprint arXiv:2501.19393*, 2025.

**573 574 575**

<span id="page-10-8"></span>**572**

Archiki Prasad, Weizhe Yuan, Richard Yuanzhe Pang, Jing Xu, Maryam Fazel-Zarandi, Mohit Bansal, Sainbayar Sukhbaatar, Jason Weston, and Jane Yu. Self-consistency preference optimization. *arXiv preprint arXiv:2411.04109*, 2024.

<span id="page-10-13"></span>**576**

<span id="page-10-6"></span>**579**

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36:53728–53741, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36, 2024.

<span id="page-10-12"></span>**584**

<span id="page-10-2"></span>**585**

David Saxton, Edward Grefenstette, Felix Hill, and Pushmeet Kohli. Analysing mathematical reasoning abilities of neural models. *arXiv preprint arXiv:1904.01557*, 2019.

**586 587 588**

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al. Beyond human data: Scaling self-training for problem-solving with language models. *arXiv preprint arXiv:2312.06585*, 2023.

<span id="page-10-11"></span>**590 591 592**

**593**

<span id="page-10-1"></span>**589**

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020.

<span id="page-11-13"></span><span id="page-11-6"></span><span id="page-11-2"></span>**604 605 606**

<span id="page-11-9"></span><span id="page-11-8"></span>**617**

<span id="page-11-15"></span><span id="page-11-11"></span><span id="page-11-7"></span>**619**

<span id="page-11-14"></span><span id="page-11-5"></span><span id="page-11-4"></span>**634**

<span id="page-11-10"></span><span id="page-11-3"></span><span id="page-11-0"></span>**636**

- <span id="page-11-12"></span><span id="page-11-1"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
  - Alexander Sasha Vezhnevets, John P Agapiou, Avia Aharon, Ron Ziv, Jayd Matyas, Edgar A Due´nez-Guzm ˜ an, William A Cunningham, Simon Osindero, Danny Karmon, and Joel Z Leibo. ´ Generative agent-based modeling with actions grounded in physical, social, or digital space using concordia. *arXiv preprint arXiv:2312.03664*, 2023.
  - Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. *arXiv preprint arXiv:2406.12845*, 2024.
  - Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*, 2022a.
  - Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. *arXiv preprint arXiv:2212.10560*, 2022b.
  - Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. *arXiv preprint arXiv:2204.07705*, 2022c.
  - Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. *arXiv preprint arXiv:2109.01652*, 2021.
  - Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. *arXiv preprint arXiv:2407.19594*, 2024.
  - Canwen Xu, Daya Guo, Nan Duan, and Julian McAuley. Baize: An open-source chat model with parameter-efficient tuning on self-chat data. *arXiv preprint arXiv:2304.01196*, 2023.
  - Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. *arXiv preprint arXiv:2502.03387*, 2025.
  - Ping Yu, Weizhe Yuan, Olga Golovneva, Tianhao Wu, Sainbayar Sukhbaatar, Jason Weston, and Jing Xu. Rip: Better models by survival of the fittest prompts. *arXiv preprint arXiv:2501.18578*, 2025.
  - Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models.
  - Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models. *arXiv preprint arXiv:2308.01825*, 2023.
  - Weihao Zeng, Can Xu, Yingxiu Zhao, Jian-Guang Lou, and Weizhu Chen. Automatic instruction evolving for large language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 6998–7018, 2024.
  - Chaoning Zhang, Chenshuang Zhang, Sheng Zheng, Yu Qiao, Chenghao Li, Mengchun Zhang, Sumit Kumar Dam, Chu Myaet Thwal, Ye Lin Tun, Le Luang Huy, et al. A complete survey on generative ai (aigc): Is chatgpt from gpt-4 to gpt-5 all you need? *arXiv preprint arXiv:2303.11717*, 2023.
  - Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36, 2024.

### <span id="page-12-0"></span>Figure 2: No CoT Prompt Generation Template for Non-verifiable instruction following task.

```
Below are sample tasks from user.
1. <begin>{INSTRUCTION 1}</end>
2. <begin>{INSTRUCTION 2}</end>
Come up with one new task, wrapped with <begin>and </end>
```

### <span id="page-12-1"></span>Figure 3: Short CoT Prompt Generation Template for Non-verifiable instruction following task.

```
Below are sample tasks from user.
1. <begin>{INSTRUCTION 1}</end>
2. <begin>{INSTRUCTION 2}</end>
Come up with one new task, wrapped with <begin>and </end>. Please provide Chain-of-
Thoughs first and then provide the new generated task
```

## 7 APPENDIX