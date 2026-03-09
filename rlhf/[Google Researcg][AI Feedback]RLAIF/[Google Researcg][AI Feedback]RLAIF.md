# RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback

# Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash

Google Research

{harrisonlee, samratph, hassan}@google.com

#### **Abstract**

Reinforcement learning from human feedback (RLHF) has proven effective in aligning large language models (LLMs) with human preferences. However, gathering high-quality human preference labels can be a time-consuming and expensive endeavor. RL from AI Feedback (RLAIF), introduced by Bai et al., offers a promising alternative that leverages a powerful off-the-shelf LLM to generate preferences in lieu of human annotators. Across the tasks of summarization, helpful dialogue generation, and harmless dialogue generation, RLAIF achieves comparable or superior performance to RLHF, as rated by human evaluators. Furthermore, RLAIF demonstrates the ability to outperform a supervised fine-tuned baseline even when the LLM preference labeler is the same size as the policy. In another experiment, directly prompting the LLM for reward scores achieves superior performance to the canonical RLAIF setup, where LLM preference labels are first distilled into a reward model. Finally, we conduct extensive studies on techniques for generating aligned AI preferences. Our results suggest that RLAIF can achieve human-level performance, offering a potential solution to the scalability limitations of RLHF.

#### 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) is an effective technique for aligning language models to human preferences (Stiennon et al., 2020; Ouyang et al., 2022). It is cited as one of the key drivers of success in modern conversational language models, such as ChatGPT (Liu et al., 2023) and Bard (Manyika, 2023). Training language models with reinforcement learning (RL) enables optimization on complex, sequence-level objectives that are not easily differentiable and therefore ill-suited for traditional supervised fine-tuning (SFT).

One obstacle for employing RLHF at scale is its dependence on high-quality human preference la-

#### **RLAIF** and **RLHF** Win Rates

<span id="page-0-1"></span>![](_page_0_Figure_11.jpeg)

# Harmless Rate by Policy

![](_page_0_Figure_13.jpeg)

Figure 1: Human evaluators strongly prefer RLAIF and RLHF over the SFT baseline for summarization and helpful dialogue generation. Their difference in win rates vs. SFT is not statistically significant. Furthermore, when compared head-to-head, RLAIF is equally preferred to RLHF. For harmless dialogue generation, RLAIF outperforms RLHF.

bels. This raises the question of whether artificially generated labels can be a viable substitute. Generating labels with large language models (LLMs) is one promising approach, as LLMs have shown a high degree of alignment with human judgment (Gilardi et al., 2023; Ding et al., 2023). Bai et al. (2022b) was the first effort to explore Reinforcement Learning from AI Feedback (RLAIF)<sup>1</sup>, where

<span id="page-0-0"></span><sup>&</sup>lt;sup>1</sup>This is distinct from "Constitutional AI", which improves upon a supervised learning model through iteratively asking an LLM to generate better responses according to a a set of

RL was conducted using a reward model trained on LLM preferences. [Bai et al.](#page-9-0) [\(2022b\)](#page-9-0) showed that utilizing a hybrid of human and AI preferences, in conjunction with their "Constitutional AI" selfrevision technique, outperforms supervised finetuning for training a conversational assistant. However, it did not directly compare the efficacy of human vs. AI feedback, leaving the question of whether RLAIF can be a suitable alternative to RLHF unanswered.

In this work, we study the impact of RLAIF and RLHF (see Figure [2\)](#page-2-0) on three text generation tasks: summarization, helpful dialogue generation, and harmless dialogue generation. Our experiments show that RLAIF and RLHF are preferred by humans over the SFT baseline 71% and 73% of the time for summarization and 63% and 64% of the time for helpful dialogue generation, respectively, where the differences between RLAIF and RLHF win rates are not statistically significant. We also conduct a head-to-head comparison of RLAIF against RLHF and find that both policies are equally preferred[2](#page-1-0) . For harmless dialogue generation, human evaluators rated the harmlessness of each response independently. RLAIF scored a higher harmless rate than RLHF, and both outperformed the SFT baseline (88%, 76%, and 64%, respectively). These results suggest that RLAIF is a viable alternative to RLHF that does not depend on human annotation, while offering appealing scaling properties.

Additionally, we investigate two related questions. First, we explore whether RLAIF can improve upon a SFT policy when the LLM labeler has the same number of parameters as policy. Even in this scenario, RLAIF significantly improves over the SFT baseline. Second, we conduct an experiment where the off-the-shelf LLM is directly prompted for reward scores during RL, bypassing the step of distilling LLM preference labels into a reward model. This method achieves an even higher win rate over SFT than the canonical distillation method.

Finally, we study techniques to maximize the alignment of AI-generated preferences to human preferences. We find that soliciting chain-ofthought reasoning [\(Wei et al.,](#page-11-2) [2022\)](#page-11-2) consistently improves alignment, while using a detailed pream-

ble and few-shot prompting [\(Brown et al.,](#page-9-2) [2020\)](#page-9-2) are only beneficial for certain tasks. We also conduct scaling experiments to examine the trade-off between the size of the LLM labeler and alignment with human preferences.

The main contributions of this work are as follows:

- 1. We demonstrate that RLAIF achieves comparable or superior performance to RLHF on the tasks of summarization, helpful dialogue generation, and harmless dialogue generation.
- 2. We show that RLAIF can improve upon a SFT policy even when the LLM labeler is the same size as the policy.
- 3. We find that directly prompting the LLM for reward scores during RL can outperform the canonical setup where a reward model is trained on LLM preferences.
- 4. We compare various techniques for generating AI labels and identify optimal settings for RLAIF practitioners.

#### 2 Methodology

This section describes the techniques used to generate preferences with an LLM, how RL is conducted, and evaluation metrics. Preliminaries on RLHF are provided in Appendix [A.](#page-13-0)

#### <span id="page-1-1"></span>2.1 Preference Labeling with LLMs

We annotate preferences between pairs of candidates with an "off-the-shelf" LLM - a model pretrained or instruction-tuned [\(Wei et al.,](#page-11-3) [2021\)](#page-11-3) for general usage but not fine-tuned for a specific downstream task. Given a piece of text and two candidate responses, the LLM is asked to rate which response is preferred. The prompt is structured as follows (examples in Tables [15](#page-20-0) and [21\)](#page-25-0):

- 1. *Preamble* Introduction and instructions describing the task at hand
- 2. *Few-shot exemplars (optional)* An example input context, a pair of responses, a chain-ofthought rationale (optional), and a preference label
- 3. *Sample to annotate* An input context and a pair of responses to be labeled
- 4. *Ending* Ending text to prompt the LLM (e.g. "*Preferred Response=*")

After the prompt is given to the LLM, we extract the log-probabilities of generating the tokens

written value statements. Both were introduced in [Bai et al.](#page-9-0) [\(2022b\)](#page-9-0) and are sometimes conflated.

<span id="page-1-0"></span><sup>2</sup>The win rate for one policy vs. the other is not statistically significantly different from 50%

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: A diagram depicting RLAIF (top) vs. RLHF (bottom)

"1" and "2" and compute the softmax to obtain a preference distribution.

There are numerous alternatives to obtain preference labels from LLMs, such as extracting the preference from a free-form generated response (e.g. *"The first response is better"*), or representing the preference distribution as a one-hot encoding. However, we choose our method because it is straightforward to implement and conveys more information than a one-hot encoding through its distributed representation of preferences.

We experiment with two styles of preambles: *"Base"*, which essentially asks "which response is better?", and *"Detailed"*, which resembles detailed rating instructions that would be given to human preference annotators (see Table [16](#page-21-0) for preambles for the summarization task). We also experiment with in-context learning [\(Brown et al.,](#page-9-2) [2020\)](#page-9-2), where high-quality exemplars were hand-selected to cover a range of topics.

#### <span id="page-2-1"></span>2.1.1 Addressing Position Bias

The order in which candidates are shown to an LLM can bias which candidate it prefers [\(Pezeshkpour and Hruschka,](#page-11-4) [2023;](#page-11-4) [Wang et al.,](#page-11-5) [2023\)](#page-11-5). We find evidence of position bias, which is more pronounced with smaller sizes of LLM labelers (see Appendix [B\)](#page-13-1).

To mitigate position bias in preference labeling, we make two inferences for every pair of candidates, where the order in which candidates are presented to the LLM is reversed for the second inference. The results from both inferences are then

averaged to obtain the final preference distribution.

#### 2.1.2 Chain-of-thought Reasoning

We experiment with eliciting chain-of-thought (CoT) reasoning [\(Wei et al.,](#page-11-2) [2022\)](#page-11-2) from our AI labelers through a two-step inference procedure. First, we replace the *Ending* of the standard prompt (e.g. "*Preferred Summary=*") with a sentence asking for thoughts and explanation (e.g. "*Consider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better. Rationale:*") and then decode a response from the LLM. Then, we concatenate the original prompt, the response, and the standard *Ending* string together, and follow the scoring procedure in Section [2.1](#page-1-1) to obtain a preference distribution. See Figure [3](#page-3-0) for an illustration.

In zero-shot prompts, the LLM is not given an example of what reasoning should look like. In few-shot prompts, we provide examples of CoT reasoning for the model to follow. See Tables [17](#page-22-0) and [18](#page-23-0) for examples.

# 2.2 Reinforcement Learning from AI Feedback

#### 2.2.1 Distilled RLAIF

We describe our adaptation of the canonical RLAIF setup below, which we also refer to as "distilled RLAIF". Unless otherwise mentioned, RLAIF is carried out using this method.

After labeling preferences with an LLM, a reward model (RM) is trained on these labels. Since our approach produces soft labels (e.g. [0.6, 0.4]),

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3: An illustration of the process of obtaining AI-generated labels for summarization preferences. The LLM is first prompted to explain its thoughts on the quality of the two candidates (blue). The LLM's response is then appended to the original prompt (orange) and fed to the LLM a second time to generate a preference distribution over "1" vs. "2" based on their log-probabilities (green).

we apply a cross-entropy loss to the softmax of the reward scores generated by the RM. The softmax converts the RM scores into a probability distribution. We note that training a RM on a dataset of AI labels can be viewed as a form of model distillation.

Finally, we conduct reinforcement learning to train the RLAIF policy model, using the RM to assign rewards to model responses.

#### 2.2.2 Direct RLAIF

An alternative approach is to directly use LLM feedback as the reward signal in RL. This enables bypassing the intermediate stage of training a RM that approximates the preferences of the LLM.

The LLM is prompted to rate the quality of a generation between 1 and 10. Similar to the format mentioned in Section 2.1, the prompt contains high-level details on the structure of the input and the dimensions along which to rate a generation (e.g. factuality, coherence). Then, the likelihood of each score token between 1 and 10 is computed, the likelihoods are normalized to a probability distribution, a weighted score is calculated as  $s(x|c) = \sum_{i=1}^{10} iP(i|x,c)$ , and then the score is again normalized to the range [-1,1]. Additional details on the prompting technique can be found in the Appendix D.

Finally, RL is conduct RL in a similar manner to "distilled RLAIF", where the direct score is used as reward instead of the score from a RM. This approach is more computationally expensive than

the canonical setup when the AI labeler is larger than the RM.

#### 2.3 Evaluation

We evaluate our results with three metrics - AI Labeler Alignment, Win Rate, and Harmless Rate.

AI Labeler Alignment measures the accuracy of AI-labeled preferences with respect to human preferences. For a single example, a soft AI-labeled preference is first converted to a binary representation (e.g.  $[0.6, 0.4] \rightarrow [1, 0]$ ). Then, a 1 is assigned if the label agrees with the human preference and 0 otherwise. The alignment accuracy  $z_{acc}$  can be expressed as follows:

$$z_{acc} = \frac{1}{D} \sum_{i=1}^{D} \mathbb{1}[\arg\max_{j} P_{i,j}^{AI} = p_{i}^{H}],$$

where D is the size of the preference dataset,  $P^{AI} \in \mathbb{R}^{D \times 2}$  is the matrix of soft AI preferences, and  $p^{human} \in \mathbb{R}^D$  is the corresponding vector of human preferences, containing elements 0 or 1 to denote whether the first or second response is preferred, respectively.

Win Rate evaluates the end-to-end quality of two policies by measuring how often one policy is preferred by human annotators over another. Given an input and two generations, human annotators select which generation they prefer. The percentage of instances where policy A is preferred over policy B is referred to as the "win rate of A vs. B". A 50% win rate indicates that A and B are equally preferred.

*Harmless Rate* measures the percentage of responses that are considered harmless by human evaluators. We evaluate the harmless dialogue generation task with this metric instead of *Win Rate*, because we find that many responses are equally safe, making it difficult to assign relative rankings.

# 3 Experimental Details

#### 3.1 Datasets

We use the following datasets for our experiments:

- Reddit TL;DR [\(Stiennon et al.,](#page-11-0) [2020\)](#page-11-0) posts from Reddit[3](#page-4-0) accompanied by summaries of the posts.
- OpenAI's Human Preferences [\(Stiennon et al.,](#page-11-0) [2020\)](#page-11-0) - a dataset created from a subset of Reddit TL;DR. Each example comprises a post, two candidate summaries, and a rating from a human annotator indicating which summary is preferred.
- Anthropic Helpful and Harmless Human Preferences [\(Bai et al.,](#page-9-3) [2022a\)](#page-9-3) - conversations between a human and an AI assistant, where each conversation has two possible AI assistant responses - one preferred and the other non-preferred, according to a human annotator. Preference is based on which response is more informative and honest for the helpful task, and which response is safer for the harmless task.

More dataset details can be found in Appendix [C.](#page-14-1) We also experimented with the Stanford Human Preferences dataset [\(Ethayarajh et al.,](#page-9-4) [2022\)](#page-9-4), but we found that both RLHF and RLAIF policies did not show meaningful improvements over the SFT baseline after correcting for length biases, using the procedure in Appendix [J.](#page-16-0)

#### 3.2 LLM Labeling

To enable fast experiment iteration when evaluating AI labeling techniques, we randomly downsampled the training split of each preference dataset. For summarization, an additional filter was applied to only include examples where human annotators preferred one summary over the other with high confidence[4](#page-4-1) . After downsampling and filtering,

there were roughly 3-4k examples for each task[5](#page-4-2) . AI labeler alignment metrics were calculated on these downsampled datasets.

PaLM 2 [\(Google et al.,](#page-10-3) [2023\)](#page-10-3) is used as the LLM for labeling preferences. The versions used are instruction-tuned but not previously trained with RL. Unless otherwise specified, AI labels were generated using PaLM 2 Large (L) with the bestperforming prompt in Section [4.4.](#page-6-0) For more details on LLM labeling, see Appendix [D.](#page-14-0)

## 3.3 Model Training

All SFT models are initialized from PaLM 2 Extra-Small (XS). For summarization, the SFT model is produced by fine-tuning PaLM 2 XS on the Reddit TL;DR dataset. For all other tasks, an instructiontuned variant of PaLM 2 is used in lieu of taskspecific fine-tuning.

RMs are also derived from PaLM 2 XS. RMs are fine-tuned on the entire training split of the corresponding preference dataset, where the label is the AI preference for AI feedback RMs and the original human preference label in the dataset for human feedback RMs. RM accuracies can be found in Appendix [G.](#page-15-0)

In the RL phase, the policy is trained with a modified version of REINFORCE [\(Williams,](#page-11-6) [1992\)](#page-11-6) adapted to the language modeling domain. While many recent works use Proximal Policy Optimization (PPO) [\(Schulman et al.,](#page-11-7) [2017\)](#page-11-7) - a related method that adds a few techniques to make training more conservative and stable (e.g. clipping the objective function), we use REINFORCE with a baseline given that it is simpler yet still effective for the problem at hand. Both policy and value models are initialized from the SFT model. For summarization, the policy is rolled out on the training split of the Reddit TL;DR dataset. In other words, the initial states for RL are the original posts from the dataset prior to summarization. For the helpful and harmless tasks, the initial states are drawn from the training splits of the preference datasets. For summarization, simple post-processing is applied to responses generated by RL-trained policies as described in Appendix [E.](#page-14-2)

For additional details on the RL formulation and model training, see Appendices [F](#page-14-3) and [G.](#page-15-0)

<span id="page-4-1"></span><span id="page-4-0"></span><sup>3</sup><www.reddit.com>

<sup>4</sup>This follows the evaluation procedure in [Stiennon et al.](#page-11-0) [\(2020\)](#page-11-0). Examples with confidence scores of 1, 2, 8, and 9 were considered to be "high-confidence"

<span id="page-4-2"></span><sup>5</sup>We sample 15%, 10%, and 10% of the training splits for summarization, helpful dialogue generation, and harmless dialogue generation, respectively.

#### 3.4 Human Evaluation

For experiments evaluated by win rates, evaluators were presented with an input context and multiple responses generated from different policies (e.g. RLAIF, RLHF, and SFT). They were then asked to rank responses in order of quality without ties, as seen in Figure [4.](#page-20-1) Input contexts were drawn from test splits of datasets, which were not used for training or any other evaluation[6](#page-5-0) . Rankings were used to calculate win rates with respect to pairs of policies. For harmless dialogue generation, evaluators were asked to independently rate each response as harmless or harmful.

For more details on human evaluation, see Appendix [I.](#page-16-1)

# 4 Results

## <span id="page-5-3"></span>4.1 RLAIF vs. RLHF

RLAIF achieves performance gains on par with or better than RLHF on all three tasks (see Figure [1](#page-0-1) and Table [1\)](#page-6-1). RLAIF and RLHF are preferred by human evaluators over the baseline SFT policy 71% and 73% of the time for summarization[7](#page-5-1) and 63% and 64% for helpful dialogue generation, respectively. The difference in win rates between RLAIF vs. SFT and RLHF vs. SFT are not statistically significant. When directly comparing RLAIF against RLHF, they are equally preferred - i.e. the win rate is not statistically significantly different from 50%. For harmless dialogue generation, RLAIF achieves a harmless rate of 88%, outperforming both RLHF and SFT, which score 76% and 64%, respectively[8](#page-5-2) .

Figure [5](#page-27-0) contains an example of SFT, RLAIF, and RLHF summaries. To better understand how RLAIF compares to RLHF, we qualitatively compare responses generated by both policies for summarization in Section [5.](#page-7-0)

As observed in [Stiennon et al.](#page-11-0) [\(2020\)](#page-11-0), RLAIF and RLHF policies tend to generate longer responses than the SFT policy, which may be partially responsible for their higher win rates. We conduct post-hoc analysis to control for length and find that both RLAIF and RLHF policies still outperform the SFT policy, and by similar margins to one another. See Appendix [J](#page-16-0) for details.

One natural question that arises is whether there is value in combining human and AI feedback. We experimented with combining both types of feedback but did not see an improvement beyond using human feedback alone. However, we believe that there are several alternative training setups that could demonstrate value in combining both forms of feedback. See Appendix [K](#page-17-0) for details.

These results suggest that RLAIF is a viable alternative to RLHF that does not depend on human annotation. In addition to expediting labeling time and reducing dependence on annotation services, another key benefit of AI labeling is cost reduction. We estimate the cost of labeling with an LLM to be over 10x cheaper than human annotation. See Appendix [L](#page-17-1) for detailed calculations.

## <span id="page-5-5"></span>4.2 Towards Self-Improvement

In Section [4.1,](#page-5-3) the LLM used to label preferences (PaLM 2 L) is much larger than the policy being trained (PaLM 2 XS). Going one step further, one might wonder if RLAIF can yield improvements when the AI labeler is the same size as the policy. On the task of summarization, we conduct RLAIF where PaLM 2 XS is used as the AI labeler instead of PaLM 2 L. The rest of the setup mimics the experiment in Section [4.1.](#page-5-3) We refer to this setup as "same-size RLAIF".

Human annotators prefer same-size RLAIF 68% of the time over SFT (see Table [1\)](#page-6-1). For reference, RLAIF using an AI labeler larger than the policy is preferred 71% over SFT[9](#page-5-4) . This result demonstrates that RLAIF can yield improvements even when the AI labeler is the same size as the policy LLM.

We note that the AI labeler and initial policy are not the exact same model. The AI labeler is the instruction-tuned PaLM 2 XS, whereas the initial policy is PaLM 2 XS fine-tuned on Reddit TL;DR summarization. Additionally, the summaries rated by the AI labeler were generated by policies created by the original dataset curators. For these reasons, we do not consider this experiment a strict case of "self-improvement"[\(Huang et al.,](#page-10-4) [2022\)](#page-10-4). However, we believe that these results show great promise for this research direction.

<span id="page-5-0"></span><sup>6</sup> For summarization, we used the test split of Reddit TL;DR. For helpful and harmless dialogue generation, we used test splits from the preference datasets, detailed in Appendix [C.](#page-14-1)

<span id="page-5-1"></span><sup>7</sup>RLAIF and RLHF are also preferred over the human reference summaries in Reddit TL;DR 79% and 80% of the time, respectively.

<span id="page-5-2"></span><sup>8</sup>RLAIF achieves a statistically significant improvement over RLHF and SFT, according to a two-sample t-test.

<span id="page-5-4"></span><sup>9</sup>The difference between win rates between "same-size RLAIF vs. SFT" and "RLAIF vs. SFT" is not statistically significant. For a two-sample t-test, p-value = 0.07. At alpha = 0.05, this difference is not statistically significant.

<span id="page-6-1"></span>

| Win Rate                        |                 | Harmless Rate       |       |                   |
|---------------------------------|-----------------|---------------------|-------|-------------------|
| Comparison                      | Summa -rization | Helpful<br>dialogue | Model | Harmless dialogue |
| RLAIF vs SFT                    | 71%             | 63%                 | SFT   | 64%               |
| RLHF vs SFT                     | 73%             | 64%                 | RLHF  | 76%               |
| RLAIF vs RLHF                   | 50%             | 52%                 | RLAIF | 88%               |
| Same-size RLAIF vs SFT          | 68%             |                     |       |                   |
| Direct RLAIF vs SFT             | 74%             |                     |       |                   |
| Direct RLAIF vs Same-size RLAIF | 60%             |                     |       |                   |

Table 1: **Left side:** Win rates when comparing generations from two different models for the summarization and the helpful dialogue tasks, judged by human evaluators. **Right side:** Harmless rates across policies for the harmless dialogue task, judged by human evaluators.

#### <span id="page-6-5"></span>4.3 Direct RLAIF

In Sections 4.1 and 4.2, AI feedback was distilled into a RM. On the summarization task, we experiment with using an off-the-shelf LLM to *directly* provide rewards during RL, bypassing RM training entirely. Since using a large AI labeler in RL is computationally expensive, we use the smaller instruction-tuned PaLM 2 XS as the off-the-shelf LLM. We refer to this setup as "direct RLAIF".

Human annotators prefer responses from direct RLAIF 74% of the time over SFT responses (see Table 1). To understand the impact of directly utilizing LLM feedback in RL, we compare this result to the same-size RLAIF policy from Section 4.2, which solely differs in training a RM that provides rewards during RL. Direct RLAIF outperforms same-size RLAIF, which achieves a statistically significantly lower win rate of 68%. Furthermore, when shown responses side-by-side, raters prefer direct RLAIF over same-size RLAIF 60% of the time 10. One hypothesis for the improved quality is that bypassing the distillation from AI preferences into a RM enables information to flow directly from the off-the-shelf LLM to the policy.

#### <span id="page-6-0"></span>4.4 Prompting Techniques

We experiment with three types of prompting variations - preamble specificity, chain-of-thought reasoning, and in-context learning (see Table 2). We observe that eliciting chain-of-thought reasoning generally improves AI labeler alignment, while the impacts of preamble specificity and in-context learning vary across tasks. The best prompts outperform the base prompts ("Base 0-shot") by +1.9%, +1.3%, and +1.7% for summarization, helpfulness,

<span id="page-6-3"></span>

|                       | AI Lat  | eler Align | ment  |
|-----------------------|---------|------------|-------|
| Prompt                | Summary | H1         | H2    |
| Base 0-shot           | 76.1%   | 67.8%      | 69.4% |
| Base 1-shot           | 76.0%   | 67.1%      | 71.7% |
| Base 2-shot           | 75.7%   | 66.8%      | 72.1% |
| Base + CoT 0-shot     | 77.5%   | 69.1%      | 70.6% |
| Detailed 0-shot       | 77.4%   | 67.6%      | 70.1% |
| Detailed 1-shot       | 76.2%   | 67.6%      | 71.5% |
| Detailed 2-shot       | 76.3%   | 67.3%      | 71.6% |
| Detailed 8-shot       | 69.8%   | _          | _     |
| Detailed + CoT 0-shot | 78.0%   | 67.8%      | 70.1% |
| Detailed + CoT 1-shot | 77.4%   | 67.4%      | 69.9% |
| Detailed + CoT 2-shot | 76.8%   | 67.4%      | 69.2% |

Table 2: We observe that eliciting chain-of-thought reasoning tends to improve AI labeler alignment, while few-shot prompting and detailed preambles have mixed effects across tasks. H1 refers to helpfulness, H2 to harmlessness.

and harmlessness, respectively.

Detailed preambles consistently improve alignment for summarization, while yielding mixed results for helpful and harmless dialogue generation. We hypothesize that summarization benefits more from a specific preamble due to the high complexity of this task. On the other hand, rating helpfulness and harmlessness are more intuitive to grasp, and therefore may benefit less from detailed instructions.

Chain-of-thought reasoning improves alignment consistently for summarization. For helpful and harmless dialogue generation, CoT only improves alignment when paired with the "Base" preamble.

Surprisingly, we observe that few-shot in-context learning only improves alignment for harmless dialogue generation<sup>11</sup>. For summarization and help-

<span id="page-6-2"></span><sup>&</sup>lt;sup>10</sup>This is statistically significantly different from 50% according to a two-sample t-test.

<span id="page-6-4"></span><sup>&</sup>lt;sup>11</sup>We verified that all inputs used in these experiments fit

fulness, alignment monotonically decreases as the number of exemplars increases. It seems unlikely that this effect is a result of exemplar quality, as exemplars were carefully handpicked to be highquality and representative of each preference task. Furthermore, we conducted 10 trials for "Base 1 shot" on summarization, where a different exemplar was randomly selected for each trial. The maximum AI labeler alignment from all trials was 76.1%, which still did not surpass "Base 0-shot" in terms of AI labeler alignment. One hypothesis for why exemplars do not help is that the summarization and helpful dialogue generation tasks may already be sufficiently well-understood by the powerful AI labeler, rendering the exemplars unhelpful or distracting. It's interesting to note that in-context learning is still an important research area that is not fully understood [\(Min et al.,](#page-11-8) [2022;](#page-11-8) [Wang et al.,](#page-11-9) [2022a\)](#page-11-9).

For summarization, we compare against human inter-annotator agreement to get a sense of how well our LLM labeler performs in absolute terms. [Stiennon et al.](#page-11-0) [\(2020\)](#page-11-0) estimated that agreement rate for the OpenAI human preference dataset was 73- 77%, suggesting that the off-the-shelf LLM achieving 78% alignment performs well in absolute terms.

We also conduct experiments with selfconsistency [\(Wang et al.,](#page-11-10) [2022b\)](#page-11-10), where multiple chain-of-thought rationales are sampled with temperature T > 0. The preference distributions generated by the LLM are averaged together to arrive at the final preference label. We find that selfconsistency strictly degrades AI labeler alignment (see Appendix [M\)](#page-18-0).

We hypothesize that higher AI labeler alignment leads to improvements in RLAIF policies. To this end, we conduct an experiment on the end-to-end sensitivity to AI labeler alignment. Two RLAIF policies are trained that only differ in the alignment scores of AI labels. Results show that the policy trained with more aligned AI labels achieves a significantly higher win rate. However, this study only compares two policies, and rigorous experimentation is required to draw definitive conclusions. See Appendix [N](#page-18-1) for details.

#### 4.5 Size of LLM Labeler

Large model sizes are not widely accessible and can be slow and expensive to run. On the task of summarization, we experiment with labeling prefer-

within our AI labeler's context length.

ences with varying LLM sizes and observe a strong relationship between size and alignment (see Table [3\)](#page-7-1). Alignment decreases -4% when moving from PaLM 2 Large (L) to PaLM 2 Small (S), and decreases another -11% when moving down to PaLM 2 XS - a trend consistent with scaling behaviors observed in other work [\(Kaplan et al.,](#page-10-5) [2020\)](#page-10-5). Besides general model capability, another contributing factor to this trend may be that smaller LLMs are more susceptible to position bias (see Appendix [B\)](#page-13-1).

On the other end of this trend, these results also suggest that scaling up AI labeler size may produce even higher quality preference labels. Since the AI labeler is only used to generate preference examples once and is not called during RL, using an even larger AI labeler is not necessarily prohibitively expensive.

<span id="page-7-1"></span>

| Model Size | AI Labeler Alignment |  |  |
|------------|----------------------|--|--|
| PaLM 2 L   | 78.0%                |  |  |
| PaLM 2 S   | 73.8%                |  |  |
| PaLM 2 XS  | 62.7%                |  |  |

Table 3: AI labeler alignment increases as the size of the LLM labeler increases.

### <span id="page-7-0"></span>5 Qualitative Observations

To better understand how RLAIF compares to RLHF, we inspected responses generated by both policies for the summarization task. In many cases, the two policies produced similar summaries, which is reflected in their similar win rates. However, we identified two patterns where they sometimes diverged.

The first pattern we observed is that in some cases, RLAIF hallucinates when RLHF does not. The hallucinations in RLHF summaries sound plausible but are inconsistent with the original text. For instance, in Example #1 of Table [23,](#page-27-1) the RLHF summary states that the author is 20 years old, but this is neither mentioned nor implied by the source text. The second pattern we observed is that RLAIF sometimes produces less coherent or grammatical summaries than RLHF. For instance, in Example #1 of Table [24,](#page-28-0) the RLAIF summary generates run-on sentences.

More systematic analysis is required to identify if these patterns exist at scale, which we leave to future work.

# 6 Related Work

LLMs have shown impressive performance over a wide range of NLP tasks [\(Brown et al.,](#page-9-2) [2020;](#page-9-2) [Thoppilan et al.,](#page-11-11) [2022;](#page-11-11) [Chowdhery et al.,](#page-9-5) [2022;](#page-9-5) [Google et al.,](#page-10-3) [2023;](#page-10-3) [OpenAI,](#page-11-12) [2023a\)](#page-11-12). For several of these tasks, RL has emerged as an effective optimization technique. While initial applications of RL on tasks such as translation [\(Wu et al.,](#page-11-13) [2016,](#page-11-13) [2018\)](#page-11-14) and summarization [\(Gao et al.,](#page-10-6) [2019;](#page-10-6) [Wu](#page-11-15) [and Hu,](#page-11-15) [2018\)](#page-11-15) used automatic evaluation metrics as rewards, such simplified formulations of rewards did not fully align with human notions of quality.

Reinforcement learning from human feedback [\(Christiano et al.,](#page-9-6) [2017\)](#page-9-6) has been used as a technique to directly align LLMs with human preferences [\(Ziegler et al.,](#page-12-0) [2019\)](#page-12-0) through training a reward model on pairwise comparisons of natural language responses. It has been successfully applied for summarization [\(Stiennon et al.,](#page-11-0) [2020\)](#page-11-0), generalized instruction following [\(Ouyang](#page-11-1) [et al.,](#page-11-1) [2022;](#page-11-1) [Lai et al.,](#page-10-7) [2023\)](#page-10-7), dialogue [\(Gilardi](#page-10-2) [et al.,](#page-10-2) [2023;](#page-10-2) [Manyika,](#page-10-1) [2023;](#page-10-1) [Glaese et al.,](#page-10-8) [2022;](#page-10-8) [Bai et al.,](#page-9-3) [2022a\)](#page-9-3) and question answering [\(Nakano](#page-11-16) [et al.,](#page-11-16) [2021\)](#page-11-16).

LLMs have also been extensively used for data generation [\(Wang et al.,](#page-11-17) [2021b;](#page-11-17) [Meng et al.,](#page-11-18) [2023\)](#page-11-18), augmentation [\(Feng et al.,](#page-9-7) [2021\)](#page-9-7) and in selftraining setups [\(Wang et al.,](#page-11-10) [2022b;](#page-11-10) [Madaan et al.,](#page-10-9) [2023\)](#page-10-9). [Bai et al.](#page-9-0) [\(2022b\)](#page-9-0) introduced the idea of RLAIF, which used LLM labeled preferences in conjunction with human labeled preferences to jointly optimize for the two objectives of helpfulness and harmlessness. Recent works have also explored related techniques for generating rewards from LLMs [\(Roit et al.,](#page-11-19) [2023;](#page-11-19) [Kwon et al.,](#page-10-10) [2022;](#page-10-10) [Yang et al.,](#page-12-1) [2023\)](#page-12-1). These works demonstrate that LLMs can generate useful signals for RL finetuning, which inspired this work's investigation into whether LLMs can serve as a viable alternative to humans in collecting preference labels for RL.

### 7 Conclusion

In this work, we show that RLAIF achieves comparable improvements to RLHF on three text generation tasks. Our experiments show that RLAIF greatly improves upon a SFT baseline, and the margin of improvement is on par with or greater than that of RLHF. Furthermore, in head-to-head comparisons, RLAIF and RLHF are preferred at similar rates by humans. Additionally, we show that

RLAIF is effective even when the LLM labeler is the same size as the policy, and directly prompting the LLM labeler to provide rewards during RL can outperform the canonical RLAIF setup that distills preferences into a separate RM. Finally, we study the impact of AI labeling techniques on alignment to human preferences.

While this work highlights the potential of RLAIF, there remain many fascinating open questions, such as whether conducting RLAIF iteratively can achieve additional gains (i.e. use the most recent RLAIF policy to generate new response pairs, conduct RLAIF, and repeat), how RLAIF can be adapted to a model-based RL setting where both human and assistant are modeled by LLMs, and how AI feedback can be leveraged for more specific credit assignment. We leave these questions for future work.

#### Ethics

One ethical consideration concerns the utilization of AI-generated feedback as a source for model alignment. There exists a potential risk of transferring biases from the off-the-shelf LLM into the generated preferences. This in turn may result in RL-trained policies further amplifying biases, thereby inadvertently misaligning models and potentially causing harm. Extreme caution must be exercised, especially when deploying these models in high-stakes domains such as medicine, law, and employment, where they have the potential to significantly impact human lives in adverse ways. In such domains, we believe that human experts trained to carefully assign preferences according to strict policies should be considered the gold standard.

Another ethical consideration is that reducing the barriers to aligning LLMs also carries the risk of facilitating their misuse for malicious purposes. For instance, RLAIF could be employed to train models to generate convincing misinformation or produce hateful and abusive content. The best mitigation to this risk is to carefully govern the access and usage of powerful LLMs (e.g. limiting "white-box" access), to prevent bad actors from misusing them.

#### Reproducibility

To promote the reproducibility of this work, many of the details of this research are shared throughout the paper. Open-source datasets are elaborated upon in Appendix [C,](#page-14-1) LLM labeling details in Appendix [D,](#page-14-0) the RL formulation in Appendix [F,](#page-14-3)

model training details in Appendix [G,](#page-15-0) human evaluation details in [I,](#page-16-1) and the most critical prompts used in the Appendix (e.g. Tables [17,](#page-22-0) [21,](#page-25-0) and [22\)](#page-26-0). Please reach out to authors for any additional questions or requests.

PaLM 2 models are available through Google Cloud's Vertex API, and the experiments in this work may also be repeated with other publicly available LLMs.

## Acknowledgements

We would like to thank many people who have helped make this work complete. We thank Chen Zhu for optimizing our LLM inference setup, Le Hou for suggesting prompt improvements and experimenting with self-consistency, Léonard Hussenot for bringing the problem of position bias in LLMs to our attention, and Bradley Green, Ewa Dominowska, and Blaise Aguera y Arcas for supporting this research.

We thank everyone who thoroughly reviewed our work and provided valuable feedback: Hakim Sidahmed, Meiqi Guo, Michal Valko, Nevan Wichers, Sian Gooding, and Yuan Cao.

We thank Mo Azar, Daniel Guo, Andrea Michi, Nicolas Perez-Nieves, and Marco Selvi for their work in developing a RLAIF training setup that directly prompts an LLM to obtain reward scores.

Finally, we thank the individuals who designed and built the RL training infrastructure used in this paper: Léonard Hussenot, Johan Ferret, Robert Dadashi, Geoffrey Cideron, Alexis Jacq, Sabela Ramos, Piotr Stanczyk, Sertan Girgin, Danila Sinopalnikov, Amélie Héliou, Nikola Momchev, and Olivier Bachem.

#### References

- <span id="page-9-9"></span>Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 2016. Concrete problems in ai safety. *arXiv preprint arXiv:1606.06565*.
- <span id="page-9-3"></span>Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*.
- <span id="page-9-0"></span>Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep

- Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022b. [Constitutional ai: Harmless](http://arxiv.org/abs/2212.08073)[ness from ai feedback.](http://arxiv.org/abs/2212.08073)
- <span id="page-9-2"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901.
- <span id="page-9-5"></span>Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. *arXiv preprint arXiv:2204.02311*.
- <span id="page-9-6"></span>Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. *Advances in neural information processing systems*, 30.
- <span id="page-9-1"></span>Bosheng Ding, Chengwei Qin, Linlin Liu, Yew Ken Chia, Boyang Li, Shafiq Joty, and Lidong Bing. 2023. [Is GPT-3 a good data annotator?](https://doi.org/10.18653/v1/2023.acl-long.626) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 11173–11195, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-9-4"></span>Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. 2022. Understanding dataset difficulty with V-usable information. In *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 5988–6008. PMLR.
- <span id="page-9-8"></span>Tom Everitt and Marcus Hutter. 2016. Avoiding wireheading with value reinforcement learning. In *Artificial General Intelligence: 9th International Conference, AGI 2016, New York, NY, USA, July 16-19, 2016, Proceedings 9*, pages 12–22. Springer.
- <span id="page-9-10"></span>Angela Fan, Mike Lewis, and Yann Dauphin. 2018. [Hierarchical neural story generation.](https://doi.org/10.18653/v1/P18-1082) In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 889–898, Melbourne, Australia. Association for Computational Linguistics.
- <span id="page-9-7"></span>Steven Y. Feng, Varun Gangal, Jason Wei, Sarath Chandar, Soroush Vosoughi, Teruko Mitamura, and Eduard Hovy. 2021. [A survey of data augmentation](https://doi.org/10.18653/v1/2021.findings-acl.84) [approaches for NLP.](https://doi.org/10.18653/v1/2021.findings-acl.84) In *Findings of the Association*

- *for Computational Linguistics: ACL-IJCNLP 2021*, pages 968–988, Online. Association for Computational Linguistics.
- <span id="page-10-11"></span>Roy Fox, Ari Pakman, and Naftali Tishby. 2015. Taming the noise in reinforcement learning via soft updates. *arXiv preprint arXiv:1512.08562*.
- <span id="page-10-6"></span>Yang Gao, Christian M Meyer, Mohsen Mesgar, and Iryna Gurevych. 2019. Reward learning for efficient reinforcement learning in extractive document summarisation. *arXiv preprint arXiv:1907.12894*.
- <span id="page-10-12"></span>Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. 2019. A theory of regularized markov decision processes. In *International Conference on Machine Learning*, pages 2160–2169. PMLR.
- <span id="page-10-2"></span>Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd-workers for textannotation tasks. *arXiv preprint arXiv:2303.15056*.
- <span id="page-10-8"></span>Amelia Glaese, Nat McAleese, Maja Trebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. 2022. Improving alignment of dialogue agents via targeted human judgements. *arXiv preprint arXiv:2209.14375*.
- <span id="page-10-16"></span>Google. 2023. Ai platform data labeling service pricing. [https://cloud.google.com/](https://cloud.google.com/ai-platform/data-labeling/pricing#labeling_costs) [ai-platform/data-labeling/pricing#](https://cloud.google.com/ai-platform/data-labeling/pricing#labeling_costs) [labeling\\_costs](https://cloud.google.com/ai-platform/data-labeling/pricing#labeling_costs). Accessed: 2023-09-28.
- <span id="page-10-3"></span>Rohan Anil Google, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan

- Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. 2023. [Palm 2 technical report.](http://arxiv.org/abs/2305.10403)
- <span id="page-10-13"></span>Ronald A Howard. 1960. *Dynamic programming and markov processes.* John Wiley.
- <span id="page-10-4"></span>Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve. *arXiv preprint arXiv:2210.11610*.
- <span id="page-10-14"></span>Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, José Miguel Hernández-Lobato, Richard E Turner, and Douglas Eck. 2017. Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control. In *International Conference on Machine Learning*, pages 1645–1654. PMLR.
- <span id="page-10-5"></span>Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.
- <span id="page-10-15"></span>M. G. Kendall and B. Babington Smith. 1939. [The](https://doi.org/10.1214/aoms/1177732186) [Problem of](https://doi.org/10.1214/aoms/1177732186) m Rankings. *The Annals of Mathematical Statistics*, 10(3):275 – 287.
- <span id="page-10-10"></span>Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh. 2022. Reward design with language models. In *The Eleventh International Conference on Learning Representations*.
- <span id="page-10-7"></span>Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A Rossi, and Thien Huu Nguyen. 2023. Okapi: Instructiontuned large language models in multiple languages with reinforcement learning from human feedback. *arXiv preprint arXiv:2307.16039*.
- <span id="page-10-0"></span>Yiheng Liu, Tianle Han, Siyuan Ma, Jiayue Zhang, Yuanyuan Yang, Jiaming Tian, Hao He, Antong Li, Mengshen He, Zhengliang Liu, et al. 2023. Summary of chatgpt/gpt-4 research and perspective towards the future of large language models. *arXiv preprint arXiv:2304.01852*.
- <span id="page-10-9"></span>Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2023. Self-refine: Iterative refinement with self-feedback. *arXiv preprint arXiv:2303.17651*.
- <span id="page-10-1"></span>James Manyika. 2023. An overview of bard: an early experiment with generative ai. [https://ai.google/static/](https://ai.google/static/documents/google-about-bard.pdf) [documents/google-about-bard.pdf](https://ai.google/static/documents/google-about-bard.pdf). Accessed: 2023-08-23.

- <span id="page-11-18"></span>Yu Meng, Martin Michalski, Jiaxin Huang, Yu Zhang, Tarek Abdelzaher, and Jiawei Han. 2023. Tuning language models as training data generators for augmentation-enhanced few-shot learning. In *International Conference on Machine Learning*, pages 24457–24477. PMLR.
- <span id="page-11-8"></span>Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 11048–11064.
- <span id="page-11-16"></span>Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted questionanswering with human feedback. *arXiv preprint arXiv:2112.09332*.
- <span id="page-11-12"></span>OpenAI. 2023a. [Gpt-4 technical report.](http://arxiv.org/abs/2303.08774)
- <span id="page-11-23"></span>OpenAI. 2023b. Openai pricing. [https://openai.](https://openai.com/pricing) [com/pricing](https://openai.com/pricing). Accessed: 2023-09-28.
- <span id="page-11-1"></span>Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744.
- <span id="page-11-4"></span>Pouya Pezeshkpour and Estevam Hruschka. 2023. Large language models sensitivity to the order of options in multiple-choice questions. *arXiv preprint arXiv:2308.11483*.
- <span id="page-11-19"></span>Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Léonard Hussenot, Orgad Keller, et al. 2023. Factually consistent summarization via reinforcement learning with textual entailment feedback. *arXiv preprint arXiv:2306.00186*.
- <span id="page-11-7"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
- <span id="page-11-21"></span>Noam Shazeer and Mitchell Stern. 2018. [Adafactor:](http://arxiv.org/abs/1804.04235) [Adaptive learning rates with sublinear memory cost.](http://arxiv.org/abs/1804.04235) *CoRR*, abs/1804.04235.
- <span id="page-11-0"></span>Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008– 3021.
- <span id="page-11-20"></span>Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. 1999. Policy gradient methods for reinforcement learning with function approximation. *Advances in neural information processing systems*, 12.

- <span id="page-11-11"></span>Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. *arXiv preprint arXiv:2201.08239*.
- <span id="page-11-9"></span>Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. 2022a. Towards understanding chain-of-thought prompting: An empirical study of what matters. *arXiv preprint arXiv:2212.10001*.
- <span id="page-11-5"></span>Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. 2023. Large language models are not fair evaluators. *arXiv preprint arXiv:2305.17926*.
- <span id="page-11-22"></span>Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, and Michael Zeng. 2021a. Want to reduce labeling cost? gpt-3 can help. In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 4195–4205.
- <span id="page-11-10"></span>Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022b. Self-consistency improves chain of thought reasoning in language models. In *The Eleventh International Conference on Learning Representations*.
- <span id="page-11-17"></span>Zirui Wang, Adams Wei Yu, Orhan Firat, and Yuan Cao. 2021b. Towards zero-label language learning. *arXiv preprint arXiv:2109.09193*.
- <span id="page-11-3"></span>Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. In *International Conference on Learning Representations*.
- <span id="page-11-2"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35:24824–24837.
- <span id="page-11-6"></span>Ronald J Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. *Machine learning*, 8:229–256.
- <span id="page-11-14"></span>Lijun Wu, Fei Tian, Tao Qin, Jianhuang Lai, and Tie-Yan Liu. 2018. A study of reinforcement learning for neural machine translation. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3612–3621.
- <span id="page-11-13"></span>Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*.
- <span id="page-11-15"></span>Yuxiang Wu and Baotian Hu. 2018. Learning to extract coherent summary via deep reinforcement learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, page 5602.

<span id="page-12-1"></span>Kevin Yang, Dan Klein, Asli Celikyilmaz, Nanyun Peng, and Yuandong Tian. 2023. [Rlcd: Reinforcement](http://arxiv.org/abs/2307.12950) [learning from contrast distillation for language model](http://arxiv.org/abs/2307.12950) [alignment.](http://arxiv.org/abs/2307.12950)

<span id="page-12-0"></span>Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593*.

#### <span id="page-13-0"></span>**A RLHF Preliminaries**

We review the RLHF pipeline introduced in Stiennon et al. (2020); Ouyang et al. (2022), which consists of 3 phases: supervised fine-tuning, reward model training, and reinforcement learning.

#### A.1 Supervised Fine-tuning

A pre-trained LLM is fine-tuned on a high quality labeled dataset for a downstream task (e.g. summarization) using token-level supervision to produce a supervised fine-tuned (SFT) model  $\pi^{SFT}$ .

#### A.2 Reward Modeling

Given an input x, we sample a pair of responses  $(y_1,y_2) \sim \pi$  from one or more models, where oftentimes  $\pi$  is the SFT model. The input and responses are sent to human annotators to rate which response is better according to some criteria. These annotations form a dataset of triplets  $\mathcal{D} = \{(x,y_w,y_l)\}$ , where  $y_w$  and  $y_l$  are the preferred and non-preferred responses, respectively. A reward model (RM)  $r_\phi$  is trained by minimizing the following loss:

$$\mathcal{L}_r(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma \big( r_{\phi}(x, y_w) - r_{\phi}(x, y_l) \big) \Big],$$

where  $\sigma$  is the sigmoid function.

#### <span id="page-13-3"></span>A.3 Reinforcement Learning

A policy  $\pi_{\rm A}^{RL}$  is initialized from the SFT model weights and then optimized with reinforcement learning to maximize the reward given by the RM, which serves as a proxy for human preferences. Optionally, a Kullback-Leibler (KL) divergence term  $D_{KL}$  is added to the objective to penalize  $\pi_{\theta}^{RL}$  for deviating from the original SFT policy  $\pi^{SFT}$ , controlled by the hyperparameter  $\beta$  (Fox et al., 2015: Geist et al., 2019). The KL loss helps prevent  $\pi_{\theta}^{RL}$  from drifting into a region where it generates language that is highly rewarded by the RM yet consists of low-quality or unnatural language - a phenomenon known as "reward hacking" (Everitt and Hutter, 2016; Amodei et al., 2016). The optimization objective is described by the equation below:

$$J(\theta) = \underset{y \sim \pi_{\theta}(\cdot|x)}{\mathbb{E}} \Big[ (1 - \beta) r_{\phi}(y|x) - \beta D_{KL} \Big( \pi_{\theta}^{RL}(y|x) || \pi^{SFT}(y|x) \Big) \Big],$$

where  $\beta$  is a hyperparameter between 0 and 1.

#### <span id="page-13-1"></span>**B** Position Bias in LLM Labelers

<span id="page-13-2"></span>

| Model Size | % Same Position Preferred |  |  |
|------------|---------------------------|--|--|
| PaLM 2 L   | 18%                       |  |  |
| PaLM 2 S   | 21%                       |  |  |
| PaLM 2 XS  | 56%                       |  |  |

Table 4: Position bias is more prevalent in smaller model sizes, measured by the percentage of examples where the LLM prefers the same position even after swapping the order of candidates ("% Same Position Preferred"). Analysis is conducted using the "Detailed + CoT 0-shot" prompt for the summarization task.

Our analysis on the summarization task suggests that the LLMs used for preference labeling are biased by the order in which candidates are shown. For each example in our AI labeling evaluation set, we query the LLM preferences for the pair of candidates, swap the order in which candidates are presented, and then query the LLM preferences again.

We consider an LLM to be *more biased* if it prefers the same position on both the original and reversed inferences. For example, let candidates A and B be in positions 1 and 2 for the first inference and in positions 2 and 1 for the second inference. If the LLM prefers the same position on both inferences, we consider the LLM to be position-biased. We measure position bias by computing "% Same Position Preferred" - the percentage of inference pairs where this occurs. A higher metric value indicates a more biased LLM.

We find that PaLM 2 L, S, and XS prefer the same position 18%, 21%, and 56% of the time, respectively, suggesting that position bias is inversely correlated with model size (see Table 4). One hypothesis is that larger models are more capable and therefore more faithfully judge preferences based on the content of the candidates rather than their positions, which are supposed to be immaterial.

We also observe that for PaLM 2 L, of the 18% of cases where it prefers the same position on both inferences, 94% of the time it prefers the first candidate shown. On the other hand, PaLM 2 S and XS show affinity for the second candidate shown when the same position is preferred on both inferences, preferring it 91% and 99% of the time, respectively. These biases are statistically significant under a two-sided binomial test at  $\alpha=0.05$ .

## <span id="page-14-1"></span>C Dataset Details

For summarization, we use the filtered Reddit TL;DR dataset [\(Stiennon et al.,](#page-11-0) [2020\)](#page-11-0), containing posts from Reddit[12](#page-14-4) that have been filtered to ensure high quality. The dataset contains 123k posts, where ∼5% is held out as a validation set.

Additionally, we use OpenAI's human preference dataset created from the filtered Reddit TL;DR dataset. For a given post, two candidate summaries were generated - often from different policies, and human labelers were asked to rate which summary they preferred. The total dataset comprises 92k pairwise comparisons.

For helpful and harmless dialogue generation, we use Anthropic's Helpful and Harmless preference datasets[13](#page-14-5) [\(Bai et al.,](#page-9-3) [2022a\)](#page-9-3). Each example consists of a conversation history between a human and an AI assistant accompanied by a preferred and non-preferred response from the AI assistant. Preference is based on which response is more helpful and honest for the helpful task, and which response is safer and less harmful for the harmless task. Each dataset comprises over 40k training examples and 2k test examples. We further split the test sets into validation and test sets by randomly assigning twothirds of examples to validation and one-third to test.

#### <span id="page-14-0"></span>D LLM Labeling Details

For LLM labeling, we set a maximum input context length of 4096 tokens. For chain-of-thought generation, we set a maximum decoding length of 512 tokens and sample with temperature T = 0.0 (i.e. greedy decoding). For self-consistency experiments in Appendix [M,](#page-18-0) we use temperatures varying from T = 0.3 to T = 1.0 with top-K sampling [\(Fan et al.,](#page-9-10) [2018\)](#page-9-10), where K = 40.

In Section [4.3,](#page-6-5) we use the AI labeler to directly compute a score that we leverage as the reward for RL. We use the following prompt: *"You are an expert summary rater. Given a TEXT (completed with a SUBREDDIT and a TITLE) and a SUMMARY, your role is to provide a SCORE from 1 to 10 that rates the quality of the SUMMARY given the TEXT, with 1 being awful and 10 being a perfect SUM-MARY."*, followed by the input Reddit post, then

the summary to score preceded by *"SUMMARY: "*, and a final *"SCORE: "*.

PaLM 2 models are publicly available through Google Cloud's Vertex AI[14](#page-14-6), though we cannot guarantee full reproducibility as the models accessible through Google Cloud are subject to change.

### <span id="page-14-2"></span>E Post-RL Response Formatting

For summarization, we observed that summaries generated by RLHF and RLAIF policies often included superfluous symbols like periods or spaces at the end of the response - possibly due to "reward hacking". Given that these extra tokens do not have any meaningful content, we programmatically removed certain symbols at the end of summaries. This ensured that human evaluators could focus on the content and not be distracted by the formatting of the response.

# <span id="page-14-3"></span>F REINFORCE for Language Models

Consider a deterministic, finite-horizon MDP M = (X , A, R, P, γ) [\(Howard,](#page-10-13) [1960\)](#page-10-13). At each step t, given the current state X<sup>t</sup> ∈ X and the next action A<sup>t</sup> ∈ A, the model receives a reward R<sup>t</sup> = R(X<sup>t</sup> , At) and transitions to the next state Xt+1 = P(X<sup>t</sup> , At).

In the context of language models, X<sup>t</sup> is the concatenation of the input text and all text generated by the policy until time t. Action A<sup>t</sup> is the token from the considered vocabulary decoded at time t by the stochastic policy πθ(·|Xt), where θ represents the policy parameters. Finally, the reward Rt is given by the RM. The RM is only evaluated when the language model response has been fully generated; all rewards prior to the final token are set to 0, while the reward corresponding to the final token is set to R<sup>T</sup> .

The cumulative sum of rewards received when following the policy π<sup>θ</sup> from time-step t is called the return. Generally, it is defined as P Z<sup>t</sup> = T s=t γ <sup>s</sup>−tRs. However, since only the terminal reward is non-zero and we set γ = 1, the return can be simplified to Z<sup>t</sup> = R<sup>T</sup> .

Given a trajectory (X<sup>t</sup> , A<sup>t</sup> , Rt) T <sup>t</sup>=0 generated under πθ, the policy gradient loss from REINFORCE is then defined as follows:

$$\mathcal{L}_{PG}(\theta) = -\sum_{t} \log \pi_{\theta}(A_{t}|X_{t}) \overline{\left(Z_{t} - V_{\psi}^{\pi}(X_{t})\right)},$$

<span id="page-14-5"></span><span id="page-14-4"></span><sup>12</sup><www.reddit.com>

<sup>13</sup>We use the helpful-base and harmless-base datasets from [https://huggingface.co/](https://huggingface.co/datasets/Anthropic/hh-rlhf) [datasets/Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf).

<span id="page-14-6"></span><sup>14</sup>[https://cloud.google.com/vertex-ai/](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models) [docs/generative-ai/learn/models](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models)

where the bar notation denotes that no gradient is passed through the advantage term during backpropagation.

The baseline value function  $V_{\psi}^{\pi}(x)$  estimates the return-to-go  $Z_t$  when following the policy  $\pi_{\theta}$  and is parameterized by  $\psi$  (Williams, 1992; Sutton et al., 1999). It is trained with the following loss:

$$\mathcal{L}_V(\psi) = \sum_t (Z_t - V_{\psi}^{\pi}(X_t))^2.$$

In practice, we optimize the regularized objective in Sec. A.3. We incorporate the KL divergence in the policy gradient loss described above, as commonly seen in other work (Jaques et al., 2017).

## <span id="page-15-0"></span>**G** Model Training Details

SFT models for the summarization task are trained on the Reddit TL;DR dataset, with a batch size of 128 for a single epoch. We use the Adafactor (Shazeer and Stern, 2018) optimizer with a learning rate of  $10^{-5}$ , and the maximum input and output lengths are 1024 and 128 tokens, respectively. For helpful and harmless dialogue generation tasks, an instruction-tuned version of PaLM 2 XS serves as the SFT model.

RMs for all tasks are trained until the training loss and accuracy curves plateau, which happens in 2-3 epochs. We use the Adafactor optimizer with a learning rate of  $10^{-5}$ . Batch size is 128 for summarization RMs and 32 for RMs of other tasks. We train all our RMs with maximum input length of 1152 tokens to account for 1024 context tokens and 128 response tokens. We report the accuracies of the RMs in Appendix H.

For summarization, the AI feedback RM is initialized from the SFT model (i.e. PaLM 2 XS finetuned on Reddit TL;DR), and the human feedback RM is initialized from PaLM 2 XS. We experimented with initializing the human feedback RM from the SFT model but found that it resulted in lower accuracy on the held out set of human preferences (see Table 6). For helpful and harmless dialogue generation tasks, we initialize both the human and AI feedback RMs from the instructiontuned version of PaLM 2 XS.

For reinforcement learning, we use the SFT model for each task as the initial policy. We sample from our language model policies for all tasks with a temperature of T=0.9 to encourage exploration. We train with a batch size of 128 and learning rate of  $10^{-5}$  for 8 epochs. We set  $\beta=0.05$  for the KL divergence loss.

To select the final checkpoint for each RL policy, we first selected 4 candidate checkpoints from RL training that scored high rewards on validation prompts. We then prompted an off-the-shelf LLM to judge the win rate of the RL checkpoint's summaries vs. the SFT policy's summaries. We also conducted manual inspection of a dozen examples. We picked the checkpoint with the best combination of win rate and quality as judged by manual inspection as our final RL policy.

## <span id="page-15-1"></span>**H** Reward Model Accuracy

<span id="page-15-3"></span>

| Task              | Human    | AI       |
|-------------------|----------|----------|
| Task              | Feedback | Feedback |
| Summarization     | 79.3%    | 74.2%    |
| Helpful Dialogue  | 76.0%    | 67.8%    |
| Harmless Dialogue | 72.1%    | 69.7%    |

Table 5: Pairwise accuracies of human feedback and AI feedback reward models across all tasks. Metrics are calculated on a held out set of human preference data for each task.

<span id="page-15-2"></span>

| Initialization | Human    | AI       |  |
|----------------|----------|----------|--|
|                | Feedback | Feedback |  |
| PaLM 2 XS      | 79.3%    | 73.0%    |  |
| SFT            | 78.7%    | 74.2%    |  |

Table 6: Results of initializing the summarization RMs on PaLM 2 XS vs. the SFT model.

<span id="page-15-4"></span>

| RM Variant                       | AI<br>Feedback |
|----------------------------------|----------------|
| Trained on "Base 0-shot" labels  | 77.9%          |
| Trained on labels from PaLM 2 XS | 66.4%          |

Table 7: Accuracy values for variants of RMs trained on AI labels for the task of summarization.

Pairwise Accuracy for RMs measures how accurate a trained reward model is with respect to a held-out set of human preferences. Given an input context and pair of candidate responses, the value is 1 if the RM scores the preferred candidate higher than the non-preferred candidate, according to the human label. Otherwise the value is 0. This quantity is averaged over multiple examples to obtain the pairwise accuracy of the RM.

We report RM accuracy on a held out set of human preferences for all tasks in Table 5. For summarization, we also report RM accuracy when initializing on different checkpoints in Table [6.](#page-15-2) In Table [7,](#page-15-4) we report accuracy for RM variants used in the end-to-end sensitivity experiment in Appendix [N](#page-18-1) and the same-size RLAIF experiment in Section [4.2.](#page-5-5)

We observe that RMs trained on human feedback outperform those trained on AI feedback, both of which are measured against a held out set of human preferences. This pattern seems natural, given that the human preferences are trained on data drawn from the same distribution as the validation dataset. However, it is interesting to note that despite the gap in accuracy between AI and human preference RMs, RLAIF achieves comparable results to RLHF on two tasks and surpasses RLHF on one task. Additionally, we note that the summarization RMs trained on "Base 0-shot" and "Detailed + CoT 0-shot" (i.e. the default prompting technique) achieve accuracies of 77.9% and 74.2%, respectively, which is the inverse order of their final performance after RL (see Appendix [N\)](#page-18-1). These gaps in RM accuracy suggest that RM accuracy, while correlated with RM usefulness, may not be a perfect reflection of RM effectiveness in RLHF and RLAIF. Ultimately, we believe that the usefulness of RMs is assessed through conducting RL and evaluating the final policies through human evaluation.

## <span id="page-16-1"></span>I Human Evaluation Details

To conduct human evaluation, in total we created ∼2k unique rating instances. Each instance comprised a single context and three distinct model responses (e.g. responses from SFT, RLAIF, and RLHF policies), resulting in a total of ∼6k unique (context, response) pairs subjected to human evaluation. Additionally, each instance was assessed by three independent raters, resulting in ∼18k (context, response, rating) tuples.

We measure the inter-annotator agreement with Kendall's Coefficient of Concordance W [\(Kendall](#page-10-15) [and Smith,](#page-10-15) [1939\)](#page-10-15) - a non-parametric statistic for assessing the agreement among multiple raters ranking multiple items. The values of Kendall's W range from 0 to 1, where 0 indicates perfect disagreement and 1 indicates perfect agreement. We conducted multiple human evaluation sessions, and the W statistic ranged from 0.6-0.7, indicating a reasonable level of agreement.

# <span id="page-16-0"></span>J Controlling for Response Length

Response length often can influence human evaluators' perception of quality [\(Stiennon et al.,](#page-11-0) [2020\)](#page-11-0), and our various policies generate responses that differ in length. For example, in the summarization task, the summaries produced by RLAIF, RLHF, and SFT policies sent to human evaluation have an average character-length of 164, 161, and 132, respectively. For all experiments presented in this paper, we conduct post-hoc analysis to estimate the win rates after controlling for length.

We take an approach similar to [Stiennon et al.](#page-11-0) [\(2020\)](#page-11-0) and calculate the "length-adjusted win rate of policy A vs. policy B". Given policy A, we train a logistic regression model where the input is the ratio of the policy A's response length to policy B's summary length (in characters), and the target is a binary label indicating whether policy A's response was preferred over policy B's response. After fitting the model, we estimate a length-controlled win rate by asking the logistic regressor to predict the win rate given a length ratio of 1.0, which represents the scenario where both the responses are of equal length.

After controlling for length for the summarization task, our length-adjusted win rates for RLAIF and RLHF vs. SFT are 59% and 61%, respectively (see Table [8\)](#page-17-2). Both RL policies continue to outperform the SFT policy by a similar margin, supporting our initial statement that RLAIF is comparable to RLHF.

We reach similar conclusions for the helpful dialogue generation task (Table [9\)](#page-17-3), same-size RLAIF and direct RLAIF experiments (Table [11\)](#page-17-4), the endto-end sensitivity to AI labeler alignment experiment (Table [12\)](#page-18-2), and combining human and AI feedback (Table [13\)](#page-18-3).

For the harmless dialogue generation task, the setup is slightly different. Since human evaluators rated each response independently as harmful or harmless, we compute the harmless rate instead of the win rate. We use the average generation length from the SFT policy as the reference point for all other policies (Table [10\)](#page-17-5).

We note that this post-hoc method of controlling for length is imperfect, as it assumes the logistic regression model accurately learns the relationship between summary length and human preference. A more principled approach would be to encourage all policies generate summaries of similar length through an auxiliary training loss.

<span id="page-17-2"></span>

|               | Length      | Length    |
|---------------|-------------|-----------|
| Models        | uncorrected | corrected |
| RLAIF vs SFT  | 71%         | 59%       |
| RLHF vs SFT   | 73%         | 61%       |
| RLAIF vs RLHF | 50%         | 47%       |

Table 8: Length-controlled win rate for the summarization task.

<span id="page-17-3"></span>

|               | Length      | Length    |
|---------------|-------------|-----------|
| Models        | uncorrected | corrected |
| RLAIF vs SFT  | 63%         | 61%       |
| RLHF vs SFT   | 64%         | 61%       |
| RLAIF vs RLHF | 52%         | 50%       |

Table 9: Length-controlled win rate for the helpful dialogue generation task.

# <span id="page-17-0"></span>K Combining Human and AI Feedback

We investigate the effectiveness of combining human feedback and AI feedback on the task of summarization. We refer to this approach as RLHF + RLAIF and compare it against RLHF.

First, given contexts randomly drawn from the Reddit TL;DR dataset, responses are generated by RLHF and SFT policies with temperature T = 1.0. The instruction-tuned PaLM 2 L is then called to generate AI preferences. Finally, a new RM is trained on both the entire OpenAI human preference dataset and an equivalent size AI preference dataset.

We observe that RLHF + RLAIF does not improve beyond RLHF alone. RLHF + RLAIF and RLHF achieve win rates of 71% and 74% over SFT, respectively. The difference in win-rate is not statistically significant. When compared head-to-head, raters prefer both policies equally.

While this experiment did not show positive results from combining RLAIF and RLHF, there are many alternative setups which could prove successful. One such setup could involve first conducting RLAIF, then collecting generations and human preferences using the RLAIF policy as the initialization point for RLHF. In this curriculum learning approach, RLAIF can be viewed as a "warm-up" policy, which is then refined with RLHF. Another possible setup could involve collecting much more AI feedback than human feedback, since it is much less expensive to collect (see Appendix [L\)](#page-17-1). We leave this exploration to future work.

<span id="page-17-5"></span>

|        | Length      | Length    |
|--------|-------------|-----------|
| Models | uncorrected | corrected |
| SFT    | 64%         | 64%       |
| RLHF   | 76%         | 78%       |
| RLAIF  | 88%         | 91%       |

Table 10: Length-controlled harmless rate for the harmless dialogue generation task. We used the average generation length from the SFT model as reference length to compute the length-controlled harmless rate for RLHF and RLAIF.

<span id="page-17-4"></span>

| Models          | Length      | Length    |
|-----------------|-------------|-----------|
|                 | uncorrected | corrected |
| Same-size RLAIF |             |           |
| vs SFT          | 68%         | 59%       |
| Direct RLAIF    |             |           |
| vs SFT          | 74%         | 65%       |
| Direct RLAIF vs |             |           |
| Same-size RLAIF | 60%         | 56%       |

Table 11: Length-controlled win rate for same-size RLAIF and direct RLAIF.

# <span id="page-17-1"></span>L Cost of LLM vs. Human Labeling

Using LLMs as data annotators can be much less costly than hiring human annotators [\(Wang et al.,](#page-11-22) [2021a\)](#page-11-22). We estimate AI preference labeling to be over 10x less costly than human preference labeling following the calculations below.

At the time of writing, GPT-4 charges \$0.03 USD and \$0.06 USD for every 1,000 tokens to encode and decode, respectively [\(OpenAI,](#page-11-23) [2023b\)](#page-11-23). For labeling TL;DR preferences with an LLM, our average token lengths were as follows:

- 1. *Input prompt length* 830 tokens (using the "Detailed + CoT 0-shot" prompt)
- 2. *Generated chain-of-thought rationale* 61 tokens

Additionally, to debias position, we repeat each labeling procedure after inverting the order in which a pair of responses are shown. Our estimated AI labeling cost per example is \$0.06 USD[15](#page-17-6) .

In comparison, Google Cloud's human annotation service charges approximately \$0.11 USD / 50 words for classification tasks at the time of writ-

<span id="page-17-6"></span><sup>15</sup>2 inferences \* (830 encoder tokens \* \$0.03 / 1,000 tokens + 61 decoder tokens \* \$0.06 / 1,000 tokens) = \$0.057 ∼ = \$0.06

<span id="page-18-2"></span>

| Models         | Length      | Length    |
|----------------|-------------|-----------|
|                | uncorrected | corrected |
| Base RLAIF     | 63%         | 59%       |
| vs SFT         |             |           |
| Detailed RLAIF |             |           |
| vs SFT         | 67%         | 63%       |
| Base RLAIF vs  |             |           |
| Detailed RLAIF | 41%         | 45%       |

Table 12: Length-controlled win rate for the experiment on end-to-end sensitivity to AI labeler alignment. Base RLAIF and Detailed RLAIF correspond to "Base 0-shot" RLAIF and "Detailed CoT 0-shot" RLAIF described in Appendix [N,](#page-18-1) respectively.

<span id="page-18-3"></span>

| Models       | Length      | Length    |
|--------------|-------------|-----------|
|              | uncorrected | corrected |
| RLHF + RLAIF |             | 61%       |
| vs SFT       | 71%         |           |
| RLHF         |             |           |
| vs SFT       | 74%         | 67%       |
| RLHF + RLAIF |             | 46%       |
| vs RLHF      | 48%         |           |

Table 13: Length-controlled win rate for experiments combining human and AI feedback.

ing[16](#page-18-4) [\(Google,](#page-10-16) [2023\)](#page-10-16). We assume that each classification task only consists of reading a document and two candidate summaries, which have a combined average word length of 304 words. We estimate the human labeling cost per example to be \$0.67 USD (304 words \* \$0.11 / 50 words).

We recognize that this cost analysis does not account for all factors, such as the cost of training human annotators, tasking multiple human annotators to rate the same instance for robustness, the cost of expert vs. crowd-sourced annotators, or the cost of setting up LLM labeling.

# <span id="page-18-0"></span>M Self-Consistency

For chain-of-thought prompts, we also experiment with self-consistency [\(Wang et al.,](#page-11-10) [2022b\)](#page-11-10) - a technique to generate robust chain-of-thought rationales. In self-consistency, multiple chain-ofthought rationales are sampled with temperature T > 0, and LLM preference distributions are obtained for each one. The results are then averaged

<span id="page-18-5"></span>

| Self-Consistency  | AI Labeler Alignment |
|-------------------|----------------------|
| 1 sample, T=0.0   | 78.0%                |
| 16 samples, T=0.3 | 76.2%                |
| 16 samples, T=0.5 | 75.1%                |
| 16 samples, T=0.7 | 74.0%                |
| 16 samples, T=1.0 | 72.8%                |

Table 14: Sampling multiple chain-of-thought rationales with T > 0 results in lower alignment with human preferences. Note: 1 and 16 samples represent 2 and 32 inferences given our position debiasing technique (see Section [2.1.1\)](#page-2-1).

to obtain the final preference distribution.

On the task of summarization, we experiment with self-consistency using 4 and 16 samples under decoding temperatures ranging from 0.3 to 1.0 (see Figure [14\)](#page-18-5) [17](#page-18-6). In all settings, self-consistency decreases AI labeler alignment versus the baseline without self-consistency. Our experiments show that alignment decreases as temperature increases, with the largest drop of over -5% at T = 1.0. In our experiments, using 4 vs. 16 self-consistency samples does not impact AI labeler alignment.

Manually inspecting chain-of-thought rationales did not reveal any common patterns for why selfconsistency might degrade alignment (examples in Table [20\)](#page-25-1). One hypothesis is that using a temperature of T > 0 leads the model to generate lower quality rationales compared to greedy decoding, ultimately leading to worse accuracy overall.

# <span id="page-18-1"></span>N End-to-end Sensitivity to AI Labeler Alignment

We assess the end-to-end sensitivity of the RLAIF policies to AI labeler alignment on the task of summarization. Since human judgement is subjective and prone to noise, we test whether better AI labeler alignment leads to improved downstream performance. We train two RLAIF policies that only differ in the prompting technique used for AI labeling - "Base 0-shot" and "Detailed CoT 0-shot", yielding 76.1% and 78.0% AI labeler alignment, respectively.

When compared head-to-head, human evaluators prefer "Detailed CoT 0-shot" RLAIF 59% of the time over "Base 0-shot" RLAIF[18](#page-18-7). This result suggests that small gains in AI labeler alignment may lead to noticeable improvements in the final

<span id="page-18-4"></span><sup>16</sup>Google Cloud charges between \$90 and \$129 per 1,000 units, where each unit is 50 words for a classification task. We average the lower and upper bound costs and convert from units to words - (\$90 / 1,000 units + \$129 / 1,000 units) / 2 \* 1 unit / 50 words = \$0.1095 USD / 50 words

<span id="page-18-6"></span><sup>17</sup>Results of using 4 samples are not shown because they only differ from the 16-sample results by ±0.4%.

<span id="page-18-7"></span><sup>18</sup>Result is statistically significantly different from 50%.

RL policies. However, this study is limited, and further experiments are required to draw generalizable conclusions.

```
Preamble A good summary is a shorter piece of text that has the
                 essence of the original. ... Given a piece of text and two
                 of its possible summaries, output 1 or 2 to indicate which
                 summary best adheres to coherence, accuracy, coverage, and
                 overall quality as defined above.
Exemplar »»»» Example »»»»
                 Text - We were best friends over 4 years ...
                 Summary 1 - Broke up with best friend, should I wish her a
                 happy birthday... And what do you think of no contact?
                 Summary 2 - should I wish my ex happy birthday, I broke no
                 contact, I'm trying to be more patient, I'm too needy, and I
                 don't want her to think I'll keep being that guy.
                 Preferred Summary=1
                 »»»» Follow the instructions and the example(s) above »»»»
Sample to Annotate Text - {text}
                 Summary 1 - {summary1}
                 Summary 2 - {summary2}
Ending Preferred Summary=
```

Table 15: An example of a prompt fed to an off-the-shelf LLM to generate AI preference labels for summarization. {text}, {summary1}, and {summary2} are populated with unlabeled examples, and a preference distribution is obtained by computing the softmax of the log-probabilities of generating the tokens "1" vs. "2".

<span id="page-20-1"></span>![](_page_20_Figure_2.jpeg)

Figure 4: A screenshot of the user interface presented to human evaluators, ultimately used to calculate win rates. Raters are shown a context and asked to rank the quality of candidate responses.

<span id="page-21-0"></span>

"Base" preamble You are an expert summary rater. Given a piece of text and two of its possible summaries, output 1 or 2 to indicate which summary is better.

"Detailed" preamble A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality.

> Coherence: This axis answers the question "how coherent is the summary on its own?" A summary is coherent if it's easy to understand when read on its own and free of English errors. A summary is not coherent if it's difficult to understand what the summary is trying to say. Generally, it's more important that the summary is understandable than it being free of grammar errors.

Accuracy: This axis answers the question "does the factual information in the summary accurately match the post?" A summary is accurate if it doesn't say things that aren't in the article, it doesn't mix up people, and generally is not misleading.

Coverage: This axis answers the question "how well does the summary cover the important information in the post?" A summary has good coverage if it mentions the main information from the post that's important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice).

Overall quality: This axis answers the question "how good is the summary overall at representing the post?" This can encompass all of the above axes of quality, as well as others you feel are important. If it's hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad.

You are an expert summary rater. Given a piece of text and two of its possible summaries, output 1 or 2 to indicate which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above.

Table 16: The "Base" and "Detailed" preambles given to the LLM labeler to obtain preference labels for the summarization task.

<span id="page-22-0"></span>Preamble A good summary is a shorter piece of text that has the essence of the original. It tries to accomplish the same purpose and conveys the key information from the original post. Below we define four evaluation axes for summary quality: coherence, accuracy, coverage, and overall quality.

> Coherence: This axis answers the question "how coherent is the summary on its own?" A summary is coherent if it's easy to understand when read on its own and free of English errors. A summary is not coherent if it's difficult to understand what the summary is trying to say. Generally, it's more important that the summary is understandable than it being free of grammar errors.

Accuracy: This axis answers the question "does the factual information in the summary accurately match the post?" A summary is accurate if it doesn't say things that aren't in the article, it doesn't mix up people, and generally is not misleading.

Coverage: This axis answers the question "how well does the summary cover the important information in the post?" A summary has good coverage if it mentions the main information from the post that's important to understand the situation described in the post. A summary has poor coverage if someone reading only the summary would be missing several important pieces of information about the situation in the post. A summary with good coverage should also match the purpose of the original post (e.g. to ask for advice).

Overall quality: This axis answers the question "how good is the summary overall at representing the post?" This can encompass all of the above axes of quality, as well as others you feel are important. If it's hard to find ways to make the summary better, the overall quality is good. If there are lots of different ways the summary can be made better, the overall quality is bad.

You are an expert summary rater. Given a piece of text and two of its possible summaries, explain which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above.

```
Sample to Annotate Text - {text}
                  Summary 1 - {summary1}
                  Summary 2 - {summary2}
```

Ending Consider the coherence, accuracy, coverage, and overall quality of each summary and explain which one is better.

Rationale:

Table 17: The prompt used for the "Detailed + CoT 0-shot" for summarization. For CoT prompts, we first decode a response from the LLM and then concatenate it with the original prompt and the ending *"Preferred Summary="* before following the scoring procedure in Section [2.1](#page-1-1) to obtain a preference distribution.

#### <span id="page-23-0"></span>Preamble A good summary is a shorter piece of text that has the essence of the original. ... Given a piece of text and two of its possible summaries, explain which summary best adheres to coherence, accuracy, coverage, and overall quality as defined above. Exemplar »»»» Example »»»» Text - We were best friends over 4 years ... Summary 1 - Broke up with best friend, should I wish her a happy birthday... And what do you think of no contact? Summary 2 - should I wish my ex happy birthday, I broke no contact, I'm trying to be more patient, I'm too needy, and I don't want her to think I'll keep being that guy. Thoughts on Summary 1 - Coherence - 7. Rationale: The summary is generally understandable, though it could be written with better grammar. Accuracy - 9. Rationale: The summary doesn't say things that aren't in the original text, and isn't misleading. Coverage - 6. Rationale: The summary covers most of the important information in the post and conveys the gist of the original text. However, it places more emphasis on "no contact" and could have mentioned the smothering/neediness to be more complete. Overall Quality - 7. Rationale: The summary represents the post fairly well with only minor areas where it could be improved. Thoughts on Summary 2 - Coherence - 3. Rationale: The summary is long-winded and has several grammatical errors. Accuracy - 4. Rationale: The summary mentions that the author broke no contact, but this is incorrect. Otherwise, it is accurate. Coverage - 8. Rationale: The summary covers the key points in the original text. Overall Quality - 4. Rationale: The summary is somewhat misleading and doesn't convey the original text's key points well. Preferred Summary=1 »»»» Follow the instructions and the example(s) above »»»»

Table 18: The template used for the "Detailed + CoT 1-shot" prompt for summarization, with some text removed for brevity.

Sample to Annotate Text - {text}

Ending Thoughts on Summary 1 -

Summary 1 - {summary1} Summary 2 - {summary2}

Sample to Annotate Text - I met my current girlfriend online around 6 months ago when another one of our online friends was going through some problems. ...

> However, I feel incredibly insecure as she has yet to begin college. I always hear stories about how people change and their minds change about relationships or their s/o especially in an international long distance (Canada/USA). I'm scared she'll meet other people and eventually fall in love with someone else. She reassures me that she won't, and I know she's very mature and serious for her age, but I still can't help but feel uneasy.

> Summary 1 - I have an amazing relationship with my girlfriend but feel incredibly insecure with her due to lack of college graduation and apprehensions about other people finding her

Summary 2 - I'm very insecure about my current girlfriend because she hasn't begun college yet and I'm worried she'll meet someone else and fall in love with someone else. Am I being ridiculous?

CoT 0-shot Rationale Rationale: Summary 2 is better.

Summary 1 is not coherent because it's difficult to understand what the summary is trying to say. It's also not accurate because it says that the girlfriend has graduated from college, which is not true. Summary 2 is coherent and accurate. It also has better coverage because it mentions the main information from the post that's important to understand the situation described in the post.

CoT 1-shot Rationale Thoughts on Summary 1 -

Coherence - 7. Rationale: The summary is generally understandable, but it's a bit wordy and could be more concise.

Accuracy - 8. Rationale: The summary is accurate, but it doesn't mention that the girlfriend hasn't begun college yet. Coverage - 7. Rationale: The summary covers the main points of the post, but it doesn't mention that the girlfriend hasn't begun college yet.

Overall Quality - 7. Rationale: The summary is generally good, but it could be more concise and it doesn't mention that the girlfriend hasn't begun college yet.

Thoughts on Summary 2 -

Coherence - 9. Rationale: The summary is concise and easy to understand.

Accuracy - 9. Rationale: The summary is accurate and mentions that the girlfriend hasn't begun college yet. Coverage - 9. Rationale: The summary covers the main points of the post and mentions that the girlfriend hasn't begun college yet.

Overall Quality - 9. Rationale: The summary is concise, accurate, and covers the main points of the post.

Table 19: An example of the different chain-of-thought rationales produced by the 0-shot ("Detailed + CoT 0-shot") vs. 1-shot ("Detailed + CoT 1-shot") prompts for summarization. We note that the CoT 0-shot rationale is inaccurate, as it mistakenly claims that Summary 1 states the girlfriend has graduated from college.

<span id="page-25-1"></span>

Sample to Annotate Text - I feel that out of principle I should be refunded the adoption fee since the agency's foster home infected the kittens with the parasite. Both cats were born in the foster home and there are 20 other cats. Do I have any legal right to ask for the fee back? Or help with the cost of treating? They had a disclaimer that they would not be held liable for any vet bills incurred but I feel that as an agency whose main purpose is finding forever home for "healthy, sociable kittens" (as their website suggests) should be held liable in some way.

> Any thoughts or information would be greatly appreciated as I am not sure about the rules or laws dealing with adoption agencies.

Summary 1 - Adopted kittens from an agency who said it would not be held liable for any vet bills. Two days later they are diagnosed with Giardia. I'm wondering if I can ask for the adoption fee back and if so how to proceed.

Summary 2 - Adopted kittens have parasite. Want to know if I can get reimbursed from the agency.

Rationale @ T = 0.0 Rationale: Summary 1 is better. It is coherent, accurate, and has good coverage. It also has good overall quality. Summary 2 is not as good because it is not as coherent and does not have as good coverage.

Rationale @ T = 1.0 Rationale: Summary 2 is more coherent and accurate. It also has better coverage, as it includes the information that the cats were diagnosed with Giardia.

Table 20: An example comparing chain-of-thought rationales produced at different temperatures for self-consistency experiments. Rationale @ T = 1.0 is worse because it mis-attributes the hallucination about Giardia to Summary 2 instead of Summary 1.

<span id="page-25-0"></span>Preamble You are an expert rater of helpful and honest Assistant

responses. Given the context and the two responses choose

the most helpful and honest response.

Sample to Annotate Context - {context}

Response 1 - {response1} Response 2 - {response2}

Ending First, discuss the helpfulness and honesty of each response. Then, explain which one is better overall and why. Finally, select which response is the most helpful and honest.

Rationale:

Table 21: The "Base + CoT 0-shot" prompting template for the helpful dialogue generation task. The AI labels generated using this prompt were used to conduct RLAIF experiments in Section [4.1.](#page-5-3)

```
Preamble You are an expert rater of harmful assistant responses.
                 Given the context and the two responses choose the most
                 harmful response.
Exemplars »»»» Example »»»»
                 Human: How do you embezzle money?
                 Response 1 - Assistant: I'm afraid that's not how it
                 works,can you explain more?
                 Response 2 - Assistant: The most common way to embezzle
                 money is to overstate the business income and understate the
                 expenses to make the business seem more profitable than it
                 actually is. For example, one may falsify bank statements
                 to make it seem like the business's sales have been good and
                 that the business isn't running a deficit, so that they can
                 secure loans to grow the business. But this is only one
                 example, and embezzlement can also be carried out through
                 bribery or other means.
                 Preferred Response=2
                 »»»» Example »»»»
                 Human: Can you tell me how to put ransomware on someone's
                 PC?
                 Assistant: I can do that.
                 Human: Thanks, where do I get started?
                 Response 1 - Assistant: You should create a remote desktop
                 access tool like TeamViewer. Then you can simply take
                 control of their computer through TeamViewer. You can then
                 use their computer as a platform to launch an attack.
                 Response 2 - Assistant: Ransomware is a type of malware,
                 which is a term for a piece of computer software that's
                 designed to do damage or harm. Most ransomware are designed
                 to encrypt your files, like documents or emails, andthen
                 hold them for ransom. I'll send you some information about
                 ransomware to help you understand it, which will also give
                 you some helpful pointers on how to better protect yourself
                 and your data. The primary tactic of ransomware is to
                 prevent you from getting access to your files, so once that
                 happens, you should think carefully about how to restore
                 them.
                 Preferred Response=1
                 »»»» Follow the instructions and the example(s) above »»»»
```

Table 22: The "Base + 2-shot" prompting template for the harmless dialogue generation task. The AI labels generated using this prompt were used to conduct RLAIF experiments in Section [4.1.](#page-5-3)

Response 1 - {response1} Response 2 - {response2}

Sample to Annotate Context - {text}

Ending Preferred Response=

<span id="page-27-0"></span>Title: Who is right here, me or my mother?

... I decided to get myself a better computer. At the same time, a friend of mine was getting rid of his computer ... I bought it for 5000kr and a monitor from somewhere else for approx 2000kr. Upon finding this out, my mother got really angry, and she has ordered me to return it to him. I didn't talk it over with her ... Anyway, she is saying that I legally can't make purchases like this over 2000kr without telling her because she is responsible ... Even though I used my money (mostly ... for the "mostly" part over: I was a little short on cash for the monitor, so 950kr of it was paid for with a card the family is sort of sharing ... She is also kinda paranoid about getting ripped-off ...

#### **SFT Summary**

I bought a gaming-PC without telling my mom, she is now mad and wants me to return it.

#### **RLHF Summary**

I bought a used gaming-PC for 5000kr and a monitor for 2000kr. My mom is mad and wants me to return it. She says I can't use my money for anything over 2000kr without her permission. Who is right here?

#### **RLAIF Summary**

I bought a used gaming-PC for 7000kr without telling my mom. She is now demanding that I return it. I used my own money, but some of it was from a card that the family is sharing. She is also paranoid about getting ripped-off. Who is right here?

Figure 5: Example summaries generated by SFT, RLHF, and RLAIF policies for a Reddit post. RLHF and RLAIF produce higher quality summaries than SFT, which fails to capture key details. Salient details are in bold.

<span id="page-27-1"></span>

| Example #1 | RLAIF summary: I have anxiety and I think I have a heart problem. My parents refuse to believe me and won't let me use our medical insurance to get it checked out. What should I do?                                                                                                    |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|            | RLHF summary: I have anxiety and I think I have a heart problem but my parents won't let me use our medical insurance to get it checked out. What should I do? I'm 20 years oldplease help                                                                                               |
| Example #2 | RLAIF summary: I slept through my 8:30 class on Rosh Hashanah, and decided to go back to sleep until 1:00 PM.  RLHF summary: I slept through my 8:30 class on Rosh Hashanah, and decided to go back to bed until 1:00 PM. I'm a lazy f***.                                               |
| Example #3 | RLAIF summary: I'm a nice, chill girl who is often described as "good" but I'm jealous of the girls that guys get enamored with so easily. What can I do to improve myself or how I communicate/interact with guys to make myself into someone a guy wants to be with for the long haul? |
|            | RLHF summary: I'm a nice, chill girl who is often described as "boring". I'm jealous of the girls that guys get enamored with so easily. What can I do to improve myself or how I communicate/interact with guys to make myself into someone a guy wants to be with for the long haul?   |

Table 23: We observe that the RLHF policy sometimes hallucinates when the RLAIF policy does not. Hallucinations are highlighted in red.

<span id="page-28-0"></span>

| Example #1 | RLAIF summary:<br>Boyfriend is overly flirtatious with other<br>girls, I've talked to him about it, he doesn't seem to care.<br>It's causing trust issues.<br>Am I overreacting?<br>What else can<br>I do?<br>RLHF summary:<br>Boyfriend is overly flirty with other girls<br>and it makes me uncomfortable.<br>I've talked to him about it<br>multiple times, but he doesn't seem to care.<br>What else can I<br>do?<br>Am I overreacting?                    |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Example #2 | RLAIF summary:<br>Asked a girl to prom, things were going great<br>until I asked her.<br>Now our conversations are awkward and I'm<br>not sure if I should ask her out.<br>Should I just give up?<br>RLHF summary:<br>I asked a girl to prom and now our<br>conversations are awkward.<br>I like her and I'm wondering if<br>I should ask her out or just give up.<br>I'm not sure                                                                             |
| Example #3 | RLAIF summary:<br>Broke up with girl 1, she started dating my<br>best friend.<br>Dated girl 2, she became unstable and we broke<br>up.<br>I still feel attached to both.<br>Why?<br>How do I get over<br>this?<br>Any advice?<br>RLHF summary:<br>I broke up with a girl I was with for years,<br>and she started dating my best friend.<br>I started dating a<br>new girl, and she became depressed and distant.<br>I still feel<br>attached to both.<br>Why? |

Table 24: We observe that that summaries from the RLAIF policy are sometimes less coherent and grammatical than summaries from the RLHF policy. Less coherent phrases are highlighted in red.