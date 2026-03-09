# **ORPO**: Monolithic Preference Optimization without Reference Model

# Jiwoo Hong Noah Lee James Thorne

# KAIST AI

{jiwoo\_hong, noah.lee, thorne}@kaist.ac.kr

### Abstract

While recent preference alignment algorithms for language models have demonstrated promising results, supervised fine-tuning (SFT) remains imperative for achieving successful convergence. In this paper, we study the crucial role of SFT within the context of preference alignment, emphasizing that a minor penalty for the disfavored generation style is sufficient for preference-aligned SFT. Building on this foundation, we introduce a straightforward and innovative reference model-free monolithic odds ratio preference optimization algorithm, ORPO, eliminating the necessity for an additional preference alignment phase. We demonstrate, both empirically and theoretically, that the odds ratio is a sensible choice for contrasting favored and disfavored styles during SFT across the diverse sizes from 125M to 7B. Specifically, fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on the UltraFeedback alone surpasses the performance of state-of-the-art language models with more than 7B and 13B parameters: achieving up to 12.20% on AlpacaEval2.<sup>0</sup> (Figure [1\)](#page-0-0), 66.19% on IFEval (instruction-level loose, Table [6\)](#page-15-0), and 7.32 in MT-Bench (Figure [12\)](#page-18-0). We release code[1](#page-0-1) and model checkpoints for Mistral-ORPOα (7B)[2](#page-0-2) and Mistral-ORPO-β (7B).[3](#page-0-3)

#### 1 Introduction

Pre-trained language models (PLMs) with vast training corpora such as web texts [\(Gokaslan and](#page-10-0) [Cohen,](#page-10-0) [2019;](#page-10-0) [Penedo et al.,](#page-11-0) [2023\)](#page-11-0) or textbooks [\(Li](#page-10-1) [et al.,](#page-10-1) [2023c\)](#page-10-1) have shown remarkable abilities in diverse natural language processing (NLP) tasks [\(Brown et al.,](#page-9-0) [2020;](#page-9-0) [Zhang et al.,](#page-11-1) [2022;](#page-11-1) [Touvron](#page-11-2) [et al.,](#page-11-2) [2023;](#page-11-2) [Jiang et al.,](#page-10-2) [2023;](#page-10-2) [Almazrouei et al.,](#page-9-1) [2023\)](#page-9-1). However, the models must undergo further

<span id="page-0-0"></span>![](_page_0_Figure_12.jpeg)

Figure 1: AlpacaEval2.<sup>0</sup> result of Llama-2 (7B) and Mistral (7B) fine-tuned with ORPO (blue) in comparison to the state-of-the-art models. Notably, Mistral-ORPO-α & β surpasses Zephyr β and Llama-2-Chat (13B) with a single epoch training exclusively on the UltraFeedback.

tuning to be usable in general-domain applications, typically through processes such as *instruction tuning* and *preference alignment*.

Instruction-tuning [\(Wei et al.,](#page-11-3) [2022;](#page-11-3) [Taori et al.,](#page-11-4) [2023;](#page-11-4) [Wang et al.,](#page-11-5) [2023;](#page-11-5) [Zhou et al.,](#page-12-0) [2023a\)](#page-12-0) trains models to follow task descriptions given in natural language, which enables models to generalize well to previously unseen tasks. However, despite the ability to follow instructions, models may generate harmful or unethical outputs [\(Carlini et al.,](#page-9-2) [2021;](#page-9-2) [Gehman et al.,](#page-10-3) [2020;](#page-10-3) [Pryzant et al.,](#page-11-6) [2023\)](#page-11-6). To further align these models with human values, additional training is required with pairwise preference data using techniques such as reinforcement learning with human feedback [\(Ziegler et al.,](#page-12-1) [2020;](#page-12-1) [Stiennon et al.,](#page-11-7) [2022,](#page-11-7) RLHF) and direct preference optimization [\(Rafailov et al.,](#page-11-8) [2023,](#page-11-8) DPO).

Preference alignment methods have demonstrated success in several downstream tasks beyond reducing harm. For example, improving factuality [\(Tian et al.,](#page-11-9) [2023;](#page-11-9) [Cheng et al.,](#page-9-3) [2024;](#page-9-3) [Chen and](#page-9-4) [Li,](#page-9-4) [2024\)](#page-9-4), code-based question answering [\(Gorba](#page-10-4)[tovski and Kovalchuk,](#page-10-4) [2024\)](#page-10-4), and machine translation [\(Ramos et al.,](#page-11-10) [2023\)](#page-11-10). The versatility of alignment algorithms over a wide range of downstream

<span id="page-0-2"></span><span id="page-0-1"></span><sup>1</sup> <https://github.com/xfactlab/orpo>

<sup>2</sup> [https://huggingface.co/kaist-ai/](https://huggingface.co/kaist-ai/mistral-orpo-alpha) [mistral-orpo-alpha](https://huggingface.co/kaist-ai/mistral-orpo-alpha)

<span id="page-0-3"></span><sup>3</sup> [https://huggingface.co/kaist-ai/](https://huggingface.co/kaist-ai/mistral-orpo-beta) [mistral-orpo-beta](https://huggingface.co/kaist-ai/mistral-orpo-beta)

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 2: Comparison of model alignment techniques. ORPO aligns the language model *without a reference model* in a single-step manner by assigning a weak penalty to the rejected responses and a strong adaptation signal to the chosen responses with a simple log odds ratio term appended to the negative log-likelihood loss.

tasks highlights the necessity of understanding the alignment procedure and further improving the algorithms in terms of efficiency and performance. However, existing preference alignment methods normally consist of a multi-stage process, as shown in Figure [2,](#page-1-0) typically requiring a second reference model and a separate warm-up phase with supervised fine-tuning (SFT) [\(Ziegler et al.,](#page-12-1) [2020;](#page-12-1) [Rafailov et al.,](#page-11-8) [2023;](#page-11-8) [Wu et al.,](#page-11-11) [2023\)](#page-11-11).

In this paper, we study the role and impact of SFT in pairwise preference datasets for model alignment in Section [3](#page-2-0) and propose a simple and novel monolithic alignment method, odds ratio preference optimization (ORPO), which efficiently penalizes the model from learning undesired generation styles during SFT in Section [4.](#page-3-0) In contrast to previous works, our approach requires neither an SFT warm-up stage nor a reference model, enabling resource-efficient development of preference-based aligned models.

We demonstrate the effectiveness of our method with the evaluation of model alignment tasks and popular leaderboards in Section [6.1](#page-4-0) and [6.2](#page-5-0) by fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO. Then, we conduct controlled experiments comparing ORPO against established methods for model alignment, RLHF, and DPO for different datasets and model sizes in Section [6.3.](#page-5-1) Along with the post-hoc analysis of generation diversity in Section [6.4,](#page-6-0) we expound on the theoretical, empirical, and computational justification of utilizing the odds ratio in monolithic preference alignment in Section [7.3.](#page-8-0) We release the training code and the checkpoints for Mistral-ORPO-α (7B) and Mistral-ORPO-β (7B). These models achieve 7.24 and 7.32 in MT-Bench, 11.33% and 12.20%

on AlpacaEval2.<sup>0</sup> , and 61.63% and 66.19% in IFEval instruction-level loose accuracy, respectively.

# 2 Related Works

Alignment with Reinforcement Learning Reinforcement learning with human feedback (RLHF) commonly applies the Bradley-Terry model [\(Bradley and Terry,](#page-9-5) [1952\)](#page-9-5) to estimate the probability of a pairwise competition between two independently evaluated instances. An additional reward model is trained to score instances. Reinforcement learning algorithms such as proximal policy optimization (PPO) [\(Schulman et al.,](#page-11-12) [2017\)](#page-11-12) are employed to train the model to maximize the score of the reward model for the chosen response, resulting in language models that are trained with human preferences [\(Ziegler et al.,](#page-12-1) [2020;](#page-12-1) [Stiennon](#page-11-7) [et al.,](#page-11-7) [2022\)](#page-11-7). Notably, [Ouyang et al.](#page-10-5) [\(2022\)](#page-10-5) demonstrated the scalability and versatility of RLHF for instruction-following language models. Extensions such as language model feedback (RLAIF) could be a viable alternative to human feedback [\(Bai et al.,](#page-9-6) [2022b;](#page-9-6) [Lee et al.,](#page-10-6) [2023;](#page-10-6) [Pang et al.,](#page-11-13) [2023\)](#page-11-13). However, RLHF faces challenges of extensive hyperparameter searching due to the instability of PPO [\(Rafailov et al.,](#page-11-8) [2023;](#page-11-8) [Wu et al.,](#page-11-11) [2023\)](#page-11-11) and the sensitivity of the reward models [\(Gao et al.,](#page-10-7) [2022;](#page-10-7) [Wang et al.,](#page-11-14) [2024\)](#page-11-14). Therefore, there is a crucial need for stable preference alignment algorithms.

Alignment without Reward Model Several techniques for preference alignment mitigate the need for reinforcement learning [\(Rafailov et al.,](#page-11-8) [2023;](#page-11-8) [Song et al.,](#page-11-15) [2023;](#page-11-15) [Azar et al.,](#page-9-7) [2023;](#page-9-7) [Etha](#page-10-8)[yarajh et al.,](#page-10-8) [2024\)](#page-10-8). [Rafailov et al.](#page-11-8) [\(2023\)](#page-11-8) introduce direct policy optimization (DPO), which combines the reward modeling stage into the preference learning stage. [Azar et al.](#page-9-7) [\(2023\)](#page-9-7) prevented potential overfitting problems in DPO through identity preference optimization (IPO). [Ethayarajh et al.](#page-10-8) [\(2024\)](#page-10-8) and [Cai et al.](#page-9-8) [\(2023\)](#page-9-8) proposed Kahneman-Tversky Optimisation (KTO) and Unified Language Model Alignment (ULMA) that does not require the pairwise preference dataset, unlike RLHF and DPO. [Song et al.](#page-11-15) [\(2023\)](#page-11-15) further suggests incorporation of the softmax value of the reference response set in the negative log-likelihood loss to merge the supervised fine-tuning and preference alignment.

Alignment with Supervised Fine-tuning Preference alignment methods in reinforcement learning (RL) often leverage supervised fine-tuning (SFT) to ensure the stable update of the active policy in relation to the old policy [\(Schulman et al.,](#page-11-12) [2017\)](#page-11-12). This is because the SFT model is the old policy in the context of RLHF [\(Ziegler et al.,](#page-12-1) [2020\)](#page-12-1). Furthermore, empirical findings indicate that, even in non-RL alignment methods, the SFT model is crucial for achieving convergence to desired results [\(Rafailov et al.,](#page-11-8) [2023;](#page-11-8) [Tunstall et al.,](#page-11-16) [2023\)](#page-11-16).

In contrast, there have been approaches to build human-aligned language models by conducting SFT only with filtered datasets [\(Zhou et al.,](#page-12-0) [2023a;](#page-12-0) [Li et al.,](#page-10-9) [2023a;](#page-10-9) [Haggerty and Chandra,](#page-10-10) [2024;](#page-10-10) [Zhou](#page-12-2) [et al.,](#page-12-2) [2023b\)](#page-12-2). [Zhou et al.](#page-12-0) [\(2023a\)](#page-12-0) demonstrated that SFT with a small amount of data with finegrained filtering and curation could be sufficient for building helpful language model assistants. Furthermore, [Li et al.](#page-10-9) [\(2023a\)](#page-10-9) and [Haggerty and Chan](#page-10-10)[dra](#page-10-10) [\(2024\)](#page-10-10) proposed an iterative process of finetuning the supervised fine-tuned language models with their own generations after fine-grained selection of aligned generations and [Zhou et al.](#page-12-2) [\(2023b\)](#page-12-2) suggested that a curated subset of preference dataset is sufficient for alignment. While these works highlight the impact and significance of SFT in the context of alignment, the actual role of SFT and the theoretical background for incorporating preference alignment in SFT remains understudied.

#### <span id="page-2-0"></span>3 The Role of Supervised Fine-tuning

We study the behavior of supervised fine-tuning (SFT) as an initial stage of preference alignment methods [\(Ziegler et al.,](#page-12-1) [2020;](#page-12-1) [Rafailov et al.,](#page-11-8) [2023\)](#page-11-8) through analysis of the loss function in SFT and empirical demonstration of the preference comprehension ability of the trained SFT model. SFT plays a significant role in tailoring the pre-trained language models to the desired domain [\(Zhou et al.,](#page-12-0) [2023a;](#page-12-0) [Dong et al.,](#page-10-11) [2024\)](#page-10-11) by increasing the log probabilities of pertinent tokens. Nevertheless, this inadvertently increases the likelihood of generating tokens in undesirable styles, as illustrated in Figure [3.](#page-3-1) Therefore, it is necessary to develop methods capable of preserving the domain adaptation role of SFT while concurrently discerning and mitigating unwanted generation styles.

Absence of Penalty in Cross-Entropy Loss The goal of cross-entropy loss model fine-tuning is to penalize the model if the predicted logits for the reference answers are low, as shown in Equation [2.](#page-2-1)

$$\mathcal{L} = -\frac{1}{m} \sum_{k=1}^{m} \log P(\mathbf{x}^{(k)}, \mathbf{y}^{(k)})$$
 (1)

<span id="page-2-1"></span>
$$= -\frac{1}{m} \sum_{k=1}^{m} \sum_{i=1}^{|V|} y_i^{(k)} \cdot \log(p_i^{(k)})$$
 (2)

where y<sup>i</sup> is a boolean value that indicates if ith token in the vocabulary set V is a label token, p<sup>i</sup> refers to the probability of ith token, and m is the length of sequence. Using cross-entropy alone gives no direct penalty or compensation for the logits of non-answer tokens [\(Lin et al.,](#page-10-12) [2017\)](#page-10-12) as y<sup>i</sup> will be set to 0. While cross-entropy is generally effective for domain adaptation [\(Mao et al.,](#page-10-13) [2023\)](#page-10-13), there are no mechanisms to penalize rejected responses when compensating for the chosen responses. Therefore, the log probabilities of the tokens in the rejected responses increase along with the chosen responses, which is not desired from the viewpoint of preference alignment.

Generalization over Both Response Styles We conduct a pilot study to empirically demonstrate the miscalibration of chosen and rejected responses with supervised fine-tuning alone. We fine-tune OPT-350M [\(Zhang et al.,](#page-11-1) [2022\)](#page-11-1) on *the chosen responses only* from the HH-RLHF dataset [\(Bai et al.,](#page-9-6) [2022b\)](#page-9-6). Throughout the training, we monitor the log probability of rejected responses for each batch and report this in Figure [3.](#page-3-1) Both the log probability of chosen and rejected responses exhibited a simultaneous increase. This can be interpreted from two different perspectives. First, the cross-entropy loss effectively guides the model toward the intended domain (e.g., dialogue). However, the absence of a penalty for unwanted generations results in rejected responses sometimes having even higher log probabilities than the chosen ones.

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 3: Log probabilities for chosen and rejected responses during OPT-350M model fine-tuning on HH-RLHF dataset. Despite only chosen responses being used for supervision, rejected responses show a comparable likelihood of generation.

**Penalizing Undesired Generations** Appending an unlikelihood penalty to the loss has demonstrated success in reducing unwanted degenerative traits in models (Welleck et al., 2019; Li et al., 2020). For example, to prevent repetitions, an unwanted token set of previous contexts,  $k \in \mathcal{C}_{recent}$ , is disfavored by adding the following term to  $(1-p_i^{(k)})$  to the loss (such as Equation 2) which penalizes the model for assigning high probabilities to recent tokens. Motivated by SFT ascribing high probabilities to rejected tokens (Figure 3) and the effectiveness of appending penalizing unwanted traits, we design a monolithic preference alignment method that dynamically penalizes the disfavored response for each query without the need for crafting sets of rejected tokens.

### <span id="page-3-0"></span>4 Odds Ratio Preference Optimization

We introduce a novel preference alignment algorithm, Odds Ratio Preference Optimization (ORPO), which incorporates an odds ratio-based penalty to the conventional negative log-likelihood (NLL) loss for differentiating the generation styles between favored and disfavored responses.

#### 4.1 Preliminaries

Given an input sequence x, the average log-likelihood of generating the output sequence y, of length m tokens, is computed as Equation 3. The odds of generating the output sequence y given an input sequence x is defined in Equation 4:

<span id="page-3-2"></span>
$$\log P_{\theta}(y|x) = \frac{1}{m} \sum_{t=1}^{m} \log P_{\theta}(y_t|x, y_{< t}) \quad (3)$$

<span id="page-3-3"></span>
$$\mathbf{odds}_{\theta}(y|x) = \frac{P_{\theta}(y|x)}{1 - P_{\theta}(y|x)} \tag{4}$$

Intuitively,  $\mathbf{odds}_{\theta}(y|x) = k$  implies that it is k times more likely for the model  $\theta$  to generate the output sequence y than not generating it. Thus, the odds ratio of the chosen response  $y_w$  over the rejected response  $y_l$ ,  $\mathbf{OR}_{\theta}(y_w, y_l)$ , indicates how much more likely it is for the model  $\theta$  to generate  $y_w$  than  $y_l$  given input x, defined in Equation 5.

<span id="page-3-4"></span>
$$\mathbf{OR}_{\theta}(y_w, y_l) = \frac{\mathbf{odds}_{\theta}(y_w|x)}{\mathbf{odds}_{\theta}(y_l|x)}$$
 (5)

#### 4.2 Objective Function of ORPO

The objective function of ORPO in Equation 6 consists of two components: 1) supervised fine-tuning (SFT) loss ( $\mathcal{L}_{SFT}$ ); 2) relative ratio loss ( $\mathcal{L}_{OR}$ ).

<span id="page-3-5"></span>
$$\mathcal{L}_{ORPO} = \mathbb{E}_{(x, y_w, y_l)} \left[ \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR} \right]$$
 (6)

 $\mathcal{L}_{SFT}$  follows the conventional causal language modeling negative log-likelihood (NLL) loss function to maximize the likelihood of generating the reference tokens as previously discussed in Section 3.  $\mathcal{L}_{OR}$  in Equation 7 maximizes the odds ratio between the likelihood of generating the disfavored response  $y_w$  and the disfavored response  $y_l$ . We wrap the log odds ratio with the log sigmoid function so that  $\mathcal{L}_{OR}$  could be minimized by increasing the log odds ratio between  $y_w$  and  $y_l$ .

<span id="page-3-6"></span>
$$\mathcal{L}_{OR} = -\log\sigma\left(\log\frac{\mathbf{odds}_{\theta}(y_w|x)}{\mathbf{odds}_{\theta}(y_t|x)}\right) \qquad (7)$$

Together,  $\mathcal{L}_{SFT}$  and  $\mathcal{L}_{OR}$  weighted with  $\lambda$  tailor the pre-trained language model to adapt to the specific subset of the desired domain and disfavor generations in the rejected response sets.

#### <span id="page-3-11"></span>4.3 Gradient of ORPO

The gradient of  $\mathcal{L}_{Ratio}$  further justifies using the odds ratio loss. It comprises two terms: one that penalizes the wrong predictions and one that contrasts between chosen and rejected responses, denoted in Equation  $8^4$  for  $d = (x, y_l, y_w) \sim D$ .

<span id="page-3-10"></span><span id="page-3-9"></span><span id="page-3-7"></span>
$$\nabla_{\theta} \mathcal{L}_{OR} = \delta(d) \cdot h(d) \tag{8}$$

$$\delta(d) = \left[1 + \frac{\mathbf{odds}_{\theta} P(y_w|x)}{\mathbf{odds}_{\theta} P(y_l|x)}\right]^{-1} \tag{9}$$

$$h(d) = \frac{\nabla_{\theta} \log P_{\theta}(y_w|x)}{1 - P_{\theta}(y_w|x)} - \frac{\nabla_{\theta} \log P_{\theta}(y_l|x)}{1 - P_{\theta}(y_l|x)}$$
(10)

<span id="page-3-8"></span><sup>&</sup>lt;sup>4</sup>The full derivation for  $\nabla_{\theta} \mathcal{L}_{OR}$  is in Appendix A.

When the odds of the favored responses are relatively higher than the disfavored responses, δ(d) in Equation [9](#page-3-9) will converge to 0. This indicates that the δ(d) will play the role of a penalty term, accelerating the parameter updates if the model is more likely to generate the rejected responses.

Meanwhile, h(d) in Equation [10](#page-3-10) implies a weighted contrast of the two gradients from the chosen and rejected responses. Specifically, 1−P(y|x) in the denominators amplifies the gradients when the corresponding side of the likelihood P(y|x) is low. For the chosen responses, this accelerates the model's adaptation toward the distribution of chosen responses as the likelihood increases.

# 5 Experimental Settings

# 5.1 Training Configurations

Models We train a series of OPT models [\(Zhang](#page-11-1) [et al.,](#page-11-1) [2022\)](#page-11-1) scaling from 125M to 1.3B parameters comparing supervised fine-tuning (SFT), proximal policy optimization (PPO), direct policy optimization (DPO), and compare these to our ORPO. PPO and DPO models were fine-tuned with TRL library [\(von Werra et al.,](#page-11-18) [2020\)](#page-11-18) on top of SFT models trained for a single epoch on the chosen responses following [Rafailov et al.](#page-11-8) [\(2023\)](#page-11-8) and [Tunstall et al.](#page-11-16) [\(2023\)](#page-11-16). We notate this by prepending "+" to each algorithm (e.g., +DPO). Additionally, we train Phi-2 (2.7B) [\(Javaheripi and Bubeck,](#page-10-15) [2023\)](#page-10-15), a pre-trained language model with promising downstream performance [\(Beeching et al.,](#page-9-9) [2023\)](#page-9-9), as well as Llama-2 (7B) [\(Touvron et al.,](#page-11-2) [2023\)](#page-11-2) and Mistral (7B) [\(Jiang et al.,](#page-10-2) [2023\)](#page-10-2). Further training details for each method are in Appendix [C.](#page-14-0)

Datasets We test each training configuration and model on two datasets: 1) Anthropic's HH-RLHF [\(Bai et al.,](#page-9-10) [2022a\)](#page-9-10), 2) Binarized UltraFeedback [\(Tunstall et al.,](#page-11-16) [2023\)](#page-11-16). We filtered out instances where y<sup>w</sup> = y<sup>l</sup> or where y<sup>w</sup> = ∅ or where y<sup>l</sup> = ∅.

Reward Models We train OPT-350M and OPT-1.3B on each dataset for a single epoch for reward modeling with the objective function in Equation [11](#page-4-1) [\(Ziegler et al.,](#page-12-1) [2020\)](#page-12-1). The OPT-350M reward model was used for PPO, and OPT-1.3B reward model was used to assess the generations of finetuned models. We refer to these reward models as RM-350M and RM-1.3B in Section [6.](#page-4-2)

<span id="page-4-1"></span>
$$-\mathbb{E}_{(x,y_l,y_w)}\left[\log\sigma\left(r(x,y_w)-r(x,y_l)\right)\right] \quad (11)$$

#### 5.2 Leaderboard Evaluation

In Section [6.1,](#page-4-0) we evaluate the models using the AlpacaEval1.<sup>0</sup> and AlpacaEval2.<sup>0</sup> [\(Li et al.,](#page-10-16) [2023b\)](#page-10-16) benchmarks, comparing ORPO to other instructiontuned models reported in the official leaderboard,[5](#page-4-3) including Llama-2 Chat (7B) and (13B) [\(Touvron](#page-11-2) [et al.,](#page-11-2) [2023\)](#page-11-2), and Zephyr α and β [\(Almazrouei](#page-9-1) [et al.,](#page-9-1) [2023\)](#page-9-1). Similarly, in Section [6.2,](#page-5-0) we evaluate the models with MT-Bench [\(Zheng et al.,](#page-12-3) [2023\)](#page-12-3) and report the results and the scores of the same models reported in the official leaderboard.[6](#page-4-4) Using GPT-4 [\(Achiam et al.,](#page-9-11) [2023\)](#page-9-11) as an evaluator in AlpacaEval1.<sup>0</sup> , we assess if the trained model can be preferred over the responses generated from text-davinci-003. For AlpacaEval2.<sup>0</sup> , we used GPT-4-turbo[7](#page-4-5) as an evaluator following the default setting. We assess if the generated responses are favored over those generated from GPT-4. Finally, using GPT-4 as an evaluator in MT-Bench, we check if the models can follow the instructions with hard answers in a multi-turn conversation.

# <span id="page-4-2"></span>6 Results and Analysis

First, we assess the general instruction-following abilities of the models by comparing the preference alignment algorithms in Sections [6.1](#page-4-0) and [6.2.](#page-5-0) Second, we measure the win rate of OPT models trained with ORPO against other alignment methods training OPT 1.3B as a reward model in Section [6.3.](#page-5-1) Then, we measure the lexical diversity of the models trained with ORPO and DPO in Section [6.4.](#page-6-0)

#### <span id="page-4-0"></span>6.1 Single-turn Instruction Following

Phi-2 (2.7B) ORPO improved pre-trained Phi-2 to exceed the performance of the Llama-2 Chat instruction-following language model by *only using UltraFeedback* as the instruction-tuning dataset, as shown in Table [1.](#page-5-2) λ of 0.25 was applied for Phi-2, resulting in 71.80% and 6.35% in AlpacaEval.

Llama-2 (7B) Notably, UltraFeedback and ORPO with λ of 0.2 on Llama-2 (7B) resulted in higher AlpacaEval scores than the chat versions of both 7B and 13B scale trained with RLHF, eventually showing 81.26% and 9.44% in both AlpacaEvals.

In contrast, in our controlled experimental setting of conducting one epoch of SFT and three

<span id="page-4-4"></span><span id="page-4-3"></span><sup>5</sup> [https://tatsu-lab.github.io/alpaca\\_eval/](https://tatsu-lab.github.io/alpaca_eval/) 6 [https://huggingface.co/spaces/lmsys/](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

<span id="page-4-5"></span>[chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 7 [https://platform.openai.com/docs/models/](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo) [gpt-4-and-gpt-4-turbo](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)

<span id="page-5-2"></span>

| Model Name               | Size | AlpacaEval1.0 | AlpacaEval2.0 |
|--------------------------|------|---------------|---------------|
| Phi-2 + SFT              | 2.7B | 48.37% (1.77) | 0.11% (0.06)  |
| Phi-2 + SFT + DPO        | 2.7B | 50.63% (1.77) | 0.78% (0.22)  |
| Phi-2 + ORPO<br>(Ours)   | 2.7B | 71.80% (1.59) | 6.35% (0.74)  |
| Llama-2 Chat *           | 7B   | 71.34% (1.59) | 4.96% (0.67)  |
| Llama-2 Chat *           | 13B  | 81.09% (1.38) | 7.70% (0.83)  |
| Llama-2 + ORPO<br>(Ours) | 7B   | 81.26% (1.37) | 9.44% (0.85)  |
| Zephyr (α) *             | 7B   | 85.76% (1.23) | 8.35% (0.87)  |
| Zephyr (β) *             | 7B   | 90.60% (1.03) | 10.99% (0.96) |
| Mistral-ORPO-α<br>(Ours) | 7B   | 87.92% (1.14) | 11.33% (0.97) |
| Mistral-ORPO-β<br>(Ours) | 7B   | 91.41% (1.15) | 12.20% (0.98) |

Table 1: Table of instruction-following abilities of each checkpoint measured through AlpacaEval. While clearly showing the improvements in instruction-following abilities after training with ORPO, it is notable that ORPO models exceed RLHF or DPO models of Llama-2 and Mistral (\* indicates the results from the official leaderboard.)

Loading [MathJax]/extensions/MathMenu.js

epochs of DPO following [Tunstall et al.](#page-11-16) [\(2023\)](#page-11-16) and [Rafailov et al.](#page-11-8) [\(2023\)](#page-11-8), Llama-2 + SFT and Llama-2 + SFT + DPO yielded models with outputs that could not be evaluated. This supports the efficacy of ORPO, in which the model can rapidly learn the desired domain and the preference with limited data. This aligns with the h(d) examination in the gradient of our method studied in Section [4.3.](#page-3-11)

Mistral-**ORPO**-α (7B) Furthermore, fine-tuning Mistral (7B) with single-turn conversation dataset, UltraFeedback, and ORPO with λ of 0.1 outperforms Zephyr series, which are the Mistral (7B) models fine-tuned with SFT on 20K UltraChat [\(Ding et al.,](#page-10-17) [2023\)](#page-10-17) and DPO on the full UltraFeedback. As shown in Table [1,](#page-5-2) Mistral-ORPO-α (7B) achieves 87.92% and 11.33%, which exceeds Zephyr α by 1.98% and Zephyr β by 0.34% in AlpacaEval2.<sup>0</sup> . The sample responses and corresponding references from GPT-4 can be found in Appendix [I.](#page-20-0)

Mistral-**ORPO**-β (7B) Using the same configuration of Mistral-ORPO-α (7B), we additionally compare fine-tuning Mistral on the cleaned version of the UltraFeedback[8](#page-5-3) to demonstrate the effect of the data quality [\(Bartolome et al.,](#page-9-12) [2023\)](#page-9-12). While the actual sizes of datasets are similar, ORPO gains further advantages from the dataset quality by scoring over 91% and 12% on AlpacaEval, as shown in Table [1.](#page-5-2) Further instruction-following evaluation on two Mistral-based models with IFEval [\(Zhou](#page-12-4) [et al.,](#page-12-4) [2023c\)](#page-12-4) is reported in the Appendix [D.](#page-15-1)

<span id="page-5-4"></span>![](_page_5_Figure_6.jpeg)

Figure 4: MT-Bench result of Mistral-ORPO-α (7B) and Mistral-ORPO-β (7B) by the category. Further comparison can be found in the Appendix [G.](#page-18-1)

#### <span id="page-5-0"></span>6.2 Multi-turn Instruction Following

With our best model, Mistral-ORPO-α (7B) and Mistral-ORPO-β (7B), we also assess the multi-turn instruction-following skills with deterministic answers (e.g., math) through MT-Bench.

As shown in Figure [4,](#page-5-4) ORPO-Mistral (7B) series achieve comparable results to either larger or the proprietary models, including Llama-2-Chat (70B) and Claude. Eventually, Mistral-ORPO-α (7B) and Mistral-ORPO-β (7B) scored 7.23 and 7.32 in MT-Bench without being exposed to the multi-turn conversation dataset during training.

#### <span id="page-5-1"></span>6.3 Reward Model Win Rate

We assess the win rate of ORPO over other preference alignment methods, including supervised fine-tuning (SFT), PPO, and DPO, using RM-1.3B to understand the effectiveness and scalability of ORPO in Tables [2](#page-6-1) and [3.](#page-6-2) Additionally, we visually verify that ORPO can effectively enhance the

<span id="page-5-3"></span><sup>8</sup> [https://huggingface.co/datasets/argilla/](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) [ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)

<span id="page-6-3"></span>![](_page_6_Figure_0.jpeg)

![](_page_6_Figure_1.jpeg)

![](_page_6_Figure_2.jpeg)

Figure 5: Reward distribution comparison between OPT-125M (left), OPT-350M (middle), and OPT-1.3B (right) trained with SFT (blue), RLHF (green), DPO (orange), and ORPO (red) on the test set of UltraFeedback using the RM-1.3B. While the rewards of the trained models are roughly normal and preference optimization algorithms (RLHF, DPO, and ORPO) tend to move the reward distribution in the positive direction, ORPO is on par or better than RLHF and DPO in increasing the expected reward. The same plot for the HH-RLHF dataset is in Appendix [F.](#page-18-2)

expected reward compared to SFT in Figure [5.](#page-6-3)

HH-RLHF In Table [2,](#page-6-1) ORPO outperforms SFT and PPO across all model scales. The highest win rate against SFT and PPO across the size of the model was 78.0% and 79.4%, respectively. Meanwhile, the win rate over DPO was correlated to the model's size, with the largest model having the highest win rate: 70.9%.

<span id="page-6-1"></span>

| ORPO vs  | SFT         | +DPO        | +PPO        |
|----------|-------------|-------------|-------------|
| OPT-125M | 84.0 (0.62) | 41.7 (0.77) | 66.1 (0.26) |
| OPT-350M | 82.7 (0.56) | 49.4 (0.54) | 79.4 (0.29) |
| OPT-1.3B | 78.0 (0.16) | 70.9 (0.52) | 65.9 (0.33) |

Table 2: Average win rate (%) and its standard deviation of ORPO and standard deviation over other methods on HH-RLHF dataset for three rounds. Sampling decoding with a temperature of 1.0 was used on the test set.

UltraFeedback The win rate in UltraFeedback followed similar trends to what was reported in HH-RLHF, as shown in Table [3.](#page-6-2) ORPO was preferred over SFT and PPO for maximum 80.5% and 85.8%, respectively. While consistently preferring ORPO over SFT and PPO, the win rate over DPO gradually increases as the size of the model increases. The scale-wise trend exceeding DPO will be further shown through 2.7B models in Section [6.1.](#page-4-0)

Overall Reward Distribution In addition to the win rate, we compare the reward distribution of the responses generated with respect to the test set of the UltraFeedback dataset in Figure [5](#page-6-3) and HH-RLHF dataset in Appendix [F.](#page-18-2) Regarding the SFT reward distribution as a default, PPO, DPO, and ORPO shift it in both datasets. However, the magnitude of reward shifts for each algorithm differs.

<span id="page-6-2"></span>

| ORPO vs  | SFT         | +DPO        | +PPO        |
|----------|-------------|-------------|-------------|
| OPT-125M | 73.2 (0.12) | 48.8 (0.29) | 71.4 (0.28) |
| OPT-350M | 80.5 (0.54) | 50.5 (0.17) | 85.8 (0.62) |
| OPT-1.3B | 69.4 (0.57) | 57.8 (0.73) | 65.7 (1.07) |

Table 3: Average win rate (%) and its standard deviation of ORPO and standard deviation over other methods on UltraFeedback dataset for three rounds. Sampling decoding with a temperature of 1.0 was used.

In Figure [5,](#page-6-3) RLHF (i.e., SFT + PPO) has some abnormal properties of the distribution with a low expected reward. We attribute this to empirical evidence of the instability and reward mismatch problem of RLHF [\(Rafailov et al.,](#page-11-8) [2023;](#page-11-8) [Gao et al.,](#page-10-7) [2022;](#page-10-7) [Shen et al.,](#page-11-19) [2023\)](#page-11-19) as the RLHF models were trained with RM-350M and assessed with RM-1.3B. Meanwhile, it is notable that the ORPO distribution (red) is mainly located on the very right side of each subplot, indicating higher expected rewards. Recalling the intent of preference alignment methods, the distributions in Figure [5](#page-6-3) indicate that ORPO tends to fulfill the aim of preference alignment for all model sizes.

#### <span id="page-6-0"></span>6.4 Lexical Diversity

The lexical diversity of the preference-aligned language models was studied in previous works [\(Kirk](#page-10-18) [et al.,](#page-10-18) [2024\)](#page-10-18). We expand the concept of per-input and across-input diversity introduced in [Kirk et al.](#page-10-18) [\(2024\)](#page-10-18) by using Gemini-Pro [\(Gemini Team et al.,](#page-10-19) [2023\)](#page-10-19) as an embedding model, which is suitable for assessing the diversity of instruction-following language models by encoding a maximum of 2048 tokens. The diversity metric with the given set of

sampled responses is defined as Equation 13.

$$\mathcal{O}_{\theta}^{i} := \{ y_{i} \sim \theta(y|x_{i}) | j = 1, 2, ..., K \}$$
 (12)

<span id="page-7-0"></span>
$$D(\mathcal{O}_{\theta}^{i}) = \frac{1}{2} \cdot \frac{\sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \cos(h_{i}, h_{j})}{N \cdot (N-1)}$$
 (13)

where  $\cos(h_i,h_j)$  refers to the cosine similarity between the embedding  $h_i$  and  $h_j$ . 5 different responses are sampled with a temperature of 1.0 to 160 queries in AlpacaEval (i.e., K=5, N=160) using Phi-2 and Llama-2 trained with ORPO and DPO. We report the results in Table 4.

**Per Input Diversity (PID)** We average the inputwise average cosine similarity between the generated samples with Equation 14 to assess the perinput diversity. In Table 4, ORPO models have the highest average cosine similarity in the first column for both models, which implies the lowest diversity per input. This indicates that ORPO generally assigns high probabilities to the desired tokens, while DPO has a relatively smoother logit distribution.

<span id="page-7-2"></span>
$$PID_D(\theta) = \frac{1}{N} \sum_{i=1}^{N} D(\mathcal{O}_{\theta}^i)$$
 (14)

Across Input Diversity (AID) Using 8 samples generated per input, we sample the first item for each input and examine their inter cosine similarity with Equation 15 for across-input diversity. Unlike per-input diversity, it is noteworthy that Phi-2 (ORPO) has lower average cosine similarity in the second row of Table 4. We can infer that ORPO triggers the model to generate more instruction-specific responses than DPO.

<span id="page-7-3"></span>
$$AID_D(\theta) = D\left(\bigcup_{i=1}^N \mathcal{O}^i, \theta, j=1\right)$$
 (15)

<span id="page-7-1"></span>

|                     | Per Input↓ | Across Input↓ |
|---------------------|------------|---------------|
| Phi-2 + SFT + DPO   | 0.8012     | 0.6019        |
| Phi-2 + ORPO        | 0.8909     | 0.5173        |
| Llama-2 + SFT + DPO | 0.8889     | 0.5658        |
| Llama-2 + ORPO      | 0.9008     | 0.5091        |

Table 4: Lexical diversity of Phi-2 and Llama-2 finetuned with DPO and ORPO. Lower cosine similarity is equivalent to higher diversity. The highest value in each column within the same model family is bolded.

#### 7 Discussion

In this section, we expound on the theoretical and computational details of ORPO. The theoretical analysis of ORPO is studied in Section 7.1, which will be supported with the empirical analysis in Section 7.2. Then, we compare the computational load of DPO and ORPO in Section 7.3.

#### <span id="page-7-4"></span>7.1 Comparison to Probability Ratio

The rationale for selecting the odds ratio instead of the probability ratio lies in its stability. The probability ratio for generating the favored response  $y_w$  over the disfavored response  $y_l$  given an input sequence x can be defined as Equation 16.

<span id="page-7-5"></span>
$$\mathbf{PR}_{\theta}(y_w, y_l) = \frac{P_{\theta}(y_w|x)}{P_{\theta}(y_l|x)}$$
(16)

While this formulation has been used in previous preference alignment methods that precede SFT (Rafailov et al., 2023; Azar et al., 2023), the odds ratio is a better choice in the setting where the preference alignment is incorporated in SFT as the odds ratio is more sensitive to the model's preference understanding. In other words, the probability ratio leads to more extreme discrimination of the disfavored responses than the odds ratio.

We visualize this through the sample distributions of the log probability ratio  $\log \mathbf{PR}(X_2|X_1)$  and log odds ratio  $\log \mathbf{OR}(X_2|X_1)$ . We sample 50,000 samples each with Equation 17 and plot the log probability ratio and log odds ratio in Figure 6. We multiply  $\beta$  for the probability ratio as it is practiced in the probability ratio-based methods and report the cases where  $\beta=0.2$  and  $\beta=1.0$ .

<span id="page-7-6"></span>
$$X_1, X_2 \sim \text{Unif}(0, 1) \tag{17}$$

$$Y \sim \beta \left(\log X_1 - \log X_2\right) \tag{18}$$

$$Y \sim \log \frac{X_1}{1 - X_1} - \log \frac{X_2}{1 - X_2}$$
 (19)

Recalling that the log sigmoid function is applied to the log probability ratio and log odds ratio, each ratio's scale determines the expected margin between the likelihood of the favored and disfavored styles when the loss is minimized. In that sense, the contrast should be relatively extreme to minimize the log sigmoid loss when  $\mathbf{PR}(X_2|X_1)$  is inputted instead of  $\mathbf{OR}(X_2|X_1)$  to the log sigmoid function, regarding the sharp distribution of  $\log \mathbf{PR}(X_2|X_1)$ 

<span id="page-8-2"></span>![](_page_8_Figure_0.jpeg)

Figure 6: Sampled distribution of log PR(X2|X1) and log OR(X2|X1). log OR(X2|X1) has a wider range given the same input probability pairs (X1, X2).

in Figure [6.](#page-8-2) This results in overly suppressing the logits for the tokens in the disfavored responses in the setting where SFT and preference alignment are incorporated, as the model is not adapted to the domain. We empirically support this analysis through the ablation study in Appendix [B.](#page-14-1) Therefore, the odds ratio is a better choice when the preference alignment is done with SFT due to the mild discrimination of disfavored responses and the prioritizing of the favored responses to be generated.

Throughout fine-tuning, minimizing the log sigmoid loss leads to either PR(X2|X1) or OR(X2|X1) to be larger. This is equivalent to the rejected responses' token-wise likelihood, which will generally get smaller. In this context, it is essential to avoid an overly extreme contrast. This precaution is especially important given the sharp distribution of log PR(X2|X1) depicted in Figure [6.](#page-8-2) The excessive margin could lead to the unwarranted suppression of logits for tokens in disfavored responses within the incorporated setting, potentially resulting in issues of degeneration.

# <span id="page-8-1"></span>7.2 Minimizing LOR

We demonstrate that models trained with ORPO learned to reflect the preference throughout the training process. We monitored the log probabilities of the chosen and rejected responses and the log odds ratio with λ = 1.0. With the same dataset and model as Figure [3,](#page-3-1) Figure [7](#page-8-3) shows that the log probability of rejected responses is diminishing while that of chosen responses is on par with Figure [3](#page-3-1) as the log odds ratio increases. This indicates that ORPO is successfully preserving the domain adaptation role of SFT while the penalty term LOR induces the model to lower the likelihood of unwanted generations. We discuss the effect of λ in Equation [6](#page-3-5) in Appendix [E,](#page-16-0) studying the proclivity

of the log probability margin between the favored and disfavored responses with respect to λ.

<span id="page-8-3"></span>![](_page_8_Figure_7.jpeg)

Figure 7: Average log-likelihood for chosen and rejected responses and log odds ratio per batch. The odds consistently increase during training with ORPO.

### <span id="page-8-0"></span>7.3 Computational Efficiency

As depicted in Figure [2,](#page-1-0) ORPO does not require a reference model, unlike RLHF and DPO. In that sense, ORPO is computationally more efficient than RLHF and DPO in two perspectives: 1) memory allocation and 2) fewer FLOPs per batch.

The reference model (πSF T ) in the context of RLHF and DPO denotes the model trained with supervised fine-tuning (SFT), which will be the baseline model for updating the parameters with RLHF or DPO [\(Ziegler et al.,](#page-12-1) [2020;](#page-12-1) [Rafailov et al.,](#page-11-8) [2023\)](#page-11-8). Thus, two πSF T s, a frozen reference model and the model undergoing tuning, are required during training. Furthermore, in theory, two forward passes should be calculated for each model to acquire the logits for the chosen and rejected responses. In other words, four forward passes happen in total for a single batch. On the other hand, a reference model is not required in ORPO as πSF T is directly updated. This leads to half the number of forward passes required for each batch during training.

# 8 Conclusion

In this paper, we introduced a reference-free monolithic preference alignment method, odds ratio preference optimization (ORPO), by revisiting and understanding the value of the supervised fine-tuning (SFT) phase in the context of preference alignment. ORPO was consistently preferred by the fine-tuned reward model against SFT and RLHF across the scale, and the win rate against DPO increased as the size of the model increased. Furthermore, we validate the scalability of ORPO with 2.7B and 7B pre-trained language models by exceeding the larger state-of-the-art instruction-following language models in AlpacaEval. Specifically, Mistral-ORPO-α and Mistral-ORPO-β achieved 11.33% and

12.20% in AlpacaEval2.<sup>0</sup> , 7.23 and 7.32 in MT-Bench, thereby underscoring the efficiency and effectiveness of ORPO. We release fine-tuning code and model checkpoints for Mistral-ORPO-α and Mistral-ORPO-β to aid reproducibility.

# Limitations

While conducting a comprehensive analysis of the diverse preference alignment methods, including DPO and RLHF, we did not incorporate a more comprehensive range of preference alignment algorithms. We leave the broader range of comparison against other methods as future work, along with scaling our method to over 7B models. In addition, we will expand the fine-tuning datasets into diverse domains and qualities, thereby verifying the generalizability of our method in various NLP downstream tasks. Finally, we would like to study the internal impact of our method on the pre-trained language model, expanding the understanding of preference alignment procedure to not only the supervised fine-tuning stage but also consecutive preference alignment algorithms.

# References

- <span id="page-9-11"></span>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*.
- <span id="page-9-1"></span>Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023. [The falcon series of open language](http://arxiv.org/abs/2311.16867) [models.](http://arxiv.org/abs/2311.16867)
- <span id="page-9-7"></span>Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. 2023. [A general theoret](http://arxiv.org/abs/2310.12036)[ical paradigm to understand learning from human](http://arxiv.org/abs/2310.12036) [preferences.](http://arxiv.org/abs/2310.12036)
- <span id="page-9-10"></span>Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022a. [Training a](http://arxiv.org/abs/2204.05862) [helpful and harmless assistant with reinforcement](http://arxiv.org/abs/2204.05862) [learning from human feedback.](http://arxiv.org/abs/2204.05862)

- <span id="page-9-6"></span>Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022b. [Constitutional ai: Harmless](http://arxiv.org/abs/2212.08073)[ness from ai feedback.](http://arxiv.org/abs/2212.08073)
- <span id="page-9-12"></span>Alvaro Bartolome, Gabriel Martin, and Daniel Vila. 2023. Notus. [https://github.com/argilla-io/](https://github.com/argilla-io/notus) [notus](https://github.com/argilla-io/notus).
- <span id="page-9-9"></span>Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. 2023. Open llm leaderboard. [https://huggingface.co/](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) [spaces/HuggingFaceH4/open\\_llm\\_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).
- <span id="page-9-5"></span>Ralph Allan Bradley and Milton E. Terry. 1952. [Rank](http://www.jstor.org/stable/2334029) [analysis of incomplete block designs: I. the method](http://www.jstor.org/stable/2334029) [of paired comparisons.](http://www.jstor.org/stable/2334029) *Biometrika*, 39(3/4):324– 345.
- <span id="page-9-0"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. [Language models are few-shot learners.](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc.
- <span id="page-9-8"></span>Tianchi Cai, Xierui Song, Jiyan Jiang, Fei Teng, Jinjie Gu, and Guannan Zhang. 2023. [Ulma: Unified lan](https://api.semanticscholar.org/CorpusID:265659430)[guage model alignment with demonstration and point](https://api.semanticscholar.org/CorpusID:265659430)[wise human preference.](https://api.semanticscholar.org/CorpusID:265659430) *ArXiv*, abs/2312.02554.
- <span id="page-9-2"></span>Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Raffel. 2021. [Extracting training data from large language models.](http://arxiv.org/abs/2012.07805)
- <span id="page-9-4"></span>Weixin Chen and Bo Li. 2024. [Grath: Gradual self](http://arxiv.org/abs/2401.12292)[truthifying for large language models.](http://arxiv.org/abs/2401.12292)
- <span id="page-9-3"></span>Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang Li,

- Kai Chen, and Xipeng Qiu. 2024. [Can ai assistants](http://arxiv.org/abs/2401.13275) [know what they don't know?](http://arxiv.org/abs/2401.13275)
- <span id="page-10-20"></span>Tri Dao. 2023. [Flashattention-2: Faster attention with](http://arxiv.org/abs/2307.08691) [better parallelism and work partitioning.](http://arxiv.org/abs/2307.08691)
- <span id="page-10-22"></span>Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. [Qlora: Efficient finetuning](http://arxiv.org/abs/2305.14314) [of quantized llms.](http://arxiv.org/abs/2305.14314)
- <span id="page-10-17"></span>Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. [Enhancing chat language](http://arxiv.org/abs/2305.14233) [models by scaling high-quality instructional conver](http://arxiv.org/abs/2305.14233)[sations.](http://arxiv.org/abs/2305.14233)
- <span id="page-10-11"></span>Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. 2024. [How](http://arxiv.org/abs/2310.05492) [abilities in large language models are affected by](http://arxiv.org/abs/2310.05492) [supervised fine-tuning data composition.](http://arxiv.org/abs/2310.05492)
- <span id="page-10-8"></span>Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. *arXiv preprint arXiv:2402.01306*.
- <span id="page-10-7"></span>Leo Gao, John Schulman, and Jacob Hilton. 2022. [Scal](http://arxiv.org/abs/2210.10760)[ing laws for reward model overoptimization.](http://arxiv.org/abs/2210.10760)
- <span id="page-10-23"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. [A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836) [evaluation.](https://doi.org/10.5281/zenodo.10256836)
- <span id="page-10-3"></span>Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020. [RealToxi](https://doi.org/10.18653/v1/2020.findings-emnlp.301)[cityPrompts: Evaluating neural toxic degeneration](https://doi.org/10.18653/v1/2020.findings-emnlp.301) [in language models.](https://doi.org/10.18653/v1/2020.findings-emnlp.301) In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 3356–3369, Online. Association for Computational Linguistics.
- <span id="page-10-19"></span>Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*.
- <span id="page-10-0"></span>Aaron Gokaslan and Vanya Cohen. 2019. Openwebtext corpus. [http://Skylion007.github.io/](http://Skylion007.github.io/OpenWebTextCorpus) [OpenWebTextCorpus](http://Skylion007.github.io/OpenWebTextCorpus).
- <span id="page-10-4"></span>Alexey Gorbatovski and Sergey Kovalchuk. 2024. [Re](http://arxiv.org/abs/2401.10882)[inforcement learning for question answering in pro](http://arxiv.org/abs/2401.10882)[gramming domain using public community scoring](http://arxiv.org/abs/2401.10882) [as a human feedback.](http://arxiv.org/abs/2401.10882)
- <span id="page-10-10"></span>Hamish Haggerty and Rohitash Chandra. 2024. [Self](http://arxiv.org/abs/2401.00692)[supervised learning for skin cancer diagnosis with](http://arxiv.org/abs/2401.00692) [limited training data.](http://arxiv.org/abs/2401.00692)

- <span id="page-10-15"></span>Mojan Javaheripi and Sébastien Bubeck. 2023. [Phi-2:](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) [The surprising power of small language models.](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- <span id="page-10-2"></span>Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. [Mistral 7b.](http://arxiv.org/abs/2310.06825)
- <span id="page-10-18"></span>Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2024. [Understanding the ef](http://arxiv.org/abs/2310.06452)[fects of rlhf on llm generalisation and diversity.](http://arxiv.org/abs/2310.06452)
- <span id="page-10-6"></span>Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. 2023. [Rlaif: Scaling reinforcement](http://arxiv.org/abs/2309.00267) [learning from human feedback with ai feedback.](http://arxiv.org/abs/2309.00267)
- <span id="page-10-14"></span>Margaret Li, Stephen Roller, Ilia Kulikov, Sean Welleck, Y-Lan Boureau, Kyunghyun Cho, and Jason Weston. 2020. [Don't say that! making inconsistent dialogue](https://doi.org/10.18653/v1/2020.acl-main.428) [unlikely with unlikelihood training.](https://doi.org/10.18653/v1/2020.acl-main.428) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 4715–4728, Online. Association for Computational Linguistics.
- <span id="page-10-9"></span>Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. 2023a. [Self-alignment with instruction back](http://arxiv.org/abs/2308.06259)[translation.](http://arxiv.org/abs/2308.06259)
- <span id="page-10-16"></span>Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023b. Alpacaeval: An automatic evaluator of instruction-following models. [https://github.com/tatsu-lab/alpaca\\_eval](https://github.com/tatsu-lab/alpaca_eval).
- <span id="page-10-1"></span>Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023c. [Textbooks are all you need ii: phi-1.5 technical re](http://arxiv.org/abs/2309.05463)[port.](http://arxiv.org/abs/2309.05463)
- <span id="page-10-12"></span>Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision*, pages 2980–2988.
- <span id="page-10-21"></span>Ilya Loshchilov and Frank Hutter. 2019. [Decoupled](http://arxiv.org/abs/1711.05101) [weight decay regularization.](http://arxiv.org/abs/1711.05101)
- <span id="page-10-13"></span>Anqi Mao, Mehryar Mohri, and Yutao Zhong. 2023. [Cross-entropy loss functions: Theoretical analysis](http://arxiv.org/abs/2304.07288) [and applications.](http://arxiv.org/abs/2304.07288)
- <span id="page-10-5"></span>Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. [Training language models to follow instructions with](http://arxiv.org/abs/2203.02155) [human feedback.](http://arxiv.org/abs/2203.02155)

- <span id="page-11-13"></span>Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. 2023. [Language model self-improvement by](http://arxiv.org/abs/2305.14483) [reinforcement learning contemplation.](http://arxiv.org/abs/2305.14483)
- <span id="page-11-0"></span>Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. [The refinedweb dataset for](http://arxiv.org/abs/2306.01116) [falcon llm: Outperforming curated corpora with web](http://arxiv.org/abs/2306.01116) [data, and web data only.](http://arxiv.org/abs/2306.01116)
- <span id="page-11-6"></span>Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. 2023. [Automatic prompt op](https://doi.org/10.18653/v1/2023.emnlp-main.494)[timization with "gradient descent" and beam search.](https://doi.org/10.18653/v1/2023.emnlp-main.494) In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 7957–7968, Singapore. Association for Computational Linguistics.
- <span id="page-11-8"></span>Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. [Direct preference optimization: Your language](http://arxiv.org/abs/2305.18290) [model is secretly a reward model.](http://arxiv.org/abs/2305.18290)
- <span id="page-11-10"></span>Miguel Moura Ramos, Patrick Fernandes, António Farinhas, and André F. T. Martins. 2023. [Aligning neu](http://arxiv.org/abs/2311.09132)[ral machine translation models: Human feedback in](http://arxiv.org/abs/2311.09132) [training and inference.](http://arxiv.org/abs/2311.09132)
- <span id="page-11-20"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. [Deepspeed: System opti](https://doi.org/10.1145/3394486.3406703)[mizations enable training deep learning models with](https://doi.org/10.1145/3394486.3406703) [over 100 billion parameters.](https://doi.org/10.1145/3394486.3406703) In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD '20, page 3505–3506, New York, NY, USA. Association for Computing Machinery.
- <span id="page-11-12"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. [Proximal policy](http://arxiv.org/abs/1707.06347) [optimization algorithms.](http://arxiv.org/abs/1707.06347)
- <span id="page-11-19"></span>Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023. [Loose lips sink ships: Mitigating length bias](https://doi.org/10.18653/v1/2023.findings-emnlp.188) [in reinforcement learning from human feedback.](https://doi.org/10.18653/v1/2023.findings-emnlp.188) In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 2859–2873, Singapore. Association for Computational Linguistics.
- <span id="page-11-15"></span>Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. 2023. [Pref](http://arxiv.org/abs/2306.17492)[erence ranking optimization for human alignment.](http://arxiv.org/abs/2306.17492)
- <span id="page-11-7"></span>Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022. [Learning](http://arxiv.org/abs/2009.01325) [to summarize from human feedback.](http://arxiv.org/abs/2009.01325)
- <span id="page-11-4"></span>Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. [https://](https://github.com/tatsu-lab/stanford_alpaca) [github.com/tatsu-lab/stanford\\_alpaca](https://github.com/tatsu-lab/stanford_alpaca).

- <span id="page-11-9"></span>Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, and Chelsea Finn. 2023. [Fine](http://arxiv.org/abs/2311.08401)[tuning language models for factuality.](http://arxiv.org/abs/2311.08401)
- <span id="page-11-2"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. [Llama: Open](http://arxiv.org/abs/2302.13971) [and efficient foundation language models.](http://arxiv.org/abs/2302.13971)
- <span id="page-11-16"></span>Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023. [Zephyr: Direct distillation of lm alignment.](http://arxiv.org/abs/2310.16944)
- <span id="page-11-18"></span>Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, and Shengyi Huang. 2020. Trl: Transformer reinforcement learning. [https://github.](https://github.com/huggingface/trl) [com/huggingface/trl](https://github.com/huggingface/trl).
- <span id="page-11-14"></span>Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, Songyang Gao, Nuo Xu, Yuhao Zhou, Xiaoran Fan, Zhiheng Xi, Jun Zhao, Xiao Wang, Tao Ji, Hang Yan, Lixing Shen, Zhan Chen, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. 2024. [Secrets of](http://arxiv.org/abs/2401.06080) [rlhf in large language models part ii: Reward model](http://arxiv.org/abs/2401.06080)[ing.](http://arxiv.org/abs/2401.06080)
- <span id="page-11-5"></span>Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023. [How](http://arxiv.org/abs/2306.04751) [far can camels go? exploring the state of instruction](http://arxiv.org/abs/2306.04751) [tuning on open resources.](http://arxiv.org/abs/2306.04751)
- <span id="page-11-3"></span>Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. [Finetuned](http://arxiv.org/abs/2109.01652) [language models are zero-shot learners.](http://arxiv.org/abs/2109.01652)
- <span id="page-11-17"></span>Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. 2019. Neural text generation with unlikelihood training. *arXiv preprint arXiv:1908.04319*.
- <span id="page-11-11"></span>Tianhao Wu, Banghua Zhu, Ruoyu Zhang, Zhaojin Wen, Kannan Ramchandran, and Jiantao Jiao. 2023. [Pair](http://arxiv.org/abs/2310.00212)[wise proximal policy optimization: Harnessing rela](http://arxiv.org/abs/2310.00212)[tive feedback for llm alignment.](http://arxiv.org/abs/2310.00212)
- <span id="page-11-1"></span>Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. [Opt: Open pre](http://arxiv.org/abs/2205.01068)[trained transformer language models.](http://arxiv.org/abs/2205.01068)

- <span id="page-12-5"></span>Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. 2023. [Pytorch fsdp: Experiences on scaling](http://arxiv.org/abs/2304.11277) [fully sharded data parallel.](http://arxiv.org/abs/2304.11277)
- <span id="page-12-3"></span>Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. [Judging](https://doi.org/10.48550/arXiv.2306.05685) [LLM-as-a-Judge with MT-Bench and Chatbot Arena.](https://doi.org/10.48550/arXiv.2306.05685) ArXiv:2306.05685 [cs].
- <span id="page-12-0"></span>Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023a. [Lima:](http://arxiv.org/abs/2305.11206) [Less is more for alignment.](http://arxiv.org/abs/2305.11206)
- <span id="page-12-2"></span>Haotian Zhou, Tingkai Liu, Qianli Ma, Jianbo Yuan, Pengfei Liu, Yang You, and Hongxia Yang. 2023b. [Lobass: Gauging learnability in supervised fine](https://api.semanticscholar.org/CorpusID:264406231)[tuning data.](https://api.semanticscholar.org/CorpusID:264406231) *ArXiv*, abs/2310.13008.
- <span id="page-12-4"></span>Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023c. [Instruction-following evaluation for](http://arxiv.org/abs/2311.07911) [large language models.](http://arxiv.org/abs/2311.07911)
- <span id="page-12-1"></span>Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. [Fine-tuning lan](http://arxiv.org/abs/1909.08593)[guage models from human preferences.](http://arxiv.org/abs/1909.08593)

### <span id="page-13-0"></span>**A** Derivation of $\nabla_{\theta} \mathcal{L}_{OR}$ with Odds Ratio

Suppose that  $g(x, y_l, y_w) = \frac{\text{odds}_{\theta} P(y_w|x)}{\text{odds}_{\theta} P(y_l|x)}$ 

$$\nabla_{\theta} \mathcal{L}_{OR} = \nabla_{\theta} \log \sigma \left( \log \frac{\mathbf{odds}_{\theta} P(y_w | x)}{\mathbf{odds}_{\theta} P(y_l | x)} \right)$$
(20)

$$= \frac{\sigma'(\log g(x, y_l, y_w))}{\sigma(\log g(x, y_l, y_w))}$$
(21)

$$= \sigma\left(-\log g(x, y_l, y_w)\right) \cdot \nabla_{\theta} \log g(x, y_l, y_w) \tag{22}$$

$$= \frac{\sigma\left(-\log g(x, y_l, y_w)\right)}{g(x, y_l, y_w)} \cdot \nabla_{\theta} g(x, y_l, y_w)$$
(23)

$$= \sigma\left(-\log g(x, y_l, y_w)\right) \cdot \nabla_{\theta} \log g(x, y_l, y_w) \tag{24}$$

<span id="page-13-1"></span>
$$= \left(1 + \frac{\operatorname{odds}_{\theta} P(y_w|x)}{\operatorname{odds}_{\theta} P(y_l|x)}\right)^{-1} \cdot \nabla_{\theta} \log \frac{\operatorname{odds}_{\theta} P(y_w|x)}{\operatorname{odds}_{\theta} P(y_l|x)}$$
(25)

In Equation 25, the remaining derivative can be further simplified by replacing  $1 - P_{\theta}(y|x)$  terms where  $P(y|x) = \sqrt[N]{\prod_{t}^{N} P_{\theta}(y_{t}|x, y_{< t})}$  in  $\mathbf{odds}_{\theta}(y|x)$  as follows.

$$\nabla_{\theta} \log (1 - P_{\theta}(y|x)) = \frac{\nabla_{\theta} (1 - P_{\theta}(y|x))}{1 - P_{\theta}(y|x)}$$
(26)

$$=\frac{-\nabla_{\theta}P_{\theta}(y|x)}{1-P_{\theta}(y|x)}\tag{27}$$

$$= -\frac{P_{\theta}(y|x)}{1 - P_{\theta}(y|x)} \cdot \nabla_{\theta} \log P_{\theta}(y|x)$$
(28)

$$= \mathbf{odds}_{\theta}(y|x) \cdot \nabla_{\theta} \log P_{\theta}(y|x) \tag{29}$$

$$\nabla_{\theta} \log \frac{\mathbf{odds}_{\theta} P(y_w | x)}{\mathbf{odds}_{\theta} P(y_l | x)} = \nabla_{\theta} \log \frac{P_{\theta}(y_w | x)}{P_{\theta}(y_l | x)} - \left(\nabla_{\theta} \log(1 - P_{\theta}(y_w | x)) - \nabla_{\theta} \log(1 - P_{\theta}(y_l | x))\right)$$
(30)

$$= (1 + \mathbf{odds}_{\theta} P(y_w|x)) \nabla_{\theta} \log P_{\theta}(y_w|x) - (1 + \mathbf{odds}_{\theta} P(y_l|x)) \nabla_{\theta} \log P_{\theta}(y_l|x)$$
(31)

Therefore, the final form of  $\nabla_{\theta} \mathcal{L}_{OR}$  would be

$$\nabla_{\theta} \mathcal{L}_{OR} = \frac{1 + \mathbf{odds}_{\theta} P(y_w | x)}{1 + \frac{\mathbf{odds}_{\theta} P(y_w | x)}{\mathbf{odds}_{\theta} P(y_l | x)}} \cdot \nabla_{\theta} \log P_{\theta}(y_w | x) - \frac{1 + \mathbf{odds}_{\theta} P(y_l | x)}{1 + \frac{\mathbf{odds}_{\theta} P(y_w | x)}{\mathbf{odds}_{\theta} P(y_l | x)}} \cdot \nabla_{\theta} \log P_{\theta}(y_l | x)$$
(32)

$$= \left(1 + \frac{\operatorname{odds}_{\theta} P(y_w|x)}{\operatorname{odds}_{\theta} P(y_l|x)}\right)^{-1} \cdot \left(\frac{\nabla_{\theta} \log P_{\theta}(y_w|x)}{1 - P(y_w|x)} - \frac{\nabla_{\theta} \log P_{\theta}(y_l|x)}{1 - P(y_l|x)}\right)$$
(33)

#### <span id="page-14-1"></span>B Ablation on Probability Ratio and Odds Ratio

In this section, we continue the discussion in Section [7.1](#page-7-4) through empirical results comparing the log probabilities of chosen and rejected responses in UltraFeedback when trained with probability ratio and odds ratio. Recalling the sensitivity of each ratio discussed in Section [7.1,](#page-7-4) it is expected for the probability ratio to lower the log probabilities of the rejected responses with a larger scale than the odds ratio. This is well-shown in Figure [8,](#page-14-2) which is the log probabilities of each batch while fine-tuning with probability ratio (left) rapidly reaches under -4, while the same phenomenon happens after the over-fitting occurs in the case of odds ratio (right).

<span id="page-14-2"></span>![](_page_14_Figure_2.jpeg)

Figure 8: The log probability trace when the model is trained with the probability ratio (left) and the odds ratio (right) given the same hyperparameters. The probability ratio leads the rejected responses to have relatively lower log probabilities.

#### <span id="page-14-0"></span>C Experimental Details

Flash-Attention 2 [\(Dao,](#page-10-20) [2023\)](#page-10-20) is applied for all the pre-trained models for computational efficiency. In particular, the OPT series and Phi-2 (2.7B) were trained with DeepSpeed ZeRO 2 [\(Rasley et al.,](#page-11-20) [2020\)](#page-11-20), Llama-2 (7B) and Mistral (7B) were trained with Fully Sharded Data Parallel(FSDP) [\(Zhao et al.,](#page-12-5) [2023\)](#page-12-5). 7B and 2.7B models were trained with four and two NVIDIA A100, and the rest were trained on four NVIDIA A6000. For optimizer, AdamW optimizer [\(Loshchilov and Hutter,](#page-10-21) [2019\)](#page-10-21) and paged AdamW [\(Dettmers et al.,](#page-10-22) [2023\)](#page-10-22) were used, and the linear warmup with cosine decay was applied for the learning rate. For input length, every instance was truncated and padded to 1,024 tokens and 2,048 tokens for HH-RLHF and UltraFeedback, respectively. To guarantee that the models can sufficiently learn to generate the proper response to the conversation history or the complex instruction, we filtered instances with prompts with more than 1,024 tokens.

Supervised Fine-tuning (SFT) For SFT, the maximum learning rate was set to 1e-5. Following [Ziegler](#page-12-1) [et al.](#page-12-1) [\(2020\)](#page-12-1) and [Rafailov et al.](#page-11-8) [\(2023\)](#page-11-8), the training epoch is set to 1.

Reinforcement Learning with Human Feedback (RLHF) For RLHF, the hyperparameters were set as Table [5](#page-15-2) for UltraFeedback. For the HH-RLHF dataset, the output\_min\_length and output\_max\_length were set to 64 and 256.

Direct Preference Optimization (DPO) For DPO, β was set to 0.1 for every case. The learning rate was set to 5e-6, and the model was trained for three epochs to select the best model by evaluation loss in each epoch. However, in most cases, the first or the second checkpoint was selected as the best model as the evaluation loss increased from the third epoch.

<span id="page-15-2"></span>

| Hyperparameter              | Setting |  |
|-----------------------------|---------|--|
| ppo_epoch                   | 4       |  |
| init_kl_coef                | 0.1     |  |
| horizon                     | 2,000   |  |
| batch_size                  | 64      |  |
| mini_batch_size             | 8       |  |
| gradient_accumulation_steps | 1       |  |
| output_min_length           | 128     |  |
| output_max_length           | 512     |  |
| optimizer                   | AdamW   |  |
| learning_rate               | 1e-05   |  |
| gamma                       | 0.99    |  |

Table 5: Hyperparameter settings for RLHF.

Odds Ratio Preference Optimization (**ORPO**) As ORPO does not require any special hyperparameter, only the learning rate and epoch were the only hyperparameter to set. For ORPO, the maximum learning rate was set to 8e-6 and trained for 10 epochs. The best model is selected based on the lowest evaluation loss for the OPT series, Phi-2 (2.7B) and Llama-2 (7B).

# <span id="page-15-1"></span>D IFEval Result for Mistral-**ORPO**-α and Mistral-**ORPO**-β

<span id="page-15-0"></span>Along with the AlpacaEval results reported in Section [6.1,](#page-4-0) we report the results of Mistral-ORPO-α and Mistral-ORPO-β on IFEval [\(Zhou et al.,](#page-12-4) [2023c\)](#page-12-4), calculated with the codes from [Gao et al.](#page-10-23) [\(2023\)](#page-10-23). Additional information can be found in: <https://jiwooya1000.github.io/posts/orpo/#ifeval>.

| Model Type     | Prompt-Strict | Prompt-Loose | Inst-Strict | Inst-Loose |
|----------------|---------------|--------------|-------------|------------|
| Mistral-ORPO-α | 0.5009        | 0.5083       | 0.5995      | 0.6163     |
| Mistral-ORPO-β | 0.5287        | 0.5564       | 0.6355      | 0.6619     |

Table 6: IFEval scores of Mistral-ORPO-α and Mistral-ORPO-β.

### <span id="page-16-0"></span>E Ablation on the Weighting Value (λ)

For the weighting value λ in Equation [6,](#page-3-5) we conduct an ablation study with {0.1, 0.5, 1.0}. Mistral (7B) and UltraFeedback were used for the base model and dataset. In Section [E.1,](#page-16-1) we compare the log probability trends by the value of λ, and we assess the downstream effect of λ in Section [E.2.](#page-16-2)

### <span id="page-16-1"></span>E.1 Log Probability

<span id="page-16-3"></span>![](_page_16_Figure_3.jpeg)

Figure 9: The log probability trend by λ. With larger λ (e.g., λ = 1.0), LOR gets more influential in fine-tuning the models with ORPO.

In Figure [9,](#page-16-3) we find that larger λ leads to stronger discrimination of the rejected responses in general. With λ = 0.1, the average log probability of the chosen and the rejected responses stay close as the fine-tuning proceeds. Also, unlike other settings, the log probabilities for the rejected responses do not decrease, but rather, the log probabilities of the chosen responses increase to minimize LOR term.

Moreover, in λ = 0.5, there exists a similar trend of further increasing the log probabilities of the chosen responses, but the log probabilities of the rejected responses are diminishing simultaneously. Lastly, in λ = 1.0, the chosen responses diminish along with the rejected responses while enlarging the margin between them. However, this does not mean smaller λ is always the better. It will depend on the specific need and model.

#### <span id="page-16-2"></span>E.2 MT-Bench

![](_page_16_Figure_8.jpeg)

Figure 10: MT-Bench result comparison by differing λ = 0.1 and λ = 1.0.

The downstream impact of λ stands out in the MT-Bench result. In comparison to λ = 0.1, Mistral+ORPO (7B) with λ = 1.0 performs worse in extraction, math, and reasoning, which are the categories that generally require deterministic answers. On the other hand, it performs better in STEM, humanities, and roleplay, which ask the generations without hard answers. Along with the amount of discrepancy between the trend in the logits of chosen and rejected responses, we can infer that making a more significant margin between the chosen and the rejected responses through higher λ in ORPO leads to overly adapting to the chosen responses set in the training dataset. This proclivity results in open-ended generations generally being preferred by the annotator while showing weaker performance in the hard-answered questions.

### <span id="page-18-2"></span>F Test Set Reward Distribution on HH-RLHF

Along with Figure [11,](#page-18-3) which depicts the reward distribution of OPT2-125M, OPT2-350M, and OPT2-1.3B on the UltraFeedback dataset, we report the reward distribution of each pre-trained checkpoint trained on the HH-RLHF dataset. As discussed in Section [6.3,](#page-5-1) ORPO consistently pushes the reward distribution of SFT to the right side.

<span id="page-18-3"></span>![](_page_18_Figure_2.jpeg)

Figure 11: Reward distribution comparison between OPT-125M (left), OPT-350M (middle), and OPT-1.3B (right) trained with SFT (blue), RLHF (green), DPO (orange), and ORPO (red) on the test set of HH-RLHF using the 1.3B reward model. General tendency follows that of Figure [5.](#page-6-3)

# <span id="page-18-1"></span>G MT-Bench Result of Mistral-**ORPO**-α (7B) and Mistral-**ORPO**-β (7B)

For the MT-Bench result in Section [6.2,](#page-5-0) we report the category-wise scores of Mistral-**ORPO**-α (7B) and Mistral-**ORPO**-β (7B) in Figure [12.](#page-18-0) While surpassing Llama-2 Chat (13B) and Llama-2 Chat (70B) in most cases, Mistral-**ORPO**-β (7B) is comparable to GPT-3.5-turbo in the categories that require descriptive generations. However, it lacks coding and math skills, which we speculate is due to the lack of training data, as we used 61k instances in UltraFeedback.

<span id="page-18-0"></span>![](_page_18_Figure_6.jpeg)

Figure 12: MT-Bench result of Mistral-**ORPO**-α (7B) and Mistral-**ORPO**-β (7B) by the category.

### H Special Instructions for Verbosity Assessment

For the succinctness and verboseness instructions, we generated five different instructions, each with ChatGPT [9](#page-19-0) . From the instructions in Table [7,](#page-19-1) we randomly sampled one prompt each for every batch to prevent potential word bias.

<span id="page-19-1"></span>

| # | Succinctness                                  | Verboseness                                         |
|---|-----------------------------------------------|-----------------------------------------------------|
| 1 | Please generate a short and concise response. | Please generate an elaborative and chatty response. |
| 2 | Provide a brief and concise answer.           | Provide a detailed answer.                          |
| 3 | Keep your reply short and to the point.       | Keep your reply elaborative and intricate.          |
| 4 | Keep your answer brief for clarity.           | Keep your answer detailed.                          |
| 5 | Generate a brief and to-the-point answer.     | Generate a chatty and step-wise answer.             |

Table 7: Instructions prepended to the queries from AlpacaEval. Each instruction set asks the model to generate either shorter or longer responses given the query, respectively.

<span id="page-19-0"></span><sup>9</sup> <https://chat.openai.com/>

### <span id="page-20-0"></span>I Generation Samples from Mistral-**ORPO**-α (7B)

In this section, we two sample generations from Mistral-**ORPO**-α (7B) on AlpacaEval, including the one which is preferred over the response of GPT-4 and the one in which GPT-4 was preferred. For readability, the instances with short responses are selected for both cases. Additionally, the responses for Mistral-ORPO-α (7B) were sampled with a temperature of 0.7. The checkpoints for Mistral-ORPO-α (7B) and Mistral-ORPO-β (7B) can be found in <https://huggingface.co/kaist-ai/mistral-orpo-alpha> and <https://huggingface.co/kaist-ai/mistral-orpo-beta>.

Query 1 (GPT-4 Win): Give a brief description of the book that you are given. Matilda

#### 1. GPT-4:

"Matilda" is a children's novel written by the British author Roald Dahl. It was first published in 1988. The story revolves around a very intelligent and precocious young girl named Matilda Wormwood. Despite being neglected and mistreated by her ignorant and self-absorbed parents, Matilda finds solace in reading books and quickly develops a love for learning.

At the age of five, she begins attending school and meets Miss Honey, a kindhearted teacher who recognizes Matilda's extraordinary intellectual abilities. However, the school's tyrannical headmistress, Miss Trunchbull, is a cruel and oppressive figure who takes pleasure in punishing the children with her unreasonable and harsh methods.

Matilda discovers that she possesses a unique telekinetic ability, which she learns to control and use to her advantage. With her wit, intelligence, and special powers, Matilda sets out to defend her friends from Miss Trunchbull's abuses and to help Miss Honey overcome her own difficult past.

The book is a tale of good triumphing over evil, and it celebrates the power of knowledge, kindness, and standing up for what is right. It has been adapted into a successful film, stage musical, and has remained a beloved classic in children's literature.

#### 2. Mistral-**ORPO**-β:

"Matilda" is a children's book written by Roald Dahl, published in 1988. The story revolves around the life of Matilda Wormwood, a brilliant and gifted five-year-old girl who has telekinetic powers. The story explores themes of education, family, and the importance of reading and intellectual curiosity. The book has been praised for its wit, humor, and powerful message, making it a classic and beloved children's book.

#### 1. GPT-4:

As an AI developed by OpenAI, I don't have personal opinions. However, I can share that ChatGPT is an advanced language model that has received considerable attention for its ability to generate human-like text. It's designed to assist users in a wide range of tasks, from answering questions to creating content and engaging in conversation.

ChatGPT has been praised for its fluency and versatility in handling various topics and for its potential to streamline and enhance communication in both personal and professional settings. However, like any technology, it also faces challenges and criticisms, such as ensuring the accuracy of information, preventing misuse, and managing ethical considerations surrounding AI-generated content.

Overall, ChatGPT represents a significant achievement in natural language processing and has sparked discussions about the future of AI in our daily lives.

# 2. Mistral-**ORPO**-β:

As an AI language model, I am an example of the technology used in ChatGPT. I can provide a personal opinion, but please keep in mind that I am designed to be socially unbiased and positive in nature.

ChatGPT is an impressive advancement in artificial intelligence and natural language processing. It has the ability to understand and generate human-like text, making it possible for users to have engaging and informative conversations with the AI. This technology has the potential to revolutionize various industries, including customer support, education, content creation, and more.

However, it's important to recognize that ChatGPT, like any AI technology, is not perfect. It may sometimes provide inaccurate or biased information due to the limitations of its training data or the complexity of human language. It's crucial to verify the facts and use critical thinking when interacting with AI systems like ChatGPT.

Overall, ChatGPT represents a significant step forward in the field of AI, and its potential applications are vast. As a responsible user, it's important to approach this technology with a critical and respectful mindset, recognizing its strengths and limitations.