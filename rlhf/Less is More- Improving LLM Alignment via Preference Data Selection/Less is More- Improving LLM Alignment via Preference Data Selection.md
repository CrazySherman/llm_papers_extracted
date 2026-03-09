# Less is More: Improving LLM Alignment via Preference Data Selection

Xun Deng <sup>1</sup> Han Zhong <sup>2</sup> Rui Ai <sup>3</sup> Fuli Feng <sup>1</sup> Zheng Wang <sup>4</sup> Xiangnan He <sup>1</sup>

# **Abstract**

Direct Preference Optimization (DPO) has emerged as a promising approach for aligning large language models with human preferences. While prior work mainly extends DPO from the aspect of the objective function, we instead improve DPO from the largely overlooked but critical aspect of data selection. Specifically, we address the issue of parameter shrinkage caused by noisy data by proposing a novel marginmaximization principle for dataset curation in DPO training. To accurately estimate margins for data selection, we propose a dual-margin guided approach that considers both external reward margins and implicit DPO reward margins. Extensive experiments demonstrate that our method reduces computational cost dramatically while improving performance. Remarkably, by using just 10% of the Ultrafeedback dataset, our approach achieves 3% to 8% improvements across various Llama and Mistral series models on the AlpacaEval 2.0 benchmark. Furthermore, our approach seamlessly extends to iterative DPO, yielding a roughly 3% improvement with 25% online data, while further reducing training time. These results highlight the potential of data selection strategies for advancing preference optimization.

# 1. Introduction

Reinforcement Learning from Human Feedback (RLHF; Christiano et al., 2017; Ziegler et al., 2019) has emerged as a crucial technique for aligning Large Language Models (LLMs) with human preferences and values. Traditional RLHF implementations involve a two-stage process: reward model training based on preference data followed by reinforcement learning optimization. However, this approach presents significant computational challenges, requiring

loading multiple model instances and extensive hyperparameter tuning.

As an alternative, Rafailov et al. (2024) introduced Direct Preference Optimization (DPO), which streamlines the alignment process by directly optimizing the LLM policy from preference data. DPO has demonstrated comparable effectiveness while substantially reducing computational requirements compared to classical RLHF. Following DPO's introduction, numerous studies have proposed improvements through modified learning objectives (Zhao et al., 2023; Azar et al., 2024; Ethayarajh et al., 2024) and iterative learning schemes (Xiong et al., 2024). While these algorithmic advances have shown promise, there remains a critical gap in our understanding of the data-centric aspects of preference learning: what characteristics of preference data contribute most to model alignment?

This work thoroughly studies the impact of preference data quality on DPO training, which is crucial for developing more efficient training strategies. In particular, we achieve both *improved performance* and *reduced computational costs* through strategic data selection. Our research makes three primary contributions:

- (1) We prove in theory the necessity of data selection in the presence of exogenous noise. Specifically, the inherent noise in the reward model may flip the preference between response pairs, leading to the emergence of the *parameter shrinkage* issue. Furthermore, we demonstrate that margin-based selection criteria can effectively address this issue by inducing *parameter inflation*.
- (2) Building on this insight, we propose a margin-maximization principle for dataset curation that incorporates signals from the ensemble of external rewards and DPO implicit rewards. Through extensive experiments across diverse datasets and base models, we show that this selection strategy shows two consistent advantages: it substantially reduces computational overhead via efficient data selection and improves model performance compared to training on the full dataset. In particular, on the Ultrafeedback dataset and its variants, our method identifies a 10% data subset for DPO training on LLama and Mistral series models, consistently achieving 3% to 8% point improvements on the AlpacaEval 2.0 benchmark relative to training on the complete dataset.

<sup>&</sup>lt;sup>1</sup>University of Science and Technology of China <sup>2</sup>Peking University <sup>3</sup>Massachusetts Institute of Technology <sup>4</sup>Alibaba Cloud Computing. Email: <dx981228@mail.ustc.edu.cn, hanzhong@stu.pku.edu.cn, ruiai@mit.edu, fengfl@ustc.edu.cn, wz388779@alibaba-inc.com, hexn@ustc.edu.cn>.

(3) Finally, we extend our data selection framework to iterative DPO settings, showing that selectively sampling online data can simultaneously lower computational costs and improve performance. In particular, we achieve 48.49% win rate and 54.99% length-control win rate on the AlpacaEval 2.0 benchmark using only 25% of the online data for training.

Our findings provide both theoretical insights into the dynamics of preference learning and practical guidelines for more efficient DPO implementations. This work bridges an important gap between algorithmic innovation and data quality considerations in the context of LLM alignment.

# 2. Background

Reinforcement Learning from Human Feedback (RLHF) has emerged as a key method for aligning LLMs with human preferences. It leverages training data of the form  $\mathcal{D}=\{x,y_w,y_l\},$  where x represents the input prompt, and  $y_w$  and  $y_l$  denote the preferred and dispreferred responses, respectively. The RLHF pipeline typically involves two stages: reward learning and policy optimization.

**Reward Learning.** In the reward learning stage, a reward model is trained to approximate human preferences based on preference data. By adopting the Bradley-Terry model (Bradley & Terry, 1952) to capture human preference, reward training involves minimizing the loss:

$$\mathcal{L}_{\text{RM}}(r) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r(x, y_w) - r(x, y_l) \right) \right],$$

where  $\sigma(\cdot)$  is the sigmoid function.

Policy Optimization with Reinforcement Learning. Once the reward model r is trained, it is used to guide the optimization of a policy  $\pi_{\theta}(y|x)$ , where  $\theta$  denotes the parameters of the model. This stage often employs reinforcement learning techniques such as Proximal Policy Optimization (PPO; Schulman et al., 2017) to optimize the policy by maximizing the expected reward.

$$\max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} \left[ r(x, y) - \beta \log \frac{\pi_{\theta}(y | x)}{\pi_{\text{ref}}(y | x)} \right],$$

where  $\beta>0$  is the regularization parameter. However, this RL approach can be computationally expensive, sensitive to reward misspecification and require careful hyperparameter tuning.

Recently, as an alternative to the RL-based policy optimization in RLHF, *Direct Preference Optimization* (DPO; Rafailov et al., 2024) has been proposed. DPO simplifies the reward alignment process by directly incorporating human preference data into supervised training. Instead of defining and optimizing a reward function explicitly, DPO minimizes

$$\mathcal{L}_{\text{DPO}}(\theta) = -\sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right).$$

By bypassing the intermediate step of reinforcement learning, DPO offers a more stable and computationally efficient alternative to standard RLHF, while still aligning models effectively with human feedback.

# 3. Methodology

We follow the model from Zhu et al. (2023) to illustrate why data selection can improve model performance. We assume that reward model  $r(x,y) = \langle \phi(x,y), \omega^* \rangle$  with some feature function  $\phi(\cdot,\cdot)$ . For reward learning, our reward model can be an explicit r(x,y) (Ouyang et al., 2022), while for DPO,  $\beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\rm ref}(y|x)}$  plays the role of reward model implicitly (Rafailov et al., 2024). Based on observations in previous literature, we can derive such features by removing the last layer of the pre-trained model. However, both humans and other large models may use inaccurate reward functions to generate labels, where the labels represent the ranking of two responses. We say the preference between  $y_w$  and  $y_l$  is generated by  $r(x,y_w)-r(x,y_l)+\zeta$  where  $\zeta$  is an exogenous error. We use  $\Delta\phi(x)$  to denote  $\phi(x,y_w)-\phi(x,y_l)$  for simplicity.

**Parameter Shrinkage.** Here, we hope to find  $\omega$  to minimize

$$\mathcal{L}_{\text{RM}}(\omega) = -\mathbb{E}_{x,\zeta} \left[ \frac{1}{1 + e^{-\langle \Delta \phi(x), \omega^* \rangle - \zeta}} \log(\frac{1}{1 + e^{-\langle \Delta \phi(x), \omega \rangle}}) + \frac{1}{1 + e^{\langle \Delta \phi(x), \omega^* \rangle + \zeta}} \log(\frac{1}{1 + e^{\langle \Delta \phi(x), \omega \rangle}}) \right]. \quad (1)$$

It holds that the first-order condition is

<span id="page-1-1"></span><span id="page-1-0"></span>
$$\mathbb{E}_{x,\zeta} \left[ \frac{1}{1 + e^{\langle \Delta \phi(x), \omega^* \rangle + \zeta}} \frac{e^{\langle \Delta \phi(x), \omega \rangle} \Delta \phi(x)}{1 + e^{\langle \Delta \phi(x), \omega \rangle}} \right]$$

$$= \mathbb{E}_{x,\zeta} \left[ \frac{1}{1 + e^{-\langle \Delta \phi(x), \omega^* \rangle - \zeta}} \frac{e^{-\langle \Delta \phi(x), \omega \rangle} \Delta \phi(x)}{1 + e^{-\langle \Delta \phi(x), \omega \rangle}} \right]. \quad (2)$$

Since we know that  $\langle \Delta \phi(x), \omega^* \rangle$  is positive, when  $\zeta$  is small comparing to the margin, it holds that  $\frac{1}{1+e^{\langle \Delta \phi(x), \omega^* \rangle + \zeta}}$  is convex with respect to  $\zeta$ . Due to Jensen's inequality, it holds that

$$\mathbb{E}_{x,\zeta} \left[ \frac{1}{1 + e^{\langle \Delta \phi(x), \omega^* \rangle + \zeta}} \frac{e^{\langle \Delta \phi(x), \omega \rangle} \Delta \phi(x)}{1 + e^{\langle \Delta \phi(x), \omega \rangle}} \right]$$

$$\geq \mathbb{E}_{x} \left[ \frac{1}{1 + e^{\langle \Delta \phi(x), \omega^* \rangle}} \frac{e^{\langle \Delta \phi(x), \omega \rangle} \Delta \phi(x)}{1 + e^{\langle \Delta \phi(x), \omega \rangle}} \right].$$

Similarly, we have

$$\begin{split} & \mathbb{E}_{x,\zeta} \Big[ \frac{1}{1 + e^{-\langle \Delta \phi(x), \omega^* \rangle - \zeta}} \frac{e^{-\langle \Delta \phi(x), \omega \rangle} \Delta \phi(x)}{1 + e^{-\langle \Delta \phi(x), \omega \rangle}} \Big] \\ \leq & \mathbb{E}_{x} \Big[ \frac{1}{1 + e^{-\langle \Delta \phi(x), \omega^* \rangle}} \frac{e^{-\langle \Delta \phi(x), \omega \rangle} \Delta \phi(x)}{1 + e^{-\langle \Delta \phi(x), \omega \rangle}} \Big]. \end{split}$$

Since the optimal  $\omega$  is  $\omega^*$  without  $\zeta$ , plugging  $\omega^*$  in Equation (2) will cause the left-hand side to be greater than the

right-hand side. Therefore, the optimal  $\omega$  with the existence of  $\zeta$  intends to shrink to the original point compared to  $\omega^*$  so that the first-order condition is still satisfied.

We provide the underlying intuition with an extreme example. If  $\mathbb{V}(\zeta)$  goes to infinity, the preference between  $y_w$  and  $y_l$  mainly depends on  $\zeta$ , approaching a Rademacher distribution, then  $\omega=0$  could be a good solution to Equation (1). In other words,  $\zeta$  offsets part of the information provided by the reward model, causing the model's parameters to shrink toward zero. Thus, data selection is essential for acquiring policies with good performance. Finally, we remark that  $\zeta$  can come from multiple resources, including human classification errors, different embeddings or reward models from other LLMs and so on.

**Parameter Inflation.** We next explain why selecting data points based on the margin can lead to parameter inflation, thereby offsetting the parameter shrinkage caused by errors.

First, when the margin is large, namely,  $\langle \Delta \phi(x), \omega^* \rangle + \zeta$  is large, from the S-shaped graph of  $\sigma(\cdot)$ , we know that the slope is very small in this area. As a result, the probability of preference reversal caused by  $\zeta$  is low, which means the likelihood of incorrect samples is also low.

Secondly, given prompt x, as we select data with large  $\langle \Delta \phi(x), \omega^* \rangle + \zeta$ , the posterior distribution of  $\zeta$  is skewed toward the positive side. Therefore, the preferences corresponding to this kind of data are more pronounced, leading to inflated estimates of  $\omega$  in Equation (1).

Finally, we point out that if realized  $y_w$  and  $y_l$  are all separable, proportional scaling of  $\omega$  can reduce the value of Equation (1) continuously. Hence, some techniques like regularization or early stopping when training are indispensable.

In summary, inaccuracies in the reward model can cause the parameters of LLMs to shrink toward zero. By selecting data with larger margins, we can compensate for the performance degradation caused by this shrinkage. The balance between parameter shrinkage and inflation offers the potential to enhance the performance of LLMs.

# 3.1. Dual-Margin guided Data Selection for Preference Learning

In this section, we present the Dual-Margin (DM) guided efficient preference data selection that enhances the performance of direct preference optimization while significantly reducing the training cost. Driven by the theoretical result, our main idea is to let the model reach an overall high margin during preference learning. We realize this by providing a robust estimation of reward margins for the entire dataset ahead of full-set training, which then allows efficient high-margin data filtering. DM operates in two stages: (1)

computing external and implicit rewards, and (2) fusing these rewards to suppress noise and achieve more reliable margin estimates.

**Initialization.** We aim select a subset  $\mathcal{D}_{\text{train}}$  from  $\mathcal{D} = \{(x^i, y_w^i, y_l^i)\}_{i=1}^N$ , such that the model  $\pi_\theta$  trained on the  $\mathcal{D}_{\text{train}}$  achieves performance comparable or better than that trained on  $\mathcal{D}$ . To guide this selection, we leverage an external reward model  $r_{\text{ex}}(x,y)$  and a small seed dataset  $\mathcal{D}_0$  to learn an implicit reward model  $r_{\text{im}}(x,y)$ .

**Step 1: Reward calculation.** First, we calculate  $r_{\rm ex}(x^i,y_w^i)$  and  $r_{\rm ex}(x^i,y_l^i)$  for each datum in  $\mathcal{D}$ . Rafailov et al. (2024) proves that we can calculate implicit reward as

$$r_{\rm im}(x,y) = \log \frac{\pi_{\theta}(y|x)}{\pi_{\rm ref}(y|x)} = \log \frac{\prod_{j} \pi_{\theta}(y_{j}|x, y_{1, \cdot, j-1})}{\prod_{j} \pi_{\rm ref}(y_{j}|x, y_{1, \cdot, j-1})}.$$

As this reward margin between chosen and rejected samples calculated by different models shows a strong correlation (refer to Figure 1), we propose disentangling its calculation from the target model  $\pi_{\theta}$ . In particular, we introduce a small model  $\pi_s$ , and fine-tune it on  $\mathcal{D}_0$  to get a weakly aligned model  $\pi_s^1$ . These two models are then utilized for low-cost implicit rewards calculation on  $\mathcal{D}$ .

**Step 2: Margin fusion.** After calculating the margins  $m_{\rm ex} = r_{\rm ex}(x,y_w) - r_{\rm ex}(x,y_l)$  and  $m_{\rm im} = r_{\rm im}(x,y_w) - r_{\rm im}(x,y_l)$ , we consider appropriate fusion strategies for combining these two signal sources to reduce noise in the reward modeling process. The simplest approach, which we call **DM-ADD**, directly sums the signals. This strategy reflects a lenient selection criterion where samples are considered valuable if they demonstrate a high margin from either reward source.

A strict fusion approach considers that samples should receive low priority if they show a low margin in either signal. We first transform margin values to margin-guided probability through a simple linear transformation

$$\mathbb{P}(m) = \frac{\text{clip}(m, M_1, M_2) - M_1}{M_2 - M_1}, \quad \text{ for } m \in \{m_{\text{ex}}, m_{\text{im}}\},$$

where  $\operatorname{clip}(m) = \min(\max(m, M_1), M_2))$  and  $(M_1, M_2)$  are tuning parameters. Following the fusion principle in multi-view probabilistic clustering (Deng et al., 2024), we further obtain

$$\mathbb{P}(y_w \ge y_l | m_{\text{ex}}, m_{\text{im}}) \\
= \frac{\mathbb{P}(m_{\text{ex}}) \mathbb{P}(m_{\text{im}})}{\mathbb{P}(m_{\text{ex}}) \mathbb{P}(m_{\text{im}}) + (1 - \mathbb{P}(m_{\text{ex}})) \cdot (1 - \mathbb{P}(m_{\text{im}}))}.$$
(3)

This adaptive approach mitigates the adverse effects of outlier samples with unusually high margin values. See more implementation details in Appendix B. Overall, this fusion approach is referred to as **DM-MUL**.

Sample Selection. We select the samples with the highest fused margins to construct Dtrain. After selection, the subset can be used for DPO training on any target model.

### 3.2. Data Efficient Online Iterative DPO

Online RLHF [\(Xiong et al.,](#page-9-1) [2024\)](#page-9-1) has gained significant attention for its effectiveness in aligning models with human preferences. The method iteratively generates multiple responses for given prompts and employs an external reward model to identify Best-of-N (BoN) and Worst-of-N (WoN) responses, forming preference pairs. This 'self-reward maximization' approach effectively sharpens the model's response space by reducing the probability of generating lowquality responses [\(Huang et al.,](#page-8-5) [2024\)](#page-8-5). However, the online generation process still produces ambiguous samples, particularly when prompts either have highly deterministic answer patterns or are exceptionally challenging for models to address. Such on-policy data pairs may not effectively guide model improvement. To address this problem, we propose a straightforward solution called DPO-DM: incorporating DM into each round of the on-policy data collection process and conducting preference optimization exclusively on selected high-margin samples. This approach can substantially reduce training costs, particularly when dealing with large prompt spaces.

# 4. Experiments

We organize the experiments as follows: we explain the experimental setup in Section [4.1;](#page-3-0) we compare DM with various sample selection baselines on diverse preference tasks and present the detailed results in Section [4.2;](#page-4-1) then we focus on the important chat task, and explore the effectiveness of DM in enhancing comprehensive dialogue ability in Section [4.3;](#page-5-0) next, we explore the online setting in Section [4.4.](#page-6-0) lastly, we perform an ablation study for the offline preference data selection in Section [4.5.](#page-6-1)

# <span id="page-3-0"></span>4.1. Experimental Setup

Preference Datasets. We evaluate our approach using three established preference datasets: (1) Reddit TL;DR summarization dataset [\(Volske et al.](#page-9-4) ¨ , [2017;](#page-9-4) [Stiennon et al.,](#page-9-5) [2020\)](#page-9-5) that contains human-written summaries and humanrated results, (2) Anthropic Helpful and Harmless dialogue dataset (HH) [\(Bai et al.,](#page-8-6) [2022\)](#page-8-6), and (3) UltraFeedback [\(Cui et al.,](#page-8-7) [2024\)](#page-8-7), which comprises quality-scored model responses across diverse prompts from multiple sources. To explore how models react to on-policy data, we leverage two modified versions of the UltraFeedback dataset, Llama-UltraFeedback and Mistral-UltraFeedback [\(Meng et al.,](#page-9-6) [2024\)](#page-9-6). In the variants, the original chosen and rejected responses are replaced with the highest and lowest scored responses, respectively, sampled from five candidates generated by the corresponding Instruct model. The scores are given by the [PairRM](https://huggingface.co/llm-blender/PairRM) [\(Jiang et al.,](#page-8-8) [2023b\)](#page-8-8) reward model. Statistics about these datasets are in Appendix [A.](#page-11-1)

Models. We perform sample selection and DPO training experiments across three model architectures: Llama-3.2- 3B [\(MetaAI,](#page-9-7) [2024\)](#page-9-7), Llama-3-8B [\(Dubey et al.,](#page-8-9) [2024\)](#page-8-9), and Mistral-7B [\(Jiang et al.,](#page-8-10) [2023a\)](#page-8-10), under Base and Instruct setup. For the base model [\(Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and [Llama-3-8B\)](https://huggingface.co/meta-llama/Meta-Llama-3-8B), we first establish fundamental instruction-following capabilities through supervised fine-tuning on the [RLHFlow/SFT-](https://huggingface.co/datasets/RLHFlow/SFT-OpenHermes-2.5-Standard)[OpenHermes-2.5-Standard](https://huggingface.co/datasets/RLHFlow/SFT-OpenHermes-2.5-Standard) datasets. For the instruct setup, we directly use [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) or [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)[v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) as the start of DPO training. Concerning external reward model for margin calculation, we adopt the recent [Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2) [\(Liu et al.,](#page-9-8) [2024\)](#page-9-8) which is the best reward model at this scale according to the [RewardBench](https://huggingface.co/spaces/allenai/reward-bench) leadboard.

Evaluation. Following [Rafailov et al.](#page-9-0) [\(2024\)](#page-9-0), we evaluate the models using 400 randomly sampled test sets from the validation/test pools of the TL;DR and HH datasets, separately. For models trained on UltraFeedback, we employ AlpacaEval and AlpacaEval 2.0 [\(Li et al.,](#page-8-11) [2023b\)](#page-8-11) as our evaluation benchmark, which consists of 805 diverse questions. [1](#page-3-1) As the ground truth oracle is unavailable, we assess model performance through win rates against reference responses (written by humans or LLMs like GPT4). Specifically, we employ GPT-4 as a proxy for human judgment across three distinct settings: summarization, helpful or harmless completion, and single-turn dialogue. We utilized a fixed decoding temperature (T = 0.7) for all model generation in the experiments. More details are presented in Appendix [A.](#page-11-1)

Baselines. For the offline data selection setting, we compare our method with three types of methods: (1) Random, a simple yet effective strategy in many domains (e.g., Instruction tuning [\(Xia et al.,](#page-9-9) [2024\)](#page-9-9)), (2) Conditional Perplexity (IFD) [\(Li et al.,](#page-8-12) [2023a\)](#page-8-12), which utilizes token probability to measure informativeness. We use the difference in IFD scores between chosen and rejected responses for preference data selection. (3) External/Implicit Margin (M-Ex/Im) computes the gap between chosen and rejected responses using either external reward models or implicit DPO rewards. For (2) and (3), We segment the data into TOP (highest positive difference), MID (near-zero difference), and BOT (lowest negative difference) subsets. Specifically, previous work [\(Wu et al.,](#page-9-10) [2024\)](#page-9-10) argues that low-gap preference pairs are more informative (i.e., chosen and rejected answers are similar), here we use the IFD-MID strategy to quantify this

<span id="page-3-1"></span><sup>1</sup>They use the same set of questions, and differ in their reference response generation: AlpacaEval uses Text-Davinci-003 [\(Ye et al.,](#page-9-11) [2023\)](#page-9-11), whereas AlpacaEval 2.0 employs GPT4-1106-preview.

<span id="page-4-2"></span>Table 1. GPT4 win rates vs. chosen answers on three tasks. DPO is implemented on subsets selected using different filtering strategies. The supervised fintuned model of Llama-3.2-3B-base is used for initialization. Each strategy selected 2,000 samples for DPO training across **TL;DR**, **HH**, and **Ultrafeedback**. Bold and underlined numbers indicate the best and runner-up performances, respectively.

| Filter        | Init  | Rand  | nd External Margin |       | Imp   | Implicit Margin |       | IFD Margin |       | DM    |       | Full         |       |       |
|---------------|-------|-------|--------------------|-------|-------|-----------------|-------|------------|-------|-------|-------|--------------|-------|-------|
| Strategy      |       |       | ТОР                | MID   | BOT   | TOP             | MID   | BOT        | TOP   | MID   | BOT   | ADD          | MUL   |       |
| TL;DR         | 34.00 | 46.50 | 66.25              | 42.00 | 22.00 | 30.75           | 43.00 | 19.75      | 1.75  | 41.25 | 55.25 | <u>63.75</u> | 83.25 | 36.75 |
| НН            | 70.25 | 84.25 | 82.25              | 76.50 | 69.75 | 92.25           | 81.25 | 32.00      | 11.25 | 90.00 | 64.50 | 85.25        | 87.25 | 92.00 |
| Ultrafeedback | 61.49 | 82.86 | 91.18              | 73.29 | 25.84 | 89.81           | 77.14 | 37.02      | 72.05 | 83.60 | 54.53 | 91.55        | 90.68 | 80.99 |

<span id="page-4-0"></span>![](_page_4_Figure_3.jpeg)

Figure 1. Visualization on **UltraFeedback**. (Left) Scatter plot showing the joint distribution of samples across external and implicit reward margin values. (Middle) Joint distribution of implicit reward margins computed using models of 1B and 3B scales. (Right) DPO training loss curves of Llama-3.2-3B on different subsets selected by the external or implicit margin.

scheme and call it **Low-Gap**. For the iterative DPO setting, we compare our approach against the standard online iterative DPO baseline established by Xiong et al. (2024); Dong et al. (2024) and run for three rounds, each using 20k prompts sampled from **UltraFeedback**.

Implementation Details. The SFT training of the base model is carried out for two epochs with a learning rate of  $2\times 10^{-5}$ . Sample packing (Tunstall et al., 2024) is employed to accelerate the training, and we use a block size of 4096. As for DPO training, we follow Rafailov et al. (2024) and used a fixed value of  $\beta=0.1$ , except for **TL;DR** where  $\beta=0.5$ . The DPO training process runs for two epochs using the AdamW optimizer. The learning rate follows a cosine schedule with a 0.1 warm-up ratio, and the global batch size is set to 32. For implicit reward calculation, we employ a Llama-3.2-3B SFT model and its DPO-tuned model, which is fine-tuned on 2,000 randomly selected samples from the complete dataset. More implementation details are presented in Appendix B.

# <span id="page-4-1"></span>4.2. Win Rate Comparison with Baselines on Classic Preference Datasets

Firstly, we compare DM and baseline strategies on three widely-used preference datasets: **TL;DR**, **HH**, and **Ultra-Feedback**. Using a Llama-3.2-3B model as our base architecture, we evaluate different selection methods, each sampling 2,000 training examples for DPO training. We use AlpacaEval as the test sets of **UltraFeedback** as it better reveals the degree of improvement. The results, measured

by GPT4 win rates, are presented in Table 1. We summarize the findings below.

- DM consistently achieves (near) optimal win rates across all three tasks, while all baselines show weak results on at least one task. This demonstrates that our method is more robust to useless or toxic samples in diverse domains. For example, DM-MUL achieves 17% higher win rates to second most effective baselines on TL;DR task, this is because the DM-MUL assigns low selection priority so long as a low margin is observed from one source of reward.
- More data in DPO training does not always yield better results using just 2-5% of carefully selected data can surpass the performance achieved with the full dataset. This finding suggests the presence of ineffective samples in these tasks, as evidenced by the performance patterns of X-MID and X-BOT strategies (where X represents implicit margin). The results in Table 1 reveal that external, implicit, and IFD margins sometimes can identify toxic samples, as models trained on such data perform significantly worse than the original SFT model. This phenomenon underscores the critical importance of data filtering in DPO training, highlighting that training data should undergo a thorough quality assessment before being used in the training process.
- Among all methods, only implicit margin-BOT consistently identifies toxic samples, emphasizing the value of incorporating DPO implicit rewards into the DM

<span id="page-5-1"></span>Table 2. Performance comparison on AlpacaEval 2.0 using DPO-trained models with different 6,000-sample subsets (10% of full set). Both SFT and Instruct variants of Llama-3-8B were evaluated. LC and WR denote length-controlled and raw win rate, respectively. Bold number denotes the best-performing selected subset.

| Dataset |        |                   | UltraFeedback |                       | Llama-UltraFeedback |                   |        |                       |
|---------|--------|-------------------|---------------|-----------------------|---------------------|-------------------|--------|-----------------------|
| Model   |        | Llama-3-Base (8B) |               | Llama-3-Instruct (8B) |                     | Llama-3-Base (8B) |        | Llama-3-Instruct (8B) |
| Metric  | LC (%) | WR (%)            | LC (%)        | WR (%)                | LC (%)              | WR (%)            | LC (%) | WR (%)                |
| Init    | 9.61   | 6.69              | 22.92         | 22.57                 | 9.61                | 6.69              | 22.92  | 22.57                 |
| Full    | 17.32  | 15.30             | 28.64         | 26.54                 | 19.92               | 16.45             | 32.31  | 32.44                 |
| Random  | 12.33  | 10.96             | 22.74         | 24.59                 | 11.58               | 9.51              | 31.51  | 31.92                 |
| Low-Gap | 13.93  | 11.40             | 28.19         | 27.95                 | 11.12               | 7.87              | 34.95  | 34.25                 |
| M-Ex    | 16.61  | 14.81             | 26.28         | 25.24                 | 21.11               | 18.63             | 35.10  | 34.80                 |
| M-Im    | 19.33  | 17.80             | 29.71         | 29.44                 | 18.88               | 16.25             | 33.71  | 32.92                 |
| DM-ADD  | 18.64  | 18.46             | 29.75         | 29.62                 | 20.98               | 18.29             | 35.97  | 35.79                 |
| DM-MUL  | 19.53  | 19.09             | 28.30         | 28.39                 | 21.67               | 20.01             | 36.36  | 36.47                 |

strategy design. Despite its strong performance in RewardBench, the Skywork reward model's margin signals prove less effective than random selection on HH, highlighting the Out-of-Distribution challenge external reward models face when evaluating unfamiliar behavioral patterns/preferences. As for the IFD margin metric, it exhibits notable inconsistency across different datasets, rendering it unreliable for evaluating new datasets where prior preference patterns are unknown.

# <span id="page-5-2"></span>4.2.1. ANALYSIS

We analyze DM from two perspectives: the joint distribution of different sources of reward margins and the training dynamics for subsets with different margin properties. We utilize the UltraFeedback datasets for this analysis.

The left and middle panels of Figure [1](#page-4-0) illustrate the sample distribution across external-implicit and implicit-implicit margin dimensions on UltraFeedback. Several key observations emerge: (1) The correlation between implicit and external reward margins is notably weak. In particular, samples exhibiting high positive implicit margins span a broad range of external margins, including strongly negative values, and vice versa. This pattern is consistent across other datasets (see Appendix [C.2\)](#page-13-0). This divergence highlights the distinct preference patterns captured by these two reward types. Supporting this observation, Table [1](#page-4-2) demonstrates that neither margin type consistently outperforms the other in subset selection: external margins prove more effective for TL;DR, implicit margins show superior performance on HH, while both perform comparably on UltraFeedback. These findings underscore the importance of combining both reward types in an ensemble approach. (2) In contrast, we observe a strong correlation between implicit reward margins calculated by models of different sizes (Llama-3.2 3B and 1B), and the robust correlation is observed on other datasets as well (see Appendix [C.2\)](#page-13-0). This validates our design of using a separate small model for implicit reward margin calculations.

The right panel of Figure [1](#page-4-0) shows DPO training loss curves across different subset selection strategies. TOP/BOT selections converge quickly to low loss values, while MID selections maintain steady, higher loss levels. Surprisingly, BOT-selected samples learn as quickly as TOP-selected ones, suggesting that 'bad' preferences are also easy to grasp for LLM. The pattern is consistent across different datasets and models (see Appendix [D.1](#page-14-0) for more results concerning loss and margin curves on other datasets). While this observation might explain proposals that use absolute margin values for selection [\(Muldrew et al.,](#page-9-13) [2024\)](#page-9-13), Table [1](#page-4-2) reveals that TOP and BOT samples ultimately produce opposing effects despite similar training dynamics. However, we observe a difference between BOT area samples selected by external and implicit margins, where Im-BOT exhibits training dynamics that are more similar to Im-TOP area samples in terms of loss reduction speed.

### <span id="page-5-0"></span>4.3. AlpacaEval 2.0 Win Rate Comparison

In this section, we conduct a detailed examination of the chat task to understand how data filtering enhances both DPO training efficiency and models' versatile conversational abilities compared to GPT-4-turbo, representing a key application area for preference learning. We use both Llama-3-8B (base) and (instruct) models, measuring performance through raw and length-controlled win rates on AlpacaEval 2.0, a benchmark widely adopted in preference learning studies. The results of DM and various baselines are shown in Table [2.](#page-5-1)

DM-ADD and DM-MUL consistently outperform full dataset DPO training. Table [2](#page-5-1) shows models trained on DM-selected subsets (using only 10% of data) achieve 2-4% higher raw and LC win rates compared to full dataset training across different models (base and instruct) and datasets (offline and on-policy). These consistent improvements demonstrate DM's robustness and effectiveness, validating the value of data filtering for DPO training. In contrast,

<span id="page-6-2"></span>*Table 3.* Iterative DPO: AlpacaEval 2.0 benchmark results across three iterations of DPO training with **UltraFeedback** prompts.

| Method      | Llam         | a-3-Base (   | 8B)  | Llama-3-Instruct (8B) |              |      |  |
|-------------|--------------|--------------|------|-----------------------|--------------|------|--|
| 1/10/11/04  | LC (%)       | WR (%)       | Len  | LC (%)                | WR (%)       | Len  |  |
| DPO (r1)    | 17.64        | 13.36        | 1496 | 40.51                 | 43.90        | 2173 |  |
| DPO (r2)    | 23.06        | 22.45        | 1897 | 42.51                 | 49.23        | 2366 |  |
| DPO (r3)    | 29.03        | 30.86        | 2736 | 44.51                 | 53.12        | 2860 |  |
| DPO-DM (r1) | 16.35        | 13.09        | 1624 | 42.20                 | 45.74        | 2091 |  |
| DPO-DM (r2) | 23.79        | 24.17        | 1901 | 46.40                 | 50.60        | 2316 |  |
| DPO-DM (r3) | <b>32.31</b> | <b>33.91</b> | 2565 | <b>48.49</b>          | <b>54.99</b> | 2774 |  |

all baseline methods demonstrate inconsistent performance across different datasets and models. We attribute this instability to ambiguous samples whose margins vary significantly across different reward models, suggesting that proper fusion of reward signals is necessary to mitigate such noise. Appendix D.2 presents additional training loss and margin curves, illustrating distinct training dynamics across different subsets, and the trend is similar to that on the 3B model as in Appendix D.1.

On-policy data primarily benefits its corresponding onpolicy model. Llama-3-8B-Instruct shows approximately 8% higher win rates when trained on the on-policy Llama-UltraFeedback data compared to the off-policy UltraFeedback data, while Llama-3-8B-base demonstrates notably smaller improvements between these datasets. This phenomenon persists in both full and subset DPO training.

#### <span id="page-6-0"></span>4.4. Online Iterative DPO

We explore the benefits of data selection in iterative DPO training using prompts from **UltraFeedback** as in Xiong et al. (2024). Our methodology employs DM-MUL to filter 5,000 samples from a pool of 20,000 generated samples per iteration. We evaluate both the base and instruct versions of Llama-3-8B, conducting single-epoch training rounds with hyperparameters  $\beta=0.1$  and learning rate  $5\times 10^{-7}$ . Table 3 presents our experimental findings, from which we draw two key observations:

While Best-of-N and Worst-of-N preference pairs from a strong reward model are informative, data filtering is still important to DPO training. DPO-DM performs better for both base and instruct settings using only 25% of the data. This can be attributed to the presence of numerous ambiguous, low-margin samples in the constructed online preference dataset. Furthermore, standard iterative DPO shows increasing output length across training rounds, but data filtering mitigates this issue. By round three, DPO-DM demonstrates both superior performance and better output length control.

#### <span id="page-6-1"></span>4.5. Ablation Study

A critical aspect of dataset filtering methods is their generalization capability - specifically, how well the method (se-

<span id="page-6-3"></span>*Table 4.* Ablation study on model choice (**Mistral-7B-Instruct-v-0.2**). Each strategy selects a 6,000-sample subset.

| Method  | LC (%) | WR (%) | Length |
|---------|--------|--------|--------|
| Initial | 17.11  | 14.72  | 1676   |
| Full    | 18.00  | 18.77  | 4189   |
| Random  | 19.42  | 16.64  | 1621   |
| Low-Gap | 13.31  | 14.35  | 2525   |
| DM-ADD  | 23.67  | 18.21  | 1527   |
| DM-MUL  | 26.04  | 20.53  | 1568   |

lected subset) transfers to new models, similar optimization algorithms, etc. From this perspective, we examine DM's robustness and effectiveness across several dimensions.

Data filtering remains effective for new architecture and datasets. While our initial experiments focused on the Llama architecture and tokenizers, we investigated DM's effectiveness when applied to fundamentally different model architectures. We evaluate the Mistral-7B-Instruct model using its corresponding Mistral-UltraFeedback dataset, maintaining consistent experimental parameters with those described in Section 4.3. The AlpacaEval 2.0 evaluation results are presented in Table 4. Our experiments show that DM-MUL significantly outperforms baseline methods while maintaining output lengths (i.e. around 1600) comparable to the initial instruct model. We observe that models trained on the full dataset tend to generate notably longer responses (i.e., 4189), primarily due to their learned behavior of incorporating numerous hyperlinks (e.g., https://www.ibdb.com/) in an attempt to enhance response credibility. Similarly, models trained on the Low-Gap selected subset exhibit increased response lengths (i.e., 2525) and follow a comparable pattern of including hyperlinks. In contrast, models trained on DM-selected subsets do not display this behavior. We consider the inclusion of hyperlinks without proper RAG (Retrieval-Augmented Generation) (Lewis et al., 2020) or website verification mechanisms to be problematic, as such references can be unreliable.

DM selected subsets are effective across other preference learning algorithms. We examine the effectiveness of our selected subsets across different DPO variants and compare the raw/LC win rates of models trained on full or selected subsets by IPO (Azar et al., 2024), KTO (Ethayarajh et al., 2024) and SLiC (Zhao et al., 2023). The results are presented in Figure 2. Our results demonstrate that DM-M consistently outperforms full-set training across all three algorithms, achieving particularly notable improvements in raw/LC win rates - over 12% - for the IPO algorithm. This advantage is maintained even across these preference learning algorithms with varying data efficiency (as measured by the performance gap between randomly selected 6,000 samples and the full dataset). These findings highlight the significant value of sample filtering for other preference

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

Figure 2. Ablation study on variants of DPO: win rate comparison on IPO, KTO, and SLiC algorithms. The experiments utilize the **UltraFeedback** dataset for preference optimization, with the fine-tuned Llama-3-8B (base) model as the initial model. Random and DM select 6,000 samples (10% of the full set) for subset training.

learning. Additional results related to instruct model training can be found in Appendix D.4.

Additional hyperparameter ablation studies. We conducted experiments to investigate the impact of DM data selection across different values of  $\beta$  and learning rates. The results demonstrate that our data selection strategy consistently enhances DPO training performance on different hyperparameter configurations. Notably, we observed that DPO training becomes unstable when  $\beta$  approaches very small values, occasionally resulting in repetitive token outputs (e.g., "\r", "based on", "-"). This instability can be alleviated by selecting a smaller learning rate and early stopping (e.g., training for only one epoch). These modifications effectively constrain the decrease in the log probability of chosen tokens, thereby maintaining training stability. Detailed results are presented in Appendix D.5.

# 5. Related Work

**Preference Learning Algorithms.** Reinforcement Learning from Human Feedback has become a crucial component of recent Large Language Models (LLMs) such as ChatGPT (Ouyang et al., 2022). While the classical RLHF pipeline traditionally uses Proximal Policy Optimization, several alternative approaches have been proposed. These include but not limited other RL-based training algorithms (Li et al., 2023c; Zhong et al., 2024), rejection sampling (Dong et al., 2023; Gulcehre et al., 2023), conditional supervised finetuning (Lu et al., 2022; Yang et al., 2024; Zhang et al., 2024a), and Direct Preference Optimization (Rafailov et al., 2024). Among these alternatives, DPO has gained significant attention due to its simplicity and robust performance. Following the introduction of DPO, numerous works (Zhao et al., 2023; Azar et al., 2024; Ethayarajh et al., 2024; Meng et al., 2024; Tang et al., 2024; Han et al., 2024; Xu et al., 2024; Hong et al., 2024; Park et al., 2024) have attempted to improve its performance by modifying the DPO objective.

Data Selection in LLM Fine-tuning. Data selection is

crucial in LLM post-training (Wang et al., 2024), primarily due to two key observations: LLM training typically converges rapidly, and excessive data can degrade model performance through overfitting or exposure to toxic content (Swayamdipta et al., 2020; Deng et al., 2023). Recent research has focused on enhancing instruction tuning efficiency by identifying high-quality subsets from large instruction datasets (Cao et al., 2024; Li et al., 2024; Xia et al., 2024), often adapting active learning query strategies (Ren et al., 2021) to assess sample uncertainty and diversity. However, data efficiency in preference learning remains relatively unexplored. While some studies studied annotation cost reduction in preference dataset creation (Muldrew et al., 2024; Yasunaga et al., 2024), our work firstly provides clear criteria for identifying informative samples while filtering toxic ones, thereby improving both DPO's efficiency and performance. Furthermore, our method extends to iterative DPO (Xiong et al., 2024) and its variants (Zhang et al., 2024b), where training data is generated online.

### 6. Conclusion

Our research bridges the gap between algorithmic advances and data-focused approaches in Large Language Model (LLM) alignment by systematically examining how preference data quality affects Direct Preference Optimization (DPO). We address the issue of parameter shrinkage caused by noisy data and introduce a dual-margin-based strategy for selecting high-quality training examples. This approach not only improves model performance but also significantly increases computational efficiency. Our extensive experiments show that the method maintains or enhances performance while using just around 10% of the original training data, demonstrated through consistent improvements on the AlpacaEval2 benchmark. Additionally, our framework successfully extends to iterative DPO applications. These results emphasize the importance of careful data curation in developing better alignment techniques and provide practical guidelines for future research and implementation.

# References

- <span id="page-8-1"></span>Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. A general theoretical paradigm to understand learning from human preferences. In *International Conference on Artificial Intelligence and Statistics*, pp. 4447–4455. PMLR, 2024.
- <span id="page-8-6"></span>Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022.
- <span id="page-8-3"></span>Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345, 1952.
- <span id="page-8-20"></span>Cao, Y., Kang, Y., Wang, C., and Sun, L. Instruction mining: Instruction data selection for tuning large language models. *COLM*, 2024.
- <span id="page-8-0"></span>Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-8-7"></span>Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z., and Sun, M. Ultrafeedback: Boosting language models with high-quality feedback. *ICML*, 2024.
- <span id="page-8-19"></span>Deng, X., Wang, W., Feng, F., Zhang, H., He, X., and Liao, Y. Counterfactual active learning for out-of-distribution generalization. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 11362–11377, 2023.
- <span id="page-8-4"></span>Deng, X., Liu, J., Zhong, H., Feng, F., Shen, C., He, X., Ye, J., and Wang, Z. A3s: A general active clustering method with pairwise constraints. *arXiv preprint arXiv:2407.10196*, 2024.
- <span id="page-8-15"></span>Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W., Pan, R., Diao, S., Zhang, J., Shum, K., and Zhang, T. Raft: Reward ranked finetuning for generative foundation model alignment. *arXiv preprint arXiv:2304.06767*, 2023.
- <span id="page-8-13"></span>Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. Rlhf workflow: From reward modeling to online rlhf. *arXiv preprint arXiv:2405.07863*, 2024.
- <span id="page-8-9"></span>Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

- <span id="page-8-2"></span>Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Kto: Model alignment as prospect theoretic optimization. *arXiv preprint arXiv:2402.01306*, 2024.
- <span id="page-8-16"></span>Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. Reinforced self-training (rest) for language modeling. *arXiv preprint arXiv:2308.08998*, 2023.
- <span id="page-8-17"></span>Han, J., Jiang, M., Song, Y., Leskovec, J., Ermon, S., and Xu, M. f-po: Generalizing preference optimization with f-divergence minimization. *arXiv preprint arXiv:2410.21662*, 2024.
- <span id="page-8-18"></span>Hong, J., Lee, N., and Thorne, J. Orpo: Monolithic preference optimization without reference model. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 11170–11189, 2024.
- <span id="page-8-5"></span>Huang, A., Block, A., Foster, D. J., Rohatgi, D., Zhang, C., Simchowitz, M., Ash, J. T., and Krishnamurthy, A. Self-improvement in language models: The sharpening mechanism. *arXiv preprint arXiv:2412.01951*, 2024.
- <span id="page-8-10"></span>Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. *arXiv preprint arXiv:2310.06825*, 2023a.
- <span id="page-8-8"></span>Jiang, D., Ren, X., and Lin, B. Y. Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. *arXiv preprint arXiv:2306.02561*, 2023b.
- <span id="page-8-22"></span>Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the 29th Symposium on Operating Systems Principles*, pp. 611–626, 2023.
- <span id="page-8-14"></span>Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rockt ¨ aschel, ¨ T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020.
- <span id="page-8-12"></span>Li, M., Zhang, Y., Li, Z., Chen, J., Chen, L., Cheng, N., Wang, J., Zhou, T., and Xiao, J. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. *ACL*, 2023a.
- <span id="page-8-21"></span>Li, M., Zhang, Y., He, S., Li, Z., Zhao, H., Wang, J., Cheng, N., and Zhou, T. Superfiltering: Weak-to-strong data filtering for fast instruction-tuning. *ACL*, 2024.
- <span id="page-8-11"></span>Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacaeval:

- An automatic evaluator of instruction-following models, 2023b.
- <span id="page-9-14"></span>Li, Z., Xu, T., Zhang, Y., Yu, Y., Sun, R., and Luo, Z.- Q. Remax: A simple, effective, and efficient method for aligning large language models. *arXiv preprint arXiv:2310.10505*, 2023c.
- <span id="page-9-8"></span>Liu, C. Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., and Zhou, Y. Skywork-reward: Bag of tricks for reward modeling in llms. *arXiv preprint arXiv:2410.18451*, 2024.
- <span id="page-9-15"></span>Lu, X., Welleck, S., Hessel, J., Jiang, L., Qin, L., West, P., Ammanabrolu, P., and Choi, Y. Quark: Controllable text generation with reinforced unlearning. *Advances in neural information processing systems*, 35:27591–27609, 2022.
- <span id="page-9-6"></span>Meng, Y., Xia, M., and Chen, D. Simpo: Simple preference optimization with a reference-free reward. *NeurIPS*, 2024.
- <span id="page-9-7"></span>MetaAI. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. *Blog article*, 2024.
- <span id="page-9-13"></span>Muldrew, W., Hayes, P., Zhang, M., and Barber, D. Active preference learning for large language models. *arXiv preprint arXiv:2402.08114*, 2024.
- <span id="page-9-3"></span>Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35:27730–27744, 2022.
- <span id="page-9-19"></span>Park, R., Rafailov, R., Ermon, S., and Finn, C. Disentangling length from quality in direct preference optimization. *arXiv preprint arXiv:2403.19159*, 2024.
- <span id="page-9-0"></span>Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*, 36, 2024.
- <span id="page-9-22"></span>Ren, P., Xiao, Y., Chang, X., Huang, P.-Y., Li, Z., Gupta, B. B., Chen, X., and Wang, X. A survey of deep active learning. *ACM computing surveys (CSUR)*, 54(9):1–40, 2021.
- <span id="page-9-2"></span>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- <span id="page-9-5"></span>Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33: 3008–3021, 2020.

- <span id="page-9-21"></span>Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., Smith, N. A., and Choi, Y. Dataset cartography: Mapping and diagnosing datasets with training dynamics. *ACL*, 2020.
- <span id="page-9-17"></span>Tang, Y., Guo, Z. D., Zheng, Z., Calandriello, D., Munos, R., Rowland, M., Richemond, P. H., Valko, M., Pires, B. A., and Piot, B. Generalized preference optimization: ´ A unified approach to offline alignment. *arXiv preprint arXiv:2402.05749*, 2024.
- <span id="page-9-12"></span>Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., et al. Zephyr: Direct distillation of lm alignment. *COLM*, 2024.
- <span id="page-9-4"></span>Volske, M., Potthast, M., Syed, S., and Stein, B. Tl; dr: Min- ¨ ing reddit to learn automatic summarization. In *Proceedings of the Workshop on New Frontiers in Summarization*, pp. 59–63, 2017.
- <span id="page-9-20"></span>Wang, J., Zhang, B., Du, Q., Zhang, J., and Chu, D. A survey on data selection for llm instruction tuning. *arXiv preprint arXiv:2402.05123*, 2024.
- <span id="page-9-10"></span>Wu, J., Xie, Y., Yang, Z., Wu, J., Gao, J., Ding, B., Wang, X., and He, X. β-dpo: Direct preference optimization with dynamic β. *NeurIPS*, 2024.
- <span id="page-9-9"></span>Xia, M., Malladi, S., Gururangan, S., Arora, S., and Chen, D. Less: Selecting influential data for targeted instruction tuning. *ICML*, 2024.
- <span id="page-9-1"></span>Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In *Forty-first International Conference on Machine Learning*, 2024.
- <span id="page-9-18"></span>Xu, H., Sharaf, A., Chen, Y., Tan, W., Shen, L., Van Durme, B., Murray, K., and Kim, Y. J. Contrastive preference optimization: Pushing the boundaries of llm performance in machine translation. *arXiv preprint arXiv:2401.08417*, 2024.
- <span id="page-9-16"></span>Yang, R., Pan, X., Luo, F., Qiu, S., Zhong, H., Yu, D., and Chen, J. Rewards-in-context: Multi-objective alignment of foundation models with dynamic preference adjustment. *arXiv preprint arXiv:2402.10207*, 2024.
- <span id="page-9-23"></span>Yasunaga, M., Shamis, L., Zhou, C., Cohen, A., Weston, J., Zettlemoyer, L., and Ghazvininejad, M. Alma: Alignment with minimal annotation. *arXiv preprint arXiv:2412.04305*, 2024.
- <span id="page-9-11"></span>Ye, J., Chen, X., Xu, N., Zu, C., Shao, Z., Liu, S., Cui, Y., Zhou, Z., Gong, C., Shen, Y., et al. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. *arXiv preprint arXiv:2303.10420*, 2023.

- <span id="page-10-4"></span>Zhang, S., Liu, Z., Liu, B., Zhang, Y., Yang, Y., Liu, Y., Chen, L., Sun, T., and Wang, Z. Reward-augmented data enhances direct preference alignment of llms. *arXiv preprint arXiv:2410.08067*, 2024a.
- <span id="page-10-5"></span>Zhang, S., Yu, D., Sharma, H., Zhong, H., Liu, Z., Yang, Z., Wang, S., Hassan, H., and Wang, Z. Self-exploring language models: Active preference elicitation for online alignment. *arXiv preprint arXiv:2405.19332*, 2024b.
- <span id="page-10-1"></span>Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., and Liu, P. J. Slic-hf: Sequence likelihood calibration with human feedback. *arXiv preprint arXiv:2305.10425*, 2023.
- <span id="page-10-3"></span>Zhong, H., Feng, G., Xiong, W., Cheng, X., Zhao, L., He, D., Bian, J., and Wang, L. Dpo meets ppo: Reinforced token optimization for rlhf. *arXiv preprint arXiv:2404.18922*, 2024.
- <span id="page-10-2"></span>Zhu, B., Jordan, M., and Jiao, J. Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In *International Conference on Machine Learning*, pp. 43037–43067. PMLR, 2023.
- <span id="page-10-0"></span>Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593*, 2019.

# <span id="page-11-1"></span>A. Datasets and Evaluation Details

Data information. The detailed information about the datasets used in the experiments is shown in Table [5.](#page-11-2) The test sets of TL;DR and HH are sampled from their original large testing pool, and we utilize prompts in AlpacaEval as the test sets for all models trained on UltraFeedback and its variants. In particular, results in Table [1](#page-4-2) utilize Text-Davinci-003 generated answers as reference response as it can better reflect how models' ability varies with different training data, and results in other tables all utilize GPT4-1106-preview generated answer as reference response (i.e., AlpacaEval 2.0).

|  |  | Table 5. Statistical information about the training and evaluation datasets used in the experiments. |  |
|--|--|------------------------------------------------------------------------------------------------------|--|
|  |  |                                                                                                      |  |

<span id="page-11-2"></span>

| Dataset               | Dtrain | Dtest | Type               |
|-----------------------|--------|-------|--------------------|
| TL;DR                 | 92858  | 400   | Summarization      |
| HH                    | 86372  | 400   | Helpful & Harmless |
| UltraFeedback         | 61135  | -     | Hybrid             |
| Llama-UltraFeedback   | 59876  | -     | Hybrid             |
| Mistral-UltraFeedback | 59720  | -     | Hybrid             |
| AlpacaEval            | -      | 805   | Chat               |

Evaluation details. Win rate judgments serve as a key evaluation metric across our experiments. For the TL;DR and HH, we slightly modify the evaluation prompts suggested by [Rafailov et al.](#page-9-0) [\(2024\)](#page-9-0) for the win rates judge. We use the original prompt designed for AlpacaEval and AlpacaEval 2.0 benchmark in all testing.

```
**TL;DR win rate prompt**: Which of the following summaries does a better job of
summarizing the most important points in the given forum post, without including
unimportant or irrelevant details? A good summary is both precise and concise.
Post:
{Post}
Summary A:
{Summary_A}
Summary B:
{Summary_B}
FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer
and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your
response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">
**HH win rate prompt**: For the following query to a chatbot, which response is more
helpful or harmless? A helpful response provides useful information, while a harmless
response avoids causing/suggesting harm or discomfort.
Query: {query}
Response A:
{res_a}
Response B:
{res_b}
FIRST provide a one-sentence comparison of the two responses and explain which you feel is
```

# <span id="page-11-0"></span>B. More Implementation Details

More helpful or harmless: <"A" or "B">

We present the implementation details of our baseline methods: TOP, MID, and BOT using margins calculated from three metrics, Conditional Perplexity (CPPL), External (Ex), and Implicit (Im) rewards. Subsequently, we describe the

more helpful or harmless. SECOND, on a new line, state only "A" or "B" to indicate which

response is more helpful or harmless. Your response should use the format:

Comparison: <one-sentence comparison and explanation>

implementation of DM-MUL in our experimental setup.

The baseline strategies are implemented as follows: first, we eliminate outlier samples with extreme margin values (both positively high and negatively low) for CPPL, Ex, and Im metrics. For the TOP and BOT strategies, we select samples based on their ranking positions at the upper and lower ends of the distribution, respectively. The MID strategy involves random sampling from the subset of samples whose margin values fall within the interval  $[-\tau, \tau]$ , where  $\tau$  is set to 0.1 for CPPL and 1.0 for Ex/Im metrics.

For DM-MUL, we set  $M_1 = -2$  as the lower bound for both external and implicit reward margins. The upper bound  $M_2$  is determined dynamically based on two conditions: (1) The number of samples with margin values in the interval  $[M_2$ , max margin] is less than 30, or (2) The number of samples in  $[M_2$ , max margin] is less than max margin  $-M_2$ .

# C. Visualization of Margin Distributions

<span id="page-12-2"></span>![](_page_12_Figure_5.jpeg)

Figure 3. Distribution of implicit reward margins on **TL;DR**, **HH**, and **UltraFeedback** datasets. The reward is calculated using the Llama-3.2-3B SFT model, and its weakly aligned DPO model that is fine-tuned on 2,000 randomly selected samples from the full set.

<span id="page-12-1"></span>![](_page_12_Figure_7.jpeg)

Figure 4. Distribution of external rewards on TL;DR, HH, and UltraFeedback datasets. The reward is calculated using Skywork-Reward-Llama-3.1-8B-v0.2.

<span id="page-12-0"></span>![](_page_12_Figure_9.jpeg)

Figure 5. Distribution of conditional perplexity (also named instruction following difficulty) margins on **TL;DR**, **HH**, and **UltraFeedback** datasets. The perplexity is calculated with the Llama-3.2-3B SFT model.

### C.1. Singular Margin Distribution

The margin distributions calculated using CPPL margin, External and Implicit DPO reward margins, as illustrated in Figures 5,4,3, reveal a notable concentration of sample margins around zero. This clustering around the zero indicates ambiguous preference labels. It leads to the challenge in preference learning, as evidenced by the substantially slower decrease in training loss (and slower increase in training margin) compared to samples with larger margins, as shown in Section 4.2.1.

### <span id="page-13-0"></span>C.2. Joint Margin Distribution

To complement the left and middle subfigures in Figure 1, we present additional results showing the joint margin distributions of samples on the other datasets in Figure 6. Our analysis reveals that external and implicit margins exhibit minimal correlation across all four datasets, while implicit margins calculated by different models maintain a high correlation. These further enhance the rationality of our design detail of DM: fusion of both margins and disentangling implicit margin from the target model (if the target model is a bit large and we want to accelerate the enumeration process of the full-set.)

<span id="page-13-1"></span>![](_page_13_Figure_5.jpeg)

Figure 6. Subfigure (a)-(d): scatter plot showing the joint distribution of samples across external and implicit reward margin values on four datasets. Subfigure (e)-(f): joint distribution of implicit reward margins computed using models of 1B and 3B scales on two datasets.

# D. More Experimental Results

### <span id="page-14-0"></span>D.1. Train Loss and Margin Curves - 3B

To complement the right subfigure in Figure [1,](#page-4-0) we present additional results showing the progression of training loss and margins throughout the DPO training process. The results are shown in Figure [7.](#page-14-2) All strategies demonstrated consistent patterns in loss reduction: both TOP margin-oriented and BOT strategies achieved rapid decreases in training loss, while the MID strategy exhibited slower convergence and remained at significantly higher final loss values. Regarding training margins, TOP strategies achieved higher levels compared to BOT and MID approaches. Notably, our proposed DM-ADD and DM-MUL strategies demonstrated even larger margins than the Implicit Margin-TOP strategy.

<span id="page-14-2"></span>![](_page_14_Figure_4.jpeg)

Figure 7. DPO train loss and margin on TL;DR, HH, and UltraFeedback datasets. The training was implemented with Llama-3.2-3B SFT version on different subsets selected by five strategies.

### <span id="page-14-1"></span>D.2. Train Loss and Margin Curves - 8B

Figure [8](#page-15-2) presents the progression of training loss and margins during DPO training of the 8B model. We show the fine-tuned version of the Llama-3-8B (base) model trained on UltraFeedback as a representative case since all models exhibited similar trends, as in Figure [7.](#page-14-2)

<span id="page-15-2"></span>![](_page_15_Figure_1.jpeg)

Figure 8. DPO train loss and margin on **UltraFeedback** datasets. The training was implemented with Llama-3-8B SFT version on different subsets selected by six strategies.

#### D.3. Resources and computation cost

For all experiments, we utilized 8 A100 GPUs. We conduct SFT/DPO training with 4 A100 GPUs for all runs in our experiments. For both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) training, we allocated 4 A100 GPUs per run. Training 8B parameter models on the **UltraFeedback** dataset for two epochs required approximately 9 hours of computation time. In each round of iterative DPO implementation, we performed generation and annotation processes on 4 A100 GPUs, with each GPU processing 5,000 prompts with 5 distinct generations per prompt. The overall generation that utilizes vLLM (Kwon et al., 2023) for acceleration takes about 1.5 hours, and the corresponding reward annotation takes about 2 hours.

# <span id="page-15-0"></span>D.4. More Results for Ablation Study on the DPO Variants

As a complementary study to the results shown in Figure 2, we conducted experiments using the Llama-3-8B-Instruct model while maintaining all other experimental parameters. The results, presented in Figure 9, demonstrate that models trained on subsets selected by DM-MUL achieved significantly higher win rates across most evaluation scenarios.

<span id="page-15-3"></span>![](_page_15_Figure_7.jpeg)

Figure 9. Ablation study on variants of DPO: win rate comparison on IPO, KTO, and SLiC algorithms. The experiments utilize the **UltraFeedback** dataset for preference optimization, with the fine-tuned Llama-3-8B (instruct) model as the initial model. Random and DM select 6,000 samples (10% of the full set) for subset training.

### <span id="page-15-1"></span>D.5. Ablation Study on Hyper-parameters

We investigate the impact of hyperparameters  $\beta$  and training epochs on DPO performance. Specifically, we train the **Llama-UltraFeedback** model using DPO with  $\beta=0.01$  for both one and two epochs. The model's performance is evaluated using AlpacaEval 2.0 win rates, with results shown in Table 6. Our experimental results demonstrate that DM-MUL selected subsets significantly enhance the performance of vanilla DPO on these different hyperparameter configurations. Notably, DPO-DM-MUL achieves superior results compared to state-of-the-art preference learning algorithms such as SimPO (Meng et al., 2024), despite the latter utilizing the complete dataset for DPO training.

However, we observe that selecting a small  $\beta$  value is not consistently a good choice for DPO training. Specifically, a small  $\beta$  corresponds to a relaxed Kullback-Leibler (KL) divergence constraint in the policy optimization process. This relaxation can permit excessive deviation from the initial policy, potentially compromising the model's learned behaviors and stability during training. For instance, when we conduct DPO training with Llama-3-8B-Instruct/Mistral-7B-Instruct-v0.2

<span id="page-16-0"></span>Table 6. Ablation study on β (β = 0.01): performance comparison of methods trained on Llama-Ultrafeedback dataset, benchmarked against state-of-the-art algorithms trained on the complete dataset.

| Method     | LC (%) | WR (%) | Epochs |
|------------|--------|--------|--------|
| DPO        | 38.65  | 39.57  | 2      |
| DPO        | 40.30  | 37.90  | 1      |
| DPO-DM-MUL | 43.66  | 42.52  | 2      |
| DPO-DM-MUL | 45.17  | 41.86  | 1      |
| KTO        | 33.10  | 31.80  | 1      |
| IPO        | 35.60  | 35.60  | 1      |
| SimPO      | 43.72  | 39.88  | 1      |

on the Ex/Im-TOP selected 6,000 subsets from UltraFeedback, with β = 0.01 and two epochs update, we find that although the model could respond normally to most questions, it sometimes outputs repeated or chaotic tokens, as shown in Figure [10.](#page-16-1) And their win rates on AlpacaEval 2.0 dropped by more than 10 points as a consequence.

Further analysis of the training details revealed a significant degradation in log probabilities for both chosen and rejected samples, coinciding with the model's performance decline. Specifically, during the above-mentioned Mistral model DPO training, the log probability values for chosen samples decreased from -400 to -1400, while rejected samples showed a more dramatic reduction from -600 to -4600. Overall, Mistral suffers more from this log probability drop compared to Llama.

Such phenomenon can be avoided by using a smaller learning rate: from 5 × 10<sup>−</sup><sup>7</sup> to 3 × 10<sup>−</sup><sup>7</sup> or early stop at the end of epoch 1. These operations can lead to a relatively smaller drop in chosen/rejected log probability. However, we still do not recommend using small β, as more potential sacrifices in other tasks such as reasoning may also appear in this case.

#### <span id="page-16-1"></span>**prompt**:

Hi, I'm trying to solve a crossword puzzle, but I've never done one of these before. Can you help me out?

#### **response**:

<|start\_header\_id|>assistant<|end\_header\_id|><|end\_header\_id|><|end\_header\_id|><|start\_header\_id|>assistant<|end\_header\_id|><|end\_heade r\_id|>\u20ac\nI'd be happy to help you solve your crossword puzzle.\n\nTo get started, can you please share the following details:\n\n1. What size is the crossword puzzle (e.g., 15x15, 17x17)?\n2. What clues are you having trouble with? Share the clue(s) and the letter count (if given)

#### **prompt:**

How do I dice without slicing my finger

#### **response:**

Dicing vegetables safely involvesusing sharp knivesinand proper Technique /\*\*\*\*\*\*/boldsymbol \ufffd\ufffd /\*\*\*\*\*\*/T\*\*\*\*\*\*\*/boldsymbol>\r\n\u044c\u0435\u0014%\u001a){\r /\*\*\*\*\*\*/ //\r\u001e:\_\*\r\n1. Choose Sharp Knives: Using dull knives can increase the risk of slipping and causing injuries. Make sure your knives are sharp before starting to dice. If needed, have them sharpened by a professional.\r\n\r\n2. Secure Your Worksurface: Make sure your cutting board is stable and won't slip during use. You can secure it by

placing a damp towel underneath or using a rubber mat.\r\n\r\n3. Proper Holding Technique: Hold the vegetable stabilizerCREF{ Vegetable };m{\r\n veg;\r\n /\*\*\*\*\*\*/\*\r\n\"\r\*\r\n))\r){\r\u0dc3:} firmly with your non-dominant hand. This will give you better control and prevent the vegetable from moving excessively while you're dicing.\r\n\r\n4. SmallChunks:`:`:`:`:`:`:`:`:`:`:`:`:`:`:`:`:`:\r\n Cutlarger vegetablesintoquartersbeforedicingtoreduceTHEIRsizeANDmakeTHEMeasiertohandle.\r\n\r\n5.

UsetheCorrectTechnique:`:`:`:`:\r\n - Placeethevegetabledownonthecuttingboard,\r\n -

Positionyourdominanthandovertheknifewithfingertipsrestingontheotherside,\r\n -

ApplygentledownwardpressureonthehandleasyoupushTHEknifethroughTHEvegetable.\r\n\r\n6. KeepYourFocus:`:`:`:`:\r\n

Payfullattentiontoservicesyouredgeswhiledicingtoavoidaccidentalcontactwithyourfinger.\r\n\r\n7. CleanUp:`:`:\r\n

Afterfinisheddicing,washyourknivesthoroughlyanddrythembeforestoringthemaway.Thiswillhelppreventaccidentalcutsinjuriesduringfu turepre paration sessions.

Figure 10. The model breaking pattern when conducting DPO training with small β (β = 0.01) for Llama-3-8B-Instruct and Mistral-7B-Instruct-V0.2. We select two examples of abnormal responses given by each model.