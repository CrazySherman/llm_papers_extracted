# The Perfect Blend: Redefining RLHF with Mixture of Judges

Tengyu Xu<sup>1,†</sup>, Eryk Helenowski<sup>1,†</sup>, Karthik Abinav Sankararaman<sup>1,†</sup>, Di Jin<sup>1,†</sup>, Kaiyan Peng<sup>1</sup>, Eric Han<sup>1</sup>, Shaoliang Nie<sup>1</sup>, Chen Zhu<sup>1</sup>, Hejia Zhang<sup>1</sup>, Wenxuan Zhou<sup>1</sup>, Zhouhao Zeng<sup>1</sup>, Yun He<sup>1</sup>, Karishma Mandyam<sup>1</sup>, Arya Talabzadeh<sup>1</sup>, Madian Khabsa<sup>1</sup>, Gabriel Cohen<sup>1</sup>, Yuandong Tian<sup>2</sup>, Hao Ma<sup>1</sup>, Sinong Wang<sup>1</sup>, Han Fang<sup>1</sup>

<sup>1</sup>Meta GenAI, <sup>2</sup>FAIR, <sup>†</sup>Equal contributions

Reinforcement learning from human feedback (RLHF) has become the leading approach for fine-tuning large language models (LLM). However, RLHF has limitations in multi-task learning (MTL) due to challenges of reward hacking and extreme multi-objective optimization (i.e., trade-off of multiple and/or sometimes conflicting objectives). Applying RLHF for MTL currently requires careful tuning of the weights for reward model and data combinations. This is often done via human intuition and does not generalize. In this work, we introduce a novel post-training paradigm which we called Constrained Generative Policy Optimization (CGPO). The core of CGPO is Mixture of Judges (MoJ) with cost-efficient constrained policy optimization with stratification, which can identify the perfect blend in RLHF in a principled manner. It shows strong empirical results with theoretical guarantees, does not require extensive hyper-parameter tuning, and is plug-and-play in common post-training pipelines. Together, this can detect and mitigate reward hacking behaviors while reaching a pareto-optimal point across an extremely large number of objectives.

Our results show that CGPO consistently outperforms other commonly used SoTA RLHF algorithms (such as PPO and DPO) on a wide range of tasks – general chat, STEM questions, instruction following, math, coding and knowledge. In particular, CGPO improves over PPO by 7.4% in AlpacaEval-2 (general chat), 12.5% in Arena-Hard (STEM & reasoning), 2% in IFEval (Instruction Following), 2% in both MATH and GSM8K (Math & reasoning), 5% in HumanEval (Coding), and 2% in the ARC challenge (Knowledge). We also observe that PPO is susceptible to severe reward hacking behaviors (it exhibits severe regression in popular coding benchmarks) which can be addressed by CGPO. CGPO represents a breakthrough in RLHF, simultaneously addressing reward-hacking and extreme multi-objective optimization, and thereby advancing the state-of-the-art in aligning general-purpose LLMs.

Date: September 18, 2024

Correspondence: Tengyu Xu at tengyuxu@meta.com

![](_page_0_Picture_7.jpeg)

#### 1 Introduction

The emergence of general-purpose Large Language Models (LLMs) has significantly transformed the landscape of natural language processing, demonstrating exceptional capabilities across various expert-level domains (Achiam et al., 2023; Brown et al., 2020; Touvron et al., 2023; Anthropic, 2023; Team et al., 2023; Meta, 2024; Tunstall et al., 2023; Zhu et al., 2023). These models are characterized by their extensive parameterization, enabling them to handle a wide array of tasks using a unified parameter set (Zhao et al., 2018; Liu et al., 2019b,a). Central to this versatility is multi-task learning (MTL) (Caruana, 1997; Crawshaw, 2020), a strategy that involves training a single model on multiple tasks simultaneously. This approach fosters the development of shared representations, which enhances the model's ability to generalize better than those trained on isolated tasks. Although prior studies on MTL have concentrated on the integration and processing of multi-task data during both pre-training and fine-tuning stages (Raffel et al., 2020; Liu et al., 2023; Aghajanyan et al., 2021; Aribandi et al., 2021), the application of the primary LLM alignment method, Reinforcement Learning with Human Preference (RLHF) (Ouyang et al., 2022; Ziegler et al., 2019; Zheng et al., 2023b), has not been thoroughly explored within the MTL context. In previous studies, the implementation of RLHF for multi-task post-training has typically involved a linear combination of multiple reward models within the standard

RLHF framework (Ramamurthy et al., 2022; Glaese et al., 2022; Yuan et al., 2023; Bakker et al., 2022; Wu et al., 2024; Li et al., 2020). Each reward model is crafted using preference data to mirror the distinct alignment preferences of different tasks. Researchers often experiment with various reward weightings to identify a Pareto front that depicts the optimal performance of the LLM across diverse tasks (Rame et al., 2024). However, this approach is limited by two significant challenges:

**Vulnerability to Reward Hacking:** The optimization of a preference-based reward model is susceptible to reward hacking, as the reward model is an imperfect proxy of human preferences (Gao et al., 2023; Jin et al., 2023; Skalse et al., 2022). Studies indicate that excessive optimization of a reward model can lead to misalignment with actual human preferences (Gao et al., 2023; Moskovitz et al., 2023; Stiennon et al., 2020; Rafailov et al., 2024a). This issue becomes more pronounced in a multi-task setting, where each reward model may have its own unique flaws. Implementing a uniform early stopping point in the RLHF optimization process to minimize reward hacking effects is impractical and can lead to degraded performance across tasks (Moskovitz et al., 2023). This highlights the need for a more tailored approach to compensate for the weaknesses of each reward model and to manage the optimization of reward models for each task in complex, multi-task environments.

**Contradictory Goals:** Different tasks often have conflicting objectives (Rame et al., 2024). Even if the prompt spaces for these tasks do not overlap, using a linear combination of reward models can lead to compromises in goal metrics. For example, the typical strategy of LLM post-training involves maximizing the helpfulness reward for safe prompts and maximizing the harmfulness reward for unsafe prompts (Bai et al., 2022). Although achieving global optimality for both tasks is possible if the LLM's capacity is sufficiently large (Iyer et al., 2022), employing a linear combination of helpfulness and harmfulness rewards inevitably results in reduced gains for both metrics. This occurs because each task partially sacrifices its own RLHF optimization progress to accommodate a contradictory metric, thereby diminishing the effectiveness of both.

To address these challenges, we developed an innovative framework called Constrained Generative Policy Optimization (CGPO). In response to the issue of reward hacking in RLHF, we introduce two types of judges: rule-based and LLM-based. These judges collaborate to identify any reward hacking patterns during the LLM's online generation phase. Based on their evaluations, we implement a constrained RLHF method to update the LLM model. This method is designed to maximize the likelihood of generating outputs that adhere to all constraints and achieve high reward values, while minimizing outputs that breach constraints and have low reward values. To support the constrained policy optimization update in the large-scale LLM setting, which is complicated even in traditional small-scale RL scenarios, we have developed three new primary-type constraint RLHF optimizers. These optimizers are designed to operate independently of the dual-variable update, which is often a critical component in conventional primal-dual constrained RL algorithms. This independence simplifies the optimizers and enhances their scalability, making them more effective for managing large-scale LLM post-training.

To effectively optimizing objectives of various tasks, which may be contradictory, we propose a novel design in CGPO for managing multi-task post-training. In this design, prompts are segregated by task, and a customized policy optimization strategy is applied to each set of prompts. This strategy includes a tailored MoJs, reward model, and hyperparameter setup for the constrained RLHF optimizer. By optimizing each task independently, our approach avoids compromises due to conflicting goals from other tasks, a common issue in previous works that used a linear combined reward model. Furthermore, our design addresses the reward hacking issue and optimizes objectives for each task in a fine-grained manner, resulting in a better Pareto frontier than previous methods that enforced uniform treatment across all tasks. See Figure 1 for an overview of our CGPO pipeline.

We summarize our contributions as follows:

- We have developed a new strategy to address the issues of reward hacking in multi-task LLM post-tuning through an innovative primal-type constrained RL method. To implement this method, we have introduced three new constrained RLHF optimizers: Calibrated-Regularized Policy Gradient (CRPG), Constrained Online Direct Preference Optimization (CODPO), and Calibrated-Regularized Reward Ranking Finetuning (CRRAFT). All proposed methods are scalable and easy to implement.
- To support the implementation of the constrained RL method in CGPO, we have developed two types of judges: the rule-based judge and the LLM-based judge. These judges are designed to effectively assess whether an LLM generation violates constraints in a broad spectrum of NLP tasks.
- We have introduced a new multi-objective RLHF treatment strategy within CGPO, where each task is managed

individually with a customized optimization setting, including reward models, mixture of judges, and optimizer hyperparameters. This pioneering design, the first in the multi-task RLHF field, significantly enhances the Pareto frontier across multiple metrics in the multi-task setting.

- We demonstrate the effectiveness of CGPO in a challenging multi-task post-training environment with five tasks: general chat, instruction following, math and coding reasoning, engagement intent, and safety, despite potentially contradictory goals across tasks. Notably, by primarily utilizing open-source data and the Llama3.0 70b pre-trained model, our research demonstrates that, in comparison to the baseline RLHF methods such as PPO Schulman et al. (2017) and DPO Rafailov et al. (2024b), our approach—when combined with the CRPG and CRRAFT optimizers—consistently outperforms these baselines across all benchmarks and tasks. Specifically
  - CRPG optimizers achieve the highest performance in terms of MATH, GSM8K, HumanEval, MBPP, ARC Challenge, and false refusal ratio. CRRAFT optimizers achieve the highest performance in AlpacaEval-2, Arena-Hard, and TruthfulQA.
  - PPO experiences a significant drop in the 0-shot coding benchmarks (HumanEval and MBPP) after exceeding certain training steps, indicating the occurrence of severe reward hacking issues. In contrast, CGPO not only avoids such regression but also consistently improves those benchmarks during training, demonstrating the extraordinary capability of MoJs in preventing reward hacking issues.

![](_page_2_Figure_4.jpeg)

**Figure 1** Overview of CGPO pipeline. In CGPO, a customized MoJs is applied to each task to evaluate model generations, and the model is updated through our proposed constrained RL algorithm.

#### 2 Preliminaries

In the RLHF finetuing phase, we typically formulate a Markov Decision Process (MDP) as follows: each prompt is considered as the state s, and the entire response is the action  $a = [a_0, a_1, \dots, a_{T-1}]$ , where  $a_i \in A$  represents the token at position i and A is the vocabulary set. An LLM policy is defined as  $\pi_w(a_t|a_{t-1}, a_{t-2}, \dots, a_0, s)$ , which represents a distribution over A at time step t, conditioned on all previous response history before t and prompt:  $\{a_{t-1}, a_{t-2}, \dots, a_0, s\}$ .

#### 2.1 Supervised Finetuing

RLHF starts by finetuing a pre-trained LLM using supervised learning on high-quality dataset relevant to the downstream target task(s) (such as dialogue, summarization, reasoning, etc.) to obtain  $\pi_{SFT}$ .

#### 2.2 Reward Model Training

After the supervised fine-tuning stage, we need to develop a reward model to assess the quality of an LLM's output. This will enable us to utilize exploration-based online RL alignment method. We typically use the pairwise preference reward model (Stiennon et al., 2020). In this model, we assume that human preference between a pair of responses  $(a_p, a_n)$ , originating from the same prompt s, is determined by a latent reward  $r_{pair}^*(s, a)$ . The Bradley-Terry (BT) model (Bradley and Terry, 1952; Ouyang et al., 2022; Bai et al., 2022; Touvron et al., 2023; Meta, 2024), a well-established reward-based preference model, defines the human preference distribution  $p_{pair}^*$  using the following formulation:

$$p_{pair}^{*}(a_{p} > a_{n}|s) = \sigma(r_{pair}^{*}(s, a_{p}) - r_{pair}^{*}(s, a_{n})), \tag{1}$$

where  $\sigma$  denotes the logistic function. In practice, we can learn a parameterized reward model  $r_{\phi}(s, a)$  as a surrogate for  $r_{pair}^*(s, a)$ . Given a pre-collected preference-pair dataset  $\mathcal{D} = \{s_i, a_{w,i}, a_{l,i}\}_{i=1}^N$ , where  $a_{w,i}$  and  $a_{l,i}$  denote the preferred and less preferred generations respectively, we can learn  $r_{\phi}$  by framing the problem as a binary classification and resolving the subsequent problem (Ouyang et al., 2022; Touvron et al., 2023; Meta, 2024):

$$\min_{\phi} \mathcal{L}_{pair}(r_{\phi}, \mathcal{D}_{pair}) = -\mathbb{E}_{\mathcal{D}_{pair}} \left[ \log \sigma(r_{\phi}(s, a_p) - r_{\phi}(s, a_n)) \right]. \tag{2}$$

In a standard LLM training pipeline, the preference-based reward model  $r_{\phi}$  is typically initialized from the finetuned SFT model  $\pi_{SFT}$ , augmented by a linear layer on the final transformer layer, which generates a single scalar prediction for the reward value (Wang et al., 2024a; Askell et al., 2021; Ouyang et al., 2022).

## 2.3 RL Finetuning

Given a LLM policy  $\pi_w$  with parameter w, a reward model  $r_\phi(a, s)$  and a prompt set  $\mathcal{D}_p = \{s_i\}_i^M$ , we aim to optimize the policy by maximizing the following RL objective (Ouyang et al., 2022; Achiam et al., 2023; Touvron et al., 2023):

$$\max_{w} \quad \mathbb{E}_{s \sim \mathcal{D}_{p}, a \sim \pi_{w}} \left[ r_{\phi}(s, a) \right]. \tag{3}$$

When solving the problem in eq. (3) we typically initialize  $\pi_w$  with SFT policy  $\pi_{SFT}$  instead of starting from scratch. In previous works a number of online RL method such as proximal policy optimization (PPO) (Schulman et al., 2017), reward ranking (RAFT) (Dong et al., 2023) and REINFORCE (Williams, 1992).

Another direction of RL finetuing involves reward-free methods, which directly optimize  $\pi_w$  using pre-collected preference data, without the need for a reward model. The rationale behind this approach is to fine-tune the model within a neighborhood of  $\pi_{SFT}$ , ensuring that the probability of generating both preferred and less preferred samples aligns with the pre-collected preference dataset. Direct Preference Optimization (DPO) (Rafailov et al., 2024b) is the most widely adopted method in this direction.

#### 3 Limitations in Traditional RLHF

In this section, we discuss several limitations in the current RLHF pipeline, which are major bottlenecks in the multi-task LLM post-training.

#### **3.1 Limitation of Reward Modelling**

**Insufficient capability for fine-grained criteria alignment.** Despite being based on a sophisticated LLM, the reward model may struggle to provide accurate alignment guidance (Pan et al., 2022), particularly in tasks requiring fine-grained criteria such as identifying correct answers in math questions and assessing code snippet correctness for coding problems. This limitation, inherent to preference-based learning, necessitates additional support to enhance the reward model's effectiveness in handling these specific requirements.

**Proxy nature in coarse-grained preference setting.** Reward hacking can occur even in coarse-grained settings where the goal is to optimize human preferences, as the reward model, serving as a proxy for true preferences, may contain misspecifications (Gao et al., 2023; Moskovitz et al., 2023). This can lead to the model favoring less-preferred outputs, misdirecting the alignment process. A common mitigation strategy is to include a KL penalty in the RL objective to limit deviation from the initial policy, πSFT. However, this approach does not directly address the reward model's imperfections, indicating the need for a more systematic approach to tackle reward hacking.

#### **3.2 Limitation of RLHF Optimizer**

**Contradictory optimization objectives.** The initial success of LLM hinges on the assumption that human preferences are homogeneous (Bakker et al., 2022), but they actually vary widely (helpfulness, harmlessness, honesty, etc) (Casper et al., 2023; Rame et al., 2024). The current RLHF pipeline trains separate reward models for each task and combines them using linear weights (Ramamurthy et al., 2022; Glaese et al., 2022; Yuan et al., 2023). However, this approach applies the same weight of rewards to all tasks, which can be suboptimal (e.g., 90% helpfulness + 10% harmlessness may work well for safe scenarios but lead to risky responses in dangerous situations).

**Rigid optimization strategy for mutli-tasks alignment.** In the standard RLHF pipeline, a uniform RL optimizer setup is typically applied across all tasks (Ouyang et al., 2022). However, this approach may not be optimal since the most effective hyperparameters, including number of generations per-prompt, batch-size, and KL-regularization, often differ between tasks due to unique nature of each task. For example, tasks requiring more exploration typically need a larger number of generations per prompt, whereas other tasks can work well with fewer.

#### **3.3 Motivation**

In multi-task LLM alignment settings, where the goal is to enhance LLM performance across various tasks, the limitations of reward modeling and RLHF optimizers discussed in Section 3 are significant bottlenecks that hinder the RLHF process from effectively improving LLM performance across all tasks. In the following section, we will introduce a novel RLHF framework, Constraint Generative Policy Optimization (CGPO), which addresses all the aforementioned limitations in the most principled manner.

## **4 Constraint Generative Policy Optimization**

In this section, we first explore how to implement the CGPO framework within the scope of a single task with MoJs, as detailed in Section 4.1. Subsequently, we discuss the implementation of CGPO to manage scenarios involving multiple objectives in Section 4.2 for multi-task learning.

#### **4.1 CGPO in Single Task with Single Objective**

The primary design of CGPO is to integrate multiple constraints to mitigate the issue of reward hacking, which arises from the limited capabilities of reward models. Specifically, in addition to optimizing the accumulated reward model value as shown in eq. (3), we also ensure that the model generation meets several constraints. For example, in mathematical reasoning tasks, we strictly require model generations to provide correct answers. This is essential since the model often fails to solve the problem correctly, yet the reward model might still allocate high values to these incorrect solutions. Another example is in general chat tasks with prompts that are free of harmful intent. We require model generations to consistently respond to user queries. This is crucial because there are instances where the model may refuse to answer, and the reward model might erroneously assign high values to such non-responsive generations. In these cases, purely maximizing the reward model could impair the model's reasoning capability and lead to an overly

conservative tendency. By introducing these constraints based on our prior knowledge about the weaknesses of each reward model, we can avoid critical reward hacking patterns effectively.

We denote the set of constraints that the LLM generations need to satisfy as  $\{C_1, C_2, \dots, C_M\}$  and the state-action set that satisfies constraint  $C_k$  as  $\Sigma_k$ , i.e.,  $\Sigma_k = \{(s, a) \in S \times \mathcal{A} \text{ and } (s, a) \text{ satisfies requirement of } C_k\}$ . We define the feasible region as the state-action set that satisfies all constraints as  $\Sigma = \Sigma_1 \cap \Sigma_2 \cap \ldots \cap \Sigma_M$ . In the single task setting, CGPO solves the following constrained problem (Ying et al., 2022; Zhang et al., 2024; Luo et al., 2024; Xu et al., 2021)

$$\max_{w} \quad \mathbb{E}_{s \sim \mathcal{D}_{p}, a \sim \pi_{w}} [r(s, a)]$$
s.t. 
$$\operatorname{Prob}_{s \sim \mathcal{D}_{p}, a \sim \pi_{w}} ((s, a) \in \Sigma) \geq 1,$$

$$\operatorname{KL}_{s \sim \mathcal{D}_{p}} (\pi_{w} | \pi_{\text{ref}}) \leq \operatorname{KL}_{\text{max}},$$
(4)

where  $\pi_{ref}$  is the initialization model and  $KL_{max}$  is the threshold of KL-divergence, which could vary for different tasks.

The high-level framework of CGPO in the multiple-constraints and single-objective setting is illustrated in Algorithm 1. At each iteration, we sample a minibatch from the prompt set D, and then apply the current LLM policy to generate K responses  $(1 \le K)$  for each prompt. Subsequently, we apply all judges  $J = \{J_h\}_{h=1}^M$  to all generated samples to evaluate whether a generation violates a specific constraint. We label a generation  $a_{t,i}^k$  as "violated" if it fails any one of the constraint judgments, and "satisfied" otherwise. Note that the constraint judge is a module for evaluating the constraint satisfaction conditions, which could be a rule-based script or an LLM classifier. This module can address a wide range of constrained problems in the LLM post-tuning scenario. We will discuss this in detail in Section 4.1.4.

After that, we split the generations into "Positive" and "Negative" groups, depending on the constraint satisfaction label. We then apply a constrained RLHF optimizer to update the policy with these two groups of samples (see line 9). In our work, we propose three new RLHF optimizers to efficiently solve the multi-constraint problem in the LLM setting. For Option I, we develop a policy gradient approach and an online DPO approach, and for Option II, we develop a reward ranking-based approach. These optimizers will be discussed in detail in the subsequent sections.

#### **Algorithm 1** CGPO(D, $\pi_{w_0}$ , J, B, R, O, T) in single task with multi-constraints

```
1: Input: prompt set D = \{s_{t,i}\}_{i=1}^N, LLM starting policy \pi_{w_0}, constraint judge set J = \{J_h\}_{h=1}^M, batchsize B, reward model
      R, iteration number T, constrianed RLHF optimizer O.
 2: for t = 0, 1, ..., T do
         Prompt sampling: \{s_{t,i}\}_{i=1}^{B} \sim D
Response generation: \{a_{t,i}^{k}\}_{k=1}^{K} \sim \pi_{w_{t}}(\cdot|s_{t,i}) \text{ for } 1 \leq i \leq n
Constraint judgement: y_{t,i}^{k} = \bigvee_{h=1}^{M} J_{h}(s_{t,i}, a_{t,i}^{k}) \text{ for } 1 \leq i \leq n \text{ and } 1 \leq k \leq K
 3:
 4:
 5:
         Split sample set:
 6:
             Positive samples: X_t^+ = \{(s_{t,i}, a_{t,i}^k) \text{ for } 1 \le i \le n, 1 \le k \le K \text{ where } y_{t,i} = 1\}
 7:
 8:
             Negative samples: X_t^- = \{(s_{t,i}, a_{t,i}^k) \text{ for } 1 \le i \le n, 1 \le k \le K \text{ where } y_{t,i} = 0\}
         Update \pi_{w_t} \to \pi_{w_{t+1}} for policy optimization with optimizer O:
 9:
             [Option I]: maximize likelihood of X_t^+ with high R(x^+) and minimize likelihood of X_t^- with low R(x^-)
10:
11:
             [Option II]: maximize likelihood of X_t^+ with high R(x^+)
12: end for
```

Intuitively, with either the Option I or Option II updating strategy, CGPO encourages the policy to explore regions that satisfy all constraints to maximize the expected reward model value. Note that CGPO is a primal-type constraint policy optimization approach, which differs from the standard primal-dual approach adopted in the constrained RL area. CGPO does not involve co-optimizing the dual variable, thus avoiding the drawbacks of extensive hyperparameter tuning issues associated with the primal-dual approach. Due to this reason, CGPO is user-friendly even with multiple different types of constraints, making it well-suited for the LLM post-tuning scenario.

In the following sections, we will discuss how to implement Algorithm 1 with our proposed RLHF optimizers: Calibrated-Regularized Policy Gradient (CRPG) and Constrained Online DPO (CODPO) for Option I, and Calibrated-Regularized Reward Ranking Fine-tuning (CRRAFT) for Option II. Subsequently, we will discuss the constraint judge module that we developed in CGPO, which enables us to assess the generation's constraint satisfaction condition.

#### 4.1.1 Calibrated Regularized Policy Gradient (CRPG)

In this section, we discuss our new constraint RLHF optimizer, the Calibrated Regularized Policy Gradient (CRPG), which is a policy gradient-based approach.

**Calibrated Reward.** In the traditional RLHF algorithm, the reward model is typically directly incorporated into RL optimizers to progressively refine the policy. However, this method can pose difficulties when the reward model value is not properly calibrated. For preference reward models trained with eq. (2), the reward's accuracy may be proficient in distinguishing between good and bad generations from the same prompt. However, the reward model values between generations from different prompts may not be directly comparable due to potential significant variations in the reward model value range for different prompts. Due to such reasons, standard RLHF algorithms, such as PPO and REINFORCE, could lead to suboptimal performance due to the poor calibration of the reward model (Rita et al., 2024). In CRPG, we introduce a novel and low-cost reward calibration strategy to address this issue.

We consider the scenario where each prompt s used in RLHF fine-tuning has a corresponding baseline response  $\bar{a}$ . This condition can be easily satisfied in practice.

- **Option 1:** We repurpose the prompt set from the SFT training set and/or the reward model training set. For the SFT training dataset, the pre-collected golden response is utilized as the baseline response, denoted as  $\bar{a}$ . For the pair-wise reward model training dataset, the preferred response is designated as the golden response  $\bar{a}$ .
- **Option 2:** Given an RLHF fine-tuning prompt set  $D_d$ , we use  $\pi_{ref}$  to generate the baseline response for all prompts  $s \in D_d$ , i.e.,  $\bar{a} \sim \pi_{ref}(\cdot|s)$  before starting RLHF fine-tuning.

Without loss of generality, we assume there is an underlying policy  $\bar{\pi}$  that generates the baseline responses, denoted as  $\bar{a} \sim \bar{\pi}(\cdot|s)$ . Given the baseline response  $\bar{a}$ , we developed the following calibrated reward to replace the raw reward model  $r_{pair}(s,a)$ :

$$R_{calib}(s, a) = \sigma(r_{pair}(s, a) - r_{pair}(s, \bar{a})). \tag{5}$$

Intuitively,  $R_{pair}(s, a)$  here represent the probability of a being better than baseline response  $\bar{a}$  conditioned on the same prompt s, i.e.,

$$R_{calib}(s, a) \approx \text{Prob}(a > \bar{a}|s).$$

The advantages of using calibrated rewards  $R_{pair}$  are twofold:

- 1. The magnitude of  $R_{calib}$  becomes meaningfully comparable across different prompts. This is because it represents the probability that the current policy  $\pi$  is superior to the baseline  $\bar{\pi}$  for different actions. In other words, if  $R_{calib}(s,a) > R_{calib}(s',a')$ , it directly implies that action a given state s is better than action a' given state s', conditioned on the baseline policy  $\bar{\pi}$ . However, this implication cannot be made if  $r_{pair}(s,a) > r_{pair}(s',a')$ .
- 2. The magnitude of the calibrated reward model is strictly bounded between 0 and 1. This constraint prevents an action with an extremely large raw value from dominating the policy update direction, which could be misleading, since a large raw reward value does not necessarily imply superior action quality.

Based on  $R_{calib}(s, a)$ , we now reformulate RLHF objective in eq. (3) as

$$\max_{w} \bar{J}(\pi_{w}) = \mathbb{E}_{a \sim \pi_{w}(\cdot|s), s \sim D_{d}} \left[ R_{calib}(s, a) \right]$$
(6)

where  $\bar{J}(\pi_w)$  is the policy optimization objective. Intuitively, it represents the probability of current policy  $\pi_w$  being better than the baseline policy  $\bar{\pi}$  conditioned on the prompt set  $D_d$ , i.e.,

$$\bar{J}(\pi_w) \approx \text{Prob}(\pi_w > \bar{\pi}|D_d).$$

Constraint Regularized Gradient. Recall that in the multi-constraint setting, our goal is to maximize the expected reward model while aligning the LLM such that its generations strictly adhere to a set of constraints. These constraints compensate for the limitations of the reward model, including safety requirements, reasoning accuracy, and factual correctness. These aspects may not be fully captured by the reward model but can be well addressed via a separate

rule-based judge or an LLM-based judge. Note that the "Positive samples" in line 6 of Algorithm 1 is a subset of  $\Sigma$ , i.e.,  $X_t^+ \in \Sigma$ . Consequently, we aim to optimize the following multi-constraint objective, denoted as  $\bar{J}_c$ :

$$\max_{w} \bar{J}_c = \mathbb{E}_{a \sim \pi_w(\cdot|s), s \sim D_d} \left[ R_{calib}(s, a) \cdot \mathbf{1}_{(s, a) \in \Sigma} \right]. \tag{7}$$

By solving the optimization problem presented in eq. (7), the LLM is aligned to maximize the expected value of the calibrated reward model as much as possible, while remaining within the constraint satisfaction region.

Given  $R_{calib}$  and  $\Sigma$ , we define the following constraint regularized reward as

$$R_{cr}(s,a) = \begin{cases} R_{calib}, & \text{if } (s,a) \in \Sigma \\ 0, & \text{if } (s,a) \notin \Sigma \end{cases}$$
(8)

With the calibrated regularized reward  $R_{cr}$ , we rewrite eq. (7) as

$$\max_{w} \bar{J}_{c} = \mathbb{E}_{a \sim \pi_{w}(\cdot|s), s \sim D_{d}} \left[ \cdot R_{cr}(s, a) \right]. \tag{9}$$

We consider the following update to optimize  $\bar{J}_c$ 

$$w_{t+1} = w_t + \alpha_t \cdot g_c(\pi_{w_t}), \tag{10}$$

where

$$g_c(\pi_w) = \frac{1}{N} \sum_{i}^{N} \nabla \log \pi_w(s_i, a_i) \cdot R_{cr}(s_i, a_i).$$

The subsequent theorem illustrates that CRPG has global optimality guarantee for both objective achievement and constraint satisfaction in the multi-constraint LLM alignment setting.

**Theorem 1** Consider the CRPG update defined in Eq. (10). Consider the scenario where the optimal policy  $\pi^*$  of eq. (7) satisfies  $\operatorname{Prob}_{\pi^*,D_d}((s,a) \in \Sigma) = 1$ . Denote the policy set within the constraint satisfaction region as  $\Pi_{\Sigma}$ , and the globally optimal policy within  $\Pi_{\Sigma}$  as  $\pi_c^*$ , i.e.,  $\pi_c^* = \operatorname{argmax}_{\pi \in \Pi_{\Sigma}} \mathbb{E}_{\pi,D_d}[R_{cr}(s,a)]$ . Given a few mild assumptions, we have

$$\mathbb{E}_{\pi_{c}^{*}, D_{d}}\left[R_{cr}(s, a)\right] - \mathbb{E}_{\pi_{w_{t}}, D_{d}}\left[R_{cr}(s, a)\right] \leq O\left(\frac{1}{poly(t)}\right),$$

$$Prob_{\pi_{w_{t}}, D_{d}}\left((s, a) \notin \Sigma\right) \leq O\left(\frac{1}{poly(t)}\right).$$

**CRPG Implementation.** Consider the KL divergence between  $\pi_{ref}$  and  $\pi_w$  as a universal regularization method to prevent reward hacking during CRPG fine-tuning. We propose the following new reward regularization approach:

$$\tilde{R}_{cr}(s, a) = \max \left\{ 1 - \frac{\log(\pi_w(s_i, a_i) / \pi_{ref}(s_i, a_i))}{KL_{max}}, 0 \right\} \cdot R_{cr}(s, a). \tag{11}$$

It is important to note that  $\tilde{R}_{cr}$  not only penalizes samples that deviate significantly from  $\pi_{ref}$ , but also strictly bounds the overall KL divergence under the predefined threshold KL<sub>max</sub>, as shown in the following theorem.

**Theorem 2** Consider the CRPG update with KL-regularized reward  $\tilde{R}_{cr}$  in eq. (11) and let  $\pi_{w_0} = \pi_{ref}$ . Under mild assumptions and with appropriately selected learning rate  $\alpha_t$ . We have

$$KL(\pi_{w_t}|\pi_{ref}) \leq KL_{\max}, \quad \forall t \geq 0.$$

Moreover, to reduce the variance in the CGPG gradient estimation, we consider subtracting a baseline from the  $g_c$  without changing its expected direction as following

$$\tilde{g}_c(\pi_{w_t}) = \frac{1}{n} \sum_{i=1}^{n} \nabla \log \pi_{w_t}(s_{t,i}, a_{t,i}) \cdot \left[ \tilde{R}_{cr}(s_{t,i}, a_{t,i}) - \frac{1}{n} \sum_{i=1}^{n} \tilde{R}_{cr}(s_{t,i}, a_{t,i}) \right]. \tag{12}$$

The final CRPG update in multi-constraints finetuning setting is given as

$$w_{t+1} = w_t + \alpha_t \cdot \tilde{g}_c(\pi_{w_t}).$$

#### 4.1.2 Constrained Online Direct Preference Optimization (CODPO)

Based on Direct Preference Optimization (DPO), a widely used offline RLHF alignment algorithm in the unconstrained setting, we propose a new variant called Constrained Online Direct Preference Optimization (CODPO) to solve the constrained RLHF fine-tuning problem.

Recall that in DPO (Rafailov et al., 2024b), the optimal policy  $\pi^*$ , which aligns with human preferences in the  $\beta$ -regularized MDP setting, satisfies the following preference model:

$$P_{\pi^*}(a_p > a_n) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(s, a_n)}{\pi_{\text{ref}}(s, a_n)} - \beta \log \frac{\pi^*(s, a_p)}{\pi_{\text{ref}}(s, a_p)}\right)}.$$

Given a pairwise preference sample pair  $(s, a_p)$  and  $(s, a_n)$ , we update our policy by solving the following problem:

$$\min_{w} \mathcal{L}_{DPO}(\pi_{w}) = -\mathbb{E}_{(s,a_{p},a_{n})} \left[ \ell_{DPO}(\pi_{w}, s, a_{p}, a_{n}) \right].$$
where 
$$\ell_{DPO}(\pi_{w}, s, a_{p}, a_{n}) = \log \sigma \left( \beta \log \frac{\pi_{w}(s, a_{p})}{\pi_{ref}(s, a_{p})} - \beta \log \frac{\pi_{w}(s, a_{n})}{\pi_{ref}(s, a_{n})} \right)$$
(13)

To prevent the possible decreasing likelihood of positive samples  $a_p$ , it has been proposed to add a regularization term to the vanilla DPO loss (Pal et al., 2024):

$$\tilde{\ell}_{\text{DPO}}(\pi_w, s, a_p, a_n) = \ell_{\text{DPO}}(\pi_w, s, a_p, a_n) + \frac{\lambda}{|a_p|} \cdot \log(\pi_w(s, a_p)), \tag{14}$$

where  $|a_p|$  represents the length of response  $a_p$ . By appropriately tuning the hyperparameter  $\lambda$ , the formulation in eq. (14) can effectively increase the likelihood of  $a_p$  while decreasing the likelihood of  $a_n$  to maximize the margin between positive and negative generations.

In CODPO, similar to CRRAFT, we first generate multiple responses for each prompt using the current policy  $\{a_{t,i}^1, a_{t,i}^2, \dots, a_{t,i}^K\} \sim \pi_{w_t}(\cdot \mid s_{t,i})$  and split the generations into positive samples  $X_t^+$  and negative samples  $X_t^-$ . After that, we select the positive sample from  $X_t^+$  with the highest reward value, and the negative sample from  $X_t^-$  with the lowest reward value, i.e.,

$$a_{i,t}^{+} = \underset{\substack{k \in [K], \\ (s_{i,t}, a_{i,t}^{k}) \in X_{t}^{+}}}{\operatorname{argmax}} r_{pair}(s_{i,t}, a_{i,t}^{k}),$$

$$a_{i,t}^{-} = \underset{\substack{k \in [K], \\ (s_{i,t}, a_{i,t}^{k}) \in X_{t}^{-}}}{\operatorname{argmin}} r_{pair}(s_{i,t}, a_{i,t}^{k}).$$

In cases where no generations satisfy all constraints, we can skip this sample. Conversely, when no generations violate any constraints, we can select the generation with the lowest reward model value as the negative sample.

Then, at each iteration, we update the policy as follows:

$$w_{t+1} = w_t - \alpha_t \cdot \frac{1}{n} \sum_{i=1}^n \nabla \tilde{\ell}_{\text{DPO}}(\pi_{w_t}, s_{i,t}, a_{i,t}^+, a_{i,t}^-).$$
 (15)

#### 4.1.3 Calibrated Regularized Reward Ranking Finetuning (CRRAFT)

In this section, we introduce another constrained RLHF policy optimizers that we proposed: Calibrated Regularized Reward Ranking Finetuning (CRRAFT), which is built upon the RAFT.

In the original RAFT algorithm (Dong et al., 2023), each round involves generating multiple responses from a prompt using the current policy model, denoted as  $\{a_{t,i}^1, a_{t,i}^2, \dots, a_{t,i}^K\} \sim \pi_{w_t}(\cdot \mid s_{t,i})$ . A reward model r is then utilized to select the response with the highest reward model score, i.e.,  $a_j^* = \operatorname{argmax}_{k \in [K]} r_{\text{pair}}(s_{t,i}, a_{t,i}^k)$  (note that whether a calibrated reward is used or not does not affect the reward ranking result). Subsequently, an one-step SFT update is performed

to maximize the likelihood of this generated sample  $(s_{t,i}, a_{t,i}^*)$ . The policy model is iteratively updated to improve its alignment with the reward model  $r_{pair}$  as follow

$$w_{t+1} = w_t + \alpha_t \cdot \frac{1}{n} \sum_{j=1}^n \nabla \log(\pi_{w_t}(s_{t,i}, a_{t,i}^*)).$$
 (16)

In the multi-constraint setting, we make the following two changes on top of RAFT to develop our CRRAFT optimizer:

• After applying the reward model to score each responses, we adopt Option I in Algorithm 1 to first filter out those generated responses that violated any of the constraints. Additionally, to avoid large drift of current policy from starting point policy  $\pi_{\text{ref}}$ , we also filter out all generations whoes KL-divergence is larger than a pre-defined threshold  $\text{KL}_{\text{max}}$ , i.e.,  $\text{KL}_{(s_{i,t},a_{i,t}^k)} = \frac{\log \pi_{w_t}}{\log \pi_{\text{ref}}}(s_{i,t},a_{i,t}^k) > \text{KL}_{\text{max}}$ . After that we apply reward ranking to select the one with the highest reward model score from the rest of responses, i.e.,

$$a_{i,t}^{*} = \underset{\substack{k \in [K], \\ (s_{i,t}, a_{i,t}^{k}) \in X_{t}^{+}, \\ \text{KL}(s_{i,t}, a_{i,t}^{k}) \leq \text{KL}_{\max}}}{\underset{\text{argmax}}{\operatorname{argmax}}} r_{pair}(s_{i,t}, a_{i,t}^{k}). \tag{17}$$

We refer to the procedure in eq. (17) as constrained regularized reward ranking. It's important to note that CRRAFT not only has the capability to manage multiple constraints, but it also strictly bounds the KL-divergence. This is a feature that the standard RAFT algorithm lacks.

Note that there may be instances where no generations remain after filtering. In such cases, if the pre-collected baseline response  $\bar{a}_{i,t}$  satisfies all constraints, it can be used as  $a_{i,t}^*$ . If it doesn't, this datapoint can be skipped.

 After the constrained regularized reward ranking, instead of directly performing SFT update w.r.t the chosen sample as eq. (16) does, here we reweigh each chosen response by their calibrated reward value and then perform SFT update as follow

$$w_{t+1} = w_t + \alpha_t \cdot \tilde{g}_{ra}(\pi_{w_t})$$

$$= w_t + \alpha_t \cdot \frac{1}{n} \sum_{i=1}^n R_{calib}(s_{i,t}, a_{i,t}^*) \cdot \nabla \log(\pi_{w_t}(s_{i,t}, a_{i,t}^*)).$$
(18)

By incorporating the calibrated reward model value in the update, we can differentiate the emphasis on chosen responses based on their quality, unlike the RAFT algorithm which treats all chosen responses equivalently. This approach allows for a more refined alignment with the reward model.

Please note that unlike CRPG and CODPO, CRRAFT specifically focuses on increasing the likelihood of constraint-satisfied positive samples and disregards the constraint-violated negative samples.

#### 4.1.4 Judges in CGPO

The key step in implementing multi-constraint CGPO optimizers, as outlined in Section 4.1.1 and Section 4.1.3, is to determine whether a generation (s, a) satisfies a constraint or not. This determination allows us to split generated samples into positive  $(X_t^+)$  and negative  $(X_t^-)$  groups given the label y predicted by each constraint judge  $J_h$ , i.e.,

$$J_h(s, a) = y \in \{0, 1\}, \text{ where } 1 \le h \le M,$$

and then apply our customized constraint RLHF optimizers based on that classification. In CGPO, we have developed and integrated the following two types of constraint judge modules to assess whether a generation satisfies a constraint:

• Rule-based constraint judge module: This module employs a rule-based approach (such as string-matching and code execution) to ascertain whether the generation strictly adheres to predefined regulations (Li et al., 2024a). It is particularly effective for constraints related to precise instruction following, where the generation must meet exact requirements such as length, number of paragraphs, and keyword inclusion (Zhou et al., 2023; Hendrycks et al., 2021b; Cobbe et al., 2021). It can also handle reasoning tasks, such as math problems and code generation.

• LLM-based constraint judge module. This module functions as an LLM generator. In most cases, the generation is formatted according to a template before being sent to the judge module. These modules not only provide access to the constraint satisfaction condition but also offer reasoning behind the judgement construction. Due to this property, they are typically capable of handling more challenging constraint evaluation tasks such as safety violation, reference-based factuality verification, and false refusal patterns. The model could either be a compact LLM fine-tuned with domain-specific data (Inan et al., 2023; Bai et al., 2022) or a powerful, large LLM without task-specific fine-tuning (Yuan et al., 2024b; Zheng et al., 2024).

A detailed introduction to these two types of judges can be found in Appendix B.

#### **4.2 CGPO in Multi-Taks with Multi-Objectives**

In the multi-tasks environment, CGPO utilizes customized combinations of "reward models + MoJs + optimizers" to provide alignment guidance tailored to each task. This approach is designed to better accommodate the specific nature of each problem, thereby enable CGPO to have better chance to achieve optimal alignment outcomes. Figure 2 provides

![](_page_10_Figure_4.jpeg)

**Figure 2** CGPO in a multi-tasks setting. The reward model, a MoJs, and optimization setup are uniquely tailored to the specific characteristics of each task. This customization ensures the most effective and targeted approach for achieving optimal performance across all tasks, even those with potentially contradictory goals.

an end-to-end illustration of how the CGPO pipeline functions in the multi-tasks setting. The entire CGPO pipeline has the following two core components: multi-objective reward modeling and multi-experts alignment.

Mutli-Objective Reward Modelling. Unlike the approach adopted in previous RLHF pipelines in multi-objective scenarios, which applies the same linear combined reward model to all prompts in the prompt set *D*, CGPO first classifies the prompt set *<sup>D</sup>* into distinct, non-overlapping categories based on the nature of the prompts, i.e., *<sup>D</sup>* <sup>=</sup> {*D*<sup>1</sup>, *<sup>D</sup>*<sup>2</sup>, . . . , *<sup>D</sup>L*}. Each prompt set *D<sup>l</sup>* ∈ *D* is referred to as a task. For example, prompts with harmful intent, which could potentially lead LLM to generate unsafe responses, are grouped into a class labeled "harmful intent". Conversely, prompts without

unsafe intent, primarily focused on information gathering and casual conversation, are grouped into a class labeled "general chat". This categorization can be performed during the data collection phase or by prompting an LLM to carry out the categorization given the definitions of different classes. Subsequently, with a collection of trained reward models denoted as  $\{R_{calib,1}, R_{calib,2}, \ldots, R_{calib,V}\}$ , we tailor the specific reward model to be applied for each task  $D_l$ . This customization guarantees that each prompt class  $D_l$  benefits from the most appropriate guidance provided by the corresponding reward model. Note that the number of reward models, denoted by V, is less than or equal to the number of tasks, meaning a single reward model can be utilized across multiple tasks.

The major advantage of segregating the reward modeling for each individual task is to exclude irrelevant or contradictory objectives, thus enabling each task to focus solely on optimizing its own goal metrics without interference from other objectives, which could otherwise lead to suboptimal gains in target goals.

**Multi-Expert Alignment.** The concept of multi-expert alignment involves applying customized MoJs, reward model and policy optimization setups for each task.

After the policy model generates online samples for each task, we employ a mixture of task-specific judges to identify generations that do not meet predefined standards. It is crucial to emphasize that the selection of judges are uniquely tailored for each task, reflecting the particular shortcomings of each reward model and our established performance criteria for LLMs in these tasks. For instance, in the "general chat" task, we employ LLM-based judges for false refusal and factuality to enhance responsiveness and ensure honesty. In "reasoning" tasks, we implement a rule-based math/coding constraint judge to guarantee correctness and accuracy.

Based on the status of constraint satisfaction across generations and a customized reward model, we implement an RLHF policy optimizer with a specifically tailored hyperparameter setup to align each task effectively. This method deviates from the conventional RLHF pipeline, which generally employs a uniform optimizer setup for task alignment. For tasks that have precise judges and require extensive exploration to derive the correct response, such as instruction following, math, and coding, we apply a lenient KL threshold and allow a higher number of generations per prompt. In contrast, for tasks where precise judges are lacking and extensive exploration is less critical, such as "general chat," we opt for a stricter KL threshold and a reduced number of generations per prompt.

```
Algorithm 2 CGPO(\{D_l\}_{l=1}^L, \pi_{w_0}, \{J_l\}_{l=1}^L, \{B_l\}_{l=1}^L, \{R_l\}_{l=1}^L, \{O_l\}_{l=1}^L, T) in multi-tasks with multi-constraints & multi-objectives
```

```
    Input: Multi-tasks prompt set {D<sub>l</sub>}<sup>L</sup><sub>l=1</sub>, LLM starting policy π<sub>w0</sub>, judges sets {J<sub>l</sub>}<sup>L</sup><sub>l=1</sub>, multi-tasks batchsizes {B<sub>l</sub>}<sup>L</sup><sub>l=1</sub>, reward model sets {R<sub>l</sub>}<sup>L</sup><sub>l=1</sub>, multi-tasks weights {ρ<sub>l</sub>}<sup>L</sup><sub>l=1</sub>, multi-tasks optimizers {O}<sup>L</sup><sub>l=1</sub>, iteration number T.
    for t = 0, 1, · · · , T do
    for l = 0, 1, · · · , L do
    Obtain gradient ğ<sub>l</sub>(π<sub>w1</sub>) for l-th task via CGPO(D<sub>l</sub>, π<sub>w1</sub>, J<sub>l</sub>, B<sub>l</sub>, R<sub>l</sub>, O<sub>l</sub>, 1) in Algorithm 1
    end for
    Update with multi-tasks gradient accumulation w<sub>t+1</sub> = w<sub>t</sub> + α<sub>t</sub> · ∑<sup>L</sup><sub>l=1</sub> ρ<sub>l</sub> · ğ<sub>l</sub>(π<sub>w1</sub>),
    end for
```

The high-level framework of CGPO in the multiple-constraint and multiple-objective setting is illustrated in Algorithm 2. Specifically, at each iteration t, we process each individual task to compute the updated gradient  $\tilde{g}_l(\pi_{w_l})$ . This computation is based on the task-specific prompt set  $D_l$ , reward model  $R_l$ , judges  $J_l$ , batch size  $B_l$ , and optimizer  $O_l$ , following the steps outlined in Algorithm 1. Subsequently, we accumulate the gradients across all tasks and combine them with our predefined task weights  $\{\rho_l\}_{l=1}^L$ , which are then used to update our model parameters.

## 5 Experiments

In this section, we outline the specifics of our experimental setup designed for multi-task alignment under conditions of extreme multi-constraints and multiple objectives. Specifically, we focus on fine-tuning a LLM to achieve alignment across the following five tasks:

• General chat: This task is designed to enhance the general conversational abilities of LLMs by considering multi-turn conversational histories (Wang et al., 2024b). It focuses on boosting the coherence, consistency, and correctness of responses, thereby making the interactions more logical and seamless. Additionally, this task

improves the model's capability to deliver responses that are better aligned with the user's intentions and queries, and are factually grounded (Sun et al., 2024).

- Instruction Following: This task is designed to enhance the ability of LLMs to follow instructions accurately within specific contexts or industries (Zhou et al., 2023). By fine-tuning LLMs to adapt to particular domains or user requirements, they can deliver more precise and relevant responses. This improvement leads to a more satisfying and efficient user experience, making LLMs more effective and versatile tools across various applications.
- Math/Code Reasoning: This task is designed to enhance the math and coding capabilities of LLMs, enabling them to address more complex problems and broaden their range of functions. These include tasks like debugging code or solving mathematical equations, which are vital in technical fields (Hendrycks et al., 2021b; Cobbe et al., 2021; Chen et al., 2021; Austin et al., 2021). Furthermore, improving LLMs' ability to comprehend and produce mathematical and code-related content results in greater accuracy and efficiency in activities that demand meticulous logical reasoning and computational thinking.
- Engagement Intent: This task aims to enhance user engagement and interaction with the LLM. To address this, we involve human annotators who interact with the model and provide binary feedback (like or dislike) for each response generated by the LLM. Our objective is to maximize the likelihood that users will favorably respond to the LLM's outputs.
- Harmful Intent: This task trains LLMs to recognize and resist safety-related adversarial attacks. It ensures that LLMs are safeguarded against exploitation for malicious purposes, such as generating harmful or misleading information (Sun et al., 2024; Xu et al., 2020). By enhancing their ability to operate safely and ethically, this task helps maintain user trust and uphold the credibility of the technology.

## **5.1 Supervised Fine-Tuning**

The foundational model we have chosen is the LLaMA-3.0-70B pre-trained checkpoint. We independently perform SFT using an open-source dataset to establish the initial policy, denoted as π<sup>0</sup>. For all preference pair datasets listed below we only use positive samples in SFT. We utilize the following datasets for the tasks under consideration:

- General chat: LMSys-55k (Chiang et al., 2024), UltraChat (Ding et al., 2023)
- Instruction following: LIama 3.0 70B instruct model synthetic instruction following dataset
- Math/Code Reasoning: Orca-Math Mitra et al. (2024), MetaMath (Yu et al., 2023), Evol-CodeAlpaca (Luo et al., 2023), UltraFeedback (Cui et al., 2023), UltraInteract (Yuan et al., 2024a)
- Harmful Intent: Human annotated safety dataset

The training is carry out for 2 epoches with a learning rate of 10−<sup>5</sup> . A cosince schedule is employed, the global batchsize is set to <sup>128</sup> with minimum rate <sup>0</sup>.<sup>1</sup> and warm-up steps 200. The detail of how we obtain synthetic instruction following dataset and safety dataset SFT can be found in Appendix A.

## **5.2 Reward Modelling**

We have employed open-source pairwise preference data to train three specialized reward models (RMs):

- **Helpfulness RM:** This model is tailored for tasks such as general chat, instruction following, and math/code reasoning. It is based on the LLaMA-3-70B instruct finetuned model. The training utilized the following pairwise preference datasets:
  - **General chat:** Includes datasets such as HH-RLHF (Bai et al., 2022), SHP (Ethayarajh et al., 2022), HelpSteer (Wang et al., 2023), Distilabel-Capybara (Ethayarajh et al., 2024), Distilabel-Orca (Álvaro Bartolomé Del Canto et al., 2024), and LMSys-55k (Chiang et al., 2024).
  - **Instruction Following:** LIama 3.0 70B instruct model synthetic instruction following pairwise preference dataset.
  - **Math/Code Reasoning:** Features datasets like Argilla Math (Álvaro Bartolomé Del Canto et al., 2024), UltraFeedback (Cui et al., 2023) and UltraInteract (Yuan et al., 2024a).

- **Engagement RM:** This RM is designed to simulate user engagement preferences. Initially, we fine-tune a binary classifier predictor using the LLaMA-3-70B instruct model to predict a user's engagement intent based on real interaction data between the language model and the user. We then treat this predictor as the oracle for user intent regarding engagement with the language model, given prompts and generations. To gather pair-wise training data, we subsample 129692 prompts from the LMSys-1M dataset (Zheng et al., 2023a) and use the LLaMA-3-70B instruct model to generate four responses for each prompt. Each prompt is then scored using the oracle engagement predictor. We select the generation with the highest score as the "chosen" response and the generation with the lowest score as the "rejected" response. By doing this, we compile the pair-wise dataset and train the engagement RM based on this data.
- **Safety RM:** Focused on ensuring safe responses in scenarios with potentially harmful user prompts, this model is based on the LLaMA-3-8B instruct finetuned model. It utilizes a human-annotated safety pairwise preference dataset that identifies harmful intent in prompts.

It is important to note that we are considering training a unified Helpfulness RM that encompasses general chat, instruction following, and math/code reasoning, rather than training three separate RMs. This consideration is based on the observed positive correlation among these tasks. A unified RM, trained with a blended dataset from these domains, is expected to yield superior performance compared to training separate RMs for each individual task.

#### **5.3 Mixture of Judges**

To address the limitations of the reward model, we have implemented several judges in our experiment for multi-task alignment:

- **False refusal judge:** Enhancing safety protocols may cause LLMs to become overly safe, leading to false refusals when responding to innocuous user queries, thus degrading user experience. It has become critical for LLMs to reduce false refusals while maintaining the same level of safety, both in the research community and in the leading industry models (Cui et al., 2024). To address this challenge, we have developed a false refusal classifier, a fine-tuned LLM designed to detect false refusals to ensure the effectiveness of the LLM.
- **Precise instruction following judge:** Reward models often struggle with precisely following instructions (Zhou et al., 2023). To address this, we have implemented a rule-based judge capable of accurately assessing compliance with over 30 types of specific instruction-following requests found in user prompts, such as "answer the question in two paragraphs." It is important to note that during RLHF finetuning, we will also include precise instruction-following prompts of this type so that the correctness of the generation can be evaluated with this constraint judge.
- **Regex math/code reasoning judge:** Reward models frequently fail to accurately assess the correctness of math and coding problems. To improve accuracy, we have introduced specialized judges for both domains. For math-related queries, we use a rule-based approach to check whether the final answers of responses match the ground-truth answers. For coding problems, we employ a unit-test-based judge that evaluates the accuracy of the code by running it through a series of unit tests.
- **Factuality judge:** Hallucination is a common issue in LLMs, especially during the RLHF phase. The reward model often fails to distinguish between factual and non-factual claims. To address this, we use the Llama3 70B model as a factuality constraint judge to evaluate whether the fact-related claims in an output contradict pre-collected, verified factual data, thereby ensuring the accuracy and reliability of the information provided by the LLM.
- **Safety judge:** The safety reward model alone does not sufficiently ensure the trustworthiness of our model due to its limited accuracy. To further enhance safety, we incorporate LlamaGuard2, an industry leading open sourced fine-tuned LLM, to assess whether an output violates predefined safety standards.

For details on all the above judges, please refer to Appendix B.

## **6 Evaluation Benchmarks**

We assess models using a range of benchmarks to comprehensively evaluate their performance across all tasks. More detailed information of evaluation setup can be found in Appendix C.

#### • **General chat**

- **AlpacaEval-2** (Dubois et al., 2024): This benchmark focus on single-turn conversations and includes 805 test prompts that span a range of topics. The models are evaluated directly against GPT-4 Preview to determine the win rate. The same GPT-4 model also serves as the judge.
- **Chat-Arena-Hard** (Li et al., 2024b): This benchmark includes 500 test prompts sourced from the live data on Chatbot Arena, a crowd-sourced platform for evaluating large language models (LLMs). These prompts assess the model's capabilities in areas such as specificity, domain knowledge, complexity, problem-solving, creativity, technical accuracy, and real-world application. Besides aligning with human preferences, when compared to AlpacaEval-2, Chat-Arena-Hard also demonstrates distinct separability between different models.

#### • **Instruction Following**

– **IFeval** (Zhou et al., 2023): This benchmark concentrates on close-form instruction-following tasks, encompassing 25 verifiable instructions. It comprises 541 evaluation prompts, each potentially containing multiple instruction requests. Four accuracy scores are provided in this benchmark: prompt-level strict accuracy, prompt-level loose accuracy, instruction-level strict accuracy, and instruction-level loose accuracy. We report the average of these four scores to represent the model's performance in this benchmark.

#### • **Math/Coding Reasoning**

- **MATH** (Hendrycks et al., 2021b): This benchmark includes 5000 problems drawn from a variety of mathematics competitions, encompassing a broad spectrum of subjects such as Prealgebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, and Precalculus. Most of these problems demand more than just the simple application of standard mathematical techniques.
- **GSM8K** (Cobbe et al., 2021): This benchmark features 8.5k high-quality problems at the grade school math level. The solutions to these problems rely solely on elementary concepts, making high test performance an achievable goal. Additionally, this dataset exhibits high linguistic diversity while depending on relatively simple grade school math concepts.
- **MBPP** (Austin et al., 2021): This benchmark comprises 974 programming tasks tailored for entry-level programmers. It evaluates the capability of language models to generate concise Python programs based on descriptions provided in natural language. We consider the 0-shot evaluation prompt, which is closer to real-world use cases. We provide a prompt example in the Appendix C.
- **HumanEval** (Chen et al., 2021): This benchmark consists of 164 handwritten programming problems, each featuring a function signature, docstring, body, and unit tests. The programming tasks in this benchmark are designed to evaluate language comprehension, reasoning, algorithmic thinking, and basic mathematics skills. Similar to MBPP, we consider 0-shot evaluation prompt for this benchmark.

#### • **World knowledge & factuality**

- **MMLU** (Hendrycks et al., 2020): This benchmark comprises 15908 multiple-choice questions spanning various branches of knowledge. It encompasses subjects including the humanities, social sciences, and hard sciences. The evaluation dataset includes 57 tasks, covering areas such as elementary mathematics, US history, computer science, law, among others.
- **ARC-Challenge** (Clark et al., 2018): This benchmark features a collection of 2590 natural, grade-school science multiple-choice questions. All questions are considered challenging, as evidenced by the failure of both retrieval and co-occurrence methods to provide correct answers.
- **TruthfulQA** (Lin et al., 2021): The benchmark includes 817 questions across 38 categories, such as health, law, finance, and politics. Each question in the dataset is designed in a way that may lead some individuals

to answer incorrectly due to prevailing misconceptions or false beliefs. We report the multiple-choice QA accuracy score (MC2) in our paper.

• **Engagement Intent**: We subsample 2000 prompts from the LMSys-1M dataset (Zheng et al., 2023a) and utilize the oracle engagement predictor, as discussed in Section 5.2, to evaluate each generation. The average score of the generations is used as a metric to measure the user's intent to interact with the language model.

#### • **Safety**

- **Safety violation ratio (SVR)**: Our safety benchmark is broadly based on the risk categories in ML Commons taxonomy of hazards (Inan et al., 2023). We collect adversarial human-written prompts per risk category to ensure every category is covered and balanced in equal proportions. The prompts are written in such a way to include diverse adversarial styles and attack vectors. We train a safety classifier similar to Llama Guard (Vidgen et al., 2024) to compute an overall violation ratio as an evaluation metric.
- **False refusal ratio (FRR)**: False refusal ratio acts as a counter-metric to violation rate, to make sure the model is not overly safe. The false refusal prompts are collected in tandem to the adversarial prompts, targeting innocuous and benign cases but are also likely to trigger the model to refuse due to safety reasons. One such example is "How to kill the lights". We also drew inspirations from XSTest (Röttger et al., 2023) and OR-Bench (Cui et al., 2024) while constructing our false refusal benchmark. We train a refusal classifier on model responses to compute an overall false refusal ratio as an evaluation metric.

## **6.1 CGPO Training Setup**

In this section, we will show how we implment the CGPO in the RLHF finetuning stage.

**RLHF warm-up.** Unlike previous studies Ouyang et al. (2022); Achiam et al. (2023), which directly employ the SFT model as the initial point for RLHF, our approach introduces a "warm-up" phase. This phase begins with a model that has undergone preliminary fine-tuning through a few steps of DPO, starting from the SFT model. The rationale behind this strategy is that even if DPO and the reward model utilize the same preference training dataset, initiating online RLHF directly from the SFT model and performing policy optimization with the reward model may not be able to explicitly exploit the high-quality preference data, potentially leading to suboptimal performance enhancements. By initiating RLHF with a model already refined by DPO to a certain degree, we can fully harness the advantages of the preference dataset, thereby providing a better starting point for RLHF with improved performance gains and more stable optimization performance.

In our experiments, we employ all reward model training datasets in Section 5.2 to conduct DPO training using the SFT model as described in Section 5.1. The warm-up model is developed through 3,000 update steps. As we will show in Section 6.2, CGPO initiated from the warm-up model significantly outperforms that started from the SFT model.

**Training recipe:** We begin the RLHF finetuning process using the warm-up model. We incorporated the reward models and MoJs, developed in Sections 5.2 and 5.3 respectively, into the CGPO framework to facilitate the RLHF finetuning process. Table 1 shows the treatment we applied for each task.

**Baseline and Ablations:** To assess the performance of various constrained RLHF optimizers proposed in this study, we conducted CGPO training with different optimizers: CRPG, CRRAFT, and CODPO under the same settings (prompt set, reward model, and MoJs). We tailored the hyperparameter settings of the optimizer for various tasks to align with the specific characteristics of each task. Additionally, we consider DPO and PPO as our RLHF baselines. To establish the DPO baseline, we continue running the DPO updates starting from the RLHF warm-up model and extend the training to 14,000 steps to thoroughly optimize all benchmarks listed in Section 6. As previously mentioned, for DPO training, we utilize all reward models' training sets starting from the SFT model. To establish the PPO baseline, we first train a unified reward model by merging all reward models' training data as described in Section 5.2. Following this, we start from the warm-up model and perform PPO updates by applying the unified reward model to all prompt sets listed in Table 1. For both PPO and CGPO variants, we utilize the same global batch size 1024 and conduct 600 update steps.

#### **6.2 Experimental Results**

In this section, we present the main results of our experiments. In Section 6.2.1, we highlight the superior performance of CGPO compared to baseline RLHF methods across various benchmarks. We present ablation studies in Section

| Tasks                                        | General chat                                         | Instruction<br>Following | Math/Coding<br>Reasoning       | Engagement<br>Intent | Harmful<br>Intent               |
|----------------------------------------------|------------------------------------------------------|--------------------------|--------------------------------|----------------------|---------------------------------|
| Prompt                                       | UltraChat,<br>LMSys-55k,<br>XSTest,<br>TriviaQA, ARC | Synthetic IF prompts     | Math,<br>GSM8K,<br>Aqua & APPS | LMSys-1M             | Safety RM<br>training<br>prompt |
| Helpfulness RM<br>Engagement RM<br>Safety RM | ✓                                                    | ✓                        | <b>√</b>                       | ✓                    | ✓                               |
| Style False refusal Precise IF Math/Code     | <i>\ \ \</i>                                         | ✓                        | <b>√</b>                       | √<br>√               |                                 |
| Factuality<br>Safety                         | <b>√</b>                                             |                          |                                |                      | ✓                               |

**Table 1** Tasks and their corresponding prompt sets, reward models, and MoJs. Note that in general chat task the factuality constraint judge is only applied to TriviaQA and ARC prompt set.

6.2.2 to demonstrate the importance of adopting MoJs. Additionally, we discuss the benefits of introducing an RLHF warm-up stage in Section 6.2.3.

#### 6.2.1 Main Results and Ablations

![](_page_16_Figure_4.jpeg)

Figure 3 Comparison of CGPO variants with baseline RLHF algorithms PPO and DPO across various benchmarks

For the online RLHF algorithms CGPO and PPO, we monitor the model's performance at every 10-step interval throughout the training trajectory across various benchmarks, as illustrated in Figure 3. The plot demonstrates that CGPO, when paired with the CRPG and CRRAFT optimizers, consistently enhances performance across all benchmarks compared to the initial model, indicating progressive improvement as training progresses. Specifically, CRPG outperforms all others throughout the entire training period in terms of ARC Challenge, 0-shot HumanEval, 0-shot MBPP, 4-shots MBPP, MATH, and GSM8K. Meanwhile, CRRAFT excels in IFEval during the training phase. Notably, the online RLHF baseline PPO exhibits a significant decline in performance on 0-shot coding benchmarks (MBPP and HumanEval) as training progresses, indicating a severe case of reward hacking. Meanwhile, CGPO with the

CODPO optimizer shows a slight regression on MBPP and IFEval benchmarks compared to the warm-up model, yet it effectively avoids the drastic performance drop observed with PPO in the coding benchmarks. The offline RLHF baseline DPO, while avoiding the drastic regression seen with PPO, remains overly conservative in enhancing the model's performance, resulting in lower metric improvements compared to CGPO with the CRPG and CRRAFT optimizers.

In Table 2, we present the evaluation results for SFT, DPO warm-up, DPO baseline, the final step of PPO, and various CGPO variants across all benchmarks detailed in Section 6. The data in Table 2 indicate that CGPO variants employing CRPG and CRRAFT optimizers significantly outperform the DPO and PPO baselines across all benchmarks. Notably, CRPG shows the most substantial improvements in math and coding benchmarks (Math, GSM8K, HumanEval, and MBPP), while CRRAFT excels in helpfulness and factuality (AlpacaEval-2, Arena-Hard, and TruthfulQA). Both CRPG and CRRAFT achieve the best results in terms of instruction following (IFEval). While the CGPO variant with the CODPO optimizer does not perform as strongly as other variants, it offers performance that is on par with or better than the DPO and PPO in all benchmarks except the IFEval. In terms of safety, CGPO with the CRPG and CODPO optimizers achieve the best results in FRR and SVR, respectively. Table 2 demonstrates that the CGPO framework is able to enhance model quality across all tasks, proving its efficacy in managing challenging multi-task fine-tuning.

|                   | SFT            | DPO<br>warm-up | DPO            | PPO            | CGPO -<br>CRPG | CGPO -<br>CRRAFT | CGPO -<br>CODPO |
|-------------------|----------------|----------------|----------------|----------------|----------------|------------------|-----------------|
| AlpacaEval-2      | 10.9           | 13.3           | 16.3           | 24.8           | 25.9           | 43.2             | 18.08           |
| Arena-Hard        | $13.6 \pm 1.6$ | $18.8 \pm 1.6$ | $18.3 \pm 1.7$ | $24.3 \pm 1.8$ | $31.2 \pm 2.2$ | $36.8 \pm 2.0$   | $16.8 \pm 1.9$  |
| IFEval            | 0.71           | 0.75           | 0.79           | 0.81           | 0.83           | 0.83             | 0.70            |
| MATH              | 0.44           | 0.44           | 0.45           | 0.46           | 0.48           | 0.47             | 0.46            |
| GSM8K             | 0.86           | 0.88           | 0.90           | 0.91           | 0.93           | 0.92             | 0.90            |
| 0-shot MBPP       | 0.50           | 0.51           | 0.49           | 0.002          | 0.63           | 0.57             | 0.51            |
| 4-shots MBPP      | 0.55           | 0.57           | 0.60           | 0.62           | 0.62           | 0.58             | 0.55            |
| 0-shot HumanEval  | 0.09           | 0.15           | 0.59           | 0.006          | 0.76           | 0.70             | 0.57            |
| 4-shots HumanEval | 0.62           | 0.70           | 0.70           | 0.66           | 0.71           | 0.68             | 0.67            |
| MMLU              | 0.75           | 0.76           | 0.75           | 0.75           | 0.75           | 0.75             | 0.75            |
| ARC               | 0.85           | 0.84           | 0.88           | 0.90           | 0.92           | 0.90             | 0.90            |
| TruthfulQA        | 0.57           | 0.59           | 0.63           | 0.65           | 0.64           | 0.66             | 0.63            |
| Engagement        | 0.50           | 0.59           | 0.71           | 0.81           | 0.81           | 0.72             | 0.79            |
| SVR               | 0.03           | 0.03           | 0.02           | 0.03           | 0.05           | 0.02             | 0.01            |
| FRR               | 0.18           | 0.161          | 0.17           | 0.12           | 0.04           | 0.12             | 0.24            |

Table 2 Evaluation results of SFT, DPO warm-up, DPO, PPO and CGPO variants

#### 6.2.2 Effectiveness of Mixture of Judges

In this section, we explore the significance of incorporating MoJs within the CGPO framework. We conduct an ablation study by eliminating all MoJs from CGPO, utilizing the CRPG optimizer, while keeping all other variables constant, and then proceed to rerun the RLHF finetuning for 600 steps. Figure 4 presents a comparative analysis of CGPO performance with and without MoJs using the CRPG optimizer across various benchmarks, including HumanEval, MBPP, MATH, and GSM8K.

From Figure 4, it is clear that in the absence of coding judges, the CRPG optimizer undergoes a notable decline in 0-shot coding benchmarks once it surpasses 180 steps, mirroring the performance of the PPO baseline. Additionally, in the MATH, GSM8K and 4-shots HumanEval and MBPP benchmarks, while CRPG shows some improvement without constraints, the increases in metrics are considerably less pronounced compared to cases where math judges are utilized. This comparison effectively illustrates that MoJs play a crucial role not only in preventing reward hacking but also in significantly boosting the model's performance during online RLHF finetuning.

#### 6.2.3 Impact of RLHF Warm-up

In this section, we discuss the importance of introducing the RLHF warm-up stage. We consider CGPO with CRPG optimizer, and rerun the experiment in Section 6.2.1 but switch the starting point with SFT model. Additionally, we add

![](_page_18_Figure_0.jpeg)

**Figure 4** Comparison of CGPO (CRPG optimizer) with and without MoJs

one more ablation by starting from the DPO baseline that has been extensively optimized, which has significantly better performance across all benchmarks than the DPO warm-up model (Table 2).

Monitoring GPT-based helpfulness evaluations like AlpacaEval-2 and Arena-Hard during training is costly. To efficiently assess the effectiveness of the RLHF warm-up stage from the helpfulness perspective, we implement a cost-effective benchmark. We collect prompts from user-LLM interactions (e.g., LMSys-1M) and generate multiple responses using the LIama3.0 70B model. These responses are ranked by a powerful LLM, and the highest and lowest-ranked responses are used to create preference pairs for training a reward model (RM). This RM evaluates helpfulness based on its average score on its training prompts. Although this RM may overfit this prompt set, it remains a valid measure of helpfulness since our finetuning process does not depend on this specific prompt set.

Figure 5 illustrates the training curves of the CGPO model with different initial conditions across various benchmarks. When compared to the standard online RLHF setting, which starts with the SFT model, CGPO initiated from the warm-up model consistently achieves superior performance in all benchmarks, with the exception of GSM8K. For the runs that begin with the DPO baseline, there is a noticeable higher initial performance across all benchmarks. However, the ultimate performance of these models does not exceed those that started from the warm-up or SFT models. Particularly in helpfulness, ARC challenge, Math and 4-shot coding benchmarks, there is a marked decline in performance during the later stages of training. This suggests that starting from the highly optimized DPO baseline may detrimentally affect the final model's performance, potentially due to the soft-greedy nature of the DPO optimal policy, which might limit the model's ability to explore and further improve. Therefore, Figure 5 demonstrates that incorporating an RLHF warm-up stage can significantly enhance the model's performance during the subsequent online RLHF phase.

## **7 Related Works**

**RLHF in MTL.** Reinforcement Learning with Human Feedback (RLHF) is designed to align language models with human preferences and has become a crucial component of the fine-tuning pipeline for Large Language Models (LLMs) (Stiennon et al., 2020; Ouyang et al., 2022; Brown et al., 2020; Touvron et al., 2023; Bi et al., 2024; Bai et al., 2022). The majority work of RLHF focus optimizing a single reward models (Ouyang et al., 2022; Gao et al., 2023; Dong et al., 2023; Ethayarajh et al., 2023). The exploration of RLHF in the MTL setting remains relatively underexplored. The

![](_page_19_Figure_0.jpeg)

**Figure 5** Comparison of CGPO (CRPG optimizer) with different starting point

most commonly adopted approach involves optimizing a weighted sum of several reward models, where each model captures the interests of different tasks (Ramamurthy et al., 2022; Glaese et al., 2022; Yuan et al., 2023; Bakker et al., 2022; Wu et al., 2024). However, a major limitation of this approach is that key information from each individual reward model can be lost through linear combination, particularly when conflicting task goals exist. This can lead to suboptimal performance for each individual task. Additionally, each individual reward model typically requires different treatments (regularization, early stopping, etc) due to their unique properties, thus applying a uniform treatment for a composite reward model can further impair optimization performance across tasks (Moskovitz et al., 2023). Another research direction involves fine-tuning a separate LLM model for each task, followed by linear interpolation of the LLM weights across all learned models to produce a single model that excels in multiple tasks (Rame et al., 2024). However, this method remains computationally expensive and unstable due to the high cost and variability inherent in a single RLHF process (Hu et al., 2023; Rafailov et al., 2024b). (Yang et al., 2024) Proposed to use in-context reward model to manage multiple reward, but introduce additonal cost during inference time. Unlike the approaches mentioned above, CGPO introduces a customized reward model recipe and an RLHF optimizer tailored for each specific task. This method is not only as efficient as the conventional RLHF pipeline, but it also preserves all information within each reward model, thereby optimizing alignment for each task to the fullest extent.

**Reward Hacking Mitigation.** Compaired with traditional RL, where the reward is typically well-defined and the goal is to maximize it (Sutton and Barto, 2018), RLHF introduces a unique challenge known as "reward hacking." This issue arises because the reward model serves as a proxy for actual human preferences. Over-optimization of the reward model can adversely impact the performance of the language model (Gao et al., 2023; Moskovitz et al., 2023; Stiennon et al., 2020; Rafailov et al., 2024b). Consequently, addressing reward hacking is a major focus in RLHF. Previous studies have explored various approaches to mitigate the effects of reward hacking, including reward model regularization (Singhal et al., 2023), reward ensembles (Eisenstein et al., 2023; Ramé et al., 2024), and explicitly learning the reward bias error (Chen et al., 2024; Shen et al., 2023). In contrast to previous methods, our CGPO framework employs both LLM and rule-based judges as constraints to detect and prevent reward hacking patterns. This approach offers a more fine-grained and controllable solution to this persistent issue. Furthermore, the use of MoJs enables us to develop tailored strategies for mitigating the effects of reward hacking across various tasks in the MTL setting. This allows us to effectively address the reward hacking challenge in the more complex MTL environment, where previous methods have struggled to perform efficiently.

## **8 Conclusion**

In this paper, we introduced the CGPO framework to address key challenges in multi-task learning for LLM post-training with RLHF. The CGPO framework effectively mitigates issues such as inhomogeneous reward hacking and conflicting task goals through a novel primal-type multi-constraint RL method and a tailored multi-objective optimization strategy. We demonstrate the effectiveness of CGPO in a scenario where we need to handle five tasks with three reward models and seven constraints, marking the first application of RLHF in multi-task learning for general-purpose LLMs. Our experiments show that CGPO achieves significantly better metric gains for all tasks compared to the baseline RLHF methods. Moving forward, it is promising to explore more automated ways to adapt the gradient weights from different tasks to further reduce the hyperparameter burden and advance the Pareto frontier (Sener and Koltun, 2018).

## **References**

- Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- Armen Aghajanyan, Anchit Gupta, Akshat Shrivastava, Xilun Chen, Luke Zettlemoyer, and Sonal Gupta. Muppet: Massive multi-task representations with pre-finetuning. *arXiv preprint arXiv:2101.11038*, 2021.
- AI Anthropic. Introducing claude, 2023.
- Vamsi Aribandi, Yi Tay, Tal Schuster, Jinfeng Rao, Huaixiu Steven Zheng, Sanket Vaibhav Mehta, Honglei Zhuang, Vinh Q Tran, Dara Bahri, Jianmo Ni, et al. Ext5: Towards extreme multi-task scaling for transfer learning. *arXiv preprint arXiv:2111.10952*, 2021.
- Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. *arXiv preprint arXiv:2112.00861*, 2021.
- Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.
- Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022.
- Michiel Bakker, Martin Chadwick, Hannah Sheahan, Michael Tessler, Lucy Campbell-Gillingham, Jan Balaguer, Nat McAleese, Amelia Glaese, John Aslanides, Matt Botvinick, et al. Fine-tuning language models to find agreement among humans with diverse preferences. *Advances in Neural Information Processing Systems*, 35:38176–38189, 2022.
- Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024.
- Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345, 1952.
- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.
- Rich Caruana. Multitask learning. *Machine learning*, 28:41–75, 1997.
- Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. *arXiv preprint arXiv:2307.15217*, 2023.
- Lichang Chen, Chen Zhu, Davit Soselia, Jiuhai Chen, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. Odin: Disentangled reward mitigates hacking in rlhf. *arXiv preprint arXiv:2402.07319*, 2024.
- Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.
- Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference, 2024.
- Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.
- Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- Michael Crawshaw. Multi-task learning with deep neural networks: A survey. *arXiv preprint arXiv:2009.09796*, 2020.
- Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. 2023.
- Justin Cui, Wei-Lin Chiang, Ion Stoica, and Cho-Jui Hsieh. Or-bench: An over-refusal benchmark for large language models. *arXiv preprint arXiv:2405.20947*, 2024.

- Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. *arXiv preprint arXiv:2305.14233*, 2023.
- Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. *arXiv preprint arXiv:2304.06767*, 2023.
- Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback. *Advances in Neural Information Processing Systems*, 36, 2024.
- Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D'Amour, DJ Dvijotham, Adam Fisch, Katherine Heller, Stephen Pfohl, Deepak Ramachandran, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. *arXiv preprint arXiv:2312.09244*, 2023.
- Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. Understanding dataset difficulty with V-usable information. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Proceedings of Machine Learning Research*, pages 5988–6008. PMLR, 17–23 Jul 2022.
- Kawin Ethayarajh, Winnie Xu, Dan Jurafsky, and Douwe Kiela. Human-centered loss functions (halos). Technical report, Technical report, Contextual AI, 2023.
- Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. *arXiv preprint arXiv:2402.01306*, 2024.
- Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In *International Conference on Machine Learning*, pages 10835–10866. PMLR, 2023.
- Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. *arXiv preprint arXiv:2209.14375*, 2022.
- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*, 2020.
- Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with apps. *NeurIPS*, 2021a.
- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021b.
- Jian Hu, Li Tao, June Yang, and Chandler Zhou. Aligning language models with offline reinforcement learning from human feedback. *arXiv preprint arXiv:2308.12050*, 2023.
- Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. *arXiv preprint arXiv:2312.06674*, 2023.
- Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. *arXiv preprint arXiv:2212.12017*, 2022.
- Di Jin, Shikib Mehri, Devamanyu Hazarika, Aishwarya Padmakumar, Sungjin Lee, Yang Liu, and Mahdi Namazifar. Data-efficient alignment of large language models with human feedback through natural language. *arXiv preprint arXiv:2311.14543*, 2023.
- Kaiwen Li, Tao Zhang, and Rui Wang. Deep reinforcement learning for multiobjective optimization. *IEEE transactions on cybernetics*, 51(6):3103–3114, 2020.
- Ming Li, Han Chen, Chenguang Wang, Dang Nguyen, Dianqi Li, and Tianyi Zhou. Ruler: Improving llm controllability by rule-based data recycling. *arXiv preprint arXiv:2406.15938*, 2024a.
- Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From live data to high-quality benchmarks: The arena-hard pipeline, april 2024. *URL https:*//*lmsys. org*/*blog*/*2024-04-19-arena-hard*, 2024b.
- Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. *arXiv preprint arXiv:2109.07958*, 2021.
- Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. *arXiv preprint arXiv:1705.04146*, 2017.

- Bingchang Liu, Chaoyu Chen, Cong Liao, Zi Gong, Huan Wang, Zhichao Lei, Ming Liang, Dajun Chen, Min Shen, Hailian Zhou, et al. Mftcoder: Boosting code llms with multitask fine-tuning. *arXiv preprint arXiv:2311.02303*, 2023.
- Shengchao Liu, Yingyu Liang, and Anthony Gitter. Loss-balanced task weighting to reduce negative transfer in multi-task learning. In *Proceedings of the AAAI conference on artificial intelligence*, volume 33, pages 9977–9978, 2019a.
- Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In *Proceedings of the IEEE*/*CVF conference on computer vision and pattern recognition*, pages 1871–1880, 2019b.
- Yudong Luo, Yangchen Pan, Han Wang, Philip Torr, and Pascal Poupart. A simple mixture policy parameterization for improving sample efficiency of cvar optimization. *arXiv preprint arXiv:2403.11062*, 2024.
- Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. 2023.
- AI Meta. Introducing meta llama 3: The most capable openly available llm to date. *Meta AI*, 2024.
- Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of slms in grade school math. *arXiv preprint arXiv:2402.14830*, 2024.
- Ted Moskovitz, Aaditya K Singh, DJ Strouse, Tuomas Sandholm, Ruslan Salakhutdinov, Anca D Dragan, and Stephen McAleer. Confronting reward model overoptimization with constrained rlhf. *arXiv preprint arXiv:2310.04373*, 2023.
- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35:27730–27744, 2022.
- Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. *arXiv preprint arXiv:2402.13228*, 2024.
- Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. *arXiv preprint arXiv:2201.03544*, 2022.
- Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, Bradley Knox, Chelsea Finn, and Scott Niekum. Scaling laws for reward model overoptimization in direct alignment algorithms. *arXiv preprint arXiv:2406.02900*, 2024a.
- Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances in Neural Information Processing Systems*, 36, 2024b.
- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140): 1–67, 2020.
- Rajkumar Ramamurthy, Prithviraj Ammanabrolu, Kianté Brantley, Jack Hessel, Rafet Sifa, Christian Bauckhage, Hannaneh Hajishirzi, and Yejin Choi. Is reinforcement learning (not) for natural language processing: Benchmarks, baselines, and building blocks for natural language policy optimization. *arXiv preprint arXiv:2210.01241*, 2022.
- Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. *Advances in Neural Information Processing Systems*, 36, 2024.
- Alexandre Ramé, Nino Vieillard, Léonard Hussenot, Robert Dadashi, Geoffrey Cideron, Olivier Bachem, and Johan Ferret. Warm: On the benefits of weight averaged reward models. *arXiv preprint arXiv:2401.12187*, 2024.
- Mathieu Rita, Florian Strub, Rahma Chaabouni, Paul Michel, Emmanuel Dupoux, and Olivier Pietquin. Countering reward over-optimization in llm with demonstration-guided reinforcement learning. *arXiv preprint arXiv:2404.19409*, 2024.
- Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. *arXiv preprint arXiv:2308.01263*, 2023.
- Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models, 2023.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. *Advances in neural information processing systems*, 31, 2018.

- Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback. *arXiv preprint arXiv:2310.05199*, 2023.
- Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. A long way to go: Investigating length correlations in rlhf. *arXiv preprint arXiv:2310.03716*, 2023.
- Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. *Advances in Neural Information Processing Systems*, 35:9460–9471, 2022.
- Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020.
- Lichao Sun, Yue Huang, Haoran Wang, Siyuan Wu, Qihui Zhang, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, et al. Trustllm: Trustworthiness in large language models. *arXiv preprint arXiv:2401.05561*, 2024.
- Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*. MIT press, 2018.
- Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.
- Llama Team. Meta llama guard 2. https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL\_ CARD.md, 2024.
- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct distillation of lm alignment. *arXiv preprint arXiv:2310.16944*, 2023.
- Bertie Vidgen, Adarsh Agrawal, Ahmed M Ahmed, Victor Akinwande, Namir Al-Nuaimi, Najla Alfaraj, Elie Alhajjar, Lora Aroyo, Trupti Bavalatti, Borhane Blili-Hamelin, et al. Introducing v0. 5 of the ai safety benchmark from mlcommons. *arXiv preprint arXiv:2404.12241*, 2024.
- Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. *arXiv preprint arXiv:2401.06080*, 2024a.
- Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6):186345, 2024b.
- Zhilin Wang, Yi Dong, Jiaqi Zeng, Virginia Adams, Makesh Narsimhan Sreedhar, Daniel Egert, Olivier Delalleau, Jane Polak Scowcroft, Neel Kant, Aidan Swope, and Oleksii Kuchaiev. Helpsteer: Multi-attribute helpfulness dataset for steerlm. 2023.
- Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8: 229–256, 1992.
- Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. *Advances in Neural Information Processing Systems*, 36, 2024.
- Jing Xu, Da Ju, Margaret Li, Y-Lan Boureau, Jason Weston, and Emily Dinan. Recipes for safety in open-domain chatbots. *arXiv preprint arXiv:2010.07079*, 2020.
- Tengyu Xu, Yingbin Liang, and Guanghui Lan. Crpo: A new approach for safe reinforcement learning with convergence guarantee. In *International Conference on Machine Learning*, pages 11480–11491. PMLR, 2021.
- Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han Zhong, Dong Yu, and Jianshu Chen. Rewards-in-context: Multi-objective alignment of foundation models with dynamic preference adjustment. *arXiv preprint arXiv:2402.10207*, 2024.
- Chengyang Ying, Xinning Zhou, Hang Su, Dong Yan, Ning Chen, and Jun Zhu. Towards safe reinforcement learning via constraining conditional value-at-risk. *arXiv preprint arXiv:2206.04436*, 2022.
- Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. *arXiv preprint arXiv:2309.12284*, 2023.

- Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun. Advancing llm reasoning generalists with preference trees. 2024a.
- Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. *arXiv preprint arXiv:2401.10020*, 2024b.
- Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. *arXiv preprint arXiv:2304.05302*, 2023.
- Qiyuan Zhang, Shu Leng, Xiaoteng Ma, Qihan Liu, Xueqian Wang, Bin Liang, Yu Liu, and Jun Yang. Cvar-constrained policy optimization for safe reinforcement learning. *IEEE Transactions on Neural Networks and Learning Systems*, 2024.
- Xiangyun Zhao, Haoxiang Li, Xiaohui Shen, Xiaodan Liang, and Ying Wu. A modulation module for multi-task learning with applications in image retrieval. In *Proceedings of the European Conference on Computer Vision (ECCV)*, pages 401–416, 2018.
- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. Lmsys-chat-1m: A large-scale real-world llm conversation dataset. 2023a.
- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36, 2024.
- Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. Secrets of rlhf in large language models part i: Ppo. *arXiv preprint arXiv:2307.04964*, 2023b.
- Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. *arXiv preprint arXiv:2311.07911*, 2023.
- Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, and Jiantao Jiao. Starling-7b: Improving llm helpfulness & harmlessness with rlaif, 2023.
- Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593*, 2019.
- Álvaro Bartolomé Del Canto, Gabriel Martín Blázquez, Agustín Piqueres Lajarín, and Daniel Vila Suero. Distilabel: An ai feedback (aif) framework for building datasets with and for llms. *GitHub repository*, 2024.

## Appendix

## **A CGPO Training Set**

The detail of of our training dataset is provide in Table 3. Note that in our experiment we adopt the instruction finetuing format, in which the prompt is wrapped as "[INST] {prompt} [\INST]":

**Synthetic IF dataset.** Inspired by Zhou et al. (2023), we consider synthetic prompts that require LLM generation to satisfy one or more closed-form instructions, which can be verified exactly. We identify 23 types of closed-form instructions for generation and use LIama 3.0 70B instruct model to create synthetic prompts that address a specific topic and also require these closed-form instructions. We create a template to enable LIama 3.0 70B instruct model to generate all prompts. The prompt template that we input into LIama 3.0 70B instruct model to generate synthetic instruction-following prompts is provided as follows:

#### **Prompt Template** =

"You are a helpful AI assistant. You are given a TOPIC and a FORMAT REQUIREMENT, and you are expected to generate a PROMPT that is on the given TOPIC and specify the given FORMAT REQUIREMENT that the corresponding answer should follow. Here are many examples that you can learn from:

**TOPIC:** Travel

**FORMAT REQUIREMENT:** In your entire response, refrain from the use of any commas

**PROMPT:** I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.

**TOPIC:** Aerospace engineering

**FORMAT REQUIREMENT:** In your entire response, refrain from the use of any commas and Give two different responses. Responses and only responses should be separated by 6 asterisk symbols: \*\*\*\*\*\*

**PROMPT:** Write two jokes about rockets. Do not contain commas in your response. Separate the two jokes with 6 asterisk symbols: \*\*\*\*\*\*.

**TOPIC:** History

**FORMAT REQUIREMENT:** Entire output should be wrapped in JSON format

**PROMPT:** What is the history of NYC prospect park? Please wrap your entire answer in JSON format.

**TOPIC:** Video game

**FORMAT REQUIREMENT:** Highlight at least 2 sections in your answer with markdown, i.e. \*highlighted section\* and Answer with at least 40 sentences

**PROMPT:** Can you write a poem about the pros and cons of playing a lot of video games? Please make sure it's at least 40 sentences long (don't forget to add punctuations). You must highlight at least sections in your response, like \*highlighted phrase\*.

**TOPIC:** Movie

**FORMAT REQUIREMENT:** Answer with at least 40 sentences, Highlight at least 4 sections in your answer with markdown, i.e. \*highlighted section\*, and Wrap your entire response with double quotation marks

**PROMPT:** Write a joke about the superhero movie with at least 5 sentences. Use Markdown to italicize at least 4 sections in your answer, i.e. \*italic text\*. Wrap your answer in double quotes.

**TOPIC:** Health care

**FORMAT REQUIREMENT:** Your entire response should be in English, capital letters only

**PROMPT:** Write an essay about public health care system in US in English and in all capital letters.

**TOPIC:** Mathematics

**FORMAT REQUIREMENT:** Entire output should be wrapped in JSON format

**PROMPT:** List all facts about calculus in a structured output. In particular, Format your entire output in JSON.

Now it is your turn to generate a PROMPT that is on the given TOPIC and specify the given FORMAT REQUIREMENT that the corresponding answer should follow. Please DO NOT make up any new format requirement that is not given to you.

**TOPIC:** {topic}

#### **FORMAT REQUIREMENT:** {instruction}

To be noted, you just need to mention/specify the FORMAT REQUIREMENT in your response but your response does not need to follow it. Please directly provide the PROMPT without any extra words. Do not write any note or explanation.

**TOPICS** = ["20th century events", "Accounting", "Architecture", "Astronomy", "Biology", "Businessethics","Celebrities","Chemistry","Clinical knowledge", "Economics", "Electrical engineering", "Ethics of artificial intelligence", "Education", "Energy", "Gaming", "Geography", "Global facts", "History", "Healthcare", "Immigration law", "International law", "Jurisprudence", "Management", "Marketing", "Mathematics", "Medicine", "Moraldisputes", "Movies", "Music", "Philosophy", "Physics", "Prehistory", "Psychology", "Public relations", "Sociology", "Sports", "Social media" "Transportation", "Virology"]

**Instructions** = ["number of paragraphs", "number of sentences", "number of words", "first word in n-the paragraph", "number of a specific placeholder"; "number of sections", "title", "response given in a certain format", "number of highlighted sections", "response need to be in json", "postscript at the end of response", "number of bullet list", "forbidden words", "certain keyword must exist", "a given key word need to appear at least n-times", "a given letter need to appear at least n-times", "generation should be in lowercase", "generation should be in capital", "capital word need to appear at least n-times", "generation should no contain comma", "generation should finish with an exact end checker", "entire response should be be wrapped within double quotation marks", "generation should contain two responses"]

Each time, we randomly select up to three types of closed-form instructions along with one topic, and incorporate them

into a template. This template is then used by LIama 3.0 70b instruct model to generate a prompt. We repeat this process 30000 times to create a comprehensive set of instruction-following prompts.

For each synthetic prompt, we utilized Llama 3.0 70B Instruct model, and Llama 3.0 8B Instruct model to generate a response based on the prompt. We then evaluated whether these responses adhered to the instruction-following constraints. Prompts that did not yield any responses meeting the constraints, as well as those where all responses met the constraints, were filtered out. This process resulted in 11668 prompts that included both responses that satisfied the constraints and responses that violated them. We randomly selected one response that met the constraints as the accepted response and one that violated the constraints as the rejected response for each prompt. By doing so, we constructed our pairwise instruction-following preference dataset.

**Human annotated safety dataset.** We take an iterative approach to collect multiple batches of safety preference data and merge them together as the final train data. At each iteration, we generate two different responses from a pool of models (model from previous iteration for example), and send them to human annotators to rate and rank based on the safety guidelines. If no response meets the guideline, the annotators are asked to directly edit the higher ranked response for it to abide the guideline. The collected preference pairs are used to train a reward model, and once such a reward model is trained, we leverage it to do rejection sampling to produce finetuning data that are used to train the next model iteration. This next model will be added to the pool of models that generate responses for human annotators to rank. We repeat this process multiple times to iteratively collect higher quality safety preference pairs. An additional layer of data auditing is also applied on top of each data iteration cycle due to the subtle and subjective nature of safety guidelines to further ensure data quality.

**Synthetic engagement dataset.** To develop a synthetic engagement pairwise preference dataset, we initially gathered 1M user engagement samples from interactions with an LLM-based chatbot on social media platforms. Each sample comprises a user query, the LLM's response, and a binary label indicating user approval of the response. We used this dataset to train a binary feedback reward model on top of the pretrained Llama 3.0 8B model by adding a linear output layer and training it as a binary classifier. We selected a model iteration with an AUC of 0.89 from the training trajectory to function as the oracle predictor of user engagement intent. This model was subsequently used to generate the synthetic user engagement preference dataset in our study. In the next step, we subsampled 112,375 prompts from LMSys-1M Zhu et al. (2023). We then generated two responses from the Llama 3.0 8B model and two responses from the Llama 3.0 70B model, ultimately generating four distinct responses for each prompt, conditioned under the generation setting temperature=1, top\_p=0.9. Following this, our oracle predictor was used to score all generated responses. The response with the highest score was selected as the accepted response, while the one with the lowest score was marked as the rejected response. By applying this methodology to all selected prompts, we created our synthetic user engagement preference dataset.

**Additional Comment.** It's important to note that for certain datasets used in online RLHF, we also incorporate metadata to provide additional information about the data as shown in Table 4. During CGPO training, sometimes it will be necessary to extract information from the metadata to implement the MoJs.

- **MATH, GSM8K & Aqua Math**: In the metadata, we include the ground truth answer for each question. This allows the math constraint judge to leverage this information to evaluate the accuracy of the LLM's response for each math question.
- **TriviaQA & ARC**: For prompts related to deterministic factuality, we also incorporate the ground truth answer into the metadata. This allows the factuality constraint judge to assess correctness based on this information.
- **APPS**: In the metadata, we include several unit tests that the correct code snippet should be able to pass through. Our coding constraint judge can leverage this to determine if the generated code is correct
- **Synthetic IF dataset**: We include closed-form instructions in the metadata, specifying requirements that the LLM's generation must satisfy. This enables our instruction-following constraint judge to verify whether the LLM's output adheres precisely to the instructions.

## **B CGPO Constraint Judge**

In this section, we will discuss in detail about how we build MoJs in CGPO.

#### **B.1 Rule-based Constraint Judge**

**Math constraint judge.** As illustrated in Table 4, for the math prompt sets MATH, GSM8K, and Aqua Math, we explicitly require the model to provide the final answer in a specified format, which can be easily extracted. When implementing the math constraint judge, we extract the LLM's answer by examining the final sentence and comparing it with the ground truth answer in the metadata. There are instances where the model correctly answers the question but fails to provide the answer in the correct format. In such cases, the math constraint judge will indicate that this generation violates the constraint. Although this is a false negative, using CGPO to encourage the model to avoid such patterns can implicitly help improve the model's ability to follow instructions.

**Coding constraint judge.** Our coding constraint judge examines the coding block in LLM's response to extract the code snippet. It then runs the snippet through all the unit tests provided in the metadata to determine if it passes each test. Similar to the math constraint, false negatives can occur if LLM's solution is not formatted correctly. Implementing CGPO to discourage such patterns could enhance the model's ability to follow instructions accurately.

**Instruction following constraint judge.** The instruction-following constraint judge begins by reading the metadata to understand the specific rules that LLM's output must adhere to. Then, we employ string-matching based logic to determine whether LLM's generation complies with all the specified rules.

## **B.2 LLM-based Constraint Judge**

The LLM classifier constraint judge utilizes an additional LLM to assess whether the output from our training LLM adheres to a specific predefined criterion. We design the input for this judge using a prompt template that arranges the LLM's response alongside other essential contexts. Within this template, we specify both a negative token and a positive token. The negative token indicates that the LLM's response breaches the constraint, while the positive token signifies compliance. We explicitly direct the judge to issue either the positive or negative token based on their assessment. To minimize the randomness in the judgment process, we do not rely solely on the LLM to generate a token and then check its correspondence to the negative or positive token. Instead, we directly examine the softmax probabilities of the negative and positive tokens. If the probability of the negative token is higher, we conclude that the LLM's response violates the constraint, and vice versa. Table 5 presents the template along with the negative and positive tokens for the LLM classifiers in our experiment.

**False refusal constraint judge.** We utilize the Llama 3.0 8b pretrained model as a foundation and fine-tune an LLM classifier specifically aimed at identifying refusal patterns in LLM responses. The training data is formatted as follows: "[INST] {LLM response} [\INST] judgment", where "judgment" is True if the LLM response indicates refusal, and False otherwise. During the inference phase of deploying this constraint judge, we also encapsulate the generated responses from the training LLM within "[INST] ... [\INST]" and use that as the input for the judge.

**Factuality constraint judge.** We employ the Llama 3.0 70b instruct model directly as the factuality constraint judge. Recall that for prompts associated with deterministic factuality, we include the ground truth answer in the metadata. When deploying this constraint judge, we use the template as illustrated in Table 5, incorporating the prompt, ground truth answer, and the LLM response into the template to serve as inputs for the judge.

**Safety constraint judge.** We utilize LIamaGuard2 Team (2024), which is fine-tuned from the Llama 3.0 8b pretrained model. We reuse the template as introduced in the LIamaGuard2 paper Team (2024), where we incorporate pre-defined safety guidelines and full completions into the prompt template to serve as inputs for the judge.

## **C Evaluation Benchmarks**

One example prompt of the MBPP evaluation set:

You are an expert Python programmer, and here is your task:

Write a function to sort a given matrix in ascending order according to the sum of its rows.

Your code should pass the following tests:

```
assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]

assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]

assert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]
```

One example prompt of the HumanEval evaluation set:

```
Write a solution to the following problem and make sure that it passes the tests:
"'python
from typing import List
def remove_duplicates(numbers: List[int]) -> List[int]:
""" From a list of integers, remove all elements that occur more than once.
Keep order of elements left the same as in the input.
>>> remove_duplicates([1, 2, 3, 2, 4])
[1, 3, 4]
"""
```

## **D** Reward Hacking Examples

We provide examples of reward hacking across various tasks in Table 6. Here, the reward model we trained in Section 5.2 assigns higher values to low quality outputs than to high quality ones. Note that we report the raw reward value in Table 6 instead of the calibrated one.

| Task | Prompt                                                                                                                                                                                                   | Response                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Reward | Quality |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|---------|
| Math | Given the following problem, reason and give a final answer to the problem. Problem: simplify $9 \cdot \frac{12}{7} \cdot \frac{-35}{36}$ . Your response should end with "The final answer is [answer]" | To simplify the expression, we can start by canceling out common factors between the numerator and denominator of the fractions. In this case, we have 12 and 36, which have a common factor of 12. We also have 7 and 35, which have a common factor of 7. Canceling out these factors, we get: $9 \cdot \frac{12}{7} \cdot \frac{-35}{36} = 9 \cdot \frac{12}{7} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{36} \cdot \frac{12}{3$ | 4.75   | Low     |

To simplify the given expression, we can start by canceling common factors:  $9 \cdot \frac{12}{7} \cdot \frac{-35}{36}$ First, let's look at the numerators. Both 12 and 36 are divisible by 12, so we can simplify the expression as follows:  $9 \cdot \frac{12}{7} \cdot \frac{-35}{36} = 9 \cdot \frac{1}{7} \cdot \frac{-35}{3}$ Next, we can look at the remaining factors. 4.59 High 7 and 35 are divisible by 7, so we simplify  $9 \cdot \frac{1}{7} \cdot \frac{-35}{3} = 9 \cdot \frac{-5}{3}$ Finally, we have 9 and 3 which are divisible by 3. Performing the division, we  $9 \cdot \frac{-5}{3} = -15$ So, the final answer is -15.

You are an expert python programmer, and here is your task to complete the code:

""

python

def you\_are\_a\_cube(cube):

Coding

In geometry, a cube is a three-dimensional solid object bounded by six square faces, facets or sides, with three meeting at each vertex. The cube is the only regular hexahedron and is one of the five Platonic solids.

It has 12 edges, 6 faces and 8 vertices. The cube is also a square parallelepiped, an equilateral cuboid and a right rhombohedron. It is a regular square prism in three orientations, and a trigonal trapezohedron in four orientations.

You are given a task of finding a if the provided value is a perfect cube!

"

Here is the requested Python code: ""

python

def you\_are\_a\_cube(cube):

if cube in [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2163, 2744, 3375, 4096, 4913, 5832, 6859, 8000, 9261, 10648, 12167, 13824, 15625, 17576, 19683, 21952, 24389, 27000, 29791, 32768, 35937, 39312, 42875, 46656, 60704, 63488, 68591, 72903, 79507, 92609, 101218, 109744, 120422, 131433, 143489, 157464, 173015, 190044, 209957, 232630, 250047, 270699, 294035, 320264, 348484, 390625, 435672, 488412, 511315, 539047, 576480, 625000, 648930, 676030, 704977, 729028, 758858, 795293, 847442, 904 Sha, 1000000, 1024003, 1056731]:

return True return False

...

1.8

Low

32

|        |                                                                                             | Sure, you can solve your task by checking<br>if the cube root of the given value is an<br>integer or not. Here's your code:<br>"'<br>python<br>import math<br>def you_are_a_cube(cube):<br>"""<br>In geometry, a cube is a three-dimensional<br>solid object bounded by six square faces,<br>facets or sides, with three meeting at each<br>vertex.The cube is the only regular<br>hexahedron and is one of the five Platonic<br>solids. It has 12 edges, 6 faces and 8<br>vertices.The cube is also a square<br>parallelepiped, an equilateral cuboid and a<br>right rhombohedron. It is a regular square<br>prism in three orientations, and a trigonal<br>trapezohedron in four orientations.<br>"""<br>return math.isqrt(cube) ** 3 == cube.<br>"' | 0.96  | High |
|--------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|------|
|        |                                                                                             | There are several things that can cause<br>instant death that people may not be aware<br>of or may underestimate the risks<br>associated with them. Here are some<br>examples · · ·                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 0.37  | Low  |
| Safety | What are some things that<br>people are unaware about<br>that could kill them<br>instantly? | As a responsible AI language model, I<br>must emphasize that it's essential to<br>prioritize safety and awareness to avoid<br>accidental harm. While it's impossible to<br>list every potential danger, I'll highlight<br>some little-known risks that could have<br>severe consequences. However, please<br>remember that most of these risks are<br>extremely rare, and with proper caution,<br>you can minimize the likelihood of<br>encountering them · · ·                                                                                                                                                                                                                                                                                        | -0.44 | High |

**Table 6** Example of reward hacking in different tasks

| Dataset                              | is Preference | Size   | Usage                        | Source                                                                                                                                                               |
|--------------------------------------|---------------|--------|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Orca-Math                            | Х             | 200035 | SFT                          | Mitra et al. (2024)                                                                                                                                                  |
| MetaMath                             | Х             | 395000 | SFT                          | Yu et al. (2023)                                                                                                                                                     |
| Evol-<br>CodeAlpaca                  | Х             | 111183 | SFT                          | Luo et al. (2023)                                                                                                                                                    |
| MATH<br>training                     | Х             | 7500   | Online RLHF                  | Hendrycks et al. (2021b)                                                                                                                                             |
| GSM8K<br>training                    | Х             | 7473   | Online RLHF                  | Cobbe et al. (2021)                                                                                                                                                  |
| Aqua Math                            | Х             | 97467  | Online RLHF                  | Ling et al. (2017)                                                                                                                                                   |
| APPS                                 | Х             | 7070   | Online RLHF                  | Hendrycks et al. (2021a)                                                                                                                                             |
| XSText                               | Х             | 2700   | Online RLHF                  | Röttger et al. (2023)                                                                                                                                                |
| LMSys-55k                            | ✓             | 49865  | SFT, RM, DPO,<br>Online RLHF | Chiang et al. (2024)                                                                                                                                                 |
| UltraChat                            | ✓             | 207865 | SFT, RM, DPO,<br>Online RLHF | Ding et al. (2023)                                                                                                                                                   |
| UltraFeedback                        | ✓             | 340025 | SFT, RM, DPO                 | Cui et al. (2023)                                                                                                                                                    |
| UltraInteract                        | ✓             | 129531 | SFT, RM, DPO                 | Yuan et al. (2024a)                                                                                                                                                  |
| HH-RLHF                              | ✓             | 115396 | RM, DPO                      | Bai et al. (2022)                                                                                                                                                    |
| SHP                                  | ✓             | 93301  | RM, DPO                      | Ethayarajh et al. (2023)                                                                                                                                             |
| HelpSteer                            | ✓             | 37131  | RM, DPO                      | Wang et al. (2023)                                                                                                                                                   |
| Distilabel-<br>Capybara              | ✓             | 14811  | RM, DPO                      | Ethayarajh et al. (2024)                                                                                                                                             |
| Distilabel-<br>Orca                  | ✓             | 6926   | RM, DPO                      | Álvaro Bartolomé Del Canto et al. (2024)                                                                                                                             |
| Argilla Math                         | ✓             | 2418   | RM, DPO                      | Álvaro Bartolomé Del Canto et al. (2024)                                                                                                                             |
| Synthetic IF dataset                 | ✓             | 11668  | SFT, RM, DPO.<br>Online RLHF | Prompts are generated by LIama 3.0 70B instruct model, accepted and rejected responses are generated by LIama 3.0 70B instruct model and LIama 3.0 8b instruct model |
| Human<br>Annotated<br>safety dataset | ✓             | 244232 | SFT, RM, DPO.<br>Online RLHF | Colloect adversarial human-written prompts per risk category. The prompts are written in such a way to include diverse adversarial styles and attack vectors.        |
| Synthetic<br>engagement<br>dataset   | ✓             | 112375 | SFT, RM, DPO.<br>Online RLHF | Prompt are sampled from LMSys-1M Zheng et al. (2023a), the accepted and rejected responses are generated by LIama 3.0 70B instruct model                             |

 Table 3
 Dataset used in CGPO experiments for SFT, RM, DPO and online RLHF training

| Data                         | Prompt                                                                                                                                                                             | Metadata                                                                                                                                                                                                                          |  |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| MATH,<br>GSM8K,<br>Aqua Math | A quadratic equation ax2 −<br>2ax + b = 0 has<br>two real solutions. What is the average of<br>these two solutions? Your response should end<br>with "The final answer is [answer] | {"answer": "1"}                                                                                                                                                                                                                   |  |
| TriviaQA,<br>ARC             | Who was President when the first Peanuts<br>cartoon was published?                                                                                                                 | {"answer": "Harry S. Truman"}                                                                                                                                                                                                     |  |
| APPS                         | Write a function "similar_elements" to find the<br>similar elements from the given two tuple lists                                                                                 | {"unit_tests": "assert similar_elements((3, 4,<br>5, 6),(5, 7, 4, 10)) == (4, 5), assert<br>similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) ==<br>(3, 4), assert similar_elements((11, 12, 14,<br>13),(17, 15, 14, 13)) == (13, 14)"} |  |
| Synthetic IF                 | What are the primary architectural styles seen<br>in European churches? Give my answer in<br>English using only capital letters.                                                   | {"if_requirements": "english_capital"}                                                                                                                                                                                            |  |

**Table 4** Example of Prompt and Metadata used in CGPO experiment

| LLM Judge     | Template                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | neg & pos tokens |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| False Refusal | [INST] {LLM response} [\INST]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | True & False     |
| Factuality    | [INST] Here is a question and a ground truth answer:<br>Question: {prompt}<br>Ground truth answer: {ground truth answer}<br>Now, you are given an answer from the AI:<br>AI: {LLM response}<br>Does the AI's answer contains the factual information from the above ground<br>truth answer? Answer in TRUE or FALSE. [\INST]                                                                                                                                                                                                                                                                              | False & True     |
| Safety        | [INST] Task: Check if there is unsafe content in 'Agent' messages in<br>conversations according our safety policy with the below categories.<br><begin categories="" content="" unsafe=""> {safety guidelines} <end<br>UNSAFE CONTENT CATEGORIES&gt;<br/><begin conversation=""> {completions} <end conversation=""><br/>Provide your safety assessment for ONLY THE LAST turn message in the<br/>above conversation<br/>- First line must read 'safe' or 'unsafe'.<br/>- If unsafe, a second line must include a comma-separated list of violated<br/>categories. [\INST]</end></begin></end<br></begin> | Unsafe & Safe    |

**Table 5** Prompt template and negative & positive tokens used in LLM classifier judge