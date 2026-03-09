# Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents

Yifan Song<sup>2,3</sup> Da Yin<sup>4</sup> Xiang Yue<sup>5</sup> Jie Huang<sup>6</sup> Sujian Li<sup>2,3</sup> Bill Yuchen Lin<sup>1</sup>

<sup>1</sup>Allen Institute for AI <sup>2</sup>School of Computer Science, Peking University <sup>3</sup>National Key Laboratory for Multimedia Information Processing, Peking University <sup>4</sup>UCLA <sup>5</sup>Ohio State University <sup>6</sup>UIUC

{yfsong, lisujian}@pku.edu.cn yuchenl@allenai.org

#### **Abstract**

Large Language Models (LLMs) have become integral components in various autonomous agent systems. In this study, we present an exploration-based trajectory optimization approach, referred to as ETO. This learning method is designed to enhance the performance of open LLM agents. Contrary to previous studies that exclusively train on successful expert trajectories, our method allows agents to learn from their exploration failures. This leads to improved performance through an iterative optimization framework. During the exploration phase, the agent interacts with the environment while completing given tasks, gathering failure trajectories to create contrastive trajectory pairs. In the subsequent training phase, the agent utilizes these trajectory preference pairs to update its policy using contrastive learning methods like DPO (Rafailov et al., 2023). This iterative cycle of exploration and training fosters continued improvement in the agents. Our experiments on three complex tasks demonstrate that ETO consistently surpasses baseline performance by a large margin. Furthermore, an examination of task-solving efficiency and potential in scenarios lacking expert trajectory underscores the effectiveness of our approach.<sup>1</sup>

#### 1 Introduction

Large language models (LLMs) have demonstrated impressive capabilities in addressing complex interactive tasks through action planning for interactions with environments and tools (Wang et al., 2023a; Xi et al., 2023). Systems using ChatGPT (OpenAI, 2022) and GPT-4 (OpenAI, 2023) as principal controllers have been developed for a range of applications. These include web browsing (Deng et al., 2023; Zhou et al., 2023), embodied tasks (Yao et al., 2022b; Lin et al., 2023), multi-modal reasoning (Lu et al., 2023), and complex question answering. However, recent research suggests that

<span id="page-0-1"></span>![](_page_0_Picture_10.jpeg)

Figure 1: Exploration-based Trajectory Optimization (ETO) allows an LLM agent to iteratively collect failure trajectories and update its policy by learning from contrastive failure-success trajectory pairs.

open-source LLMs are considerably less effective than GPT-4 in constructing agents (Liu et al., 2023; Wang et al., 2023d; Mialon et al., 2023).

The standard approach to constructing open LLM agents involves imitation learning, which involves fine-tuning LLMs based on expert trajectories. Behavioral cloning (BC) (Pomerleau, 1991) is a simple yet effective imitation learning technique that derives a policy through supervised learning from observation-action pairs. Recent efforts (Chen et al., 2023; Zeng et al., 2023), including Agent LUMOS (Yin et al., 2023), have explored the use of BC to develop open LLM agents by implementing supervised fine-tuning (SFT) on expert trajectories. These methods employ the teacherforcing algorithm to train LLMs, enabling them to learn a policy for generating subsequent actions based on observations and past actions. However, these SFT methods, which rely entirely on expert demonstrations, may yield sub-optimal policies due to inadequate exploration of target environments, thereby limiting their generalizability.

The process of human learning not only involves observing successful demonstrations but

<span id="page-0-0"></span><sup>&</sup>lt;sup>1</sup>Code & Data: https://github.com/Yifan-Song793/ETO.

also includes experiencing and exploring failures through trial-and-error interactions with the environment. Drawing inspiration from this, we propose a novel learning approach for LLM agents, which we call Exploration-based Trajectory Optimization (ETO). Unlike previous methods that solely rely on successful trajectories, our approach capitalizes on the exploration failures of the current policy to enhance the learning process of an agent.

In particular, we first use SFT-based behavioral cloning to construct a base agent, as depicted in Figure [1.](#page-0-1) During the *exploration* phase, this base agent interacts with the target environment to undertake a set of given tasks and receive feedback from the environment. We gather failed trajectories from the base agents and pair them with expert trajectories previously collected for those tasks. Subsequently, we apply the DPO loss [\(Rafailov](#page-9-0) [et al.,](#page-9-0) [2023\)](#page-9-0) to fine-tune the LLM policy with these contrastive trajectory pairs, thereby continually improving the agent. The ETO can be expanded to multiple rounds by collecting failure cases from previously ETO-tuned agents.

We assessed our approach using three representative datasets: *WebShop* [\(Yao et al.,](#page-10-6) [2022a\)](#page-10-6) for web navigation, *ScienceWorld* [\(Wang et al.,](#page-10-7) [2022\)](#page-10-7) for simulated science experiments, and *ALF-World* [\(Shridhar et al.,](#page-9-9) [2021\)](#page-9-9) for embodied household tasks. Across these datasets, our ETO consistently exceeded the performance of SFT behavioral cloning and other robust baselines by a significant margin, thereby demonstrating the effectiveness of learning from exploration failures. Furthermore, our approach demonstrated an impressive performance improvement of 22% over SFT on the challenging out-of-distribution test set in ScienceWorld, indicating its strong generalization capability. Our analysis also highlighted the task-solving efficiency of our method, as it achieves higher rewards with fewer action steps. In extreme scenarios where expert trajectories are not available, our ETO still delivers impressive performance in a self-play mode, further emphasizing the potential of our approach.

In this paper, our contributions are as follows:

- Method. We introduce exploration-based trajectory optimization, ETO, a learning algorithm which iteratively collects failure trajectories and refines the agent policy via contrastive learning.
- Evaluation. Extensive experiments on three complex interactive tasks show that our method outperforms SFT behavioral cloning and other

strong baselines by a large margin.

• Analysis. We conduct in-depth analysis to carefully validate the effectiveness of ETO in multiple aspects, including out-of-distribution generalization, action efficiency, and its feasibility without the need for expert trajectories.

# 2 Task Formulation

The agent task with environment feedback can be formalized as a partially observable Markov decision process (POMDP) (U, S, A, O, T , R) with instruction space U, state space S, action space A, observation space O, transition function T : S×A → S, and reward function R : S × A → [0, 1]. Note that in our LLM-based agent scenario, U, A, O are subsets of natural language space.

Given a task instruction u ∈ U, the LLM agent with parameter θ generates the action a<sup>1</sup> ∼ πθ(·|u) ∈ A according to its policy πθ. The action incurs a change in the latent state space s<sup>t</sup> ∈ S, and an execution feedback as observation o<sup>t</sup> ∈ O. Then the agent generates the corresponding action in the t + 1 step at+1 ∼ πθ(·|u, a1, o1, ..., ot−1, at) ∈ A. The interaction loop repeats until the task completes or exceeds the maximum steps, and the trajectory is denoted as:

$$e = (u, a_1, o_1, ..., o_{n-1}, a_n) \sim \pi_{\theta}(e|u),$$
 (1)

$$\pi_{\theta}(e|u) = \prod_{j=1}^{n} \pi_{\theta}(a_j|u, a_1, o_1, ..., o_{j-1}), \quad (2)$$

where n is the trajectory length. Finally, the final reward r(u, e) ∈ [0, 1] is computed, with 1 representing successful task completion.

# 3 Method

Our method, ETO, starts by training a base agent through behavioral cloning. Based on the base agent, our framework continually enhanced the policy from trial and error in an iterative manner.

#### 3.1 Behavioral Cloning

Behavioral cloning (BC) has demonstrated promising results through supervised fine-tuning on the expert interaction trajectory data, serving as a solid starting point for building a powerful agent. In this work, we employ ReAct-style [\(Yao et al.,](#page-10-2) [2022b\)](#page-10-2) trajectory to conduct BC, which additionally generates Chain-of-Thought (CoT) rationales [\(Wei et al.,](#page-10-8) [2022\)](#page-10-8) before each action. Considering that the CoT and action are generated together in the ReAct

<span id="page-2-2"></span>![](_page_2_Figure_0.jpeg)

Figure 2: An illustrative overview of **Exploration-based Trajectory Optimization** (ETO). Starting from a base LLM agent trained through behavioral cloning, our method allows the agent to iteratively collect failure trajectories and update its policy by continually learning from contrastive failure-success trajectory pairs.

framework, we use  $\boldsymbol{a}$  to represent the action with CoT for simplicity.

Given an expert trajectory dataset  $\mathcal{D} = \left\{(u,e)^{(i)}\right\}_{i=1}^{|\mathcal{D}|}$ , where  $|\mathcal{D}|$  is the number of trajectories, we fine-tune an LLM on auto-regressive loss to get the base agent  $\pi_{\text{base}}$ :

<span id="page-2-0"></span>
$$\mathcal{L}_{SFT}(\pi_{\theta}) = -\mathbb{E}_{e \sim \mathcal{D}} \left[ \pi_{\theta}(e|u) \right], \tag{3}$$

where  $e = (u, a_1, o_1, ...o_{n-1}, a_n) \sim \mathcal{D}$  is an expert interaction trajectory.

Since  $\pi_{\theta}(e|u) = \prod_{j=1}^{n} \pi_{\theta}(a_{j}|u,...,o_{j-1})$ , in practice, we first concatenate the instruction, actions and observations in trajectory e as a text sequence t:

$$t = \operatorname{concat}(u, a_1, o_1, ..., o_{n-1}, a_n)$$
  
=  $(t_1, t_2, ..., t_l)$ , (4)

where  $t_k$  is the k-th token in the result sequence. Then the probability of the trajectory in Eq. (3) can be obtained by directly computing the probability of actions with observation tokens masked:

$$\pi_{\theta}(e|u) = -\sum_{k} \log \pi_{\theta}(t_k|t_{< k}) \times \mathbf{1}(t_k \in A),$$
(5)

where  $\mathbf{1}(t_k \in A)$  is an indicator function about whether  $t_k$  is a token belonging to actions produced by the agent.

#### 3.2 Learning From Exploration Failures

Behavioral cloning exclusively depends on expert trajectories and lacks the ability to explore the environment, leading to sub-optimal policies. To train a more powerful agent, it is important for the model to also explore failure trajectories. To achieve this, a viable approach is reinforcement learning, which empowers agents to actively explore the environment to get rewards and refine the policy through trial and error (Ouyang et al., 2022):

<span id="page-2-1"></span>
$$\max_{\pi_{\theta}} \mathbb{E}_{u \sim \mathcal{D}, e \sim \pi_{\theta}(e|u)} \left[ r(u, e) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(e|u) \mid\mid \pi_{\text{ref}}(e|u) \right],$$
(6)

where the KL term with weighting parameter  $\beta$  controls the deviation from the base reference policy  $\pi_{\rm ref}$ , *i.e.*, the base agent  $\pi_{\rm base}$ . In practice, the agent to be trained  $\pi_{\theta}$  is also initialized to  $\pi_{\rm base}$ . Then the optimization problem in Eq. (6) can be solved via RL methods such as PPO (Schulman et al., 2017; Ouyang et al., 2022).

However, directly applying online RL on LLM agents will present practical challenges such as instability and low efficiency (Shen et al., 2023; Rafailov et al., 2023). Therefore, we instead design an iterative offline learning framework and train the agent with contrastive trajectory data. As shown in Figure 2, the training process can be formulated in an iterative exploration-training loop. In the

#### Algorithm 1: ETO: Exploration-based Trajectory Optimization for LLM Agents

**Input:**  $\mathcal{D} = \{(u, a_1, o_1, ... o_{n-1}, a_n)^{(i)}\}$ : expert trajectory dataset for behavioral cloning,  $T_1$ :

```
number of behavioral cloning steps, I: number of iterations for ETO, T_2: number of steps
            in training phase, \pi_{\theta}: initial LLM policy.
Output: Final policy \pi_{\theta}
// Behavioral cloning
for i = 1 to T_1 do
 Optimize \theta on BC objective: \mathcal{L}_{SFT}(\pi_{\theta}) = -\mathbb{E}_{e \sim \mathcal{D}} \left[ \pi_{\theta}(e|u) \right]
// Iteratively learning from exploration failures
for i = 1 to I do
      \pi_{\text{base}} = \pi_{\theta}; \pi_{\text{ref}} = \pi_{\theta}
      Get base agent trajectories on \mathcal{D}: \hat{e} = (u, \hat{a}_1, \hat{o}_1, ..., \hat{o}_{m-1}, \hat{a}_m) \sim \pi_{\text{base}}(e|u)
      Compare rewards of \hat{e} with expert trajectory e to get the failure-success pair: e_w > e_l \mid u
      Construct contrastive trajectory dataset: \mathcal{D}_p = \left\{ (u, e_w, e_l)^{(i)} \right\}
      for j = 1 to T_2 do
           Optimize \bar{\theta} on trajectory contrastive objective:
             \mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(u, e_w, e_l) \sim \mathcal{D}_p} \left| \log \sigma \left( \beta \log \frac{\pi_{\theta}(e_w | u)}{\pi_{\theta}(e_l | u)} - \beta \log \frac{\pi_{\mathrm{ref}}(e_w | u)}{\pi_{\mathrm{ref}}(e_l | u)} \right) \right|
return \pi_{\theta}
```

<span id="page-3-3"></span>exploration phase of ETO, the agent explores the environment to collect failure trajectories. During the training phase, the agent learns the contrastive information from the "failure-success" trajectory pairs to update the policy.

**Exploration Phase** In this phase, the base agent  $\pi_{\text{base}}$  explores the environment to get the trajectories on the instructions of training data for BC:

$$\hat{e} = (u, \hat{a}_1, \hat{o}_1, ..., \hat{o}_{m-1}, \hat{a}_m) \sim \pi_{\text{base}}(e|u).$$
 (7)

The environments then return a reward  $\hat{r}$  corresponding to the trajectory  $\hat{e}$ .

Then we construct failure-success trajectory pairs, denote as  $e_w \succ e_l \mid u$ , based on the final rewards. Here,  $e_w$  and  $e_l$  represent the trajectories with higher and lower rewards, chosen from the expert trajectory e and agent-generated trajectory e respectively. Note that we only collect pairs where two trajectories have different rewards. If both e and e successfully complete the task, we simply discard the pair. Finally, we get the contrastive trajectory dataset  $\mathcal{D}_p = \left\{ (u, e_w, e_l)^{(i)} \right\}_{i=1}^{|\mathcal{D}_p|}$ .

<span id="page-3-4"></span>**Training Phase** In this phase, the agent policy is updated by modeling the contrastive failure-success information in the trajectory pair data.

Given trajectory pair  $e_w \succ e_l \mid u$ , the failuresuccess relation can be modeled via Bradley-Terry (BT) (Bradley and Terry, 1952) model:

<span id="page-3-1"></span>
$$p(e_w \succ e_l | u) = \frac{\exp(r(u, e_w))}{\exp(r(u, e_w)) + \exp(r(u, e_l))}.$$
(8)

On the other hand, under the optimal policy  $\pi_r(e|u)$  of Eq. (6), the reward function in the environment can be written as (Peng et al., 2019; Rafailov et al., 2023):

<span id="page-3-0"></span>
$$r(u,e) = \beta \log \frac{\pi_r(e|u)}{\pi_{\text{ref}}(e|u)} + \beta \log Z(x), \quad (9)$$

where  $Z(u)=\sum_{e}\pi_{\mathrm{ref}}(e|u)\exp\left(\frac{1}{\beta}r(u,e)\right)$  is the partition function. Substitute Eq. (9) into Eq. (8) to get the BT model over policy:

$$p(e_w \succ e_l|u) = \sigma\left(\beta \log \frac{\pi_{\theta}(e_w|u)}{\pi_{\theta}(e_l|u)} - \beta \log \frac{\pi_{\text{ref}}(e_w|u)}{\pi_{\text{ref}}(e_l|u)}\right),$$
(10)

where  $\sigma$  is the sigmoid function. Then the optimal policy  $\pi_{\theta}$  can be achieved by applying maximum likelihood:

<span id="page-3-2"></span>
$$\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(u, e_w, e_l) \sim \mathcal{D}_p} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(e_w | u)}{\pi_{\theta}(e_l | u)} - \beta \log \frac{\pi_{\text{ref}}(e_w | u)}{\pi_{\text{ref}}(e_l | u)} \right) \right]. \tag{11}$$

This optimization objective aims to increase the likelihood of the success trajectories e<sup>w</sup> and decrease the likelihood of failure trajectories e<sup>l</sup> , with a constraint term to maintain the basic agent capabilities. Moreover, as a reformulation of RL objective Eq. [\(6\)](#page-2-1), Eq. [\(11\)](#page-3-2) is directly maximizing the final reward while avoiding the need to perform RL optimization.

Iteration To further improve the agent's performance, ETO adopts an iterative explorationtraining manner. After the training phase, the agent policy can be used to gather new failure cases and create contrastive trajectory pairs. These new data are then used to further enhance the agent through trajectory contrastive learning. The complete learning process of ETO is shown in Algorithm [1.](#page-3-3)

#### 4 Experiments

In this section, we conduct extensive experiments to validate the effectiveness of ETO. Our method demonstrates superior performance compared to baselines across three datasets, and it exhibits enhanced advantages when dealing with out-ofdomain unseen tasks. The analysis further showcases the efficiency of our method. Furthermore, our method also holds promise in scenarios where expert trajectories are unavailable.

## 4.1 Experimental Settings

Datasets We conduct experiments on three representative agent datasets, WebShop [\(Yao et al.,](#page-10-6) [2022a\)](#page-10-6) for web navigation, ScienceWorld [\(Wang](#page-10-7) [et al.,](#page-10-7) [2022\)](#page-10-7) for embodied science experiments, and ALFWorld [\(Shridhar et al.,](#page-9-9) [2021\)](#page-9-9) for embodied house holding tasks. Both WebShop and Science-World environments provide dense final rewards ranging from 0 to 1, while ALFWorld only provides binary rewards indicating whether the task is completed. All three environments can be formally described as partially observable Markov decision processes. For details of the datasets and the expert trajectory collection process, please refer to Appendix [A.](#page-11-0)

The statistical information of our datasets is presented in Table [1.](#page-4-0) It is important to mention that, in addition to the in-distribution test sets, both ScienceWorld and ALFWorld contain test sets that include out-of-distribution unseen variations. These additional test sets allow us to assess the generalization capabilities of different agents.

<span id="page-4-0"></span>

| Dataset      |      |     | #Train #Test-Seen #Test-Unseen #Turns |      |
|--------------|------|-----|---------------------------------------|------|
| WebShop      | 1938 | 200 | -                                     | 4.9  |
| ScienceWorld | 1483 | 194 | 241                                   | 14.4 |
| ALFWorld     | 3321 | 140 | 134                                   | 10.1 |

Table 1: Statistics of datasets. "Test-Seen" and "Test-Unseen" are test set with seen and unseen scenarios respectively. "#Turns" denotes the average number of interaction turns for the expert trajectories.

Training Setup We mainly use Llama-2-7B-Chat [\(Touvron et al.,](#page-9-14) [2023\)](#page-9-14) as the base model for building LLM agents. To provide more comprehensive results, we also conduct experiments on Llama-2-13B-Chat and Mistral-7B [\(Jiang et al.,](#page-9-15) [2023\)](#page-9-15). We utilize the AdamW optimizer. For the SFT phase, the batch size is 64 and the learning rate is set to 1e-5 with 3% warm up and a cosine scheduler. Then the base agent will explore once for each instance in the training set to collect failure trajectories. For the training phase of ETO, the batch size is 32 and the learning rate is set to 1e-6. The β in DPO loss is set to 0.1. The learning epochs of SFT phase and training phase in ETO are set to 3. The number of iterations of ETO is set to 2 for WebShop and ScienceWorld, 1 for ALFWorld. All experiments are conducted on 8 NVIDIA A100 80G GPUs.

Baselines We compare ETO with SFT behavioral cloning and other post-imitation baseline methods. 1) SFT [\(Chen et al.,](#page-8-1) [2023;](#page-8-1) [Zeng et al.,](#page-10-4) [2023\)](#page-10-4) conducts behavioral cloning on expert trajectories, which is the base agent for ETO and other baselines. 2) Best-of-N sampling employs SFT base agent and selects the trajectory with the best reward within N samplings. Here we set N to 10. 3) RFT (Rejection sampling Fine-Tuning) [\(Yuan](#page-10-9) [et al.,](#page-10-9) [2023\)](#page-10-9) is a strong baseline which adds the success trajectories to the expert trajectory dataset and trains the agent on new augmented trajectories. 4) PPO (Proximal Policy Optimization) [\(Schulman](#page-9-11) [et al.,](#page-9-11) [2017\)](#page-9-11) is an RL method directly optimizing the SFT agents to maximize the final task reward. We also include GPT-3.5-Turbo [\(OpenAI,](#page-9-2) [2022\)](#page-9-2), GPT-4 [\(OpenAI,](#page-9-3) [2023\)](#page-9-3), and untuned Llama-2-7B-Chat for comparison.

Evaluation All methods are evaluated using the ReAct-style interaction format [\(Yao et al.,](#page-10-2) [2022b\)](#page-10-2), with CoT rationale generated before the action. See Appendix [D](#page-14-0) for the detailed prompts. We add 1 shot in-context example in the instruction prompt

<span id="page-5-0"></span>

| Method                       | WebShop | ScienceWorld   |      | ALFWorld |        |
|------------------------------|---------|----------------|------|----------|--------|
|                              |         | Seen<br>Unseen |      | Seen     | Unseen |
| GPT-4                        | 63.2    | 64.8           | 64.4 | 42.9     | 38.1   |
| GPT-3.5-Turbo                | 62.4    | 16.5           | 13.0 | 7.9      | 10.5   |
| Llama-2-7B-Chat              | 17.9    | 3.8            | 3.1  | 0.0      | 0.0    |
| Llama-2-7B-Chat + SFT        | 63.1    | 67.4           | 53.0 | 60.0     | 67.2   |
| Llama-2-7B-Chat + Best-of-N  | 63.8    | 70.2           | 57.6 | 62.1     | 69.4   |
| Llama-2-7B-Chat + RFT        | 63.6    | 71.6           | 54.3 | 62.9     | 66.4   |
| Llama-2-7B-Chat + PPO        | 64.2    | 59.4           | 51.7 | 22.1     | 29.1   |
| Llama-2-7B-Chat + ETO (ours) | 67.4    | 73.8           | 65.0 | 68.6     | 72.4   |

Table 2: The average reward of different methods on three agent datasets. "Seen" denotes the held-out test set with task types seen during training, while "Unseen" refers to the test set with critical unseen task variations.

for each task. The decoding temperature of the LLMs is set to be 0.0 for deterministic generation, except for Best-of-N method. We employ Average Reward as the metric, which represents the average reward of all task instances in the test set.

## 4.2 Results

Table [2](#page-5-0) presents the performance comparison of ETO and baselines on three agent datasets. As shown, ETO demonstrates a significant improvement over SFT imitation learning, leading to an average reward increase of 8% and 9.5% for Web-Shop and ScienceWorld. Our method also outperforms all other baselines on all datasets. On WebShop dataset, ETO even outperforms GPT-4 on the average reward, showing the extraordinary performance of our method. Although the RFT method also exhibits improvement compared to SFT, its performance remains constrained as it is an augmented version of behavioral cloning and solely learns from success trajectories. This comparison indicates the comparison between failure and expert trajectories is essential to improve the performance of the agent. Meanwhile, though PPO gains improved performance on WebShop, it struggles to achieve satisfactory results on the other two datasets due to the inherent instability in RL optimization procedures, particularly on ALFWorld dataset which only provides binary final rewards. In Appendix [C,](#page-11-1) we present case studies to show the task-solving trajectories of our method.

Notably, our approach showcases enhanced advantages in out-of-domain unseen scenarios, achieving an impressive performance boost of 20% on ScienceWorld-Unseen. Moreover, ETO exhibits strong effectiveness on the unseen scenarios

<span id="page-5-1"></span>

| Base LLM    | Method | WebShop | ScienceWorld   |      |  |
|-------------|--------|---------|----------------|------|--|
|             |        |         | Unseen<br>Seen |      |  |
| Llama-2-13B | SFT    | 66.3    | 68.1           | 57.6 |  |
|             | ETO    | 70.7    | 71.4           | 68.6 |  |
| Mistral-7B  | SFT    | 60.1    | 63.8           | 52.2 |  |
|             | ETO    | 66.2    | 68.5           | 62.5 |  |

Table 3: The average reward of different base LLMs on WebShop and ScienceWorld.

in ALFWorld and outperforms the RFT and PPO baselines, both of which suffer from performance degradation. These results underscore that learning by trial and error can further enhance the agent's generalization capabilities, particularly in out-ofdistribution unseen scenarios.

Results on Different LLMs To further demonstrate the effectiveness of our method, we present the results based on other base LLMs, including Llama-2-13B-Chat and Mistral-7B. Table [3](#page-5-1) showcases the consistent improvement in agent performance achieved by ETO across different LLMs. Notably, when compared to Llama-2-7B, the 13B model displays a relatively smaller performance gain on both datasets, suggesting that our method can provide greater benefits to weaker agents. Despite Mistral-7B is a more powerful LLM than Llama-2-13B, it falls short of Llama-2-7B after either SFT or ETO. This finding indicates that there is not a strong correlation between the basic LLM capabilities and the agent capabilities.

Analysis on Efficiency We evaluate the tasksolving efficiency of agents in ScienceWorld environment, which provides fine-grained subgoals for each task. The reward of a task is updated

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Figure 3: Cases of ScienceWorld reward trajectory for ETO, SFT Base Agent and Oracle. X: time steps  $(0 \rightarrow T)$ ; Y: scores  $(0 \rightarrow 100)$ . Task IDs are shown at the bottom-right. Best viewed in color.

upon the accomplishment of a subgoal. By assessing the agent's ability to achieve higher rewards within fewer action steps, we can determine its efficiency. Figure 3 showcases the score trajectories of ScienceWorld-Seen test set, comparing ETO with the SFT base agent, and the oracle agent. As depicted, ETO can reach higher rewards in fewer action steps than the SFT base agent. Interestingly, in certain cases like 15-90 and 19-23, our method outperforms even the oracle agent, reaching a score of 100 earlier. These results demonstrate that by learning from failure trajectories, our method also acquires a more powerful task-solving efficiency.

#### 4.3 Ablation of Iterations

In this section, we present a study on the impact of iteration numbers in ETO. The results are shown in Figure 4. As depicted, ETO demonstrates the ability to enhance the performance of agents in the first two iterations on both the WebShop and ScienceWorld datasets. However, further increasing the iterations does not lead to continuous improvement. Instead, the performance starts to decline after the third iteration. Regarding the ALFWorld dataset, only the first iteration of ETO shows an improvement. Surprisingly, the performance on the second and third iterations even falls behind that of the SFT base agent.

To explain this, it is important to note that the learning process of ETO relies on a fixed expert trajectory set, and the exploration phase of the agent is performed on the same training set. Consequently, the diversity and quantity of failure-success contrastive trajectory data are constrained. Initially, the policy can be improved by learning from past mistakes. However, the model gets overfitting on

<span id="page-6-1"></span>![](_page_6_Figure_6.jpeg)

Figure 4: ETO performance on multiple iterations.

<span id="page-6-2"></span>

| Method | Level      | lr   | β   | Avg. Reward |
|--------|------------|------|-----|-------------|
| SFT    | -          | -    | -   | 63.1        |
| ЕТО    | Trajectory | 1e-6 | 0.1 | 67.4        |
|        | Step       | 1e-6 | 0.1 | 8.3         |
|        | Step       | 1e-7 | 0.5 | 62.8        |
|        | Mixture    | 1e-6 | 0.1 | 64.3        |

Table 4: The average reward on WebShop of agents trained on different level contrastive data.

the contrastive information in subsequent iterations, resulting in a decline in performance. In the case of ALFWorld, the coarse-grained binary rewards further hinder the agent from getting improvement from iterative training. As a potential solution, future work could explore the incorporation of GPT-4 to dynamically construct more diverse contrastive trajectory data.

#### 4.4 Strategy of Contrastive Data Construction

In this section, we delve deeper into the contrastive trajectory pair construction strategy used in our method. In Section 3.2, we directly learn from failure-success trajectory pairs (Eq. (11)), referred to trajectory-wise contrastive. Alternatively, inspired by previous work (Lightman et al., 2023), we introduce a fine-grained variation of ETO that captures step-wise contrastive information by comparing "good-bad" action pairs. To achieve this, we use the expert trajectory to conduct teacher forcing for first t-1 steps, and then have the agent predict the action of t-th step. Then the quality of t-th action is determined by the final rewards. We also implement a *mixture* variation by combining the above two strategies. For further details regarding the step-wise variation of ETO, please refer to Appendix B.

The comparison of different methods is presented in Table 4. As the results demonstrate,

trajectory-wise contrastive yields the best performance. On the other hand, we observed that stepwise contrastive modeling tends to be less stable, necessitating a lower learning rate and a higher constraint parameter β to maintain the basic capabilities of the agent. This instability may be attributed to the inaccurate estimation of the action quality, as we simply utilize the final rewards to construct step-wise contrastive pairs. Moreover, the performance of mixture strategy also falls short compared to trajectory-level contrastive.

#### 4.5 Self-Play w/o Expert Trajectory

In this section, we explore a challenging scenario where no expert trajectory is available. In such cases, the agent is compelled to explore the environment and depend on self-play to enhance its capabilities. To achieve this, we eliminate the behavioral cloning phase of ETO and allow the LLM agent to explore the environments using a decoding temperature of 1.0. Subsequently, we compare different trajectories associated with the same instruction based on their final rewards, creating trajectory preference data. Finally, the agent is trained exclusively on the preference data generated by itself.

On WebShop, the untuned Llama-2-7B-Chat achieves relatively high rewards. Thus, we use this dataset to conduct the experiment. We also employ rejection sampling fine-tuning (RFT) as a baseline. The results in Table [5](#page-7-0) show that ETO alone does not improve performance without behavioral cloning. In contrast, RFT shows promising ability to enhance the agent's capabilities without relying on expert trajectories. However, when combining RFT with ETO, we observe a further enhancement in the agent's performance. These findings suggest that in scenarios without expert trajectories, it may be beneficial to first employ RFT and then allow the agent to learn from exploration failures. These results further highlight the potential of our method when expert trajectories are unavailable.

# 5 Related Work

Imitation Learning Imitation learning is a learning paradigm where an agent learns a policy by mimicking expert demonstrations [\(Hussein](#page-9-17) [et al.,](#page-9-17) [2017;](#page-9-17) [Fang et al.,](#page-8-3) [2019\)](#page-8-3). A prevalent approach in imitation learning is behavioral cloning (BC) [\(Pomerleau,](#page-9-8) [1991\)](#page-9-8), which utilizes expert trajectories to learn a direct mapping from states to actions. There are various methods to mitigate

<span id="page-7-0"></span>

| Method           | w/ BC? | Avg. Reward |
|------------------|--------|-------------|
| SFT              | ✓      | 63.1        |
| RFT              | ✓      | 63.6        |
| ETO              | ✓      | 67.4        |
| Llama-2-7B-Chat† | ✗      | 17.9        |
| RFT              | ✗      | 48.4        |
| ETO              | ✗      | 12.5        |
| RFT+ETO          | ✗      | 51.2        |

Table 5: Performance of self-play without behavioral cloning from expert trajectories. † means directly prompting an untuned Llama-2-7B-Chat.

the limitations of BC [\(Ross et al.,](#page-9-18) [2011;](#page-9-18) [Ross and](#page-9-19) [Bagnell,](#page-9-19) [2014\)](#page-9-19). Our method, ETO, shares a similar spirit with DAgger [\(Ross et al.,](#page-9-18) [2011\)](#page-9-18), an approach used to enhance the agent's performance by learning from failure cases. However, unlike DAgger which gathers additional expert trajectories on agent-failed cases, ETO improves the policy through learning from contrastive trajectory pairs.

LLM Agents With the various emergent abilities of LLMs, researchers have explored building agent systems based on LLMs [\(Xi et al.,](#page-10-0) [2023\)](#page-10-0). Recent projects such as AutoGPT [\(Richards,](#page-9-20) [2023\)](#page-9-20), BabyAGI [\(Nakajima,](#page-9-21) [2023\)](#page-9-21) have employed LLMs as core controllers, building powerful agent frameworks capable of solving realistic tasks. While GPTs have shown strong agent intelligence, opensource LLMs still lag far behind [\(Liu et al.,](#page-9-6) [2023;](#page-9-6) [Wang et al.,](#page-10-3) [2023d\)](#page-10-3). To bridge this gap, recent studies, including FireAct [\(Chen et al.,](#page-8-1) [2023\)](#page-8-1), Agent-Tuning [\(Zeng et al.,](#page-10-4) [2023\)](#page-10-4), and Lumos [\(Yin et al.,](#page-10-5) [2023\)](#page-10-5), construct expert trajectory data from teacher agents (*e.g.*, GPT-4) and perform BC on opensource LLMs. Taking a step further, [Aksitov et al.](#page-8-4) [\(2023\)](#page-8-4) refine the agent through iterative BC on success trajectories generated by the previous policy. Concurrently with our work, [Yang et al.](#page-10-10) [\(2023\)](#page-10-10) use the DAgger framework [\(Ross et al.,](#page-9-18) [2011\)](#page-9-18) and also employ DPO loss to develop embodied multimodal agents.

LLM Policy Learning Learning from preference has shown promise for learning an enhanced LLM policy, particularly in LLM alignment research. Reinforcement Learning from Human Feedback (RLHF) is a method that learns a reward model and then utilizes proximal policy optimization to update the policy model [\(Christiano et al.,](#page-8-5) [2017;](#page-8-5) [Ouyang et al.,](#page-9-10) [2022\)](#page-9-10). Despite its attractive advantages, RLHF presents limitations regarding training

<span id="page-8-6"></span>

| Method     | Exploration | Reward | Efficiency | Robustness |
|------------|-------------|--------|------------|------------|
| SFT        | ✗           | ✗      | ✓          | ✓          |
| RFT        | ✓           | ✗      | ✓          | ✓          |
| Online RL  | ✓           | ✓      | ✗          | ✗          |
| Offline RL | ✗           | ✓      | ✓          | ✗          |
| ETO        | ✓           | ✓      | ✓          | ✓          |

Table 6: ETO vs alternatives: ETO can leverage exploration failures to optimize the policy with high computational efficiency and robustness.

efficiency and instability. To address these issues, [Rafailov et al.](#page-9-0) [\(2023\)](#page-9-0) reformulate the optimization objective of RLHF, introducing the DPO loss to directly model preferences. Similar to our work, ReST [\(Gulcehre et al.,](#page-9-22) [2023\)](#page-9-22) iteratively generates new samples from the current policy and refines the policy using offline RL methods. Recent studies have explored the application of LLM policy learning in other domains [\(Lightman et al.,](#page-9-16) [2023;](#page-9-16) [Wang](#page-9-23) [et al.,](#page-9-23) [2023b\)](#page-9-23). For example, [Wang et al.](#page-10-11) [\(2023c\)](#page-10-11) train a step-wise reward model to improve the performance of LLMs in mathematical reasoning. The comparison of ETO with several alternatives in Table [6](#page-8-6) highlights the strength of our method in LLM agent policy learning.

#### 6 Conclusion

In this work, we present ETO, a method aimed at enhancing the capabilities of LLM agents. Our approach allows the agent to learn by trial and error, thereby improving the performance of the base agent acquired through behavioral cloning. ETO uses an exploration-training iteration framework. During the exploration phase, the agent explores the environment, gathering failure trajectories and constructing trajectory preference pairs. Subsequently, in the training phase, the agent learns from the preference information using DPO loss. This iterative process of exploration and training enables further improvement in the agent's performance. Extensive experiments on three agent datasets demonstrate our method outperforms behavioral cloning and strong baselines by a large margin. Moreover, our method exhibits remarkable efficiency and shows great potential in scenarios where expert trajectories are unavailable.

# Limitations

Our method, ETO, demonstrates effective learning of powerful LLM agents through trial and error. However, it is important to acknowledge several

limitations of this work. 1) ETO simplifies the comparison of failure-success trajectories by assuming that the agent generates wrong actions right from the beginning. However, in realistic cases, the agent may start executing incorrect actions from some intermediate step. If we can identify when the agent makes a bad action (*e.g.*, aˆ<sup>3</sup> at 3-th step), we should then collect the expert trajectory for the remaining actions at>3. Unfortunately, most current environments do not contain such information, making it challenging to conduct action-wise or process-level reward modeling. A potential solution is to employ GPT-4 to identify the bad action and construct fine-grained contrastive trajectory data. 2) This work primarily focuses on developing specialized LLM agents for a specific agent task, with limited exploration into the construction of strong generalized agents. For future work, we will investigate the transferability of the policies trained by ETO and try to apply our method in a multi-task training scenario.

#### References

<span id="page-8-4"></span>Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, et al. 2023. [Rest meets react: Self](https://arxiv.org/abs/2312.10003)[improvement for multi-step reasoning llm agent.](https://arxiv.org/abs/2312.10003) *ArXiv preprint*, abs/2312.10003.

<span id="page-8-2"></span>Ralph Allan Bradley and Milton E Terry. 1952. [Rank](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) [analysis of incomplete block designs: I. the method](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) [of paired comparisons.](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091) *Biometrika*, 39(3/4):324– 345.

<span id="page-8-1"></span>Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. 2023. [Fireact:](https://arxiv.org/abs/2310.05915) [Toward language agent fine-tuning.](https://arxiv.org/abs/2310.05915) *ArXiv preprint*, abs/2310.05915.

<span id="page-8-5"></span>Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. [Deep](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) [reinforcement learning from human preferences.](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) In *Advances in Neural Information Processing Systems*, pages 4299–4307.

<span id="page-8-0"></span>Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. [Mind2web: Towards a generalist agent for the](https://arxiv.org/abs/2306.06070) [web.](https://arxiv.org/abs/2306.06070) *Advances in Neural Information Processing Systems*.

<span id="page-8-3"></span>Bin Fang, Shidong Jia, Di Guo, Muhua Xu, Shuhuan Wen, and Fuchun Sun. 2019. [Survey of imitation](https://link.springer.com/article/10.1007/s41315-019-00103-5) [learning for robotic manipulation.](https://link.springer.com/article/10.1007/s41315-019-00103-5) *International Journal of Intelligent Robotics and Applications*, 3:362– 369.

- <span id="page-9-22"></span>Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, et al. 2023. [Reinforced self](https://arxiv.org/abs/2308.08998)[training \(rest\) for language modeling.](https://arxiv.org/abs/2308.08998) *ArXiv preprint*, abs/2308.08998.
- <span id="page-9-17"></span>Ahmed Hussein, Mohamed Medhat Gaber, Eyad Elyan, and Chrisina Jayne. 2017. [Imitation learning: A sur](https://dl.acm.org/doi/abs/10.1145/3054912)[vey of learning methods.](https://dl.acm.org/doi/abs/10.1145/3054912) *ACM Computing Surveys (CSUR)*, 50(2):1–35.
- <span id="page-9-15"></span>Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. [Mistral](https://arxiv.org/abs/2310.06825) [7b.](https://arxiv.org/abs/2310.06825) *ArXiv preprint*, abs/2310.06825.
- <span id="page-9-16"></span>Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. [Let's verify step by step.](https://arxiv.org/abs/2305.20050) *ArXiv preprint*, abs/2305.20050.
- <span id="page-9-4"></span>Bill Yuchen Lin, Yicheng Fu, Karina Yang, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Prithviraj Ammanabrolu, Yejin Choi, and Xiang Ren. 2023. [Swiftsage: A generative agent with fast and slow](https://arxiv.org/abs/2305.17390) [thinking for complex interactive tasks.](https://arxiv.org/abs/2305.17390) *Advances in Neural Information Processing Systems*.
- <span id="page-9-6"></span>Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. 2023. [Agentbench: Eval](https://arxiv.org/abs/2308.03688)[uating llms as agents.](https://arxiv.org/abs/2308.03688) In *The Twelfth International Conference on Learning Representations*.
- <span id="page-9-5"></span>Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023. [Chameleon: Plug-and-play compo](https://api.semanticscholar.org/CorpusID:258212542)[sitional reasoning with large language models.](https://api.semanticscholar.org/CorpusID:258212542) *ArXiv*, abs/2304.09842.
- <span id="page-9-7"></span>Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. [Gaia: a benchmark for general ai assistants.](https://arxiv.org/abs/2311.12983) *ArXiv preprint*, abs/2311.12983.
- <span id="page-9-21"></span>Yohei Nakajima. 2023. [Babyagi.](https://github.com/yoheinakajima/babyagi)
- <span id="page-9-2"></span>OpenAI. 2022. [Introducing chatgpt.](https://openai.com/blog/chatgpt)
- <span id="page-9-3"></span>OpenAI. 2023. [Gpt-4 technical report.](http://arxiv.org/abs/2303.08774) *arXiv*, pages 2303–08774.
- <span id="page-9-10"></span>Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. [Training language models to follow instruc](https://arxiv.org/abs/2203.02155)[tions with human feedback.](https://arxiv.org/abs/2203.02155) *Advances in Neural Information Processing Systems*, 35:27730–27744.
- <span id="page-9-13"></span>Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. 2019. [Advantage-weighted regression: Sim](https://arxiv.org/abs/1910.00177)[ple and scalable off-policy reinforcement learning.](https://arxiv.org/abs/1910.00177) *ArXiv preprint*, abs/1910.00177.

- <span id="page-9-8"></span>Dean A Pomerleau. 1991. [Efficient training of artificial](https://direct.mit.edu/neco/article-abstract/3/1/88/5572) [neural networks for autonomous navigation.](https://direct.mit.edu/neco/article-abstract/3/1/88/5572) *Neural computation*, 3(1):88–97.
- <span id="page-9-0"></span>Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. 2023. [Direct preference optimization: Your language](https://arxiv.org/abs/2305.18290) [model is secretly a reward model.](https://arxiv.org/abs/2305.18290) *ArXiv preprint*, abs/2305.18290.
- <span id="page-9-20"></span>Toran Bruce Richards. 2023. [Auto-gpt: An autonomous](https://github.com/Significant-Gravitas/AutoGPT) [gpt-4 experiment.](https://github.com/Significant-Gravitas/AutoGPT)
- <span id="page-9-19"></span>Stephane Ross and J Andrew Bagnell. 2014. [Reinforce](https://arxiv.org/abs/1406.5979)[ment and imitation learning via interactive no-regret](https://arxiv.org/abs/1406.5979) [learning.](https://arxiv.org/abs/1406.5979) *arXiv preprint arXiv:1406.5979*.
- <span id="page-9-18"></span>Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. 2011. [A reduction of imitation learning and struc](https://arxiv.org/abs/1011.0686)[tured prediction to no-regret online learning.](https://arxiv.org/abs/1011.0686) In *Proceedings of the fourteenth international conference on artificial intelligence and statistics*, pages 627– 635. JMLR Workshop and Conference Proceedings.
- <span id="page-9-11"></span>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. [Proximal](https://arxiv.org/abs/1707.06347) [policy optimization algorithms.](https://arxiv.org/abs/1707.06347) *ArXiv preprint*, abs/1707.06347.
- <span id="page-9-12"></span>Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, and Deyi Xiong. 2023. [Large language model align](https://arxiv.org/abs/2309.15025)[ment: A survey.](https://arxiv.org/abs/2309.15025) *ArXiv preprint*, abs/2309.15025.
- <span id="page-9-24"></span>Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. 2020. [ALFRED: A](https://doi.org/10.1109/CVPR42600.2020.01075) [benchmark for interpreting grounded instructions](https://doi.org/10.1109/CVPR42600.2020.01075) [for everyday tasks.](https://doi.org/10.1109/CVPR42600.2020.01075) In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10737–10746. IEEE.
- <span id="page-9-9"></span>Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. 2021. [Alfworld: Aligning text and em](https://openreview.net/forum?id=0IOX0YcCdTn)[bodied environments for interactive learning.](https://openreview.net/forum?id=0IOX0YcCdTn) In *9th International Conference on Learning Representations*.
- <span id="page-9-14"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. [Llama 2: Open founda](https://arxiv.org/abs/2307.09288)[tion and fine-tuned chat models.](https://arxiv.org/abs/2307.09288) *ArXiv preprint*, abs/2307.09288.
- <span id="page-9-1"></span>Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2023a. [A survey on large](https://arxiv.org/abs/2308.11432) [language model based autonomous agents.](https://arxiv.org/abs/2308.11432) *ArXiv preprint*, abs/2308.11432.
- <span id="page-9-23"></span>Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, and Zhifang Sui. 2023b. [Making large language models better reasoners with](https://arxiv.org/abs/2309.02144) [alignment.](https://arxiv.org/abs/2309.02144) *ArXiv preprint*, abs/2309.02144.

- <span id="page-10-11"></span>Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. 2023c. [Math-shepherd: A label-free step-by-step](https://arxiv.org/abs/2312.08935) [verifier for llms in mathematical reasoning.](https://arxiv.org/abs/2312.08935) *ArXiv preprint*, abs/2312.08935.
- <span id="page-10-7"></span>Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. 2022. [ScienceWorld: Is](https://aclanthology.org/2022.emnlp-main.775) [your agent smarter than a 5th grader?](https://aclanthology.org/2022.emnlp-main.775) In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 11279–11298, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
- <span id="page-10-3"></span>Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. 2023d. [Mint:](https://arxiv.org/abs/2309.10691) [Evaluating llms in multi-turn interaction with tools](https://arxiv.org/abs/2309.10691) [and language feedback.](https://arxiv.org/abs/2309.10691) In *The Eleventh International Conference on Learning Representations*.
- <span id="page-10-8"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. [Chain-of-thought prompting elicits rea](http://arxiv.org/abs/2201.11903)[soning in large language models.](http://arxiv.org/abs/2201.11903) *Advances in Neural Information Processing Systems*, 35:24824–24837.
- <span id="page-10-0"></span>Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. [The rise and](https://arxiv.org/abs/2309.07864) [potential of large language model based agents: A](https://arxiv.org/abs/2309.07864) [survey.](https://arxiv.org/abs/2309.07864) *ArXiv preprint*, abs/2309.07864.
- <span id="page-10-10"></span>Yijun Yang, Tianyi Zhou, Kanxue Li, Dapeng Tao, Lusong Li, Li Shen, Xiaodong He, Jing Jiang, and Yuhui Shi. 2023. [Embodied multi-modal agent trained by](http://arxiv.org/abs/2311.16714) [an llm from a parallel textworld.](http://arxiv.org/abs/2311.16714) *arXiv preprint arXiv:2311.16714*.
- <span id="page-10-6"></span>Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022a. [Webshop: Towards scalable](http://arxiv.org/abs/2207.01206) [real-world web interaction with grounded language](http://arxiv.org/abs/2207.01206) [agents.](http://arxiv.org/abs/2207.01206) *Advances in Neural Information Processing Systems*, 35:20744–20757.
- <span id="page-10-2"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022b. [React: Synergizing reasoning and acting in language](https://arxiv.org/abs/2210.03629) [models.](https://arxiv.org/abs/2210.03629) In *The Eleventh International Conference on Learning Representations*.
- <span id="page-10-5"></span>Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. 2023. [Lumos: Learning agents](https://arxiv.org/abs/2311.05657) [with unified data, modular design, and open-source](https://arxiv.org/abs/2311.05657) [llms.](https://arxiv.org/abs/2311.05657) *ArXiv preprint*, abs/2311.05657.
- <span id="page-10-9"></span>Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. 2023. [Scal](https://arxiv.org/abs/2308.01825)[ing relationship on learning mathematical reason](https://arxiv.org/abs/2308.01825)[ing with large language models.](https://arxiv.org/abs/2308.01825) *ArXiv preprint*, abs/2308.01825.
- <span id="page-10-4"></span>Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2023. [Agenttuning:](https://arxiv.org/abs/2310.12823) [Enabling generalized agent abilities for llms.](https://arxiv.org/abs/2310.12823) *ArXiv preprint*, abs/2310.12823.

<span id="page-10-1"></span>Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. 2023. [Webarena:](https://arxiv.org/abs/2307.13854) [A realistic web environment for building autonomous](https://arxiv.org/abs/2307.13854) [agents.](https://arxiv.org/abs/2307.13854) *ArXiv preprint*, abs/2307.13854.

# <span id="page-11-0"></span>A Datasets

WebShop WebShop [\(Yao et al.,](#page-10-6) [2022a\)](#page-10-6) is an online shopping website environment where agents navigate the platform to make purchases based on user instructions. Once the agent selects the "buy" action, the environment provides a final reward, which is calculated based on the matching heuristics of the product's attributes and price.

ScienceWorld ScienceWorld [\(Wang et al.,](#page-10-7) [2022\)](#page-10-7) is a text-based virtual environment centered around accomplishing elementary science experiments, including 10 different task types such as thermodynamics and electrical circuits. The agents need to be grounded in embodied interactive environments to engage with and comprehend scientific concepts through practical experience. Each task in ScienceWorld includes several optional subgoals, and the overall final reward is computed based on the achievement of these subgoals.

The original test set in ScienceWorld consists of critical unseen task variations. For instance, in the training set, the task may involve boiling water, whereas in the test set, the task is to boil lead. Consequently, we employ the original test set to evaluate our model's generalization performance on unseen scenarios. We utilize the original development set as our test set with seen scenarios. We exclude Task-9 and Task-10 due to their excessively long task-solving trajectories. Following [Lin et al.](#page-9-4) [\(2023\)](#page-9-4), we use the first 10 instances for task types with more than 10 test variations for fair and cost-effective comparisons.

ALFWorld ALFWorld [\(Shridhar et al.,](#page-9-9) [2021\)](#page-9-9) consists of interactive TextWorld environments that parallel embodied worlds in the ALFRED [\(Shridhar et al.,](#page-9-24) [2020\)](#page-9-24) dataset. In this environment, agents are required to explore and complete high-level house-holding instructions. The original ALFWorld dataset comprises both seen and unseen evaluation sets. The seen set is designed to assess in-distribution generalization, whereas the unseen set with new task instances measures out-of-distribution generalization of the agents.

CoT Annotation Webshop and ALFWorld provide a few human-annotated trajectories for imitation learning. We also employ GPT-4 as the teacher agent to explore in the WebShop environment and select trajectories which have a reward greater than 0.7. ScienceWorld environment provides heuristic searching algorithms to generate golden trajectories for each sub-task. Since the original trajectories do not contain CoT information for each action step, we further utilize GPT-4 to generate the corresponding rationales.

#### <span id="page-11-2"></span>B Details for Step-Wise Contrastive

We implement a variation of ETO which learns from contrastive good-bad action pairs. Specifically, for a task instruction u with expert trajectory e = (u, a1, ..., on−1, an), we utilize teacher forcing for the first t − 1 steps (a1, o1, ...at−1, ot−1), and let the agent predict the actions from t-th step to get the trajectory:

$$\hat{e} = (u, a_1, o_1, ..., o_{t-1}, \hat{a}_t, \hat{o}_t, ..., \hat{o}_{m-1}, \hat{a}_m)$$
(12)

The environments return a reward rˆ for the trajectory eˆ. If we denote the golden trajectory for the first t − 1 steps as e(t−1), then the good-bad action pairs a<sup>w</sup> ≻ a<sup>l</sup> | u, e(t−1) is constructed based on the final rewards. Here, a<sup>w</sup> and a<sup>l</sup> represent the actions with higher and lower final rewards, chosen from (a<sup>t</sup> , aˆt) respectively. Then the contrastive relation of the action pair can also be utilized in DPO loss to improve the policy:

$$\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(a_w | u, e_{(t-1)})}{\pi_{\theta}(a_l | u, e_{(t-1)})} - \beta \log \frac{\pi_{\text{ref}}(a_w | u, e_{(t-1)})}{\pi_{\text{ref}}(a_l | u, e_{(t-1)})} \right) \right]. \tag{13}$$

#### <span id="page-11-1"></span>C Case Study

Here, we present case studies to compare ETO agents with SFT behavioral cloning agents. We illustrate examples of WebShop and ScienceWorld in Figure [5](#page-12-0) and [6,](#page-13-0) respectively.

In the WebShop scenario, the SFT agent failed to select the "3pc" color option, resulting in a suboptimal trajectory. Conversely, ETO demonstrated the ability to learn from past failures and adeptly chose the appropriate attribute option.

In the ScienceWorld example, the task is to find a(n) animal. The SFT agent, however, persisted in performing the incorrect action, focusing on a non-living object instead. Additionally, despite generating a correct CoT, the SFT agent's action selection was inconsistent with the CoT rationale. In contrast, ETO successfully accomplished the task, showcasing the effectiveness of trial and error.

## Case Study of WebShop

# SFT Agent

Task Instruction: I am looking for a hair growth treatment in the color 3pc, and price lower than 40.00 dollars

Thought 1: I should search for "hair growth treatment 3pc" first.

Action 1: search[hair growth treatment 3pc] Observation 1: *<searching results>*

Thought 2: There are several options, but I need to find one that's within my budget. The first option, B09S12LVLZ, is a 3-piece ginger hair growth cream set priced at \$15.99, which is under \$50. This fits my criteria, so I'll take a closer look at this product.

Action 2: click[B09S12LVLZ] Observation 2: *<product details>*

Thought 3: The product details confirm it's a 3-piece set and the price is still within my budget. I don't need to check the size options since it's already the 3-piece set I'm looking for. The price is good, and although there's no rating, I'm willing to try it out. I'll go ahead and make the purchase.

Action 3: click[buy now]

Reward: 0.5

<span id="page-12-0"></span>*# Did not select the color option.*

# ETO Agent

Task Instruction: I am looking for a hair growth treatment in the color 3pc, and price lower than 40.00 dollars

Thought 1: I should search for "hair growth treatment 3pc" first.

Action 1: search[hair growth treatment 3pc] Observation 1: *<searching results>*

Thought 2: B09S12LVLZ and B09QMJY2ZC are hair growth treatments in a 3pc set and priced lower than \$50. I can check B09S12LVLZ first.

Action 2: click[B09S12LVLZ] Observation 2: *<product details>*

Thought 3: For a hair growth treatment in a 3pc set, the item has options for different sizes. I should click on the '3pcs' option.

Action 3: click[3pcs]

Observation 3: You have clicked [3pcs] option.

Thought 4: Now that I have selected the 3pc option, I should proceed to purchase the item.

Action 4: click[buy now]

Reward: 1.0

Figure 5: Case study for WebShop.

# SFT Agent

Task Instruction: Your task is to find a(n) animal. First, focus on the thing. Then, move it to the yellow box in the bedroom.

Thought 1: I need to find an animal as per the task description. I should start by looking around the current room to identify if there's an animal here.

Action 1: look around

Observation 1: This room is called the living room. In it, you see: the agent, a substance called air, a chair. On the chair is: nothing. a couch. On the couch is: a white pillow. a desk. On the desk is: a drawer. a finger painting. You also see: A door to the hallway (that is open). Thought 2: There are no animals in the living room, but I see a finger painting that could be considered an animal-related item. I should look around again to ensure I didn't miss anything before deciding on my next step.

Action 2: look around *# Repeat wrong action* Observation 2: This room is called the living room. In it, you see: the agent, a substance called air, a chair. On the chair is: nothing. a couch. On the couch is: a white pillow. a desk. On the desk is: a drawer. a finger painting. You also see: A door to the hallway (that is open). Thought 3: The finger painting is a potential animal-related item, but it's not the target of the task. I should focus on finding an actual animal.

<span id="page-13-0"></span>Action 3: focus on finger painting *# Not a(n) animal, contradict with the CoT* Reward: 0.0

# ETO Agent

Task Instruction: Your task is to find a(n) animal. First, focus on the thing. Then, move it to the yellow box in the bedroom.

Thought 1: I need to find an animal as per the task description. I should start by looking around the current location to identify any potential animals or related items.

Action 1: look around

Observation 1: This room is called the living room. In it, you see: ...

Thought 2: The living room doesn't seem to have any animals, but I might have missed something. I should check another area.

Action 2: teleport to outside

Observation 2: You teleport to the outside.

Thought 3: Now that I'm outside, I should look around to see if there are any animals or indications of animals here.

Action 3: look around

Observation 3: This outside location is called the outside. Here you see: the agent, an axe, a frog egg, a baby wolf ...

Thought 4: I see a frog egg here, which is a developmental stage of an animal. This fits the task description of finding an animal.

Action 4: focus on frog egg

Observation 4: You focus on the frog egg.

Thought 5: The frog egg is the early stage of a common toad, which is an animal. I should pick it up to move it to the specified location.

Action 5: pick up frog egg

Observation 5: You move the common toad to the inventory.

Thought 6: With the common toad (frog egg) in my inventory, I need to go to the bedroom to place it in the yellow box as instructed.

Action 6: teleport to bedroom

Observation 6: You teleport to the bedroom. Thought 7: Now that I'm in the bedroom, I should complete the task by moving the common toad from my inventory to the yellow box.

Action 7: move common toad to yellow box

Reward: 1.0

## <span id="page-14-0"></span>D Prompt for Evaluation

We show the instruction prompts for WebShop, ScienceWorld, ALFWorld in Figure [7,](#page-14-1) [8,](#page-15-0) [9,](#page-15-1) respectively.

#### Instruction Prompt for WebShop

You are doing a web shopping task. I will give you instructions about what to do. You have to follow the instructions. Every round I will give you an observation and a list of available actions, you have to respond to an action based on the state and instruction. You can use search action if search is available. You can click one of the buttons in clickables. An action should be one of the following structure: search[keywords] or click[value]

If the action is not valid, perform nothing. Keywords in search are up to you, but the value in click must be a value in the list of available actions. Remember that your keywords in search should be carefully designed.

Your response should use the following format:

Thought: I think ... Action: click[something]

Figure 7: Instruction prompt for WebShop.

#### <span id="page-14-1"></span>Instruction Prompt for ScienceWorld

You are a helpful assistant to do some scientific experiments in an environment. In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway You should explore the environment and find the items you need to complete the experiment. You can teleport to any room in one step. All containers in the environment have already been opened, you can directly get items from the containers.

The available actions are:

open OBJ: open a container close OBJ: close a container activate OBJ: activate a device deactivate OBJ: deactivate a device

connect OBJ to OBJ: connect electrical components disconnect OBJ: disconnect electrical components

use OBJ [on OBJ]: use a device/item look around: describe the current room examine OBJ: describe an object in detail look at OBJ: describe a container's contents

read OBJ: read a note or book

move OBJ to OBJ: move an object to a container pick up OBJ: move an object to the inventory pour OBJ into OBJ: pour a liquid into a container

mix OBJ: chemically mix a container teleport to LOC: teleport to a specific room focus on OBJ: signal intent on a task object

wait: task no action for 10 steps wait1: task no action for a step

Your response should use the following format:

Thought: <your thoughts> Action: <your next action>

Figure 8: Instruction prompt for ScienceWorld.

# <span id="page-15-0"></span>Instruction Prompt for ALFWorld

Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given a detailed description of the current environment and your goal to accomplish.

For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts. Action: your next action".

The available actions are:

- 1. go to recep
- 2. task obj from recep
- 3. put obj in/on recep
- 4. open recep
- 5. close recep
- 6. toggle obj recep
- 7. clean obj with recep
- 8. heat obj with recep
- 9. cool obj with recep

where obj and recep correspond to objects and receptacles.

After each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment outputs "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

<span id="page-15-1"></span>Thought: <your thoughts> Action: <your next action>

Figure 9: Instruction prompt for ALFWorld.