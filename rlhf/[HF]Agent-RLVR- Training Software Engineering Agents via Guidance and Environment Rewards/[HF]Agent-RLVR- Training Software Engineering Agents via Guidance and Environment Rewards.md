# <span id="page-0-0"></span>Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards

Jeff Da Clinton Wang Xiang Deng

Yuntao Ma Nikhil Barhate Sean Hendryx

Scale AI

## Abstract

Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models (LLMs) and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces *agent guidance*, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. We curated a dataset of 817 training environments with problem statements, environments, and guidance in the software engineering domain. Agent-RLVR elevates the PASS@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-BENCH VERIFIED. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting PASS@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle.

## 1 Introduction

Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a prevalent method to train language models for reasoning tasks. Recent models such as OpenAI's o1 and DeepSeek-R1 [\[1\]](#page-9-0) have used RLVR to achieve state-of-the-art performance on math and coding tasks. Relying only on reward signals from problem graders or unit test feedback, RLVR yields exceptional improvements in performance along with generalization to related tasks.

However, equivalent success has not been achieved by RLVR in agentic settings. These settings require reasoning across multiple turns, navigating complex problem statements with sequential

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: Agent-RLVR is a framework for training agents with RLVR using environment feedback and guidance. ① The agent attempts the problem without any additional guidance and the environment runs unit tests on the generated patch to determine the correctness. ② We generate guidance for failed patches by leveraging environment information. We provide several types of guidance – a plan, environment feedback, and environment interaction. The agent reattempts the problem with guidance. ③ Positive trajectories are sampled for instruct-tuning, and trajectories are then used for RLVR to update the agent policy via offline DPO in an iterative manner.

decision-making, and interacting with external environments via tool use or other interfaces. In these contexts, the probability of generating a correct trajectory across multiple attempts diminishes significantly. As well, interacting with the environment requires additional infrastructure setup and execution time, adding further complexity to training. These complications make it difficult for RLVR to perform well in agentic settings.

To address these issues, we introduce Agent-RLVR, a framework for training language model agents that relies solely on rewards produced by the agent-environment interaction. We demonstrate the success of the method in software engineering (SWE) tasks. We choose this setting for several reasons: the setting has objectively verifiable solutions (e.g. passing unit tests), previous work such as SWE-Bench [2] has provided standardized evaluation infrastructure, and advances in this area offer significant practical value for improving developer productivity. The core of our method is the incorporation of *agent guidance*—a multi-agent framework where teacher guidance, which provides targeted information about failing code patches or test expectations, is incorporated during RL training. These hints steer the agent towards successful trajectory by providing these diverse information clues, similar to how a junior software engineer may receive guidance from a teacher or tech lead when first exploring a new codebase.

To summarize, the key contributions of our work are as follows: (1) We propose Agent-RLVR, a framework to enable effective agentic RLVR training. Agent-RLVR addresses the fundamental challenge of sparse rewards in multi-step reasoning environments by incorporating pedagogical elements that guide agents through complex solution spaces without compromising the verifiability of the ultimate reward signal. (2) We curate a dataset of SWE tasks that includes problem statements, environments, and expert guidance designed for Agent-RLVR. This dataset goes beyond traditional input-output pairs by capturing complete coding environments with integrated guidance signals, providing a rich resource for training SWE agents. (3) We empirically demonstrate the improvement of Agent-RLVR on SWE agent performance. On SWE-bench Verified, our method increases PASS@1 from 9.4% to 22.4% with just 817 training environments. This dramatic improvement with a relatively small training dataset validates the efficiency of our approach and its ability to help agents learn complex multi-step reasoning processes. We show that guidance is a critical component, as the guidance model improves over the guidance in both PASS@1 (19.8%  $\rightarrow$ 22.4%) and PASS@32 (34.2%  $\rightarrow$  38.4%). Finally, we show that the RLVR data has additional utility for reward model training, as using a reward model trained on the same RLVR data to rank k=32patch generations increases the PASS@1 to 27.8%. These results demonstrate the synergistic benefits of combining our guided RLVR approach with other advanced techniques, establishing a foundation for future research in training agents for complex real-world environments.

# 2 Agent-RLVR: An RL Framework for SWE Agents via Guidance and Environment Rewards

We introduce Agent-RLVR, a framework for training agents via RLVR. Agent-RLVR encourages agents to explore environments more effectively via agent guidance, a method of steering agents towards correct trajectories on challenging tasks. Figure [1](#page-1-0) shows an illustration of the Agent-RLVR training loop. Note that experiments in this paper describe one iteration cycle.

## 2.1 Background: Reinforcement Learning from Verifiable Rewards

In RLVR, prompts x are fed to an LLM π<sup>θ</sup> that produces responses y. A reward function r(x, y) judges prompt-response pairs and provides feedback to the LLM. Following prior works [\[3,](#page-9-2) [4\]](#page-9-3), the reward function is modulated by a KL divergence penalty that prevents the policy from drifting too far from the reference policy πref (typically the LLM before RLVR began):

$$R(x,y) = r(x,y) - \beta \log \left[ \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$
 (1)

where β is a hyperparameter. LLM parameters θ are optimized to maximize this expected reward using an algorithm like proximal policy optimization (PPO) or direct policy optimization (DPO) [\[5\]](#page-9-4). DPO removes the need to explicitly model the reward function, and instead samples pairs of winning responses y<sup>w</sup> (that produce higher rewards) and losing responses y<sup>l</sup> (that produce lower rewards) for the same input x. The DPO loss function drives the LLM towards preferred responses:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_{\theta}(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_{\theta}(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right) \right]$$
(2)

where σ is the sigmoid function. In Agent-RLVR, we leverage the environment as a source of verifiable feedback for RLVR. In this approach, the preference data D consists of code solution pairs (yw, yl) where y<sup>w</sup> passes all unit tests and y<sup>l</sup> fails one or more unit tests.

#### 2.2 Extending RLVR to Agentic Tasks with Agent-RLVR

While RLVR has shown to be successful at math and competitive coding tasks [\[1\]](#page-9-0), there are several challenges in adapting RLVR to agentic settings. Environment complexity. For math problems, it's possible to grade the answer via a simple string grader or fine-tuned LM grader. However, in agentic settings, an environment takes a significant infrastructure load and execution time to give a reward signal. Multi-turn. Agentic problems are multi-turn, requiring the model to interact with the environment in a series of steps. Problem difficulty. Agentic tasks can be very complex, requiring the model to complete a series of steps in which the agent might never succeed. While these problems are evergreen for agentic tasks, Agent-RLVR helps to mitigate the issues: generated guidance assists the agents in difficult problems and helps steer the model throughout multiple turns, and (by showing the effectiveness of offline RL) our framework enables provided environments to be run asynchronously to speed up reward generation.

In this section, we describe the training dataset, problem formulation, guidance generated, and training loop needed to execute Agent-RLVR. In Section [2.2.1](#page-2-0) and Section [2.2.2,](#page-3-0) we describe the format of the problem statements and guidance needed for Agent-RLVR. In Section [2.2.3](#page-4-0) we describe our training dataset, and in Section [2.2.4](#page-4-1) we describe the Agent-RLVR training loop.

## <span id="page-2-0"></span>2.2.1 Problem Formulation

Issue description. Each task presents the agent with a detailed GitHub issue description that outlines a bug report or feature request from real-world open-source projects. These descriptions often include error messages, expected behavior, actual behavior, and sometimes steps to reproduce the issue. The agent must carefully analyze this natural language problem statement to understand the underlying

<sup>&</sup>quot;'latex "'

#### Algorithm 1 Agent-RLVR

```
1: Input: M_{\theta} = agent model, T = teacher, \mathcal{D} = dataset of issue descriptions, repositories, and test
       suites, \mathcal{T}_{correct}, \mathcal{T}_{incorrect} = \emptyset: correct and incorrect agent trajectories
 2: Initialize \mathcal{D}_{RLVR} = \emptyset
 3: ① Initial agent attempt
 4: for d_i in dataset \mathcal{D} do
             Generate agent trajectory t_i: t_i = M_{\theta}(d_i)
 5:
             Generate reward r_i: r_i = evaluate(d_i, t_i)
 6:
             if r_i = 1 then
 7:
                   (d_i, t_i) \to \mathcal{T}_{correct}
 8:
 9:
                    (d_i, t_i) \to \mathcal{T}_{incorrect}
10:
13: Add to \mathcal{D}_{RLVR}: \{(d_i, t_{i,j}^+, t_{i,k}^-) : r_{i,j} > r_{i,k}\} \triangleright Generate preference pairs from multiple rollouts
14: ② Generate agent guidance
15: for (d_i, t_i) in \mathcal{T}_{incorrect} do
             Generate teacher guidance g_i: g_i = T(d_i, t_i)
16:
17:
             Generate new solution with guidance: t'_i = M_{\theta}(d_i, g_i)
             Evaluate new solution: r'_i = evaluate (d'_i, t'_i)
18:
19:
             if r'_i = 1 then
                   Add (d_i, t'_i, t_i) to \mathcal{D}_{RLVR}
20:
21:
22: end for
23: 3 Update policy
24: \mathcal{D}_{SFT} \subset \mathcal{D}_{\mathcal{RLVR}}

25: \mathcal{L}_{SFT}(\theta) = -\frac{1}{|\mathcal{D}_{SFT}|} \sum_{\substack{(x_i, r_i, y_i^*) \in \mathcal{D}_{SFT}}} \log p_{\theta}(y_i^* | x_i, r_i)
26: \theta_{SFT} = \arg\min_{\theta} \mathcal{L}_{SFT}(\theta)
27: \pi_{\text{ref}} = M_{\theta_{\text{SFT}}}
28: \mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{RLVR}} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right]
29: \theta_{RLVR} = \arg\min_{\theta} \mathcal{L}_{DPO}(\theta)
30: return M_{\theta_{\text{RLVR}}}
```

<span id="page-3-1"></span>technical issue, identify relevant parts of the codebase, and formulate a strategy for implementing a solution.

**Repository.** The agent is given access to a complete snapshot of a real-world Python codebase drawn from popular open-source projects. The agent must efficiently navigate the codebases and locate relevant code components. We are careful not to collect any issues from any repository in the SWE-Bench test set, to avoid any potential data contamination issues.

**Environment.** The agent operates within a fully functional Docker container that provides an interactive development environment mirroring real-world software engineering workflows. The environment allows the agent to execute code, run specific modules, and observe runtime behavior to better understand the issue. This is similar to other works where unit tests are executed e.g. for coding challenges [6, 7], however in our case, the infrastructure setup incurs additional complexity as each problem has it's own runtime environment.

**Unit tests.** Each task includes test suites that serve as the ultimate validation mechanism for the agent's solution. These tests include both regression tests that verify existing functionality remains intact, as well as focused tests that specifically check whether the issue has been resolved. These tests are used to provide verifiable feedback to the agent in RL training.

#### <span id="page-3-0"></span>2.2.2 Guidance

At the core of Agent-RLVR is the introduction of guidance – a method of assisting the agent in the trajectory rollout stage. During guidance generation, the annotator receives access to the problem statement, patch, and (in the case of environment feedback guidance) the previous trajectory

and environment feedback produced by the agent. Although the model is not required to use the guidance, it can help the model to generate the correct trajectory, especially for more difficult problem statements. This promotes active self-improvement for the model, as it is encouraged to explore additional trajectories that it otherwise would not have generated. For this work, we use an external LLM in place of human-generated guidance due to cost considerations. Note that guidance is included during train-time only, at test-time, the inference is the same with and without guidance-trained models. We show that guidance reduces reward landscape sparsity and improves agent PASS@k in Table [2](#page-6-0) and Figure [2.](#page-6-0) We include the following types of guidance.

Plan. We include guidance to assist the agent in solving the given problem. Derived from the problem statement, the plan is either a series of suggested steps to solve the problem and/or a pointer regarding the crux of the problem. Similar to other works finding that the plan helps to improve generation diversity [\[8\]](#page-9-7), we hypothesize that the plan encourages the agent to explore additional trajectories beyond the range of it's initial policy.

Environment feedback. Given the feedback from the environment, the annotator (or external model) creates guidance that corrects the previous mistake. This aims to guide the agent to reconsider a previous trajectory in which the patch caused an error or specific test to fail.

Environment interaction. We include environment interaction following Learn-by-Interact [\[9\]](#page-10-0). In particular, we guide the annotator or model to produce guidance with respect to the correct file and patch location. This is to help mitigate potential issues with agent navigation during the localization phase.

#### <span id="page-4-0"></span>2.2.3 Training Dataset

We create a training dataset for Agent-RLVR complete with environments and guidance generations. The dataset is a set of 817 agent environments for SWE tasks. Given a codebase along with an issue to be solved, the agent is tasked with editing the codebase to address this issue. This requires the agent to navigate the given codebase, coordinate changes across multiple functions, classes, and files, and complete complex reasoning while interacting with execution environments. We include issues from SWE-Gym [\[10\]](#page-10-1), and source additional ones using a similar scraping and annotation pipeline. In total, we include 27 different repos, with 593 problems from SWE-Gym and 219 self-collected problems. The problems are then fed into the Agent-RLVR pipeline to collect trajectories. The final training dataset consists of 8186 trajectories in total, out of which 4093 are positive trajectories. Instruct-tuning is done on a subset (20%) of positive trajectories and the same set of problems is used for instruct-tuning and DPO.

#### <span id="page-4-1"></span>2.2.4 Agent-RLVR Training Loop

We describe the training loop for Agent-RLVR. First, agent trajectories are generated using a scaffold. In our experiments, we use the scaffold described in Section [3.1.](#page-4-2) If the trajectory successfully generates a patch, it is validated by executing the Docker image and running the unit tests. For failing trajectories, where either the unit tests fail or the patch is not generated, the problem is reattempted with guidance. At this stage, the guidance is generated and the agent reattempts the problem with guidance and the new set of generated patches is graded. We find that the guidance augmentation enables the agent to complete more trajectories, as shown in Table [2.](#page-6-0) For each correct trajectory, an incorrect (non-guidance) trajectory for that problem is sampled to create a pair of trajectories for DPO training. If the model has not yet been fine-tuned, the model goes through instruct-tuning where a subset of 20% of the (non-guidance augmented) trajectories are sampled for SFT training. Finally, DPO training occurs using each trajectory pair as training data, concluding the training iteration. The algorithm is illustrated in Algorithm [1.](#page-3-1)

## 3 Experiments

#### <span id="page-4-2"></span>3.1 Setup

Dataset. We use the aforementioned training dataset for RLVR training, with a total of 817 environments. For guidance generation, we use an external LLM (claude-3-7-sonnet-20250219). We include both on-policy (trajectories generated by Qwen2.5-72B-Instruct) and off-policy (trajectories generated by the LLM used for guidance) data.

Trajectory collection. We sample from the model multiple times to collect trajectories, similar to works in which PASS@k is distilled into PASS@1 [\[11,](#page-10-2) [12\]](#page-10-3). Our dataset of 8186 trajectories is generated from sampling from 16 trajectories for each problem during the guidance and non-guidance trajectory sampling. 20% of the dataset is used for instruct-tuning, where we adapt the model to the task, and the entire dataset is use for DPO training.

Model training. For model training, we use the Qwen-2.5 family of models. We reserve a portion of the dataset for an instruct-tuning phase, such that the model is adapted to learn the format of the task. All models are instruct-tuned for five epochs with a learning rate of 1e-5, and sequence length of 8k. For RLVR training, we use DPO with a learning rate of 1e-6 for 1 epoch.

Evaluation dataset. We evaluate our models on the verified portion of SWE-Bench [\[2\]](#page-9-1). We use the evaluation harness provided by the authors[1](#page-0-0) , and use Modal as our infrastructure stack during evaluation. Several environments in SWE-Bench Verified have been noted to be broken by other works. We found similar issues in our experiments, where 6 instances could not be compiled. Nevertheless, we include all instances in our metrics to remain consistent with other methods.

Scaffolding. We use a basic scaffolding based off Agentless [\[13\]](#page-10-4), which is widely used for SWE agent evaluation. We pick Agentless for its popularity and maturity at the time of publication. To simplify the scaffold for our experiments, we make some minor modifications and only kept the localization and repair steps. We removed the patch selection step that require rather expensive test generation and execution, so our scaffold is significantly simpler, faster, and less compute intensive than Agentless. During the self-improvement stage with guidance, we include an extra prompt in which the model is given access to the additional hints for localization and repair.

Metrics. We report PASS@1 to compare similar methodologies and for ablation studies, as well as BEST@1 to compare with frontier models and other open source models on SWE-Bench. For PASS@1, we use greedy decoding at all steps. For PASS@k and BEST@1, we use greedy decoding for the first generation and then use temperature=0.6 for the rest of the generations. We additionally report standard deviation in Table [1.](#page-6-1)

SFT baseline. We create a strong SFT baseline (denoted as Agent-SFT) to compare. To select trajectories to train on, we filter by using the environment unit tests (filtering for trajectories that pass unit tests, aka rejection sampling [\[11\]](#page-10-2)) We rollout using the same set of 817 tasks. We train the model for 5 epochs with a learning rate of 1e-5 and cosine learning schedule. We optimize learning rate and number of epochs via grid search. Note that we do not include any trajectories with guidance in the SFT baseline, as when comparing an SFT baseline trained with and without guidance, we find that performance is better without guidance likely due to overfitting to the guidance prompt during SFT.

Reward model. To further explore test-time scaling, we train a reward model using the data generated by Agent-RLVR (including guidance). The reward model takes as input the generated patch and problem statement and produces a reward score.For the base model, we use Qwen-2.5-Coder-32B. The reward model is trained with 1e-5 learning rate for 500 steps and a warmup ratio of 0.05 with a pairwise loss. At inference time, we generate k = 32 patches and rank them with the reward model to select one final patch for evaluation.

Compute. We trained on Nvidia H100 GPUs. For the 14B and 32B experiments, we use two nodes with 8 H100 GPUs each. For the 72B experiments, we use 4 nodes with 8 H100 GPUs each. The train time is 4, 6, and 10 hours for each of the 14B, 32B and 72B experiments respectively.

#### 3.2 Evaluation Results

Agent-RLVR enables models to specialize as SWE agents. After training via Agent-RLVR, the base model (Qwen-2.5-72B-Instruct) improves from 9.4% → 22.4% on SWE-Bench Verified. This highlights that the agent can learn the task almost entirely through Agent-RLVR, and that the method enables weaker models to become strong agent models. This also demonstrates that environment feedback is an effective reward signal for agent training. We hypothesize that this effect would be magnified by an online RL algorithm, as online and on-policy RL generally outperforms offline DPO. We test different model sizes for Agent-RLVR, and find that performance generally scales linearly with model size.

<sup>1</sup> <https://github.com/SWE-bench/SWE-bench>

<span id="page-6-1"></span>

| Model                               | PASS@1 (STD) |  |
|-------------------------------------|--------------|--|
| Base Models                         |              |  |
| Qwen-2.5-Coder-14B                  | 6.8% (±2.2)  |  |
| Qwen-2.5-Coder-32B                  | 8.8% (±2.5)  |  |
| Qwen-2.5-72B-Instruct               | 9.4% (±2.5)  |  |
| Trained Models                      |              |  |
| Agent-RLVR-Qwen-2.5-Coder-14B       | 18.0% (±3.4) |  |
| Agent-RLVR-Qwen-2.5-Coder-32B       | 21.6% (±3.5) |  |
| Agent-SFT-Qwen-2.5-72B-Instruct     | 20.8% (±3.5) |  |
| Agent-RLVR-Qwen-2.5-72B-Instruct    | 22.4% (±3.6) |  |
| Agent-RLVR-RM-Qwen-2.5-72B-Instruct | 27.8% (±3.8) |  |

Table 1: Comparison between base and trained models on SWE-Bench Verified. Models trained via Agent-RLVR are better as SWE agents, e.g. from 9.4% → 22.4% for Qwen-2.5-72B-Instruct. We also incorporate an SFT baseline with rejection sampling, and an additional reward model (RM) trained with RLVR data. The reward model provides additional test-time scaling (from 22.4% → 27.8%) while incurring minimal overhead.

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Figure 2: We compare PASS@k (k = 2[0,5]) between the guidance and non-guidance versions of Agent-RLVR. We find that the guidanceaugmented method performance is higher at PASS@1 and that the performance gap increases at higher PASS@k, suggesting that the guidancetrained model is able to sample more diverse and accurate generations.

| Metric                                          | Value                |  |
|-------------------------------------------------|----------------------|--|
| Without Guidance                                |                      |  |
| # of Successful Rollouts (avg)<br>Empty Patch % | 138 (16.9%)<br>8.71% |  |
| With Guidance                                   |                      |  |
| # of Successful Rollouts (avg)<br>Empty Patch % | 165 (20.3%)<br>7.20% |  |

Table 2: We compare trajectory rollout success rate with and without guidance. When including guidance, an average of 27 trajectories are more successful and the empty patch percent reduces by 1.51%, highlighting that the agent is able to complete additional problems when given adequate guidance.

Agent-RLVR vs Agent-SFT. We compare the RLVR trained model to an SFT baseline trained using rejection sampling. We find that the RLVR model performs better, with a score of 22.4% for the RLVR trained model versus 20.8% for the SFT trained model. However, it is important to note that these two are not mutually exclusive. For example, in a post-training pipeline the same trajectories can be included in the SFT pipeline, and RL data produced by Agent-RLVR can be included during RL training for additional boosts in performance, which can be looked at as a future experiment.

Guidance is critical for Agent-RLVR. We find that guidance is needed for successful RLVR training. Figure [2](#page-6-0) highlights the gap between the model trained with and without guidance. For the non-guidance model, we perform the same training cycle but exclude any guidance-augmented data during the DPO training. We find that as PASS@k increases, the gap between the guidance and non-guidance model increases. We hypothesize that the reward landscape sparsity is a key reason for the gap between guidance and non-guidance RLVR trained models.

RLVR data can be used as an effective test-time reward model. Another benefit to gathering RLVR data is that it can be used to train an effective reward model. For this step, we train a reward model by using the same dataset of RLVR problems. We train on the repair patches only and include the problem statements during train and inference time. Finally, at test-time we generate 32 patches

| Base Model                                       | PASS@1 |  |
|--------------------------------------------------|--------|--|
| Other Models                                     |        |  |
| Qwen-2.5-72B-Instruct                            | 9.4%   |  |
| SWE-Gym-Qwen-2.5-Coder-32B [10] (OpenHands [14]) | 20.6%  |  |
| GPT-4o (Agentless [13])                          | 38.8%  |  |
| SWE-Fixer-72B [15]                               | 30.2%  |  |
| Llama3-SWE-RL-70B [16] (Agentless [13])          | 41.0%  |  |
| DeepSeek-V3 (Agentless [13])                     | 42.0%  |  |
| DeepSeek-R1 (Agentless [13])                     | 49.2%  |  |
| Claude-3.5-Sonnet (Agentless [13])               | 50.8%  |  |
| Our Models                                       |        |  |
| Agent-RLVR-Qwen-2.5-Coder-32B                    | 21.6%  |  |
| Agent-RLVR-Qwen-2.5-72B-Instruct                 | 22.4%  |  |
| Agent-RLVR-RM-Qwen-2.5-72B-Instruct              | 27.8%  |  |

Table 3: Comparison of different models on SWE-bench Verified. Agent-RLVR is effective while relying on a smaller training set, simpler scaffold, and/or less specialized training procedure. For example, SWE-RL [\[16\]](#page-10-7) has 11M training instances and SWE-Gym [\[10\]](#page-10-1) has 2k.

and use the reward model scores for filtering. We find that the reward model is able to discern between correct and incorrect patches at test-time and improves PASS@1 from 22.4% to 27.8%.

# 4 Broader Impacts and Limitations

## 4.1 Broader Impacts

Highly capable SWE agents could have a significant impact on the tech industry. We hope that in the long term, SWE agents will serve as a collaborative tool for experienced human SWEs, eliminating tedious work and accelerating a team's ability to transform ideas into production-ready code. We believe that the demand for high-quality SWE work still outstrips the supply of SWE talent, and that advancements in training SWE agents will yield a large net social good.

## 4.2 Limitations

Programming languages. Our work is limited to Python as it is the most widely used programming language, and is supported by other infrastructure components such as SWE-Bench and Agentless. We leave other languages for future work.

Dataset limitations. Our training data is limited to a relatively small set of repositories due to cost considerations. Additionally, there is (to our knowledge) only one substantial test set for SWE agents that has been human-validated for reliability. It's possible that the dataset contains contamination with respect to pre-training [\[17\]](#page-10-8), although we ensure that repositories do not overlap between training and test sets.

Online versus offline RL. In this work, we explore Agent-RLVR as an offline algorithm. However, it is possible to apply the same methods to online RL. For example, Agent-RLVR could run for several iterations with GRPO or PPO. Previous work [\[18,](#page-10-9) [1,](#page-9-0) [19\]](#page-10-10) has found that online RL is an effective way to train language models for reasoning and coding tasks. Since this requires additional infrastructure setup for the online environment, we focus on offline RL for this work.

## 5 Related Work

#### 5.1 Training software engineering agents

Our work mainly focuses on training software engineering agents, for instance, agents able to autonomously solve issues and debug coding problems in a large repository, which was first defined by SWE-Bench [\[9\]](#page-10-0). These agents are assessed on their ability to produce a patch that passes a set of unit tests, and problem statements are derived from previous pull-requests or commits in that repo. More recently, commit0 [\[20\]](#page-11-0) assesses agent ability to create an repository from scratch.

Early software engineering agent designs involved improvements on the agent scaffold, with a fixed model and no additional training. SWE-Agent [\[21\]](#page-11-1) uses an agent-computer interface to allow the agent to iteratact with the environment via tool-use. Agentless [\[13\]](#page-10-4) takes a multi-step approach with localization, repair, and validation steps. React [\[22\]](#page-11-2) incorporates tool-use with reasoning traces to generate agent trajectories.

Several works have explored training software engineering agents by keeping the scaffold constant and training the model itself. SWE-Gym [\[10\]](#page-10-1) trained the model via rejection sampling, and also explores test-time scaling for model patches. SWE-RL [\[16\]](#page-10-7) uses reinforcement learning on Github repos via GRPO and a cosine similarity scoring function. SWE-Fixer [\[15\]](#page-10-6) trained a code retrieval and code editing module for the localization and repair steps respectively.

## 5.2 Reinforcement Learning from Verifiable Rewards

RLVR (coined by [\[18\]](#page-10-9)) denotes a set of research where RLHF training methods (such as PPO [\[23\]](#page-11-3), DPO [\[5\]](#page-9-4), and GRPO [\[24\]](#page-11-4)) are applied using verifiable feedback as a reward signal. Typical RLHF pipelines involve collecting a set of human preferences to train a reward model, however, in RLVR the reward model is not used and instead the reward score is via a grader or set of unit tests. DeepSeek r1 [\[1\]](#page-9-0) shows the utility of scaling test-time thoughts, which are generated by the model during the reinforcement learning process. In other post-training pipelines, RLVR is a critical part of the training process. For example, in Tulu3, the last stage of the post-training pipeline is an RLVR stage [\[18\]](#page-10-9). In DeepSeek v3, the model goes through an RLVR stage for code and reasoning training [\[19\]](#page-10-10). In terms of guidance, PlanSearch [\[8\]](#page-9-7) shows that including a plan helps to improve search diversity in coding tasks.

#### 5.3 Progress in large language model agents

Apart from coding agents, there has been a significant amount of progress in LM agents. Similar to SWE-Bench, several benchmarks exist for enviroment-based computer-use and web agents. Some examples are OSWorld [\[25\]](#page-11-5) which provides environment infrastructure for computer-use agents, Mind2Web [\[26\]](#page-11-6) which evaluates agents on a set of open-ended website tasks, and ToolComp [\[27\]](#page-11-7) for tool-use. Works that involve training models for agentic tasks are more sparse, with UI-Tars [\[28\]](#page-11-8) training a model using screenshots as input for GUI agents. More recent works, such as OpenAI Operator and WebRL [\[29\]](#page-11-9) show the effectiveness of training a model for computer-use and web-use respectively.

## 6 Conclusion

In this paper, we discuss Agent-RLVR, a pipeline for training language models via RLVR. We introduce agent guidance, which enables efficient use of provided environments and reduces sparsity in reward signal landscapes for agents via agent self-improvement from additional environment exploration. We prove our method on SWE-Agents, achieving improved performance by training via RL and beating SFT baselines. Through analysis of test-time scaling, we find that the guidance gap widens at higher PASS@k and that the guidance-based RL data can be used to train a reward model for test-time patch selection. Our findings indicate the promise of RL for agent training and reveal potential for future avenues of environment-based RL training.

# References

- <span id="page-9-0"></span>[1] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Jun-Mei Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiaoling Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bing-Li Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dong-Li Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Jiong Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, M. Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, Ruiqi Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shao-Kang Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wen-Xia Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyu Jin, Xi-Cheng Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yi Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yu-Jing Zou, Yujia He, Yunfan Xiong, Yu-Wei Luo, Yu mei You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yao Li, Yi Zheng, Yuchen Zhu, Yunxiang Ma, Ying Tang, Yukun Zha, Yuting Yan, Zehui Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhen guo Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zi-An Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *ArXiv*, abs/2501.12948, 2025.
- <span id="page-9-1"></span>[2] Neil Chowdhury, James Aung, Chan Jun Shern, Oliver Jaffe, Dane Sherburn, Giulio Starace, Evan Mays, Rachel Dias, Marwan Aljubeh, Mia Glaese, Carlos E. Jimenez, John Yang, Leyton Ho, Tejal Patwardhan, Kevin Liu, and Aleksander Madry. Introducing SWE-bench verified, 2024.
- <span id="page-9-2"></span>[3] Nisan Stiennon, Ouyang Long, Jeffrey Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Francis Christiano. Learning to summarize with human feedback. In *Neural Information Processing Systems*, 2020.
- <span id="page-9-3"></span>[4] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. *ArXiv*, abs/2203.02155, 2022.
- <span id="page-9-4"></span>[5] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *ArXiv*, abs/2305.18290, 2023.
- <span id="page-9-5"></span>[6] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. *ArXiv*, abs/2108.07732, 2021.
- <span id="page-9-6"></span>[7] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring coding challenge competence with apps. *ArXiv*, abs/2105.09938, 2021.
- <span id="page-9-7"></span>[8] Evan Wang, Federico Cassano, Catherine Wu, Yunfeng Bai, Will Song, Vaskar Nath, Ziwen Han, Sean M. Hendryx, Summer Yue, and Hugh Zhang. Planning in natural language improves llm search for code generation. *ArXiv*, abs/2409.03733, 2024.

- <span id="page-10-0"></span>[9] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? *ArXiv*, abs/2310.06770, 2023.
- <span id="page-10-1"></span>[10] Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with swe-gym. *ArXiv*, abs/2412.21139, 2024.
- <span id="page-10-2"></span>[11] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. STar: Bootstrapping reasoning with reasoning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, *Advances in Neural Information Processing Systems*, 2022.
- <span id="page-10-3"></span>[12] Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alexa Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, A. Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (rest) for language modeling. *ArXiv*, abs/2308.08998, 2023.
- <span id="page-10-4"></span>[13] Chun Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying llm-based software engineering agents. *ArXiv*, abs/2407.01489, 2024.
- <span id="page-10-5"></span>[14] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai software developers as generalist agents. In *International Conference on Learning Representations*, 2024.
- <span id="page-10-6"></span>[15] Chengxing Xie, Bowen Li, Chang Gao, He Du, Wai Lam, Difan Zou, and Kai Chen. Swefixer: Training open-source llms for effective and efficient github issue resolution. *ArXiv*, abs/2501.05040, 2025.
- <span id="page-10-7"></span>[16] Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriele Synnaeve, Rishabh Singh, and Sida Wang. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution. *ArXiv*, abs/2502.18449, 2025.
- <span id="page-10-8"></span>[17] Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Dylan Slack, Qin Lyu, Sean M. Hendryx, Russell Kaplan, Michele Lunati, and Summer Yue. A careful examination of large language model performance on grade school arithmetic. *ArXiv*, abs/2405.00332, 2024.
- <span id="page-10-9"></span>[18] Nathan Lambert, Jacob Daniel Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, Nouha Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hanna Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. *ArXiv*, abs/2411.15124, 2024.
- <span id="page-10-10"></span>[19] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bing-Li Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dong-Li Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Jun-Mei Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, Ruiqi Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shao-Ping Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wen-Xuan Yu, Wentao Zhang, X. Q. Li, Xiangyu

- Jin, Xianzu Wang, Xiaoling Bi, Xiaodong Liu, Xiaohan Wang, Xi-Cheng Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yao Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yi Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yi-Bing Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxiang Ma, Yuting Yan, Yu-Wei Luo, Yu mei You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Zehui Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhen guo Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zi-An Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report. *ArXiv*, abs/2412.19437, 2024.
- <span id="page-11-0"></span>[20] Wenting Zhao, Nan Jiang, Celine Lee, Justin T. Chiu, Claire Cardie, Matthias Gall'e, and Alexander M. Rush. Commit0: Library generation from scratch. *ArXiv*, abs/2412.01769, 2024.
- <span id="page-11-1"></span>[21] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Adriano Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. *ArXiv*, abs/2405.15793, 2024.
- <span id="page-11-2"></span>[22] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. *ArXiv*, abs/2210.03629, 2022.
- <span id="page-11-3"></span>[23] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *ArXiv*, abs/1707.06347, 2017.
- <span id="page-11-4"></span>[24] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Jun-Mei Song, Mingchuan Zhang, Y. K. Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *ArXiv*, abs/2402.03300, 2024.
- <span id="page-11-5"></span>[25] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. *ArXiv*, abs/2404.07972, 2024.
- <span id="page-11-6"></span>[26] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. *ArXiv*, abs/2306.06070, 2023.
- <span id="page-11-7"></span>[27] Vaskar Nath, Pranav Raja, Claire Yoon, and Sean M. Hendryx. Toolcomp: A multi-tool reasoning & process supervision benchmark. *ArXiv*, abs/2501.01290, 2025.
- <span id="page-11-8"></span>[28] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haolin Chen, Zhaojian Li, Haihua Yang, Hai-Yi Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. Ui-tars: Pioneering automated gui interaction with native agents. *ArXiv*, abs/2501.12326, 2025.
- <span id="page-11-9"></span>[29] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Xinyue Yang, Jiadai Sun, Yu Yang, Shuntian Yao, Tianjie Zhang, Wei Xu, Jie Tang, and Yuxiao Dong. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. *ArXiv*, abs/2411.02337, 2024.

## A Technical Appendices and Supplementary Material

| Method               | Score |
|----------------------|-------|
| SFT with guidance    | 16.8  |
| SFT without guidance | 20.8  |

Table 4: Comparison of SFT with and without guidance.

```
You are an expert software engineer helping to generate helpful hints for
    debugging and fixing issues.
REPOSITORY: {repo}
PROBLEM STATEMENT:
{problem_statement}
PATCH:
{patch}
STACKTRACE:
{stacktrace_hint}
Based on the information above and the following hints, create a concise,
    helpful hint that would guide a developer to fix the issue efficiently:
The hint should be in the following format:
PLAN HINT: Explain what steps need to be taken based on the problem statement.
    Be specific and actionable, focusing on the core issue. Be concise yet
    comprehensive (200-300 words maximum)
ENVIRONMENT FEEDBACK HINT: Explain what needs to be fixed based on the
    stacktrace. Be specific and actionable, focusing on the core issue. Be
    concise yet comprehensive (200-300 words maximum)
ENVIRONMENT INTERACTION HINT: Clearly identify potential file(s) and location(s)
     need to be modified.
Format your hint with clear section headings (PLAN HINT, ENVIRONMENT FEEDBACK
    HINT, ENVIRONMENT INTERACTION HINT).
Each part should have 1-2 sentences, except for ENVIRONMENT INTERACTION which
    should just be a location-based hint (file paths, and usually line number,
    function name, etc.).
```

Figure 3: Guidance generation prompt. repo refers to the name of the repo, problem\_statement is the task statement, patch is the reference patch, and stacktrace\_hint is the stacktrace from the previous attempt.

```
In addition, please carefully consider the following hint, which includes a
suggested path(s) to file(s) that may be problematic.
### Hint ###
{hint}
###
Please solve the problem starting from the very beginning. Use the hint as
guidance, but do not assume any steps have already been completed. When
using the hint, you still need to explain your thought process as if you
did not have access to the hint.
```

Figure 4: Example of including guidance during trajectory generation, in which the above prompt is appended to the trajectory.