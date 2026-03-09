![](_page_0_Picture_2.jpeg)

# **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning**

**Amrith Setlur**1,3,\***, Chirag Nagpal**1,\***, Adam Fisch**<sup>2</sup> **, Xinyang Geng**<sup>2</sup> **, Jacob Eisenstein**<sup>2</sup> **, Rishabh Agarwal**<sup>2</sup> **, Alekh Agarwal**<sup>1</sup> **, Jonathan Berant**†,2 **and Aviral Kumar**†,2,3

**A promising approach for improving reasoning in large language models is to use process reward models (PRMs). PRMs provide feedback at each step of a multi-step reasoning trace, potentially improving credit assignment over outcome reward models (ORMs) that only provide feedback at the final step. However, collecting dense, per-step human labels is not scalable, and training PRMs from automatically-labeled data has thus far led to limited gains. To improve a** *base* **policy by running search against a PRM or using it as dense rewards for reinforcement learning (RL), we ask: "How should we design process rewards?". Our key insight is that, to be effective, the process reward for a step should measure** *progress***: a change in the likelihood of producing a correct response in the future, before and after taking the step, corresponding to the notion of step-level advantages in RL. Crucially, this progress should be measured under a** *prover* **policy distinct from the base policy. We theoretically characterize the set of good provers and our results show that optimizing process rewards from such provers improves exploration during test-time search and online RL. In fact, our characterization shows that weak prover policies can substantially improve a stronger base policy, which we also observe empirically. We validate our claims by training** *process advantage verifiers (PAVs)* **to predict progress under such provers, and show that compared to ORMs, test-time search against PAVs is** > 8% **more accurate, and** 1.5 − 5× **more compute-efficient. Online RL with dense rewards from PAVs enables** *one of the first results* **with** 5 − 6× **gain in sample efficiency, and** > 6% **gain in accuracy, over ORMs.**

## **1. Introduction**

Trained reward models or *verifiers* are often used to improve math reasoning in large language models, either by re-ranking solutions at test-time [\(Collins,](#page-15-0) [2000\)](#page-15-0) or via reinforcement learning (RL) [\(Uesato](#page-17-0) [et al.,](#page-17-0) [2022\)](#page-17-0). Typically, verifiers are trained to predict the outcome of an entire reasoning trace, often referred to as *outcome* reward models (ORM) [\(Cobbe et al.,](#page-15-1) [2021b;](#page-15-1) [Hosseini et al.,](#page-15-2) [2024\)](#page-15-2). However, ORMs only provide a sparse signal of correctness, which can be hard to learn from and inefficient to search against. This challenge is alleviated by fine-grained supervision, in theory. For reasoning, prior works train *process* reward models (PRMs) that assign intermediate rewards after each step of search [\(Snell](#page-16-0) [et al.,](#page-16-0) [2024\)](#page-16-0) or during RL. While [Lightman et al.](#page-16-1) [\(2023\)](#page-16-1) obtains PRM annotations from human raters, this approach is not scalable. More recent works [\(Luo et al.,](#page-16-2) [2024;](#page-16-2) [Wang et al.,](#page-17-1) [2024\)](#page-17-1) train PRMs to predict automatically-generated annotations that estimate future success of solving the problem, akin to value functions in RL. So far, automated PRMs, especially as dense rewards in RL, only improve by 1-2% over ORMs [\(Shao et al.,](#page-16-3) [2024\)](#page-16-3), raising serious doubts over their utility.

To resolve these uncertainties, in this paper, we train PRMs with automated annotations, such that optimizing the dense rewards from trained PRMs can improve a *base* policy compute- and sampleefficiently, during test-time search and online RL. For this, we first ask: **(i)** what should the *per-step* process rewards measure, and **(ii)** what kind of automated data collection strategy should we use to train PRMs that predict this measure. For **(i)**, conventional belief [\(Lightman et al.,](#page-16-1) [2023;](#page-16-1) [Uesato et al.,](#page-17-0) [2022\)](#page-17-0)

<sup>1</sup>Google Research, <sup>2</sup>Google DeepMind, <sup>3</sup>Carnegie Mellon University, \*Equal contribution, †Equal advising

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1 | *Process advantage verifiers (PAV)*: Process reward for a step is defined as progress (advantage) under the prover policy, *i.e.*, change in prover policy's success rate before and after the step. (a): The base policy samples both correct ① and incorrect ② steps but struggles to succeed from either. A strong prover policy completes the solution from both steps, and is unable to adequately reflect progress made by ① and ② (both scored 0.0). Conversely, a complementary prover policy distinguishes ①, ② more prominently (only succeeds from ①). (b,c): Compared to ORMs, PAVs are 5x more compute efficient, 10% more accurate in test-time search, and 6x more sample efficient, 7% more accurate for online reinforcement learning (RL).

has been to measure mathematical correctness or relevance of steps. But, it is unclear if this supervision yields the most improvement in the base policy (e.g., a policy may need to generate simpler, repetitive, and even incorrect steps to explore and discover the final answer during test-time search and RL). **Our key insight** is that per-step, process rewards that measure a notion of progress: change in the likelihood of arriving at a correct final answer before and after taking the step, are effective, for both test-time beam search and online RL. Reinforcing steps that make progress regardless of whether they appear in a correct or incorrect trace diversifies the **exploration** of possible answers at initial steps, which is crucial when the approach to solve a problem is not clear. Formally, such rewards correspond to per-step advantages of steps from the RL literature (Sutton and Barto, 2018). We empirically show that using advantages in addition to ORM rewards outperforms the typical use of future probabilities of success or Q-values (Wang et al., 2024) for both search and RL. This is because, when given a combinatorial space of responses, under bounded computational and sampling constraints, Q-values mainly "exploit" states whereas advantages also "explore" steps that make the most progress towards the final answer (Fig. 2).

To answer (ii), we first note that advantages under a poor base policy are  $\approx 0$  on most steps, and thus will not be informative for search or RL. In addition, regardless of the strength of the base policy, using its own per-step advantages as process rewards in RL will result in base policy updates equivalent to *only* 

using outcome rewards for RL (since a standard policy gradient algorithm already computes advantages). Hence, we propose to use advantages estimated via rollouts under a different *prover policy* as process rewards (Fig. 1(a)). How should we choose this prover policy? A natural guess would be to use a very capable prover. However, we show advantages under an overly capable prover policy, that can succeed from any step, fail to distinguish good and bad steps. A similar argument holds for very weak provers.

In theory, we formalize this intuition to define good provers as policies that are *complementary* to the base policy (*i.e.*, policies with advantages that can contrast steps produced by the base policy sufficiently), while still producing step-level advantages correlated with those of the base policy. For *e.g.*, for Best-of-K policies (Nakano et al., 2021) corresponding to a base policy, we empirically find that provers corresponding to K > 1 (but not too large) are more capable at improving the base policy. Contrary to intuition, the set of complementary provers also contains policies that are worse than the base policy. To predict the advantages of such provers we train dense verifiers, called *process advantage verifiers* (*PAVs*), that accelerate sample and compute efficiency of RL and search.

With the conceptual design of PAVs in place, we prescribe practical workflows for training PAVs and demonstrate their efficacy on a series of 2B, 9B, and 27B Gemma2 models (Gemma Team et al., 2024). PAV training data is gathered by sampling "seed" solution traces from the prover and partial rollouts from the same to estimate the *Q*-value at each prefix of the seed trace. Our workflow prescribes favorable ratios for seed and partial rollouts. Our first set of empirical results show that for an equal budget on test-time compute, beam search against trained PAVs is >8% better in accuracy, and 1.5 – 5× more compute efficient compared to re-ranking complete traces against an ORM (Fig. 1(b)). Dense rewards from PAVs improve the efficiency of step-level exploration during search by pruning the combinatorial space of solutions aggressively and honing in on a diverse set of possible sequences. Finally, we demonstrate *for the first time*, that using PAVs as dense rewards in RL scales up data efficiency by 6× compared to only using outcome rewards (Fig. 1(c)). Moreover, base policies trained with PAVs also achieve 8× better Pass @N performance (probability of sampling the correct solution in N attempts), and consequently afford a higher ceiling on the performance of any test-time re-ranker. Finally, running RL with PAVs discovers solutions to hard problems that sampling from the SFT policy with a very large budget can't solve.

#### <span id="page-2-0"></span>2. Preliminaries, Definitions, and Notation

Following protocols from Lightman et al. (2023); Uesato et al. (2022), a reasoning trace from an LLM consists of multiple logical steps separated by a demarcation token. An outcome reward model (ORM) is a trained verifier that assigns a numerical score after the last step of the trace, and a process reward model (PRM) is a trained verifier that scores each step of the trace individually.

**Problem setup and notation.** Given a math problem  $x \in \mathcal{X}$ , our goal is to improve a *base policy*  $\pi$  that samples a response  $y \sim \pi(\cdot \mid x)$  in the set  $\mathcal{Y}$ . A response y consists of multiple reasoning steps (maximum H), separated by a delimiter ('next line' in our case), *i.e.*,  $y = (a_1, a_2, \ldots, a_H)$ . Since sampling is auto-regressive, we can view each step as an action taken by the agent  $\pi$  in a Markov decision process (MDP) with deterministic dynamics. Specifically, we treat the prefix  $(x, a_1, \ldots, a_{h-1})$  as the current *state*  $s_h$  and next step  $a_h \sim \pi(\cdot \mid x)$  as the *action* taken by  $\pi$  at  $s_h$ , resulting in the next state  $s_{h+1}$ . For problem x, with ground-truth response  $y_x^*$ , we can evaluate the accuracy of  $\pi$  by running a regular expression match on the final answer (Hendrycks et al., 2021):  $\text{Rex}(y, y_x^*) \mapsto \{0, 1\}$ , *i.e.*, accuracy is given by  $\mathbb{E}_{y \sim \pi(\cdot \mid x)} \left[ \text{Rex}(y, y_x^*) \right]$ . Now, given a dataset  $\mathcal{D} = \{(x_i, y_{x_i}^*)\}_i$  of problem-solution pairs, the main goal is to learn a good base policy by optimizing this outcome reward on  $\mathcal{D}$ . Next, we see how we can leverage the final answer verifier Rex available on  $\mathcal{D}$  to train ORMs and PRMs.

Outcome reward model (ORM). Given a response y, an ORM estimates the ground-truth correctness  $\text{Rex}(y,y_x^\star)$ . To train such a model we first take problems in  $\mathcal{D}$ , and collect training data of the form  $\{(x,y\sim\pi(\cdot\mid x),\text{Rex}(y,y_x^\star))\}$ . Then we train an ORM that takes as input a problem-response pair (x,y) and predicts  $\text{Rex}(y,y_x^\star)$ . At test time, when  $y_x^\star$  is unknown, the ORM is used to score candidate solutions revealed by test-time search. Given a base policy  $\pi$ , a Best-of-K policy:  $\text{BoK}(\pi)$ , is a policy that samples K responses from  $\pi$ , scores them against an ORM, and returns the one with the highest score. Whenever the ORM matches Rex, the performance of  $\text{BoK}(\pi)$  is referred to as Pass @K. Furthermore, when the likelihood of  $\pi$  solving problem K is K, then for K by this likelihood is given by the expression:  $1-(1-p_x)^K$ . In general, this is larger than K, making K by stronger than K for K.

**Standard process reward models (PRMs).** A PRM scores every step  $a_h$  in a multi-step response  $y \sim \pi$  (e.g., in Lightman et al. (2023) PRMs are trained to score correct steps over incorrect and irrelevant ones). But, unlike ORMs, which only require Rex for data collection, PRM training data requires expensive step-level human annotations. Prior works (Luo et al., 2024; Wang et al., 2024) attempted to scale process rewards automatically by sampling from the model to provide a heuristic understanding of when a step is actually correct. In particular, they evaluate a prefix by computing the expected future accuracy of multiple completions sampled from  $\pi$ , after conditioning on the prefix, *i.e.*, value function  $Q^{\pi}$  (Eq. 1) from RL. Similarly, we define  $V^{\pi}(s_h) := \mathbb{E}_{a_h \sim \pi(\cdot|s_h)} Q^{\pi}(s_h, a_h)$  as value of state  $s_h$ . These works use  $Q^{\pi}$  as the PRM that assigns a score of  $Q^{\pi}(s_h, a_h)$  to the action  $a_h$ , at state  $s_h$ .

<span id="page-3-0"></span>
$$Q^{\pi}(\underbrace{(\mathbf{x}, a_{1}, \dots, a_{h-1})}_{\text{state } \mathbf{s}_{h}}, \underbrace{a_{h}}_{\text{action } a_{h}}) = \underbrace{\mathbb{E}_{a_{h+1}, \dots, a_{H} \sim \pi(\cdot | \mathbf{s}_{h}, a_{h})} \left[ \text{Rex}\left((a_{1}, \dots, a_{H}), \mathbf{y}_{\mathbf{x}}^{\star}\right) \right]}_{\text{likelihood of future success}}, \tag{1}$$

Using PRMs for beam search at test-time. Given a PRM, a natural way to spend test-time compute is to use it as a step-level re-ranker within a beam search procedure (Snell et al., 2024). For each problem, at step 0, a beam of maximum width B, is initialized with a single state consisting of just the problem. At step h, a beam contains partial responses unrolled till a set of states or prefixes  $\{s_i\}_{i=1}^B$ . From each state  $s_i$  in this set, C independent actions or steps  $\{a_{i,j}\}_{j=1}^C$  are sampled from  $\pi(\cdot \mid s_i)$ , each of which leads to a new state. Process rewards from PRMs assign a score to every new state  $(s_i, a_{i,j})$ , and only the states corresponding to the top B values are retained in the beam for the next step.

# 3. How Should we Define Process Rewards and Why?

Ultimately, we are interested in test-time search and RL methods that can most efficiently and reliably discover solution traces with the correct final answer, thus maximizing Rex. To this end, *process rewards should serve as step-level supervision to indirectly maximize outcome-level* Rex. Our position contrasts with conventional belief that process rewards should mainly evaluate mathematical correctness or relevance of individual steps (Lightman et al., 2023; Uesato et al., 2022), since LLMs might need to generate trivial or repetitive intermediate steps in order to discover a trace with the correct final answer. With this insight, in this section we approach the design of dense automated step rewards as a form of supervision to be used in conjunction with sparse outcome rewards to improve the base policy.

In an MDP, a starting point to design step-level dense feedback that is eventually meant to optimize a sparse outcome reward Rex is to consider the notion of a *potential function* (Ng et al., 1999): in our case, this is a function that summarizes the difference between some statistic of the policy at the future state and the same statistic computed at the current state. By appealing to this framework, in Sec. 3.1, we

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

**Figure 2** | *Issues with using Q-values as process rewards*: (a): Unlike  $A^{\pi}$ ,  $Q^{\pi}$  mixes action evaluation with the *Q*-value of the previous state. Beam search with  $Q^{\pi}$  exploits high-likelihood states, while adding  $A^{\pi}$  (*e.g.*,  $Q^{\pi} + \alpha A^{\pi}$  in Eq. 5) aids in exploring states reached by making actions that induce progress, *i.e.*, increase likelihood of success. (b):  $Q^{\mu}$  from a strong prover  $\mu$  can assign unmerited bonuses to trivial actions.

show that advantages – not value functions (Luo et al., 2024; Wang et al., 2024) – that measure a notion of "progress" at each new step are more appropriate for use as dense rewards in search and RL (primarily for exploration). Then in Secs. 3.3 and 3.4, we show that this progress or advantage vakue is measured best under a policy  $\mu$ , different from the base policy  $\pi$ . We call this policy  $\mu$ , the *prover policy*.

#### <span id="page-4-1"></span>3.1. Process Rewards Should be Advantages, Not Value Functions

To understand the relationship to potential functions, we first study test-time beam search, and present some challenges with the reward design of Snell et al. (2024), that uses value function  $Q^{\pi}(s, a)$  of the base policy  $\pi$  to reward action a at state s. Consider the example in Fig. 2(a), where from the 2 states in the beam, we sample 3 actions. If we pick next states purely based on highest values of  $Q^{\pi}$ , we would be comparing steps sampled from different states (e.g.,  $a_{1,1}$  vs.  $a_{2,1}$ ) against each other. Clearly, a reduction in expected final outcome, i.e.,  $Q^{\pi}(s_1, a_{1,1}) - V^{\pi}(s_1)$ , means that  $a_{1,1}$  by itself has a negative effect of -0.05 on the probability of success from  $s_1$ , whereas  $a_{2,1}$  has a positive effect of +0.20 from  $s_2$ . However, expanding the beam based on absolute values of  $Q^{\pi}$  retains the action that makes negative progress, and removes state  $s_2$  from the beam (as beam size is 2). In other words,  $Q^{\pi}$  fails to decouple the "evaluation" of an action (step), from the "promise" shown by the previous state. This will not be an issue for every problem, and particularly not when the beam capacity is unbounded, but under finite computational and sampling constraints, using  $Q^{\pi}$  might retain states with potentially unfavorable steps that hurt the overall likelihood of success. If we could also also utilize the progress made by the previous step along with the likelihood of success  $Q^{\pi}$  when deciding what to retain in the beam, then we can address this tradeoff.

How can we measure the "progress" made by a step? One approach is to consider the relative increase/decrease in the likelihood of success, before and after the step. This notion is formalized by the advantage (Eq. 2) of a step under policy  $\pi$ . Furthermore, since advantages can attach either positive or negative values to a step, training the base policy against advantages supervises the base policy when it generates a step that makes progress (where  $A^{\pi} > 0$ ), and also when it fails to produce one, employing a "negative gradient" that speeds up RL training (Tajwar et al., 2024).

<span id="page-4-2"></span>
$$A^{\pi}(s_h, a_h) := Q^{\pi}(s_h, a_h) - V^{\pi}(s_h) = Q^{\pi}(s_h, a_h) - Q^{\pi}(s_{h-1}, a_{h-1}).$$
 (2)

Recall that since we view process rewards as potential functions in the MDP, they can be computed under

any policy  $\mu$ , which can be the base policy. However, in the above example, reasons for which  $Q^{\pi}$  is a seemingly unfit choice for process rewards also apply to  $Q^{\mu}$ . Nevertheless, we can possibly use advantage under  $\mu$ :  $A^{\mu}$ , which measures the progress made by a step to improve the likelihood of success under  $\mu$ . In that case, how should we choose this policy  $\mu$ , that we call the prover policy, and should it be necessarily different from base policy  $\pi$ ? Before diving into the choice of  $\mu$ , we discuss a more pertinent question: how should we use  $A^{\mu}$  in conjunction with outcome rewards for improving the base policy  $\pi$ ? We will then formally reason about the choice of  $\mu$  in Secs. 3.3 and 3.4.

#### <span id="page-5-2"></span>3.2. Our Approach: Process Advantage Verifiers (PAV)

For building an approach that uses process rewards  $A^{\mu}$  together with the outcome reward Rex to improve the base policy  $\pi$ , we situate ourselves in the context of improving  $\pi$  with online RL. If all we had was access to Rex on  $\mathcal{D}$ , the standard RL objective is given by:

<span id="page-5-3"></span>
$$\ell_{\text{ORM-RL}}(\pi) := \mathbb{E}_{\mathbf{x} \sim \mathcal{D}, (a_1, \dots, a_H) \sim \pi(\cdot | \mathbf{x})} \left[ \text{Rex} \left( (\mathbf{x}, a_1, \dots, a_H), \mathbf{y}_{\mathbf{x}}^{\star} \right) \right]. \tag{3}$$

Inspired by how reward bonuses (and potential functions) are additive (Bellemare et al., 2016; Ng et al., 1999), one way to use process rewards  $A^{\mu}$  is to combine it with the standard RL objective as:

<span id="page-5-1"></span><span id="page-5-0"></span>
$$\ell_{\text{PAV-RL}}^{\pi'}(\pi) := \ell_{\text{ORM-RL}}(\pi) + \alpha \cdot \sum_{h=1}^{H} \mathbb{E}_{s_h \sim d_h^{\pi'}} \mathbb{E}_{a_h \sim \pi(\cdot | s_h)} \left[ A^{\mu}(s_h, a_h) \right]$$
(4)

The term in red is the difference in likelihoods of success of the prover  $\mu$ , summed over consecutive steps (a notion of *progress*). Here,  $d_h^{\pi'}$  denotes the distribution over states at step h, visited by the old policy  $\pi'$  (policy at previous iterate). Following policy gradient derivations (Williams, 1992):

$$\left| \nabla_{\pi} \ell_{\text{PAV-RL}}^{\pi'}(\pi) \right|_{\pi' = \pi} = \sum_{h=1}^{H} \left| \nabla_{\pi} \log \pi(a_h \mid s_h) \cdot \underbrace{\left( Q^{\pi}(s_h, a_h) + \alpha \cdot A^{\mu}(s_h, a_h) \right)}_{\text{effective reward}} \right|$$
 (5)

At a glance, we can view  $Q^{\pi}(s_h, a_h) + \alpha A^{\mu}(s_h, a_h)$  as the effective reward for step  $a_h$  when scored against a combination of the outcome evaluation Rex, i.e.,  $Q^{\pi}$ , and process rewards  $A^{\mu}$ . Thus, we can optimize Eq. 4 indirectly via (a) running beam-search against the effective reward; or (b) online RL where the policy gradients are given by Eq. 5. For either of these, we need access to verifiers that are trained to predict the advantage  $A^{\mu}(s_h, a_h)$  under the prover. We refer to these verifiers as **process advantage verifiers** (**PAVs**). In Sec. 4.2 we describe how to train PAVs, but now we use the above formulation to reason about how to choose prover  $\mu$  that is most effective at improving base  $\pi$ .

We also remark that the term in red resembles prior work on imitation learning via policy optimization (Ross and Bagnell, 2014; Sun et al., 2017), where the main aim is to learn a policy  $\pi$  that imitates the prover  $\mu$ , or to improve upon it to some extent. Of course, this is limiting since our goal is to not just take actions that perform at a similar level as  $\mu$ , but to improve the base policy even further, and using a combination of  $Q^{\pi}$  and  $A^{\mu}$  is critical towards this goal.

How should we choose the prover  $\mu$ ? Perhaps a natural starting point is to set the prover to be identical to the base policy, *i.e.*,  $\mu = \pi$ , which produces process rewards that prior works have considered Shao et al. (2024). However, setting  $A^{\pi} = A^{\mu}$  in Eq. 5 results in exactly the same policy gradient update as only optimizing outcome evaluation Rex. Moreover, for a poor base policy  $\pi$ , where  $Q^{\pi} \approx 0$  on most states,

the term  $A^{\pi}$  would also be  $\approx$  0, and hence running beam search with the effective rewards would not be informative at all. Hence, *a better approach is to use a different prover policy*, but a very weak prover  $\mu$  will likely run into similar issues as a poor base policy. We could instead use a very capable prover  $\mu$ , but unfortunately even this may not be any better than optimizing only the outcome reward either. To see why, consider a scenario where  $\pi$ 's response contains an intermediate step that does not help make progress towards the solution (*e.g.*,  $\pi$  simply restates the question, see Fig. 2(b)). Here,  $Q^{\mu}$  for a capable prover before and after this irrelevant step will be identical since  $\mu$  can succeed from either step. This means that  $\mu$  fails to distinguish steps, resulting in  $A^{\mu} \approx 0$  in most cases. Training with this process reward during RL will then lead to gradients that are equivalent to those observed when purely optimizing  $\ell_{\text{ORM-RL}}$ . In fact, empirically, we observe that online RL with  $Q^{\mu}$  from strong provers leads to polices that only produce re-phrasings of the question (App. G) and do not succeed at solving the question. Clearly, *any* policy different from the base policy cannot serve as a prover. So, how do we identify a set of good provers? Can they indeed be weaker than the base policy? We answer next.

## Takeaway: What should process rewards measure during test-time search and online RL?

- Process rewards should correspond to progress, or **advantage**, as opposed to absolute *Q*-values, for a better explore-exploit tradeoff during beam search and online RL.
- <span id="page-6-0"></span>• Advantages should be computed using a **prover** policy, different from the base policy.

#### 3.3. Analysis in a Didactic Setting: Learning a Planted Sub-sequence

In this section, we aim to characterize prover policies that are effective in improving the base policy. To do so, we first introduce a didactic example, representative of real reasoning scenarios to illustrate the main intuition. Then, we will formalize these intuitions in the form of theoretical results.

**Didactic example setup.** Given an unknown sub-sequence  $y^*$  consisting of tokens from vocabulary  $\mathcal{V} := \{1, 2, \dots, 15\}$ , we train a policy  $\pi$  to produce a response which contains this sub-sequence. The task completion reward is terminal and sparse, *i.e.*,  $r(y, y^*) = 1$  for a y if and only if  $y^*$  appears in y. By design, the reward  $r(y, y^*)$  resembles outcome reward  $\text{Rex}(y, y^*_x)$  in Sec. 2. The prover policy  $\mu$  is a procedural policy, parameterized by a scalar  $\gamma > 0$  (details in App. B). As  $\gamma$  increases, the performance of  $\mu$  improves and  $\to 1$  as  $\gamma \to \infty$ . For simplicity, we assume oracle access to ground-truth  $A^{\mu}$  and  $Q^{\pi}$ , and alleviate errors from learned verifiers approximating these values.

- (1) RL with effective reward  $Q^{\pi} + \alpha A^{\mu}$  is  $10 \times$  more sample-efficient than only outcome reward. In Fig. 3(a), we first note that training  $\pi$  with this effective reward under a prover  $\mu$  with strength  $\gamma = 10$ , produces optimal performance (100% accuracy) in 350 iterations, despite starting from a mediocre initialization for  $\pi$  ( $\gamma = 5.0$ ). Training with only outcome reward is ineffective. More importantly, in Fig. 3(b), we note that effective rewards only help for a set of provers, in  $\gamma \in [8.0, 15.0]$ . Outside this range, we observed advantages  $A^{\mu}$  were close to 0 on most states, either because  $\mu$  was poor (small  $\gamma$ ) and was unable to generate  $\gamma$  even when  $\pi$  got the sequence partially correct, or because  $\mu$  was strong (large  $\gamma$ ) that it generated  $\gamma$  with almost equal likelihood from all prefixes.
- (2) Effective reward improves Pass @N by  $5 \times$  over only outcome reward. We report the "Pass @N" performance in Fig. 3(c), which measures the maximum reward r across N traces sampled i.i.d. from  $\pi$  and hence, represents the ceiling on the performance of any test-time search method that picks a single response from multiple draws (e.g., as in Best-of-N). For a policy trained with the effective reward for 100 iterations, the Pass @N performance grows  $5 \times$  faster with N, compared to the policy trained with only

<span id="page-7-1"></span>![](_page_7_Figure_1.jpeg)

**Figure 3** | *Results for our didactic analysis:* (a): We train base policy via RL with either effective reward  $Q^{\pi} + \alpha A^{\mu}$ , or the typical  $Q^{\pi}$  (computed via Monte-Carlo sampling). (b): We vary the strength  $\gamma$  of the prover  $\mu$  used to compute advantages  $A^{\mu}$  in the effective reward, and plot the base policy accuracy averaged over the RL run. (c): We plot the max score out of N responses (Pass @N) sampled i.i.d. from an undertrained base policy (iter 100) .

the outcome reward. Due to only sparse feedback, the latter policy does not learn to sample partially correct  $y^*$ , whereas a policy trained with the effective reward produces partially correct  $y^*$ , and is able to sample the complete  $y^*$  with higher likelihood during Pass @N.

#### Takeaway: Online RL with process rewards from different prover policies.

Effective rewards  $Q^{\pi} + \alpha A^{\mu}$  from prover  $\mu$ : (i) improve sample efficiency of online RL, and (ii) yield policies with better Pass @N performance, over using only outcome rewards. But, advantages of very capable or poor  $\mu$  do not improve base policy beyond outcome rewards.

#### <span id="page-7-0"></span>3.4. Theory: Provers Complementary to the Base Policy Boost Improvement

From our didactic analysis, it is clear that process rewards  $A^{\mu}$  under different provers  $\mu$  disparately affect the base policy that optimizes  $Q^{\pi} + \alpha A^{\mu}$  via online RL. We now present a formal analysis of why this happens and characterize a class of provers that can guarantee non-trivial improvements to the base policy. For simplicity, we assume oracle access to  $Q^{\pi}$ ,  $A^{\mu}$  at every state-action pair  $(s_h, a_h)$  and prove our result in the tabular RL setting, where the policy class is parameterized using the softmax parameterization in Agarwal et al. (2021). Proofs for this section are in App. F.

Main intuitions. We expect a prover  $\mu$  to improve a base policy  $\pi$  only when  $\mu$  is able to distinguish different actions taken by  $\pi$ , by attaining sufficiently varying advantage values  $A^{\mu}(s_h, a)$  for actions a at state  $s_h$ . This can be formalized under the notion of sufficiently large variance across actions,  $\mathbb{V}_{a\sim\pi}\left[A^{\mu}(s_h,a)\right]$ . In that case, can we simply use a policy with large advantage variance under any measure? No, because when the prover  $\mu$  ranks actions at a given state very differently compared to the base policy  $\pi$  (e.g., if  $A^{\mu}$  and  $A^{\pi}$  are opposite), then effective rewards  $Q^{\mu} + \alpha A^{\pi}$  will be less reliable due to conflicting learning signals. Thus, we want  $\mathbb{E}_{\pi}\left[\langle A^{\mu}, A^{\pi} \rangle\right]$  to not be too negative, so that  $\mu$  and  $\pi$  are reasonably aligned on their assessment of steps from  $\pi$ .

In Theorem 3.1, we present our result on policy improvement where the base policy is updated with natural policy gradient (Kakade, 2001a):  $\pi_{t+1}(a \mid s_h) \propto \exp(\gamma \cdot (Q^{\pi}(s_h, a) + A^{\mu}(s_h, a)))$ . We note that in this idealized update rule, swapping Q values (of  $\mu$  or  $\pi$ ) with advantages does not affect the update since we assume access to all possible actions when running the update. Nonetheless, despite this simplifying assumption, the analysis is able to uncover good choices for the prover policy  $\mu$  for computing process

reward  $A^{\mu}$ , and is orthogonal to the design consideration of advantages or Q-values as process rewards that we have discussed so far in this paper. Theorem 3.1 formalizes our intuition by showing that policy improvement at iteration t, grows as the variance in  $A^{\mu}$  values increases (higher distinguishability) and reduces when  $A^{\mu}$  and  $A^{\pi}$  become extremely misaligned. This will then allow us to discuss a special case for the case of Best-of-K policies as provers as an immediate corollary.

<span id="page-8-0"></span>**Theorem 3.1** (Lower bound on policy improvement; informal). For base policy iterate  $\pi_t$ , after one step of policy update, with learning rate  $\gamma \ll 1$ , the improvement over a distribution of states  $\rho$ :

$$\mathbb{E}_{s \sim \rho} \left[ V^{\pi_{t+1}}(s) - V^{\pi_t}(s) \right] \gtrsim \gamma \cdot \mathbb{E}_{s \sim \rho} \mathbb{V}_{a \sim \pi_t} \left[ A^{\mu}(s, a) \right] + \gamma \cdot \mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_t} \left[ A^{\mu}(s, a) A^{\pi_t}(s, a) \right]$$

$$\qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad$$

It may seem that the base policy  $\pi$  can only learn from an improved prover  $\mu$ , but our result shows that a **weak prover can also amplify a stronger base policy**, since a weak prover  $\mu$  may have a lower average of  $Q^{\mu}$  under its own measure, but still have higher variance across  $Q^{\mu}$  (compared to  $Q^{\pi}$ ) when evaluated under  $\pi$  (see Proposition F.1 in App. F.5 for formal discussion). This tells us that *rewarding progress under a prover is different from typical knowledge distillation or imitation learning algorithms* (Hinton, 2015; Rusu et al., 2015) that in most cases remain upper bounded by the performance of the stronger teacher. So provers cannot be characterized purely by strength, what is a class of provers that is a reasonable starting point if we were to improve any base policy  $\pi$ ?

The policy class of "Best-of-K" (computed over base policies) contain complementary provers. A good starting point to identify good provers for a base policy  $\pi$ , is the class of Best-of-K policies or BoK( $\pi$ ). Recall from Sec. 2 that the performance of BoK( $\pi$ ) increases monotonically with K. Applying Theorem 3.1 to this class, we arrive at Remark 3.1 that recommends using BoK( $\pi$ ) with K > 1 as a prover policy for a poor base policy  $\pi$ . However, K cannot be too large always since when  $Q^{\pi}(s,a) \approx 1$ , increasing K too much can hurt distinguishability of different steps at that state. In the next section, we empirically note that the policies in the class of BoK( $\pi$ ) indeed induce different performance gains when used as prover policies, and we find Bo4 to be a good choice for test-time search over most base policies.

<span id="page-8-1"></span>**Remark 3.1.** When  $Q^{\pi}(s, a) = O(1/K)$ ,  $\forall s$ , a, using  $BoK(\pi)$  as a prover for base  $\pi$  improves distinguishability (and improvement) by  $\Omega(K^2)$ , and make alignment worse at most by O(K).

#### Takeaway: Formal characterization of good prover policies that improve the base policy.

Provers with advantages that can **distinguish** actions taken by the base policy (more strongly than the base policy itself) but are **not too misaligned** from the base, boost improvements on each update of the base policy. We call such policies *complementary provers*. BoK( $\pi$ ) for any base policy  $\pi$  for K > 1 can provide a good starting choice of prover policies.

# <span id="page-8-2"></span>4. Results: Scaling Test-Time Compute with PAVs

Now, we study how process verifiers can scale up test-time compute. While our derivations from Sec. 3.2 were with RL, we can also use the *effective reward*  $Q^{\pi}(s_h, a_h) + \alpha \cdot A^{\mu}(s_h, a_h)$  for running beam search over intermediate steps sampled from base policy  $\pi$ . To do so, we train a process advantage verifier to predict  $A^{\mu}$ , along with a process reward model  $Q^{\pi}$ . PAV training is done using procedures discussed in Sec. 4.2. While the candidates of the beam are selected using a combination of both the PAV and the PRM  $Q^{\pi}$ ,

<span id="page-9-0"></span>![](_page_9_Figure_1.jpeg)

Figure 4 | For test-time search, PAVs are 8 – 10% more accurate and 1.5 –  $5\times$  more compute efficient over ORMs: On samples from (a) Gemma-2B, (b) 9B, and (c) 27B SFT policies, we run test-time beam search with the estimate of effective reward  $Q^{\pi} + \alpha A^{\mu}$  (PAV), where  $\mu$  is the Bo4( $\pi$ ) policy. We compare beam search performance with best-of-N, re-ranking with a trained outcome verifier (ORM), or the oracle Rex (Pass @N).

the final candidate is selected using the outcome reward prediction from  $Q^{\pi}$  itself (i.e., we repurpose the PRM representing  $Q^{\pi}$  as an ORM). For clarity, we abuse notation and refer to the estimated effective reward (ORM +  $\alpha$  PAV) as PAV directly.

**Setup**. We finetune Gemma 2B, 9B, and 27B (Gemma Team et al., 2024) on MATH (Hendrycks et al., 2021) via supervised fine-tuning (SFT) to get three base policies. The set of provers consists of the three base SFT policies themselves as well as their best-of-K policies for different values of  $K \in \{2^0, \ldots, 2^5\}$ . Additional details for the experiments in this section are in App. C.

#### <span id="page-9-1"></span>**4.1.** PAVs Scale Test-Time Compute by $5 - 10 \times$ Over ORMs

**Result 1: PAVs are more compute efficient than ORMs.** In Fig. 4, we plot the performance of beam search with PAVs for different sizes of the beam N, and compare it with best-of-N using ORMs, i.e., sampling N complete solutions from the base policy and returning the one with the highest ORM score. To compare PAVs and ORMs, we evaluate the *compute efficiency* of PAVs over ORMs, given by the ratio of total compute needed by PAVs to obtain the same performance as running best-of-128 with ORM. Even when accounting for the fact that running beam search with PAVs does require additional compute *per solution trace* (since each element in the beam samples C = 3 next steps, before scoring and pruning the beam), PAVs are able to scale the compute efficiency by  $\mathbf{10x}$  over ORMs for Gemma-2B, 9B base models, and by  $5 \times$  for Gemma-27B model. We use  $\mathbf{BoK}(\pi)$  with K = 4 as the prover policy for all base policies  $\pi$ .

We also compare performance with beam search using process verifiers that only predict  $Q^{\pi}$ , and best-of-N where the ORM is replaced with PAV (PAV-as-ORM). At N=128, similar to Luo et al. (2024), we note a similar gain of 4% for "PAV-as-ORM" Fig. 5(a) over only ORMs, for base Gemma-9B  $\pi$ . When comparing beam search with  $Q^{\pi}$  (Snell et al., 2024), we find that PAVs scale compute efficiency by 8×. Evidently, advantages from the prover in the effective reward positively impact the beam search. Why does  $A^{\mu}$  help, and for what choice of the prover  $\mu$ ?

Result 2: Beam search with too weak/strong provers is sub-optimal. In Fig. 5(b), for the setting when the base policy  $\pi$  is a Gemma-2B SFT model, we compare beam search with PAVs where the provers are given by BoK( $\pi$ ), for different values of K. Recall that as K increases, BoK( $\pi$ ) becomes stronger. Corroborating our analysis in Sec. 3.4, our results show that neither too weak (Bo2) or too strong (Bo32) provers perform best. Instead, across all values of N, we find Bo4 to be dominant. The advantage values

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

Figure 5 | Comparing PAVs with search baselines and ablating over the prover policy: (a): We compare beam search over Gemma 9B SFT, using either effective reward (PAV), or  $Q^{\pi}$  (Snell et al., 2024), and report best-of-N performance where the re-ranker is either the ORM or PAV-as-ORM. (b): For the base Gemma 2B SFT policy, we run beam search with the effective reward where the prover is BoK( $\pi$ ) for different values of K. In both (a), (b) the x-axis scales the size of the beam or N for best-of-N. (c): For each base policy in the set: Gemma 2B, 9B, 27B policies, we run beam search with PAVs (beam size of 16) where the prover is another policy from the same set.

 $A^{\mu} \approx 0$  on all steps for very large K, since  $Q^{\mu}(s_h, a_h) = 1 - (1 - Q^{\pi}(s_h, a_h))^K \to 1$  on all steps, as we increase K. Hence, *in order to succeed we need an intermediate-level prover policy*.

We make similar observations in Figure 5(c) where we use the three base policies (Gemma 2B/9B/27B) as provers for training PAVs. In this scenario, we evaluate beam search with PAVs at N=16 on top of different base policies. We find that for the 2B and 9B base models, the 9B and 27B provers are most effective respectively, whereas for the 27B model, *surprisingly a weaker 9B policy is more effective than the stronger 27B model*. The weaker model presumably offers a complementary signal that distinguishes between different actions taken by 27B, aligning with our theoretical observations in Sec. 3.4.

Result 3: Advantages from the prover policy enable exploration. As discussed in Sec. 3.1, advantage  $A^{\mu}$  measures the progress made by an action agnostic of the value of the previous state, where as  $Q^{\pi}$  measures the promise of a particular state. Given a finite capacity beam, our effective reward (Eq. 5), which linearly combines  $Q^{\pi}$  and  $A^{\mu}$  induces a better tradeoff between exploring new prefixes (states) from where progress can be made and exploiting currently known prefixes with high Q-values. Exploration at

<span id="page-10-1"></span>![](_page_10_Figure_6.jpeg)

**Figure 6** | **(a):** Beam search with PAVs improves exploration efficiency (higher Pass@N), over typical PRMs. **(b):** Performance of beam search over Gemma 9B SFT for PAVs trained on datasets with different  $n_{\rm mc}/n_{\rm cov}$ .

initial steps is critical to ensure that the beam at later steps covers diverse partial rollouts each with a high likelihood of producing the correct answer. Thus over-committing to the beam with actions from the same state, regardless of the progress made by each can prove to be sub-optimal over a selection strategy that balances rewarding previous actions  $A^{\mu}$  and current states  $Q^{\pi}$ . Indeed, we observe in Fig. 6(a), beam search with PAV enhances pass@N performance vs. beam search with  $Q^{\pi}$  and i.i.d. sampling.

## Takeaways: Scaling test-time compute with process advantage verifiers.

- Beam search with PAVs boosts accuracy by >8% & compute efficiency by 1.5-5x over ORMs.
- Utilizing Best-of-K policies (corresponding to the base policy) as provers induce better exploration to maximize outcome reward. Optimal provers for a base policy appear at K > 1.

#### <span id="page-11-0"></span>4.2. How to Collect Data to Train PAVs?: PAV Training Data Scaling Laws

We now describe the procedure for training outcome verifiers and PAVs. We can learn to predict  $Q^{\pi}$  for a policy  $\pi$  (similar for  $Q^{\mu}$ ) by finetuning LLMs with a cross-entropy loss on the following data with triplets  $(s, a, Q_{\text{mc}}^{\pi}(s, a))$ . To collect this data, we first sample  $n_{\text{cov}}$  "seed" rollouts from the base or prover policy respectively for ORM and PAVs, to promote coverage over prefixes and steps. Then we sample  $n_{\text{mc}}$  additional rollouts, conditioned on each prefix in the seed rollout to compute the Monte-Carlo estimate of  $Q^{\pi}$  at each prefix. In Fig. 6(b) we plot the beam search performance of PAVs trained with different ratios of  $n_{\text{mc}}/n_{\text{cov}}$ , as we scale the total dataset size. Here, the beam size is fixed to 128 and the base policy is the Gemma 9B SFT policy and prover is Bo4 policy. We find that under low sampling budgets, optimizing for coverage  $(n_{\text{cov}} > n_{\text{mc}})$  is better for performance, and when budget is higher, reducing label noise in  $Q_{\text{mc}}^{\pi}$  by setting  $n_{\text{mc}} > n_{\text{cov}}$  gets us more improvements. In addition, we also spend some initial sampling budget is spent to identify "high value" states where  $Q^{\pi}$  is larger than a threshold, and identify the first step with low  $Q^{\pi}$  on an incorrect partial rollout from this state. We found this strategy to scale better with dataset size, as we discuss in App. D.

# <span id="page-11-1"></span>5. Results: Scaling Dense-Reward RL with PAVs

We can also use PAVs to train policies via online reinforcement learning (RL), by using the effective reward  $Q^{\pi} + \alpha A^{\mu}$  as dense, per-step rewards. We compare the sample efficiency of PAV-RL (i.e.,  $\ell_{\text{PAV-RL}}$  in Eq. 4) with standard ORM-RL (i.e.,  $\ell_{\text{ORM-RL}}$  in Eq. 3) on Gemma 2B and Gemma 9B SFT models, which are further optimized via rejection finetuning (RFT) (Yuan et al., 2023), before using them to initialize RL. To our knowledge, no prior work has successfully demonstrated the use of dense per-step feedback with a process reward model for RL, and we present the first significant set of results establishing the efficacy of this approach. We show that PAV-RL is much more sample-efficient, and enjoys a higher ceiling on the performance of any test-time re-ranker. Additional details for the experiments are in App. E.

Result 1: PAV-RL is > 7% better than ORM-RL in test accuracy, and  $6\times$  sample efficient. In Fig. 7(a), we report the test accuracies of Gemma 2B and 9B models trained with SFT, RFT, ORM-RL and PAV-RL. PAV-RL improves the RFT policy by 11% for 2B, and 15% for 9B, with > 7% gain over ORM-RL in both cases. Not only do the effective rewards from PAV improve the raw accuracy after RL, this higher accuracy is attained  $6\times$  faster (see Fig. 7(b)) for the 2B run and similarly for the 9B RL run (Fig. 7(c)). For both 2B and 9B, RL runs, we experiment with two options for the prover policy: (i) 2B SFT policy; and (ii) 9B SFT policy. While both of these provers rapidly become weaker than the base policy within a few gradient steps of RL, a fixed PAV trained with each of these provers is able to still sustain performance gains in RL. More interestingly, we find that the 2B SFT policy serves as the best choice of the prover for both 2B and

<span id="page-12-0"></span>![](_page_12_Figure_1.jpeg)

Figure 7 | PAVs as dense rewards in RL improve sample efficiency compared to ORMs, along with gains on raw accuracy: (a) We report the performance of a base policy trained using RL with effective rewards (PAV-RL), or only outcome rewards (ORM-RL), and baselines SFT, RFT. (b,c): Across training iterations, we report the test performance of policies trained with PAV-RL and ORM-RL, on Gemma 2B and 9B SFT base policies.

9B policies. This observation that a weak prover can still improve the base policy corroborates our results in the didactic setup and our analysis in Sec. 3.4. While we were not able to run experiments where the prover policy is dynamically updated on the fly, we believe that updating the prover through the process of RL training should only amplify these benefits.

Result 2: PAV-RL achieves higher performance ceiling on test-time re-ranking. In Fig. 8(a), for Gemma 2B, we plot the Pass @N performance for each method, and find (i) Pass @N is higher (> 7%) for PAV-RL, compared to ORM-RL, for any  $N \le 128$ ; and (ii) the rate at which Pass @N improves for PAV-RL is higher than ORM-RL. Both trends are consistent with our observations on the didactic example in Sec. 3.3. Notably, for  $N \ge 64$ , ORM-RL is worse than the SFT policy, perhaps due to lower entropy over the distribution at the next step resulting in non-diverse candidates. Why does PAV-RL produce diverse candidates, and does not suffer from the low diversity problem in ORM-RL? We answer this with a key insight on how *the primary benefit of PAVs is to promote efficient exploration*.

<span id="page-12-1"></span>![](_page_12_Figure_5.jpeg)

**Figure 8** | **(a):** For the policies trained in (a) we report the best-of-N performance where the oracle reward Rex is used to rank N candidates sampled from the base policy (Pass @N). **(b):** Amongst hard problems that remain unsolved by Best-of-256 over the base SFT policy, we check how many are solved by Best-of-N over PAV-RL or ORM-RL. PAV-RL is able to solve a substantially more problems than what ORM-RL was able to solve.

**Result 3: PAVs improve exploration and discover correct solutions to novel problems.** An outcome reward model rewards downweight all steps in an incorrect rollout equally during RL, whereas the effective reward + in PAVs, up-weights steps that make progress under the prover, even when the complete rollout is incorrect. This increases the coverage over individual steps that can improve the likelihood of the base policy to succeed (since the prover policy is not too misaligned with the base policy). These can now be proposed by the base policy at a given prefix. This mechanism for exploration is analogous to test-time search we discussed in Sec. [4.1.](#page-9-1) Hence, the directed supervision from PAVs improves sample-efficiency throughout the course of training (Fig. [7\(](#page-12-0)c)). In fact, we also find that combining the PAV-RL policy with test-time beam search is able to solve a substantially larger number of *new* problems within smaller compute budgets ( = 16, 32) that the SFT policy cannot solve with a much larger budget = 256 (Fig. [8\(](#page-12-1)b)).

#### Takeaway: RL with process advantage verifiers (PAVs) as dense rewards

- Using trained PAVs as dense rewards in RL boosts scales sample efficiency by 5 − 6×, compared to only using sparse ORM rewards, and results in policies with a higher Pass @N performance.
- Advantages from a complementary prover policy improves the sample efficiency of exploration in RL, and produces policies that can discover solutions to hard novel questions.

## <span id="page-13-0"></span>**6. Related Work**

We briefly discuss some key related works here, and leave the detailed discussion for App. [A.](#page-18-0) To address issues of sparse feedback in ORMs [\(Cobbe et al.,](#page-15-1) [2021b\)](#page-15-1), recent works [\(Lightman et al.,](#page-16-1) [2023;](#page-16-1) [Uesato](#page-17-0) [et al.,](#page-17-0) [2022\)](#page-17-0) trained process reward models (PRMs) to densely predict incorrect steps in a multi-step reasoning trace. Since human data collection for process labels is not scalable enough, recent work [\(Luo](#page-16-2) [et al.,](#page-16-2) [2024;](#page-16-2) [Wang et al.,](#page-17-1) [2024\)](#page-17-1) used automated supervision to annotate steps with values under the base policy, *i.e.*, the PRMs score a step with the likelihood of future success, when continuing to sample from the step. While -value PRMs in [Lightman et al.](#page-16-1) [\(2023\)](#page-16-1); [Luo et al.](#page-16-2) [\(2024\)](#page-16-2) were mainly used as verifiers for re-ranking, [Snell et al.](#page-16-0) [\(2024\)](#page-16-0) used them for test-time beam search. [Shao et al.](#page-16-3) [\(2024\)](#page-16-3) uses PRMs for RL but found a gain of only 1 − 2% with PRMs. In our work, we question solely relying on -values or advantages of the base policy, and find that measuring progress (i.e., advantages) under a different prover policy can amplify exploration, thus boosting test-time search and RL. To our knowledge, we are the first to show substantial gains in compute and sample efficiency with PRMs. Our methodology for data collection is similar to [\(Hwang et al.,](#page-15-7) [2024;](#page-15-7) [Setlur et al.,](#page-16-9) [2024\)](#page-16-9) (i.e., identify "first pits" in reasoning traces), these works only use it to collect preference pairs. Beyond all of these, we also characterize which policy to use for computing advantages.

Concurrently to us, akin to the methodology in [Hwang et al.](#page-15-7) [\(2024\)](#page-15-7); [Setlur et al.](#page-16-9) [\(2024\)](#page-16-9), [Kazemnejad](#page-15-8) [et al.](#page-15-8) [\(2024\)](#page-15-8) optimize the base policy with online RL, where the dense step-level rewards correspond to advantages under the base policy itself. This is a special case of our setting, where the prover policy = , but as we note in Sec. [3.2,](#page-5-2) setting = in our effective reward (Eq. [5\)](#page-5-0) results in exactly the same policy gradient updates as only optimizing the outcome reward. Since [Kazemnejad et al.](#page-15-8) [\(2024\)](#page-15-8) use "on-the-fly" Monte-Carlo rollout estimation to estimate advantages, they are able to avoid estimation errors in the process reward model. Nonetheless, our theoretical result and the didactic example (both of which assume access to perfect advantage estimates) show that gains from this approach are significantly smaller than using an appropriate prover policy, which is distinct from the base policy.

## **7. Discussion and Conclusion**

We began our exposition with the following question: how to define process rewards such that optimizing the base policy against process rewards ultimately improves the outcome level correctness of the final answer? Our key finding is that process rewards defined as advantages of a prover policy, distinct from the base policy improve the efficiency of exploration for steps sampled from the base policy during test-time search and online RL. This improved exploration in turn leads to discovery of better solutions, resulting in a higher accuracy on the math reasoning task. We also formally characterized the set of good prover policies as policies with step-level advantages that meaningfully contrast steps generated by the base policy, while still producing step-level advantages that are aligned with the base policy. Having trained process advantage verifiers (PAVs) to predict advantages under the prover policy, we empirically observed that test-time search against the trained PAVs improve the compute-efficiency of search by 1.5 − 5×, and accuracy of search by over 8% compared to running best-of- against an ORM. Next, we present one of the significant results that validate the use of dense supervision when optimizing the base policy with online RL. Specifically, we show that dense online RL with rewards from our trained PAVs, improves sample efficiency of online RL by 5 − 6×, and results in an accuracy gain of over 6%.

**Limitations.** Despite the promise of our results, there are several limitations to our work that present important avenues for future research. First, while we can easily compute the right hand side of our result in Theorem [3.1](#page-8-0) to understand whether a given prover policy will improve a fixed base policy, it is unclear how to automatically design a flexible class of optimal (or very good) prover policies for a sequence of base policy iterates. Perhaps simultaneously optimizing the prover and the base policy (in a two-player game) might provide for an approach to obtain the best prover during RL, but this is largely an open question. Second, since inevitably learning a process advantage verifier (PAV) will incur fitting errors and this upper bounds peformance of our method. Fitting errors can be circumvented if our approach if we can simply run rollouts from prover policies during online RL or search to estimate advantages without training verifiers, and extending our approach to this setup is a good avenue for future work.

## **Acknowledgements**

The authors would like to thank Charlie Snell, Yi Su, Katherine Heller, and Virginia Smith for feedback on an earlier version of this paper. We also thank Ahmad Beirami, Sergey Levine, Victor Veitch, Idan Shenfeld, Arian Hosseini, Stephen Pfohl, Xiangyu Qi, Tianhe Yu, and Christina Baek for technical discussions. AS and CN also thank Preston Robinette, Sho Kannan, Tianze Shi, Diana Mincu, Hritik Bansal, and Liangchen Luo for code, infrastructure and data analytics support.

## **References**

- <span id="page-14-1"></span>A. Agarwal, S. M. Kakade, J. D. Lee, and G. Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. *Journal of Machine Learning Research*, 22(98):1–76, 2021.
- <span id="page-14-2"></span>T. Anthony, Z. Tian, and D. Barber. Thinking fast and slow with deep learning and tree search. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-14-0"></span>M. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos. Unifying count-based exploration and intrinsic motivation. In *Advances in Neural Information Processing Systems*, pages 1471–1479, 2016.

- <span id="page-15-10"></span>X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024.
- <span id="page-15-13"></span>J. D. Chang, K. Brantley, R. Ramamurthy, D. Misra, and W. Sun. Learning to generate better than your llm. *arXiv preprint arXiv:2306.11816*, 2023.
- <span id="page-15-12"></span>K.-W. Chang, A. Krishnamurthy, A. Agarwal, H. Daumé III, and J. Langford. Learning to search better than your teacher. In *International Conference on Machine Learning*, pages 2058–2066. PMLR, 2015.
- <span id="page-15-9"></span>K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021a.
- <span id="page-15-1"></span>K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021b.
- <span id="page-15-0"></span>M. Collins. Discriminative reranking for natural language parsing. In *Proceedings of the International Conference on Machine Learning*, 2000.
- <span id="page-15-3"></span>Gemma Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, et al. Gemma: Open models based on gemini research and technology. *arXiv preprint arXiv:2403.08295*, 2024.
- <span id="page-15-14"></span>M. Germain, K. Gregor, I. Murray, and H. Larochelle. Made: Masked autoencoder for distribution estimation. In *International conference on machine learning*, pages 881–889. PMLR, 2015.
- <span id="page-15-11"></span>A. Havrilla, Y. Du, S. C. Raparthy, C. Nalmpantis, J. Dwivedi-Yu, M. Zhuravinskyi, E. Hambro, S. Sukhbaatar, and R. Raileanu. Teaching large language models to reason with reinforcement learning. *arXiv preprint arXiv:2403.04642*, 2024.
- <span id="page-15-4"></span>D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. *NeurIPS*, 2021.
- <span id="page-15-6"></span>G. Hinton. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.
- <span id="page-15-2"></span>A. Hosseini, X. Yuan, N. Malkin, A. Courville, A. Sordoni, and R. Agarwal. V-star: Training verifiers for self-taught reasoners. *arXiv preprint arXiv:2402.06457*, 2024.
- <span id="page-15-7"></span>H. Hwang, D. Kim, S. Kim, S. Ye, and M. Seo. Self-explore to avoid the pit: Improving the reasoning capabilities of language models with fine-grained rewards. *arXiv preprint arXiv:2404.10346*, 2024.
- <span id="page-15-16"></span>S. Kakade and J. Langford. Approximately optimal approximate reinforcement learning. In *Proceedings of the Nineteenth International Conference on Machine Learning*, pages 267–274, 2002.
- <span id="page-15-5"></span>S. M. Kakade. A natural policy gradient. In *Advances in neural information processing systems*, volume 14. Advances in neural information processing systems, 2001a.
- <span id="page-15-15"></span>S. M. Kakade. A natural policy gradient. *Advances in neural information processing systems*, 14, 2001b.
- <span id="page-15-8"></span>A. Kazemnejad, M. Aghajohari, E. Portelance, A. Sordoni, S. Reddy, A. Courville, and N. L. Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. *arXiv preprint arXiv:2410.01679*, 2024.

- <span id="page-16-1"></span>H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let's verify step by step. *arXiv preprint arXiv:2305.20050*, 2023.
- <span id="page-16-2"></span>L. Luo, Y. Liu, R. Liu, S. Phatale, H. Lara, Y. Li, L. Shu, Y. Zhu, L. Meng, J. Sun, et al. Improve mathematical reasoning in language models by automated process supervision. *arXiv preprint arXiv:2406.06592*, 2024.
- <span id="page-16-12"></span>Q. Ma, H. Zhou, T. Liu, J. Yuan, P. Liu, Y. You, and H. Yang. Let's reward step by step: Step-level reward model as the navigators for reasoning. *arXiv preprint arXiv:2310.10080*, 2023.
- <span id="page-16-4"></span>R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. *arXiv preprint arXiv:2112.09332*, 2021.
- <span id="page-16-5"></span>A. Y. Ng, D. Harada, and S. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In *ICML*, volume 99, pages 278–287, 1999.
- <span id="page-16-14"></span>L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022.
- <span id="page-16-13"></span>R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. *arXiv preprint arXiv:2305.18290*, 2023.
- <span id="page-16-6"></span>S. Ross and J. A. Bagnell. Reinforcement and imitation learning via interactive no-regret learning. *arXiv preprint arXiv:1406.5979*, 2014.
- <span id="page-16-8"></span>A. A. Rusu, S. G. Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell. Policy distillation. *arXiv preprint arXiv:1511.06295*, 2015.
- <span id="page-16-9"></span>A. Setlur, S. Garg, X. Geng, N. Garg, V. Smith, and A. Kumar. Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold. *arXiv preprint arXiv:2406.14532*, 2024.
- <span id="page-16-3"></span>Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.
- <span id="page-16-11"></span>A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, P. J. Liu, J. Harrison, J. Lee, K. Xu, A. Parisi, et al. Beyond human data: Scaling self-training for problem-solving with language models. *arXiv preprint arXiv:2312.06585*, 2023a.
- <span id="page-16-10"></span>I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg. Progprompt: Generating situated robot task plans using large language models. In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 11523–11530. IEEE, 2023b.
- <span id="page-16-0"></span>C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*, 2024.
- <span id="page-16-7"></span>W. Sun, A. Venkatraman, G. J. Gordon, B. Boots, and J. A. Bagnell. Deeply aggrevated: Differentiable imitation learning for sequential prediction. In *International conference on machine learning*, pages 3309–3318. PMLR, 2017.

- <span id="page-17-2"></span>R. S. Sutton and A. G. Barto. *Reinforcement learning: An introduction*. The MIT Press, second edition, 2018.
- <span id="page-17-10"></span>R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour. Policy gradient methods for reinforcement learning with function approximation. *Advances in neural information processing systems*, 12, 1999.
- <span id="page-17-3"></span>F. Tajwar, A. Singh, A. Sharma, R. Rafailov, J. Schneider, T. Xie, S. Ermon, C. Finn, and A. Kumar. Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data, 2024.
- <span id="page-17-0"></span>J. Uesato, N. Kushman, R. Kumar, F. Song, N. Siegel, L. Wang, A. Creswell, G. Irving, and I. Higgins. Solving math word problems with process-and outcome-based feedback. *arXiv preprint arXiv:2211.14275*, 2022.
- <span id="page-17-1"></span>P. Wang, L. Li, Z. Shao, R. X. Xu, D. Dai, Y. Li, D. Chen, Y. Wu, and Z. Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024.
- <span id="page-17-4"></span>R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8(3-4):229–256, 1992.
- <span id="page-17-9"></span>Y. Wu, Z. Sun, S. Li, S. Welleck, and Y. Yang. An empirical analysis of compute-optimal inference for problem-solving with language models. *arXiv preprint arXiv:2408.00724*, 2024.
- <span id="page-17-7"></span>F. Yu, A. Gao, and B. Wang. Outcome-supervised verifiers for planning in mathematical reasoning. *arXiv preprint arXiv:2311.09724*, 2023.
- <span id="page-17-5"></span>Z. Yuan, H. Yuan, C. Li, G. Dong, C. Tan, and C. Zhou. Scaling relationship on learning mathematical reasoning with large language models. *arXiv preprint arXiv:2308.01825*, 2023.
- <span id="page-17-6"></span>E. Zelikman, Y. Wu, J. Mu, and N. Goodman. Star: Bootstrapping reasoning with reasoning. *Advances in Neural Information Processing Systems*, 35:15476–15488, 2022.
- <span id="page-17-8"></span>L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, and R. Agarwal. Generative verifiers: Reward modeling as next-token prediction. *arXiv preprint arXiv:2408.15240*, 2024.

# **Appendices**

## <span id="page-18-0"></span>**A. Additional Related Work**

In this section, we highlight works from four relevant streams, expanding on discussion in Section [6.](#page-13-0) First, we look at works that train verifiers to provide outcome level feedback [\(Cobbe et al.,](#page-15-1) [2021b;](#page-15-1) [Hosseini](#page-15-2) [et al.,](#page-15-2) [2024;](#page-15-2) [Singh et al.,](#page-16-10) [2023b;](#page-16-10) [Zelikman et al.,](#page-17-6) [2022\)](#page-17-6) on the correctness of the full response (ORM). Here, the trained ORMs are mainly used for test-time search (best-of-). Next, we look at works that alleviate issues with sparse feedback in ORMs, and instead train process reward models (PRMs), that can perform credit assignment. PRMs are trained either through human annotations [\(Lightman et al.,](#page-16-1) [2023;](#page-16-1) [Uesato et al.,](#page-17-0) [2022\)](#page-17-0), or automated forms of supervision [\(Luo et al.,](#page-16-2) [2024;](#page-16-2) [Snell et al.,](#page-16-0) [2024;](#page-16-0) [Wang et al.,](#page-17-1) [2024\)](#page-17-1). While some works use PRMs and ORMs to collect data for supervised fine-tuning [Hosseini et al.](#page-15-2) [\(2024\)](#page-15-2) or offline RL [Setlur et al.](#page-16-9) [\(2024\)](#page-16-9), other works directly use them as rewards in online RL [\(Shao](#page-16-3) [et al.,](#page-16-3) [2024;](#page-16-3) [Uesato et al.,](#page-17-0) [2022;](#page-17-0) [Wang et al.,](#page-17-1) [2024\)](#page-17-1). Finally, we contrast our work against papers on imitating stronger teacher policies via RL objectives that optimize potential functions of teacher policies.

**Outcome reward models.** ORMs are verifiers [\(Cobbe et al.,](#page-15-1) [2021b;](#page-15-1) [Uesato et al.,](#page-17-0) [2022\)](#page-17-0) commonly used to improve the test-time performance using best-of-, where we generate multiple candidate solutions from the base policy (LLM), rank them using the ORM, and pick the best one. ORMs are trained to assess correctness of a solution either using binary classification [\(Cobbe et al.,](#page-15-9) [2021a;](#page-15-9) [Yu et al.,](#page-17-7) [2023\)](#page-17-7), preference optimization using DPO [\(Hosseini et al.,](#page-15-2) [2024\)](#page-15-2), or next-token prediction [\(Zhang et al.,](#page-17-8) [2024\)](#page-17-8). Furthermore, prior works train LLMs on self-generated data using ground-truth outcome rewards (Rex), either via supervised fine-tuning [\(Singh et al.,](#page-16-11) [2023a;](#page-16-11) [Yuan et al.,](#page-17-5) [2023;](#page-17-5) [Zelikman et al.,](#page-17-6) [2022\)](#page-17-6), or online RL [\(Bi et al.,](#page-15-10) [2024\)](#page-15-10). In contrast to these approaches, our work focuses on process reward models (PRMs) for improving performance with beam-search at test time as well as online RL where we maximize the effective reward in Eq. [5](#page-5-0) which linearly combines both Rex (outcome supervision) and process supervision in the form of advantages under a prover policy .

**PRMs and credit assignment.** Several works focus on training step-level PRMs on math reasoning tasks, either using human labels [\(Lightman et al.,](#page-16-1) [2023\)](#page-16-1) or automated LLM-generated data to estimate value functions [\(Luo et al.,](#page-16-2) [2024;](#page-16-2) [Wang et al.,](#page-17-1) [2024\)](#page-17-1). Our work also focus on automated data collection for PRMs but empirically argues for using the advantage function as step-level rewards along with , with a conceptual explanation in Section [3.1.](#page-4-1) Several prior works have explored step-level search algorithms with PRMs, such as beam search [\(Snell et al.,](#page-16-0) [2024\)](#page-16-0), heuristic greedy search [\(Ma et al.,](#page-16-12) [2023\)](#page-16-12), and reward-balanced tree search [\(Wu et al.,](#page-17-9) [2024\)](#page-17-9). [Hwang et al.](#page-15-7) [\(2024\)](#page-15-7); [Setlur et al.](#page-16-9) [\(2024\)](#page-16-9) use advantages to identify the "first pit" in an incorrect reasoning trace. Specifically, they collect data by computing advantages at each step using Monte Carlo rollouts. Then in an incorrect trace, they identify the step with the least advantage, and use the prefix of that step to construct preference pairs for offline direct preference optimization [\(Rafailov et al.,](#page-16-13) [2023\)](#page-16-13). In contrast, our work computes advantages under a prover policy, that we formally characterize, and use the computed advantages for improving test-time search and efficiency of online reinforcement learning.

**Online RL for math reasoning.** Once we have a trained outcome or process verifiers, it is natural update a policy by optimizing it against the learned signal, similar to how learned reward models are optimized in RLHF [\(Ouyang et al.,](#page-16-14) [2022\)](#page-16-14). In the context of math reasoning, [Havrilla et al.](#page-15-11) [\(2024\)](#page-15-11); [Shao](#page-16-3) [et al.](#page-16-3) [\(2024\)](#page-16-3); [Uesato et al.](#page-17-0) [\(2022\)](#page-17-0) trained policies with RL, experimenting with both dense and sparse

```
Planted sub-sequence v^*: [3 6 0 3 5]
Samples from policy corresponding to \gamma = 15
                                                        Samples from policy corresponding to \gamma = 100
sample=[ 3 6 0
                 3
                    5 14 14 14 14 14
                                        reward=1.0
                                                        sample=[3 6
                                                                         3
                                                                             5 14 14 14 14 14]
                                                                                                reward=1.0
                                                                      0
sample=[3
           6
               0
                 4
                    0 3 1 11 7 3]
                                        reward=0.0
                                                        sample=[3 6]
                                                                         3
                                                                             5 14 14 14 14 14]
                                                                                                reward=1.0
                                                        sample=[ 3
                                                                      0
                                                                          3
sample=[ 3 6
              0 3 5 14 13 3 2 11]
                                                                   6
                                                                             5 14 14 14 14 14]
                                                                                                reward=1.0
                                        reward=1.0
sample=[ 3
           6
              0 13
                    5
                       3
                             3 14
                                        reward=0.0
                                                        sample=[ 3
                                                                   6
                                                                      0
                                                                         3
                                                                             5 14 14 14 14 14]
                                                                                                reward=1.0
                       3
                                                                      0
                                                                         3 5 14 14 14 14 14]
                                                        sample=[ 3 6
                                                                                                reward=1.0
sample=[12 8
              3
                6 14
                          6 0
                                3
                                   5]
                                        reward=0.0
```

**Figure 9** | *Pictorial description of our planted sub-sequence didactic setup*: An example showing five samples drawn *i.i.d.* from a very strong policy ( $\gamma = 100$ ), and a relatively weaker ( $\gamma = 15$ ) policy in our didactic setup.

rewards. In all three works, the gains observed by using PRMs that predict step-level correctness (similar to Lightman et al. (2023)) is quite small, compared to simply using trained ORMs, or the ground-truth outcome supervision Rex. In fact, Havrilla et al. (2024) states that the only algorithm that does well is a form of expert iteration (Anthony et al., 2017), which does not inhibit exploration as severely as some other approaches they compare with. Our work presents one of the first results, where trained PRMs, used in conjunction with the outcome rewards during online RL, result in policies with substantially higher (+6%) performance, than the one trained only with outcome supervision. Our results also indicate a  $5-6\times$  sample efficiency boost for online RL, with our trained PAVs.

Connections to imitation learning through RL. The idea of mixing potential functions from different policies  $\mu$  and  $\pi$ , in order to improve upon a sub-optimal expert  $\mu$  appears in Chang et al. (2015), but this work considers the structured prediction problem which is vastly different from our setting. Related to this, is the work by Chang et al. (2023), which uses a "guide" policy to rollout from prefixes generated by a base policy. The base policy can now imitate the guide by cloning those rollouts, and eventually surpass. Our work also uses a prover policy which can complete rollouts from states where the base policy fails. But, we also show that weak provers in many cases are able to improve the base policy, or search over its responses, better than a stronger prover policy. We tie this observation to the insight that the main goal of the prover policy is to distinguish steps taken by the base policy, as measured by advantages under the prover. Thus, we do not require the prover policy to be something better than the base policy, which is a key distinction with Chang et al. (2023).

# <span id="page-19-0"></span>**B.** Didactic Analysis

We consider sequences of length 10 from a 15-token vocabulary  $\mathcal{V} := \{1, 2, \dots, 14\}$ , where the end-of-sequence token is given by 14, and all tokens following the end-of-sequence token (including it) are masked. Given an unknown planted sequence  $y^*$  (in Fig. 9), we train a policy  $\pi$  with policy gradient, where the outcome reward we wish to optimize is terminal and sparse, *i.e.*, for  $y \sim \pi$  we have  $r(y, y^*) = 1$  if and only if  $y^*$  appears in y, and 0 otherwise (Fig. 9). The policy  $\pi$  in our experiments is represented by a multi-layer neural network, similar to the MADE architecture (Germain et al., 2015). The prover policy  $\mu$  is parameterized by a scalar  $\gamma > 0$ . In particular, at any state s, where the last k tokens leading up to s match first k tokens of  $y^*$ , then:

$$\mu(\mathbf{y}_{k+1}^{\star} \mid \mathbf{s}) \propto \gamma,$$

and uniform on all other tokens. Thus, as  $\gamma$  increases, the performance of  $\mu$  improves and  $\to 1$  as  $\gamma \to \infty$ . For our experiments, we assume (almost) oracle access to ground-truth advantage and Q-values, thus

mitigating any confounding issues due to usage of a learned verifier. We are able to approximate exact Q-values very accurately by using Monte Carlo estimates with large > 100 rollouts. With the goal of optimizing the terminal reward  $r(y, y^*)$ , we optimize  $\pi$  with two types of rewards: (i) only the outcome reward  $r(y, y^*)$ , which is equivalent to using only  $Q^{\pi}$  as step-level rewards; and (ii) using the effective reward:  $Q^{\pi} + \alpha A^{\mu}$  as the step-level reward.

**Training details.** We use effective rewards with  $\alpha=1$ , and use the gradient in Eq. 5 to update the policy via policy gradient iterations. For the ORM runs, where we only use the outcome reward  $r(y,y^*)$ , the policy gradient is equivalent to the case where  $\alpha=0$  in Eq. 5. We train for 10,000 iterations in both cases, with a batch size of 64, and a constant learning rate of 1e-3 for the Adam optimizer. The RL runs are initialized with a supervised finetuned policy. For this we take a randomly initialized network, based on the MADE architecture (Germain et al., 2015), with 3 layers, and 128 hidden units in each. Then we train it with supervised next-token prediction loss for 50 iterations on a dataset of 3200 samples from a weak policy ( $\gamma=5.0$ ). The batch size for the SFT training is also set to 64. For evaluating Pass @N performance, we either sample N independent trajectories (temperature 1.0) from the base policy trained using effective rewards, or only  $Q^{\pi}$ . We also evaluate Pass @N for the SFT policy for comparison.

# <span id="page-20-0"></span>C. Additional: Experiments on Test-time Search with PAVs

**Implementation details.** For our experiments in Sec. 4, we use three pretrained models: Gemma 2B, 9B and 27B. We finetune each of these on the MATH (Hendrycks et al., 2021) dataset. The finetuning is done for 5000 iterations, with a batchsize of 32, and a maximum learning rate of 5e - 6 for 2B, 9B and 5e - 7 for the 27B models. We trained the policies using the Adam optimizer, with a linear warm up and cosine decay learning rate schedule. The linear warm up is done for the first 500 iterations. For the base policies, we choose the SFT checkpoints with the best accuracy on a holdout validation set of the MATH dataset. Given the SFT checkpoints, we next train PAVs using the procedure in Sec. 4.2. We do this for a class of provers, which include the base policies themselves. As we discuss in Sec. 4, the prover class also includes the best-of-K policy for K in  $\{2, 4, 8, 16, 32\}$ .

We use the hold out validation set to ascertain the value of  $\alpha$  in the effective reward. For each base policy we run beam search with a beam size of 16 on this hold out validation set, and using the base policy itself as a prover, we evaluate the value of  $\alpha$  that works best in the effective reward. We find that  $\alpha=0.5$  worked best for Gemma 2B and 9B base policies, while a lower value of  $\alpha=0.2$  was optimal for Gemma 27B. To tune  $\alpha$  we ran a grid search over the range [0.0, 1.0], evaluating at an interval of 0.1. We observe that the choice of  $\alpha$  is a relatively robust one, since for all three base policies, we saw improvements (over only  $Q^{\pi}$  as the reward) for values in the range of [0.2, 0.6]. Having a separate value of  $\alpha$  for each base policy, we use the same value in the effective reward given by any choice of the prover policy that is used for that base policy. Next, we present an experiment that compares the predictive power of effective reward vs. just  $Q^{\pi}$  at initial states of a rollout under the base policy  $\pi$ , when either is used to predict the final outcome given by Rex.

**Experiment:** Is the effective reward able to predict the final outcome better than  $Q^{\pi}$ ? In Fig. 10, we describe an experiment where for both the effective reward  $Q^{\pi} + \alpha A^{\mu}$  (PAV) and just  $Q^{\pi}$  (PQV), we compute the error of the classifier that makes a prediction on the final outcome by thresholding on either reward value at each step of the rollout. This threshold is computed using a validation set, and is separate for each step and reward combination. The figure tells us that the outcome prediction error drops for both rewards as the base policy is rolled out more, but clearly the effective reward dominates  $Q^{\pi}$  (PQV)

<span id="page-21-1"></span>across all steps. Thus, the effective reward is a more informative signal (lower classification error) for the problem of predicting the success of a partial rollout, especially in the earlier steps of the rollout. This helps to explain the better performance of beam search with a finite capacity beam that re-ranks partial rollouts with the effective reward. For this experiment, we use the Gemma 9B SFT policy as the base policy and the best-of-4 policy corresponding to the same SFT policy as the prover.

![](_page_21_Figure_2.jpeg)

Figure 10 | Effective rewards at any step are able to predict the outcome rewards at the final step, better than  $Q^{\pi}$ : For both the effective reward  $Q^{\pi} + \alpha A^{\mu}$  and just  $Q^{\pi}$ , we compute the error of the classifier that makes a prediction on the final outcome by thresholding on either reward value at each step of the rollout. This threshold is computed using a validation set, and is separate for each step and reward combination. The figure tells us that the outcome prediction error drops for both rewards, as the base policy is rolled out more, but compared to  $Q^{\pi}$  at an intermediate step, the effective reward  $Q^{\pi} + \alpha A^{\mu}$  at an intermediate step is able to reliably predict the final outcome level correctness of the future rollout under the base policy  $\pi$ .

## <span id="page-21-0"></span>D. Details on Collecting Data and Training PAVs

In Fig. 6(b) in Sec. 4.2, our seed rollouts are *i.i.d.* sampled from  $\pi$ , but prior works (Hwang et al., 2024; Luo et al., 2024; Setlur et al., 2024) employed a "first pit" strategy for coverage. Here, some initial sampling budget is spent to identify "high value" states where  $Q^{\pi}$  is larger than a threshold. Then, for any incorrect sample from these high value states greater budget is spent to estimate the first step (first pit) with  $Q^{\pi} = 0$ . All prefixes (and their estimated Q-values) until the first pit are then added to the training data. In Fig. 11, we compare beam search using PAVs trained using data from the first pit strategy, and the random sampling strategy. Both of them use the best value of  $n_{\text{mc}}/n_{\text{cov}}$  from Fig. 6(b) for every dataset size. We find the first pit strategy to be better than random, especially when the number of seed rollouts are limited. Once we get coverage over such pits, we sample a large number of partial rollouts conditioned on each prefix until the first pit. This is used to compute the Monte Carlo estimate of Q values more accurately on the path to the first pit. Each prefix and estimated Q value pair is then added to the dataset used to train PAVs.

**Training details.** All PAVs used in this work are trained by taking the Gemma 9B pretrained checkpoint and finetuning it on the data collected from the above strategy. The data collection uses first pit strategy for better coverage over pits in the seed rollouts. Based on findings in Fig. 6(b), we use a high value of  $n_{\text{mc}} = 20$  to estimate the *Q*-values accurately for each step in the seed rollout. For each base policy,

in total, we collect a dataset of over 300,000 (prefix,  $\hat{Q}^{\pi}$ -value) pairs. Here,  $\hat{Q}^{\pi}$  is the Monte Carlo estimate for the *Q*-value at the prefix, under the policy  $\pi$ . on which we finetune the Gemma 9B model with cross-entropy loss. Since the distribution of values for  $\hat{Q}^{\pi}$  can be skewed, we split the range of  $\hat{Q}^{\pi}$ -values into two buckets, based on which we also partition the training data. The first bucket is the set of all prefixes with  $\hat{Q}^{\pi} < 0.5$  and the second is the set of all prefixes with  $\hat{Q}^{\pi} \geq 0.5$ . Then, we use class-balanced sampling over these buckets to finetune the pretrained model for 20000 training iterations, using a batch size of 32. We use an Adam optimizer with a maximum learning rate of 5e-7. We use a linear warm up (till 2000 steps), followed by a cosine decay learning rate schedule to train the models. Since a pretrained LLM would output a matrix of logits (vocabulary size × sequence length) we fix a token as the "scoring token" to be the end of the sequence / prefix that needs to be scored. The logits of this scoring token are then used to determine the prediction for the LLM being trained.

<span id="page-22-1"></span>Once we have models that predict the  $Q^{\pi}$  for a base policy  $\pi$ , we compute the Q-value under the BoK policy corresponding to  $\pi$ , by setting:  $Q^{\text{BoK}(\pi)}(s,a) = 1 - (1 - Q^{\pi}(s,a))^{K}$ . Next, given the Q-values we for  $\pi$ , and their corresponding best-of-K policies, we can compute any effective reward in Eq. 5, using the definition of advantage value of a step in Eq. 2.

![](_page_22_Figure_3.jpeg)

**Figure 11** | *First pit strategy from Luo et al. (2024); Setlur et al. (2024)*: We compare the beam search performance (with beam size 128) for a PAV trained on data collected using two types of seed rollouts. For the seed rollouts, we either randomly sample from the distribution of state action pairs induced by the base policy. Or we improve coverage particularly by using the "first pit" strategy of identifying the first state with low *Q* values on a partial rollout that starts from a high *Q*-value state and ends with an outcome reward of 0, i.e., *Q* value is 0.

# <span id="page-22-0"></span>E. Additional: Experiments on RL Training with PAVs

**Training details.** As discussed in Section 5, the initialization for RL training is the RFT (rejection finetuned) checkpoint for the corresponding base policies. More specifically, we consider two base policies Gemma 2B SFT, and Gemma 9B SFT, where the RL training is initialized with the policy obtained by further optimizing the base policy via rejection finetuning (RFT) Yuan et al. (2023). This is done to improve coverage over states and actions during the initial iterations of RL training. For rejection finetuning we train the SFT model (base policy) on all correct trajectories sampled when collecting seed rollouts for training PAVs (that predict the advantage of the same base policy). The training details for RFT remain same as SFT and are detailed in Appendix C. We use the REINFORCE (Sutton et al., 1999)

algorithm to improve the base policy. The RL training is run for 10000 iterations for the 2B model, and for 5000 iterations for the 9B model. For both we use the Adam optimizer with a learning rate of 1e-7, and a batchsize of 32. The maximum response length is set to 512. For both we used a learning rate schedule with a linear warm up for 10% of the total training iterations, followed by a cosine decay. The implementation of our policy gradient algorithm also uses a token-level value function that is initialized to be the base policy itself, and is trained with a square loss. The value function is only used as a baseline during RL training, i.e., at any state s it is only predicting  $\mathbb{E}_{a \sim \pi(\cdot|s)} Q^{\pi}(s,a) + \alpha A^{\mu}(s,a)$ .

Most importantly, we use a validation set to identify a good choice of  $\alpha$  in the effective reward (in Eq. 5). For Gemma 2B this value is 5.0 and for the 9B policy  $\alpha$  = 3.0 works best. Similar to the choice of  $\alpha$  for test-time search, we find that most values of  $\alpha$  in the range of 0.5 to 6.0 improved performance over ORM-RL, to different degrees. Both policies are also optimized with KL regularization, against the initial iterate of the RL policy, where the strength of the KL penalty is set to 0.001 for both.

## <span id="page-23-0"></span>F. Theoretical Analysis: Complementary Provers Amplify Base Policy Improvement

In this section, we present the proofs for our theoretical results in the main paper (Sec. 3.4). We begin by describing some notation we use in our proofs, the natural policy gradient algorithm we use for the policy update, followed by the proof for Theorem 3.1. We also present a simple application of this result in Proposition F.1. Our results in this section are in the tabular setting, with softmax parameterization of the policies. Note that for the deterministic Markov decision process induced by the LLM, we are indeed in a tabular setting, where the set of states and actions is discrete, but large and grows exponentially in sequence length.

**Notation and preliminaries.** We use  $d_h^{\pi}$ ,  $d_h^{\mu}$  to denote the distribution over states  $s_h$  at time step h, starting from the initial state distribution given by the empirical distribution over the questions in the dataset  $\mathcal{D}$ , and following the base policy  $\pi$ , or prover policy  $\mu$  respectively. The term  $d_s^{\pi}$  denotes the distribution over future states, starting from state s, and following policy  $\pi$ . Here, s can be a state at any time  $h \in [0, \ldots, H]$ . For convenience, we overload the notation  $d_s^{\pi}$  (the distribution over future states induced by a policy starting from state s), and use  $d_{\rho}^{\pi}$  to denote the mixture distribution over  $d_s^{\pi}$  starting from a random state s drawn from  $s \sim \rho$ , and following policy  $\pi$ .

The term  $Q^{\pi}(s_h, a_h)$  refers to the value of state-action pair  $s_h, a_h, i.e.$ , the expected return in the future, starting the policy from state  $s_h$  and taking action  $a_h$ :

$$Q^{\pi}(\mathbf{s}_h, a_h) := \mathbb{E}_{a_h, \dots, a_H \sim \pi(\cdot | \mathbf{s}_h, a_h)} \left[ \text{Rex} \left( (a_1, \dots, a_H), \mathbf{y}_{\mathbf{x}}^{\star} \right) \right]. \tag{7}$$

Note that  $y_x^*$  is known on the dataset  $\mathcal{D}$ , and state  $s_h$  contains the question x as part of it. Similarly, we can define the value function  $V^{\pi}(s_h)$  of a state  $s_h$  as:

$$V^{\pi}(\mathbf{s}_h) := \mathbb{E}_{a_{h+1} \sim \pi(\cdot | \mathbf{s}_h)} Q^{\pi}(\mathbf{s}_h, a_h). \tag{8}$$

The advantage function is then given by:

$$A^{\pi}(s_h, a_h) := Q^{\pi}(s_h, a_h) - V^{\pi}(s_h). \tag{9}$$

The policy gradient algorithm we use to update the base policy iteratively is natural policy gradient (Kakade, 2001b), and we use  $\pi_t$  to refer to the base policy iterate at time t of this iterative algorithm.

Finally, we use S to denote the set of all states (prefixes) and  $\mathcal{A}$  for the set of all actions (steps) that the LLM can take at any state.

Parameterization of the base policy. We adopt the softmax parameterization for the base policy:

$$\pi_{\theta}(a \mid s_h) = \frac{\exp(\theta_{s_h,a})}{\sum_{a' \in \mathcal{A}} \exp(\theta_{s_h,a'})}.$$
 (10)

Here  $\theta_{s_h,a} \in \Theta \subseteq \mathbb{R}^d$  controls the probability of taking action a at state  $s_h$ . The full set of parameters across all states and actions is denoted by  $\theta \in \mathbb{R}^{d \times |\mathcal{S}| \times |\mathcal{A}|}$ . Whenever clear from context, we overload the notation  $\pi_t$  to denote both the policy at iterate t, i.e.,  $\pi_{\theta_t}$  and the parameter  $\theta_t$  itself. E.g., the gradient operator  $\nabla_{\pi_t}[\cdot]$  is referring to  $\nabla_{\theta_t}[\cdot]$ .

**Defining policy improvement.** Let  $\rho$  be a distribution over all states  $\{s_h : h \in [0, 1, ..., H]\}$ , then  $\mathbb{E}_{s \sim \rho} V^{\pi}(s)$ , and  $\mathbb{E}_{s \sim \rho} V^{\mu}(s)$  give us the expected value functions over states across time steps, measured under  $\rho$ , for policies  $\pi$  and  $\mu$  respectively. We assume that  $d_h^{\pi}$  and  $d_h^{\mu}$  are both absolutely continuous with respect to  $\rho$ , and use the expected value function over  $\rho$  as the quantity we track before and after a policy update. A positive change in  $\mathbb{E}_{s \sim \rho} V^{\pi}(s)$  implies a net positive improvement in the base policy. Thus, progress is made at each update of the policy when:

$$\mathbb{E}_{\boldsymbol{s}\sim\rho}V^{\pi_{t+1}}(\boldsymbol{s}) - \mathbb{E}_{\boldsymbol{s}\sim\rho}V^{\pi_t}(\boldsymbol{s}) > 0.$$

#### F.1. Natural Policy Gradient

The natural policy gradient (NPG) algorithm (Kakade, 2001a) defines a Fisher information matrix (induced by the policy), and performs gradient updates in the geometry induced by the following matrix:

$$F_{\rho}(\pi) = \mathbb{E}_{s \sim d_{\rho}^{\pi}} \mathbb{E}_{a \sim \pi(\cdot \mid s)} \left[ \nabla_{\pi} \log \pi(a \mid s) \left( \nabla_{\pi} \log \pi(a \mid s) \right)^{\top} \right]$$
(11)

Typically, the NPG update does gradient updates on the objective  $\ell_{ORM-RL}$  in Eq. 3, but in our case, the objective of interest is  $\ell_{PAV-RL}$  in Eq. 4, and thus the natural gradient is given by:

<span id="page-24-0"></span>
$$\pi_{t+1} = \pi_t + \gamma \cdot F_{\rho}(\pi^t)^{\dagger} \left( \nabla_{\pi} \ell_{\text{PAV-RL}}(\pi) \Big|_{\pi = \pi_t} \right), \tag{12}$$

where  $M^{\dagger}$  denotes the Moore-Penrose pseudoinverse of the matrix M. We restrict to using the initial state distribution  $\rho$  in our update rule, *i.e.*, we restrict attention to states s reachable from  $\rho$ , since  $\rho$  governs the performance measure of interest when evaluating the expected value of a policy. Thus, without loss of generality, we can exclude states that are not reachable under  $\rho$ . Specifically, we restrict the MDP to the set:  $\{s_h : \exists \pi \text{ such that } d^{\pi}_{\rho}(s_h) > 0, h \in [0, \ldots, H]\}$ . The scalar  $\gamma > 0$  determines the learning rate.

#### F.2. Useful Lemmas

<span id="page-24-1"></span>**Lemma F.1.** [The performance difference lemma; (Kakade and Langford, 2002)] For all policies  $\pi$ ,  $\pi'$  and states  $s_0$ ,

$$V^{\pi}(\mathbf{s}) - V^{\pi'}(\mathbf{s}) = \mathbb{E}_{\mathbf{s}_h \sim d_{\mathbf{s}}^{\pi}} \mathbb{E}_{a_h \sim \pi(\cdot | \mathbf{s}_h)} \left[ A^{\pi'}(\mathbf{s}_h, a_h) \right].$$

*Proof.* See proof of Lemma 6.1 in Kakade and Langford (2002).

<span id="page-25-0"></span>**Lemma F.2** (Natural policy gradient update). For the natural policy gradient in Eq. 12, the corresponding policy update is given by:

$$\pi^{t+1}(a \mid s) = \pi^{t}(a \mid s) \cdot \frac{\exp(\gamma \cdot (Q^{t}(s, a) + \alpha \cdot A^{\mu}(s, a)))}{Z^{t}(s)},$$
(13)

$$Z^{t}(s) = \gamma \cdot \sum_{a \in \mathcal{A}} \left( Q^{t}(s, a) + \alpha \cdot A^{\mu}(s, a) \right)$$
(14)

*Proof.* We use arguments similar to the proof of Lemma 15 in Agarwal et al. (2021), with the key difference of separately accounting for the term  $A^{\mu}(s,a)$  in the effective reward. For the sake of completeness we reproduce some of the derivation, accounting for the  $A^{\mu}$  term in the process. We follow compatible function approximation in Sutton et al. (1999) and Kakade (2001a). For a vector  $\mathbf{w} \in \mathbb{R}^{d \times |\mathcal{S}||\mathcal{A}|}$ , we define the error function

$$L^{\pi}(\boldsymbol{w}) = \mathbb{E}_{\boldsymbol{s} \sim d_{\rho}^{\pi}}, \mathbb{E}_{\boldsymbol{a} \sim \pi(\cdot | \boldsymbol{s})} \left[ \boldsymbol{w}^{\mathsf{T}} \nabla_{\boldsymbol{\pi}} \log \pi(\cdot | \boldsymbol{s}) - \left( A^{\pi}(\boldsymbol{s}, \boldsymbol{a}) + \alpha A^{\mu}(\boldsymbol{s}, \boldsymbol{a}) - \alpha \mathbb{E}_{\boldsymbol{a} \sim \pi_{t}(\cdot | \boldsymbol{s})} A^{\mu}(\boldsymbol{s}, \boldsymbol{a}) \right) \right]^{2}.$$
 (15)

Let  $\mathbf{w}^{\star}$  be the minimizer of  $L^{\pi}(\mathbf{w})$  with the smallest  $\ell_2$  norm. Then by definition of Moore-Penrose pseudoinverse:

$$\boldsymbol{w}^{\star} = F_{\rho}(\pi)^{\dagger} \mathbb{E}_{\boldsymbol{s} \sim d_{\rho}^{\pi}, a \sim \pi(a|\boldsymbol{s})} \left[ \nabla_{\pi} \log \pi(a|\boldsymbol{s}) \left( A^{\pi}(\boldsymbol{s}, a) + \alpha A^{\mu}(\boldsymbol{s}, a) - \alpha \mathbb{E}_{a \sim \pi_{t}(\cdot|\boldsymbol{s})} A^{\mu}(\boldsymbol{s}, a) \right] \right]$$

$$= F_{\rho}(\pi)^{\dagger} \nabla_{\pi} \ell_{\text{PAV-RL}}(\pi). \tag{16}$$

In other words,  $w^*$  is precisely proportional to the NPG update direction. Note further that for the Softmax policy parameterization, we have:

$$\boldsymbol{w}^{\top} \nabla \log \pi(a|s) = \boldsymbol{w}_{s,a} - \sum_{a' \in \mathcal{A}} \boldsymbol{w}_{s,a'} \pi(a'|s).$$

Since  $\sum_{a \in \mathcal{A}} \pi(a|s) A^{\pi}(s, a) = 0$ , this immediately yields that:

$$L^{\pi}(A^{\pi}(\boldsymbol{s},a) + \alpha A^{\mu}(\boldsymbol{s},a)) = 0.$$

However, this might not be the unique minimizer of  $L^{\pi}$ , which is problematic since  $\boldsymbol{w}^{\star}(\pi)$  as defined in terms of the Moore-Penrose pseudoinverse is formally the smallest norm solution to the least-squares problem, which  $A^{\pi} + \alpha A^{\mu}$  may not be. However, given any vector  $\boldsymbol{v} \in \mathbb{R}^{|S| \times |\mathcal{A}|}$ , let us consider solutions of the form  $A^{\pi} + \alpha A^{\mu} + \boldsymbol{v}$ . Due to the form of the derivatives of the policy for the softmax parameterization, we have for any state  $\boldsymbol{s}$ ,  $\boldsymbol{a}$  such that  $\boldsymbol{s}$  is reachable under  $\boldsymbol{\rho}$ ,

$$\boldsymbol{v}^{\top} \nabla_{\boldsymbol{\pi}} \log \boldsymbol{\pi}(\boldsymbol{a} \mid \boldsymbol{s}) = \sum_{\boldsymbol{a}' \in \mathcal{A}} (\boldsymbol{v}_{\boldsymbol{s}, \boldsymbol{a}'} \mathbf{1}[\boldsymbol{a} = \boldsymbol{a}'] - \boldsymbol{v}_{\boldsymbol{s}, \boldsymbol{a}'} \boldsymbol{\pi}(\boldsymbol{a}' \mid \boldsymbol{s})) = \boldsymbol{v}_{\boldsymbol{s}, \boldsymbol{a}} - \sum_{\boldsymbol{a}' \in \mathcal{A}} \boldsymbol{v}_{\boldsymbol{s}, \boldsymbol{a}'} \boldsymbol{\pi}(\boldsymbol{a}' \mid \boldsymbol{s}).$$

This is because  $\pi$  is a stochastic policy with  $\pi(a \mid s) > 0$  for all actions a in each state s, so that if a state is reachable under  $\rho$ , it will also be reachable using  $\pi$ , and hence the zero derivative conditions apply at each reachable state. For  $A^{\pi} + \alpha A^{\mu} + v$  to minimize  $L^{\pi}$ , we would like  $v^{\top} \nabla_{\pi} \log \pi(a \mid s) = 0$  for all s, a so that  $v_{s,a}$  is independent of the action and can be written as a constant  $c_s$  for each s by the above equality. Hence, the minimizer of  $L^{\pi}(w)$  is determined up to a state-dependent offset, and

$$F_{\rho}(\theta)^{\dagger}\nabla_{\pi}\ell_{PAV-RL}(\pi) = Q^{\pi} + \alpha A^{\mu} + \boldsymbol{v},$$

where  $v_{s,a} = c_s$  for some  $c_s \in \mathbb{R}$  for each state s and action a. Finally, we observe that this yields the updates

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}^t + \gamma (Q^{\pi} + \alpha A^{\mu} + \boldsymbol{\nu}) \quad \text{and} \quad \boldsymbol{\pi}_{t+1}(a \mid \boldsymbol{s}) = \boldsymbol{\pi}_t(a \mid \boldsymbol{s}) \frac{\exp(\gamma Q^t(s, a) + \gamma \alpha A^{\mu}(s, a))}{Z^t(\boldsymbol{s})}.$$

Owing to the normalization factor  $Z^t(s)$ , the state dependent offsets cancel in the updates for  $\pi$ , which yields the statement of the lemma.

#### F.3. Proof of Theorem 3.1

*Proof.* For some notational convenience, we use  $V^t, V^{t+1}$ , to denote the value functions  $V^{\pi_t}, V^{\pi_{t+1}}$  for policies at  $\pi_t$ ,  $\pi_{t+1}$  at time t and t+1 respectively. Similarly, we use  $A^t, A^{t+1}$  for  $A^{\pi_t}, A^{\pi_{t+1}}$  respectively. For the distribution over states  $d_{\rho}^{\pi_{t+1}}$  induced by the policy  $\pi_{t+1}$ , starting from an initial distribution of states given by  $\rho$ , we simplify the notation and use  $d_{\rho}^{t+1}$ . Similarly we use  $d_{\rho}^t$  for  $d_{\rho}^{\pi_t}$ .

Next, for simplicity we set  $\alpha = 1$  in the natural policy gradient update in Lemma F.2. It is easy to see that the lower bound result we show holds for any value of  $\alpha > 0$ , and the term in the lower bound scales linearly with  $\alpha$ . Note, that this is not a free variable, and  $\alpha$  has to be O(1), since as we increase the value of  $\alpha$  we would have to correspondingly reduce  $\gamma$  for our result to hold. For now, we fix  $\alpha = 1$ .

From the policy difference Lemma F.1, we can write:

$$\mathbb{E}_{s \sim \rho} V^{t+1}(s) - \mathbb{E}_{s \sim \rho} V^t(s) = \mathbb{E}_{s \sim d_{\rho}^{t+1}} \mathbb{E}_{a \sim \pi^{t+1}(a|s)} \left[ A^t(s, a) \right]$$

$$\tag{17}$$

Next, from the natural policy gradient update in Lemma F.2, we can write  $A^{t+1}(s, a)$  as:

<span id="page-26-1"></span><span id="page-26-0"></span>
$$A^{t}(s,a) = \frac{1}{\gamma} \cdot \log \left( \frac{\pi^{t+1}(a \mid s) \cdot Z^{t}(s,a)}{\pi^{t}(a \mid s)} \right) - A^{\mu}(s,a)$$

$$\tag{18}$$

Substituting Eq. 18 in Eq. 17 we get:

$$\mathbb{E}_{s \sim \rho} V^{t+1}(s) - \mathbb{E}_{s \sim \rho} V^{t}(s) = \frac{1}{\gamma} \mathbb{E}_{s \sim d_{\rho}^{t+1}} \left[ KL(\pi^{t+1}(\cdot \mid s)) \| \pi^{t}(\cdot \mid s)) \right] + \frac{1}{\gamma} \log Z^{t}(s) - \mathbb{E}_{s \sim d_{\rho}^{t+1}} \mathbb{E}_{a \sim \pi^{t+1}(a|s)} A^{\mu}(s, a).$$
(19)

Recall that for  $\alpha = 1$ ,

$$\log Z^{t}(s) = \log \mathbb{E}_{a \sim \pi^{t}(\cdot \mid s)} \exp \left( \gamma \cdot (A^{t}(s, a) + A^{\mu}(s, a)) \right)$$
 (20)

Applying Jensen's inequality we get:

$$\log Z^{t}(s) \ge \gamma \cdot \mathbb{E}_{a \sim \pi^{t}(\cdot \mid s)} \left[ A^{t}(s, a) + A^{\mu}(s, a) \right]$$
 (21)

<span id="page-26-3"></span><span id="page-26-2"></span>
$$= \gamma \cdot \mathbb{E}_{a \sim \pi^t(\cdot | s)} \left[ A^{\mu}(s, a) \right], \tag{22}$$

since  $\mathbb{E}_{\pi^t}[A^t(s,a)] = 0$ . Note that in Eq. 20 the KL term is always non-negative. Thus, we can lower bound our policy improvement:

$$\mathbb{E}_{s \sim \rho} V^{t+1}(s) - \mathbb{E}_{s \sim \rho} V^{t}(s) \ge \mathbb{E}_{s \sim d_{\rho}^{t+1}} \langle \pi^{t+1}(\cdot \mid s) - \pi^{t}(\cdot \mid s), A^{\mu}(s, \cdot) \rangle, \tag{23}$$

where the inner product is the standard euclidean product as our actions space  $\mathcal{A}$  is discrete.

In the following we will treat the distribution  $\pi^{t+1}(\cdot \mid s)$  as a vector denoted by  $\pi$  Next, from the NPG update we know that:

<span id="page-27-0"></span>
$$\pi^{t+1}(a \mid s) - \pi^{t}(a \mid s) = \pi^{t}(a \mid s) \left( \frac{\exp(\gamma A^{t}(s, a) + \gamma A^{\mu}(s, a))}{Z^{t}(s)} - 1 \right)$$
 (24)

We note that for  $\gamma \ll 1$ ,  $\exp \gamma(A^t(s,a)) + \gamma(A^\mu(s,a)) = \Theta(1 + \gamma(A^t(s,a) + A^\mu(s,a)))$ , where the terms that grow as  $\omega(\gamma)$  are being ignored. Based on this, for  $\gamma \ll 1$ ,  $\exists$  constants  $0 < C_1 < C_2$  such that:

$$\exp(\gamma A^{t}(s, a) + \gamma A^{\mu}(s, a)) - 1 \in [C_{1}\gamma(A^{t}(s, a) + A^{\mu}(s, a)), C_{2}\gamma(A^{t}(s, a) + A^{\mu}(s, a))]$$

Applying the above claim in Eq. 24 gives us:

$$\pi^{t+1}(a \mid s) - \pi^{t}(a \mid s) \geq \pi^{t}(a \mid s) \left( \frac{1 + C_{1}\gamma(A^{t}(s, a) + A^{\mu}(s, a))}{1 + C_{2}\gamma\mathbb{E}_{a\mid\pi^{t}(\cdot\mid s)} \left[A^{t}(s, a) + A^{\mu}(s, a)\right]} - 1 \right)$$

$$\geq C_{3}\gamma \frac{\left(\pi^{t}(a \mid s) \left(A^{t}(s, a) + A^{\mu}(s, a)\right) - \pi_{t}(a \mid s)\mathbb{E}_{a \sim \pi_{t}(a\mid s)} \left[A^{t}(s, a) + A^{\mu}(s, a)\right]\right)}{1 + C_{2}\gamma\mathbb{E}_{a \sim \pi_{t}(a\mid s)} \left[A^{t}(s, a) + A^{\mu}(s, a)\right]}$$

$$(25)$$

$$= C_3 \gamma \frac{\left(\pi^t(a \mid s) \left(A^t(s, a) + A^{\mu}(s, a)\right) - \pi_t(a \mid s) \mathbb{E}_{a \sim \pi_t(a \mid s)} \left[A^{\mu}(s, a)\right]\right)}{1 + \gamma C_2 \mathbb{E}_{a \sim \pi_t(a \mid s)} \left[A^t(s, a) + A^{\mu}(s, a)\right],}$$
(26)

where we reused:  $\mathbb{E}_{\pi^t}[A^t(s, a)] = 0$ . Here,  $C_3 > 0$  is a constant.

We now plug in the above lower bound into Eq. 21 to get the final lower bound on the policy improvement in Theorem 3.1. For this, we will once again use the assumption that the learning rate  $\gamma \ll 1$ , which allows us to use  $1 + \gamma \mathbb{E}_{a \sim \pi_t(a|s)} \left[ A^t(s,a) + A^{\mu}(s,a) \right] \ge C_4$  for some constant  $C_4 > 0$ . This is because, in our setting the range of the advantages is [-1,1].

$$\mathbb{E}_{s \sim \rho} \left[ V^{t+1}(s) - V^{t}(s) \right] \gtrsim \gamma \mathbb{E}_{s \sim d_{\rho}^{t+1}} \left[ \mathbb{E}_{a \sim \pi^{t}(a|s)} \left[ A^{\mu}(s, a) A^{t}(s, a) \right] \right]$$

$$+ \gamma \mathbb{E}_{s \sim d_{\rho}^{t+1}} \left[ \mathbb{E}_{a \sim \pi^{t}(a|s)} \left[ \left( A^{\mu}(s, a) \right)^{2} \right] \right]$$

$$- \gamma \mathbb{E}_{s \sim d_{\rho}^{t+1}} \left[ \left( \mathbb{E}_{a \sim \pi^{t}(a|s)} \left[ A^{\mu}(s, a) \right] \right)^{2} \right]$$

$$(27)$$

Now Eq. 27 gives us,

$$\mathbb{E}_{s \sim \rho} \left[ V^{t+1}(s) - V^t(s) \right] \gtrsim \gamma \mathbb{E}_{s \sim d_{\rho}^{t+1}} \left[ \mathbb{V}_{a \sim \pi_t(a|s)} \left[ A^{\mu}(s, a) \right] - \mathbb{E}_{a \sim \pi_t(a|s)} \left[ A^{\mu}(s, a) A^t(s, a) \right] \right]$$
(28)

Now, for the last step we note that  $d_{\rho}^{t+1}$  is component wise larger than  $\rho$ , and this gives us the final result:

$$\mathbb{E}_{s \sim \rho} \left[ V^{t+1}(s) - V^{t}(s) \right] \gtrsim \gamma \mathbb{E}_{s \sim \rho} \left[ \mathbb{V}_{a \sim \pi_{t}(a|s)} \left[ A^{\mu}(s, a) \right] - \mathbb{E}_{a \sim \pi_{t}(a|s)} \left[ A^{\mu}(s, a) A^{t}(s, a) \right] \right]$$
(29)

#### F.4. Discussion on Remark 3.1

First, we note that if the *Q*-value of a base policy  $\pi$  at state, action pair (s, a) is  $Q^{\pi}(s, a)$ , then for the prover  $\mu$  set to the best-of-K policy BoK $(\pi)$ , the *Q*-value at the same state, action pair is:

$$Q^{\mu}(\mathbf{s}, a) = 1 - (1 - Q^{\pi}(\mathbf{s}, a))^{K}$$
(30)

<span id="page-27-2"></span>28

<span id="page-27-1"></span>

This is because, in our setup the final outcome of correctness given by Rex is a random variable taking values in  $\{0, 1\}$ , with expectation  $Q^{\pi}(s, a)$ , when completing a rollout starting from prefix (s, a). Thus, we can treat the outcome of function Rex as a Bernoulli random variable. Next, we can simply compute the probability of sampling a single correct answer out of K attempts, which is also a Bernoulli random variable with mean given by  $Q^{\mu}(s, a)$  in Eq. 30.

Now for Remark 3.1, we observe that whenever  $Q^{\pi}(s, a) \ll 1$ , e.g., when  $Q^{\pi}(s, a) = O(1/K)$ , for all values of (s, a), we can do the Taylor approximation of  $Q^{\mu}(s, a)$  around 0, and note that  $Q^{\mu}(s, a) = \Theta(K \cdot Q^{\pi}(s, a))$ . Next, note the following calculation for the first term (distinguishability) in Theorem 3.1:

$$\begin{aligned} \mathbb{V}_{\pi} A^{\mu}(\boldsymbol{s}, a) &= \mathbb{V}_{\pi} Q^{\mu}(\boldsymbol{s}, a) \\ &= \mathbb{V}_{\pi} \left[ 1 - (1 - Q^{\pi}(\boldsymbol{s}, a))^{K} \right] \\ &= \mathbb{V}_{\pi} \left[ (1 - Q^{\pi}(\boldsymbol{s}, a))^{K} \right] \end{aligned}$$

This means that the first term in Theorem 3.1 which measures "distinguishability" now increases by a factor of  $K^2$ . Similarly, we can see that the term which measures "misalignment" can change in magnitude by atmost a factor of O(K), since the misalignment term is linear in  $Q^{\mu}$ . These two observations combined lead us to the conclusion in Remark 3.1.

#### <span id="page-28-1"></span>F.5. Improving a Stronger Base policy with a Weaker Prover Policy

In Proposition F.1, we consider the case where the  $\pi$  and  $\mu$  differ in performance, as measured under the distribution of states  $\rho$  in the following way:

$$\mathbb{E}_{\boldsymbol{s}\sim\rho}[|V^{\mu}(\boldsymbol{s})-V^{\pi_t}(\boldsymbol{s})|]=\eta.$$

Next, whenever the prover's preference over actions is complementary to the base policy, by a factor of  $\eta$ , i.e.,

$$\mathbb{E}_{s\sim\rho}\mathbb{E}_{a\sim\pi}[|Q^{\mu}(s,a)-Q^{\pi_t}(s,a)|]=\Theta(\eta),$$

then the variance of  $A^{\mu}$  or  $A^{\pi_t}$  under  $\pi_t$  should scale as  $\eta^2$ .

Thus, we see that when  $\pi_t$  fails to distinguish actions (*i.e.*,  $\mathbb{E}_{s \sim \rho} \mathbb{V}_{\pi_t}[A^{\pi_t}(s, a)]$  is small) regardless of the strength of prover policy  $\mu$ , as long as it is sufficiently complementary to  $\pi_t$ , the prover policy induces an improvement in base policy, that scales as  $\eta^2$ .

<span id="page-28-0"></span>**Proposition F.1** (Complementary  $\mu$  boosts improvements in  $\pi$ ). Under the distribution over states given by  $\rho$ , let prover  $\mu$  and base policy at iterate t,  $\pi_t$ , differ in absolute performance, i.e.,

$$\mathbb{E}_{\boldsymbol{s}\sim\rho}[|V^{\mu}(\boldsymbol{s})-V^{\pi_t}(\boldsymbol{s})|]=\eta.$$

When  $\mathbb{E}_{s \sim \rho} \mathbb{V}_{a \sim \pi_t}[A^{\pi_t}(s, a)] < \mathbb{E}_{s \sim \rho} \mathbb{V}_{a \sim \pi_t}[A^{\mu}(s, a)]$ , and  $\mu$  is complementary to  $\pi_t$ , i.e.,

$$\mathbb{E}_{\boldsymbol{s}\sim\rho}\mathbb{E}_{\boldsymbol{a}\sim\pi_t}|Q^{\pi_t}(\boldsymbol{s},\boldsymbol{a})-Q^{\mu}(\boldsymbol{s}_h,\boldsymbol{a})|=\Theta(\eta),$$

then 
$$\mathbb{E}_{s\sim\rho}[V^{\pi_{t+1}}(s)-V^{\pi_t}(s)]\gtrsim\eta^2$$
.

*Proof.* We begin by proving an upper bound on the disagreement between prover and base policy:

$$\begin{split} & \mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_{t}} | Q^{\mu}(s, a) - Q^{\pi_{t}}(s, a) | \\ & \leq \mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_{t}} | Q^{\mu}(s, a) - V^{\mu}(s) | + \mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_{t}} | Q^{\pi_{t}}(s, a) - V^{\mu}(s) | \\ & \leq \mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_{t}} | Q^{\mu}(s, a) - V^{\mu}(s) | + \mathbb{E}_{s \sim \rho} [|V^{\mu}(s) - V^{\pi_{t}}(s)|] + \mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_{t}} | Q^{\pi_{t}}(s, a) - V^{\pi_{t}}(s, a) | \\ & \leq \eta + \mathbb{E}_{s \sim \rho} \sqrt{\mathbb{V}_{\pi_{t}} [A^{\mu}(s, a)]} + \mathbb{E}_{s \sim \rho} \sqrt{\mathbb{V}_{\pi_{t}} [A^{\pi_{t}}(s, a)]}, \end{split}$$

where the last inequality uses Cauchy-Schwartz. Next we apply Jensen's inequality on the terms in the square root, and the conditions that  $\mathbb{E}_{s \sim \rho} \mathbb{E}_{a \sim \pi_t} |Q^{\pi_t}(s, a) - Q^{\mu}(s_h, a)| = \Omega(\eta)$  to conclude:

$$\sqrt{\mathbb{E}_{s\sim\rho}\mathbb{V}_{\pi_t}[A^{\mu}(s,a)]}\gtrsim\eta,$$

whenever  $\mathbb{E}_{s\sim\rho}\mathbb{V}_{a\sim\pi_t}[A^{\pi_t}(s,a)] < \mathbb{E}_{s\sim\rho}\mathbb{V}_{a\sim\pi_t}[A^{\mu}(s,a)]$ . Now that we lower bound "distinguishability", it is easy to see that a similar derivation would upper bound the magnitude of the misalignment term by  $O(\eta^2)$ . Invoking the result in Theorem 3.1 yields:

$$E_{\boldsymbol{s}\sim\rho}[V^{\pi_{t+1}}(\boldsymbol{s})-V^{\pi_t}(\boldsymbol{s})]\gtrsim\eta^2$$

#### 

# <span id="page-29-0"></span>G. Examples Generated by Base Policy Trained on Rewards $Q^{\pi} + \alpha Q^{\mu}$

When we train the base policy with reinforcement learning where the reward is  $Q^{\pi} + \alpha Q^{\mu}$ , instead of the effective reward  $Q^{\pi} + \alpha A^{\mu}$ , we find that the policy training converges to contrived solutions with degenerate failures.

For example, we see in multiple examples (like the four below), that for every question, the model learns to output "REPHRASE THE PROBLEM" as the first step. This is because under  $Q^{\mu}$  this trivial step achieves a positive reward. While there exists a better solution where the policy outputs the correct answer and achieves higher rewards, it fails to recover from this local optima and find the better solution. Across multiple training runs, we observed similar trends when training with  $Q^{\pi} + \alpha Q^{\mu}$ . Note that in the idealized update, where we observe advantages  $A^{\mu}$  or value function  $Q^{\mu}$  on all actions at a state (*e.g.*, in the NPG update in Sec. 3.4), using  $Q^{\pi} + \alpha Q^{\mu}$  is equivalent to  $Q^{\pi} + \alpha A^{\mu}$ . But, in practice,  $A^{\mu}$  offers significant benefits in the form of variance reduction in policy gradient iterates. In this instance, it does that by not assigning reward to steps like "REPHRASE THE PROBLEM", since the likelihood of a prover solving before or after this step is expected to be roughly the same under any choice of the prover policy (*i.e.*, the prover policy makes no progress).

```
Example 1

Q: My club has 25 members. In how many ways can I choose members to form a 4-person executive committee?

A:

>>>
REPHRASE THE PROBLEM
WE NEED TO FIND THE NUMBER OF WAYS TO CHOOSE 4 MEMBERS OUT OF 25 FOR THE EXECUTIVE COMMITTEE SO WE CAN USE THE COMBINATION FORMULA WHICH IS
```

```
N C R = N! / (R! * (N - R)!) WHERE N IS THE TOTAL NUMBER OF MEMBERS AND R IS
THE NUMBER OF MEMBERS WE CHOOSE SO IN THIS CASE N = 25 AND R = 4 SO WE PLUG THEM IN AND
SIMPLIFY 25 C 4 = 25! / (4! * (25 - 4)!) = (25 * 24 * 23 * 22) / (4 * 3 * 2 * 1) = 104005
# Answer
104005
```

#### Example 2

```
Q: We have a triangle $\triangle ABC$ and a point $K$ on $BC$ such that $AK$ is
an altitude of $\triangle ABC$. If $AC = 10,$ $BK = 7$, and $BC = 13,$ then
what is the area of $\triangle ABC$?
A:
>>>
REPHRASE THE PROBLEM
WE NEED TO FIND THE AREA OF $\triangle ABC$ SO WE CAN USE THE FORMULA $A = \frac{1}{2}bh$
WHERE $b$ IS THE BASE AND $h$ IS THE HEIGHT SO SINCE $AK$ IS AN ALTITUDE, THAT MEANS
$AK \perp BC$ SO WE CAN USE $BC$ AS THE BASE AND $AK$ AS THE HEIGHT SO NOW WE NEED
TO FIND $AK$ SO WE CAN USE THE PYTHAGORE THEOREM ON $\triangle AKC$ SINCE $AC = 10$
# Answer
48
```

## Example 3

```
Q: Suppose that $n, n+1, n+2, n+3, n+4$ are five consecutive integers.
Determine a simplified expression for the sum of these five consecutive integers.
A:
>>>
REPHRASE THE PROBLEM
WE NEED TO FIND THE SUM OF THESE FIVE CONSECUTIVE INTEGERS SO WE CAN USE THE FORMULA
$S_n = \frac{n}{2}(a_1 + a_n)$ WHERE $S_n$ IS THE SUM, $n$ IS THE NUMBER OF TERMS,
$a_1$ IS THE FIRST TERM, AND $a_n$ IS THE LAST TERM SO IN THIS CASE $n = 5, a_1 = n,
a_5 = n + 4$
# Answer
5n + 10
```

#### Example 4

```
Q: What is the value of $x^2+y^2-z^2+2xy$ when $x=-3$, $y=5$, and $z=-4$?
A:
>>>
REPHRASE THE PROBLEM
WE NEED TO FIND THE VALUE OF $x^2+y^2-z^2+2xy$ WHEN $x=-3$, $y=5$, AND $z=-4$
SO WE JUST PLUG IN THE VALUES AND SIMPLIFY
# Answer
-2
```