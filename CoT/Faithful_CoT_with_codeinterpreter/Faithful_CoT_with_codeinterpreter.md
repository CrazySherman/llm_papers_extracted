# Faithful Chain-of-Thought Reasoning

Qing Lyu <sup>∗</sup> Shreya Havaldar<sup>∗</sup> Adam Stein<sup>∗</sup> Li Zhang Delip Rao Eric Wong Marianna Apidianaki Chris Callison-Burch

University of Pennsylvania

{lyuqing, shreyah, steinad, zharry, exwong, marapi, ccb}@seas.upenn.edu, deliprao@gmail.com

# Abstract

While Chain-of-Thought (CoT) prompting boosts Language Models' (LM) performance on a gamut of complex reasoning tasks, the generated reasoning chain does not necessarily reflect how the model arrives at the answer (aka. *faithfulness*). We propose Faithful CoT, a reasoning framework involving two stages: Translation (Natural Language query → symbolic reasoning chain) and Problem Solving (reasoning chain → answer), using an LM and a deterministic solver respectively. This guarantees that the reasoning chain provides a faithful explanation of the final answer. Aside from interpretability, Faithful CoT also improves empirical performance: it outperforms standard CoT on 9 of 10 benchmarks from 4 diverse domains, with a relative accuracy gain of 6.3% on Math Word Problems (MWP), 3.4% on Planning, 5.5% on Multi-hop Question Answering (QA), and 21.4% on Relational Inference. Furthermore, with GPT-4 and Codex, it sets the new state-of-the-art few-shot performance on 7 datasets (with 95.0+ accuracy on 6 of them), showing a strong synergy between faithfulness and accuracy.[1](#page-0-0)

# <span id="page-0-2"></span>1 Introduction

Complex reasoning tasks, such as commonsense reasoning and math reasoning, have long been the Achilles heel of LMs [\(Bengio,](#page-9-0) [2019\)](#page-9-0), until a recent line of work on Chain-of-Thought (CoT) reasoning [\(Wei et al.,](#page-11-0) [2022;](#page-11-0) [Wang et al.,](#page-11-1) [2022,](#page-11-1) i.a.) brought striking performance gains. CoT prompts an LM to generate a reasoning chain along with the answer, given only a few in-context exemplars.

Besides performance improvement, CoT is also claimed to "provide an interpretable window into the behavior of the model" [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0). However, it is unclear how accurately these reason<span id="page-0-1"></span>Q: John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for \$15 each. He ends up buying 8 videogames for \$20 each and has \$5 left. How many lego sets does he still have? Input CoT output

A: He started with 13 lego sets. He sold them for \$15 each. So he must have gotten 13 \* \$15 = \$195 from selling them. He used this money to buy 8 video games for \$20 each. This means he spent 8 \* \$20 = \$160 on the video games. He had \$5 left, so in total he must have had \$195 + \$5 = \$200. The answer is 0.

Figure 1: An example of *unfaithful* output from CoT prompting [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0) on GSM8K. The answer (green) does not follow from the reasoning chain (blue).

ing chains reflect the underlying reasoning process behind the model's prediction, namely, how *faithful* they are as explanations [\(Jacovi and Gold](#page-10-0)[berg,](#page-10-0) [2020,](#page-10-0) i.a.). In standard CoT, faithfulness is not guaranteed and even systematically violated [\(Turpin et al.,](#page-11-2) [2023\)](#page-11-2), as the final answer does not necessarily follow from the generated reasoning chain. In other words, CoT can "lie" about the model's true reasoning process. Figure [1](#page-0-1) exemplifies such an unfaithful CoT generation from [Wei](#page-11-0) [et al.](#page-11-0) [\(2022\)](#page-11-0) on GSM8K: the answer "0" is not even mentioned in the reasoning chain. This, along with more examples in Appendix [B.1,](#page-12-0) illustrates that standard CoT does not provide interpretability of how the model predicts the answer.

The lack of faithfulness in CoT can be dangerous in high-stake applications because it may mislead people into believing that the model is selfinterpretable, while there is no actual causal relationship between the reasoning chain and the answer. Even worse, when an *unfaithful* explanation looks *plausible* (i.e., convincing to humans) [\(Jacovi](#page-10-0) [and Goldberg,](#page-10-0) [2020\)](#page-10-0), this makes it easier for people (e.g., legal practitioners) to over-trust the model (e.g., a recidivism predictor) even if it has implicit biases (e.g., against racial minorities) [\(Pruthi et al.,](#page-10-1) [2020;](#page-10-1) [Slack et al.,](#page-11-3) [2020\)](#page-11-3).

To address this concern, we propose Faithful CoT, a reasoning framework where the answer is the result of deterministically executing the reasoning chain. Specifically, we break down a com-

<span id="page-0-0"></span><sup>∗</sup> Equal contribution.

<sup>1</sup>Our code, data, and prompts are available at [https://](https://github.com/veronica320/Faithful-COT) [github.com/veronica320/Faithful-COT](https://github.com/veronica320/Faithful-COT).

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 2: An overview of our Faithful CoT framework, consisting of Translation, where an LM translates a query (in NL/Natural Language) into a reasoning chain (which interleaves NL and SL/Symbolic Language), and Problem Solving, where an external solver executes the reasoning chain to derive the answer.

plex reasoning task into two stages: Translation and Problem Solving (Figure [2\)](#page-1-0). During Translation, an LM translates a query into a reasoning chain, which interleaves NL and Symbolic Language (SL). The NL component decomposes the original query into multiple simpler, interdependent subproblems. Then, each subproblem is tackled in a task-dependent SL, such as Python, Datalog, or Planning Domain Definition Language (PDDL). In the Problem Solving stage, the reasoning chain is executed by a deterministic solver, e.g., a Python/Datalog interpreter, or a PDDL planner, to derive the answer.

Our reasoning chain (outcome of Translation) is guaranteed to provide a faithful explanation of how the final answer is produced (outcome of Problem Solving), therefore making our method more *interpretable* than standard CoT methods.[2](#page-1-1) While interpretability is not the same as correctness (i.e. our method can reveal the reasoning process behind *both* correct *and* wrong answers), we find that it does empirically improve correctness: when evaluated on 10 reasoning datasets from 4 diverse domains (MWP, Planning, Multi-hop QA, and Relational Inference), Faithful CoT brings consistent performance gains over three existing baselines, across different LMs and decoding strategies. With Codex, our approach outperforms vanilla CoT on 9 of the 10 datasets, with a relative accuracy gain of 6.3% on MWP, 3.4% on Planning, 5.5% on Multihop QA, and 21.4% on Relational Inference. With GPT-4, our method sets the new SOTA few-shot

performance on 7 datasets, with 95.0+ accuracy on 6 of them. This suggests that interpretability does not have to come at the cost of performance; instead, there exists a strong synergy in between.

Our key contributions are as follows:

- (a) We propose Faithful CoT, a framework that decomposes reasoning into Translation and Problem Solving. The reasoning chain interleaves userunderstandable natural language comments and executable symbolic language programs, thus providing faithful interpretability of how the model arrives at the answer.
- (b) Our approach is generalizable to multiple domains beyond arithmetic reasoning and simple symbolic reasoning, thanks to its flexible integration with any choice of SL and external solver. We set the new SOTA performance on 7 out of the 10 reasoning datasets, showing a strong synergy between faithfulness and accuracy.
- (c) We provide an extensive analysis of the strengths and weaknesses of our method, showing its generalizability across LMs, robustness to the choice of exemplars and prompt phrasing, the pivotal role of the solver, the plausibility of generated reasoning chains, as well as frequent error patterns where it still struggles.

# <span id="page-1-3"></span>2 Related Work

Faithfulness. In interpretability, *faithfulness* (also called *fidelity* or *reliability*) means that an explanation should "accurately represent the reasoning process behind the model's prediction", which is a fundamental requirement of an explanation [\(Harrington et al.,](#page-10-2) [1985;](#page-10-2) [Ribeiro et al.,](#page-11-4) [2016;](#page-11-4) [Gilpin](#page-10-3) [et al.,](#page-10-3) [2018;](#page-10-3) [Jacovi and Goldberg,](#page-10-0) [2020\)](#page-10-0).[3](#page-1-2) It should be contrasted with *plausibility* (a.k.a. *persuasiveness* or *understandability*), which refers to "how convincing an explanation is to humans" [\(Herman,](#page-10-4) [2019;](#page-10-4) [Jacovi and Goldberg,](#page-10-0) [2020\)](#page-10-0). In the context of CoT prompting, a faithful reasoning chain needs to accurately reflect how the model arrives at the final answer, whereas a plausible reasoning chain is one that looks reasonable and coherent to humans. Standard CoT [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0) generates the rea-

<span id="page-1-1"></span><sup>2</sup>Note that we do not claim that the process of generating the reasoning chain itself, i.e., the Translation stage, is interpretable. See more discussion in Limitations.

<span id="page-1-2"></span><sup>3</sup>Note that this differs from the notion of faithfulness in the Natural Language Generation (NLG) literature, primarily in what constitutes the ground truth. In interpretability, we talk about the faithfulness of an explanation w.r.t. the model's underlying reasoning mechanism – the ground truth is usually unknown. In NLG, we talk about the faithfulness of the generated text (e.g., a translated sentence, or a summary) w.r.t. some explicit source (e.g., the source sentence, or the full document) – the ground truth is transparent.

soning chain in pure NL, which may often look plausible; nevertheless, the final answer does not need to causally follow from the reasoning chain, thus not guaranteeing faithfulness.

Chain-of-Thought-style prompting. In CoTstyle prompting, given a complex question Q, an LM is prompted to generate a reasoning chain C along with the final answer A. Specifically, the prompt consists of a few examples of (Q, C, A) triples, called in-context exemplars. This allows pre-trained LMs (e.g., GPT-3 [\(Brown et al.,](#page-9-1) [2020\)](#page-9-1)) to solve unseen questions with much higher accuracy than standard prompting, where the exemplars do not contain the reasoning chain C.

We create a taxonomy of existing CoT-style prompting methods into three types: all-at-once, ensemble-based, and modularized. All-at-once prompting means that the LM produces C and A as one continuous string, without any dependencies or constraints in between. Scratchpad [\(Nye et al.,](#page-10-5) [2021\)](#page-10-5), standard CoT [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0), and "Let's think step by step" [\(Kojima et al.,](#page-10-6) [2022\)](#page-10-6), are all examples of this kind. Ensemble-based prompting is designed to overcome the local optimality issue of the one-shot generation in previous methods by sampling multiple (C, A) pairs and choosing the best answer via strategies like majority voting. Examples include Self-Consistent CoT [\(Wang et al.,](#page-11-1) [2022\)](#page-11-1), Minerva [\(Lewkowycz et al.,](#page-10-7) [2022\)](#page-10-7), and DI-VERSE [\(Li et al.,](#page-10-8) [2022\)](#page-10-8), which differ mainly in the voting granularity and the underlying LM. Modularized methods break down Q into subproblems and then conquer them individually [\(Jung et al.,](#page-10-9) [2022;](#page-10-9) [Qian et al.,](#page-11-5) [2022,](#page-11-5) i.a.). In particular, Leastto-Most prompting [\(Zhou et al.,](#page-11-6) [2022\)](#page-11-6) has a similar question decomposition process to ours, but there is still no faithfulness guarantee since the reasoning chain is entirely in NL.

Concurrent with our work, [Chen et al.](#page-10-10) [\(2022\)](#page-10-10) and [Gao et al.](#page-10-11) [\(2022\)](#page-10-11) both generate Python programs (i.e., SL-only reasoning chains) to derive the answer. We want to highlight the following qualitative differences:[4](#page-2-0) (a) In terms of motivation, our approach is interpretability-driven, whereas theirs are performance-driven. (b) Our reasoning chain involves a structured decomposition of the problem in NL, allowing users without a programming background to better understand and potentially interact with the system. (c) They only use Python as the SL and only tackle math and simple symbolic

reasoning tasks, whereas we demonstrate the generalizability of our approach to multiple symbolic languages and various other domains. In particular, we innovatively recast a diverse set of realistic tasks (Planning, Multi-hop QA, and Relational Inference) into a symbolic representation, which allows us to tackle them with a single framework. (d) We perform a more comprehensive analysis compared to previous work, especially a human evaluation of the reasoning chain correctness.

# 3 Method

Our method, Faithful CoT, is a 2-stage pipeline, as seen in Figure [2.](#page-1-0) Like previous CoT-style work, our prompt consists of (Q, C, A) triples. Notable differences lie in our unique interleaving of NL (natural language) and SL (symbolic language) in C, as well as the way we derive the final answer A.

In the Translation stage, given a complex query Q in NL, we prompt an LM to translate it into a reasoning chain C, which interleaves NL comments and SL programs. The NL component decomposes the original query into multiple simpler, interdependent subproblems. Then, each subproblem is tackled in a task-dependent SL, such as Python, Datalog, or PDDL. In the Problem Solving stage, we call a deterministic external solver, e.g., a Python interpreter, a Datalog executor, or PDDL planner, depending on the task, to obtain the answer A from the reasoning chain C. As shown in Figure [3,](#page-3-0) we define CNL to be the NL component (black) and CSL to be the SL component (blue) in C. Though we separate the two components notationally, they are interleaved in the generation. Using this approach, C is guaranteed to be a faithful model explanation, since our final A is the result of deterministically executing CSL. Moreover, CNL allows the user to better understand the reasoning process.[5](#page-2-1)

We apply this method to 4 types of complex reasoning tasks: MWP, Multi-hop QA, Planning, and Relational Inference. Next, we will illustrate how our method works for each of them, with examples from Figure [3.](#page-3-0)

# 3.1 Math Word Problems (MWP)

Given a grade-school math question Q written in NL ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", shown in green in Figure [3\)](#page-3-0), we want to

<span id="page-2-0"></span><sup>4</sup>Also see Appendix [B.3](#page-13-0) for an empirical comparison.

<span id="page-2-1"></span><sup>5</sup>While no constraints are enforced between CNL and CSL in our main experiments, we analyze this in Section [C.4.](#page-14-0)

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3: Examples from each task (Math Word Problems, Multi-hop QA, Relational Inference, and Planning) showing our 2-stage Translation and Problem Solving pipeline.

obtain A as a real-valued number (5). In the Translation stage, we prompt the LM to take in Q and generate a reasoning chain C, which interleaves CNL and CSL. Specifically, the CNL component consists of three types of information:

- (a) Subquestions: Q is broken down into multiple smaller-scale subquestions, e.g., "1. how many cars are there in the beginning?", "2. how many cars arrive?", and "3. how many cars are in the parking lot?".
- (b) Dependency Graph: Each subquestion can either be answered directly via context (subquestions 1 and 2 are "independent") or rely on answers to previous subquestions (subquestion 3 "depends on 1 and 2").
- (c) Rationales: Each subquestion is accompanied with rationale(s) to support the answer (the "support" field). The rationales can be either a subset of the original context ("2 more cars arrive") or any external knowledge ("there are 7 days in a week") relevant to the subquestion.

Each subquestion and its corresponding dependencies and rationales inform the subsequent generation of CSL. In our example in Figure [3,](#page-3-0) CSL consists of Python code generated to answer each subquestion in CNL. During the Problem Solving stage, we execute CSL using our solver, a Python interpreter, to derive A (5 cars in the end).

# 3.2 Multi-hop QA

Given a complex question Q that involves multiple steps of reasoning (e.g., "Would a pear sink in water?", shown in red in Figure [3\)](#page-3-0), we want to obtain the answer A as a Boolean value or string value variable. Similar to our MWP task formulation, C interleaves CNL (NL comments), and CSL (symbolic program). Depending on the nature of the task, the format of the reasoning chain C is slightly different: for some datasets, the LM first generates all subquestions and their answers in NL, and then represents these answers as SL to derive A (see Figure [3\)](#page-3-0); for others, the LM interleaves the NL subquestions and the SL program, similar to the case of MWP (see Table [14](#page-24-0) and Table [15](#page-24-1) for examples). In terms of SL, we use both Python and Datalog, also depending on the dataset. As Multi-hop QA problems involve multi-step reasoning to solve, CSL often utilizes Boolean algebra and string comparisons (in Python) along with relation definitions and logic programming (in Datalog). We use their corresponding interpreter as our deterministic solver to execute CSL and obtain A.

In the example from Figure [3,](#page-3-0) the LM first generates the subquestions, "1. What is the density of a pear?" and "2. What is the density of water?", which are individually answered

in NL. The answers ("Water has a density of 1g/cm<sup>3</sup> ") are converted to Datalog statements (Has\_density("water", 1)), which are then combined to formalize the truth condition of the final answer. Finally, we execute the Datalog program to determine that a pear would not sink in water.

# 3.3 Planning

In a user-robot interaction scenario, given a household task query Q from a user, we want to come up with a plan of actions A that the robot should take in order to accomplish the task. For example, in Figure [3,](#page-3-0) given user query "I spilled my coke on the table, could you throw it away and bring something to clean with?", a possible plan can be "find(coke), pick(coke), find(trash), put(coke) ...". In the Translation stage, an LM translates Q into C, consisting of CNL (which breaks down Q into subtasks) and CSL (which represents the subtasks as a symbolic goal in PDDL[6](#page-4-0) — a language to define and solve classical planning problems). Figure [3](#page-3-0) shows this translation, with CSL in blue and CNL in black. Finally, we call a PDDL Planner as the deterministic solver to obtain A, a plan to accomplish the goal CSL under the predefined scenario.

### <span id="page-4-2"></span>3.4 Relational Inference

Given a relational inference problem Q written in NL, we want to obtain A as a string-valued variable. For example, the CLUTRR [\(Sinha et al.,](#page-11-7) [2019\)](#page-11-7) dataset involves inferring the family relationship (e.g., "grandson") between two people from a short story (e.g., "[Gabrielle] drove her daughter [Dorothy] to the hospital. [Dorothy]'s son [Vincent] showed up shortly after. How is [Vincent] related to [Gabrielle]?", shown in yellow in Figure [3\)](#page-3-0). During the Translation stage, we prompt the LM to generate C, consisting of CNL and CSL. Similar to previous tasks, CNL breaks down Q into subquestions ("How is [Vincent] related to [Dorothy]" and "How is [Dorothy] related to [Gabrielle]"), as well as provide input extracts as rationales to support the answer ("[Dorothy]'s son [Vincent] showed up shortly after", etc.). Each subquestion in CNL is answered in CSL via a relational expression representing the relation between the mentioned entities, for example, relation(Vincent, Dorothy)=son denotes that Vincent is Dorothy's

son. In the Problem Solving stage, our solver is a simple relational inference engine that relies on a set of transitivity rules provided by [Zhang et al.](#page-11-8) [\(2022\)](#page-11-8) among possible family relationships, e.g., son@daughter=grandson (the son of one's daughter is one's grandson). Our solver recursively applies these rules on CSL to derive A, and determine that Vincent is Gabrielle's grandson.

# 4 Experimental setup

# <span id="page-4-3"></span>4.1 Datasets

Here, we summarize the evaluation datasets used for each domain. We select the same number (6 to 10, depending on the task) of exemplars as in [Wei](#page-11-0) [et al.](#page-11-0) [\(2022\)](#page-11-0) to form our few-shot prompt, which can be found in our repository. Unless otherwise stated, we use the official splits: training set for exemplar selection, validation set for prompt tuning, and test set for evaluation.[7](#page-4-1)

Math Word Problems (MWP). We follow [Wei et al.](#page-11-0) [\(2022\)](#page-11-0) and consider the same five MWP benchmarks: GSM8K [\(Cobbe et al.,](#page-10-12) [2021\)](#page-10-12), SVAMP [\(Patel et al.,](#page-10-13) [2021\)](#page-10-13), MultiArith [\(Roy and](#page-11-9) [Roth,](#page-11-9) [2015\)](#page-11-9), ASDiv [\(Miao et al.,](#page-10-14) [2020\)](#page-10-14), and AQuA [\(Ling et al.,](#page-10-15) [2017\)](#page-10-15). For all datasets, the input question is phrased in NL. The answer is a string-valued mathematical expression for AQuA, and one or more integer(s) for all other datasets. We use the same 8-shot prompt for all datasets except AQuA. Multi-hop QA. We consider the three datasets: StrategyQA [\(Geva et al.,](#page-10-16) [2021\)](#page-10-16), a dataset of opendomain questions that require an implicit multistep strategy to answer, e.g., "Did Aristotle use a laptop?" involves answering "1. When did Aristotle live?", "2. When was the laptop invented?", and "3. Is #2 before #1?"; Date Understanding from BIG-bench [\(BIG-Bench collaboration,](#page-9-2) [2021\)](#page-9-2), which asks the model to infer a date from a context, by performing computation on relative periods of time; and finally, Sports Understanding from BIG-bench, which asks the model to decide whether an artificially constructed statement related to sports is plausible or implausible. Since the latter two datasets do not have a training set, we follow [Wei et al.](#page-11-0) [\(2022\)](#page-11-0) and select 10 examples from the test set to form the prompt and use the rest for evaluation.

Planning. We use the SayCan dataset [\(Ahn et al.,](#page-9-3) [2022\)](#page-9-3), which assumes a scenario of a robot operat-

<span id="page-4-0"></span><sup>6</sup> [https://en.wikipedia.org/wiki/Planning\\_](https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language) [Domain\\_Definition\\_Language](https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language). A goal is a special construct in PDDL.

<span id="page-4-1"></span><sup>7</sup> See Appendix [E](#page-16-0) for dataset statistics, examples, data cleaning method, splits, prompt construction strategy, etc.

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 4: A sample output for a math question from three baselines and Faithful CoT (our method). The ground-truth answer is 280, and only our method correctly computes the answer.

ing in a kitchen, helping the user with household tasks, e.g., "bring a coke to the table". There are a number of locations and objects that the robot can interact with. The robot can only perform a fixed set of actions, including find, pick, and put. The task is to map a user query in NL to a plan of predefined actions. Following [Wei et al.](#page-11-0) [\(2022\)](#page-11-0), we manually write 7 exemplars, since no training set is provided.

Relational inference. We use the CLUTRR [\(Sinha et al.,](#page-11-7) [2019\)](#page-11-7) benchmark described in Section [3.4.](#page-4-2) The dataset has multiple splits based on the number of intermediate steps K required to reach the answer. We construct the prompt using 8 exemplars with K ∈ {2, 3}, and test the models on the remaining examples with K up to 10.

# 4.2 Evaluation Metrics

We evaluate the model performance with final answer accuracy as the main metric. Following previous work [\(Wei et al.,](#page-11-0) [2022;](#page-11-0) [Wang et al.,](#page-11-1) [2022;](#page-11-1) [Chen](#page-10-10) [et al.,](#page-10-10) [2022\)](#page-10-10), for all MWP datasets (except AQuA) where the answer contains integer(s), a correct answer is defined as the exact match between the prediction and the ground truth both converted to the nearest integer; for StrategyQA and Sports Understanding where the answer is a Boolean value, it is defined as the exact match between the prediction and the ground truth both evaluated as a Boolean variable; for SayCan, the generated plan is considered correct if it is among the ground truth plans; for all other datasets, we rely on the exact match between the prediction string and the ground truth string. Additionally, we evaluate the human-rated plausibility of the reasoning chain in Appendix [D.](#page-14-1)

## 4.3 Baselines

We compare our method to three other few-shot prompting baselines, shown in Figure [4:](#page-5-0) standard prompting, popularized by [Brown et al.](#page-9-1) [\(2020\)](#page-9-1), with demonstrations of only the question and the answer; CoT [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0), which additionally includes an NL reasoning chain; and Leastto-Most (LtM) [\(Zhou et al.,](#page-11-6) [2022\)](#page-11-6), which decomposes the question in NL but does not involve SL. All prompting methods are compared under two decoding strategies: greedy decoding, where the LM samples the most probable next token from the vocabulary (i.e., temperature = 0.0); and selfconsistency decoding [\(Wang et al.,](#page-11-1) [2022\)](#page-11-1), where the LM generates multiple reasoning chains and chooses the final chain based on majority voting on the evaluated answer (we use a temperature of 0.4 and 40 generations for all datasets).[8](#page-5-1) We reproduce the baseline results ourselves in cases when they are not reported on certain tasks or when we clean the test set.

# 4.4 LMs

We use OpenAI Codex [\(Chen et al.,](#page-9-4) [2021\)](#page-9-4) (code-davinci-002) in Section [5](#page-5-2) and experiment with four other code-generation LMs in Appendix [C.3.](#page-14-2)

# <span id="page-5-2"></span>5 Results

Our results on all datasets are shown in Table [1.](#page-6-0) With code-davinci-002 as the Translator, Faithful CoT outperforms all baselines across the vast

<span id="page-5-1"></span><sup>8</sup>Note that we do not report the performance of standard prompting with self-consistency decoding, since when the number of sampled outputs is large enough, this converges to standard prompting with greedy decoding [\(Wang et al.,](#page-11-1) [2022\)](#page-11-1).

<span id="page-6-0"></span>

|                     | Math Word Problems        |       |            |       | Planning | Multi-hop QA |            |      | Relation |        |
|---------------------|---------------------------|-------|------------|-------|----------|--------------|------------|------|----------|--------|
| Method              | GSM8K                     | SVAMP | MultiArith | ASDiv | AQuA     | SayCan       | StrategyQA | Date | Sport    | CLUTRR |
| Greedy Decoding     |                           |       |            |       |          |              |            |      |          |        |
| Standard            | 19.6                      | 69.5  | 43.8       | 72.1  | 31.5     | 82.5         | 63.9       | 51.3 | 71.9     | 42.0   |
| CoT                 | 63.3                      | 77.3  | 96.5       | 80.0  | 42.1     | 86.4         | 72.5       | 59.9 | 98.6     | 48.5   |
| LtM                 | 38.3                      | 80.3  | 74.0       | 76.5  | 40.6     | 77.7         | 72.2       | 76.6 | 99.5     | 47.2   |
| Faithful CoT (ours) | 72.3                      | 83.4  | 98.8       | 80.2  | 47.2     | 89.3         | 63.0       | 81.6 | 99.1     | 58.9   |
|                     | Self-Consistency Decoding |       |            |       |          |              |            |      |          |        |
| CoT                 | 78.0                      | 86.8  | 100.0      | 84.2  | 52.0     | 89.3         | 79.8       | 63.8 | 98.0     | 45.7   |
| LtM                 | 38.8                      | 80.5  | 74.0       | 76.3  | 44.9     | 76.7         | 71.9       | 77.2 | 99.4     | 50.9   |
| Faithful CoT (ours) | 80.0                      | 88.8  | 99.2       | 84.4  | 61.4     | 94.2         | 65.2       | 85.5 | 99.0     | 71.9   |

Table 1: Accuracy of different prompting methods on 10 reasoning datasets from 4 domains. We compare our method, Faithful CoT, with standard (Brown et al., 2020), CoT (Wei et al., 2022), and Least-to-Most prompting (Zhou et al., 2022), with code-davinci-002 as the LM. The best results within each decoding strategy are **bolded**.

majority of datasets and domains under both decoding strategies. With greedy decoding, Faithful CoT outperforms all baselines on 8 of the 10 datasets, by a relative improvement of up to 14.2% on MWP, 3.4% on Planning, 6.5% on Date Understanding from Multi-hop QA, and a surprising 21.4% on Relational Inference. Generally, we see larger gains on harder datasets. Take MWP as an example: on simpler datasets where CoT already performs decently (e.g., MultiArith and AsDiv, where most questions require only 1-2 steps to solve), the gains are smaller (0.3% to 2.4%); however, we see the largest gain (14%) on the most difficult GSM8K, which requires up to 8 steps to solve. With selfconsistency decoding, Faithful CoT still performs the best on 7 out of the 10 datasets. Compared to greedy decoding, the relative gain increases on 4 datasets (AQUA:  $12.1\% \rightarrow 18.1\%$ , SayCan: 3.4% $\rightarrow$  5.5%, Date Understanding: 6.5%  $\rightarrow$  10.8%, and CLUTRR:  $21.4\% \rightarrow 41.3\%$ ), but decreases or remains unchanged for the remaining three MWP datasets (GSM8K:  $9.0\% \rightarrow 2.6\%$ , SVAMP: 3.9% $\rightarrow$  2.3%, ASDiV 0.2%  $\rightarrow$  0.2%).

On the other hand, we do not see clear empirical gains on two multi-hop QA datasets, Sports Understanding on StrategyQA. On Sports Understanding, Faithful CoT and LtM both have near perfect accuracy (99+), suggesting that the dataset is almost saturated. On StrategyQA, however, the performance of our method is still far from the baselines. To understand why, we specifically compare the examples where CoT makes a correct prediction but our method fails. As shown in Figure 11 in Appendix F, we find that the likely primary cause is the sparsity of Datalog in the pretraining data for Codex, as an overwhelming 29% of errors are syntax-related. Moreover, including Datalog in the prompt also interferes with NL generation, making

it harder for Codex to produce relevant subquestions (17%), retrieve knowledge correctly (10%), and come up with valid reasoning from the knowledge to the answer (10%). Another potential cause is the nature of the task, as the difficulty for many StrategyQA questions does not lie in reasoning but rather in knowledge retrieval, which makes the advantages of our deterministic solver less obvious. Still, with further pretraining on Datalog, we believe that there is room for improvement.

To see how generalizable our method is, we also experiment with four alternative LMs and observe consistent gains brought by Faithful CoT over the baselines, as shown in Appendix C.3. In particular, with GPT-4, we set the new few-shot SOTA results on 7 datasets, achieving 95.0+ accuracy in four out of five MWP and two out of three Multi-hop QA datasets. Overall, these results suggest that faithfulness does empirically improve performance.

#### <span id="page-6-2"></span>6 Analysis

In this section, we perform an extensive analysis of the strengths and weaknesses of our method, to better understand the role of different components, the robustness to design choices, the plausibility of generated reasoning chains, as well as frequent error patterns where it still struggles. Here, we only show the first two aspects; see the rest in Appendix C. Unless otherwise stated, we choose one dataset from each domain (GSM8K, Date Understanding, Say-Can, and CLUTRR) and use code-davinci-002 outputs with greedy decoding.

#### <span id="page-6-1"></span>6.1 Ablation Study

Given the strong performance of Faithful CoT, we now address a natural question: **how much does each part of the prompt contribute to the accuracy?** We perform an ablation study where we re-

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 5: Ablation study results: accuracy when we remove different parts of the prompt. See Section 6.1 for details.

move different parts of the prompt and see how the performance changes. In addition to the original prompt ("Full"), we test four variations, illustrated with the example from Figure 4:

**No rationale.** We remove the rationales, i.e., everything in the brackets from the NL comments, e.g., "independent, support: ['There are 15 trees']".

**No NL but nudge.** We remove all NL comments except the "nudge" line: e.g., "# To answer this question, we write a Python program to answer the following subquestions".

No NL. We remove all NL comments.

**No solver.** Instead of calling the external solver, we add "Answer: {answer}" to the end of every exemplar and let the LM predict the answer itself.

Figure 5 shows the results of all prompt variations. On GSM8K, Date Understanding, and Say-Can, NL comments contribute little to the performance, and sometimes even slightly hurt it. On CLUTRR, however, their role is crucial, since the exclusion of each component (rationale, nudge, subquestions) results in a clear accuracy drop. In particular, comparing **No NL but nudge** and **No NL**, the nudge line itself brings a striking improvement by 31.3 points.

The external solver relieves the burden of problem solving from the LM. Without it, the accuracy suffers a huge decline on GSM8K, Date Understanding, and CLUTRR (-50.8, -22.9, and -19.4 respectively), while on SayCan it improves by 2.9 nonetheless. One potential influencing factor is that SayCan might be too homogeneous, as it contains a set of only 3 predefined actions. This can make the task relatively easy, which allows all model variants to achieve around 90% accuracy and renders the solver unnecessary. Another potential reason is the level of correspondence between the final answer and the reasoning chain for different datasets: as shown in Figure 3, the answer in SavCan is a sequence of actions (e.g., find(redbull)), each directly corresponding to one step in the reasoning

<span id="page-7-1"></span>

| Exemplars       | GSM8K | Date | SayCan | CLUTRR |
|-----------------|-------|------|--------|--------|
| Set 0 (Table 1) | 72.3  | 81.6 | 89.3   | 58.9   |
| Set 1           | 72.6  | 81.3 | 91.3   | 59.0   |
| Set 2           | 71.1  | 85.0 | 85.4   | 57.2   |
| Set 3           | 72.3  | 82.5 | 88.3   | 58.0   |
| Set 4           | 71.2  | 77.4 | 88.3   | 55.5   |
| Set 5           | 71.5  | 85.0 | 89.3   | 56.0   |
| Mean            | 71.8  | 82.1 | 88.7   | 57.4   |
| Std             | 0.6   | 2.8  | 1.9    | 1.5    |
|                 |       |      |        |        |

Table 2: Robustness to the choice of exemplars.

chain (e.g., at redbull trash). However, the answer in the other three datasets is only a single number or string, which can only be derived after executing *all* the steps in the reasoning chain. Therefore, the latter type of tasks further necessitates the presence of an external solver.

# 6.2 Robustness to Exemplars

We now answer the next question: **how much does the choice of exemplars matter**? To do this, we annotate 20 examples in total, randomly sample k (7-10, depending on the dataset) to construct the prompt, and repeat the process five times. Table 2 shows the performance of all six runs, including the original (from Table 1). The mean accuracy is close to the original (-1.5 to +1.2), still above the baselines by a large margin (7 to 17) on all datasets except the arguably easiest SayCan, considering the standard deviation (1.3 to 2.9). This strongly suggests that the benefits of Faithful CoT are minimally influenced by the choice of exemplars.

#### 6.3 Human Evaluation of Plausibility

Our main experiments use final answer accuracy as the performance measure, but this does not necessarily correspond to the validity of the reasoning chain. Technically, a model can sometimes accidentally arrive at the correct answer with an invalid reasoning chain. We then ask: when the answer is correct, how often is the reasoning chain truly correct? In other words, we want to evaluate the plausibility of the reasoning chains.

We conduct a human evaluation study on Prolific: given a generated reasoning chain that results in a correct answer, a crowd-worker selects whether it is *A*) *completely correct*, or, if incorrect, specify why with *B*) *incorrect NL* and/or *C*) *incorrect SL*. Alternatively, they can select *D*) *flawed question* and *E*) *I am confused*.

The results of our study are shown in Figure 6 (see Table 8 in the Appendix for numerical results).

<span id="page-7-2"></span><sup>&</sup>lt;sup>9</sup>See Appendix D for more details on the human study.

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 6: Human evaluation results of reasoning chain plausibility. Each column represents the percent of different answer choices selected by human evaluators in each domain/dataset.

For most domains, we see that annotators often find the reasoning chain fully correct – Sports, Say-Can, and MWP have a 90%+ correctness rate. To gain more insight into when the reasoning chain can be "incorrect", we perform an in-depth analysis of user annotations for the three worst-performing datasets – StrategyQA (66.7% correctness), Date (87.9% correctness), and CLUTRR (88.0% correctness). From our inspection, we find that annotators mistakenly mark a correct reasoning chain as incorrect at different rates based on the task (8.3% of the time for StrategyQA, 41.7% for Date Understanding, and 100% for CLUTRR). We find annotators are inaccurate for Date because they incorrectly believe the generated code misuses a Python library (relativedelta), or they complain that there is a better way to answer the question. For CLUTRR, annotators mark chains as incorrect due to known ambiguity in the dataset. For example, the grandmother of one's child may not necessarily be their parent, but also a parent-in-law.

As for the remaining truly incorrect reasoning chains, we find the LM can sometimes add unnecessary steps or arrive at the correct answer by chance. The latter is especially an issue in StrategyQA - given that all questions have a True/False answer, it is common for an incorrect reasoning chain to result in a correct answer. For example, the LM correctly answers the question "Was Karachi a part of Alexander the Great's success?" as "True." However, the reasoning chain contains the flawed subquestion "Which countries are in Pakistan? Pakistan includes Pakistan, Afghanistan, and India." Though the final answer is correct and faithful to the LM generation, the reasoning chain contains wrong knowledge.

Overall, Faithful CoT does generate valid reasoning chains for the vast majority of the time when the answer is correct. However, we still see exceptions where the model arrives at the right answer via an incorrect reasoning chain. Though this happens infrequently, it raises concerns about when people should trust LMs. To our knowledge, we are the first to conduct a systematic human study on the plausibility of CoT-style reasoning chains, and we hope to see future work further investigate and improve on the flaws that our study brings to light.

# 7 Conclusion

We propose Faithful CoT, a framework that decomposes complex reasoning into Translation and Problem Solving. It guarantees that the reasoning chain is a faithful explanation of how the model arrives at the answer. We demonstrate the efficacy of our approach on 4 types of complex reasoning problems: Math Word Problems, Multi-hop QA, Planning, and Relational Inference. Our method sets new SOTA performance on 7 of the 10 datasets, while additionally providing a faithful explanation for the final answer. These results give empirical evidence that improving model *interpretability*, by guaranteeing the faithfulness of an explanation, does not come at the expense of overall *performance*; in fact, we see a strong synergy in between. Through a comprehensive analysis of the strengths and weaknesses of our method, we show its robustness to the choice of exemplars, the pivotal role of the solver, as well as frequent error patterns where it still struggles.

# Limitations

One crucial limitation of our study is that on March 23rd, 2023, OpenAI discontinued the use of codedavinci-002. This has rendered part of our results unreplicable for any teams or researchers who have not been granted continued access to the model. This discontinuation was unexpected during our study. It raises important questions about using closed-source models for academic research.

Meanwhile, one methodological limitation of our approach lies in the scope of faithfulness. Currently, we guarantee that Problem Solving stage is faithful. However, the Translation stage is still opaque, meaning it is not self-interpretable how the LM generates the reasoning chain from the question. It is still an under-explored question whether

it is possible to improve the interpretability of the LM generation process in general, and a few recent studies have made promising early progress [\(Yin](#page-11-10) [and Neubig,](#page-11-10) [2022;](#page-11-10) [Sarti et al.,](#page-11-11) [2023\)](#page-11-11) that might be used to improve the faithfulness of the Translation stage.

Finally, it still needs further exploration of the role NL comments in the reasoning chain. From our ablation study, in terms of *performance*, whether to include the NL comments in the reasoning chain does not make a big difference on many of the datasets, especially those where the task is not knowledge-intensive. Nevertheless, speaking of *interpretability*, NL comments can make the reasoning chain more structured and understandable to the end user. Further, NL comments can be an interface that allows users *without* a programming background to interact with and debug the model, which we leave for future work.

## Ethics Statement

With the recent success of generative large LMs, they are now being used to solve complex reasoning problems. When using the output of an LM for reasoning, there is a danger that if the reasoning *appears* realistic, then the final answer or conclusion will also be considered reliable. As we highlighted in Figure [1](#page-0-1) and [7,](#page-12-1) this is often not true, since an LM may produce a reasoning chain that looks plausible, but the final answer is still wrong. This work is a step in the direction of making the use of LMs more trustworthy by using the LM for just expressing its reasoning in a symbolic program and executing the program independently. In this work, we have ensured the faithfulness of the reasoning chain w.r.t how the final answer is produced in a variety of domains, but admittedly the Translation phase is still opaque. Therefore, our pipeline is still not entirely interpretable. Furthermore, as we have stressed in Section [1,](#page-0-2) faithfulness does not guarantee correctness, so our method can still sometimes produce erroneous answers, which may pose a risk for users that rely on it for decision making.

## Acknowledgements

This research is based upon work supported in part by the DARPA KAIROS Program (contract FA8750-19-2-1004), the DARPA LwLL Program (contract FA8750-19-2-0201), the IARPA HIATUS Program (contract 2022-22072200005), and the NSF (Award 1928631). Approved for Public Release, Distribution Unlimited. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, IARPA, NSF, or the U.S. Government.

We appreciate the support from OpenAI on increasing the rate limit for the Codex API. We also thank Jiani Huang, Ziyang Li, Litao Yan, Andrew Head, Mayur Naik, and Lyle Ungar for their valuable feedback.

# References

<span id="page-9-3"></span>Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J. Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. 2022. [Do As I](https://doi.org/10.48550/arXiv.2204.01691) [Can, Not As I Say: Grounding Language in Robotic](https://doi.org/10.48550/arXiv.2204.01691) [Affordances.](https://doi.org/10.48550/arXiv.2204.01691) ArXiv:2204.01691 [cs].

<span id="page-9-0"></span>Yoshua Bengio. 2019. FROM SYSTEM 1 DEEP LEARNING TO SYSTEM 2 DEEP LEARNING.

<span id="page-9-2"></span>BIG-Bench collaboration. 2021. [Beyond the Imitation](https://github.com/google/BIG-bench/) [Game: Measuring and extrapolating the capabilities](https://github.com/google/BIG-bench/) [of language models.](https://github.com/google/BIG-bench/)

<span id="page-9-1"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. [Language Models are Few-Shot Learners.](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc.

<span id="page-9-4"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter,

- Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. [Eval](http://arxiv.org/abs/2107.03374)[uating Large Language Models Trained on Code.](http://arxiv.org/abs/2107.03374) ArXiv:2107.03374 [cs].
- <span id="page-10-10"></span>Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2022. [Program of Thoughts](https://doi.org/10.48550/arXiv.2211.12588) [Prompting: Disentangling Computation from](https://doi.org/10.48550/arXiv.2211.12588) [Reasoning for Numerical Reasoning Tasks.](https://doi.org/10.48550/arXiv.2211.12588) ArXiv:2211.12588 [cs].
- <span id="page-10-12"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. [Training Verifiers to Solve Math Word Prob](https://doi.org/10.48550/arXiv.2110.14168)[lems.](https://doi.org/10.48550/arXiv.2110.14168) ArXiv:2110.14168 [cs].
- <span id="page-10-11"></span>Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2022. [PAL: Program-aided Language](https://doi.org/10.48550/arXiv.2211.10435) [Models.](https://doi.org/10.48550/arXiv.2211.10435) ArXiv:2211.10435 [cs].
- <span id="page-10-16"></span>Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. *[Did Aristotle](https://doi.org/10.1162/tacl_a_00370) Use a Laptop?* [A Question Answering Benchmark](https://doi.org/10.1162/tacl_a_00370) [with Implicit Reasoning Strategies.](https://doi.org/10.1162/tacl_a_00370) *Transactions of the Association for Computational Linguistics*, 9:346– 361.
- <span id="page-10-3"></span>Leilani H. Gilpin, David Bau, Ben Z. Yuan, Ayesha Bajwa, Michael Specter, and Lalana Kagal. 2018. [Ex](https://doi.org/10.1109/DSAA.2018.00018)[plaining Explanations: An Overview of Interpretabil](https://doi.org/10.1109/DSAA.2018.00018)[ity of Machine Learning.](https://doi.org/10.1109/DSAA.2018.00018) In *2018 IEEE 5th International Conference on Data Science and Advanced Analytics (DSAA)*, pages 80–89.
- <span id="page-10-2"></span>L. A. Harrington, M. D. Morley, A. Šcedrov, and S. G. Simpson. 1985. *Harvey Friedman's Research on the Foundations of Mathematics*. Elsevier. Google-Books-ID: 2plPRR4LDxIC.
- <span id="page-10-4"></span>Bernease Herman. 2019. [The Promise and Peril](http://arxiv.org/abs/1711.07414) [of Human Evaluation for Model Interpretability.](http://arxiv.org/abs/1711.07414) *arXiv:1711.07414 [cs, stat]*. ArXiv: 1711.07414.
- <span id="page-10-0"></span>Alon Jacovi and Yoav Goldberg. 2020. [Towards faith](https://doi.org/10.18653/v1/2020.acl-main.386)[fully interpretable NLP systems: How should we](https://doi.org/10.18653/v1/2020.acl-main.386) [define and evaluate faithfulness?](https://doi.org/10.18653/v1/2020.acl-main.386) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 4198–4205. Association for Computational Linguistics.
- <span id="page-10-9"></span>Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, and Yejin Choi. 2022. [Maieutic Prompting: Logically](http://arxiv.org/abs/2205.11822)

- [Consistent Reasoning with Recursive Explanations.](http://arxiv.org/abs/2205.11822) ArXiv:2205.11822 [cs].
- <span id="page-10-6"></span>Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. [Large Lan](https://openreview.net/forum?id=e2TBb5y0yFf)[guage Models are Zero-Shot Reasoners.](https://openreview.net/forum?id=e2TBb5y0yFf)
- <span id="page-10-7"></span>Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. [Solving Quanti](http://arxiv.org/abs/2206.14858)[tative Reasoning Problems with Language Models.](http://arxiv.org/abs/2206.14858) ArXiv:2206.14858 [cs].
- <span id="page-10-8"></span>Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2022. [On the Ad](https://doi.org/10.48550/arXiv.2206.02336)[vance of Making Language Models Better Reasoners.](https://doi.org/10.48550/arXiv.2206.02336) ArXiv:2206.02336 [cs].
- <span id="page-10-15"></span>Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. [Program induction by rationale](https://doi.org/10.18653/v1/P17-1015) [generation: Learning to solve and explain alge](https://doi.org/10.18653/v1/P17-1015)[braic word problems.](https://doi.org/10.18653/v1/P17-1015) In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 158–167. Association for Computational Linguistics. ArXiv:1705.04146 [cs].
- <span id="page-10-14"></span>Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. [A diverse corpus for evaluating and developing](https://doi.org/10.18653/v1/2020.acl-main.92) [English math word problem solvers.](https://doi.org/10.18653/v1/2020.acl-main.92) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 975–984, Online. Association for Computational Linguistics.
- <span id="page-10-5"></span>Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2021. [Show Your Work: Scratchpads for](https://doi.org/10.48550/arXiv.2112.00114) [Intermediate Computation with Language Models.](https://doi.org/10.48550/arXiv.2112.00114) ArXiv:2112.00114 [cs].
- <span id="page-10-17"></span>OpenAI. 2023. [Gpt-4 technical report.](http://arxiv.org/abs/2303.08774)
- <span id="page-10-13"></span>Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. [Are NLP models really able to solve simple](https://doi.org/10.18653/v1/2021.naacl-main.168) [math word problems?](https://doi.org/10.18653/v1/2021.naacl-main.168) In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2080–2094, Online. Association for Computational Linguistics.
- <span id="page-10-18"></span>Silviu Pitis, Michael R Zhang, Andrew Wang, and Jimmy Ba. 2023. Boosted prompt ensembles for large language models. *arXiv preprint arXiv:2304.05970*.
- <span id="page-10-1"></span>Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, and Zachary C. Lipton. 2020. [Learning to](https://doi.org/10.18653/v1/2020.acl-main.432) [deceive with attention-based explanations.](https://doi.org/10.18653/v1/2020.acl-main.432) In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 4782– 4793, Online. Association for Computational Linguistics.

- <span id="page-11-5"></span>Jing Qian, Hong Wang, Zekun Li, Shiyang Li, and Xifeng Yan. 2022. [Limitations of Language](http://arxiv.org/abs/2208.05051) [Models in Arithmetic and Symbolic Induction.](http://arxiv.org/abs/2208.05051) ArXiv:2208.05051 [cs].
- <span id="page-11-4"></span>Marco Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. ["why should I trust you?": Explaining the pre](https://doi.org/10.18653/v1/N16-3020)[dictions of any classifier.](https://doi.org/10.18653/v1/N16-3020) In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations*, KDD '16, pages 97–101, New York, NY, USA. Association for Computational Linguistics.
- <span id="page-11-9"></span>Subhro Roy and Dan Roth. 2015. [Solving general arith](https://doi.org/10.18653/v1/D15-1202)[metic word problems.](https://doi.org/10.18653/v1/D15-1202) In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 1743–1752, Lisbon, Portugal. Association for Computational Linguistics.
- <span id="page-11-11"></span>Gabriele Sarti, Nils Feldhus, Ludwig Sickert, and Oskar van der Wal. 2023. Inseq: An interpretability toolkit for sequence generation models. *arXiv preprint arXiv:2302.13942*.
- <span id="page-11-7"></span>Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. 2019. [CLUTRR:](https://doi.org/10.18653/v1/D19-1458) [A diagnostic benchmark for inductive reasoning from](https://doi.org/10.18653/v1/D19-1458) [text.](https://doi.org/10.18653/v1/D19-1458) In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 4506–4515, Hong Kong, China. Association for Computational Linguistics.
- <span id="page-11-3"></span>Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh, and Himabindu Lakkaraju. 2020. [Fooling LIME and](https://doi.org/10.1145/3375627.3375830) [SHAP: Adversarial Attacks on Post hoc Explanation](https://doi.org/10.1145/3375627.3375830) [Methods.](https://doi.org/10.1145/3375627.3375830) In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, pages 180–186. Association for Computing Machinery, New York, NY, USA.
- <span id="page-11-2"></span>Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. [Language Models Don't](https://arxiv.org/abs/2305.04388v1) [Always Say What They Think: Unfaithful Explana](https://arxiv.org/abs/2305.04388v1)[tions in Chain-of-Thought Prompting.](https://arxiv.org/abs/2305.04388v1)
- <span id="page-11-1"></span>Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. [Self-Consistency Improves](https://doi.org/10.48550/arXiv.2203.11171) [Chain of Thought Reasoning in Language Models.](https://doi.org/10.48550/arXiv.2203.11171) ArXiv:2203.11171 [cs].
- <span id="page-11-0"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. [Chain of Thought Prompting](https://openreview.net/forum?id=_VjQlMeSB_J) [Elicits Reasoning in Large Language Models.](https://openreview.net/forum?id=_VjQlMeSB_J)
- <span id="page-11-10"></span>Kayo Yin and Graham Neubig. 2022. Interpreting language models with contrastive explanations. *arXiv preprint arXiv:2202.10419*.
- <span id="page-11-8"></span>Hanlin Zhang, Ziyang Li, Jiani Huang, Mayur Naik, and Eric Xing. 2022. [Improved logical reasoning](https://openreview.net/forum?id=8lNy3QCaxHX) [of language models via differentiable symbolic pro](https://openreview.net/forum?id=8lNy3QCaxHX)[gramming.](https://openreview.net/forum?id=8lNy3QCaxHX) In *First Workshop on Pre-training: Perspectives, Pitfalls, and Paths Forward at ICML 2022*.

<span id="page-11-6"></span>Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. 2022. [Least](http://arxiv.org/abs/2205.10625)[to-Most Prompting Enables Complex Reasoning in](http://arxiv.org/abs/2205.10625) [Large Language Models.](http://arxiv.org/abs/2205.10625) ArXiv:2205.10625 [cs].

# A Implementation Details

In all our experiments, we use OpenAI GPT-3 (text-davinci-001 and text-davinci-002) and Codex (code-davinci-001 and code-davinci-002) models through the Python API available at <beta.openai.com>, from Sept, 2022 to Jan, 2023. The inference cost per example is \$0 for all Codex models since they are in limited beta period, and \$0.01 - \$0.03 for GPT-3 models depending on the dataset. It takes 2-15 seconds to run inference on one example with Codex models under a rate limit of 150,000 tokens/minute, and 1-8 seconds with GPT-3 models under 250,000 tokens/minute, also depending on the dataset. For example, on the GSM8K test set of 1,319 examples, it takes 3.5h to finish the inference with Codex and 2.3h with GPT-3.

We use the following hyper-parameters throughout all experiments:

temperature: 0.0 for greedy decoding, 0.4 for self-consistent decoding;

max\_tokens: 1000;

n: 1 for greedy decoding, 40 for self-consistent decoding;

frequency\_penalty: 0; presence\_penalty: 0.

Any unspecified hyper-parameters are set to the default value on [https://beta.openai.com/](https://beta.openai.com/docs/api-reference/completions/create) [docs/api-reference/completions/create](https://beta.openai.com/docs/api-reference/completions/create).

# B Extended Results

In this section, we present more results that do not fit into the main text.

# <span id="page-12-0"></span>B.1 Examples of Unfaithful CoT Output

Figure [7](#page-12-1) shows examples of unfaithful output from the CoT method [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0) on three datasets: GSM8K (MWP), StrategyQA (Multi-hop QA), and SayCan (Planning). Note that here we only show model outputs that are apparently unfaithful. In other cases, there is no *known* inconsistency between the reasoning chain and the answer, but this still does not imply that the reasoning chain is a faithful explanation, since the ground truth (how the model predicts the answer) is unknown.

The GSM8K example is explained in Section [1.](#page-0-2) In the StrategyQA example, though the reasoning chain correctly identifies that a hummingbird weighs much more than a pea, the answer is still "yes"; in the SayCan example, the reasoning chain only mentions "visit the table and the counter",

<span id="page-12-1"></span>![](_page_12_Figure_13.jpeg)

Figure 7: Examples of *unfaithful* output from CoT prompting [\(Wei et al.,](#page-11-0) [2022\)](#page-11-0) on three datasets. The answer (green) does not follow from the reasoning chain (blue).

but the plan contains unnecessary "pick" and "put" operations.

[Wei et al.](#page-11-0) [\(2022\)](#page-11-0) claim that CoT "provides an interpretable window into the behavior of the model, suggesting how it might have arrived at a particular answer and providing opportunities to debug where the reasoning path went wrong". As we have pointed out in Section [1,](#page-0-2) since CoT does not guarantee faithfulness, how the model arrives at the answer could differ drastically from what is shown in the reasoning chain. Furthermore, it is still hard for the user to debug the model: even if they manually correct the reasoning chain and let the model regenerate the answer, it might still be wrong, since there is no causality between the reasoning chain and the answer.

# <span id="page-12-2"></span>B.2 Comparison with Few-shot SOTA

We compare the results of Faithful CoT with the published few-shot SOTA in Table [3.](#page-13-2) The Faithful CoT results are obtained with the best-performing LM among all five LMs we experiment with (see Section [C.3](#page-14-2) on each dataset. The SOTA results are obtained from the following sources:

GSM8K: [OpenAI](#page-10-17) [\(2023\)](#page-10-17);

MultiArith, ASDiv, StrategyQA: [Wang et al.](#page-11-1) [\(2022\)](#page-11-1);

SVAMP: [Chen et al.](#page-10-10) [\(2022\)](#page-10-10); AQuA: [Pitis et al.](#page-10-18) [\(2023\)](#page-10-18);

<span id="page-13-2"></span>

|                     | Math Word Problems |       |            |       | Planning | Multi-hop QA |            |       | Relation |        |
|---------------------|--------------------|-------|------------|-------|----------|--------------|------------|-------|----------|--------|
| Method              | GSM8K              | SVAMP | MultiArith | ASDiv | AQuA     | SayCan       | StrategyQA | Date  | Sport    | CLUTRR |
| Few-shot SOTA       | 92.0               | 89.1  | 100.0      | 87.8  | 76.4     | 88.3         | 81.6       | 76.2  | 98.5     | 50.9   |
| Faithful CoT (ours) | 95.0               | 95.4  | 99.2       | 95.6  | 73.6     | 93.2         | 65.2       | 95.8  | 99.3     | 71.9   |
| $\delta_{acc}$      | +3.0               | +6.3  | -0.8       | +7.8  | -2.8     | +4.9         | -16.4      | +19.6 | +0.8     | +21.0  |

Table 3: Comparison between the existing few-shot SOTA results and the optimal Faithful CoT results (with the best-performing LM (code-davinci-002 for SayCan and CLUTRR, and gpt-4 for the rest of the datasets). See Appendix B.2 for sources of SOTA results.

<span id="page-13-3"></span>![](_page_13_Figure_2.jpeg)

Figure 8: Accuracy of our method and two concurrent methods, Program of Thoughts (POT) (Chen et al., 2022) and Program-Aided Language Models (PAL) (Gao et al., 2022), on 10 reasoning datasets.

**SayCan, Sports Understanding**: (Wei et al., 2022);

Date Understanding: (Gao et al., 2022);

**CLUTRR**: Our implementation of LtM prompting (Zhou et al., 2022) (no existing work reports few-shot performance on CLUTRR with K up to 10).

With Codex and GPT-4, Faithful CoT sets new SOTA performance on 7 out of the 10 datasets across four domains, achieving 95.0+ accuracy on 6 of them.

# <span id="page-13-0"></span>**B.3** Empirical Comparison with Concurrent Work

Two pieces of concurrent work, Program of Thoughts (PoT) (Chen et al., 2022) and Program-Aided Language Models (PAL) (Gao et al., 2022), were announced on arXiv within three months of our work. Essentially, they both generate Python programs, or SL-only reasoning chains, to derive the answer. Our approach differs from them mainly in the additional component of structured NL comments, which decomposes the original problem into simpler, inter-dependent subproblems.

Aside from the qualitative differences highlighted in Section 2, we perform an empirical performance comparison with them on the same set of 10 datasets used in our main evaluation. Since both papers have only tackled math reasoning and symbolic reasoning tasks, we reimplement their methods by using the "noNL" prompt in our ablation study from Section 6.1. The comparison is done with code-davinci-002 as the underlying LM and greedy decoding.

As shown in Figure 8, on 6 of the 10 datasets (including most MWP datasets, SayCan, and Date Understanding), PAL/PoT and Faithful CoT have very close accuracy (<2.0 difference). On AQuA, PAL/PoT is visibly better. On the remaining three datasets (StrategyQA, Sports Understanding, and CLUTRR), Faithful CoT reasonably outperforms PAL/PoT. This may suggest that our method has an advantage when the task requires extensive external knowledge (e.g., StrategyQA and Sports Understanding) or when the SL is not frequent in the LM's pretraining data (e.g., Datalog, or our self-defined relational expressions).

Finally, note that the key contribution of our method lies in interpretability. Though the addition of structured NL comments sometimes does not make a difference in performance, it does make the reasoning chain more understandable to the user. Furthermore, it may even enable users without a programming background to debug the model, by only interacting with the NL subproblems (e.g., adding/removing/editing a subproblem), which is worth further exploration in the future.

## <span id="page-13-1"></span>C Extended Analysis

#### C.1 Ablation Study

Table 4 shows the full results of the ablation study from Section 6.1.

#### **C.2** Robustness to Prompt Phrasing

We study the sensitivity of our method to subtle differences in the prompt design. We experiment with three prompt variations: 1. randomly permuting the order of independent subquestions/reasoning steps; 2: Changing the variable names; 3. changing the nudge line (e.g. from "# To answer this

<span id="page-14-3"></span>

| Prompt          | GSM8K | Date | SayCan | CLUTRR |
|-----------------|-------|------|--------|--------|
| Full            | 72.3  | 81.6 | 89.3   | 58.9   |
| No rationale    | 75.4  | 83.0 | -      | 51.8   |
| No NL but nudge | 73.5  | 80.2 | -      | 39.6   |
| No NL           | 72.8  | 79.7 | 90.3   | 9.3    |
| No solver       | 21.5  | 57.9 | 90.3   | 40.9   |

Table 4: Ablation study results that accompany Figure 5. We report accuracy when we remove different parts of the prompt.

<span id="page-14-4"></span>

| Prompt      | GSM8K | Date | SayCan | CLUTRR |
|-------------|-------|------|--------|--------|
| Original    | 72.3  | 81.6 | 89.3   | 58.9   |
| Variation 1 | 69.1  | 84.4 | 88.3   | -      |
| Variation 2 | 70.3  | 81.6 | 90.3   | 56.2   |
| Variation 3 | 70.2  | 80.5 | 87.4   | 55.9   |
| Mean        | 70.5  | 82.0 | 88.8   | 57.0   |
| Std         | 1.3   | 1.7  | 1.3    | 1.7    |

Table 5: Robustness to prompt phrasing.

question, write a Python program to answer the following subquestions" to "# To solve this question, we answer each of the following subquestions with a Python program").

We rerun the evaluation of all three variations on 4 datasets (when applicable) used in the Section 6, under greedy decoding. Table 5 shows the results. Overall, the performance is quite stable, always above each baseline on all four datasets.

# <span id="page-14-2"></span>C.3 Model Sensitivity

In this section, we want to answer the question: how much does the choice of LM matter? All results in Section 5 are obtained using code-davinci-002. Here, we examine the effect of using four alternative code-generation models as the Translator: code-davinci-001, text-davinci-002, text-davinci-003, gpt-4. We compare our method with the three baselines using each of the above LM on five MWP datasets, using the greedy decoding strategy.

As shown in Table 6, regardless of the underlying LM, Faithful CoT consistently outperforms all baselines on the vast majority of the datasets, and performs very closely with the best method (<2.0 difference) on the remaining ones. On average, it has a relative accuracy gain of 16.1%, 11.0%, 9.4%, and 4.6% over the best-performing method among the baselines, for each LM respectively. This indicates that even though the absolute performance varies depending on the LM, Faithful CoT brings a relatively consistent accuracy gain.

Notably, with GPT-4 as the underlying LM, Faithful CoT results in 95.0+ accuracy in 4 of the 5 MWP datasets, far outperforming the previous fewshot SOTA on three of them (GSM8K, SVAMP, and ASDiv).

# <span id="page-14-0"></span>**C.4** Enforcing Constraints

Since our generated reasoning chain contains structured components (e.g., dependency graphs), another natural question to ask is: will it be helpful to enforce certain constraints on the generation? Using MWP datasets as a case study, we examine the effect of three such constraints:

**Graph validity.** The dependency graph must be a Directed Cyclic Graph (DAG), e.g., it is not allowed for a subquestion to depend on itself.

**No over-dependency.** The code cannot depend on *any* variable that its corresponding subquestion has not mentioned, e.g. in Figure 4, since Q5 says "depend on 4", then the corresponding code should *not* use the variable eggs\_in\_dozen, since it is not the output of Q4.

**No under-dependency.** The code must depend on *all* variables that its corresponding subquestion has mentioned, e.g. in the same example, since Q5 says "depend on 4", then the corresponding code *must* use the variable eggs\_in\_dozen.

We investigate the effect of adding constraints on the generations under self-consistent decoding. Starting with our original results (without any constraint), we add a different set of constraints at each time and report the accuracy change in Table 7. Individually, the graph validity constraint results in little to no change in the performance, but the other two constraints lead to a more unstable changemostly a decrease-across datasets. Adding two or more constraints further lowers the performance in almost all cases except on MultiArith (the easiest dataset), revealing the tradeoff between accuracy and satisfying the constraints. It also indicates that a proportion of generations (1.0% to 8.9%) in our existing results do not satisfy all constraints. However, it may still be worth enforcing some of these constraints (e.g., graph validity) at the cost of performance, in order for users to better control and interact with the model.

#### <span id="page-14-1"></span>**D** Human Evaluation Details

We hire crowd workers on Prolific to evaluate the correctness of model-generated reasoning chains that result in a correct answer. We sample

<span id="page-15-0"></span>

| Method / Dataset    | GSM8K | SVAMP                | MultiArith  | ASDiv | AQuA | Average |  |
|---------------------|-------|----------------------|-------------|-------|------|---------|--|
|                     |       | LM: code-davinci-001 |             |       |      |         |  |
| Standard            | 5.2   | 28.7                 | 8.6         | 38.5  | 22.8 | 20.8    |  |
| CoT                 | 14.7  | 41.2                 | 57.2        | 50.4  | 22.4 | 37.2    |  |
| LtM                 | 9.8   | 44.9                 | 18.5        | 46.6  | 20.5 | 28.1    |  |
| Faithful CoT (ours) | 27.4  | 50.8                 | 63.3        | 53.7  | 20.9 | 43.2    |  |
|                     |       | LM: te               | xt-davinci- | 002   |      |         |  |
| Standard            | 15.4  | 65.2                 | 34.1        | 64.8  | 24.0 | 40.7    |  |
| CoT                 | 47.2  | 68.0                 | 91.1        | 69.9  | 40.6 | 63.4    |  |
| LtM                 | 32.9  | 73.8                 | 68.3        | 70.9  | 34.6 | 56.1    |  |
| Faithful CoT (ours) | 62.7  | 80.0                 | 92.8        | 75.4  | 41.3 | 70.4    |  |
|                     |       | LM: te               | xt-davinci- | 003   |      | •       |  |
| Standard            | 16.9  | 69.4                 | 38.8        | 59.1  | 29.5 | 42.7    |  |
| CoT                 | 59.6  | 79.5                 | 95.0        | 69.1  | 46.9 | 70.0    |  |
| LtM                 | 34.6  | 79.5                 | 73.7        | 70.0  | 44.5 | 60.5    |  |
| Faithful CoT (ours) | 71.7  | 85.1                 | 94.5        | 80.7  | 50.8 | 76.6    |  |
|                     |       | ]                    |             |       |      |         |  |
| Standard            | 46.9  | 88.4                 | 98.7        | 70.2  | 50.4 | 70.9    |  |
| CoT                 | 64.9  | 80.0                 | 94.0        | 71.6  | 75.2 | 77.1    |  |
| LtM                 | 91.8  | 92.9                 | 98.3        | 86.7  | 72.0 | 87.5    |  |
| Faithful CoT (ours) | 95.0  | 95.4                 | 98.5        | 95.6  | 73.6 | 91.6    |  |

Table 6: Accuracy of different prompting methods with each underlying LM on 5 MWP reasoning datasets.

<span id="page-15-1"></span>

| Constraint | GSM8K | SVAMP | MultiArith | ASDiv | AQuA |
|------------|-------|-------|------------|-------|------|
| None       | 80.0  | 88.8  | 99.2       | 84.4  | 61.4 |
| + G        | 0.0   | 0.0   | 0.0        | -0.1  | -0.8 |
| + O        | -0.9  | -0.1  | -0.1       | +0.4  | -3.9 |
| + U        | -1.0  | -3.6  | 0.0        | -1.2  | +1.2 |
| + GO       | -1.7  | -0.4  | -0.1       | +0.2  | -3.9 |
| + GU       | -1.0  | -3.7  | 0.0        | -1.2  | +0.8 |
| + OU       | -4.0  | -5.4  | -0.1       | -2.6  | -4.3 |
| + GOU      | -5.0  | -5.9  | -0.1       | -3.2  | -5.5 |

Table 7: Accuracy change after enforcing different constraints on the generation. The "None" row shows the original performance without any constraint (from Table 1). Each row below adds a different set of constraints: G stands for "graph validity", O for "no overdependency", and U for "no under-dependency". Results are on all MWP datasets under self-consistent decoding.

100 reasoning chains for each domain generated by code-davinci-002 with the greedy decoding strategy, where each set of 100 contains an equal number of samples from all datasets within the domain. We further require annotators to have experience coding in the programming language of the dataset they annotate (Python for MWP/CLUTRR/Sports/Date, and Scala for StrategyQA, as Datalog was not an option in Prolific). We have a single survey for each domain, with the exception of Multi-hop QA (in this case, we have separate surveys for StrategyQA, Date, and Sports, given the different nature of each dataset). Additionally, there was no way to ensure annotators knew PPDL on Prolific. In order to ensure high-quality annotations for SayCan, the authors

<span id="page-15-2"></span>![](_page_15_Picture_5.jpeg)

Figure 9: Example of our annotation interface for the CLUTRR survey

annotate this dataset themselves.

Annotation Process Each of the 100 questions for each domain is annotated by at least three annotators. Each annotator is given 10 questions to annotate. A screenshot of our survey interface is shown in Figure 9. At the start of the survey, annotators are given examples of a correct reasoning chain, a reasoning chain with incorrect NL, and a reasoning chain with incorrect SL. The examples match the domain of reasoning chains in each survey. As an attention and understanding check, we ask annotators to label an example they had just seen. If the annotator fails this question, they are sent to the end of the survey and their responses are filtered out. We also manually filter out all spammers (annotators who answer with the same

<span id="page-16-1"></span>

| Domain     | Correct | Incorrect NL | Incorrect SL | Flawed Question | Confused | Agreement |
|------------|---------|--------------|--------------|-----------------|----------|-----------|
| MWP        | 92.0    | 3.0          | 2.0          | 2.0             | 1.0      | 0.790     |
| Planning   | 94.0    | 3.9          | 2.1          | 0.0             | 0.0      | 0.947     |
| StrategyQA | 66.7    | 30.3         | 3.0          | 0.0             | 0.0      | 0.455     |
| Date       | 87.9    | 6.1          | 6.1          | 0.0             | 0.0      | 0.788     |
| Sports     | 100.0   | 0.0          | 0.0          | 0.0             | 0.0      | 0.899     |
| Relation   | 88.0    | 6.0          | 0.0          | 5.0             | 1.0      | 0.683     |

Table 8: Numerical results for human evaluation of reasoning chain correctness, accompanying Figure [6.](#page-8-0) Each row represents the percent (0-100) of different answer choices selected by human evaluators in each domain/dataset, as well as the inter-annotator agreement.

response repeatedly or complete the survey in under 3 minutes). After the surveys are complete, we compute annotator agreement and then take the majority label for each reasoning chain as the final label in our analysis. Our annotator population consists of 100 annotators with an average age of 25 years and an average income of 40k per year. 87.9% of the annotators are males and 56% have a four year college degree. Annotators are compensated at \$16/hr, and average 2 minutes per question. Our full study cost \$280. Sample instructions for the CLUTRR survey can be found in the Supplementary Materials.

# <span id="page-16-0"></span>E Dataset Details

# E.1 Statistics

We show the dataset details in Table [9,](#page-17-0) including the statistics, the number of few-shot exemplars used in the prompt, and example inputs and outputs.

In particular, we notice that in one of our baselines [Wei et al.](#page-11-0) [\(2022\)](#page-11-0), the reported number of exemplars used in the prompt is inconsistent between the main text (10) and the appendix (6). To ensure fair comparison, we rerun the baseline with 10 exemplars for our results in Table [1,](#page-6-0) which is what we use for our method.

# E.2 URLs and Licenses

We use the same distribution of datasets following [Wei et al.](#page-11-0) [\(2022\)](#page-11-0):

# Math Word Problems

• GSM8K [\(Cobbe et al.,](#page-10-12) [2021\)](#page-10-12): [https://](https://github.com/openai/grade-school-math) [github.com/openai/grade-school-math](https://github.com/openai/grade-school-math), MIT license: [https://github.com/](https://github.com/openai/grade-school-math/blob/master/LICENSE) [openai/grade-school-math/blob/](https://github.com/openai/grade-school-math/blob/master/LICENSE) [master/LICENSE](https://github.com/openai/grade-school-math/blob/master/LICENSE).

- SVAMP [\(Patel et al.,](#page-10-13) [2021\)](#page-10-13): [https://](https://github.com/arkilpatel/SVAMP) [github.com/arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP), MIT license: [https://github.com/arkilpatel/](https://github.com/arkilpatel/SVAMP/blob/main/LICENSE) [SVAMP/blob/main/LICENSE](https://github.com/arkilpatel/SVAMP/blob/main/LICENSE).
- MultiArith [\(Roy and Roth,](#page-11-9) [2015\)](#page-11-9), license: CC BY 4.0.
- ASDiv [\(Miao et al.,](#page-10-14) [2020\)](#page-10-14): [https://github.](https://github.com/chaochun/nlu-asdiv-dataset) [com/chaochun/nlu-asdiv-dataset](https://github.com/chaochun/nlu-asdiv-dataset).
- AQuA [\(Ling et al.,](#page-10-15) [2017\)](#page-10-15): [https:](https://github.com/deepmind/AQuA) [//github.com/deepmind/AQuA](https://github.com/deepmind/AQuA), license: [https://github.com/deepmind/AQuA/](https://github.com/deepmind/AQuA/blob/master/LICENSE) [blob/master/LICENSE](https://github.com/deepmind/AQuA/blob/master/LICENSE).

#### Multi-hop QA

- StrategyQA [\(Geva et al.,](#page-10-16) [2021\)](#page-10-16): we use the open-domain setting (question-only set) from [\(BIG-Bench collaboration,](#page-9-2) [2021\)](#page-9-2): [https://github.com/google/BIG-bench/](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/strategyqa) [tree/main/bigbench/benchmark\\_tasks/](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/strategyqa) [strategyqa](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/strategyqa).
- Date Understanding and Sports Understanding from BIG-Bench [\(BIG-Bench](#page-9-2) [collaboration,](#page-9-2) [2021\)](#page-9-2): Apache License v.2: [https://github.com/google/BIG-bench/](https://github.com/google/BIG-bench/blob/main/LICENSE) [blob/main/LICENSE](https://github.com/google/BIG-bench/blob/main/LICENSE).

#### Planning

• SayCan [\(Ahn et al.,](#page-9-3) [2022\)](#page-9-3): SayCan dataset can be accessed at [https://say-can.](https://say-can.github.io/) [github.io/](https://say-can.github.io/) under CC BY 4.0 license.

# Relational Reasoning

• CLUTRR [\(Sinha et al.,](#page-11-7) [2019\)](#page-11-7): [https://](https://github.com/facebookresearch/clutrr) [github.com/facebookresearch/clutrr](https://github.com/facebookresearch/clutrr), license: [https://github.com/](https://github.com/facebookresearch/clutrr/blob/main/LICENSE) [facebookresearch/clutrr/blob/main/](https://github.com/facebookresearch/clutrr/blob/main/LICENSE)

<span id="page-17-0"></span>

| Domain                   | Dataset                 | # Shot | # Test | Example                                                                                                                                                                                                                   |  |
|--------------------------|-------------------------|--------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| Math<br>Word<br>Problems | GSM8K                   | 8      | 1,319  | Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?  A: 72                                                     |  |
|                          | SVAMP                   | 8      | 1,000  | Q: Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack. How much do you have to pay to buy each pack?  A: 51                                                                            |  |
|                          | MultiArith              | 8      | 600    | Q: For Halloween Debby and her sister combined the candy they received. Debby had 32 pieces of candy while her sister had 42. If they ate 35 pieces the first night, how many pieces do they have left?  A: 39            |  |
|                          | ASDiv                   | 8      | 2,096  | Q: Seven red apples and two green apples are in the basket. How many apples are in the basket?  A: 9                                                                                                                      |  |
|                          | AQuA                    | 8      | 254    | Q: A car finishes a journey in 20 hours at the speed of 60 km/hr. If the same distance is to be covered in 10 hours, how much speed does the car gain? A: "120 kmph"                                                      |  |
| Multi-<br>hop<br>QA      | StrategyQA              | 6      | 2,290  | Q: Did Aristotle use a laptop? A: False                                                                                                                                                                                   |  |
|                          | Date<br>Understanding   | 10     | 359    | Q: Yesterday was April 30, 2021. What is the date tomorrow in MM/DD/YYYY? A: "05/02/2021"                                                                                                                                 |  |
|                          | Sports<br>Understanding | 10     | 977    | $Q\!:$ Is the following sentence plausible: "Lebron James hit the turnaround jumper"? A: True                                                                                                                             |  |
| Planning                 | SayCan                  | 7      | 103    | Q: Could you get me a drink with caffeine? A: "1.find(redbull) 2.pick(redbull) 3.find(user) 4.put(redbull 5.done()."                                                                                                      |  |
| Relational<br>Inference  | CLUTRR                  | 8      | 1,042  | Q: [Carlos] is [Clarence]'s brother. [Carlos] and his sister, [Annie], went shopping. [Annie] asked her mom [Valerie] if she wanted anything, but [Valerie] said no. How is [Valerie] related to [Clarence]?  A: "mother" |  |

Table 9: Datasets used for evaluation. "# Shot" stands for the number of few-shot examples in the prompt (following Wei et al. (2022)) and "# Test" stands for the number of test examples.

LICENSE. We obtain the publicly distributed version available at https://drive.google.com/file/d/1SEq\_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF/view, specifically the data\_089907f8 split.

We use all the above datasets for research purposes only, consistent with their intended use.

#### E.3 Data Cleaning

We perform manual cleaning on ASDiv, Date Understanding, Sports Understanding, and SayCan as we discover a number of annotation issues. In our experiment, we rerun all baselines on our cleaned version of the test sets. They are provided in our repository to assist future research.

Specifically, we clean each of the datasets as follows:

**ASDiv**: We start with the test set used by Wei et al. (2022), which removes all questions with float-valued and string-valued answers. However, in their released version, we notice an error in the answer extraction step for questions with more than one value in the answer (e.g., "what is the width and

length of X?", where the answer consists of two values). In their implementation, only the first value is extracted as the ground truth answer, which is then compared against model outputs. This might artificially inflate the final accuracy. To fix this, we extract all values in the answer as a set and compare model outputs against it.

**Date Understanding**: We find a number of wrong answers in the test set. For example, for the question "Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date today in MM/DD/YYYY?", the provided answer is "01/02/1961", whereas the correct answer should be "01/02/1963". We manually correct these answers, and the resulting test set has the same number of examples as the original one.

**Sports Understanding**: We notice a few ambiguities with the Sports Understanding dataset. For instance, running out of bounds is illegal in many sports. The phrase "Domantas Sabonis ran out of bounds" is labeled as implausible, however, Domantas Sabonis is a basketball player, and basketball players can indeed run out of bounds on

the court. We remove 8 questions with such actionbased ambiguities. Additionally, since the release of this dataset, a few new athletes have risen to fame with identical names to those mentioned in the dataset. For example, the question "Chris Paul struck out the side" is implausible, as the referenced "Chris Paul" is a famous basketball player. However, "Chris Paul" is also the name of a new MLB baseball player, in which case this statement is plausible. We remove 5 questions with such name-based ambiguities.

SayCan: We discover a few issues in the test set: (1) the environment setup (e.g., the list of objects, the list of locations, and the initial location of each object) is not the same for all examples; (2) the annotation of the ground truth answer is often incomplete (i.e., for a given task like "visit all locations", there exist many possible plans in terms of the order of locations visited, but not all of them are included in the annotation); (3) there are ambiguous descriptions in certain queries, for example, in "Could you get me something refreshing?", it is unclear what drinks are considered "refreshing". For these questions, we complete the annotation whenever possible, and filter out the rest. The resulting test set contains 103 examples out of the original 120.

# E.4 Dataset Splits

As stated in Section [4.1,](#page-4-3) we use the official splits whenever possible: training set for exemplar selection, validation set for prompt tuning, and test set for evaluation. In cases where they are available, we adopt the following strategies for each dataset:

GSM8K: it only has training and test sets. We form the validation set by randomly sampling 1,000 examples from the training set.

Other MWP datasets: for AQuA, we use the official training/validation/test split. For the other datasets, only the test sets are used, since we have the same prompt for GSM8K and them.

Date Understanding and Sports Understanding: they only have test sets. We follow [Wei et al.](#page-11-0) [\(2022\)](#page-11-0) to select the same number of examples from the test set to form the few-shot prompt and use the remaining examples as a new test set.

SayCan: Following [Wei et al.](#page-11-0) [\(2022\)](#page-11-0), we manually write 7 few-shot exemplars, since no training set is provided. We evaluate the models on our cleaned version of the test set, described in the previous subsection.

CLUTRR: this dataset is split into multiple folds. There is a training fold with K ∈ {2, 3} (where K is the number of intermediate steps required to reach the answer), and one test fold for each K from 2 to 10. We construct the few-shot prompt using exemplars from the training fold, and test our method on the concatenation of all test folds.

# <span id="page-18-0"></span>F Error Analysis

To further investigate where our method still fails, we inspect 100 errors[10](#page-18-1) from model predictions on each of the four datasets and manually annotate the error categories.

# <span id="page-18-2"></span>F.1 GSM8K

<span id="page-18-3"></span>![](_page_18_Figure_12.jpeg)

Figure 10: Error analysis for GSM8K. For a detailed description of the error categories, see Appendix [F.1.](#page-18-2)

As shown in Figure [10,](#page-18-3) we categorize the errors on GSM8K into 6 types, inversely sorted with frequency:

Wrong Subquestion (49%): The LM produces a wrong NL subquestion, which eventually leads to the incorrect answer. While this is the majority error type in our sample, it is worth noting that in a typical human-in-the-loop collaboration, these errors are easily fixable. Even if the user is unfamiliar with programming, they can inspect the NL subquestions and potentially correct the model error by simply deleting or editing a wrong subquestion. Wrong Code (24%): The NL subquestion is correct, but the code fails to answer the subquestion correctly. For example, the code uses a variable that has not been previously defined.

Semantic Understanding Error (12%): The LM incorrectly interprets certain semantic subtleties in the query. This is the most complex and most interesting error category. For example, consider the

<span id="page-18-1"></span><sup>10</sup>To encourage sample diversity, we embed all the errors using text-davinci-002 and cluster the embeddings using spectral clustering. This produces around 70 clusters of different sizes, from which we gather 100 samples using importance sampling.

```
# 1. How many pounds will Martin lose per week if he eats Cheerios every day for breakfast? (independent, support:
["he'll lose 1.25 pounds/week"])
pounds_lost_cheerios = 1.25
# 2. How many pounds will Martin gain per week if he eats donuts every day for breakfast? (independent, support:
["he'll gain 1.75 pounds/week"]) pounds_gained_donuts = 1.75
# 3. How many weeks are there in 5 weeks? (independent, support: ["External knowledge: there are 7 days in a
week"])
```

weeks\_in\_5\_weeks = 5 # 4. How many pounds will Martin lose in 5 weeks if he eats Cheerios every day for breakfast? (depends on 1 and 3, support: [])

pounds\_lost\_cheerios\_5\_weeks = pounds\_lost\_cheerios \* weeks\_in\_5\_weeks

# 5. How many pounds will Martin gain in 5 weeks if he eats donuts every day for breakfast? (depends on 2 and 3, support: [])

pounds\_gained\_donuts\_5\_weeks = pounds\_gained\_donuts \* weeks\_in\_5\_weeks

# 6. What will be the difference in his weight at the end of 5 weeks between the two breakfast options? (depends on 4 and 5, support: [])

difference\_5\_weeks = pounds\_gained\_donuts\_5\_weeks - pounds\_lost\_cheerios\_5\_weeks

# 7. Final Answer: What will be the difference in his weight at the end of 5 weeks between the two breakfast options? (depends on 6, support: [])

answer = difference\_5\_weeks

Table 10: Generated code for the question in Appendix [F.1,](#page-18-2) as an example of "semantic understanding error".

#### following problem:

*If Martin eats Cheerios every day for breakfast, he'll lose 1.25 pounds/week. If he eats donuts every day for breakfast, he'll gain 1.75 pounds/week. What will be the difference in his weight at the end of 5 weeks between the two breakfast options?*

The generated code, as shown in Table [10,](#page-19-1) does not assign opposite polarities (signs) for "pounds lost" vs. "pounds gained". For other examples in this category, we notice errors like missing that a pair of something has 2 items in it, missing to subtract 2 for "two years ago" when it occurs as a subjunctive, and so on. Fixing these errors, in general, will require more than providing additional examples in the prompt.

Generation Cutoff (7%): The generation stops midway, mainly due to the LM producing the same steps over and over again. These errors could be easily detected in postprocessing and possibly fixed by re-prompting the LM.

Wrong Gold Label (5%): We find 5 (out of our 100) examples that are genuine annotation errors in the gold labels.

Missing Subquestion (3%): The LM misses a relevant subquestion needed for the rest of the reasoning chain to work. These errors are also potentially fixable via human-in-the-loop interaction, where the user can insert a subquestion into the reasoning chain.

## <span id="page-19-2"></span>F.2 StrategyQA

As mentioned in Section [5,](#page-5-2) Figure [11](#page-19-0) shows the error type distribution on a sample of 70 instances from StrategyQA, where we specifically compare the cases where the prediction of CoT is correct whereas ours is wrong.

<span id="page-19-0"></span>![](_page_19_Figure_18.jpeg)

Figure 11: Error analysis for StrategyQA. For a detailed description of the error categories, see Section [F.2.](#page-19-2)

There are 8 error types, from most to least frequent:

Syntax (29%): these are syntax-related errors that result in an invalid Datalog program. For example, the LM redefines a relation that is previously declared, uses an invalid string comparison operation, generates an ungrounded variable in the answer definition, and so on. With further pretraining on Datalog, we believe Codex is able to overcome these errors to a reasonable extent.

Infinite Loop (24%): the LM generates an in-

finite loop of subquestions, which are oftentimes rather philosophical. For example, for the question "Are potatoes native to the European continent?", the LM generates a list of 40 subquestions (potentially more, if not truncated by the token limit):

- 1. Where are potatoes native to? Potatoes are native to the Andes.
- 2. Which continent is the Andes in? The Andes is in South America.
- 3. Which continent is South America in? South America is in the Americas.
- 4. Which continent is the Americas in? The Americas is in the Western Hemisphere.
- 5. Which hemisphere is the Western Hemisphere in?

The Western Hemisphere is in the Earth. 6. Which continent is the Earth in? The Earth is in the Solar System.

......

- 11. Which continent is the Universe in? The Universe is in the Multiverse.
- 12. Which continent is the Multiverse in?

The Multiverse is in the Omniverse.

13. Which continent is the Omniverse in?

The Omniverse is in the Metaverse.

14. Which continent is the Metaverse in? The Metaverse is in the Hyperverse.

......

39. Which continent is the Quattuordecaverse in?

The Quattuordecaverse is in the Quindecaverse.

40. Which continent is the Quindecaverse in?

We notice that this type of errors also happen if the prompt does not include Datalog code but just the NL subquestions, but much less often.

Wrong Subquestions (17%): The LM generates an irrelevant subquestion to the original question. For instance, given the question, "Would toast for a vegan have margarine instead of butter?", one subquestion that is generated is "What is the difference between a vegan and a vegetarian?", which does not help with reaching the final answer.

Wrong Reasoning (10%): The subquestions are correct, but the reasoning from the retrieved facts to the final answer is problematic. For example, to answer the question, "Do solo pianists require a conductor?", the model correctly generates the relevant subquestions and their respective answers:

- 1. What is the definition of a solo pianist? A solo pianist is a pianist who performs alone.
- 2. What is the definition of a conductor? A conductor is a person who leads an orchestra.
- 3. What is the definition of an orchestra? An orchestra is a large instrumental ensemble that contains string, brass, woodwind, and percussion instruments.

But in deriving the final answer, the reasoning is wrong: "Now, we derive the final answer: Do solo pianists require a conductor? The answer is Yes only if a solo pianist is not a conductor."

Wrong Knowledge (10%): the LM fails to retrieve the correct knowledge to answer the subquestions. For example, given the original question "Is the largest city in New Mexico also known as Yootó?", the model correctly generates the subquestions "What is the largest city in New Mexico?" (answer: Albuquerque) and "Is Albuquerque also known as Yootó?". However, when answering the second subquestion, it retrieves a wrong piece of knowledge ("Albuquerque is also known as Yootó.", whereas in reality, it should be "Santa Fe" that is known as Yootó).

Answer Definition (6%): In our prompt, we always derive the answer in the format of "The answer is Yes only if ...", which is followed by a Datalog rule containing conditions that should be satisfied for the answer to be true. However, the LM sometimes generates this as "The answer is No only if ...", which outputs the reversed answer.

Knowledge Representation (3%): The retrieved knowledge in NL is correct, but the representation of it in Datalog is wrong. For example, for the piece of knowledge "The Lucy Show is not the same TV series as JAG (TV series)", the model represents it as follows:

```
.decl Same_TV_series(TV_series1:symbol,
TV_series2:symbol)
Same_TV_series("The Lucy Show", "JAG (TV
series)")."
```

which actually means the reverse (they are the same).

Unknown (1%): There is a very small proportion of errors (1 out of 70) where we are unsure

of the cause. Specifically, we expect the solver to output True, but it outputs False instead.

#### <span id="page-21-0"></span>F.3 Date Understanding

<span id="page-21-1"></span>![](_page_21_Figure_2.jpeg)

Figure 12: Error analysis for Date Understanding. For a detailed description of the error categories, see Appendix [F.3.](#page-21-0)

Unlike GSM8K, we only have 69 errors out of the 359 test examples, so we annotate them all, as shown in Figure [12.](#page-21-1) The error categories for date understanding are similar to GSM8K, except that we do not see any generation errors in the samples, but we see questions with ambiguous phrasing allowing both the gold and predicted answers to be correct based on interpretation.

#### <span id="page-21-2"></span>F.4 SayCan

<span id="page-21-3"></span>![](_page_21_Figure_6.jpeg)

Figure 13: Error analysis for SayCan. For a detailed description of the error categories, see Section [F.4.](#page-21-2)

Since SayCan only has 120 test examples and Faithful CoT produces 7 errors, we annotate all 7 of them, as shown in Figure [13.](#page-21-3) These 7 examples can be categorized into two types:

Additional Subgoals (64%): Cases where the model generated unnecessary subgoals in the decomposition of the original task, leading the planner astray. This is illustrated by the request "Clear the jalapeno chips off the counter":

```
(:goal
```

```
(and
            (not (at jalapeno-chips counter))
            (not (at jalapeno-chips table))
            (not (at jalapeno-chips trash))
            (not (at jalapeno-chips bowl))
            (not (at jalapeno-chips user))
    )
)
```

Wrong Object (36%): Here the model generates the wrong object/object types in the goal. For example, a request such as "I opened a pepsi earlier. How would you bring me an open can?" fails because the model generates actions with water instead of Pepsi.

# <span id="page-21-4"></span>F.5 CLUTRR

<span id="page-21-5"></span>![](_page_21_Figure_14.jpeg)

Figure 14: Error analysis for CLUTRR. For a detailed description of the error categories, see Section [F.5.](#page-21-4)

For CLUTRR, we group all error cases by K, the number of steps in their gold reasoning chain, as a proxy for problem complexity, and perform importance sampling on these groups to select 100 examples. Our annotation of these examples reveals 5 error categories, as shown in Figure [14:](#page-21-5)

Inversed Relation (41%): This stands out as the majority of the errors. These errors are caused by the reversal of directional relationships for the actors in the problem, i.e., predicting "mother" or "nephew" when the answer is "daughter" or "uncle" respectively.

Wrong Relation (30%): Here the model extracts the relation incorrectly (not even the inverse). For example, for the subquestion "How is [Donald] related to [Jason]?" with the correctly identified support "[Jason] is father of their father", the model produces relation(Donald, Jason) = son when the correct relation should be "grandson".

Nonexistent Relation (4%): The model hallucinates a non-existent relation (e.g. "adopted" for

daughter).

Wrong Path (12%): Here, the model does not generate a correct reasoning path from target entity A to target entity B in the question.

Wrong Gold Label (13%): These are annotation errors in the CLUTRR dataset. In one example, for the sentence, "[Gloria] asked her mother [Laura] if she could go outside and play with her friends.", the annotation says Laura is Gloria's grandmother.

# G Prompts

Due to the space limit, we show one exemplar in the prompt for each dataset here. The full prompts can be found in our repository.

Among all the MWP datasets, our prompt for AQuA is different from the rest, because the answers are in a multiple-choice format instead of integers. To produce a multiple-choice answer, we take a two-step approach by first producing a numerical answer in the same way as for the other math datasets. Then, we perform an additional step of converting the numerical answer into an answer choice by again prompting the language model to generate which answer choice is closest to the previously produced numerical answer. An exemplar of this 2-step prompt is shown in Table [11.](#page-23-0)

#### <span id="page-23-0"></span>EXEMPLAR FOR AQUA

#### Step 1: Answer Prediction

# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:

# Answer option: ['A)1 hour', 'B)2 hours', 'C)3 hours', 'D)4 hours', 'E)5 hours']

# Write Python code to solve the following questions. Store your result as a variable named 'answer'.

# 1. What was the duration of the flight? (independent, support: ["The duration of the flight is"])

duration = Symbol('duration', positive=True)

# 2. What is the delay of the flight? (independent, support: ["the time of flight increased by 30 minutes"]) delay = 30 / 60

# 3. What was the total flight distance? (independent, support: ["In a flight of 600 km"])

total\_distance = 600

# 4. What was the original speed? (depends on 1 and 3, support: ["External knowledge: speed is distance over time"]) original\_speed = total\_distance / duration

# 5. What was the reduced speed? (depends on 1, 2, and 3, support: [])

reduced\_speed = total\_distance / (duration + delay)

# 6. What was the duration of the flight if the original speed was 200 km/hr faster than the reduced speed? (depends on 4, 5, and 1, support: [])

solution = solve\_it(original\_speed - reduced\_speed - 200, duration)

answer = solution[duration]

#### Step 2: Multiple Choice Conversion

# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:

# Answer option: ['A)1 hour', 'B)2 hours', 'C)3 hours', 'D)4 hours', 'E)5 hours']

# Prediction: 1.00000000000000

# Closest Option: A

Table 11: An exemplar from our prompt for AQuA.

#### EXEMPLAR FOR GSM8K, SVAMP, MULTIARITH, AND ASDIV

# Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

# To answer this question, write a Python program to answer the following subquestions:

# 1. How many trees are there in the beginning? (independent, support: ["There are 15 trees"]) trees\_begin = 15

# 2. How many trees are there in the end? (independent, support: ["there will be 21 trees"])

trees\_end = 21

# 3. How many trees did the grove workers plant today? (depends on 1 and 2, support: [])

trees\_today = trees\_end - trees\_begin

# 4. Final Answer: How many trees did the grove workers plant today? (depends on 3, support: []) answer = trees\_today

Table 12: An exemplar from our prompt for GSM8K, SVAMP, MultiArith, and ASDiv.

#### EXEMPLAR FOR STRATEGYQA

// Q: Would a pear sink in water?

// To answer this question, we answer the following subquestions:

// 1. What is the density of a pear?

// The density of a pear is about 0.6g/cm<sup>3</sup> .

// 2. What is the density of water?

// Water has a density of 1g/cm<sup>3</sup> .

// Then, we represent these answers in Datalog:

// 1. The density of a pear is about 0.6g/cm<sup>3</sup> .

.decl Has\_density(Object:symbol, Density:float)

Has\_density("pear", 0.6).

// 2. Water has a density of 1g/cm<sup>3</sup> .

Has\_density("water", 1).

// Now, we derive the final answer: Would a pear sink in water?

// The answer is Yes only if the density of a pear is more than the density of water.

.decl Answer()

Answer() :- Has\_density("pear", density1), Has\_density("water", density2), density1 > density2.

.output Answer

Table 13: An exemplar from our prompt for StrategyQA.

#### <span id="page-24-0"></span>EXEMPLAR FOR DATE UNDERSTANDING # Q: Yesterday was April 30, 2021. What is the date tomorrow in MM/DD/YYYY? # To answer this question, we write a program to answer the following subquestions: # import relevant packages from datetime import date, time, datetime from dateutil.relativedelta import relativedelta # 1. What is the date yesterday? (independent, support: ["Yesterday was April 30, 2021"]) date\_yesterday = date(2021,4,30) # 2. What is the date today? (depends on 1, support: ["Yesterday was April 30, 2021"]) date\_today = date\_yesterday + relativedelta(days=1) # 3. What is the date tomorrow? (depends on 2, support: []) date\_tomorrow = date\_today + relativedelta(days=1) # 4. Final Answer: What is the date tomorrow in MM/DD/YYYY? (depends on 3, support: [])

Table 14: An exemplar from our prompt for Date Understanding.

#### <span id="page-24-1"></span>EXEMPLAR FOR SPORTS UNDERSTANDING

answer = date\_tomorrow.strftime("%m/%d/%Y")

# Q: Is the following statement plausible? Sam Darnold passed the puck

# To answer this question, write a Python program to answer the following subquestions:

# 1. Sam Darnold is a player in which sport? (independent, support: ["Sam Darnold is an NFL Quarterback", "NFL is the National Football League"])

player\_sport = "football"

# 2. The phrase "passed the puck" implies playing which sport? (independent, support: ["Players pass the puck in hockey"]) playing\_sport = "hockey"

# 3. Is the following statement plausible? Sam Darnold passed the puck (depends on 1 and 2, support: ["Sam Darnold is an NFL Quarterback", "NFL is the National Football League", "Players pass the puck in hockey"]) plausibility = (player\_sport == playing\_sport)

# 4. Is the following statement plausible? Sam Darnold passed the puck (depends on 3, support: []) answer = int(plausibility)

Table 15: An exemplar from our prompt for Sports Understanding.

```
EXEMPLAR FOR SAYCAN
User query: Bring me something not sweet to eat.
Goal in PDDL:
(:goal
    ; I need to find a snack
    (exists (?s - snack)
         ; it has to satisfy the following conditions
         (and
              ; the snack must not be sweet
              (not (is-sweet ?s))
              ; bring it to the user
              (at ?s user)
         )
    )
)
```

Table 16: An exemplar from our prompt for SayCan.

#### EXEMPLAR FOR CLUTRR

# Context: [Jason] always had some great adventure planned for his granddaughter [Guillermina] when she came to visit. So, naturally, when [Myrna] told her daughter [Guillermina] that they would be going to visit [Jason] she could hardly contain herself.

# Question: How is [Jason] related to [Myrna]?

# To answer this question, we write a program to answer the following subquestions:

# 1. How is [Jason] related to [Guillermina]? (independent, support: "[Jason] always had some great adventure planned for his granddaughter [Guillermina] when she came to visit.") relation(Jason, Guillermina) = grandfather

# 2. How is [Guillermina] related to [Myrna]? (independent, support: "So, naturally, when [Myrna] told her daughter [Guillermina] that they would be going to visit [Jason] she could hardly contain herself.") relation(Guillermina, Myrna) = daughter

# 3. Final answer: How is [Jason] related to [Myrna]? (depends on 1, 2) relation(Jason, Myrna) = relation(Jason, Guillermina) @ relation(Guillermina, Myrna)

Table 17: An exemplar from our prompt for CLUTRR.