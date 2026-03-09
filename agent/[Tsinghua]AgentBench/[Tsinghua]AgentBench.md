# AgentBench: Evaluating LLMs as Agents

Xiao Liu1<sup>∗</sup> , Hao Yu1∗†, Hanchen Zhang<sup>1</sup> , Yifan Xu<sup>1</sup> , Xuanyu Lei<sup>1</sup> , Hanyu Lai<sup>1</sup> , Yu Gu2† , Hangliang Ding<sup>1</sup> , Kaiwen Men<sup>1</sup> , Kejuan Yang<sup>1</sup> , Shudan Zhang<sup>1</sup> , Xiang Deng<sup>2</sup> , Aohan Zeng<sup>1</sup> , Zhengxiao Du<sup>1</sup> , Chenhui Zhang<sup>1</sup> , Sheng Shen<sup>3</sup> , Tianjun Zhang<sup>3</sup> , Yu Su<sup>2</sup> , Huan Sun<sup>2</sup> , Minlie Huang<sup>1</sup> , Yuxiao Dong<sup>1</sup> , Jie Tang<sup>1</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>The Ohio State University <sup>3</sup>UC Berkeley

# Abstract

Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks. As a result, there has been an urgent need to *evaluate LLMs as agents* on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 25 LLMs (including APIs and open-sourced models) shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and open-sourced competitors. It also serves as a component of an ongoing project with wider coverage and deeper consideration towards systematic LLM evaluation. Datasets, environments, and an integrated evaluation package for AgentBench are released at <https://github.com/THUDM/AgentBench>.

![](_page_0_Figure_6.jpeg)

- (a) Typical LLMs' AgentBench performance (relative) against the best in each environment.
- (b) Overall scores of AgentBench across 8 environ -ments. Dashed lines for two LLM types' average.

Figure 1: An overview of LLMs on AgentBench. While LLMs begin to manifest their proficiency in LLM-as-Agent, gaps between models and the distance towards practical usability are significant.

# 1 Introduction

Intelligent agents and autonomous entities [\[67;](#page-19-0) [49;](#page-18-0) [92\]](#page-21-0) capable of decision making and action execution in a particular environment have been key concepts of artificial intelligence (AI) historically.

<sup>\*</sup>XL and HY are lead authors that contributed equally. Email: {shawliu9,longinyh}@gmail.com

<sup>†</sup>Work partially done when HY, YG visited Tsinghua University.

<sup>‡</sup>Website for AgentBench leaderboard & demos: <https://llmbench.ai>

![](_page_1_Figure_0.jpeg)

Figure 2: AgentBench is the first systematic benchmark to evaluate LLM-as-Agent on a wide array of real-world challenges and 8 distinct environments. In total, 25 LLMs are examined in its first edition.

Notwithstanding substantial advancements in deep learning algorithms applied in both computer vision [\[31;](#page-17-0) [62\]](#page-19-1) and natural language processing (NLP) [\[39;](#page-18-1) [11\]](#page-16-0), their potential for developing effective and efficient agents to assist humans remains largely unexplored.

The advent of Large Language Models (LLMs) [\[11;](#page-16-0) [15;](#page-16-1) [101;](#page-21-1) [66;](#page-19-2) [100;](#page-21-2) [80\]](#page-20-0), such as GPT-4 [\[56\]](#page-18-2), has brought plenty of new opportunities to this realm. Through extensive alignment training [\[75;](#page-20-1) [58;](#page-19-3) [8;](#page-15-0) [90;](#page-21-3) [65\]](#page-19-4), LLMs have not only mastered traditional NLP tasks such as question answering [\[61\]](#page-19-5), natural language inference [\[10\]](#page-16-2), and text summarization [\[52\]](#page-18-3), but also showcased an impressive ability to comprehend human intent and execute instructions. This has spurred the development of various applications like AutoGPT [\[64\]](#page-19-6), BabyAGI [\[51\]](#page-18-4), and AgentGPT [\[1\]](#page-15-1), which employ LLMs for autonomous goal completion, sparking substantial public interest and numerous discussions. LLM agents are also being experimented with in social and game contexts [\[59;](#page-19-7) [86;](#page-20-2) [104\]](#page-21-4).

Despite these advancements, the lack of a systematic and standard benchmark to evaluate LLM-as-Agent presents a critical challenge. Historically, text-based game environments [\[57;](#page-18-5) [17;](#page-16-3) [30;](#page-17-1) [82\]](#page-20-3) have been employed for language agent evaluation. However, they often suffer from limitations due to their closed, discrete action spaces, and their focus is primarily narrowed on models' commonsense grounding abilities. More recently, attempts on embodied agents [\[63;](#page-19-8) [35;](#page-17-2) [2\]](#page-15-2) have employed complicated multi-modal simulators based on games [\[41;](#page-18-6) [22\]](#page-16-4), GUI [\[69;](#page-19-9) [81\]](#page-20-4), and indoor scenes [\[68;](#page-19-10) [74\]](#page-19-11). However, these simulators, despite their complexity, do not accurately reflect the practical use-cases of LLMs, and their multi-modal nature creates a hurdle for the urgent evaluation of text-only LLMs. Furthermore, most benchmarks for agents focus on single environments, limiting their capability to provide a comprehensive overview of LLMs across diverse application scenarios.

To address these challenges, we introduce AgentBench, a multi-dimensional benchmark designed to evaluate LLM-as-Agent across a spectrum of different environments. AgentBench encompasses 8 distinct environments, 5 of which are created in this paper for the first time, namely operating system (OS), database (DB), knowledge graph (KG [\[4\]](#page-15-3)), digital card game (DCG), and lateral thinking puzzles (LTP [\[5\]](#page-15-4)). The remaining 3 are recompiled from published datasets, which are house-holding (Alfworld [\[71\]](#page-19-12)), web shopping (WebShop [\[96\]](#page-21-5)), and web browsing (Mind2Web [\[19\]](#page-16-5)). All datasets are meticulously designed and reformulated to simulate interactive environments where text-only LLMs can operate as autonomous agents. Furthermore, they systematically evaluate an LLM's core abilities, including following instructions [\[58\]](#page-19-3), coding [\[12\]](#page-16-6), knowledge acquisition [\[38;](#page-17-3) [78\]](#page-20-5), and logical reasoning [\[73\]](#page-19-13). Consequently, AgentBench serves as an ideal testbed for both LLM and agent evaluation.

In addition, we have conducted a comprehensive evaluation of 25 different LLMs using AgentBench, including both API-based and open-source models. Our results reveal that top-tier models like GPT-4 are capable of handling a wide array of real-world tasks, indicating the potential for developing a potent, continuously learning agent. However, we also note a significant performance gap between these top-tier models and their open-source counterparts. Despite the recent success of open-source LLMs and their competitive scores on several benchmarks [], their performance on the challenging

<span id="page-2-0"></span>Table 1: AgentBench evaluates 25 API-based or open-sourced LLMs on LLM-as-Agent challenges.

| Model                                                                              | Model Size                                               | Form                     | Version              | Creator          |
|------------------------------------------------------------------------------------|----------------------------------------------------------|--------------------------|----------------------|------------------|
| gpt-4 [56]<br>gpt-3.5-turbo [55]<br>text-davinci-003 [58]<br>text-davinci-002 [58] | undisclosed<br>undisclosed<br>undisclosed<br>undisclosed | api<br>api<br>api<br>api | 0613<br>0613         | OpenAI           |
| claude [6]<br>claude-instant [6]                                                   | undisclosed undisclosed                                  | api<br>api               | v1.3<br>v1.1         | Anthropic        |
| text-bison-001[3]                                                                  | undisclosed                                              | api                      | -                    | Google           |
| chatglm2 [100; 21]<br>chatglm-6b [100; 21]<br>codegeex2-6b-chat [102]              | undisclosed<br>6B<br>6B                                  | api<br>open<br>open      | v0.2<br>v1.1         | Tsinghua & Zhipu |
| vicuna-13b [14]<br>vicuna-7b [14]                                                  | 13B<br>7B                                                | open<br>open             | v1.1<br>v1.1         | LMSYS            |
| llama2-chat-13b [80]<br>llama2-chat-7b [80]                                        | 13B<br>7B                                                | open<br>open             | -<br>-               | Meta             |
| openchat-8192-13b [85]<br>openchat-13b [85]                                        | 13B<br>13B                                               | open<br>open             | v1<br>v3.2           | Tsinghua         |
| wizardlm-30b [93]<br>wizardcoder-15b [48]<br>wizardlm-13b [93]                     | 30B<br>15B<br>13B                                        | open<br>open<br>open     | v1.0<br>v1.0<br>v1.0 | Microsoft        |
| koala-13b [26]                                                                     | 13B                                                      | open                     | -                    | UC Berkeley      |
| internlm-chat-7b [79]                                                              | 7B                                                       | open                     | -                    | InternLM Team    |
| baichuan-13b-chat [36]<br>baichuan-7b [36]                                         | 13B<br>7B                                                | open<br>open             | -                    | Baichuan Inc.    |
| dolly-12b [16]                                                                     | 12B                                                      | open                     | v2                   | Databricks       |
| oasst-sft-4-12b [42]                                                               | 12B                                                      | open                     | -                    | LAION            |

AgentBench tasks lags considerably. This underscores the necessity for additional efforts to enhance the learning abilities of open-source LLMs.

Lastly, there are definitely more scenarios and tasks to cover in LLM-as-agent evaluation. AgentBench is constantly evolving to be more inclusive and comprehensive. It is also part of our ongoing endeavor to establish a more systematic evaluation scheme for LLMs. We are continuously refining the scheme and anticipate its release with enriched AgentBench's next version in the near future.

In summary, the contributions of this work are:

- We introduce the concept of evaluating LLMs as agents, and present AgentBench, a comprehensive benchmark to standardize the evaluation. It defines 8 distinct environments based on real-world scenarios, offering a practical testbed for LLMs' wide array of capabilities.
- We perform a thorough evaluation of 25 different LLMs using AgentBench, uncovering a significant performance gap between leading top API-based LLMs and open-source models. It illuminates the state of the art in LLM-as-Agent and highlights areas for improvement.
- To facilitate the assessment of LLM-as-Agent, we release an integrated toolkit to handily customize the AgentBench evaluation to any LLMs based on the philosophy of "API & Docker" interaction. This toolkit, along with the associated datasets and environments, are made publicly available for the broader research community.

# 2 LLM-as-Agent: Definition and Preliminary

Here we formalize the terms for describing the evaluation for LLMs as agents and necessary preliminary knowledge for using LLMs in the context of agent evaluation.

**Definition: Interactive Evaluation of LLM-as-Agent.** The interactive evaluation of LLM-as-Agent could be regarded as a Partially Observable Markov Decision Process (S, A, T, R, U, O), which comprises state space S, action space A, transition function  $T: S \times A \to S$ , reward assigning function R, task instruction space U, and observation space O. Here we denote an LLM agent as O.

![](_page_3_Figure_0.jpeg)

#### (a) Operating System (OS)

Task: *"Find the total number of non-empty directories inside the '/etc' directory."*

Action Space: *Any valid bash commands* Observation: *System standard output*

![](_page_3_Figure_5.jpeg)

#### (c) Knowledge Graph (KG)

Task: *"Find tropical cyclones that are similar to Hurricane Marie and affected Eastern North America."*

Action space: *Basic KG-querying tools* Observation: *Query results*

![](_page_3_Figure_9.jpeg)

#### (e) Lateral Thinking Puzzles (LTP)

Task: *"A man sleeps with the lights off, and the next morning he suicides after opening windows. Why?"* Action Space: *Any binary questions*

Observation: *'Yes', 'No', or 'Irrelevant'*

![](_page_3_Figure_13.jpeg)

#### (g) Web Shopping (WS)

Task: *"Looking for a queen size bedspread set in the color redwood, and price lower than 70."*

Action space: *Search (generate keywords) and Click (choose from all clickable buttons)*

Observation: *Products' descriptions; the webpage*

#### (b) Database (DB)

Task: *"What was the total number of medals won by United States?", given the table 'Olympic Medals'* Action space: *Any valid SQL commands*

Observation: *MySQL command-line interface output*

![](_page_3_Figure_21.jpeg)

#### (d) Digital Card Game (DCG)

Task: *"Compete against another player using four 'fish' cards in 'Aquawar' game."*

Action space: *Four 'fish' cards and Assertion* Observation: *Battle process, status of 'fish'*

![](_page_3_Figure_25.jpeg)

#### (f) House-holding (HH)

Task: *"Clean some soapbar and put it in coutertop"* Action space: *A list of allowed actions in the room, or other accessible rooms*

Observation: *Results after the action.*

![](_page_3_Figure_29.jpeg)

#### (h) Web Browsing (WB)

Task: *"Find a latest post with more than 10k upvotes in r/announcements community and upvote it."*

Action space: *1) Choose one out of all HTML elements in the webpage; 2) Click, Type, or Select Options* Observation: *Webpage HTML (optional: screenshot)*

Figure 3: Examples of all environments in AgentBench. [Click to watch high-fidelity videos!]

Definition: Action Validity. Compared to previous evaluation of language agents [\[17;](#page-16-3) [74\]](#page-19-11), LLMs' open-ended generation allows a substantially more complex and open action space A. However, it also leads to a common problem for LLM agents to generate *invalid* actions [\[97\]](#page-21-8). For example, insufficiently-aligned LLMs would not follow complicated instructions [\[58\]](#page-19-3), while some other overaligned LLMs tend to refuse task instructions [\[23;](#page-16-10) [89\]](#page-20-8). In code-related tasks, LLMs are likely to generate codes that cause compiling or run-time errors [\[12;](#page-16-6) [24\]](#page-17-6). In AgentBench, we find that the requirement of action validity significantly increases the threshold for LLMs to act as agents.

Preliminary: Chain-of-Thought (CoT) and Other Reasoning Strategies. Since LLM-as-Agent requires LLMs' strong reasoning ability, CoT [\[91\]](#page-21-9), which has been considered a de facto strategy in related evaluation together with actions [\[99\]](#page-21-10), is also adopted in AgentBench. Despite many improved strategies proposed later, such as introducing ensemble [\[87\]](#page-20-9), reflection [\[70\]](#page-19-14), and search [\[98\]](#page-21-11), we evaluate LLMs with the most primitive CoT in AgentBench, as it reflects the most practical public user experience without multiple trails and repeated generation.

# 3 Composition of AgentBench

In this section, we introduce the datasets and environments that compose the AgentBench. Compared to previous agent evaluation benchmarks [\[17;](#page-16-3) [22\]](#page-16-4), AgentBench concentrates on the practical useoriented evaluation of LLMs via Chain-of-Thought (CoT) [\[91;](#page-21-9) [99\]](#page-21-10) prompting. Most of these datasets pinpoint promising directions of LLMs' applications with autonomous mission completing.

#### 3.1 Operating System (OS)

Introduction. Allowing LLMs to access and manipulate OS in terminal is a fascinating but challenging mission. Despite attempts on translating natural language to Shell commands [\[45\]](#page-18-10), few prior efforts evaluate on authentic executable environments. In this dataset, we aim to evaluate LLMs in genuine OS' interactive bash environments (i.e., Ubuntu Docker [\[50\]](#page-18-11)) on human questions with deterministic answers (e.g., *number of users with non-*/home *directories in an OS.*) or series of operations for practical goals (e.g., *recursively set all directory files to read-only, excluding mine*).

Construction Details. Each evaluation sample in OS dataset encompasses following contents:

- Instruction. The description of the problem in natural language that needs LLMs to solve.
- Docker Environment. The starting up docker image (e.g., preset default local-os/default).
- Initialization Script (Optional). The bash scripts that need to be executed independently (docker exec) before the interaction starts (e.g., user configurations, files, system statuses).
- Start Script (Optional). The bash scripts executed after shell is created and before interaction.
- Checking Pipeline. The checking method to judge the correctness of LLMs answer or operation.
- Example Script (Optional). The bash scripts that serve as reference solutions. In other words, if executing them in the interaction, results are correct. Only for unit tests that introduced below.

We design two types of tasks in the OS evaluation beyond conventional QA-only evaluation.

- Question Answering (QA): LLMs need to output commands to solve specific questions in OS (e.g., aggregate numbers, view file contents). In this case, they must commit answers finally.
- Operation: LLMs need to output commands to do some verifiable operations on the operating system (e.g., change file/user states). In this case, they do not need to commit final answers.

Thanks to the checking pipeline, two types of tasks can be evaluated in a unified solution.

Collecting challenging queries regarding OS could be difficult. In practice, about half of our instructions are created or collected from humans, while the other half are mostly QA problems generated by gpt-4 and strictly filtered by passing the unit tests (i.e., yield correct answers/states).

For human instructions, we first gather 6000 real problems and solutions with bash or shell tag from Stack Overflow[1](#page-4-0) . Then we sort them by the score (count of likes). We invite 8 annotators majored in programming to select challenging ones. For each selected problem, they create one or more task instructions and write a detailed problem description, the initialization script, the starting

<span id="page-4-0"></span><sup>1</sup> https://stackoverflow.com/

script, and the checking pipeline. Finally, we conduct a cross verification for each evaluation sample to make sure it's correct. For each problem, it takes about 2 hours to do the annotation.

For generated problems, our unit test contains the following parts. 1) Initialization Script Correction: we execute the initialization script and remove samples with wrong initialization whose exit code does not equal to 0. 2) Example Code Correction: we execute the example code and the checking pipeline to judge the correctness of the answer. We remove samples with wrong answers.

In the end, we curate 144 high-quality diverse OS evaluation samples accompanied with testing interactive environments and corresponding checking pipelines (i.e., scripts). Agents are prompted with 1-shot CoT to better format their responses (Cf. Appendix [A\)](#page-22-0).

Evaluation Setup. For each problem (i.e., instruction), the execution can be divided into 3 parts.

- Initialization. We create a docker container with a specific image, and we run an initialization bash script to set up environments specified by the instruction.
- Interaction. We start a new shell in this docker, and run the starting bash script specified by the instruction. Then the LLM to test is fed with a piece of instruction and the problem description. It starts interaction with the shell. In each turn, two actions are provides. One is to run bash script, which allows the model to generate and run a series of commands in the shell. The other is to commit answer, which allows the model to terminate the interaction process. It's notable that the model will be judged that it fail to solve the problem if exceeding round limit (8 by default).
- Checking. For each problem, there is a checking pipeline containing a list of scripts f1, f2, · · · , fn, where f<sup>k</sup> denotes the k-th script piece in the pipeline. For fk, the answer of the model, o0, and the output of ft(t < k), ot, will be fed as input arguments into fk, i.e., o<sup>k</sup> = fk(o0, o1, · · · , ok−1). The result is correct if and only if all the scripts exit with code 0.

Metrics. We measure the Success Rate for LLMs to solve problems in the execution. There are only two final status for each item of the problems, wrong or correct.

#### 3.2 Database (DB)

Introduction. Databases are another typical type of systems that real users may wish AIs to automatically interact with. Thus it is of paramount importance to examine LLMs' abilities to operate on real databases via SQL. Prior research has a large emphasis on translation between SQL and natural language, or answering questions given individual small tables. However, in AgentBench we evaluate LLMs on authentic SQL interfaces and databases as is in the real world.

Construction Details. We acquire the source queries and databases via reusing and amalgamating several established datasets: WikiSQL [\[103\]](#page-21-12), WikiTableQuestions [\[60\]](#page-19-15), SQA [\[37\]](#page-17-7), HybridaQA [\[13\]](#page-16-11), and FeTaQA [\[53\]](#page-18-12), ensuring the diversity of instructions and data.

To further enrich (and avoid leakage from) the dataset, we employed gpt-3.5-turbo to perform data augmentation. Provided with the header information and original rows of a table, gpt-3.5-turbo generates ten new rows. Using the name, header information, and some SQL examples, we task gpt-3.5-turbo with generating five additional SQL queries. Each acquired SQL statement is then fed sequentially into gpt-3.5-turbo with instructions to rephrase the sentences without changing their original meanings. The valid entries are filtered and sampled into the final dataset with 1599 entries, categorized into three basic types of DB operations: select, insert, or update.

As a result, each sample in the dataset comprises:

- Instruction. A piece of description delineating the problem and guiding the agent's action.
- Table Info. Explanations about the table name and column names (i.e., meta information).
- Table Content. The actual contents within the table, utilized to create the database.
- Correct Answer. For selection-type samples, it is a text answer; for other entry types (i.e., insert, update), it is the hash code of the correctly modified table.

Evaluation Setup. We assess each problem in the dataset through the following procedure:

• Initialization. An initial SQL script is constructed based on the table content, and a MySQL database is initialized in a docker container, which provides a forwarded port for interaction.

- Interaction. An initial prompt guides the agent to provide an executable SQL command along with its reasoning. The agent is provided with the prompt, instruction, and table information description, and it is expected to return a response in given format. We execute the SQL and directly return the result to the agent, continuing this loop until the agent commits its final answer or encounters an error (e.g., reaching the maximum round limit or failing to parse the action).
- Checking. For selection-type problems, we compare the agent's answer with the standard text answer, disregarding the order, but expecting an exact match. If the answer is a single number, all equivalent representations are accepted (e.g., 5, "5.0", '+5' are considered identical). For insertion or updating types of problems, we calculate and compare the hash of the table after the agent's operation with the hash of the table after the correct SQL operation.

Metrics. We measure the Success Rate of agents in completing instructions. Overall success rate is the macro average of the rate of three categories.

#### 3.3 Knowledge Graph (KG [\[4\]](#page-15-3))

Introduction. Engaging with a knowledge graph (KG) demands a broad range of skills from an intelligent agent [\[27\]](#page-17-8). Primarily, it needs to understand natural language, with all its intricacies and subtleties. It should also have the capacity to break down complex tasks into simpler, more manageable components. In addition, the agent requires the ability to formulate plans, strategize and adapt as necessary. Contemporary KGs are usually vast in size (e.g., FREEBASE [\[9\]](#page-15-7) has over 45M entities and 3B facts), making them too large for an agent to fully perceive. Operating in such environments, which are only partially observable, requires the agent to make decisions with incomplete information and manage inherent uncertainties. This heightens the need for the agent to demonstrate flexibility and adaptability in its decision making.

Given the challenges outlined above, a KG can act as a representative testing ground to assess the decision-making abilities of AI agents in complex real-world environments.

Construction Details. In an effort to gauge the decision-making abilities of LLMs, specifically their proficiency in long-term planning, we have meticulously compiled a dataset sourced from preexisting knowledge base question answering (KBQA) datasets on FREEBASE, including GrailQA [\[28\]](#page-17-9), ComplexWebQuestions [\[77\]](#page-20-10), and GraphQuestions [\[76\]](#page-20-11).

We envisage KBQA as a tool learning setting, thereby outfitting the LLM with an array of KGquerying tools. By leveraging the S-expressions annotated in [\[29\]](#page-17-10), we can accurately establish the optimal sequence of tool applications corresponding to each question. In order to sustain a high degree of difficulty in the tasks, we have opted to preserve only those questions which necessitate a minimum of five instances of tool invocation. Through this rigorous selection methodology, we have accrued a dataset consisting of 1,663 questions. Each data entry in the dataset has the following fields:

- Input Question. A natural language utterance that involves intricate KG information seeking.
- Topic Entities. A set of topic entities mentioned in the input question. We obviate the need of performing entity linking, allowing the LLM to focus on long-term planning.
- Action Sequence. The gold action sequence (i.e., tool invocations) that leads to the target answer.
- Gold Answer. The gold answer to the question, typically characterized by a set of KG entities.

Note that, in contrast to interacting with databases in AgentBench, where the particulars and content of the database are integrated into the input, describing an extensive KG to the LLM is not particularly feasible. This task is characterized by a partially observable environment, which is a critical aspect of its nature.

Evaluation Setup. To support our evaluation, we first host the latest version of FREEBASE using Virtuoso.[2](#page-6-0) Due to the complexity of SPARQL queries, we decide not to burden the LLM with crafting SPARQL queries by itself. Instead, we implement a series APIs that interface with the Virtuoso backend, allowing the LLM to query the KG more effortlessly.

We use the first 500 tasks from the datest for evaluation. Each task, when successfully executed, should ideally proceed through the following phases.

<span id="page-6-0"></span><https://github.com/dki-lab/Freebase-Setup>

- Initialization. We prompt the LLM with the concrete task description, including the concrete description of each KG-querying tool that we provide.
- Interaction. During this phase, the LLM is expected to invoke different tools to access the KG and accumulate the necessary information to respond accurately to the question. Importantly, the process is entirely autonomous, meaning the LLM determines the workflow entirely by itself.
- Final Answer Prediction. During its interaction with the KG, the LLM may generate a list of variables, each one representing a unique set of entities. If the LLM determines that one particular variable should signify the final answer, it will present this variable as its output and conclude the task.

Metrics. We use F1 score as the primary evaluation metric in our study, calculated by comparing the model's predicted answers to the gold standard answers. In addition to F1 score, we also use the Exact Match metric. However, unlike previous studies that measure Exact Match based on the logical form, we assess it based on the exact match between the predicted and gold answer sets. Lastly, we also evaluate the Executability of the action sequences generated by the model. If the model's action sequence produces any set of answers when executed, it scores 1.0 for Executability. If it fails to produce an answer, it scores 0.

#### 3.4 Digital Card Game (DCG)

Introduction. Games, especially those requires strategies and planning, could serve as simulated environments for intelligent agent development. Nevertheless, prior text-based games (e.g., TextWorld [\[17\]](#page-16-3)) are largely focused on commonsense grounding. Some recent efforts adopt realworld games (e.g., MineDojo [\[22\]](#page-16-4)) as environments but mostly require multi-modal capabilities beyond existing LLMs.

Digital card game (e.g., *Hearthstone* [\[34\]](#page-17-11)), instead, is an ideal option for text-only LLM evaluation. It usually involves abundant text descriptions for cards, turn-based competition, and thoughtful playing strategies to win, testing a model's understanding of game rules, operating logic, and abilities to form strategic decisions based on current conditions and past experiences in the game. In this work we adapt a simplified DCG system—*Aquawar*[3](#page-7-0)—from the 2021 Tsinghua University Agent Competition (THUAC) hosted by Student Association for Science and Technology in Department of Computer Science and Technology (CST-SAST), for evaluating LLM-as-Agent.

In Aquawar, the agent acts as a player managing a team of fishes with different talents to battle against another team (controlled by our baseline agent) in a turn-based form.

Construction Details. Aquawar has the following rules and configurations:

- Player and Cards. It is a two-player battle game with four pet fishes (i.e., cards) in each team.
- Initial State. Each fish has 400 initial health, 200 attack power, active ability, and passive ability.
- Basic Rule. Players choose a live fish to use its active skill or normal attack on an enemy fish each round. All alive fish's passive ability will automatically trigger when meeting certain conditions.
- Assertion Mechanism. The identity of a player's fish is initially hidden. The counter-player can guess one of the player's fish's identities each round. If the counter-player guesses correctly, the player's fish's identity is revealed, and all its fish will get damaged.
- Round Process. Within a round of the game, the player for that round will first assert the identity of one opponent's fish that are alive and whose identities have not been revealed. If the assertion is correct, all of the opponent's fish that remain alive get damaged. Subsequently, the player for that round can command one alive fish to execute a normal attack or an active ability. Following this, any fish that meet the condition will unleash its passive ability.
- Victory Condition. The victory condition is to have more fish alive at the end of the game.

The original design of Aquawar could be too complicated for existing LLMs. Therefore, we have made the following additional adjustments to cater to LLMs' current performance.

• We limited fish selection to the four types of fish: Spray, Flame, Eel, and Sunfish (See appendix [D](#page-27-0) for details), to reduce game complexity and help models understand the game rules.

<span id="page-7-0"></span><sup>3</sup> <https://www.saiblo.net/>

- To balance agent engagement and game complexity simultaneously, we designed two stages of game logic. We remove the assertions in the first stage while keeping assertions in the second stage. We test all the models on both the first and second stages separately.
- We choose two naive playing strategies (random and greedy search) as the baselines.

Evaluation Setup. For each time of the game playing, we evaluate with the following steps:

- Initialization. We initiated the modified game logic environment, an LLM to be tested, and the baseline game agent under the Ubuntu 20.04 environment.
- Interaction. We place rule descriptions in the isntruction prompt according to different game stages, and the LLM agent interacts and competes strategically with the baseline within the game logic environment. We give the LLM agent five chances to respond in the correct format. It will be immediately deemed defeated if it fails to output legal actions within the given number of attempts. At the same time, we encourage the model to output its reasoning process in CoT.
- Result Calculation. During the Interaction process, we will record the entire game process for battle playback and calculate the game results to obtain the metrics for the task.

Metrics. Our comprehensive evaluation uses metrics that range from basic gameplay elements such as the completion rate (Full Play) and the average number of illegal actions (Try Times) to higher-level strategic indicators like the average number of defeated enemy fish (Takedown), the total damage inflicted (Total DMG), and ultimately the main metric—game-winning rate (Win Rate).

# 3.5 Lateral Thinking Puzzles (LTP [\[5\]](#page-15-4))

#### WARNING: Some dataset contents and descriptions might induce psychological discomfort.

Introduction. Lateral thinking puzzles [\[72\]](#page-19-16), or namely situation puzzles, 海龟汤, is a popular group-playing game around the world. The game usually has a person hosting the puzzle and others guess by asking riddle-related questions. The host can only respond "yes", "no", or "irrelevant". The game is terminated when one of the player recovers the critical plot of the puzzle. Its name derives from the psychological term "lateral thinking" [\[18\]](#page-16-12), which refers to the ability of deducing facts from unconventional perspectives and facilitating the exploration of new ideas.

In this dataset, we first design an accurate LTP host system based on gpt-3.5-turbo for automatic game playing[4](#page-8-0) (Cf. Appendix [E\)](#page-29-0). To assess LLMs' lateral reasoning prowess, we have curated a diverse puzzle dataset from web. Our evaluation of LLMs will revolve around measuring their proximity to factual accuracy and the speed at which they uncover these truths. Through this assessment, we aim to gain insights into the depth and agility of LLMs' lateral reasoning abilities.

Construction Details. Each sample is constructed of a pair of story (a riddle, e.g., *A man walked into a restaurant, ordered a bowl of turtle soup, and after finishing it, he committed suicide. Why did he do that?*) and truth. We categorize samples into four levels of difficulty: easy, medium, hard, and expert. The LTP rules for LLM agent playing are as follows:

- Roles: Roles in LTP evaluation are a host and a solver. The host knows the story and truth, providing the story to the solver, and guiding it to guess out the truth. The solver, played and acted by an LLM, tries to find out the truth by asking questions and synthesizing host's answers.
- Solving Steps: There is a maximum round for each game, for example, 25. The solver needs to propose a question in each round based on known facts. The questions should be the ones that can be answered by "Yes", "No", or "Irrelevant". Host reply to the questions with correct answers. To lower the difficulty for LLM agents, sometimes the host will provides some hints in responses when solvers get trapped in wrong directions of reasoning.
- Game Termination: When the solver thinks it has guessed out the major part of the truth, it can declare the guessed plot to the host. If it is correct, the host will announce the end of the game.

Evaluation Setup. For each pair of story and truth, we evaluate the models with the following steps:

- Initialization. Setting up the LTP host system via local python package installation or web API.
- Interaction. We set up system prompts for LLMs to build their roles of players. LLMs are tested as solvers within the maximum round for each game, if the LLM does not exceed the max token

<span id="page-8-0"></span><sup>4</sup>Try the LTP host system at <https://mkw18-lateralthinkingpuzzle.hf.space/> . Warning: some riddles and answers might induce psychological discomfort.

length. In automatic evaluation, we limit the answer to be mostly "Yes", "No", or "Irrelevant", and extract the answer from gpt-3.5-turbo's responses. LLMs are also asked to summarize their reasoning in automatic evaluation in order to help the termination detection to be more accurate.

• Checking. We do the pilot study of each LLM to collect all situations in game process and design the checking plan. For automatic evaluation, we set up some key words for gpt-3.5-turbo to answer and remind the model to consider some flexible situation like synonyms.

Metrics. We evaluate LLMs' Lateral reasoning ability by two self created metrics:

- Single Game Accuracy (SGA): The proportion of rounds in which LLMs approaching the truth in a single game.
- Round Efficiency (RE): How fast the model can guess out the truth within the maximum round.
- Query Relevance (QR): Relevance between model's questions and the truth.
- Game Progress (GP): Progress before a game end, which serves as the main metric. We break down the groundtruth into several points and measure how many points are reached by an agent.

#### 3.6 House-Holding (Alfworld [\[71\]](#page-19-12))

Introduction. House-holding environments, which require strong commonsense grounding, have been well-established for language agent evaluation [\[17\]](#page-16-3). In AgentBench, we assess the model's capability in accomplishing tasks in physical house-holding environments on the classical ALFWorld [\[71\]](#page-19-12).

Construction Details. The ALFWorld benchmark comprises of textual environments designed to mimic household scenarios, providing an interactive environment where an agent can perform decision-making tasks through text-based interfaces. Given the household environment description and an target instruction, the agent's objective is to break down the complex high-level target into a sequence of straightforward actions. After each step, the agent receives environment feedback, allowing the agent to adapt the plan dynamically and move on to the subsequent task to eventually accomplish the main objective.

Each evaluation sample in ALFWorld dataset encompasses following contents:

- Environment Description. The detailed description of the whole household environment, including agent's initial position and a snapshot of the room containing objects and their IDs.
- Objective. The goal that needs the agent to accomplish in the environment, usually requiring multi-step reasoning and exploring (e.g. put the lamp on the table).
- Simulated Environment. After every action of the agent, the simulated environment gives immediate feedback and evaluates whether the agent has completed the task.

In the dataset, we utilized 134 solvable problems from the ALFWorld *eval out of distribution* split of the dataset. All the problems were categorized into six categories: *pick and place*, *pick clean then place*, *pick heat then place*, *pick cool then place*, *look at obj*, and *pick two obj*.

Evaluation Setup. Due to the inherent complexity of the problem and the high standards required for the output format, we employ a 1-shot evaluation setting. For each category of problem, we use one relatively simple and complete interact processes of the same category from the training set as an example. Following ReAct [\[99\]](#page-21-10), we adopt the few-shot examples and prompts in corresponding repository[5](#page-9-0) . Additionally, if LLM output format is invalid, we use the BLEU metric to assess the similarity of the output to all valid action options. The option with the highest similarity will be chosen as the action of the model for this round.

For each sample, the evaluation process can be divided into 2 parts.

- Initialization. We describe the task to the model and provide one successful example. Afterwards, we elaborate on the environment and delineate the objective required to be accomplished.
- Interaction. The model generates some thoughts and the next action based on the feedback received from previous interactions and the information from the environment. After receiving the action from the model, the environment provides feedback (changes to the environment or information observed by the model). This process is repeated until the model successfully achieves its goal (which is considered a success) or reaches its maximum number of actions (which is considered a failure). It is worth noting that sometimes, after several unsuccessful

<span id="page-9-0"></span><sup>5</sup> <https://github.com/ysymyth/ReAct>

attempts, the model may repeatedly output the same content. To save evaluation time, we judge that if the model outputs identical content three times consecutively, it will be deemed a failure due to repetition.

Metrics. We employ the overall Success Rate as a measure of model performance, that is, the number of tasks successfully completed by the model divided by the total number of tasks.

#### 3.7 Web Shopping (WebShop [\[96\]](#page-21-5))

Introduction. Online shopping has become an important part of modern life. Its trajectory, which comprises searching, viewing, and choosing user desirable items on a real e-commerce website, requires autonomous agents' strong reasoning and decision-making abilities. Webshop [\[96\]](#page-21-5), a simulated online shopping environment, exactly serves such purpose for evaluating language agents. While it is originally evaluated on specifically trained models, we assess LLMs with mere prompting.

Construction Detail. The environment displays the text observation of the webpage and available actions to agents. Agent may freely explore the website and browse through items with clickable buttons just as in the real world. About a million products are scraped from amazon.com to form the database of website. Then each of them is annotated with labels representing its own attribute. 12,087 human instructions are collected and linked with goals along with expected attributes. Please refer to [\[96\]](#page-21-5) for more dataset construction details.

Evaluation Setup. We adopt the first 500 entries of 12,087 instructions as test set (following [\[96\]](#page-21-5)'s official implementation). Each round of interaction can be decomposed as following steps:

- Instructing. After the initial prompt that tells environment information and the format in which LLMs should response, we give instructions about what kind of product we wish to buy.
- Interacting. Agent respond in given format, as prompted, containing their thoughts and the action they wish to take. The actions can be categorized into two types: search and click, corresponding with the actual actions of using search engine and clicking buttons in real world. The environment answers agent's action with a simplified text version of webpage and a list of available buttons. This process repeats until the agent click "buy now" button or round limit is exceeded.
- Calculating reward. We use the reward function in the paper as the metric. The reward is mapping from the similarity of the attributes we are expecting and the attributes that the bought product actually have to a number between 0 and 1.

Metrics. As there might be more than one suitable item for a given query, Webshop adopts a matching reward as its evaluation metric:

$$\mathbf{Reward} = \frac{|U_{att} \cap Y_{att}| + |U_{opt} \cap Y_{opt}| + \mathbb{I}[y_{price} \le u_{price}]}{|U_{att}| + |U_{opt}| + 1} \cdot r_{type} \tag{1}$$

where

$$r_{\rm type} = \begin{cases} 0, & \text{if TextMatch} = 0 \\ 0.1, & \text{if TextMatch} < 0.1 \\ 0.5, & \text{if TextMatch} \le 0.2 \text{ and query not match and category not match} \\ 1, & \text{otherwise} \end{cases} \tag{2}$$

U and Y stand for goal and chosen product, att and opt stand for attributes and options. TextMatch is a text match of pronoun, noun, and proper noun between chosen and goal product title.

#### 3.8 Web Browsing (Mind2Web [\[19\]](#page-16-5))

Introduction. General web environment is an ideal sandbox for training and evaluating intelligent agents. Mind2Web [\[19\]](#page-16-5) is a very recently released general benchmark for developing and assessing web agents capable of executing intricate tasks across various website domains, based on high-level user instructions. It designs feasible actions for website interactions, such as clicking, selecting, and typing, thereby facilitating a comprehensive evaluation of LLMs as web agents.

Construction Details. Mind2Web covers domains of Travel, Information, Sevice, Shopping, and Entertainment, assembled using SimilarWeb ranking as a reference. It hires annotators to first

propose task goals based on the current website, and then record their traces of interaction as expert demonstrations. Our adoption of it primarily focuses on generalization across environments, i.e., the Cross Domain test set which contains 912 tasks from 73 websites, spread among domains including Housing, Job, Social Media, Education, Health, Government, Home Service, etc. Please refer to [\[19\]](#page-16-5) for more dataset construction details. Each task sample encomposses the following contents:

- Task Description. A high-level (instead of step-by-step) goal that can be achieved on the website, such as"*Get the highest rated SAP S/4 HANA course rated 4, and up with a duration between 3 to 6 hours for an intermediate, and add this to your cart and checkout*".
- (Reference) Action Sequence. In the annotated interaction sequence, a meta-action a<sup>t</sup> at step t includes {et, ot}, where e<sup>t</sup> represents the unique backend id of the target element, and o<sup>t</sup> refers to the symbolic action operated on e<sup>t</sup> (i.e., *Click*, *Type*, and *Select Options*). For *Type* and *Select Options*, corresponding textual inputs are also included.
- Webpage Information. A detailed observation of the web browsing environment at each step. Throughout the manual annotation process, each observed step captures a snapshot, incorporating the raw HTML codes from the website as well as the previous interaction trajectory.

It has been found that LLMs consistently face challenges when handling the cumbersome raw HTML code associated with real-world web pages. Therefore, Mind2Web proposes to rank and filter the HTML elements with a small language model, e.g., DeBERTa, to enhance inference efficiency.

Given the user's high-level instruction, the agent continuously interacts with the web system by receiving the observation of the current page content and the action histories, then predicting the next action, which consists of the target element and intended operation.

Evaluation Setup. The evaluation involves a dual process to improve the efficiency following [\[19\]](#page-16-5). A fine-tuned small language model is first employed to rank HTML elements and select top-k potential candidates. Subsequently, we prompt and formulate the element selection as a multi-choice QA problem, providing five candidates for each round. For the *Type* and *Select Options* operations, agents are additionally prompted to specify the argument for the operation, i.e., textual input to type or option to select.

Metrics. For evaluation, as suggested in the original paper, we consider the following metrics:

- Element Accuracy. Calculates the accuracy of the chosen element et.
- Action F1. Determines the token-level matching score for the operation ot. It brings a distinction for Type and Select Option operations due to the existence of text values.
- Success Rate. Evaluates the predicted action correctness compared to reference actions. For Step Success Rate, we grant success if the selected element e<sup>t</sup> is correct and the predicted operation o<sup>t</sup> matches the ground truth value at the step. Likewise, for the Task Success Rate, a task is considered successful only if all the steps have been successful, making it a rigorous measure. Unfortunately, even the best LLMs now can only achieve single-digit task success percentages.

We report Step Success Rate as the main metric showing the independent accuracy of each action step, due to the current struggles for LLMs to ensure overall task success rates. Regarding the experimental setup, we select topk 10 candidates to construct multichoice questions utilizing CoT few-shot prompting. Consequently, the GPT-3.5 results can diverge from the original paper [\[19\]](#page-16-5) under topk of 50 setting and different prompting strategies.

# 4 Evaluation of AgentBench

We extensively evaluate 25 LLMs, including API-based commercial models and open-sourced LLMs, to form a systematic view of the existing performance of LLM-as-Agent. To facilitate related LLM-as-Agent research, we also design and release a simple plug-and-play evaluation toolkit.

#### <span id="page-11-0"></span>4.1 Evaluation Setup

Dataset Statistics. We report the statistics of datasets in AgentBench in Table [2.](#page-12-0) For simplicity, in the following part we use the abbreviation of each dataset. All datasets are practical multi-turn interacting challenges, and their estimated solving turns for each individual problem ranging from 5

<span id="page-12-0"></span>Table 2: Statistics and metrics of 8 environments in AgentBench evaluation. "SR" stands for *Success Rate*. "#Turn" denotes the estimated number of interacting turns to solve a single problem. In "#Dev", "#Test", and "#Extend", we provide the number of query samples and total expected interacting turns. Additionally, "Weight<sup>-1</sup>" refers to the average score for a task across all models in our evaluation. For further clarification, please refer to Section 4.1.

|                      | Operating<br>System | Database    | Knowledge<br>Graph | Digital<br>Card Game | Lateral Think<br>-ing Puzzles | House-<br>holding | Web<br>Shopping | Web<br>Browsing |
|----------------------|---------------------|-------------|--------------------|----------------------|-------------------------------|-------------------|-----------------|-----------------|
| Abbr.                | OS                  | DB          | KG                 | DCG                  | LTP                           | НН                | WS              | WB              |
| #Turn                | 8                   | 5           | 15                 | 30                   | 25                            | 35                | 5               | 10              |
| Metric               | SR                  | SR          | F1                 | Win Rate             | Game Progress                 | SR                | Reward          | Step SR         |
| #Dev                 | 26 / 240            | 60 / 300    | 20 / 300           | 12 / 360             | 40 / 1000                     | 20 / 700          | 80 / 400        | 31 / 400        |
| #Test                | 144 / 1200          | 300 / 1500  | 150 / 2250         | 20 / 600             | 100 / 2500                    | 50 / 1750         | 200 / 1000      | 177 / 1800      |
| #Extend              | -                   | 1599 / 8000 | 500 / 7500         | 200 / 6000           | 200 / 5000                    | 134 / 4690        | 500 / 2500      | 912/9000        |
| Weight <sup>-1</sup> | 11                  | 8           | 10                 | 9                    | 5                             | 10                | 21              | 8               |

to 50. Except OS, we provide three splits for each dataset: Dev, Test, and Extend. The Dev split's all environments, answers, and checking scripts are public, while Test and Extend are kept.

We also carefully balance the evaluation comprehensiveness and efficiency in AgentBench design. One may find the total number of problems in Dev and Test are a bit small—289 and 1,141 respectively. However, considering the multi-turn interaction, to test an LLM requires it to generate for about 4k and 13k times, which would certainly take some time to inference.

**LLMs to Evaluate.** As a systematic attempt to benchmark existing LLMs on LLM-as-Agent, we include in total 25 models for evaluation, which could be roughtly classified into two categories:

- **API-based LLMs:** which mostly consist LLM APIs without disclosed parameter amounts (Cf. Table 1). Due to more investments, their performances are usually better.
- Open-sourced LLMs: which mostly come from the academia and some companies, without public serving APIs (Cf. Table 1). Due to limited computing resources, we only include medium-sized open-sourced LLMs (i.e., no more than 30B) in AgentBench's first edition of evaluation.

**Evaluation Toolkit Design: "API & Docker" Interaction.** As LLM systems become increasingly complicated and only available via APIs, we base the AgentBench evaluation toolkit on the philosophy of API-orientation; in other words, the toolkit is designed to only interact with APIs to alleviate the efforts to adapt and run different LLMs. For LLMs that want to test on AgentBench, they only need to setup a model server with standard format chatting API required by AgentBench.

Besides, because of various complex interaction environments, it is extremely troublesome to configure all environments in a unified environment, and there may be environment conflicts. Therefore, we pack the tasks with complex environments into docker images, researchers can easily use them by mounting the code path and simply starting the evaluation.

**Evaluation Prompt Setup.** To accommodate the majority of existing dialogue models, our dialogue paradigm is structured around two roles (user and agent) engaging and alternating with one another. We record interaction trajectories as a conversation history  $(u_0, a_0, \cdots, u_k, a_k)$  involving user and agent, where  $u_i$ ,  $a_i$  represents the i-th round of the conversation history.

When we perform inference, the conversation history must be like  $(u_0,a_0,\cdots,u_k)$ . We select the minimum r such that count of all tokens<sup>6</sup> in  $(u_0,a_i,u_{i+1},\cdots,u_k)$  is not greater than 3500. And then we append "[NOTICE] 2i messages are omitted." into  $u_0$ . After that, the sequence  $(u_0,a_i,u_{i+1},\cdots,u_k)$  is regarded as final input in multi-turn chat format.

However, in order to take non-chat models into consideration, we append a post-processor. For chat models supporting multiple turns, we simply feed the history into the model. For models supporting only single turn (e.g., wizardcoder-15b), we repeat the single-turn prompt for multiple rounds. For models supporting only text completion (e.g., text-davinci-003), we prepend "USER:" or

<span id="page-12-1"></span><sup>&</sup>lt;sup>6</sup>Because the tokenizers of each model is different, we simply calculate tokens like this: a word with length n occupies  $\lceil n/6 \rceil$  token(s), and a non-blank character takes 1 token.

<span id="page-13-0"></span>Table 3: Test set (standard) results of AgentBench. A clear performance gap exists between top commercial LLMs (e.g., gpt-4) and medium-sized open-sourced competitors. "VER" stands for model version; "OA" stands for the overall AgentBench score derived from a weighted average of all environments (Cf. Section [4.1\)](#page-11-0).

| Type            | Models                 | VER  | OA   | OS   | DB   | KG   | DCG  | LTP  | HH   | WS   | WB   |
|-----------------|------------------------|------|------|------|------|------|------|------|------|------|------|
| API<br>based    | gpt-4                  | 0613 | 4.41 | 36.8 | 33.7 | 52.1 | 50.0 | 17.6 | 78.0 | 58.6 | 22.6 |
|                 | claude                 | v1.3 | 2.77 | 13.2 | 16.8 | 36.2 | 30.0 | 10.4 | 52.0 | 59.3 | 21.0 |
|                 | gpt-3.5-turbo          | 0613 | 2.55 | 32.6 | 15.0 | 27.2 | 30.0 | 14.9 | 14.0 | 67.2 | 15.7 |
|                 | text-davinci-003       | -    | 2.10 | 22.9 | 16.3 | 30.8 | 15.0 | 5.2  | 20.0 | 61.4 | 15.5 |
|                 | claude-instant         | v1.1 | 1.90 | 14.6 | 8.0  | 29.7 | 35.0 | 6.1  | 26.0 | 44.2 | 0.8  |
|                 | text-davinci-002       | -    | 1.46 | 4.9  | 13.7 | 18.9 | 25.0 | 2.3  | 14.0 | 60.1 | 1.1  |
|                 | text-bison-001         | -    | 1.39 | 4.2  | 12.8 | 17.1 | 20.0 | 0.1  | 4.0  | 46.1 | 20.5 |
|                 | chatglm2               | v0.2 | 1.31 | 14.6 | 13.7 | 6.8  | 10.0 | 12.6 | 6.0  | 19.4 | 12.9 |
|                 | openchat-13b           | v3.2 | 1.15 | 9.7  | 5.3  | 6.8  | 0.0  | 9.5  | 8.0  | 50.2 | 14.9 |
|                 | wizardlm-30b           | v1.0 | 0.83 | 14.6 | 12.7 | 2.3  | 10.0 | 4.5  | 6.0  | 10.6 | 3.1  |
|                 | vicuna-13b             | v1.1 | 0.62 | 8.3  | 11.3 | 1.2  | 0.0  | 8.0  | 0.0  | 12.6 | 3.9  |
|                 | wizardlm-13b           | v1.0 | 0.59 | 9.7  | 13.0 | 0.4  | 0.0  | 4.1  | 6.0  | 1.2  | 5.8  |
|                 | llama2-13b-chat        | -    | 0.55 | 10.4 | 4.5  | 3.1  | 0.0  | 3.7  | 2.0  | 3.1  | 11.9 |
|                 | codegeex2-6b           | -    | 0.53 | 12.5 | 6.5  | 6.3  | 0.0  | 1.9  | 0.0  | 11.8 | 5.4  |
| Open<br>Sourced | openchat-8192-13b      | v1   | 0.51 | 10.4 | 2.7  | 0.6  | 10.0 | 0.0  | 4.0  | 6.7  | 7.1  |
|                 | baichuan-13b-chat      | -    | 0.36 | 11.8 | 3.0  | 6.3  | 0.0  | 1.3  | 0.0  | 5.7  | 2.3  |
|                 | koala-13b              | -    | 0.34 | 2.8  | 5.3  | 0.0  | 0.0  | 2.5  | 0.0  | 6.0  | 8.1  |
|                 | llama2-7b-chat         | -    | 0.31 | 10.4 | 2.8  | 1.9  | 0.0  | 2.1  | 0.0  | 2.2  | 3.8  |
|                 | chatglm-6b             | v1.1 | 0.31 | 4.9  | 0.3  | 0.0  | 0.0  | 6.7  | 0.0  | 0.5  | 4.9  |
|                 | vicuna-7b              | v1.1 | 0.24 | 6.2  | 3.3  | 0.0  | 0.0  | 3.1  | 0.0  | 6.4  | 0.2  |
|                 | internlm-chat-7b       | -    | 0.23 | 3.5  | 6.3  | 0.0  | 0.0  | 3.4  | 0.0  | 0.0  | 0.2  |
|                 | baichuan-7b            | -    | 0.22 | 4.2  | 0.0  | 0.5  | 0.0  | 2.3  | 0.0  | 2.8  | 5.8  |
|                 | wizardcoder-15b        | v1.0 | 0.21 | 3.5  | 0.0  | 2.8  | 0.0  | 1.3  | 0.0  | 0.0  | 6.7  |
|                 | dolly-v2-12b           | -    | 0.15 | 0.0  | 0.0  | 0.0  | 0.0  | 3.4  | 0.0  | 0.4  | 4.4  |
|                 | oasst-sft-4-pythia-12b | -    | 0.07 | 2.8  | 0.0  | 0.0  | 0.0  | 1.5  | 0.0  | 0.0  | 0.3  |

<sup>&</sup>quot;AGENT:" into each item in the history and finally append the string "AGENT:" to make models generate agent's content.

On task prompt organization, for most tasks we adapted the format from [\[99\]](#page-21-10) to include both "Thought" (for CoT) and "Action" but in one single-turn. Usually a 1-shot simple CoT demonstration is provided in the task instruction for better output format. To ensure reproducible results, we set temperature=0 (i.e., greedy decoding) in the inference on all tasks except DCG following [\[91\]](#page-21-9).

Overall Score Calculation. We have observed that the score distribution for each task varies significantly, mainly because tasks differ in difficulty levels. As a consequence, the average score becomes heavily influenced by tasks that generally yield higher scores (e.g., Web Shopping), overshadowing those with lower scores. As a result, a direct calculation of the average score would not be suitable in this context.

Our approach to calculating the overall score is by averaging the scores of each task across all the models we evaluate, scaling them to an average of 1 (Cf. Table [2\)](#page-12-0). To standardize and simplify score calculations for future studies, we utilize the reciprocal of the average score of all the tested LLMs in each task as a weight. The total score is then computed as the average value obtained by multiplying the score of each task by its corresponding weight. This method ensures fairness and consistency in evaluation, enabling easier comparisons and analysis in future research.

#### 4.2 Main Results

Overall and dataset-specific scores in AgentBench are reported in Table [3.](#page-13-0) On this challenging benchmark, surprisingly we discover that some top LLMs have equipped strong capabilities in dealing with real-world environment interaction. For example, gpt-4 presents the best performance on 7 out of 8 datasets in AgentBench; on HH, it achieves a success rate of 78%, indicating its practical usability in this scenario. For other API-based LLMs, despite their relatively poorer performance, regardless of tasks most of them can solve at least a few percent of problems.

For open-sourced LLMs, however, it is common for them to fail to solve any problems in some challenging tasks, such as KG, DCG, and HH. Generally, most open-sourced LLMs also perform far poorer than API-based LLMs in AgentBench (Avg. 0.42 v.s. 2.24), and even the most capable open-sourced model openchat-13b-v3.2 presents a clear performance gap to gpt-3.5-turbo. This is in contrast to recent claims that some open-sourced LLMs are comparable to gpt-3.5-turbo and gpt-4.

#### 4.3 Analysis

In the evaluation, we identify several specific challenges for LLMs to deal with agent-formed tasks, including action validity, long context, multi-turn consistency, and the impact of code training.

Action Validity. During our evaluation, we find that the models do not follow instructions at all times. In other words, the expected outputs of the models are not always in the input space that the environments can accept. From the perspective of reinforcement learning, the model may not output a valid action or the action is not in the action space.

We observe several common errors in model outputs. In the first type, the model does not understand the task instructions and does not attempt to output an action. The second type is that the model outputs an action, but it is incorrect or incomplete (e.g., missing parameters), so it is rejected by the environment. Some models also output multiple actions in one turn, which is not allowed. Ensuring valid actions is an area for improvement.

Long Context. We noticed that the context length of some open-sourced models is only 2k tokens, which impacts their performance on these interaction tasks. Some of our tasks have lengthy instructions and feedback that exceed the context capacity, causing the model to ignore potentially useful information. Expanding the context length could improve multi-turn performance.

Multi-turn Consistency. There are tasks with terse responses per turn but many dialogue rounds (e.g., House-Holding). Some models tend to lose track of their role across multi-round dialogues. A common failure is outputting an apology and stating inability to control the environment[7](#page-14-0) , despite being instructed to interact as an agent. Maintaining consistency across turns remains difficult for some models.

Is code training a key for LLM-as-Agent capability? Surprisingly, we find codegeex2-6b-chat significantly outperforms chatglm2-6b on OS, DB, and KG, which are related to coding. However, performance on LTP, which requires logical reasoning, decreases. Meanwhile, wizardcoder underperforms despite code training. We speculate that the single-turn format of its training data hinders its multi-turn aptitude. Overall, code training helps certain abilities but has tradeoffs.

# 5 Related Work

Evaluation of LLMs. The general capabilities of self-supervised [\[47\]](#page-18-13) LLMs [\[11;](#page-16-0) [15;](#page-16-1) [101;](#page-21-1) [66;](#page-19-2) [100;](#page-21-2) [80\]](#page-20-0), especially those chat-aligned ones [\[58;](#page-19-3) [6;](#page-15-5) [56\]](#page-18-2), have refreshed people's impression on deep learning-based systems. As a result, they have significantly transcended the conventional scope of NLP evaluation and become an urgent challenge in recent study. Compared to previous efforts focusing on a subset of specified tasks [\[83;](#page-20-12) [84;](#page-20-13) [25\]](#page-17-12), an increasing number of benchmarks are including broader spectra of tasks and datasets [\[33;](#page-17-13) [44;](#page-18-14) [73\]](#page-19-13) in the evaluation. However, most of them are still limited to traditional tasks and thus fail to evaluate LLMs' open-ended generation, multi-turn interaction, and ability to act as agents.

LLM-as-Agent. In pre-LLM era, text game environments such as TextWorld [\[17\]](#page-16-3), Jericho [\[30\]](#page-17-1), and LIGHT [\[82\]](#page-20-3) are dominant in language agent study which bases on BERT [\[20\]](#page-16-13) and reinforcement learning. With the advent of LLMs, the study of LLM agents begins to thrive [\[35\]](#page-17-2), especially after Chain-of-Thought [\[91\]](#page-21-9) came out. ReAct [\[99\]](#page-21-10) is a pioneer work to combine CoT reasoning and actions in agent tasks. Later, a bunch of advanced reasoning strategies [\[40;](#page-18-15) [70;](#page-19-14) [88;](#page-20-14) [46;](#page-18-16) [98;](#page-21-11) [27\]](#page-17-8) and applications [\[59;](#page-19-7) [64;](#page-19-6) [51;](#page-18-4) [1\]](#page-15-1) for LLM-as-Agent have emerged and arouse much public interest. Nevertheless, they are often only studied on limited datasets and models, without a standard evaluation setting and comprehensive coverage of scenarios. AgentBench presents the first systematic benchmark

<span id="page-14-0"></span><sup>7</sup> e.g., *Apologize for my confusion. As a language model, I'm not able to control an operating system ... but I can provide ...*

for evaluating LLM-as-Agent with a broad coverage of practical challenges and available LLMs. Additionally, it also initiates the idea of adopting agent tasks to measure LLM performance.

Evaluating LLMs in Executive Environments. As LLMs become increasingly capable of real-world challenges, there is also a trend to evaluate them in executive environments rather than static text datasets, which also serves as a key feature of AgentBench. Besides text games (e.g., ALFWorld [\[71\]](#page-19-12)), another main stream of works lies in code execution. APPS [\[32\]](#page-17-14), HumanEval [\[12\]](#page-16-6) and MBPP [\[7\]](#page-15-8) pioneer the effort to evaluate code LLMs for functional correctness instead of text similarity. The paradigm has been later widely recognized and adopted in following works [\[43;](#page-18-17) [102;](#page-21-6) [94;](#page-21-13) [54\]](#page-18-18).

However, previous code evaluation frameworks do not consider multi-turn interactions. A concurrent work InterCode [\[95\]](#page-21-14) releases a framework that allows evaluation of interaction between models and Bash and SQL environments, which are similar to OS and DB tasks in AgentBench. However, it is different from AgentBench in several aspects: 1) It displays no command and environment output histories to agent and thus limits agents' capability. 2) Only model's execution output and the difference of file system are checked, leaving other systems (e.g., network system, user system) unverfiable. 3) Each turn an independent shell is launched, disallowing advanced commands such as variable declaration.

Most importantly, prior efforts on the topic do not present such a diverse array of task environments beyond coding. Few previous works would also have comparably extensive coverage of LLM testing as AgentBench does.

# 6 Conclusion

We present AgentBench, a systematically designed multi-dimensional evolving benchmark for evaluating LLMs as agents. For the first time, we include such a wide array of up to 8 real-world challenges to evaluate LLM agents, and establish a unified testing framework and toolkit for agile evaluation. Each challenge is accompanied with real-time interactive environments for testing, in contrast to traditional static-formed NLP datasets. An extensive study of 25 LLMs, including APIbased and Open-sourced, is carefully conducted in a standard setting. We unveil that while top LLMs are becoming capable of tackling complex real-world missions, for open-sourced competitiors there are still a long way to go. We expect AgentBench to serve as a cornerstone for later study to develop better and more applicable intelligent LLM agents.

# References

- <span id="page-15-1"></span>[1] Agentgpt. *Python. https://github.com/reworkd/AgentGPT*, 2023.
- <span id="page-15-2"></span>[2] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. *arXiv preprint arXiv:2204.01691*, 2022.
- <span id="page-15-6"></span>[3] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. Palm 2 technical report. *arXiv preprint arXiv:2305.10403*, 2023.
- <span id="page-15-3"></span>[4] Anonymous. Knowledge base question answering as tool learning. under review, 2023.
- <span id="page-15-4"></span>[5] Anonymous. Lateral thinking assessment in large language models. under review, 2023.
- <span id="page-15-5"></span>[6] Anthropic. Introducing claude, 2023.
- <span id="page-15-8"></span>[7] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.
- <span id="page-15-0"></span>[8] Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022.
- <span id="page-15-7"></span>[9] K. D. Bollacker, C. Evans, P. K. Paritosh, T. Sturge, and J. Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In J. T. Wang, editor, *Proceedings*

- *of the ACM SIGMOD International Conference on Management of Data, SIGMOD 2008, Vancouver, BC, Canada, June 10-12, 2008*, pages 1247–1250. ACM, 2008.
- <span id="page-16-2"></span>[10] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. A large annotated corpus for learning natural language inference. In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 2015.
- <span id="page-16-0"></span>[11] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, NIPS'20, Red Hook, NY, USA, 2020. Curran Associates Inc.
- <span id="page-16-6"></span>[12] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.
- <span id="page-16-11"></span>[13] W. Chen, H. Zha, Z. Chen, W. Xiong, H. Wang, and W. Y. Wang. HybridQA: A dataset of multi-hop question answering over tabular and textual data. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 1026–1036, Online, Nov. 2020. Association for Computational Linguistics.
- <span id="page-16-8"></span>[14] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. *See https://vicuna.lmsys.org (accessed 14 April 2023)*, 2023.
- <span id="page-16-1"></span>[15] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. *arXiv preprint arXiv:2204.02311*, 2022.
- <span id="page-16-9"></span>[16] M. Conover, M. Hayes, A. Mathur, J. Xie, J. Wan, S. Shah, A. Ghodsi, P. Wendell, M. Zaharia, and R. Xin. Free dolly: Introducing the world's first truly open instruction-tuned llm, 2023.
- <span id="page-16-3"></span>[17] M.-A. Côté, A. Kádár, X. Yuan, B. Kybartas, T. Barnes, E. Fine, J. Moore, M. Hausknecht, L. El Asri, M. Adada, et al. Textworld: A learning environment for text-based games. In *Computer Games: 7th Workshop, CGW 2018, Held in Conjunction with the 27th International Conference on Artificial Intelligence, IJCAI 2018, Stockholm, Sweden, July 13, 2018, Revised Selected Papers 7*, pages 41–75. Springer, 2019.
- <span id="page-16-12"></span>[18] E. De Bono. Lateral thinking. *New York*, page 70, 1970.
- <span id="page-16-5"></span>[19] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su. Mind2web: Towards a generalist agent for the web. *arXiv preprint arXiv:2306.06070*, 2023.
- <span id="page-16-13"></span>[20] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, 2019.
- <span id="page-16-7"></span>[21] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang. Glm: General language model pretraining with autoregressive blank infilling. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 320–335, 2022.
- <span id="page-16-4"></span>[22] L. Fan, G. Wang, Y. Jiang, A. Mandlekar, Y. Yang, H. Zhu, A. Tang, D.-A. Huang, Y. Zhu, and A. Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. *Advances in Neural Information Processing Systems*, 35:18343–18362, 2022.
- <span id="page-16-10"></span>[23] D. Ganguli, L. Lovitt, J. Kernion, A. Askell, Y. Bai, S. Kadavath, B. Mann, E. Perez, N. Schiefer, K. Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. *arXiv preprint arXiv:2209.07858*, 2022.

- <span id="page-17-6"></span>[24] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig. Pal: Program-aided language models. In *International Conference on Machine Learning*, pages 10764–10799. PMLR, 2023.
- <span id="page-17-12"></span>[25] S. Gehrmann, T. Adewumi, K. Aggarwal, P. S. Ammanamanchi, A. Aremu, A. Bosselut, K. R. Chandu, M.-A. Clinciu, D. Das, K. Dhole, et al. The gem benchmark: Natural language generation, its evaluation and metrics. In *Proceedings of the 1st Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021)*, pages 96–120. Association for Computational Linguistics, 2021.
- <span id="page-17-4"></span>[26] X. Geng, A. Gudibande, H. Liu, E. Wallace, P. Abbeel, S. Levine, and D. Song. Koala: A dialogue model for academic research. *Blog post, April*, 1, 2023.
- <span id="page-17-8"></span>[27] Y. Gu, X. Deng, and Y. Su. Don't generate, discriminate: A proposal for grounding language models to real-world environments. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4928–4949, Toronto, Canada, July 2023. Association for Computational Linguistics.
- <span id="page-17-9"></span>[28] Y. Gu, S. Kase, M. Vanni, B. Sadler, P. Liang, X. Yan, and Y. Su. Beyond i.i.d.: Three levels of generalization for question answering on knowledge bases. In *Proceedings of the Web Conference 2021*. ACM, apr 2021.
- <span id="page-17-10"></span>[29] Y. Gu and Y. Su. ArcaneQA: Dynamic program induction and contextualized encoding for knowledge base question answering. In *Proceedings of the 29th International Conference on Computational Linguistics*, pages 1718–1731, Gyeongju, Republic of Korea, Oct. 2022. International Committee on Computational Linguistics.
- <span id="page-17-1"></span>[30] M. Hausknecht, P. Ammanabrolu, M.-A. Côté, and X. Yuan. Interactive fiction games: A colossal adventure. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 7903–7910, 2020.
- <span id="page-17-0"></span>[31] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770– 778, 2016.
- <span id="page-17-14"></span>[32] D. Hendrycks, S. Basart, S. Kadavath, M. Mazeika, A. Arora, E. Guo, C. Burns, S. Puranik, H. He, D. Song, et al. Measuring coding challenge competence with apps. *arXiv preprint arXiv:2105.09938*, 2021.
- <span id="page-17-13"></span>[33] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations*, 2021.
- <span id="page-17-11"></span>[34] A. K. Hoover, J. Togelius, S. Lee, and F. de Mesentier Silva. The many ai challenges of hearthstone. *KI-Künstliche Intelligenz*, 34:33–43, 2020.
- <span id="page-17-2"></span>[35] W. Huang, P. Abbeel, D. Pathak, and I. Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In *International Conference on Machine Learning*, pages 9118–9147. PMLR, 2022.
- <span id="page-17-5"></span>[36] B. Inc. Baichuan 7b & 13b. <https://github.com/baichuan-inc>, 2023.
- <span id="page-17-7"></span>[37] M. Iyyer, W.-t. Yih, and M.-W. Chang. Search-based neural structured learning for sequential question answering. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1821–1831, Vancouver, Canada, July 2017. Association for Computational Linguistics.
- <span id="page-17-3"></span>[38] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, 2017.

- <span id="page-18-1"></span>[39] J. D. M.-W. C. Kenton and L. K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT*, pages 4171–4186, 2019.
- <span id="page-18-15"></span>[40] G. Kim, P. Baldi, and S. McAleer. Language models can solve computer tasks. *arXiv preprint arXiv:2303.17491*, 2023.
- <span id="page-18-6"></span>[41] H. Küttler, N. Nardelli, A. Miller, R. Raileanu, M. Selvatici, E. Grefenstette, and T. Rocktäschel. The nethack learning environment. *Advances in Neural Information Processing Systems*, 33:7671–7684, 2020.
- <span id="page-18-9"></span>[42] LAION. Open-assistant. <https://github.com/LAION-AI/Open-Assistant>, 2023.
- <span id="page-18-17"></span>[43] Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond, T. Eccles, J. Keeling, F. Gimeno, A. Dal Lago, et al. Competition-level code generation with alphacode. *Science*, 378(6624):1092–1097, 2022.
- <span id="page-18-14"></span>[44] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al. Holistic evaluation of language models. *arXiv preprint arXiv:2211.09110*, 2022.
- <span id="page-18-10"></span>[45] X. V. Lin, C. Wang, L. Zettlemoyer, and M. D. Ernst. Nl2bash: A corpus and semantic parser for natural language interface to the linux operating system. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, 2018.
- <span id="page-18-16"></span>[46] B. Liu, Y. Jiang, X. Zhang, Q. Liu, S. Zhang, J. Biswas, and P. Stone. Llm+ p: Empowering large language models with optimal planning proficiency. *arXiv preprint arXiv:2304.11477*, 2023.
- <span id="page-18-13"></span>[47] X. Liu, F. Zhang, Z. Hou, L. Mian, Z. Wang, J. Zhang, and J. Tang. Self-supervised learning: Generative or contrastive. *IEEE transactions on knowledge and data engineering*, 35(1):857– 876, 2021.
- <span id="page-18-8"></span>[48] Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and D. Jiang. Wizardcoder: Empowering code large language models with evol-instruct. *arXiv preprint arXiv:2306.08568*, 2023.
- <span id="page-18-0"></span>[49] P. Maes. Agents that reduce work and information overload. *Commun. ACM*, 37:30–40, 1994.
- <span id="page-18-11"></span>[50] D. Merkel et al. Docker: lightweight linux containers for consistent development and deployment. *Linux j*, 239(2):2, 2014.
- <span id="page-18-4"></span>[51] Y. Nakajima. Babyagi. *Python. https://github. com/yoheinakajima/babyagi*, 2023.
- <span id="page-18-3"></span>[52] R. Nallapati, B. Zhou, C. dos Santos, C. Gulcehre, and B. Xiang. Abstractive text summarization using sequence-to-sequence rnns and beyond. In *Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning*, page 280. Association for Computational Linguistics, 2016.
- <span id="page-18-12"></span>[53] L. Nan, C. Hsieh, Z. Mao, X. V. Lin, N. Verma, R. Zhang, W. Krysci ´ nski, N. Schoelkopf, ´ R. Kong, X. Tang, M. Mutuma, B. Rosand, I. Trindade, R. Bandaru, J. Cunningham, C. Xiong, and D. Radev. Fetaqa: Free-form table question answering, 2021.
- <span id="page-18-18"></span>[54] E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y. Zhou, S. Savarese, and C. Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In *The Eleventh International Conference on Learning Representations*, 2023.
- <span id="page-18-7"></span>[55] OpenAI. Introducing chatgpt, 2022.
- <span id="page-18-2"></span>[56] R. OpenAI. Gpt-4 technical report. *arXiv*, pages 2303–08774, 2023.
- <span id="page-18-5"></span>[57] P. Osborne, H. Nõmm, and A. Freitas. A survey of text games for reinforcement learning informed by natural language. *Transactions of the Association for Computational Linguistics*, 10:873–887, 2022.

- <span id="page-19-3"></span>[58] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022.
- <span id="page-19-7"></span>[59] J. S. Park, J. C. O'Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein. Generative agents: Interactive simulacra of human behavior. *ArXiv*, abs/2304.03442, 2023.
- <span id="page-19-15"></span>[60] P. Pasupat and P. Liang. Compositional semantic parsing on semi-structured tables. In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 1470–1480, Beijing, China, July 2015. Association for Computational Linguistics.
- <span id="page-19-5"></span>[61] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. Squad: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2383–2392, 2016.
- <span id="page-19-1"></span>[62] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever. Zero-shot text-to-image generation. In *International Conference on Machine Learning*, pages 8821–8831. PMLR, 2021.
- <span id="page-19-8"></span>[63] S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-maron, M. Giménez, Y. Sulsky, J. Kay, J. T. Springenberg, et al. A generalist agent. *Transactions on Machine Learning Research*, 2022.
- <span id="page-19-6"></span>[64] T. B. Richards. Auto-gpt: An autonomous gpt-4 experiment, 2023.
- <span id="page-19-4"></span>[65] V. Sanh, A. Webson, C. Raffel, S. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja, M. Dey, et al. Multitask prompted training enables zero-shot task generalization. In *International Conference on Learning Representations*, 2022.
- <span id="page-19-2"></span>[66] T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilic, D. Hesslow, R. Castagné, A. S. Luccioni, ´ F. Yvon, M. Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. *arXiv preprint arXiv:2211.05100*, 2022.
- <span id="page-19-0"></span>[67] J. R. Searle. Speech acts: An essay in the philosophy of language. *Language*, 46:217, 1970.
- <span id="page-19-10"></span>[68] B. Shen, F. Xia, C. Li, R. Martín-Martín, L. Fan, G. Wang, C. Pérez-D'Arpino, S. Buch, S. Srivastava, L. Tchapmi, et al. igibson 1.0: A simulation environment for interactive tasks in large realistic scenes. In *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 7520–7527. IEEE, 2021.
- <span id="page-19-9"></span>[69] T. Shi, A. Karpathy, L. Fan, J. Hernandez, and P. Liang. World of bits: An open-domain platform for web-based agents. In *International Conference on Machine Learning*, pages 3135–3144. PMLR, 2017.
- <span id="page-19-14"></span>[70] N. Shinn, B. Labash, and A. Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. *arXiv preprint arXiv:2303.11366*, 2023.
- <span id="page-19-12"></span>[71] M. Shridhar, X. Yuan, M.-A. Cote, Y. Bisk, A. Trischler, and M. Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In *International Conference on Learning Representations*, 2020.
- <span id="page-19-16"></span>[72] P. Sloane. *Lateral thinking puzzlers*. Sterling Publishing Company, Inc., 1992.
- <span id="page-19-13"></span>[73] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *Transactions on Machine Learning Research*, 2023.
- <span id="page-19-11"></span>[74] S. Srivastava, C. Li, M. Lingelbach, R. Martín-Martín, F. Xia, K. E. Vainio, Z. Lian, C. Gokmen, S. Buch, K. Liu, et al. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments. In *Conference on Robot Learning*, pages 477–490. PMLR, 2022.

- <span id="page-20-1"></span>[75] N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020.
- <span id="page-20-11"></span>[76] Y. Su, H. Sun, B. M. Sadler, M. Srivatsa, I. Gur, Z. Yan, and X. Yan. On generating characteristic-rich question sets for QA evaluation. In J. Su, X. Carreras, and K. Duh, editors, *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, EMNLP 2016, Austin, Texas, USA, November 1-4, 2016*, pages 562–572. The Association for Computational Linguistics, 2016.
- <span id="page-20-10"></span>[77] A. Talmor and J. Berant. The web as a knowledge-base for answering complex questions. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 641–651, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.
- <span id="page-20-5"></span>[78] A. Talmor, J. Herzig, N. Lourie, and J. Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4149–4158, 2019.
- <span id="page-20-7"></span>[79] I. Team. Internlm: A multilingual language model with progressively enhanced capabilities. <https://github.com/InternLM/InternLM>, 2023.
- <span id="page-20-0"></span>[80] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-20-4"></span>[81] D. Toyama, P. Hamel, A. Gergely, G. Comanici, A. Glaese, Z. Ahmed, T. Jackson, S. Mourad, and D. Precup. Androidenv: A reinforcement learning platform for android. *arXiv preprint arXiv:2105.13231*, 2021.
- <span id="page-20-3"></span>[82] J. Urbanek, A. Fan, S. Karamcheti, S. Jain, S. Humeau, E. Dinan, T. Rocktäschel, D. Kiela, A. Szlam, and J. Weston. Learning to speak and act in a fantasy text adventure game. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 673–683, 2019.
- <span id="page-20-12"></span>[83] A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. *Advances in neural information processing systems*, 32, 2019.
- <span id="page-20-13"></span>[84] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In *International Conference on Learning Representations*.
- <span id="page-20-6"></span>[85] G. Wang, S. Cheng, Q. Yu, and C. Liu. OpenChat: Advancing Open-source Language Models with Imperfect Data, 7 2023.
- <span id="page-20-2"></span>[86] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. J. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. *ArXiv*, abs/2305.16291, 2023.
- <span id="page-20-9"></span>[87] X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. In *The Eleventh International Conference on Learning Representations*, 2023.
- <span id="page-20-14"></span>[88] Z. Wang, S. Cai, A. Liu, X. Ma, and Y. Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. *arXiv preprint arXiv:2302.01560*, 2023.
- <span id="page-20-8"></span>[89] A. Wei, N. Haghtalab, and J. Steinhardt. Jailbroken: How does llm safety training fail? *arXiv preprint arXiv:2307.02483*, 2023.

- <span id="page-21-3"></span>[90] J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners. In *International Conference on Learning Representations*, 2022.
- <span id="page-21-9"></span>[91] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35:24824–24837, 2022.
- <span id="page-21-0"></span>[92] M. Wooldridge and N. R. Jennings. Intelligent agents: Theory and practice. *The knowledge engineering review*, 10(2):115–152, 1995.
- <span id="page-21-7"></span>[93] C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, and D. Jiang. Wizardlm: Empowering large language models to follow complex instructions. *arXiv preprint arXiv:2304.12244*, 2023.
- <span id="page-21-13"></span>[94] F. F. Xu, U. Alon, G. Neubig, and V. J. Hellendoorn. A systematic evaluation of large language models of code. In *Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming*, pages 1–10, 2022.
- <span id="page-21-14"></span>[95] J. Yang, A. Prabhakar, K. Narasimhan, and S. Yao. Intercode: Standardizing and benchmarking interactive coding with execution feedback. *arXiv preprint arXiv:2306.14898*, 2023.
- <span id="page-21-5"></span>[96] S. Yao, H. Chen, J. Yang, and K. Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. *Advances in Neural Information Processing Systems*, 35:20744–20757, 2022.
- <span id="page-21-8"></span>[97] S. Yao, R. Rao, M. Hausknecht, and K. Narasimhan. Keep calm and explore: Language models for action generation in text-based games. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 8736–8754, 2020.
- <span id="page-21-11"></span>[98] S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. *arXiv preprint arXiv:2305.10601*, 2023.
- <span id="page-21-10"></span>[99] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. In *The Eleventh International Conference on Learning Representations*, 2023.
- <span id="page-21-2"></span>[100] A. Zeng, X. Liu, Z. Du, Z. Wang, H. Lai, M. Ding, Z. Yang, Y. Xu, W. Zheng, X. Xia, et al. Glm-130b: An open bilingual pre-trained model. *arXiv preprint arXiv:2210.02414*, 2022.
- <span id="page-21-1"></span>[101] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*, 2022.
- <span id="page-21-6"></span>[102] Q. Zheng, X. Xia, X. Zou, Y. Dong, S. Wang, Y. Xue, Z. Wang, L. Shen, A. Wang, Y. Li, et al. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. *arXiv preprint arXiv:2303.17568*, 2023.
- <span id="page-21-12"></span>[103] V. Zhong, C. Xiong, and R. Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. *CoRR*, abs/1709.00103, 2017.
- <span id="page-21-4"></span>[104] X. Zhu, Y. Chen, H. Tian, C. Tao, W. Su, C. Yang, G. Huang, B. Li, L. Lu, X. Wang, Y. Qiao, Z. Zhang, and J. Dai. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. *ArXiv*, abs/2305.17144, 2023.

# Part I

# Appendix

# Table of Contents

| A | Operating System                             | 23       |
|---|----------------------------------------------|----------|
|   | A.1<br>Actions<br>A.2<br>Prompt Example      | 23<br>23 |
| B | Database                                     | 25       |
|   | B.1<br>Data Augmentation<br>                 | 25       |
|   | B.2<br>Prompt Example                        | 25       |
| C | Knowledge Graph                              | 26       |
|   | C.1<br>Prompt Example                        | 26       |
| D | Digital Card Game                            | 28       |
|   | D.1<br>The Attributes of Fish                | 28       |
|   | D.2<br>Prompt Example.<br>                   | 29       |
| E | Lateral Thinking Puzzles                     | 30       |
|   | E.1<br>Evaluation on LTP System              | 30       |
|   | E.2<br>LTP Game Progress and Termination<br> | 30       |
|   | E.3<br>Prompt Example                        | 31       |
| F | House-holding                                | 34       |
|   | F.1<br>Prompt Example                        | 34       |
|   | G Web Shopping                               | 35       |
|   | G.1<br>Prompt Example                        | 35       |
|   | H Web Browsing                               | 37       |
|   | H.1<br>Prompt Example.<br>                   | 37       |

# <span id="page-22-0"></span>A Operating System

# <span id="page-22-1"></span>A.1 Actions

In OS evaluation, we design two major types of actions: *bash* and *commit*.

- Bash: which launches a bash command (using textual inputs in *content* field)
- Commit: which announces the completion of the goal. If the task is a QA problem, then the agent should submit the final answer in *content* field; else the checking pipeline will automatically check the system status to judge the correctness.

#### <span id="page-22-2"></span>A.2 Prompt Example

A prompt for OS evaluation consists of the instruction and the formulation of interaction trajectory. An example of instruction prompt is:

You are an assistant that will act like a person , I ' will play the role of linux ( ubuntu ) operating system . Your goal is to implement the operations required by me or answer to the question proposed by me . For each of your turn , you should first think what you should do , and then take exact one of the three actions : " bash " , " finish " or " answer ".

1. If you think you should execute some bash code , take bash action , and you should print like this : Think : put your thought here . Act : bash ''' bash # put your bash code here ''' 2. If you think you have finished the task , take finish action , and you should print like this : Think : put your thought here . Act : finish 3. If you think you have got the answer to the question , take answer action , and you should print like this : Think : put your thought here . Act : answer ( Your answer to the question should be put in this pair of parentheses ) If the output is too long , I will truncate it . The truncated output is not complete . You have to deal with the truncating problem by yourself . Attention , your bash code should not contain any input

The trajectory is organized in CoT styles, and we use an 1-shot example to make model better understand the action space like the following.

actions in each turn .

operation . Once again , you should take only exact one of the three

```
< USER >:
    {{ INSTRUCTION }}
    Now , my problem is :
    Tell me how many files are in the directory "/ etc "?
< AGENT >:
    Think : To count the files in / etc , I need to print all the files
        in it .
    Act : bash
    ''' bash
    ls / etc
    '''
< USER >:
    The output of the OS :
    cpi cron . hourly fuse . conf iproute2 lvm networkd - dispatcher
        protocols selinux tmpfiles . d [ truncated because the output is
        too long ]
< AGENT >:
    Think : The output has been truncated because it is too long , so I
        need to count files by script directly .
    Act : bash
```

```
''' bash
    ls -1 / etc | wc -l
    '''
< USER >:
    The output of the OS :
    220
< AGENT >:
    Think : Now I get the answer , it is 220.
    Act : answer (220)
< USER >:
    Now , I will start a new problem in a new OS . My problem is :
    {{ PROBLEM }}
< LOOP >:
    < AGENT >:
         {{ RESPONSE }}
    Swicth AGENT . Action :
         " bash ":
              < User >:
                  The output of the OS :
                  {{ OUTPUT }}
         " answer " or " finish ":
              Executing Checking Script
              Exit Loop
         otherwise :
              Exit Loop
```

# <span id="page-24-0"></span>B Database

#### <span id="page-24-1"></span>B.1 Data Augmentation

We elaborate on the data augmentation of three types of DB tasks based on the existing SQL datasets [\[103;](#page-21-12) [60;](#page-19-15) [37;](#page-17-7) [13;](#page-16-11) [53\]](#page-18-12), which are all QA problems without some common operations including inserting and updating. We first tested the validity of the raw data and then randomly sample from each category from filtered data to form the final dataset. We adopt gpt-3.5-turbo to enrich and rewrite the original instructions.

- Insert: Given the name, the header information, and the original rows of a table, we generate 5 SQL statements for insertion. Later we rephrase the sentences without changing their meaning (using shorter or longer expressions or changing the order).
- Update: Given the name, the header information, and the previously generated 5 SQL statements for insertion, we generate 5 SQL statements for modification based on the given statements. We rephrase the sentences following the above standard.

To ensure data quality, each augmented query statement are required to pass the unit test scripts.

The *query* type of tasks fall into the traditional scope of Text-to-SQL evaluation, and we only sample and categorize for evaluation. Each query statement in existing datasets is classified into following types: 'Counting', 'Aggregation-MIN', 'Aggregation-MAX', 'Aggregation-AVG', 'Aggregation-SUM', 'Ranking', or 'Comparison'. Each one can only belong to one type. The remaining will be categorized as "Other".

#### <span id="page-24-2"></span>B.2 Prompt Example

We use the following format of prompts:

User :

```
I will ask you a question , then you should help me operate a MySQL
   database with SQL to answer the question .
```

You have to explain the problem and your solution to me and write down your thoughts .

After thinking and explaining thoroughly , every round you can choose to operate or to answer .

your operation should be like this :

Action : Operation

''' sql

SELECT \* FROM table WHERE condition ;

'''

You MUST put SQL in markdown format without any other comments . Your SQL should be in one line .

Every time you can only execute one SQL statement . I will only execute the statement in the first SQL code block . Every time you write a SQL , I will execute it for you and give you the output .

If you are done operating , and you want to commit your final answer , then write down :

Action : Answer

Final Answer : [" ANSWER1 " , " ANSWER2 " , ...]

DO NOT write this pattern unless you are sure about your answer . I expect an accurate and correct answer .

Your answer should be accurate . Your answer must be exactly the same as the correct answer .

If the question is about modifying the database , then after done operation , your answer field can be anything .

If your response cannot match any pattern I mentioned earlier , you will be judged as FAIL immediately .

Your input will be raw MySQL response , you have to deal with it by yourself .

# <span id="page-25-0"></span>C Knowledge Graph

#### <span id="page-25-1"></span>C.1 Prompt Example

Task description:

User :

You are an agent that answers questions based on the knowledge stored in a knowledge base . To achieve this , you can use the following tools to query the KB .

1. get\_relations ( variable : var ) -> list of relations

A variable can be either an entity or a set of entities ( i . e . , the result of a previous query ) . This function helps to navigate all relations in the KB connected to the variable , so you can decide which relation is the most useful to find the answer to the question .

A simple use case can be ' get\_relations ( Barack Obama ) ', which finds all relations / edges starting from the entity Barack Obama .

The argument of get\_relations should always be an entity or a variable ( e . g . , #0) and not anything else .

2. get\_neighbors ( variable : var , relation : str ) -> variable Given a variable , this function returns all entities connected to the variable via the given relation . Note that , get\_neighbors () can only be used after get\_relations () is used to find a set of viable relations .

A simple use case can be ' get\_neighbors ( Barack Obama , people . person . profession ) ', which returns the profession of Obama in Freebase .

3. intersection ( variable1 : var , variable2 : var ) -> variable Given two variables , this function returns the intersection of the two variables . The two variables MUST be of the same type !

- 4. get\_attributes ( variable : var ) -> list of attributes This function helps to find all numerical attributes of the variable . Please only use it if the question seeks for a superlative accumulation ( i . e . , argmax or argmin ) .
- 5. argmax ( variable : var , attribute : str ) -> variable Given a variable , this function returns the entity with the maximum value of the given attribute . It can only be used after get\_attributes () is used to find a set of viable attributes .
- A simple use case can be ' argmax ( variable , age ) ', which returns the oldest entity belonging to the variable .
- 6. argmin ( variable : var , attribute : str ) -> variable Given a variable , this function returns the entity with the minimum value of the given attribute . It can only be used after get\_attributes () is used to find a set of viable attributes . A simple use case can be ' argmin ( variable , age ) ', which returns the youngest entity belonging to the variable .
- 7. count ( variable : var ) -> int Given a variable , this function returns the number of entities belonging to the variable .
- After a variable is produced along the process , you need to judge whether a variable is the final answer to the question . Each variable is represented as an id starting from 0. For example , #0 is the first variable , #1 is the second variable , and so on . Once you find the answer , respond with ' Final Answer : # id ' , where id is the id of the variable that you think is the final answer . For example , if you think #3 is the final answer , you MUST respond with ' Final Answer : #3 '.
- You can only take ONE action at a time !! After you get the observation from its execution , you can take another action . You can take at most 15 actions to find the answer to the question .

Given the inherent complexity associated with enabling LLMs to query the KB, it has been observed that, in a zero-shot setting, LLMs struggle to generate any outputs of substantive relevance. As a result, we additionally provide a teaching example in our prompt:

Here is a demo that you can use to get started .

Question : the bipropellant rocket engine with kerosene and gas generator cycle is designed by who ? Entities : [ Gas - generator cycle , Kerosene ]

Thought : I need to first find engines with gas - generator cycle . To acheive this , I will query the KB to find relations connected to the entity 'Gas - generator cycle ' and see if any of them can help me find the answer .

Action : get\_relations ( Gas - generator cycle )

- Observation : [ spaceflight . rocket\_engine\_cycle . rocket\_engines , spaceflight . satellite . orbiting , spaceflight . rocket\_stage . fuel , spaceflight . satellite\_manufacturer . spacecraft\_manufactured , automotive . fuel . engines ]
- Thought : From the above relations , I might use spaceflight . rocket\_engine\_cycle . rocket\_engines to find the engines of Gas generator cycle .
- Action : get\_neighbors ( Gas generator cycle , spaceflight . rocket\_engine\_cycle . rocket\_engines )

```
Observation : variable #0 , which are instances of spaceflight .
   bipropellant_rocket_engine
Thought : I also need to find engines with kerosene . To acheive this , I
    will query the KB to find relations connected to the entity '
   Kerosene '.
Action : get_relations ( Kerosene )
Observation : [ spaceflight . satellite_manufacturer .
   spacecraft_manufactured , automotive . fuel . engines , spaceflight .
   rocket_engine_cycle . rocket_engines ]
Thought : From the above relations , I might use spaceflight .
   rocket_engine_cycle . rocket_engines to find engines with kerosene
   cycle .
Action : get_neighbors ( Kerosene , spaceflight . rocket_engine_cycle .
   rocket_engines )
Observation : variable #1 , which are instances of spaceflight .
   bipropellant_rocket_engine
Thought : The engines with both gas - generator cycle and kerosene cycle
   shoule be the intersection of variable #0 and variable #1.
Action : intersection (#0 , #1)
Observation : variable #2 , which are instances of spaceflight .
   bipropellant_rocket_engine
Thought : Now I need to find who designed such engines , which will be
   the final answer . To acheive this , I will query the KB to find
   relations connected to the variable #2.
Action : get_relations (#2)
Observation : [ spaceflight . rocket_engine . manufactured_by , spaceflight .
   rocket_engine . designed_by , spaceflight . rocket_engine . design_period
   , spaceflight . rocket_engine . status ]
Thought : From the above relations , I might use spaceflight .
   rocket_engine . designed_by to find the designer of the engines .
Action : get_neighbors (#2 , spaceflight . rocket_engine . designed_by )
Observation : variable #3 , which are instances of pspaceflight .
   rocket_engine_designer
Thought : #3 is the final answer to the question , which represents the
   target rocket engine designer .
Final Answer : #3
```

# <span id="page-27-0"></span>D Digital Card Game

#### <span id="page-27-1"></span>D.1 The Attributes of Fish

In order to simplify the game rules, we limit the types of pet fish to the following four types:

• Spray

Counter (Passive): Inflicts 30 damage to the attacker when a teammate's health is below 30%

AOE (Active): Attacks all enemies for 35% of its attack points.

#### • Flame

Counter (Passive): Inflicts 30 damage to the attacker when a teammate's health is below 30% Infight (Active): Inflicts 75 damage on one living teammate and increases your attack points by 140.

#### • Eel

Deflect (Passive): Distributes 70% damage to teammates and takes 30% when attacked. Gains 40 attack points after taking 200 damage accumulated.

AOE (Active): Attacks all enemies for 35% of its attack points.

#### • Sunfish

Deflect (Passive): Distributes 70% damage to teammates and takes 30% when attacked. Gains 40 attack points after taking 200 damage accumulated.

Infight (Active): Inflicts 75 damage on one living teammate and increases your attack points by 140.

As can be seen, there is overlap among the active and passive skills of different pet fish, which is done to better conceal the identity information of pet fish in the game and increase the strategic aspects of the game.

#### <span id="page-28-0"></span>D.2 Prompt Example.

We use the following format of prompts for actions:

```
This is a two - player battle game with four pet fish on each team .
Each fish has its 400 initial health , 200 attack power , active ability
   , and passive ability .
You can choose a live fish to use its active skill or normal attack on
    an enemy fish each round .
When the conditions are met , the fish ' s passive ability will
   automatically trigger , regardless of whether it is chosen .
The victory condition is to have more fish alive at the end of the
   game .
The following are the four types of the pet fish :
{ ' spray ': { ' passive ': \" Counter : Deals 30 damage to attacker when a
   teammate ' s health is below 30%\" , ' active ': ' AOE : Attacks all
   enemies for 35% of its attack points . '} , ' flame ': { ' passive ': \"
   Counter : Deals 30 damage to attacker when a teammate ' s health is
   below 30%. \" , ' active ': \" Infight : Attacks one alive teammate for
    75 damage and increases your own attack points by 140. Notice !
   You can ' t attack yourself or dead teamate !\"} , 'eel ': { ' passive ':
   ' Deflect : Distributes 70% damage to teammates and takes 30% when
   attacked . Gains 40 attack points after taking 200 damage
   accumulated . ', ' active ': ' AOE : Attacks all enemies for 35% of
   your attack points . '} , ' sunfish ': { ' passive ': ' Deflect :
   Distributes 70% damage to teammates and takes 30% when attacked .
   Gains 40 attack points after taking 200 damage accumulated . ', '
   active ': \" Infight : Attacks one alive teammate for 75 damage and
   increases your own attack points by 140. Notice ! You can ' t attack
   yourself or dead teamate !\"}}
Play the game with me . In each round , you should output your thinking
   process , and return your move with following json format :
{ ' pick_fish ': ' pick an alive fish ' , ' action ': ' choose from [ normal ,
   active ] ' , ' target_position ': \" target 's position , you must choose
   from [0 ,3]\"}
Notice ! You must return your move in each round . Otherwise , you will
   be considered defeated .
```

We use the following format of prompts for assertions in stage2:

```
This is a two - player battle game with four pet fish in each team .
Each fish has its initial health , attack power , active ability , and
   passive ability .
All fish \ ' s identities are initially hidden . You should guess one of
   the enemy fish \ ' s identities in each round . If you guess right ,
   the enemy fish \ ' s identity is revealed , and each of the enemy ' s
   fish will get 50 damage . You can only guess the identity of the
   live fish .
The victory condition is to have more fish alive at the end of the
   game .
The following are the four types of the pet fish :
{ ' spray ': { ' passive ': \" Counter : Deals 30 damage to attacker when a
   teammate ' s health is below 30%\" , ' active ': ' AOE : Attacks all
   enemies for 35% of its attack points . '} , ' flame ': { ' passive ': \"
   Counter : Deals 30 damage to attacker when a teammate ' s health is
   below 30%. \" , ' active ': \" Infight : Attacks one alive teammate for
    75 damage and increases your own attack points by 140. Notice !
   You can ' t attack yourself or dead teamate !\"} , 'eel ': { ' passive ':
   ' Deflect : Distributes 70% damage to teammates and takes 30% when
   attacked . Gains 40 attack points after taking 200 damage
   accumulated . ', ' active ': ' AOE : Attacks all enemies for 35% of
   your attack points . '} , ' sunfish ': { ' passive ': ' Deflect :
   Distributes 70% damage to teammates and takes 30% when attacked .
   Gains 40 attack points after taking 200 damage accumulated . ', '
   active ': \" Infight : Attacks one alive teammate for 75 damage and
   increases your own attack points by 140. Notice ! You can ' t attack
   yourself or dead teamate !\"}}
Play the game with me . In each round , you should output your thinking
   process , and return your move with following json format :
{\" guess_type \": \" the enemy ' s fish type you may guess \" , \"
   target_position \": \" guess target ' s position , you must choose from
    [0 ,3]\"}
Notice ! You must return your move in each round . Otherwise , you will
   be considered defeated .
```

# <span id="page-29-0"></span>E Lateral Thinking Puzzles

#### <span id="page-29-1"></span>E.1 Evaluation on LTP System

We evaluate the LTP System by human validation, validating system's accuracy on milestone recognition and fact verification. We compare the Single Game Accuracy and Query Relevance between automatic evaluation and human evaluation, and found that automatic evaluation sometimes more tolerate for the agent, which make SGA and QR seem better than human evaluation, especially on open-sourced models. We plan to train a model specifically for the host of the game, in order to provide a better game experience and a more precise evaluation. For Game Progress and Round Efficiency, the LTP system provides an objective evaluation, which can match the level of human evaluation.

#### <span id="page-29-2"></span>E.2 LTP Game Progress and Termination

The progress of game is defined as the proportion of hit key points in the truth. The key points are summarized by gpt-3.5-turbo, which are concluded in the dataset as "answer\_keys" (e.g., key points of the truth "*That night they went to the abandoned building to record the number of steps. They verified what was said on the Internet, and there would be one step less when counting the stairs at night. However, when they went to the abandoned building for verification the next day, they found that there were no stairs at all.*" is "*1. They want to count the steps of the abandoned building. 2. A supernatural event occurred. 3. They saw a claim online: counting stairs at night will result in one step less. 4. The next day, when they went to the abandoned building to verify, they found no stairs. 5.* *They broke down because they were terrified.*"). The number of key points varies among samples. As for the decision of whether the agent guess out key points, we first change relevant questions into declarative sentences, then simplify sentences into one sentence. After guessing out a key point, delete that key point and relevant inferences to avoid repetition. The termination of game is defined as guessing out more than 50% of key points.

#### <span id="page-30-0"></span>E.3 Prompt Example

We use the following format of prompts for agents:

You are a game player , and you are playing Lateral Thinking Puzzle , also known as Situation Puzzle .

Lateral Thinking Puzzle is a deductive reasoning game , and here are the game rules :

- 1. At the beginning of the game , you will receive a narrative , referred to as " story ". Based on the story , you need to ask questions that can be answered with " yes ", " no " , or " irrelevant " to guees out the " truth ".
- 2. By asking questions , you narrow down the range of possibilities until you eventually guess out the truth .
- 3. Each time , you can only ask one question .
- 4. Remember that your role is a player . You cannot declare the end of the game , give up on reasoning , or request a new game .
- 5. You cannot directly repeat information already provided in the story .
- 6. You cannot directly ask for details about the story in the form of " why " questions ; you need to make your own guesses for truth .
- 7. You cannot directly inquire about the story ; you must make your own deductions .

Next , please make full use of the information provided above to engage in game reasoning . Keep in mind that your questions should be answerable with " yes " , " no " , or " irrelevant " , and you can only ask one question at a time .

Here is your story : { story }

You can start guessing the content of the truth , and I will answer your questions . Please note that your questions should be answerable with " yes " , " no " , or " irrelevant ".

We use the following format of prompts for host:

# USER :

I need you to be the host of a game called Lateral Thinking Puzzle .

Lateral Thinking Puzzle is a game consist of a story and a truth . Your story is : '{ story } ' Your truth is : '{ answer } '

Here are the game rules :

- 1. You know both the " story " and the " truth ". When a user wants to play Lateral Thinking Puzzle , you provide them with the " story ". The user only knows the " story " and is unawared of the " truth ".
- 2. The user asks questions that can be answered with " yes ," " no ," or " irrelevant ". Their questions are aimed at guessing the " truth ". Based on the " truth " , you respond to the user ' s questions using " yes ," " no ," or " irrelevant " to guide them towards guessing the correct truth .
- 3. If the user directly asks for details about the truth using the form of " why " questions , inform them that they need to make their own guesses .
- 4. You must fully understand and accurately interpret the information from the truth . Based on the information of the truth and the user ' s past questions , you answer the user ' s questions . The user ' s

- questions may not necessarily contain information from the truth , but your responses must align with the facts of the truth .
- 5. You can only answer " irrelevant " when the truth cannot provide a direct or indirect answer . Note that this is the only condition for responding " irrelevant "; otherwise , you should answer " yes " or " no ."
- 6. You cannot directly disclose the information from the truth to the user , even if they ask directly .
- 7. You need to judge the user ' s questions as a whole and understand their overall intent . Avoid answering based solely on a particular point ; your responses must align with the facts of the truth .
- 8. During the user ' s process of guessing the truth , if they come close to some truths but still have gaps in understanding the complete truth of the truth , you can provide certain entry point hints . However , you cannot directly reveal information from the truth .

During the game process , please adhere to the above game rules to ensure a positive gaming experience for the users . Pay close attention to the questions asked and ensure that your responses comply with both the game rules and the information from the truth . When a user requests to play the game , provide them with the story and help them guess the truth by answering with " yes " , " no " , or " irrelevant ". Remember that with each response , you must fully understand and abide by the aforementioned game rules , as well as the story and the truth . This will ensure a smooth user experience and avoid situations where you cannot answer or violate the game rules .

#### ASSISTANT :

Alright , I understand that my role is to be the host of the Lateral Thinking Puzzle and help users guess the truth by answering their questions . I have fully grasped all the information regarding the story and the truth and have carefully read all the rules . I assure that I will abide by all the rules throughout the game process .

### USER :

Please summarize the key points of the story to ensure that you have understood it .

#### ASSISTANT : { story\_key }

# USER :

Please summarize the key points of the truth to ensure that you have understood it .

#### ASSISTANT : { answer\_key }

# USER :

Please restate the rules to ensure that you have understood all of them .

#### ASSISTANT :

- 1. You know both the " story " and the " truth ". When a user wants to play Lateral Thinking Puzzle , you provide them with the " story ". The user only knows the " story " and is unawared of the " truth ".
- 2. The user asks questions that can be answered with " yes ," " no ," or " irrelevant ". Their questions are aimed at guessing the " truth ". Based on the " truth " , you respond to the user ' s questions using " yes ," " no ," or " irrelevant " to guide them towards guessing the correct truth .

- 3. If the user directly asks for details about the truth using the form of " why " questions , inform them that they need to make their own guesses .
- 4. You must fully understand and accurately interpret the information from the truth . Based on the information of the truth and the user ' s past questions , you answer the user ' s questions . The user ' s questions may not necessarily contain information from the truth , but your responses must align with the facts of the truth .
- 5. You can only answer " irrelevant " when the truth cannot provide a direct or indirect answer . Note that this is the only condition for responding " irrelevant "; otherwise , you should answer " yes " or " no ."
- 6. You cannot directly disclose the information from the truth to the user , even if they ask directly .
- 7. You need to judge the user ' s questions as a whole and understand their overall intent . Avoid answering based solely on a particular point ; your responses must align with the facts of the truth .
- 8. During the user ' s process of guessing the truth , if they come close to some truths but still have gaps in understanding the complete truth of the truth , you can provide certain entry point hints . However , you cannot directly reveal information from the truth .

#### USER :

Alright , we can now start the game . Remember , before each response , you should review the key points of the story , the key points of the truth , and the rules . Answer with " yes " , " no " , or " irrelevant ".

#### ASSISTANT :

Alright , as the host of the game , I will adhere to the above rules and ensure that my responses comply with the rules and the information from the truth . Below is your story : { story }

You can start guessing the content of the truth , and I will answer your questions . Please note that your questions should be answerable with " yes " , " no " , or " irrelevant ".

#### USER :

{ question }

Please answer with " yes " , " no " , or " irrelevant ".

Here is the prompt to convert questions answered by "Yes" into declarative sentence.

Please restate the following content as a declarative sentence and simplify it into one sentence : { question }

Here is the prompt to convert questions answered by "No" into declarative sentence.

Please restate the following content as a declarative sentence by using the opposite meaning and then simplify it into one sentence : { question }

Here is the prompt to merge reasoned out information into one sentence to judge whether the agent guess out the key points:

Please simplify the following content into one sentence : { reasoning }

Here is the prompt to judge whether the merged sentence hit the key point.

Please compare the information between Sentence 1 and Sentence 2 to determine if Sentence 2 contains all the information in Sentence 1 , including key details and descriptions . Please answer with " yes " or " no ".

```
Sentence 1: { key }
Sentence 2: { merged sentence }"}
```

# <span id="page-33-0"></span>F House-holding

#### <span id="page-33-1"></span>F.1 Prompt Example

To align the output format with the legal commands supported by the simulated environment, we adopted a 1-shot evaluation setup where one successfully completed task example was concatenated after the instruction. At the beginning of the interaction, we describe the task to the model using the following instruction.

Interact with a household to solve a task . Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal . At the beginning of your interactions , you will be given the detailed description of the current environment and your goal to accomplish . For each of your turn , you will be given a list of actions which you can choose one to perform in this turn . You should choose from two actions : \" THOUGHT \" or \" ACTION \". If you choose \" THOUGHT \" , you should first think about the current condition and plan for your future actions , and then output your action in this turn . Your output must strictly follow this format :\" THOUGHT : your thoughts .\ n ACTION : your next action \ n \"; If you choose \" ACTION \" , you should directly output the action in this turn . Your output must strictly follow this format :\" ACTION : your next action \ n \". After your each turn , the environment will give you immediate feedback based on which you plan your next few steps . if the envrionment output \" Nothing happened \" , that means the previous action is invalid and you should try more options . Reminder : 1. the action must be chosen from the given available actions . Any actions except provided available actions will be regarded as illegal . 2. Think when necessary , try to act directly more in the process .

All the tasks in the datasets are categorized into six classes. To better guide the model in accomplishing the objectives, we have selected one relatively simple example of successful completion of similar tasks for each category as 1-shot example. Here is an example:

```
User : You are in the middle of a room . Looking quickly around you , you
    see a cabinet 4 , a cabinet 3 , a cabinet 2 , a cabinet 1 , a
   countertop 1 , a garbagecan 1 , a handtowelholder 2 , a
   handtowelholder 1 , a sinkbasin 2 , a sinkbasin 1 , a toilet 1 , a
   toiletpaperhanger 1 , and a towelholder 1. Your task is to : put
   some spraybottle on toilet .
Agent : THOUGHT : The task is to put some spraybottle on toilet , so
   first i need to search the room for some spraybottle . After
   finding the spraybottle , i need to take it to the toilet . I wil
   start my search on cabinets .\ n ACTION : go to cabinet 1
User : On the cabinet 1 , you see a cloth 1 , a soapbar 1 , a soapbottle
   1.
Agent : ACTION : go to cabinet 2
User : The cabinet 2 is closed .
Agent : ACTION : open cabinet 2
User : You open the cabinet 2. The cabinet 2 is open . In it , you see a
   candle 1 , and a spraybottle 2.
Agent : ACTION : take spraybottle 2 from cabinet 2
User : You pick up the spraybottle 2 from the cabinet 2.
Agent : ACTION : go to toilet 1
User : On the toilet 1 , you see a soapbottle 2.
Agent : ACTION : put spraybottle 2 in / on toilet 1
User : You put the spraybottle 2 in / on the toilet 1.
```

# <span id="page-34-0"></span>G Web Shopping

#### <span id="page-34-1"></span>G.1 Prompt Example

We use the following format of the prompt:

```
User :
You are web shopping .
I will give you instructions about what to do .
You have to follow the instructions .
Every round I will give you an observation and a list of available
   actions , you have to respond an action based on the state and
   instruction .
You can use search action if search is available .
You can click one of the buttons in clickables .
An action should be of the following structure :
search [ keywords ]
click [ value ]
If the action is not valid , perform nothing .
Keywords in search are up to you , but the value in click must be a
   value in the list of available actions .
Remember that your keywords in search should be carefully designed .
Your response should use the following format :
Thought :
I think ...
Action :
click [ something ]}
User :
Observation :
{ observation }
Available Actions :
{ available_actions }
```

We use an one-shot example.

```
User :
Observation :
" WebShop [ SEP ] Instruction : [ SEP ] i need a long lasting 6.76 fl oz
    bottle of l ' eau d ' issey , and price lower than 100.00 dollars [ SEP ]
     Search "
Available Actions :
{" has_search_bar ": true , " clickables ": ["..."]}
Agent :
Thought :
I think I should use the search bar to look for the product I need .
Action :
search [l ' eau d ' issey 6.76 fl oz bottle price < 100.00]
User :
Observation :
" Instruction : [ SEP ] i need a long lasting 6.76 fl oz bottle of l ' eau d
   ' issey , and price lower than 100.00 dollars [ SEP ] Back to Search [
    SEP ] Page 1 ( Total results : 50) [ SEP ] Next > [ SEP ] B000VOHH8I [ SEP
   ] L ' eau D ' issey By Issey Miyake for MenEau De Toilette Spray , 6.7
    Fl Oz Bottle [ SEP ] $64 .98 [ SEP ] B000MJZOPK [ SEP ] L ' eau d ' Issey by
    Issey Miyake for Women 3.3 oz Eau de Toilette Spray [ SEP ] $49 .98 [
    SEP ] B0012S249E [ SEP ] L ' eau D ' issey By Issey Miyake For Women .
```

```
Shower Cream 6.7 - Ounces [ SEP ] $31 .36 [ SEP ] B01H8PGKZS [ SEP ] L ' eau
   D ' Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [ SEP ] $67 .97 [
    SEP ] B00G3C8FHE [ SEP ] L ' Eau d ' Issey pour Homme - Eau de Toilette
    4.2 fl oz [ SEP ] $51 .25 [ SEP ] B000R94HRG [ SEP ] Issey Miyake L ' Eau D
    ' Issey Pour Homme Eau De Toilette Natural Spray [ SEP ] $44 .99 [ SEP ]
     B000C214CO [ SEP ] Issey Miyake L ' eau D ' issey Eau de Toilette Spray
     for Men , 4.2 Fl Oz [ SEP ] $53 .99 [ SEP ] B0018SBRDC [ SEP ] Issey
    Miyake L ' eau d ' Issey for Women EDT , White , 0.84 Fl Oz [ SEP ] $27 .04
     [ SEP ] B000XEAZ9Y [ SEP ] L ' eau De Issey By Issey Miyake For Men .
    Eau De Toilette Spray 6.7 Fl Oz [ SEP ] $67 .08 [ SEP ] B079HZR2RX [ SEP
    ] L ' eau d ' Issey Pure by Issey Miyake for Women 3.0 oz Nectar de
    Parfum Spray [ SEP ] $71 .49"
Available Actions :
{" has_search_bar ": false , " clickables ": ["..." , "..." , "..." , "..." ,
    "..." , "..." , "..." , "..." , "..." , "..." , "..." , "..."]}
Agent :
Thought :
I think I should click on the product I need , which is B000VOHH8I .
Action :
click [ B000VOHH8I ]
User :
Observation :
" Instruction : [ SEP ] i need a long lasting 6.76 fl oz bottle of l ' eau d
    ' issey , and price lower than 100.00 dollars [ SEP ] Back to Search [
    SEP ] < Prev [ SEP ] size [ SEP ] 2.5 fl oz [ SEP ] 6.76 fl oz ( pack of
    1) [ SEP ] L ' eau D ' issey By Issey Miyake for MenEau De Toilette
    Spray , 6.7 Fl Oz Bottle [ SEP ] Price : $64 .98 [ SEP ] Rating : N . A . [
    SEP ] Description [ SEP ] Features [ SEP ] Reviews [ SEP ] Buy Now "
Available Actions :
{" has_search_bar ": false , " clickables ": ["..." , "..." , "..." , "..." ,
    "..." , "..." , "..." , "..."]}
Agent :
Thought :
I think I should click on the '6.76 fl oz ( pack of 1) ' option to
    select the size I need .
Action :
click [6.76 fl oz ( pack of 1) ]
User :
Observation :
" Instruction : [ SEP ] i need a long lasting 6.76 fl oz bottle of l ' eau d
    ' issey , and price lower than 100.00 dollars [ SEP ] Back to Search [
    SEP ] < Prev [ SEP ] size [ SEP ] 2.5 fl oz [ SEP ] 6.76 fl oz ( pack of
    1) [ SEP ] L ' eau D ' issey By Issey Miyake for MenEau De Toilette
    Spray , 6.7 Fl Oz Bottle [ SEP ] Price : $64 .98 [ SEP ] Rating : N . A . [
    SEP ] Description [ SEP ] Features [ SEP ] Reviews [ SEP ] Buy Now "
Available Actions :
{" has_search_bar ": false , " clickables ": ["..." , "..." , "..." , "..." ,
    "..." , "..." , "..." , "..."]}
Agent :
Thought :
I think I should click on the ' Buy Now ' button to purchase the product
    .
Action :
click [ Buy Now ]
```

# <span id="page-36-0"></span>H Web Browsing

#### <span id="page-36-1"></span>H.1 Prompt Example.

We use the following 3-example CoT prompts for Mind2Web evaluation:

```
User :
'''
< html > <div > <div > <a tock home page / > < button id =0 book a
    reservation . toggle open > < span > Book a reservation </ span > </
    button > < button book a reservation . toggle open > </ button > </ div >
   <div > < select id =1 type > < option reservations true > Dine in </
    option > ... </ html >
'''
Based on the HTML webpage above , try to complete the following task :
Task : Check for pickup restaurant available in Boston , NY on March 18 ,
     5 pm with just one guest
Previous actions :
None
What should be the next action ? Please select from the following
    choices ( If the correct action is not in the page above , please
    select A . ' None of the above ') :
A . None of the above
B . < button id =0 book a reservation . toggle open > < span > Book a
C . < select id =1 type > < option reservations true > Dine in </ option > <
    option
D . < div id =2 > <p > Celebrating and supporting leading women shaking up
Assistant :
Thought : I need to select pickup restaurant first .
Answer : C .
Action : SELECT
Value : Pickup
User :
'''
< html > <div > < main main > < section tabpanel > <div > < ul tablist > < li tab
     heading level 3 search and > </ li > < li id =0 tab heading level 3
    search and > < span > Hotel </ span > </ li > <li tab heading level 3
    search and > </ li > < li tab heading level 3 search and > </ li > </ ul >
   < div tabpanel > < div id =1 > <div > < span > Dates * </ span > < button
    button clear dates / > </ div > ... </ html >
'''
Based on the HTML webpage above , try to complete the following task :
Task : Compare the fare types to book a 1 - adult ticket from Springfiels
   , IL to Austin , TX for April 29 th 2023
Previous actions :
[ combobox ] Enter your departing city , airport name , or airpor ... ->
    TYPE : SPRINGFIELD
[ button ] Springfield , IL , US ( SPI ) -> CLICK
[ combobox ] Enter your destination city , airport name , or airp ... ->
    TYPE : AUSTIN
[ button ] Austin , TX , US ( AUS ) -> CLICK
What should be the next action ? Please select from the following
    choices ( If the correct action is not in the page above , please
    select A . ' None of the above ') :
A . None of the above
B . < li id =0 tab heading level 3 search and > < span > Hotel
C . < div id =1 > <div > < span > Dates * </ span > < button button clear dates
D . < ul id =2 > <a mobile tools > </a > <a open united ' s tiktok
```

```
Assistant :
Thought : There are no elements that allow setting the date or viewing
   the fare , so there is no correct action .
Answer : A .
User :
'''
< html > <div > < nav main menu > <ul > <li > < div button > Car Sales </ div > <
   div id =0 > <div > <div > <div > Buy A Car </ div > <div > Plan Your
    Purchase </ div > </ div > <div > <h4 > Its Tax Refund Time . Treat
    Yourself to an Upgrade . </ h4 > <p >
With a variety of options , invest your refund in what you really want
   - a quality , used vehicle from Enterprise . </p > ... </ html >
'''
Based on the HTML webpage above , try to complete the following task :
Task : Find a mini van at Brooklyn City from April 5 th to April 8 th for
     a 22 year old renter .
Previous actions :
[ searchbox ] Pick - up & Return Location ( ZIP , City or Airport ) (... ->
   TYPE : Brooklyn
[ option ] Brooklyn , NY , US Select -> CLICK
What should be the next action ? Please select from the following
    choices ( If the correct action is not in the page above , please
    select A . ' None of the above ') :
A . None of the above
B . < div id =0 > <div > <div > <div > Buy A Car </ div > <div >
C . < div id =1 > Enterprise Fleet Management </ div >
D . < button id =2 selected pick - up date 03/19/2023 > < span > < span > 19 </
   span >
Assistant :
Thought : The search has already been set to Brooklyn . Next , I should
    choose pick - up date .
```

Answer : D . Action : CLICK