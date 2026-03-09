# Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs

Oded Ovadia \*† Menachem Brief, Moshik Mishaeli, and Oren Elisha

{odedovadia,t-mbrief,mmishaeli,oren.elisha}@microsoft.com Microsoft, Israel

#### **Abstract**

Large language models (LLMs) encapsulate a vast amount of factual information within their pre-trained weights, as evidenced by their ability to answer diverse questions across different domains. However, this knowledge is inherently limited, relying heavily on the characteristics of the training data. Consequently, using external datasets to incorporate new information or refine the capabilities of LLMs on previously seen information poses a significant challenge. In this study, we compare two common approaches: fine-tuning and retrieval-augmented generation (RAG). We evaluate both approaches on a variety of knowledge-intensive tasks across different topics. Our findings reveal that while fine-tuning offers some improvement, RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge. Moreover, we find that LLMs struggle to learn new factual information through fine-tuning, and that exposing them to numerous variations of the same fact during training could alleviate this problem.

#### 1. Introduction

Large language models (LLMs) are able to capture vast amounts of factual information (Petroni et al., 2019; Cohen et al., 2023; Hu et al., 2023). LLMs exhibit a remarkable level of knowledge in various domains due to their massive pre-training datasets. However, there are two significant limitations to this knowledge. First, it is static and does not update with time. Second, it is non-specific and thus may lack nuanced expertise in particular domains. While these are two different problems, they are deeply related since their solution is the same: enhancing the model's knowledge.

Recently, the idea of adapting LLMs to particular domains and updating their knowledge has become increasingly common (Yu et al., 2022). Various models have been suggested to improve factual knowledge and capabilities in diverse fields such as healthcare (Singhal et al., 2023a;b; Wu et al., 2023a), finance (Wu et al., 2023b; Yang et al., 2023), and law (Huang et al., 2023; Nguyen, 2023).

In this work, we focus on the evaluation of a model's knowledge and its ability to memorize, understand, and retrieve factual data. We aim to understand the concept of *knowledge injection* (Wang et al., 2020; Chen et al., 2022; Liu et al., 2020; Lauscher et al., 2020). Given some knowledge base in the form of a text corpus, what is the best way to teach a pre-trained model this knowledge?

One way to add knowledge to a pre-trained model is through fine-tuning. With fine-tuning, we continue the model's training process and adapt it using task-specific data. By exposing the model to a specific knowledge base, we expect the model weights to adapt accordingly. This process is meant to optimize the model for targeted applications, enhancing its performance and contextual relevance in specialized domains.

Another method to enhance a model's knowledge base is through the use of in-context learning (ICL) (Chen et al., 2021; Radford et al., 2019; Min et al., 2021; Lampinen et al., 2022). The main idea behind ICL is to improve the performance of pre-trained LLMs on new tasks by modifying the input query to the model without directly changing the weights of the model. One form of ICL is retrieval augmented generation (RAG) (Lewis et al., 2020; Neelakantan et al., 2022). RAG uses information retrieval techniques to enable LLMs to obtain relevant information from a knowledge source and incorporate it into generated text.

This study aims to evaluate the knowledge injection capabilities of LLMs through a comparison of fine-tuning and RAG. To illustrate the rationale, let us use an analogy. Consider three college students taking a test on a specific topic. All had access to class materials but didn't know the topic beforehand. The first student had the textbook only during

<sup>\*</sup>Corresponding author.

<sup>†</sup>Equal contribution.

the test, the second had pre-test access and studied, and the third lost access upon the test announcement. Who would probably perform better?

## <span id="page-1-0"></span>2. Background

To assess *knowledge injection*, we must first understand what *knowledge* means for LLMs.

**Knowledge and Language Models** Defining knowledge is a complex philosophical task far beyond the scope of this research. However, we can examine what factual knowledge means in the context of language models. If a model knows a fact, it can accurately and consistently answer questions about it. Furthermore, it can reliably distinguish between true and false statements related to this fact. We can then extend this definition to a whole knowledge base, not just a single fact.

Mathematically, let  $\mathcal{Q}=\{q_n\}_{n=1}^N$  be a set of N multiple choice factual questions, where each question has L possible answers and exactly one correct answer. Let  $\mathcal{A}=\{(a_n^1,\ldots,a_n^L)\}_{n=1}^N$  be the corresponding set of possible answers, and  $\mathcal{C}=\{c_n\}_{n=1}^N$  be the correct answers.

Let  $\mathcal{M}$  be a language model. We denote by  $\mathcal{M}(q_n) \in \{a_n^1, \dots, a_n^L\}$  the predicted answer of the model to the n-th question.

We define the *knowledge score*  $\mathcal{L}$  of  $\mathcal{M}$  in relation to  $\mathcal{Q}$  to be the standard accuracy score:

<span id="page-1-1"></span>
$$\mathcal{L}_{\mathcal{M},\mathcal{Q}} := \frac{\#\{q_n | \mathcal{M}(q_n) = c_n\}}{N}.$$
 (1)

We say that the model  $\mathcal{M}$  possesses *any* knowledge regarding the set of questions  $\mathcal{Q}$  if the following holds:

<span id="page-1-2"></span>
$$\mathcal{L}_{\mathcal{M},\mathcal{Q}} > \frac{1}{L}.$$
 (2)

In simpler terms, the model can consistently give correct answers, outperforming a simple random guessing baseline. Naturally, if the knowledge score  $\mathcal{L}_{\mathcal{M},\mathcal{Q}}$  is higher for one model compared to another, then we assert that the former is more knowledgeable with regards to  $\mathcal{Q}$  compared to the latter.

**Previously Seen Knowledge** One important distinction to make is between knowledge that the model has been exposed to before during pre-training as opposed to entirely new facts. Considering the size of modern LLM training sets, they cover a vast amount of information available through web-sourced text. As a result, even in niche domains, the goal of knowledge injection is not necessarily to teach the model entirely new facts but rather to "refresh" its memory by inducing a bias toward a particular domain.

**Knowledge and Reasoning** We emphasize that this knowledge evaluation framework for LLMs is imperfect. Importantly, it doesn't address other quality metrics influencing a model's response. Creating a purely knowledgeintensive dataset without involving some level of reasoning is challenging. Consequently, a model with robust reasoning abilities might excel on unfamiliar knowledge-intensive tasks by making "educated guesses" in a multiple-choice exam. Therefore, any evaluation of knowledge in LLMs should consider this, with results seen as part of a broader range of benchmarks for reasoning (Sakaguchi et al., 2021), reading comprehension (Dua et al., 2019), and general language abilities (Srivastava et al., 2022). However, this evaluation framework still strongly emphasizes factual information above all else.

**Causes for Factual Errors** There are many possible reasons for the failure of models to answer factual questions accurately. In (Wang et al., 2023), Wang *et al.* introduce a taxonomy of five main model-level causes:

- Domain knowledge deficit: A language model may lack comprehensive expertise in a specific domain to which it has not been exposed. For example, a model trained exclusively on texts written by William Shakespeare would perform poorly when asked about the works of Mark Twain.
- Outdated Information: LLMs invariably have a cutoff date determined by their training dataset. Consequently, any events, discoveries, or changes occurring after the last training update will not be within the model's knowledge without access to external sources.
- Immemorization: Sometimes, a model is exposed to knowledge during its training process but does not retain it. This is especially true for rare facts that appear in the training dataset only scarcely (Kandpal et al., 2023).
- Forgetting: Language models often undergo additional training after the pre-training phase (fine-tuning). In some cases, this might lead to a phenomenon called *catastrophic forgetting* (Kirkpatrick et al., 2017; Goodfellow et al., 2013; Chen et al., 2020; Luo et al., 2023), where models lose some of the knowledge they had prior to the fine-tuning process.
- Reasoning Failure: In certain instances, a language model might possess relevant knowledge about a fact but fail to utilize it properly. This is particularly evident in complex multi-step reasoning tasks (Tan et al., 2023) or when posed with different questions about the same fact, resulting in disparate outcomes (Berglund et al., 2023).

We observe that most of these issues arise during the pretraining phase, with catastrophic forgetting being the notable

![](_page_2_Figure_1.jpeg)

Figure 1. A visualization of the knowledge injection framework.

exception. Hence, many LLMs will suffer from factual errors of this kind regardless of any post-training process.

## 3. Injecting Knowledge to Language Models

Following the background given in Section [2,](#page-1-0) it is clear that general pre-training is insufficient for many knowledgeintensive tasks. To solve this, an additional post-processing step is essential to augment the knowledge of a pre-trained model. This step is often reffered to as *knowledge injection* [\(Wang et al.,](#page-10-4) [2020;](#page-10-4) [Chen et al.,](#page-7-0) [2022;](#page-7-0) [Liu et al.,](#page-8-3) [2020;](#page-8-3) [Lauscher et al.,](#page-8-4) [2020\)](#page-8-4).

In this section, we examine two widely used frameworks for knowledge injection: fine-tuning (FT) and retrieval augmented generation (RAG). We begin by formulating the knowledge injection problem, aiming to explain both methods using consistent terminology.

#### 3.1. Problem formulation

In Equations [\(1\)](#page-1-1) and [\(2\)](#page-1-2), we presented a formulation for knowledge in language models through the lens of questionanswering (Q&A). We now extend this formulation to the problem of knowledge injection using the same terminology.

Given a set of factual questions, there exists some text corpus containing information that is relevant to these questions. The central assumption of knowledge injection is that given

full access to this corpus, it could serve as an auxiliary knowledge base and improve the model's performance on this set of questions.

Mathematically, let M be a pre-trained model, and let Q be a set of factual questions, as before. Now, assume we have a relevant auxiliary knowledge base BQ. Our objective is to discover a transformation, denoted as F, that, when applied, would enhance the knowledge about Q:

$$\mathcal{M}' := \mathcal{F}(\mathcal{M}, \mathcal{B}_{\mathcal{Q}}) \quad s.t. \quad \mathcal{L}_{\mathcal{M}', \mathcal{Q}} > \mathcal{L}_{\mathcal{M}, \mathcal{Q}}.$$
 (3)

In this work, we aim to compare two choices for F: finetuning and RAG to see which option performs better in this problem.

#### <span id="page-2-0"></span>3.2. Fine-Tuning

Fine-tuning is the process of adjusting a pre-trained model on a specific, often narrower, dataset or task to enhance its performance in that particular domain. Here, it is vital to distinguish between different types of fine-tuning. FT techniques are commonly classified into supervised, unsupervised, and reinforcement learning (RL) based methods. We proceed by briefly reviewing these methods and their relation to the problem of knowledge injection.

Supervised Fine-Tuning Supervised fine-tuning (SFT) requires sets of labeled input-output pairs. One of the most

common SFT methods is instruction tuning (Wang et al., 2022; Mishra et al., 2021; Ouyang et al., 2022; Taori et al., 2023), which has emerged as one of the most powerful methods to improve model performance. With instruction tuning, the input is a natural language task description, and the output is an example of the desired behavior. Many current state-of-the-art LLMs have gone through instruction tuning after their pre-training phase.

Instruction tuning has been shown to be very effective at improving the overall quality of the model, with a particular emphasis on its zero-shot and reasoning capabilities. However, despite these advantages, instruction tuning does not necessarily teach the model new knowledge (Ouyang et al., 2022; Chung et al., 2022; Mitra et al., 2023; Chia et al., 2023; Zhou et al., 2023). As such, instruction tuning alone is not a viable solution to the knowledge injection problem.

**Reinforcemnt Learning** Another form of FT relies on RL or RL-inspired optimization strategies to better align the model after its pre-training phase. A few prominent examples are reinforcement learning from human feedback (RLHF) (OpenAI, 2023; Touvron et al., 2023), direct preference optimization (DPO) (Rafailov et al., 2023), and proximal policy optimization (PPO) (Schulman et al., 2017; Tunstall et al., 2023).

These techniques have been shown to be very useful, especially when used in conjunction with instruction tuning. However, similarly to instruction tuning, these methods focus on the overall quality of the response and its expected behavior and not necessarily on its breadth of knowledge.

**Unsupervised Fine-Tuning** The final FT strategy we discuss is unsupervised, meaning there are no available labels for the model to learn from. One common unsupervised FT technique is often referred to as *continual pre-training* or *unstructured* FT.

In this method, the FT process is viewed as a direct continuation of the pre-training phase. We start with a saved checkpoint of the original LLM and train it in a causal autoregressive manner, i.e., predicting the next token. One major difference in comparison to actual pre-training is the learning rate. Usually, one would need a much lower learning rate when continuing the pre-training of the model to avoid catastrophic forgetting (Kirkpatrick et al., 2017).

It is well known that LLMs store vast amounts of knowledge during their pre-training phase (Zhou et al., 2023). So, it makes sense to continue this process in order to inject knowledge into the model. Hence, we use the unsupervised FT approach throughout this work and evaluate its efficacy in enhancing the model's capacity for learning new information.

#### 3.3. Retrieval Augmented Generation

Retrieval augmented generation (RAG) (Lewis et al., 2020) is a technique that expands LLMs' capabilities, especially in knowledge-intensive tasks, by using external knowledge sources. While the original formulation involved additional training per task, it has since been demonstrated (Neelakantan et al., 2022) that a pre-trained *embedding* model can achieve improved performance with no additional training involved.

The idea is that given an auxiliary knowledge base and an input query, we use the RAG architecture to find documents within the knowledge base that resemble the input query. These documents are then added to the input query, thus giving the model further context about the subject of the query.

In practice, implementing the suggested architecture is quite straightforward: Given an auxiliary knowledge base  $\mathcal{B}_{\mathcal{Q}}$  and a pre-trained embedding model  $\mathcal{M}_e$ , we create a dense vector representation (embedding) per document  $b \in \mathcal{B}_{\mathcal{Q}}$  and store these in a vector store. Upon receiving a new query q, we use its embedding,  $\mathcal{M}_e(q)$ , to retrieve q's top-K closest neighbors,  $\mathbf{b}_q = \{b_k\}_1^K$ , according to dot-product ranking. We then update q to be  $\tilde{q} = \mathbf{b}_q \| q$ , where  $\|$  denotes string concatenation. Finally, we return  $\mathcal{M}(\tilde{q})$  as the model's output.

## 4. Knowledge Base Creation

#### <span id="page-3-1"></span>4.1. Task Selection and Rationale

MMLU Benchmark To properly evaluate the capabilities of LLMs on knowledge-intensive tasks, we selected four distinct tasks from the Massively Multilingual Language Understanding Evaluation (MMLU) benchmark (Hendrycks et al., 2021) in the topics of anatomy, astronomy, college biology, and college chemistry. The chosen tasks were selected based on their emphasis on factual knowledge and the minimal reliance on reasoning. As a heuristic, we opted for tasks where the questions are short and involve no context. This approach aims to enable us to test LLM proficiency in comprehending and manipulating information in isolation from its reasoning processes.

**Current Events Task** To further isolate LLMs' abilities to learn new knowledge, we created a task comprising multiple-choice questions about current events. This task includes multiple-choice questions about events that occurred after the cutoff of the various models' training data. Specifically, we focused on "current events" from the USA, in the time span of August-November 2023, that are included in the relevant Wikipedia indexes<sup>1</sup>. This method enables us

<span id="page-3-0"></span>Inttps://en.wikipedia.org/wiki/Category: 2023\_events\_in\_the\_United\_States\_by\_month

<span id="page-4-1"></span>

| Task                       | Model      | Base model | Base model + RAG | Fine-tuned | Fine-tuned + RAG |
|----------------------------|------------|------------|------------------|------------|------------------|
|                            | Mistral 7B | 0.556      | 0.681            | 0.570      | 0.659            |
| Anatomy (0-shot)           | Llama2 7B  | 0.422      | 0.496            | 0.430      | 0.489            |
|                            | Orca2 7B   | 0.607      | 0.644            | 0.600      | 0.637            |
|                            | Mistral 7B | 0.600      | 0.681            | 0.622      | 0.674            |
| Anatomy (5-shot)           | Llama2 7B  | 0.467      | 0.563            | 0.496      | 0.548            |
|                            | Orca2 7B   | 0.578      | 0.652            | 0.593      | 0.674            |
|                            | Mistral 7B | 0.625      | 0.684            | 0.651      | 0.697            |
| Astronomy (0-shot)         | Llama2 7B  | 0.421      | 0.461            | 0.487      | 0.520            |
|                            | Orca2 7B   | 0.651      | 0.763            | 0.651      | 0.75             |
|                            | Mistral 7B | 0.658      | 0.704            | 0.651      | 0.691            |
| Astronomy (5-shot)         | Llama2 7B  | 0.388      | 0.474            | 0.447      | 0.520            |
|                            | Orca2 7B   | 0.664      | 0.770            | 0.664      | 0.743            |
|                            | Mistral 7B | 0.681      | 0.757            | 0.701      | 0.764            |
| College biology (0-shot)   | Llama2 7B  | 0.451      | 0.486            | 0.458      | 0.465            |
|                            | Orca2 7B   | 0.576      | 0.646            | 0.604      | 0.632            |
|                            | Mistral 7B | 0.722      | 0.778            | 0.736      | 0.771            |
| College biology (5-shot)   | Llama2 7B  | 0.465      | 0.521            | 0.424      | 0.479            |
|                            | Orca2 7B   | 0.611      | 0.660            | 0.625      | 0.653            |
|                            | Mistral 7B | 0.470      | 0.500            | 0.490      | 0.500            |
| College chemistry (0-shot) | Llama2 7B  | 0.320      | 0.400            | 0.390      | 0.390            |
|                            | Orca2 7B   | 0.360      | 0.440            | 0.370      | 0.390            |
|                            | Mistral 7B | 0.470      | 0.540            | 0.500      | 0.500            |
| College chemistry (5-shot) | Llama2 7B  | 0.340      | 0.410            | 0.360      | 0.390            |
|                            | Orca2 7B   | 0.450      | 0.470            | 0.370      | 0.380            |

Table 1. Results for the MMLU datasets described in Section [4.1](#page-3-1) in terms of log-likelihood accuracy (Equation [\(4\)](#page-6-0)).

|            | Base model | Base model + RAG | FT-reg | FT-par | FT-reg + RAG | FT-par + RAG |
|------------|------------|------------------|--------|--------|--------------|--------------|
| Mistral 7B | 0.481      | 0.875            | 0.504  | 0.588  | 0.810        | 0.830        |
| Llama2 7B  | 0.353      | 0.585            | 0.219  | 0.392  | 0.326        | 0.520        |
| Orca2 7B   | 0.456      | 0.876            | 0.511  | 0.566  | 0.820        | 0.826        |

Table 2. Current events results. Models that were fine-tuned on the original dataset are labeled as *FT-reg*, while those trained on the dataset with multiple paraphrases are labeled as *FT-par*.

to mostly guarantee that the models have not been exposed to these facts, thus allowing us to directly test knowledge injection capabilities.

#### 4.2. Data Collection and Preprocessing

To effectively evaluate the LLMs' performance on these knowledge-intensive tasks, a comprehensive auxiliary dataset was collected by scraping relevant articles per topic from Wikipedia. The rationale behind selecting Wikipedia as the primary source of knowledge is its broad coverage of relevant topics and its reliability as a repository of crowdverified knowledge. All articles pertinent to the tasks were retrieved via the official Wikipedia API[2](#page-4-0) by identifying the relevant central page per topic.

<span id="page-4-0"></span><sup>2</sup>[https://www.mediawiki.org/wiki/API:](https://www.mediawiki.org/wiki/API:Main_page) [Main\\_page](https://www.mediawiki.org/wiki/API:Main_page)

Subsequently, a rigorous cleaning process was utilized to transform the data from raw subsections to clean chunks. This step was done with the "wikiextractor" tool [\(Attardi,](#page-7-3) [2015\)](#page-7-3). The division into small, clean (e.g., remove HTML, URLs, etc.) chunks was aimed at enhancing the evaluation of the LLMs' understanding across various knowledge domains and aiding the LLMs in the fine-tuning process.

#### <span id="page-5-3"></span>4.3. Current Events Task Creation

After collecting the relevant chunks from Wikipedia, we created a new multiple-choice dataset with the help of GPT-4 [\(OpenAI,](#page-9-11) [2023\)](#page-9-11). First, we removed any small chunks. For each remaining chunk in the corpus, GPT-4 was instructed to create four highly specific, high-quality multiple-choice questions with only one correct answer. By specific, we mean that the question can be answered without knowledge of which context the question refers to and with minimal ambiguity. Next, GPT-4 was asked to select the two most specific of the four. This was followed by a manual evaluation and verification step. In total, this resulted in 910 new questions.

#### 4.4. Paraphrases Generation

After creating the dataset, we utilized GPT-4 to generate augmentations of the dataset. We instructed GPT-4 to provide paraphrased versions of the input data that fully retain the information while being reworded. Each paraphrasing iteration was done with a different seed to ensure variety.

We selected 240 chunks at random for each task and created two paraphrases per chunk. These were set aside to be used as validation sets for hyperparameter tuning. For the current events dataset, we created ten paraphrases for each chunk used in the fine-tuning process described in Section [6.](#page-6-1)

## <span id="page-5-4"></span>5. Experiments and Results

Experimental Framework We used the popular LM-Evaluation-Harness [\(Gao et al.,](#page-8-20) [2021\)](#page-8-20) repository to evaluate the performance of LLMs on the selected knowledgeintensive tasks. LM-Evaluation-Harness is a robust benchmarking tool that currently serves as the industry standard for model evaluation and is the basis of the HuggingFace leaderboard[3](#page-5-0) . Leveraging this platform ensured a standardized evaluation framework and allowed consistent comparison across models, methods, and datasets. More importantly, by using the industry standard for evaluation, we could avoid any differences stemming from prompt engineering and formatting issues and replicate the reported baseline results for each model.

![](_page_5_Figure_10.jpeg)

<span id="page-5-2"></span>![](_page_5_Figure_11.jpeg)

Figure 2. Box plot comparing all knowledge-injection methods over all experiments in Table [1.](#page-4-1)

Model Selection We chose three models for inference evaluation: Llama2-7B [\(Touvron et al.,](#page-9-12) [2023\)](#page-9-12), Mistral-7B [\(Jiang et al.,](#page-8-21) [2023\)](#page-8-21), and Orca2-7B [\(Mitra et al.,](#page-8-17) [2023\)](#page-8-17). The choice of these models was meant to represent the most popular open-source base models and an instruction-tuned model across various baseline capabilities. Additionally, we selected *bge-large-en* [\(Xiao et al.,](#page-10-7) [2023\)](#page-10-7) as the embedding model for the RAG component and used FAISS [\(Johnson](#page-8-22) [et al.,](#page-8-22) [2019\)](#page-8-22) as its vector-store. This embedding model is currently the SOTA of open-source embedding models, according to the HuggingFace MTEB leaderboard[4](#page-5-1) .

Configuration Variations Our evaluation included multiple configurations, with a grid-search over them, to allow for more comprehensive benchmarking.

Firstly, we compared the baseline and fine-tuned models and their performance with the RAG component. Secondly, we explored the optimal number of text chunks to add to the context in RAG. Specifically, different values of K ∈ {0, . . . , 5} were employed to analyze the impact on model performance. Finally, we explored 5-shot performance vs. 0-shot.

Training Setup We trained all of the models using the unsupervised training procedure described in Section [3.2.](#page-2-0) For each dataset, we divided the auxiliary knowledge base into equal chunks of size 256 by concatenating or splitting the original chunks based on their length. We also added two special tokens, <BOS> and <EOS>, to demarcate the original chunks' beginnings and ends to preserve the documents' structure.

The models were trained using learning rates between 1 × 10<sup>−</sup><sup>6</sup> and 5 × 10<sup>−</sup><sup>5</sup> , which were found through a hyper-

<span id="page-5-0"></span><sup>3</sup>[https://huggingface.co/spaces/](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) [HuggingFaceH4/open\\_llm\\_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

<span id="page-5-1"></span><sup>4</sup>[https://huggingface.co/spaces/mteb/](https://huggingface.co/spaces/mteb/leaderboard) [leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

parameter search. All models were trained on 4 NVIDIA A-100 GPUs for a maximum of 5 epochs and a batch size of 64.

**Evaluation method** All evaluations were done by appending each of the multiple-choice options to the question, followed by passing the concatenation through the model to get a log probability score per option. The highest score was interpreted as the model's choice and used for accuracy calculation. More formally, this means that in Equation (1) we say that  $\mathcal{M}(q_n) = c_n$  if:

<span id="page-6-0"></span>
$$c_n = \arg\max_{l} \{ \mathcal{M}(q_n || a_n^1), \dots, \mathcal{M}(q_n || a_n^L) \}, \quad (4)$$

where  $\mathcal{M}(q_n || a_n^l) = \log P_{\mathcal{M}}(q_n || a_n^l)$ .

**MMLU Results** For each task and model, we compared four approaches: using just the base model, RAG, FT, and finally combining FT and RAG by using the fine-tuned model as the generator. Furthermore, we tested the MMLU tasks using both 0-shot and 5-shot scenarios. The full results are shown in Table 1 and summarized in Figure 2.

In all cases, RAG performed significantly better compared to the base models. Furthermore, using RAG with the base model as the generator was consistently better than only fine-tuning. In some cases, using the fine-tuned model instead of the base model as the generator in the RAG pipeline can improve results even further. However, this is not consistent and demonstrates the instability of fine-tuning. Additionally, we found that the 5-shot approach boosts the results by a small margin in most cases, although not consistently.

**Current Events Results** The evaluation on the current events task is shown in Table 2. RAG proves particularly effective due to the one-to-one correspondence between the questions and the auxiliary dataset (see Section 4.3). Fine-tuning is not competitive with RAG. However, fine-tuning with multiple paraphrases still provides a significant improvement over the baseline. We note that combining RAG with fine-tuning shows inferior performance compared to RAG alone.

It is worth noting that although the questions are based on information the models were not exposed to during training, the results of the base models surpass  $\frac{1}{L}=0.25$ . We deduce that this is due to the correlation between past and present events, along with the high reasoning capabilities of Orca2-7B and Mistral-7B.

**Fine-Tuning vs. RAG:** In the results of both the MMLU and current events tasks, a significant advantage for RAG over fine-tuning is evident. While fine-tuning improved results compared to the base model in most cases, it was not competitive with the RAG approach.

Several factors might contribute to this behavior. Firstly, RAG not only adds knowledge to a model but also incor-

porates context relevant to the question, a feature lacking in fine-tuning. Additionally, fine-tuning may impact other capabilities of the model due to a degree of catastrophic forgetting. Finally, it's plausible that unsupervised fine-tuned models might benefit from further alignment through supervised or RL-based fine-tuning, as evidenced by the vastly improved performance of Orca2 over the base Llama2.

## <span id="page-6-1"></span>6. The Importance of Repetition

Unlike the other tasks, where the model has been exposed to aspects related to the topic during pretraining, *current events* includes new information. In this case, standard regular fine-tuning not only did not improve the performance of Llama2 but also significantly degraded it. To improve the fine-tuning results, we explored augmentation of the data using paraphrases.

<span id="page-6-3"></span>![](_page_6_Figure_14.jpeg)

Figure 3. Training loss over time for Mistral-7B.

<span id="page-6-2"></span>![](_page_6_Figure_16.jpeg)

Figure 4. Model accuracy on the *current events* task as a function of the number of paraphrases.

**Data Augmentation** Data augmentation is a well-established method for enhancing the performance of lan-

guage models and has been surveyed extensively [\(Shorten](#page-9-16) [et al.,](#page-9-16) [2021\)](#page-9-16). Using generative models for augmentations has also been used successfully to improve classification models in the past [\(Sharma et al.,](#page-9-17) [2022\)](#page-9-17). An example of data augmentation using paraphrasing can be found in Appendix [B.](#page-11-0)

Monotonic Improvement This approach resulted in notable improvements in our results, showcasing a direct correlation between the number of paraphrases utilized and the models' accuracy. Our experimentation revealed a compelling trend, shown in Figure [4.](#page-6-2) For all models tested, the accuracy was a monotonically increasing function of the number of paraphrases used. This observation strongly suggests the positive impact of paraphrase augmentation, yielding information repetition, on the model's ability to comprehend and generalize new knowledge from limited data.

Learning New Information In Figure [3,](#page-6-3) we can see an interesting phenomenon observed throughout our experiments. After each epoch, i.e., completing another iteration over the entire dataset, the training loss drops significantly. This is consistent with what is known about LLMs memorizing the data during training and overfitting [\(Tirumala et al.,](#page-9-18) [2022\)](#page-9-18).

Our hypothesis is as follows

In order to teach pre-trained LLMs new knowledge, the knowledge must be repeated in numerous ways.

This is well known for LLM pre-training [\(Kandpal et al.,](#page-8-11) [2023\)](#page-8-11), and we see in this case that this holds for fine-tuning as well. The rationale for this hypothesis is that mere memorization of sentences does not entail knowledge of their content, as was already shown in [\(Berglund et al.,](#page-7-2) [2023\)](#page-7-2). By providing the information in numerous forms (like the data augmentation process we used), the various relationships in the data (e.g., a =⇒ b, b ≠⇒ c) stand a higher chance of appearing naturally. We believe this can potentially both increase L<sup>M</sup>,<sup>Q</sup> in general, as well as ameliorate Berglund et al.'s *Reversal Curse*. While promising, this result still warrants further research.

# 7. Conclusion and Future Work

Large language models possess vast amounts of knowledge on various topics. In this work, we tested their capability to adapt to new knowledge: both specialized and completely unseen. This is among the first studies to compare two prominent approaches in this domain, namely fine-tuning and retrieval augmented generation. While fine-tuning can be useful for many use-cases, we found that RAG is a more reliable choice for knowledge injection.

Some aspects of this work still warrant further research. For

example, we focused on unsupervised training as our primary fine-tuning method, as opposed to instruction-tuning or RL-based methods. Researching combinations of various techniques, with diverse auxiliary knowledge bases, may yield improved results. This approach, combined with our hypothesis from Section [6,](#page-6-1) could further enhance our understanding of knowledge injection via FT.

While we believe that this work further enhances our understanding of knowledge in LLMs, there is a lot more work to be done in this field. Specifically, more research is required regarding the question of knowledge representation in LLMs, especially from a theoretical perspective.

Finally, further efforts are needed to measure knowledge in LLMs. While we employed an empirical approach as described in Equation [\(2\)](#page-1-2), it is important to explore other definitions and perspectives on knowledge as well, and extend upon this work.

## 8. Limitations

As in all machine learning applications, the choice of hyperparameters significantly impacts the results. We therefore strongly recommend optimizing all relevant hyperparameters for specific cases.

We have supported our claims by running the experiments on three different models. However, generalization to other LLMs should be tested thoroughly. For example, GPT-4 achieves near perfect accuracy for some MMLU tasks [\(Nori](#page-9-19) [et al.,](#page-9-19) [2023\)](#page-9-19), and thus further improvement is not applicable.

Finally, while we chose various topics for the knowledge bases, all of our sources came from Wikipedia. Other datasets may yield different results, and must be evaluated carefully.

## References

<span id="page-7-3"></span>Attardi, G. Wikiextractor. [https://github.com/](https://github.com/attardi/wikiextractor) [attardi/wikiextractor](https://github.com/attardi/wikiextractor), 2015.

<span id="page-7-2"></span>Berglund, L., Tong, M., Kaufmann, M., Balesni, M., Stickland, A. C., Korbak, T., and Evans, O. The reversal curse: Llms trained on" a is b" fail to learn" b is a". *arXiv preprint arXiv:2309.12288*, 2023.

<span id="page-7-1"></span>Chen, S., Hou, Y., Cui, Y., Che, W., Liu, T., and Yu, X. Recall and learn: Fine-tuning deep pretrained language models with less forgetting. *arXiv preprint arXiv:2004.12651*, 2020.

<span id="page-7-0"></span>Chen, X., Zhang, N., Xie, X., Deng, S., Yao, Y., Tan, C., Huang, F., Si, L., and Chen, H. Knowprompt: Knowledgeaware prompt-tuning with synergistic optimization for

- relation extraction. In *Proceedings of the ACM Web conference 2022*, pp. 2778–2788, 2022.
- <span id="page-8-5"></span>Chen, Y., Zhong, R., Zha, S., Karypis, G., and He, H. Metalearning via language model in-context tuning. *arXiv preprint arXiv:2110.07814*, 2021.
- <span id="page-8-18"></span>Chia, Y. K., Hong, P., Bing, L., and Poria, S. Instructeval: Towards holistic evaluation of instruction-tuned large language models. *arXiv preprint arXiv:2306.04757*, 2023.
- <span id="page-8-16"></span>Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al. Scaling instruction-finetuned language models. *arXiv preprint arXiv:2210.11416*, 2022.
- <span id="page-8-0"></span>Cohen, R., Geva, M., Berant, J., and Globerson, A. Crawling the internal knowledge-base of language models. *arXiv preprint arXiv:2301.12810*, 2023.
- <span id="page-8-10"></span>Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. *arXiv preprint arXiv:1903.00161*, 2019.
- <span id="page-8-20"></span>Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., Phang, J., Reynolds, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, September 2021. URL [https:](https://doi.org/10.5281/zenodo.5371628) [//doi.org/10.5281/zenodo.5371628](https://doi.org/10.5281/zenodo.5371628).
- <span id="page-8-13"></span>Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., and Bengio, Y. An empirical investigation of catastrophic forgetting in gradient-based neural networks. *arXiv preprint arXiv:1312.6211*, 2013.
- <span id="page-8-19"></span>Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-8-1"></span>Hu, L., Liu, Z., Zhao, Z., Hou, L., Nie, L., and Li, J. A survey of knowledge enhanced pre-trained language models. *IEEE Transactions on Knowledge and Data Engineering*, 2023.
- <span id="page-8-2"></span>Huang, Q., Tao, M., An, Z., Zhang, C., Jiang, C., Chen, Z., Wu, Z., and Feng, Y. Lawyer llama technical report. *arXiv preprint arXiv:2305.15062*, 2023.
- <span id="page-8-21"></span>Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. *arXiv preprint arXiv:2310.06825*, 2023.
- <span id="page-8-22"></span>Johnson, J., Douze, M., and Jegou, H. Billion-scale similar- ´ ity search with GPUs. *IEEE Transactions on Big Data*, 7 (3):535–547, 2019.

- <span id="page-8-11"></span>Kandpal, N., Deng, H., Roberts, A., Wallace, E., and Raffel, C. Large language models struggle to learn long-tail knowledge. In *International Conference on Machine Learning*, pp. 15696–15707. PMLR, 2023.
- <span id="page-8-12"></span>Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, 114(13):3521–3526, 2017.
- <span id="page-8-7"></span>Lampinen, A. K., Dasgupta, I., Chan, S. C., Matthewson, K., Tessler, M. H., Creswell, A., McClelland, J. L., Wang, J. X., and Hill, F. Can language models learn from explanations in context? *arXiv preprint arXiv:2204.02329*, 2022.
- <span id="page-8-4"></span>Lauscher, A., Majewska, O., Ribeiro, L. F., Gurevych, I., Rozanov, N., and Glavas, G. Common sense or world ˇ knowledge? investigating adapter-based knowledge injection into pretrained transformers. *arXiv preprint arXiv:2005.11787*, 2020.
- <span id="page-8-8"></span>Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rockt ¨ aschel, ¨ T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. *Advances in Neural Information Processing Systems*, 33:9459–9474, 2020.
- <span id="page-8-3"></span>Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., and Wang, P. K-bert: Enabling language representation with knowledge graph. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pp. 2901–2908, 2020.
- <span id="page-8-14"></span>Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., and Zhang, Y. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *arXiv preprint arXiv:2308.08747*, 2023.
- <span id="page-8-6"></span>Min, S., Lewis, M., Zettlemoyer, L., and Hajishirzi, H. Metaicl: Learning to learn in context. *arXiv preprint arXiv:2110.15943*, 2021.
- <span id="page-8-15"></span>Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. Crosstask generalization via natural language crowdsourcing instructions. *arXiv preprint arXiv:2104.08773*, 2021.
- <span id="page-8-17"></span>Mitra, A., Del Corro, L., Mahajan, S., Codas, A., Simoes, C., Agrawal, S., Chen, X., Razdaibiedina, A., Jones, E., Aggarwal, K., et al. Orca 2: Teaching small language models how to reason. *arXiv preprint arXiv:2311.11045*, 2023.
- <span id="page-8-9"></span>Neelakantan, A., Xu, T., Puri, R., Radford, A., Han, J. M., Tworek, J., Yuan, Q., Tezak, N. A., Kim, J. W., Hallacy, C., Heidecke, J., Shyam, P., Power, B., Nekoul, T. E., Sastry, G., Krueger, G., Schnurr, D. P., Such, F. P., Hsu,

- K. S.-K., Thompson, M., Khan, T., Sherbakov, T., Jang, J., Welinder, P., and Weng, L. Text and code embeddings by contrastive pre-training. *ArXiv*, abs/2201.10005, 2022. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:246275593) [org/CorpusID:246275593](https://api.semanticscholar.org/CorpusID:246275593).
- <span id="page-9-3"></span>Nguyen, H.-T. A brief report on lawgpt 1.0: A virtual legal assistant based on gpt-3. *arXiv preprint arXiv:2302.05729*, 2023.
- <span id="page-9-19"></span>Nori, H., King, N., McKinney, S. M., Carignan, D., and Horvitz, E. Capabilities of gpt-4 on medical challenge problems. *ArXiv*, abs/2303.13375, 2023. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:257687695) [org/CorpusID:257687695](https://api.semanticscholar.org/CorpusID:257687695).
- <span id="page-9-11"></span>OpenAI. Gpt-4 technical report. *ArXiv*, abs/2303.08774, 2023. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:257532815) [org/CorpusID:257532815](https://api.semanticscholar.org/CorpusID:257532815).
- <span id="page-9-9"></span>Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022.
- <span id="page-9-0"></span>Petroni, F., Rocktaschel, T., Lewis, P., Bakhtin, A., Wu, ¨ Y., Miller, A. H., and Riedel, S. Language models as knowledge bases? *arXiv preprint arXiv:1909.01066*, 2019.
- <span id="page-9-4"></span>Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.
- <span id="page-9-13"></span>Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. *arXiv preprint arXiv:2305.18290*, 2023.
- <span id="page-9-5"></span>Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.
- <span id="page-9-14"></span>Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- <span id="page-9-17"></span>Sharma, S., Joshi, A., Mukhija, N., Zhao, Y., Bhathena, H., Singh, P., Santhanam, S., and Biswas, P. Systematic review of effect of data augmentation using paraphrasing on named entity recognition. In *NeurIPS 2022 Workshop on Synthetic Data for Empowering ML Research*, 2022. URL [https://openreview.net/forum?](https://openreview.net/forum?id=rc2h1h89aDi) [id=rc2h1h89aDi](https://openreview.net/forum?id=rc2h1h89aDi).

- <span id="page-9-16"></span>Shorten, C., Khoshgoftaar, T. M., and Furht, B. Text data augmentation for deep learning. *Journal of Big Data*, 8, 2021. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:236096559) [org/CorpusID:236096559](https://api.semanticscholar.org/CorpusID:236096559).
- <span id="page-9-1"></span>Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Large language models encode clinical knowledge. *Nature*, 620(7972):172–180, 2023a.
- <span id="page-9-2"></span>Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Hou, L., Clark, K., Pfohl, S., Cole-Lewis, H., Neal, D., et al. Towards expert-level medical question answering with large language models. *arXiv preprint arXiv:2305.09617*, 2023b.
- <span id="page-9-6"></span>Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *arXiv preprint arXiv:2206.04615*, 2022.
- <span id="page-9-8"></span>Tan, Y., Min, D., Li, Y., Li, W., Hu, N., Chen, Y., and Qi, G. Can chatgpt replace traditional kbqa models? an in-depth analysis of the question answering performance of the gpt llm family. In *International Semantic Web Conference*, pp. 348–367. Springer, 2023.
- <span id="page-9-10"></span>Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpaca: A strong, replicable instruction-following model. *Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html*, 3(6):7, 2023.
- <span id="page-9-18"></span>Tirumala, K., Markosyan, A. H., Zettlemoyer, L., and Aghajanyan, A. Memorization without overfitting: Analyzing the training dynamics of large language models. *ArXiv*, abs/2205.10770, 2022. URL [https:](https://api.semanticscholar.org/CorpusID:248986465) [//api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:248986465) [248986465](https://api.semanticscholar.org/CorpusID:248986465).
- <span id="page-9-12"></span>Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-9-15"></span>Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., et al. Zephyr: Direct distillation of lm alignment. *arXiv preprint arXiv:2310.16944*, 2023.
- <span id="page-9-7"></span>Wang, C., Liu, X., Yue, Y., Tang, X., Zhang, T., Jiayang, C., Yao, Y., Gao, W., Hu, X., Qi, Z., et al. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. *arXiv preprint arXiv:2310.07521*, 2023.

- <span id="page-10-4"></span>Wang, R., Tang, D., Duan, N., Wei, Z., Huang, X., Cao, G., Jiang, D., Zhou, M., et al. K-adapter: Infusing knowledge into pre-trained models with adapters. *arXiv preprint arXiv:2002.01808*, 2020.
- <span id="page-10-5"></span>Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A. S., Naik, A., Stap, D., et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. *arXiv preprint arXiv:2204.07705*, 2022.
- <span id="page-10-1"></span>Wu, C., Zhang, X., Zhang, Y., Wang, Y., and Xie, W. Pmcllama: Further finetuning llama on medical papers. *arXiv preprint arXiv:2304.14454*, 2023a.
- <span id="page-10-2"></span>Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., Kambadur, P., Rosenberg, D., and Mann, G. Bloomberggpt: A large language model for finance. *arXiv preprint arXiv:2303.17564*, 2023b.
- <span id="page-10-7"></span>Xiao, S., Liu, Z., Zhang, P., and Muennighoff, N. C-pack: Packaged resources to advance general chinese embedding, 2023.
- <span id="page-10-3"></span>Yang, H., Liu, X.-Y., and Wang, C. D. Fingpt: Opensource financial large language models. *arXiv preprint arXiv:2306.06031*, 2023.
- <span id="page-10-0"></span>Yu, W., Zhu, C., Li, Z., Hu, Z., Wang, Q., Ji, H., and Jiang, M. A survey of knowledge-enhanced text generation. *ACM Computing Surveys*, 54(11s):1–38, 2022.
- <span id="page-10-6"></span>Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al. Lima: Less is more for alignment. *arXiv preprint arXiv:2305.11206*, 2023.

## A. RAG Ablation Study

As mentioned in Section [5,](#page-5-4) we compared various values of K ∈ {0, . . . , 5}, shown in Table [3.](#page-11-1)We were unable to find an optimal value of K per model, per 0/5-shot, or per task. In fact, other than Anatomy that worked well with K = 2 consistently, there seems to be no patterns that aid in predicting the performance per K, unlike the results presented in [\(Lewis](#page-8-8) [et al.,](#page-8-8) [2020\)](#page-8-8) for other setups. Moreover, the gap between the best and worst performing Ks can be large.

<span id="page-11-1"></span>Unfortunately, we must conclude that this additional hyperparameter is unstable. This is a downside of using RAG in practice, and the choice of K cannot be ignored.

|                    |            | # Retrieved documents (k) |       |       |       |       |  |
|--------------------|------------|---------------------------|-------|-------|-------|-------|--|
| Task               | Model      | 1                         | 2     | 3     | 4     | 5     |  |
|                    | Mistral 7B | 0.615                     | 0.681 | 0.630 | 0.644 | 0.622 |  |
| Anatomy (0-shot)   | Llama2 7B  | 0.452                     | 0.496 | 0.452 | 0.459 | 0.474 |  |
|                    | Orca2 7B   | 0.607                     | 0.644 | 0.600 | 0.570 | 0.622 |  |
|                    |            |                           |       |       |       |       |  |
|                    | Mistral 7B | 0.659                     | 0.667 | 0.659 | 0.681 | 0.674 |  |
| Anatomy (5-shot)   | Llama2 7B  | 0.511                     | 0.563 | 0.533 | 0.526 | 0.519 |  |
|                    | Orca2 7B   | 0.622                     | 0.652 | 0.600 | 0.607 | 0.600 |  |
| Astronomy (0-shot) | Mistral 7B | 0.651                     | 0.664 | 0.658 | 0.664 | 0.684 |  |
|                    | Llama2 7B  | 0.461                     | 0.454 | 0.447 | 0.447 | 0.461 |  |
|                    | Orca2 7B   | 0.704                     | 0.730 | 0.737 | 0.763 | 0.724 |  |
|                    |            |                           |       |       |       |       |  |
|                    | Mistral 7B | 0.697                     | 0.645 | 0.664 | 0.664 | 0.704 |  |
| Astronomy (5-shot) | Llama2 7B  | 0.467                     | 0.474 | 0.461 | 0.441 | 0.461 |  |
|                    | Orca2 7B   | 0.730                     | 0.704 | 0.750 | 0.750 | 0.770 |  |
| Biology (0-shot)   | Mistral 7B | 0.736                     | 0.722 | 0.757 | 0.743 | 0.736 |  |
|                    | Llama2 7B  | 0.451                     | 0.472 | 0.486 | 0.465 | 0.479 |  |
|                    | Orca2 7B   | 0.646                     | 0.625 | 0.639 | 0.625 | 0.639 |  |
|                    |            |                           |       |       |       |       |  |
| Biology (5-shot)   | Mistral 7B | 0.722                     | 0.778 | 0.778 | 0.771 | 0.743 |  |
|                    | Llama2 7B  | 0.493                     | 0.521 | 0.493 | 0.479 | 0.472 |  |
|                    | Orca2 7B   | 0.618                     | 0.639 | 0.632 | 0.653 | 0.660 |  |
| Chemistry (0-shot) | Mistral 7B | 0.490                     | 0.480 | 0.450 | 0.420 | 0.430 |  |
|                    | Llama2 7B  | 0.360                     | 0.370 | 0.430 | 0.410 | 0.370 |  |
|                    | Orca2 7B   | 0.400                     | 0.400 | 0.440 | 0.450 | 0.380 |  |
|                    |            |                           |       |       |       |       |  |
|                    | Mistral 7B | 0.480                     | 0.440 | 0.380 | 0.430 | 0.410 |  |
| Chemistry (5-shot) | Llama2 7B  | 0.360                     | 0.380 | 0.390 | 0.330 | 0.320 |  |
|                    | Orca2 7B   | 0.430                     | 0.400 | 0.460 | 0.440 | 0.430 |  |

Table 3. RAG ablation study.

## <span id="page-11-0"></span>B. Paraphrase Examples

Below is the prompt we used to generate paraphrases with GPT-4:

Your task is to paraphrase a text paragraph. The paragraph is given below.

Make sure to keep the same meaning but change the wording. Do not change any factual information.

Try to keep roughly the same length of the original text.

Give NUM PARAPHRASES different paraphrases for each text.

These paraphrases should be as different from each other as possible.

#### Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs

Return a JSON formatted string with one key, called 'paraphrases', and a list of paraphrases. Input paragraph:

Below are some examples. First, an example taken from the following Wikipedia page: "51st International Emmy Awards".

## Paraphrase I

The fifty-first edition of the International Emmy Awards Competition accepted submissions across all categories from December 7, 2022, until the closing date of February 16, 2023.

## Paraphrase II

The International Emmy Awards Competition, now in its 51st year, accepted submissions across all categories starting from December 7, 2022, until the deadline on February 16, 2023.

## Paraphrase III

The fifty-first edition of the International Emmy Awards Competition accepted submissions across all categories from December 7, 2022, until the closing date of February 16, 2023.

The next example was taken from the following Wikipedia page: "2023 Indianapolis mayoral election".

Paraphrase I The mayoral election in Indianapolis took place on November 7, 2023, with preliminary elections occurring on May 2. The sitting mayor, Democrat Joe Hogsett, successfully ran for a third term. Both Hogsett and his Republican opponent, Jefferson Shreve, moved on to the main election.

Paraphrase II On November 7, 2023, citizens of Indianapolis cast their votes to elect their Mayor, following primary elections on May 2. Joe Hogsett, the Democrat already in office, won his bid for a third term. Hogsett and the Republican candidate, Jefferson Shreve, were the two contenders in the final electoral round.

## Paraphrase III

The mayoral election in Indianapolis took place on the 7th of November, 2023, following primary elections that occurred on the 2nd of May. Joe Hogsett, the incumbent Democrat, successfully ran for a third term. Both Hogsett and his Republican challenger, Jefferson Shreve, made it through to the final round of the election.