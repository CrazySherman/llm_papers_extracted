# OPENCODER: THE OPEN COOKBOOK FOR TOP-TIER CODE LARGE LANGUAGE MODELS

Siming Huang<sup>∗</sup> Tianhao Cheng<sup>∗</sup> Jason Klein Liu Jiaran Hao Liuyihan Song Yang Xu J. Yang J.H. Liu Chenchen Zhang Linzheng Chai Ruifeng Yuan Zhaoxiang Zhang Jie Fu Qian Liu Ge Zhang Zili Wang† Yuan Qi Yinghui Xu Wei Chu†

INF M-A-P

![](_page_0_Picture_5.jpeg)

Home Page: <https://opencoder-llm.github.io>

### ABSTRACT

Large language models (LLMs) for code have become indispensable in various domains, including code generation, reasoning tasks and agent systems. While open-access code LLMs are increasingly approaching the performance levels of proprietary models, high-quality code LLMs suitable for rigorous scientific investigation, particularly those with reproducible data processing pipelines and transparent training protocols, remain limited. The scarcity is due to various challenges, including resource constraints, ethical considerations, and the competitive advantages of keeping models advanced. To address the gap, we introduce Open-Coder, a top-tier code LLM that not only achieves performance comparable to leading models but also serves as an "open cookbook" for the research community. Unlike most prior efforts, we release not only model weights and inference code, but also the reproducible training data, complete data processing pipeline, rigorous experimental ablation results, and detailed training protocols for open scientific research. Through this comprehensive release, we identify the key ingredients for building a top-tier code LLM: (1) code optimized heuristic rules for data cleaning and methods for data deduplication, (2) recall of text corpus related to code and (3) high-quality synthetic data in both annealing and supervised finetuning stages. By offering this level of openness, we aim to broaden access to all aspects of a top-tier code LLM, with OpenCoder serving as both a powerful model and an open foundation to accelerate research, and enable reproducible advancements in code AI.

<span id="page-0-0"></span>![](_page_0_Figure_9.jpeg)

Figure 1: OpenCoder surpasses all previous fully open models (i.e., with open model weights and reproducible datasets) and other open-access models (i.e., with open model weights only) at the 6B+ parameter scale, pushing the frontier of fully open models to new heights.

<sup>∗</sup>The first two authors contributed equally to this work. Work done during the internships of Siming Huang, Tianhao Cheng and Jason Klein Liu at INF. † Correspondence to Wei Chu (chuwei@inftech.ai) and Zili Wang (ziliwang.do@gmail.com).

### CONTENTS

| 1 |                   | Introduction                                                           | 4  |  |  |  |  |  |  |
|---|-------------------|------------------------------------------------------------------------|----|--|--|--|--|--|--|
| 2 |                   | Pretraining Data                                                       | 4  |  |  |  |  |  |  |
|   | 2.1               | RefineCode<br>                                                         | 5  |  |  |  |  |  |  |
|   |                   | 2.1.1<br>Raw Code                                                      | 5  |  |  |  |  |  |  |
|   |                   | 2.1.2<br>Code-Related Web Data<br>                                     | 7  |  |  |  |  |  |  |
|   |                   | 2.1.3<br>Summary                                                       | 8  |  |  |  |  |  |  |
|   | 2.2               | Annealing Data<br>                                                     | 8  |  |  |  |  |  |  |
| 3 | Pretraining<br>10 |                                                                        |    |  |  |  |  |  |  |
|   | 3.1               | Model Architecture                                                     | 10 |  |  |  |  |  |  |
|   | 3.2               | Training Details                                                       | 10 |  |  |  |  |  |  |
| 4 |                   | Post Training                                                          | 11 |  |  |  |  |  |  |
|   | 4.1               | Data Composition                                                       | 11 |  |  |  |  |  |  |
|   | 4.2               | Two-Stage Instruction-Tuning<br>                                       | 12 |  |  |  |  |  |  |
|   | 4.3               | Training Details                                                       | 13 |  |  |  |  |  |  |
|   | 4.4               | Decontamination<br>                                                    | 13 |  |  |  |  |  |  |
| 5 |                   | Experimental Results                                                   | 13 |  |  |  |  |  |  |
|   | 5.1               | Evaluation on Base Models                                              | 13 |  |  |  |  |  |  |
|   | 5.2               | Evaluation on Instruct Model                                           | 14 |  |  |  |  |  |  |
| 6 | Analysis          |                                                                        | 16 |  |  |  |  |  |  |
|   | 6.1               | Analysis of the Deduplication Level<br>                                | 16 |  |  |  |  |  |  |
|   | 6.2               | Analysis on the Importance of High-quality Data In the Annealing Phase | 17 |  |  |  |  |  |  |
|   | 6.3               | Analysis on the Effect of GitHub Stars                                 | 18 |  |  |  |  |  |  |
|   | 6.4               | Analysis on the two-stage instruction tuning strategy<br>              | 19 |  |  |  |  |  |  |
| 7 |                   | Related Work                                                           | 19 |  |  |  |  |  |  |
| 8 |                   | Conclusion & Future Work                                               | 20 |  |  |  |  |  |  |
| A |                   | Filtering Rules                                                        | 27 |  |  |  |  |  |  |
|   | A.1               | Design of Filtering Rules                                              | 27 |  |  |  |  |  |  |
|   | A.2               | Examples of Filtering Rules<br>                                        | 27 |  |  |  |  |  |  |
| B |                   | Analysis on Chunk-level Deduplication                                  | 29 |  |  |  |  |  |  |
| C |                   | Extra Data Processing                                                  | 30 |  |  |  |  |  |  |
|   | C.1               | Chinese Code-Like Domains Annotation                                   | 30 |  |  |  |  |  |  |

|   | C.2<br>Code-Related Data from Github Text Files | 30 |
|---|-------------------------------------------------|----|
|   | C.3<br>Jupyter Notebooks<br>                    | 31 |
| D | Comparison of RefineCode with The Stack Series  | 31 |
| E | Programming Languages Categories                | 31 |
|   | E.1<br>Included Programming Languages<br>       | 31 |
|   | E.2<br>Excluded Programming Languages           | 32 |
| F | Raw Code Data Composition                       | 33 |
| G | Prompts For SFT Synthetic Data                  | 34 |

## <span id="page-3-0"></span>1 INTRODUCTION

Large Language Models (LLMs) have achieved significant success in various domains [\(Wang et al.,](#page-24-0) [2023;](#page-24-0) [Que et al.,](#page-23-0) [2024;](#page-23-0) [Liu et al.,](#page-22-0) [2024a;](#page-22-0)[c;](#page-22-1) [Wu et al.,](#page-24-1) [2024\)](#page-24-1), particularly in code-related tasks, revolutionizing the current paradigm of software development [\(Qian et al.,](#page-23-1) [2024;](#page-23-1) [Wang et al.,](#page-24-2) [2024\)](#page-24-2). Code-specific LLMs have emerged as a critical area within LLM research, with tools such as Chat-GPT, Copilot, and Cursor reshaping the workflows of developers. Despite this, the performance of open-source LLMs focused on code [\(Li et al.,](#page-22-2) [2023;](#page-22-2) [Tao et al.;](#page-24-3) [Lozhkov et al.,](#page-22-3) [2024a;](#page-22-3) [Zhang](#page-25-0) [et al.,](#page-25-0) [2024a\)](#page-25-0) still falls short compared to state-of-the-art LLMs [\(Hui et al.,](#page-21-0) [2024;](#page-21-0) [Zhu et al.,](#page-25-1) [2024\)](#page-25-1), largely because these leading models keep their training datasets—an essential factor in LLM development—proprietary. This lack of transparency limits the broader research community's ability to establish strong baselines and gain deeper insights into the workings of top-tier code LLMs.

To remedy the gap, we set forth three primary goals by releasing OpenCoder and its development material: (1) Firstly, we aim to provide scholars with a meticulously curated and fully transparent strong baseline code LLM for research on mechanical interpretability and the data distribution of code LLMs. (2) Secondly, we intend to conduct in-depth investigations into the pretrain and instruction data curation pipeline for the development of stronger code LLMs. (3) Thirdly, by enabling a detailed review of the development of the models, we hope to unlock more diverse customized solutions based on transparent code LLM. Through OpenCoder, we strive to stimulate and accelerate the growth of the open-source code LLM community.

Our comprehensive set of controlled experiments highlights key design choices for data curation for top-tier code LLMs in different training stages: (1) During the pretraining phase, the importance of data cleaning is highlighted [\(Zhou et al.,](#page-25-2) [2024\)](#page-25-2), emphasizing the removal of non-informative data such as pure hexadecimal code and excessively short code snippets that do not contribute to the learning process. (2) The impact of deduplication is significant, with file-level deduplication proving to be more effective than repository-level deduplication by maintaining data diversity and enhancing model performance on downstream tasks [\(Li et al.,](#page-22-2) [2023\)](#page-22-2). (3) The influence of GitHub stars is also examined, revealing that filtering data based on Github star count can possibly reduce data diversity and affect the overall data distribution, contributing to a suboptimal result [\(Allal et al.,](#page-19-1) [2023\)](#page-19-1). (4) In the annealing phase, the use of high-quality data is crucial for further enhancing the model's capabilities, indicating that data quality is more important than quantity in the later stages of model training. (5) Finally, during the instruction tuning phase, a two-stage instruction tuning strategy is shown to be effective, allowing the model to acquire broad capabilities initially and then refine them with code-specific tasks, resulting in improved performance on both theoretical and practical coding tasks. These five key points underscore the importance of data quality, diversity, and targeted enhancement strategies in developing a high-performing code generation model like OpenCoder.

This work introduces the OpenCoder, a completely open-source Code LLM, built on the transparent data process pipeline and reproducible dataset. As shown in Table [1,](#page-4-2) We provide the open cookbook to build a code LLM from scratch by providing the data cleaning pipeline, reproducible pretraining dataset, large-scale SFT Corpus, and intermediate checkpoints. OpenCoder, through its meticulous data processing and advanced training methods, has surpassed expectations by achieving top-tier results on multiple code LLM evaluation benchmarks. The introduction of the open cookbook of code LLM is designed to push forward the field of code intelligence studies and to encourage its broad use in the community of code intelligence.

### <span id="page-3-1"></span>2 PRETRAINING DATA

Pretraining data plays a crucial role in the development of LLMs, where the scale, quality, and diversity of the data greatly affect the model's overall performance. Therefore, we introduce an efficient and effective methodology for producing data tailored for our code LLM pretraining. In this section, we will comprehensively illustrate the data processing strategies used in both the general pretraining stage and the annealing stage.

<span id="page-4-2"></span>Table 1: The comparison of released resources between our OpenCoder with other popular open-sourced code LLMs. **HumanEval scores are reported for the corresponding chat models**.

| Models           | Data<br>Processing<br>Pipeline | Reproducible<br>Pretraining<br>Dataset | Large-<br>scale SFT<br>Dataset<br>(>1M) | Intermediate<br>Check-<br>points | Training<br>Tokens | <b>HumanEval</b><br>Pass@1 |
|------------------|--------------------------------|----------------------------------------|-----------------------------------------|----------------------------------|--------------------|----------------------------|
|                  | Open                           | Model Weights                          | & Reproducil                            | ole Datasets                     |                    |                            |
| OpenCoder-8B     | <b>✓</b>                       | <b>✓</b>                               | <b>√</b>                                | ✓                                | 2.5T               | 83.5                       |
| StarCoder2-15B   | ✓                              | ✓                                      | X                                       | X                                | 4.1T               | 72.6                       |
| Crystal-7B       | X                              | ✓                                      | X                                       | ✓                                | 1.3T               | 34.1                       |
|                  |                                | Open Mo                                | odel Weights                            |                                  |                    |                            |
| CodeLlama-7B     | X                              | X                                      | X                                       | X                                | 2.5T               | 34.8                       |
| CodeGemma-7B     | X                              | X                                      | X                                       | X                                | 6.5T               | 56.1                       |
| DS-Coder-V2-Lite | X                              | X                                      | X                                       | X                                | 10.2T              | 81.1                       |
| Yi-Coder-9B      | X                              | X                                      | X                                       | X                                | 6.0T               | 85.4                       |
| Qwen2.5-Coder-7B | X                              | X                                      | X                                       | X                                | 23.5T              | 88.4                       |

#### <span id="page-4-0"></span>2.1 RefineCode

Pretraining data forms the foundation for the capabilities of large language models. In the LLM open-source community, The Stack v2 (Lozhkov et al., 2024a) has provided a valuable code dataset, which significantly facilitates the training of code LLMs. However, the quality of the training part in The Stack v2 is insufficient to train LLMs with top-rated performance. To address this, we present **RefineCode**, a high-quality, reproducible dataset of 960 billion tokens across 607 programming languages, incorporating over 130 language-specific rules with customized weight assignments. This dataset is composed of two main parts: raw code and code-related web data. Specifically, we collect the raw code primarily from GitHub repositories up to November 2023 with non-GitHub data from The Stack v2. Additionally, the code-related web data is primarily sourced from web corpora. A detailed comparison with previous versions of The Stack is provided in the Appendix D. Besides, to ensure both quality and diversity, as shown in Figure 2, we have designed a sophisticated data processing pipeline to produce code pretraining corpus. In the following sections, we have provided a detailed description of our processing pipeline and the details of our **RefineCode** dataset.

#### <span id="page-4-1"></span>2.1.1 RAW CODE

To ensure the curation of high-quality raw code data, we have developed the code-specific data processing pipeline including modules of **preprocessing**, **deduplication**, **transformation**, **filtering**, **data sampling**. The following sections provide the details of these processes.

**Preprocessing** Initially, we exclude files exceeding 8 MB in size, as these are predominantly non-text files, which require considerable resource overhead. Furthermore, given the miscellaneous file types present on GitHub, we restrict our selection to those file types related to programming languages by their file extension referring to *linguist*<sup>1</sup>, and filter those types with low capacity or low quality. Finally, we preserve 607 different types of programming language files. A comprehensive list of the included and excluded programming languages is provided in Appendix E.

**Deduplication** The purpose of deduplication is to construct an unbiased and diverse training set while significantly reducing the data volume. Owing to the extremely high repetition of the source code in Github, we prioritize the deduplication process early in the pipeline and adopt an aggressive file-level deduplication strategy (see elaborate analysis in Section 6.1). More specifically, we leverage both exact deduplication and fuzzy deduplication methods to eliminate documents containing identical or near-identical code content shown as follows:

*Exact Deduplication*: Due to the prevalence of forking and copy-pasting within the codebase, nearly 75% of files are completely duplicated. On account of this, differing from general deduplication

<span id="page-4-3"></span>

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 2: The illustration of our pretraining data processing workflow.

process, Identity removal is applied towards code data at the first step in this module. We compute the SHA256 hash value for each document, where files with identical hash values are compared, and only the code files with the highest star count as well as the latest commit time are retained.

*Fuzzy Deduplication*: Following the fuzzy deduplication setting in the general data pipeline, we split the raw text into 5-gram pieces, and then calculate the 2048 MinHash functions (Broder, 1997). Additionally, we utilize LSH (Leskovec et al., 2014) by setting bands to 16 and rows to 128, to retain only those distinct files with the highest stars and latest commit time. This process removes 6% file volume.

**Transformation** Filtering is generally adequate for removing files that fail to meet specific criteria. However, certain issues, though small in text size, are pervasive across numerous files. In such cases, it is unacceptable to exclude all those issued files. Instead, we opt to transform these files to rectify the identified issues before the filtering module. Concretely, we implement two types of transformation rules as follows:

Copyright Removal: There are over 15% code files including the copyright notices at the beginning of the content like "Copyright Intel Corporation (C) 2014-2016", which are highly repetitive and irrelevant to the coding tasks, possibly affecting the performance of the LLM. Consequently, we specifically identified and removed these copyright notices from the initial code comments.

*PII Reduction*: Personally Identifiable Information (PII) encompasses content such as passwords, emails, IP addresses. Training on those data containing PII implies significant privacy risks. Therefore, we employ complex regular expressions to detect such information and replace them with placeholders such as "<name>" and "password>".

Filtering The quality of the original code files on GitHub exhibits significant variability, where lower-quality code potentially hinders the LLM pretraining process. Given the distinct nature of code compared to natural language, the criteria for high-quality code differ significantly from those for natural language. Furthermore, different programming languages also exhibit distinct properties. Based on this, we believe that designing a set of detailed heuristic filtering rules tailored specifically to the characteristics of pretraining data is important to enhance the model's capabilities. Drawing inspiration from the principles of high-quality code data proposed in Gunasekar et al. (2023), we consider the following guidelines when designing our filters: 1) Filter out files with poor self-containment; 2) Filter out files with poor or minimal logical structure; 3) Remove files that deviate significantly from standard formatting.

Based on these guidelines and the characteristics of our dataset, our work presents the first heuristic filtering framework by considering the unique characteristics of different programming languages. Based on RedPajama (Computer, 2023), this framework extends and refines the existing rules from StarCoder (Li et al., 2023) to better align with the unique properties of code datasets, resulting in more precise and higher-quality data cleansing. We developed the following three categories of filtering rules:

<span id="page-6-1"></span>![](_page_6_Figure_1.jpeg)

![](_page_6_Figure_2.jpeg)

Figure 3: Visualization on the PCA data distributions of RefineCode and The Stack v2.

- 1. Natural Language Filtering Rules: These rules filter data based on common properties for all text files, such as file size, number of lines, and other general metrics. Both text and code files share these filtering rules.
- 2. General Code Filtering Rules: These rules apply to all code files by filtering data based on general code characteristics, such as the number of variables, average function length, and other common features.
- 3. Language-Specific Filtering Rules: These rules are designed according to the unique characteristics of specific programming languages, such as the frequency of "pass" statements in Python or the use of "goto" statements in C. We have developed these rules for the following eight commonly used programming languages: Python, C, C++, C#, Java, JavaScript, Go, and HTML.

Heuristic rules involve extensive threshold setting. When defining these rules and determining thresholds, we consistently follow a guiding principle: to remove harmful data as much as possible, while ensuring the overall distribution of the dataset is not significantly affected. We outline our motivations for rule design in Appendix [A.1,](#page-26-1) along with a detailed explanation of the tuning process for the corresponding thresholds. Besides, we show the details of several representative rules in Appendix [A.2.](#page-26-2)

Data Sampling We try to preserve the original data distribution as much as possible to maximize the utilization of our cleaned high-quality dataset. However, we downsample certain high-resource programming languages before using our dataset in pretraining. Specifically, we downsample Java data from 409 GB to 200GB, due to its excessive volume compared to other common languages. Additionally, we downsample HTML data from 213GB to 64GB, as HTML files often contain a significant amount of non-informative structured content and lack substantial coding logic. Finally, we produce about 730B tokens in the pretraining stage.

Notably, as illustrated in Figure [3,](#page-6-1) we use PCA to visualize the embeddings extracted from Code-BERT [\(Feng et al.,](#page-21-2) [2020\)](#page-21-2) for The Stack V2 and RefineCode, and observe a clear distinction between these datasets. Specifically, in Figure [3,](#page-6-1) The Stack V2 data shows a greater number of outliers, while the embeddings of RefineCode appear more tightly clustered. Besides, after analyzing the outlier data, we observe the outliers usually show many low-quality patterns, such as pure text comments, hexadecimal-only data, and excessively short code lacking computational logic, which can distort the distribution of the pretraining dataset and ultimately hinder the efficiency of pretraining.

#### <span id="page-6-0"></span>2.1.2 CODE-RELATED WEB DATA

Inspired by the DeepSeekMath [\(Shao et al.,](#page-23-2) [2024\)](#page-23-2), we collect high-quality code-related data corpus from the Common Crawl dataset. Unlike the previous practice in the math domain, due to the lack of open-source fine-gained code corpus, we first annotate 500,000 high-quality code-like data from CommonCrawl using the Autonomous Data Selection [\(Zhang et al.,](#page-25-3) [2024b\)](#page-25-3) method as seed data for training fasttext[\(Joulin et al.,](#page-21-3) [2016\)](#page-21-3). These data serve as the initial code seed corpus.

Table 2: The Composition of RefineCode.

<span id="page-7-2"></span>

| Category              | Data Source            | # Tokens | Percentage |
|-----------------------|------------------------|----------|------------|
|                       | Github Code            | 755 B    | 78.4%      |
| Raw Code Data         | Jupyter Notebooks      | 11 B     | 1.1%       |
|                       | The Stack v2           | 120 B    | 12.5%      |
|                       | Processed CC           | 13 B     | 1.4%       |
| Code-related Web Data | Processed SkyPile      | 3 B      | 0.3%       |
|                       | Processed FineWeb      | 55 B     | 5.7%       |
| OpenSource Data       | Processed AutoMathText | 3 B      | 0.3%       |

As shown in Figure [2,](#page-5-0) the processing pipeline of code-related web data comprises four main components: 1) FastText Model Training: To maintain a controllable vocabulary size in fastText and enable tokenization of Chinese texts using spaces, we first apply the BPE (Byte Pair Encoding) tokenizer to segment the corpus. Subsequently, the open-source FastText framework is utilized for model training. 2) Recall From Common Crawl: We perform recall on Common Crawl to generate the code-related web corpus. 3) Code-related Domain Discovery, we conduct statistical analysis of the recalled data by domain URLs, and define a domain as web pages with the same base URL(e.g. stackoverflow.com), where domains with over 10% of web pages are classified as coderelated. Note that given the scarcity of Chinese data, we provide detailed annotations of domain names related to code and mathematics within the CommonCrawl dataset in the Appendix [C.](#page-29-0) 4) Url Annotation: We manually annotate the URLs associated with code content within these identified domains. For instance, we have identified all content under "stackoverflow.com/questions" as computer technology questions. Then, we include samples with URLs matching "stackoverflow.com/questions", which are not correctly classified by fastText, into our code seed corpus. After three iterations, we obtain about 220G code-related web data. Note that as the iteration progresses, the quantity and diversity of the seed corpus will be better.

We also apply the same recall pipeline to FineWeb [\(Penedo et al.,](#page-23-3) [2024a\)](#page-23-3), Skypile [\(Wei et al.,](#page-24-4) [2023a\)](#page-24-4) and web part of AutoMathText [\(Zhang et al.,](#page-25-3) [2024b\)](#page-25-3) and produce 330G code-related web data in total. Furthermore, we observe that only a very small portion of the textual data in GitHub is also related to natural language text. Therefore, we also train a classifier to determine whether the text is code-related and obtain an additional 178GB code-related web data.

#### <span id="page-7-0"></span>2.1.3 SUMMARY

Ultimately, we curated a high-quality code pretraining dataset, RefineCode, consisting of about 960 billion tokens. The composition of the data sources is illustrated in Table [2,](#page-7-2) while the distribution of different program languages is displayed in Figure [4.](#page-8-0) For more details regarding the data composition of different program languages, please refer to Appendix [F.](#page-32-0) To demonstrate the efficacy of RefineCode, we train a 1.5B code LLM up to 600B using data from RefineCode and the training subset of The Stack v2 respectively. The results in Figure [1,](#page-0-0) indicate that RefineCode significantly improves training efficiency compared to The Stack v2, highlighting the superiority of our dataset.

#### <span id="page-7-1"></span>2.2 ANNEALING DATA

The annealing stage can be seen as a bridge between the general pretraining stage and the supervised fine-tuning (SFT) stage. Following the training strategy in MiniCPM [\(Hu et al.,](#page-21-4) [2024\)](#page-21-4), our model also undergoes a rapid learning rate annealing phase after the general pretraining stage, where very high-quality training data is used to further enhance the model's capabilities. In addition to the RefineCode from the original distribution, we further incorporated the Algorithmic Corpus and synthetic data during the annealing phase. The detailed data mixture can be found in Table [3.](#page-9-3)

Original Distribution Data In the annealing stage, it's necessary to ensure that the overall data distribution remains similar to the pretraining phase. A significant distribution shift can lead to catastrophic forgetting in the model's knowledge, and we ensure that 84% of the annealing data

<span id="page-8-0"></span>![](_page_8_Figure_1.jpeg)

Figure 4: The distribution of top program languages in **RefineCode**.

comes from the original distribution of RefineCode. Note that given the limited computing budget available, this mixture ratio might not be ideal.

**Algorithmic Corpus** Algorithmic code files exhibit strong code logic and minimal dependency on external files, demonstrating excellent self-containment. Additionally, these files are more aligned with the distribution of smaller, independent tasks commonly encountered in real-world interactive scenarios. Therefore, we sample a certain proportion of the original pretraining data that contains keywords such as "leetcode,", "def solution," or "class solution" to create this corpus.

**Synthetic Data** High-quality pretraining data rewriting is also extremely important during the pretraining stage, which helps the model memorize and embed knowledge for efficient retrieval (Allen-Zhu & Li, 2023). We select Algorithmic Corpus as the seed because it encompasses a wide range of algorithmic logic. We employed two forms of rewriting enhancement: Verified Code Snippets and Code Textbooks.

- 1. **High Quality Code Snippet:** Inspired by the synthetic CodeExercises dataset in Gunasekar et al. (2023), we utilized the algorithmic corpus as seeds and employ a strong LLM to synthesize a batch of self-contained independent functions along with their corresponding test cases. We retained the data that successfully passed the test cases and included them in the annealing stage dataset. This approach was similarly extended to support multiple program languages.
- 2. **Code Textbooks:** To enable the model to understand code from multiple perspectives, we constructed educational text snippets based on the hqcode <sup>2</sup> dataset using Qwen2-72B-Instruct (Yang et al., 2024). Hqcode is a multilingual code dataset synthesized with GPT-4o-Mini, where each entry describes an independent task and provides a corresponding function as a solution. We engaged LLMs to perform interactive analysis on the code within this dataset, extracting and elaborating on abstract code knowledge. This approach aims to enable the model to learn code from diverse perspectives.

<span id="page-8-1"></span>https://huggingface.co/datasets/yuxiang630/hqcode

Table 3: Detailed data mixture for annealing data.

<span id="page-9-3"></span>

| Category       | Dataset                                     | # Token            |
|----------------|---------------------------------------------|--------------------|
| Original Data  | RefineCode<br>Algorithmic Corpus            | 84.21 B<br>12.44 B |
| Synthetic Data | High Quality Code Snippet<br>Code Textbooks | 2.71 B<br>0.91 B   |

<span id="page-9-4"></span>Table 4: Overview of the key hyperparameters of OpenCoder, including 1.5B and 8B.

|                      | OpenCoder-1.5B  | OpenCoder-8B     |  |
|----------------------|-----------------|------------------|--|
| Layers               | 24              | 32               |  |
| Model Dimension      | 2240            | 4096             |  |
| Attention Heads      | 14              | 32               |  |
| Key / Value Heads    | 14              | 8                |  |
| Activation Function  | SwiGLU          |                  |  |
| Vocab Size           | 96640           |                  |  |
| Positional Embedding | RoPE(θ = 10000) | RoPE(θ = 500000) |  |
| Context Window Size  | 4096            | 8192             |  |

### <span id="page-9-0"></span>3 PRETRAINING

#### <span id="page-9-1"></span>3.1 MODEL ARCHITECTURE

In this section, we provide a detailed overview of our model architecture. As shown in Table [4,](#page-9-4) the models are available in two sizes: 1.5 billion and 8 billion parameters. The 1.5 billion model consists of 24 layers with a hidden size of 2240, 14 attention heads, and 14 key/value heads, supporting a context window size of 4096. The 8 billion model architecture closely follows the Llama-3.1-8B architecture, with 32 layers, a hidden size of 4096, and 8 attention heads. Both models use the SwiGLU activation function and have a vocabulary size of 96,640.

#### <span id="page-9-2"></span>3.2 TRAINING DETAILS

The training process, based on the aforementioned model architecture, involved several critical details. The dataset encompassed both Chinese and English languages, alongside 607 programming languages, the complete list of which is provided in Appendix [E.](#page-30-2)

For the 1.5B model, due to the incomplete data curation, training was performed on 2 trillion tokens over four epochs. Following the pretraining phase, we conducted annealing training on an additional 100 billion tokens. The WSD learning schedule, referenced in MiniCPM [\(Hu et al.,](#page-21-4) [2024\)](#page-21-4), was employed, featuring a warm-up phase of 2,000 steps across 8 billion tokens. The peak learning rate was 3e-4, which remained constant after the warm-up and subsequently decayed exponentially to 1e-5 during the annealing phase. A micro-batch size of 4 and a global batch size of 1024 were used. Training was conducted using Megatron-LM [\(Shoeybi et al.,](#page-23-4) [2020\)](#page-23-4) with distributed optimization and DDP gradient overlap on a cluster of 256 H800 GPUs over a total of 109.5 hours, equating to 28,034 GPU hours.

For the 8B model, the WSD learning schedule was again employed with a warm-up phase covering 8 billion tokens over 2,000 steps. This model was trained for 3.5 epochs on 2.5 trillion tokens, followed by a decay phase with an additional 100 billion tokens. Unlike the 1.5 billion model, which lacked code-related recall data due to incomplete data processing, the 8 billion model incorporated this data during training. The learning rate schedule mirrored that of the 1.5B model. The microbatch size was set to 1, with a TP of 2 and a sequence length of 8192. The global batch size was 1024. Training was conducted on a cluster of 512 H100 GPUs over 187.5 hours, totaling 96,000 GPU hours. It is noteworthy that the first 130,000 steps were trained with a sequence length of 4096 and a global batch size of 2048.

![](_page_10_Figure_1.jpeg)

Figure 5: The illustration of our instruction data synthesis workflow.

#### <span id="page-10-0"></span>4 POST TRAINING

#### <span id="page-10-1"></span>4.1 DATA COMPOSITION

Open-source Training Data To enhance the model training, we collect the open-source instruction corpora from the websites, including Evol-Instruct<sup>3</sup> (Luo et al., 2024), Infinity-Instruct<sup>4</sup>, McE-val<sup>5</sup> (Chai et al., 2024; Yang et al., 2021), where the instruction data is created from the multilingual raw code snippet by language sampling with the fixed ratio. We employ an LLM to perform binary classification on the content of Infinity-Instruct, aiming to extract the segments specifically related to the code. Additionally, we sample real user queries from WildChat (Zhao et al., 2024) and Code-290k-ShareGPT<sup>6</sup>, extracting code-related dialogue histories using LLM and subsequently performing data cleaning. For low-quality responses, we employ a robust LLM to regenerate the content, enhancing the overall data quality. This RealUser-Instruct dataset not only exhibits high diversity but also aligns more closely with real-world problem complexity, focusing on addressing practical issues in authentic scenarios.

**Educational Instruction Synthesis** To ensure the diversity and richness of instruction-tuning datasets, prior work explores using code snippets sampled from real-world sources as seed data (Wei et al., 2023b), subsequently used to synthesize question-answer pairs. This approach is widely adopted in the development of large language models. In synthesizing instruction-tuning datasets for Python code, we enhance the effectiveness of this method. Specifically, we observe that the educational value of the synthesized data largely depends on the quality of the seed data. Thus, during the seed data selection phase, we use a scorer model where the input is a code snippet to

<span id="page-10-2"></span> $<sup>^3 \</sup>verb|https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1|$ 

<span id="page-10-4"></span><span id="page-10-3"></span><sup>&</sup>lt;sup>4</sup>https://huggingface.co/datasets/BAAI/Infinity-Instruct

<sup>5</sup>https://huggingface.co/datasets/Multilingual-Multimodal-NLP/ McEval-Instruct

<span id="page-10-5"></span><sup>6</sup>https://huggingface.co/datasets/cognitivecomputations/ Code-290k-ShareGPT-Vicuna

identify high-quality seed data. By using only high-quality seed data, we ensure that the resulting instruction-tuning dataset includes more educational example responses. Subsequently, we use a teacher model to generate multiple test cases for the code sections in each problem. These test cases are appended to the code snippets and executed using a Python interpreter. Only the data samples that successfully pass the tests are retained. By using this strategy, we maximize the likelihood that the generated data is both syntactically and semantically sound, thereby enhancing the reliability of the dataset.

Package-related Instruction Synthesis Due to a significant amount of outdated package usage in the pre-training data, LLM may sometimes employ methods from older versions of libraries when generating code, leading to suboptimal performance in tasks involving package invocation. For example, Python's extensive ecosystem of libraries—such as NumPy, pandas, and TensorFlow—are frequently updated, with new functions, methods, and best practices emerging over time. As a result, when users query NumPy, the model may give incorrect answers based on outdated information. Furthermore, if the model is significantly affected by outdated library syntax, it may fail to generate correct code, leading to errors when the code is executed in a Python interpreter. This problem undermines the model's ability to use tool calls to improve performance. To mitigate the impact of outdated programming syntax and obsolete external library interfaces in the pre-training dataset, we synthesized a tool usage instruction tuning dataset using up-to-date external library documentation. Specifically, we analyzed commonly used external Python libraries and retrieved API signatures and usage examples for widely used syntax and tools via PyDoc. This information was sent to prompt a teacher model that generated accurate and up-to-date question-answer pairs reflecting current usage. By fine-tuning the model on a curated set of code that includes up-to-date usage of these libraries, we ensured that it could provide accurate, contemporary answers to questions about using them effectively. This is particularly important given the rapid pace of change in software development, where outdated code and obsolete practices can lead to incorrect answers and inefficient solutions.

Large-scale Diverse Instruction Synthesis Following the previous work [\(Yue et al.,](#page-24-8) [2024\)](#page-24-8), to increase the diversity of the instruction dataset, we create a large-scale instruction data synthesis framework. The framework for synthesizing code instruction data using LLMs incorporates the following key components: (1) An LLM is used first to clean the irrelevant context (e.g. advertisements on the web) in the websites and select useful sentences as the seed for further question generation. (2) A task specification module defines programming languages, difficulty levels, and coding task types, utilizing a configuration file for easy customization. The prompt engineering component employs a template-based system to generate diverse, contextually rich prompts, incorporating realworld scenarios and best practices in software development. We set temperature T = 1.0 for diverse questions. (3) An advanced LLM with more parameters first generates the created questions and then generates the corresponding answers. The validation module combines automated code execution and unit testing to check the correctness. (4) Then an LLM is adopted to refine the response by adding code comments and more explanation.

#### <span id="page-11-0"></span>4.2 TWO-STAGE INSTRUCTION-TUNING

In developing a CodeLLM, particularly in computer science and software development, it is essential to ensure that the model excels in both theoretical knowledge and practical coding tasks. To address both needs, we implemented a two-stage instruction fine-tuning process. The detailed composition of instruction tuning is presented in Table [5.](#page-12-4)

The first stage of this fine-tuning process focused on synthesizing question-answer (QA) pairs related to theoretical computer science. Building on general-purpose pre-training data, we created a specialized dataset that enabled the model to develop a deeper understanding of theoretical computer science, such as algorithms, data structures, and networking principles. By fine-tuning the model with domain-specific QA pairs, we ensured that it could respond with greater precision to questions about concepts such as binary search trees, dynamic programming, and the intricacies of object-oriented design patterns.

In the second stage of the fine-tuning process, we shifted focus from theoretical knowledge to practical coding tasks. In this stage, we used high-quality code from GitHub to create a dataset aimed at improving the model's ability to generate and work with code. By fine-tuning the model on high-

<span id="page-12-4"></span>Table 5: Detailed data composition of our two-stage instruction-tuning.

| Stage  | Data Source                                                                     | # Examples                      |
|--------|---------------------------------------------------------------------------------|---------------------------------|
| Stage1 | RealUser-Instruct<br>Large-scale Diverse-Instruct<br>Filtered Infinity-Instruct | 0.7 M<br>2.3 M<br>1.0 M         |
| Stage2 | McEval-Instruct<br>Evol-Instruct<br>Educational-Instruct<br>Package-Instruct    | 36 K<br>111 K<br>110 K<br>110 K |

quality code from GitHub, we ensured it was exposed to real-world examples of well-maintained and formatted code. One key advantage of using high-quality code in the fine-tuning process is that it enhances the model's ability to generate code that is both syntactically and semantically correct.

The two-stage fine-tuning approach allows the model to excel in theoretical knowledge and practical coding tasks, thereby avoiding the limitations of focusing on only one area. Models that only prioritize theory may struggle with coding, while those focused solely on code generation may lack depth in explaining complex concepts. By refining both areas, the model becomes technically proficient and versatile, able to meet the needs of developers, beginners, and professionals alike.

#### <span id="page-12-0"></span>4.3 TRAINING DETAILS

In the first stage of SFT, we trained for one epoch with a batch size of 4096, a learning rate (LR) of 2e-5, warmup steps set to 100, and a cosine learning rate scheduler.In the second stage of SFT, we trained for three epochs using a batch size of 512, a learning rate of 5e-5, with 100 warmup steps, and the same cosine learning rate scheduler.

#### <span id="page-12-1"></span>4.4 DECONTAMINATION

We applied strict data deduplication for all SFT data. Specifically, we removed any data containing the entry points corresponding to test sets such as HumanEval and MBPP. Additionally, we performed 10-gram deduplication, removing any data with a 10-gram overlap with the test sets.

### <span id="page-12-2"></span>5 EXPERIMENTAL RESULTS

In this section, we conduct a comprehensive and fair evaluation to demonstrate that the model we constructed using cleaned and synthesized data performs comparably to other closed large language models. We also compared the most widely used and powerful open-source language models, including the Crystal and StarCoder series. To further highlight the practicality and effectiveness of our models, we focus on tasks such as code generation, code completion, and code understanding.

#### <span id="page-12-3"></span>5.1 EVALUATION ON BASE MODELS

For base models, we focus on evaluating their code completion ability. Code completion is a fundamental capability that enables code models to tackle complex tasks. This evaluation goal aligns with our optimization objective in the annealing stage, as code completion can be regarded as a special case of the code generation task. To ensure the reproducibility of all results, we used publicly available LLM evaluation framework OpenCodeEval[7](#page-12-5) . For comparing models, we compare open-coder-1.5B with state-of-the-art small language models.

HumanEval & MBPP We selected two widely used code completion benchmarks to evaluate OpenCoder, HumanEval [\(Chen et al.,](#page-20-3) [2021\)](#page-20-3), and MBPP [\(Austin et al.,](#page-20-4) [2021\)](#page-20-4). To further enhance the accuracy of the evaluation, EvalPlus [\(Liu et al.,](#page-22-6) [2024d\)](#page-22-6) extends HumanEval and MBPP into

<span id="page-12-5"></span><sup>7</sup><https://github.com/richardodliu/OpenCodeEval>

Table 6: Performance of various base models on HumanEval, MBPP, and the "complete" task of BigCodeBench. Models trained on reproducible datasets are marked with green.

| Model                      | Size | HumanEval |      | MBPP |       |        | BigCodeBench |      |  |
|----------------------------|------|-----------|------|------|-------|--------|--------------|------|--|
|                            |      | HE        | HE+  | MBPP | MBPP+ | 3-shot | Full         | Hard |  |
| 1B+ Models                 |      |           |      |      |       |        |              |      |  |
| DeepSeek-Coder-1.3B-Base   | 1.3B | 34.8      | 26.8 | 55.6 | 46.9  | 46.2   | 26.1         | 3.4  |  |
| Yi-Coder-1.5B              | 1.5B | 41.5      | 32.9 | 27.0 | 22.2  | 51.6   | 23.5         | 3.4  |  |
| CodeGemma-2B               | 2B   | 31.1      | 16.5 | 51.1 | 43.1  | 45.4   | 23.9         | 7.4  |  |
| Qwen2.5-Coder-1.5B         | 1.5B | 43.9      | 36.6 | 69.2 | 58.6  | 59.2   | 34.6         | 9.5  |  |
| StarCoder2-3B              | 3B   | 31.7      | 27.4 | 60.2 | 49.1  | 46.4   | 21.4         | 4.7  |  |
| OpenCoder-1.5B-Base        | 1.5B | 54.3      | 49.4 | 70.6 | 58.7  | 51.8   | 24.5         | 5.4  |  |
| 6B+ Models                 |      |           |      |      |       |        |              |      |  |
| CodeLlama-7B               | 7B   | 33.5      | 26.2 | 55.3 | 46.8  | 41.4   | 28.7         | 5.4  |  |
| CodeGemma-7B               | 7B   | 39.0      | 32.3 | 50.5 | 40.7  | 55.0   | 38.3         | 10.1 |  |
| DS-Coder-6.7B-Base         | 6.7B | 47.6      | 39.6 | 70.2 | 56.6  | 60.6   | 41.1         | 11.5 |  |
| DS-Coder-V2-Lite-Base(MoE) | 16B  | 40.9      | 34.1 | 71.9 | 59.4  | 62.6   | 30.6         | 8.1  |  |
| CodeQwen1.5-7B-Base        | 7B   | 51.8      | 45.7 | 72.2 | 60.2  | 61.8   | 45.6         | 15.6 |  |
| Yi-Coder-9B                | 9B   | 53.7      | 46.3 | 48.4 | 40.7  | 69.4   | 42.9         | 14.2 |  |
| Qwen2.5-Coder-7B-Base      | 7B   | 61.6      | 53.0 | 76.9 | 62.9  | 68.8   | 45.8         | 16.2 |  |
| Crystal-7B                 | 7B   | 22.6      | 20.7 | 38.6 | 31.7  | 31.0   | 10.8         | 4.1  |  |
| StarCoder2-7B              | 7B   | 35.4      | 29.9 | 54.4 | 45.6  | 55.2   | 27.7         | 8.8  |  |
| StarCoder2-15B             | 15B  | 46.3      | 37.8 | 66.2 | 53.1  | 15.2   | 38.4         | 12.2 |  |
| OpenCoder-8B-Base          | 8B   | 68.9      | 63.4 | 79.9 | 70.4  | 60.6   | 40.5         | 9.5  |  |

HumanEval+ and MBPP+ by adding unique and challenging test cases and correcting inaccurate ground-truth solutions. These results can be used to indicate the model's ability to understand and apply basic Python data structures and knowledge of algorithms. For HumanEval, we report the 0-shot results. For MBPP, we report 3-shots on 500 questions in the test split from original dataset, while the others following EValPlus report results on 378 questions in the sanitized part.

BigCodeBench BigCodeBench [\(Zhuo et al.,](#page-25-5) [2024\)](#page-25-5) is a challenging benchmark for code completion, designed to assess models on their ability to handle complex instructions and make accurate function calls across diverse external libraries. In the Completion setup, models are provided with a function signature and related documentation to generate appropriate code, along with a unit test for the completed function. Covering a range of practical programming tasks, it evaluates models' ability to handle real-world scenarios involving complex, task-specific libraries.

#### <span id="page-13-0"></span>5.2 EVALUATION ON INSTRUCT MODEL

LiveCodeBench LiveCodeBench is a comprehensive, contamination-free benchmark that assesses the reasoning and problem-solving abilities of highly complex algorithmic tasks. The benchmark is continuously updated with new problems from platforms such as LeetCode, AtCoder, and CodeForces, ensuring the challenges remain current and diverse. LiveCodeBench provides a robust measure of a model's ability to handle sophisticated logical processes, which are essential in competitive programming contexts. The instruct models are evaluated on the 2305-2409 data split.

MultiPL-E MultiPL-E extends the HumanEval benchmark to evaluate the code generation capabilities of large language models across multiple languages. MultiPL-E translates tasks into languages such as C++, Java, PHP, TypeScript, C#, Bash, and JavaScript, providing a consistent basis for assessing how models apply their programming skills across different syntaxes and paradigms. We follow the evaluation code of Qwencoder[8](#page-13-1) to systematically measure performance in each language, providing insights into the adaptability and code generation accuracy of LLMs in a multilingual context.

<span id="page-13-1"></span><sup>8</sup><https://github.com/QwenLM/Qwen2.5-Coder>

Table 7: Performance of various chat models on HumanEval, MBPP, the "instruct" task of Big-CodeBench and LiveCodeBench. Models trained on reproducible datasets are marked with green.

| Model                        | Size | Huma<br>HE | anEval<br>HE+ | MBPP | BPP<br>MBPP+ | BigCo<br>Full | deBench<br>Hard | LiveCodeBench Avg |  |  |
|------------------------------|------|------------|---------------|------|--------------|---------------|-----------------|-------------------|--|--|
| 1B+ Models                   |      |            |               |      |              |               |                 |                   |  |  |
| DS-coder-1.3B-Instruct       | 1.3B | 65.2       | 61.6          | 61.6 | 52.6         | 22.8          | 3.4             | 9.3               |  |  |
| Qwen2.5-Coder-1.5B-Instruct  | 1.5B | 70.7       | 66.5          | 69.2 | 59.4         | 32.5          | 6.8             | 15.7              |  |  |
| Yi-Coder-1.5B-Chat           | 1.5B | 67.7       | 63.4          | 68.0 | 59.0         | 24.0          | 6.8             | 11.6              |  |  |
| OpenCoder-1.5B-Instruct      | 1.5B | 72.5       | 67.7          | 72.7 | 61.9         | 33.3          | 11.5            | 12.8              |  |  |
| 6B+ Models                   |      |            |               |      |              |               |                 |                   |  |  |
| DS-Coder-V2-Lite-Instruct    | 16B  | 81.1       | 75.0          | 82.3 | 68.8         | 36.8          | 16.2            | 24.3              |  |  |
| CodeLlama-7B-Instruct        | 7B   | 45.7       | 39.6          | 39.9 | 33.6         | 21.9          | 3.4             | 2.8               |  |  |
| CodeGemma-7B-It              | 7B   | 59.8       | 47.0          | 69.8 | 59.0         | 32.3          | 7.4             | 14.7              |  |  |
| DS-Coder-6.7B-Instruct       | 6.7B | 78.6       | 70.7          | 75.1 | 66.1         | 35.5          | 10.1            | 20.5              |  |  |
| Yi-Coder-9B-Chat             | 9B   | 82.3       | 72.6          | 81.5 | 69.3         | 38.1          | 11.5            | 23.4              |  |  |
| CodeQwen1.5-7B-Chat          | 7B   | 86.0       | 79.3          | 83.3 | 71.4         | 39.6          | 18.9            | 20.1              |  |  |
| Qwen2.5-Coder-7B-Instruct    | 7B   | 88.4       | 84.1          | 83.5 | 71.7         | 41.0          | 18.2            | 37.6              |  |  |
| CrystalChat-7B               | 7B   | 34.1       | 31.7          | 39.1 | 32.7         | 26.7          | 2.3             | 6.1               |  |  |
| StarCoder2-15B-Instruct-v0.1 | 15B  | 72.6       | 63.4          | 75.2 | 61.2         | 37.6          | 12.2            | 20.4              |  |  |
| OpenCoder-8B-Instruct        | 8B   | 83.5       | 78.7          | 79.1 | 69.0         | 40.3          | 16.9            | 23.2              |  |  |

Table 8: Performance of various chat models on the MultiPL-E benchmark across different programming languages.

| Model                       | Size       | Python | Java | C++         | C#   | TS   | JS   | PHP  | Bash | Average |  |
|-----------------------------|------------|--------|------|-------------|------|------|------|------|------|---------|--|
|                             | 1B+ Models |        |      |             |      |      |      |      |      |         |  |
| DS-Coder-1.3B-Instruct      | 1.3B       | 65.2   | 51.9 | 45.3        | 55.1 | 59.7 | 52.2 | 45.3 | 12.7 | 48.4    |  |
| Yi-Coder-1.5B-Chat          | 1.5B       | 67.7   | 51.9 | 49.1        | 57.6 | 57.9 | 59.6 | 52.2 | 19.0 | 51.9    |  |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B       | 71.2   | 55.7 | 50.9        | 64.6 | 61.0 | 62.1 | 59.0 | 29.1 | 56.7    |  |
| OpenCoder-1.5B-Instruct     | 1.5B       | 72.5   | 64.6 | 50.9        | 61.4 | 63.5 | 62.1 | 55.3 | 29.7 | 57.5    |  |
| 6B+ Models                  |            |        |      |             |      |      |      |      |      |         |  |
| DS-Coder-6.7B-Instruct      | 6.7B       | 78.6   | 68.4 | 63.4        | 72.8 | 67.2 | 72.7 | 68.9 | 36.7 | 66.1    |  |
| DS-Coder-V2-Lite-Instruct   | 16B        | 81.1   | 76.6 | <b>75.8</b> | 76.6 | 80.5 | 77.6 | 74.5 | 43.0 | 73.2    |  |
| CodeLlama-7B-Instruct       | 7B         | 45.7   | 32.2 | 28.6        | 32.9 | 39.0 | 43.5 | 31.7 | 10.1 | 33.0    |  |
| CodeGemma-7B-It             | 7B         | 59.8   | 48.1 | 46.6        | 51.9 | 54.7 | 54.0 | 46.6 | 10.1 | 46.5    |  |
| CodeQwen1.5-7B-Chat         | 7B         | 83.5   | 70.9 | 72.0        | 75.9 | 76.7 | 77.6 | 73.9 | 41.8 | 71.6    |  |
| Yi-Coder-9B-Chat            | 9B         | 85.4   | 76.0 | 67.7        | 76.6 | 72.3 | 78.9 | 72.1 | 45.6 | 71.8    |  |
| Qwen2.5-Coder-7B-Instruct   | 7B         | 87.8   | 76.5 | 75.6        | 80.3 | 81.8 | 83.2 | 78.3 | 48.7 | 76.5    |  |
| OpenCoder-8B-Instruct       | 8B         | 83.5   | 72.2 | 61.5        | 75.9 | 78.0 | 79.5 | 73.3 | 44.3 | 71.0    |  |

**McEval** The comprehensive multilingual code evaluation benchmark McEval (Chai et al., 2024) employed a detailed assessment of OpenCoder's programming capabilities across 40 languages. In contrast to MultiPL-E, this benchmark is not derived from HumanEval or MBPP. Figure 6 depicts the results of the multilingual generation task for OpenCoder-8B-Instruct, which comprises nearly 2,000 samples. The figure illustrates that the model exhibits superior multilingual performance compared to other open-source models of comparable size.

**MdEval** OpenCoder is also evaluated on the comprehensive multilingual code debugging benchmark MdEval (Liu et al., 2024e) across 18 languages. In contrast to McEval, this benchmark focuses on the assessment of code debugging, especially for language-specific bugs. Figure 7 shows the results of the multilingual automated program repair task for OpenCoder-8B-Instruct, which comprises nearly 1.2K samples, which demonstrates that OpenCoder can effectively find the bugs and fix them compared to other open-source models of comparable size.

<span id="page-15-2"></span>![](_page_15_Figure_1.jpeg)

Figure 6: The McEval performance of OpenCoder-8B-Instruct in comparison to other open-source 1 code models of comparable size.

<span id="page-15-3"></span>![](_page_15_Figure_3.jpeg)

Figure 7: The MdEval performance of OpenCoder-8B-Instruct in comparison to other open-source 1 code models of comparable size.

### <span id="page-15-0"></span>6 ANALYSIS

#### <span id="page-15-1"></span>6.1 ANALYSIS OF THE DEDUPLICATION LEVEL

Recent studies [\(Lee et al.,](#page-22-8) [2021\)](#page-22-8) have demonstrated the significant performance improvements that can be achieved by deduplicating training datasets for LLM, where MinHash combined with LSH has emerged as the predominant method for deduplication in code training datasets [\(Li et al.,](#page-22-2) [2023;](#page-22-2) [Lozhkov et al.,](#page-22-3) [2024a;](#page-22-3) [Guo et al.,](#page-21-5) [2024;](#page-21-5) [Mishra et al.,](#page-23-5) [2024\)](#page-23-5). Recently, DeepSeekCoder [\(Guo et al.,](#page-21-5) [2024\)](#page-21-5) claims that deduplication is performed at the repository level. However, we conduct extensive experiments on the Python corpus of RefineCode by performing deduplication at both the file and repository levels, respectively.Specifically, the deduplication is conducted at both the file level and repository level across the 485 million Python files available on GitHub, respectively, and then we train two 1.5B LLMs, where the findings are as follows: First, in Table [9,](#page-16-1) the number of retained tokens at the repository level deduplication is almost three times that of the file level deduplication. Second, in Figure [8,](#page-16-2) we compare the downstream performance of the two datasets (i.e., HumanEval and MBPP) during pretraining and observe that the performance of file level deduplication is better than the performance of repository level deduplication a lot. Third, for repository level deduplication, we observe that a substantial portion of 52 billion tokens exhibits complete character-level equivalence with another file. Fourth, when conducting file-level deduplication as a post-processing step on the results of repository-level deduplication, we find that approximately 68 billion tokens (about 68.4% of the data) could be further deduplicated. Our further investigation into chunk-level deduplication revealed no observable benefits, as detailed in the Appendix [B.](#page-28-0)In summary, for largescale code datasets, performing exact deduplication followed by file-level fuzzy deduplication is an efficient and CPU-saving approach.

<span id="page-16-1"></span>Table 9: The statistics for file level deduplication and repository level deduplication on Python code. Rows for file level and repository level represent the number of files and repositories, respectively.

| Deduplication Level | # Total Rows | # Retained Rows | # Retained Tokens |
|---------------------|--------------|-----------------|-------------------|
| File level          | 485,817,123  | 30,488,834      | 32.74 B           |
| Repository level    | 11,037,352   | 7,480,488       | 99.47 B           |

<span id="page-16-2"></span>![](_page_16_Figure_4.jpeg)

Figure 8: Impact of using different deduplication strategies.

#### <span id="page-16-0"></span>6.2 ANALYSIS ON THE IMPORTANCE OF HIGH-QUALITY DATA IN THE ANNEALING PHASE

During the annealing phase of training, we conduct experiments by using different annealing data with different data distributions as shown in Figure [9.](#page-17-1) Similarly, we still train two 1.5B LLMs, where the first is trained by our original annealing data previously introduced and the second is trained by the data without using the high-quality data (i.e., Algorithmic Corpus and the Synthetic Data). From Figure [9,](#page-17-1) we observe that the performance drops a lot when the high-quality training data is removed, which demonstrates the effectiveness of our constructed high-quality data in the annealing phase.

<span id="page-17-1"></span>![](_page_17_Figure_1.jpeg)

Figure 9: Impact of using high-quality data in the annealing stage.

<span id="page-17-2"></span>![](_page_17_Figure_3.jpeg)

Figure 10: Impact of star-based data filtering on model performance.

#### <span id="page-17-0"></span>6.3 Analysis on the Effect of Github Stars

Following SantaCoder (Allal et al., 2023), we also conduct experiments by comparing the performance trained by original code data and the filtered code data based on GitHub Stars, respectively. Specifically, as shown in Figure 10, we train two 1.5B LLMs, where one is trained original data and another is trained by data filtered by GitHub stars (stars>=5), and we have the following findings. First, in Figure 10, we observe that the LLM trained by original data is better than the LLM trained by filter data, which is similar to the results of SantaCoder. Second, in Figure 11, we also provide the training losses of these two LLMs and observe that the loss of the LLM trained by filtered data is fewer than the LLM trained by original data. For this phenomenon, we assume that the data quality is better when using stars as the filter signal, but the diversity is relatively limited compared to the original data. Besides, we find that this effect can be predicted from a single data distribution through visualization alone, without the need for training. As dedicated in Figure 11, star filter significantly impacts the overall data distribution, compromising data diversity. Upon closer examination of the filtered data, we find that it still contains a considerable amount of well-structured, algorithmically rich code. Therefore, we argue that using stars as a filtering criterion is not an optimal choice.

<span id="page-18-2"></span>![](_page_18_Figure_1.jpeg)

Figure 11: Left figure: Losses of using different training data with different distributions. Right figure: Visualization of the embeddings for original data and filtered data. Note that filtering based on the number of stars can reduce data diversity and result in a lower overall loss for pretraining.

#### <span id="page-18-0"></span>6.4 ANALYSIS ON THE TWO-STAGE INSTRUCTION TUNING STRATEGY

We compared three tuning strategies for OpenCoder-1.5B-Instruct: Stage1, Stage1+Stage2, and Mix Training. Table [10](#page-18-3) indicates that the two-stage SFT training can bring consistent improvement in both public benchmarks and real-world scenarios. We observe that the data in Stage 1 exhibits significant diversity, though with relatively lower average quality. In contrast, the data in Stage 2 consists of high-quality, code-specific SFT data. This two-stage SFT strategy allows for the acquisition of broad capabilities in Stage 1, followed by targeted enhancement of code-related tasks in Stage 2. Besides, similar to Chatbot Arena, we adopt the CodeArena test set covering nearly 400 humancreated samples to emulate user code-related prompts in realistic environments. We use GPT-4 as the baseline and use GPT-4 to judge which LLM has a better response, where the reported results are win rate compared to the GPT-4. Table [10](#page-18-3) demonstrates the importance of the two-stage SFT training strategy in the algorithmic benchmarks Evalplus and the realistic benchmarks CodeArena.

<span id="page-18-3"></span>Table 10: Performance of different training strategies across benchmarks. Mix Training refers to the process of combining and shuffling the data from Stage 1 and Stage 2 for joint training.

|                 | HE   | HE+  | MBPP | MBPP+ | BigCodeBench | Code Arena |
|-----------------|------|------|------|-------|--------------|------------|
| Stage1          | 52.4 | 48.1 | 68.7 | 57.4  | 22.1         | 5.3        |
| Stage1 + Stage2 | 70.1 | 64.0 | 74.6 | 64.8  | 31.5         | 6.9        |
| Mix Training    | 55.5 | 51.2 | 52.0 | 58.7  | 23.9         | 3.8        |

## <span id="page-18-1"></span>7 RELATED WORK

Code Large Language Models. The remarkable progress in generative language modeling has sparked numerous studies on AI applications for software engineering [\(Black et al.,](#page-20-5) [2022;](#page-20-5) [Brown](#page-20-6) [et al.,](#page-20-6) [2020;](#page-20-6) [Radford et al.,](#page-23-6) [2019;](#page-23-6) [Touvron et al.,](#page-24-9) [2023;](#page-24-9) [Sun et al.,](#page-24-10) [2024;](#page-24-10) [Chai et al.,](#page-20-2) [2024;](#page-20-2) [Liu et al.,](#page-22-7) [2024e\)](#page-22-7). While proprietary models [\(Achiam et al.,](#page-19-3) [2023;](#page-19-3) [Chen et al.,](#page-20-3) [2021;](#page-20-3) [Chowdhery et al.,](#page-20-7) [2023\)](#page-20-7) achieve significant performance improvements in many code-related benchmark datasets [\(Chen](#page-20-3) [et al.,](#page-20-3) [2021;](#page-20-3) [Hendrycks et al.,](#page-21-6) [2020\)](#page-21-6), the inaccessible model checkpoints hinder further innovation. In contrast, the research community has introduced several open-source models (e.g., CodeGen [\(Ni](#page-23-7)[jkamp et al.,](#page-23-7) [2023a;](#page-23-7)[b\)](#page-23-8), StarCoder [\(Li et al.,](#page-22-2) [2023;](#page-22-2) [Lozhkov et al.,](#page-22-9) [2024b\)](#page-22-9), CodeLlama [\(Roziere et al.,](#page-23-9) [2023\)](#page-23-9) and DeepSeekCoder [\(Guo et al.,](#page-21-5) [2024\)](#page-21-5)), which greatly foster continued innovation in the field.

Code Benchmarks. Code generation models can be leveraged to address programming challenges by interpreting and acting upon input specifications, which involves the automatic creation of programming solutions based on given problem descriptions [\(Athiwaratkun et al.,](#page-19-4) [2023;](#page-19-4) [Austin et al.,](#page-20-4) [2021;](#page-20-4) [Chen et al.,](#page-20-3) [2021;](#page-20-3) [Gu et al.,](#page-21-7) [2024;](#page-21-7) [Lai et al.,](#page-21-8) [2023;](#page-21-8) [Chai et al.,](#page-20-2) [2024;](#page-20-2) [Muennighoff et al.,](#page-23-10) [2024a;](#page-23-10) [Sun et al.,](#page-24-10) [2024\)](#page-24-10). Moreover, many benchmark datasets have been proposed to comprehensively assess code large language models, such as code retrieval [\(Husain et al.,](#page-21-9) [2019;](#page-21-9) [Lu et al.,](#page-22-10) [2021\)](#page-22-10), code translation [\(Yan et al.,](#page-24-11) [2023\)](#page-24-11), code efficiency [\(Du et al.,](#page-21-10) [2024\)](#page-21-10) and the challenging repository-level code completion tasks [\(Allal et al.,](#page-19-1) [2023;](#page-19-1) [Liu et al.,](#page-22-11) [2023a;](#page-22-11) [Shrivastava et al.,](#page-24-12) [2023;](#page-24-12) [Zhang et al.,](#page-25-6) [2023;](#page-25-6) [Deng et al.,](#page-20-8) [2024;](#page-20-8) [Liu et al.,](#page-22-12) [2024b;](#page-22-12) [Deng et al.,](#page-20-8) [2024\)](#page-20-8).

Open Large Language Models. Recently, many open-sourced LLMs have been proposed to empower the open research community and inspire a new wave of innovation. Specifically, many LLMs (e.g., LLaMA [\(Touvron et al.,](#page-24-9) [2023\)](#page-24-9), Mistral [\(Jiang et al.,](#page-21-11) [2023\)](#page-21-11), Qwen [\(Bai et al.,](#page-20-9) [2023\)](#page-20-9), Chat-GLM [\(GLM,](#page-21-12) [2024\)](#page-21-12)), pretraining datasets (e.g., RedPajama [\(Computer,](#page-20-1) [2023\)](#page-20-1), SlimPajama [\(Sobol](#page-24-13)[eva et al.,](#page-24-13) [2023\)](#page-24-13), FineWeb [\(Penedo et al.,](#page-23-11) [2024b\)](#page-23-11)), and chat-related datasets (e.g., WildChat [\(Zhao](#page-25-4) [et al.,](#page-25-4) [2024\)](#page-25-4), LMSYS-Chat-1M [\(Zheng et al.,](#page-25-7) [2023\)](#page-25-7)) are open-sourced, which greatly inspire more research innovations and accelerate the improvements of LLMs. Notably, several fully open LLMs have been introduced, which provide as many details as possible to reproduce high-performance LLMs. For example, in general LLMs, OLMo [\(Groeneveld et al.,](#page-21-13) [2024\)](#page-21-13), OLMoE [\(Muennighoff](#page-23-12) [et al.,](#page-23-12) [2024b\)](#page-23-12), LLM360 [\(Liu et al.,](#page-22-13) [2023b\)](#page-22-13) and MAP-Neo [\(Zhang et al.,](#page-25-0) [2024a\)](#page-25-0) are proposed. These models release not only the final model checkpoint but also many training details (e.g., the data processing pipeline, the pretraining data, and the intermediate checkpoints). In code LLMs, Star-Coder [\(Allal et al.,](#page-19-1) [2023\)](#page-19-1) and StarCoderV2 [\(Lozhkov et al.,](#page-22-3) [2024a\)](#page-22-3) also release high-quality code pretraining corpus.

## <span id="page-19-0"></span>8 CONCLUSION & FUTURE WORK

In this paper, we present OpenCoder, an open LLM specialized in code intelligence that achieves top-tier performance. To advance research transparency and reproducibility, we release our complete training materials, including: the complete data processing pipeline, the reproducible pretraining dataset, the open code SFT dataset, rigorous experimental ablation results, detailed training protocols and intermediate checkpoints. The performance of OpenCoder is on par with leading proprietary models, and it surpasses most previous open-source models at the both 1B+ and 6B+ parameter scale. Furthermore, we conducted a series of ablation analyses on each phase of the code LLM training process, providing valuable insights and recommendations for future code LLM training. We hope the release of OpenCoder can democratize access to all aspects of a top-tier code LLM, serving as both a powerful model and an open foundation to accelerate research and enable reproducible advancements in code AI.

In the future, we will continue to update our model and data consistently, aiming to improve Open-Coder's performance and expand its influence within the community. Our commitment is to ensure that OpenCoder remains at the forefront of technological advancements, providing users with the most efficient and accurate coding assistance possible. By regularly incorporating user feedback and the latest research findings, we strive to build a more robust and versatile platform that can cater to the diverse needs of developers around the world.

### REFERENCES

<span id="page-19-3"></span>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. 2023.

<span id="page-19-1"></span>Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don't reach for the stars! *arXiv preprint arXiv:2301.03988*, 2023.

<span id="page-19-2"></span>Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction. *arXiv preprint arXiv:2309.14316*, 2023.

<span id="page-19-4"></span>Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, Xiaopeng Li, Yuchen Tian, Ming Tan, Wasi Uddin Ahmad, Shiqi Wang, Qing Sun, Mingyue Shang, Sujan Kumar Gonugondla, Hantian Ding, Varun Kumar, Nathan Fulton, Arash Farahani, Siddhartha Jain, Robert Giaquinto, Haifeng Qian, Murali Krishna Ramanathan, Ramesh Nallapati, Baishakhi Ray, Parminder Bhatia, Sudipta

- Sengupta, Dan Roth, and Bing Xiang. Multi-lingual evaluation of code generation models. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://](https://openreview.net/forum?id=Bo7eeXm6An8) [openreview.net/forum?id=Bo7eeXm6An8](https://openreview.net/forum?id=Bo7eeXm6An8).
- <span id="page-20-4"></span>Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *ArXiv preprint*, abs/2108.07732, 2021. URL [https://arxiv.org/abs/2108.](https://arxiv.org/abs/2108.07732) [07732](https://arxiv.org/abs/2108.07732).
- <span id="page-20-9"></span>Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.
- <span id="page-20-5"></span>Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPT-NeoX-20B: An open-source autoregressive language model. In *Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models*, pp. 95– 136, virtual+Dublin, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. bigscience-1.9. URL <https://aclanthology.org/2022.bigscience-1.9>.
- <span id="page-20-0"></span>Andrei Z. Broder. On the resemblance and containment of documents. In Bruno Carpentieri, Alfredo De Santis, Ugo Vaccaro, and James A. Storer (eds.), *Compression and Complexity of SE-QUENCES 1997, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997, Proceedings*, pp. 21–29. IEEE, 1997. doi: 10.1109/SEQUEN.1997.666900. URL [https://doi.org/10.](https://doi.org/10.1109/SEQUEN.1997.666900) [1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900).
- <span id="page-20-6"></span>Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL [https://proceedings.neurips.cc/paper/2020/hash/](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) [1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html).
- <span id="page-20-2"></span>Linzheng Chai, Shukai Liu, Jian Yang, Yuwei Yin, Ke Jin, Jiaheng Liu, Tao Sun, Ge Zhang, Changyu Ren, Hongcheng Guo, et al. Mceval: Massively multilingual code evaluation. *arXiv preprint arXiv:2406.07436*, 2024.
- <span id="page-20-3"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.
- <span id="page-20-7"></span>Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240): 1–113, 2023.
- <span id="page-20-1"></span>Together Computer. Redpajama: an open dataset for training large language models, 2023. URL <https://github.com/togethercomputer/RedPajama-Data>.
- <span id="page-20-8"></span>Ken Deng, Jiaheng Liu, He Zhu, Congnan Liu, Jingxin Li, Jiakai Wang, Peng Zhao, Chenchen Zhang, Yanan Wu, Xueqiao Yin, Yuanxing Zhang, Wenbo Su, Bangyu Xiang, Tiezheng Ge, and

- Bo Zheng. R2c2-coder: Enhancing and benchmarking real-world repository-level code completion abilities of code large language models. *ArXiv*, abs/2406.01359, 2024.
- <span id="page-21-10"></span>Mingzhe Du, Anh Tuan Luu, Bin Ji, Qian Liu, and See-Kiong Ng. Mercury: A code efficiency benchmark for code large language models. *arXiv preprint arXiv:2402.07844*, 2024.
- <span id="page-21-2"></span>Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, et al. Codebert: A pre-trained model for programming and natural languages. *arXiv preprint arXiv:2002.08155*, 2020.
- <span id="page-21-12"></span>Team GLM. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.
- <span id="page-21-13"></span>Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, William Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah Smith, and Hannaneh Hajishirzi. OLMo: Accelerating the science of language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 15789–15809, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
- <span id="page-21-7"></span>Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I Wang. Cruxeval: A benchmark for code reasoning, understanding and execution. 2024.
- <span id="page-21-1"></span>Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. *arXiv preprint arXiv:2306.11644*, 2023.
- <span id="page-21-5"></span>Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming– the rise of code intelligence. *arXiv preprint arXiv:2401.14196*, 2024.
- <span id="page-21-6"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. 2020.
- <span id="page-21-4"></span>Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. *arXiv preprint arXiv:2404.06395*, 2024.
- <span id="page-21-0"></span>Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al. Qwen2. 5-coder technical report. *arXiv preprint arXiv:2409.12186*, 2024.
- <span id="page-21-9"></span>Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. *arXiv preprint arXiv:1909.09436*, 2019.
- <span id="page-21-11"></span>Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L'elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. *ArXiv*, 2023.
- <span id="page-21-3"></span>Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hervé Jégou, and Tomas Mikolov. Fasttext.zip: Compressing text classification models. *arXiv: Computation and Language,arXiv: Computation and Language*, Nov 2016.
- <span id="page-21-8"></span>Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-Tau Yih, Daniel Fried, Sida I. Wang, and Tao Yu. DS-1000: A natural and reliable benchmark for data science code generation. In Andreas Krause, Emma Brunskill, Kyunghyun

- Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp. 18319–18345. PMLR, 2023. URL <https://proceedings.mlr.press/v202/lai23b.html>.
- <span id="page-22-8"></span>Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. *arXiv preprint arXiv:2107.06499*, 2021.
- <span id="page-22-4"></span>Jure Leskovec, Anand Rajaraman, and Jeffrey D. Ullman. *Mining of Massive Datasets, 2nd Ed*. Cambridge University Press, 2014. ISBN 978-1107077232. URL <http://www.mmds.org/>.
- <span id="page-22-2"></span>Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! *arXiv preprint arXiv:2305.06161*, 2023.
- <span id="page-22-0"></span>Jiaheng Liu, Zhiqi Bai, Yuanxing Zhang, Chenchen Zhang, Yu Zhang, Ge Zhang, Jiakai Wang, Haoran Que, Yukang Chen, Wenbo Su, et al. E2-llm: Efficient and extreme length extension of large language models. *arXiv preprint arXiv:2401.06951*, 2024a.
- <span id="page-22-12"></span>Jiaheng Liu, Ken Deng, Congnan Liu, Jian Yang, Shukai Liu, He Zhu, Peng Zhao, Linzheng Chai, Yanan Wu, Ke Jin, Ge Zhang, Zekun Moore Wang, Guoan Zhang, Bangyu Xiang, Wenbo Su, and Bo Zheng. M2rc-eval: Massively multilingual repository-level code completion evaluation. 2024b.
- <span id="page-22-1"></span>Jiaheng Liu, Chenchen Zhang, Jinyang Guo, Yuanxing Zhang, Haoran Que, Ken Deng, Zhiqi Bai, Jie Liu, Ge Zhang, Jiakai Wang, et al. Ddk: Distilling domain knowledge for efficient large language models. *arXiv preprint arXiv:2407.16154*, 2024c.
- <span id="page-22-6"></span>Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. *Advances in Neural Information Processing Systems*, 36, 2024d.
- <span id="page-22-7"></span>Shukai Liu, Linzheng Chai, Jian Yang, Jiajun Shi, He Zhu, Liran Wang, Ke Jin, Wei Zhang, Hualei Zhu, Shuyue Guo, Tao Sun, Jiaheng Liu, Yunlong Duan, Yu Hao, Liqun Yang, Guanglin Niu, Ge Zhang, and Zhoujun Li. Mdeval: Massively multilingual code debugging. *arXiv preprint arXiv:2411.02310*, 2024e.
- <span id="page-22-11"></span>Tianyang Liu, Canwen Xu, and Julian J. McAuley. Repobench: Benchmarking repository-level code auto-completion systems. abs/2306.03091, 2023a. doi: 10.48550/ARXIV.2306.03091. URL <https://doi.org/10.48550/arXiv.2306.03091>.
- <span id="page-22-13"></span>Zhengzhong Liu, Aurick Qiao, Willie Neiswanger, Hongyi Wang, Bowen Tan, Tianhua Tao, Junbo Li, Yuqi Wang, Suqi Sun, Omkar Pangarkar, Richard Fan, Yi Gu, Victor Miller, Yonghao Zhuang, Guowei He, Haonan Li, Fajri Koto, Liping Tang, Nikhil Ranjan, Zhiqiang Shen, Xuguang Ren, Roberto Iriondo, Cun Mu, Zhiting Hu, Mark Schulze, Preslav Nakov, Tim Baldwin, and Eric P. Xing. Llm360: Towards fully transparent open-source llms, 2023b.
- <span id="page-22-3"></span>Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. *arXiv preprint arXiv:2402.19173*, 2024a.
- <span id="page-22-9"></span>Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. 2024b.
- <span id="page-22-10"></span>Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. *arXiv preprint arXiv:2102.04664*, 2021.
- <span id="page-22-5"></span>Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024. URL [https:](https://openreview.net/forum?id=UnUwSIgK5W) [//openreview.net/forum?id=UnUwSIgK5W](https://openreview.net/forum?id=UnUwSIgK5W).

- <span id="page-23-5"></span>Mayank Mishra, Matt Stallone, Gaoyuan Zhang, Yikang Shen, Aditya Prasad, Adriana Meza Soria, Michele Merler, Parameswaran Selvam, Saptha Surendran, Shivdeep Singh, et al. Granite code models: A family of open foundation models for code intelligence. *arXiv preprint arXiv:2405.04324*, 2024.
- <span id="page-23-10"></span>Niklas Muennighoff, Qian Liu, Armel Randy Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. Octopack: Instruction tuning code large language models. In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024a. URL <https://openreview.net/forum?id=mw1PWNSWZP>.
- <span id="page-23-12"></span>Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024b. URL [https://arxiv.org/abs/2409.](https://arxiv.org/abs/2409.02060) [02060](https://arxiv.org/abs/2409.02060).
- <span id="page-23-7"></span>Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, and Yingbo Zhou. Codegen2: Lessons for training llms on programming and natural languages. *arXiv preprint arXiv:2305.02309*, 2023a. URL <https://arxiv.org/abs/2305.02309>.
- <span id="page-23-8"></span>Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In *International Conference on Learning Representations*, 2023b. URL [https:](https://openreview.net/forum?id=iaYcJKpY2B_) [//openreview.net/forum?id=iaYcJKpY2B\\_](https://openreview.net/forum?id=iaYcJKpY2B_).
- <span id="page-23-3"></span>Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro ˇ Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. *arXiv preprint arXiv:2406.17557*, 2024a.
- <span id="page-23-11"></span>Guilherme Penedo, Hynek Kydlícek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin ˇ Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024b.
- <span id="page-23-13"></span>Jim Plotts and Megan Risdal. Meta kaggle code, 2023. URL [https://www.kaggle.com/ds/](https://www.kaggle.com/ds/3240808) [3240808](https://www.kaggle.com/ds/3240808).
- <span id="page-23-1"></span>Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 15174–15186, 2024.
- <span id="page-23-0"></span>Haoran Que, Jiaheng Liu, Ge Zhang, Chenchen Zhang, Xingwei Qu, Yi Ma, Feiyu Duan, Zhiqi Bai, Jiakai Wang, Yuanxing Zhang, Xu Tan, Jie Fu, Wenbo Su, Jiamang Wang, Lin Qu, and Bo Zheng. D-cpt law: Domain-specific continual pre-training scaling law for large language models. *ArXiv*, abs/2406.01375, 2024.
- <span id="page-23-6"></span>Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. *OpenAI preprint*, 2019. URL [https://cdn.openai.com/better-language-models/language\\_models\\_](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [are\\_unsupervised\\_multitask\\_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- <span id="page-23-9"></span>Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. 2023.
- <span id="page-23-2"></span>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. Wu Y.K. Li, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL <https://arxiv.org/abs/2402.03300>.
- <span id="page-23-4"></span>Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020. URL <https://arxiv.org/abs/1909.08053>.

- <span id="page-24-12"></span>Disha Shrivastava, Hugo Larochelle, and Daniel Tarlow. Repository-level prompt generation for large language models of code. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 31693–31715. PMLR, 23–29 Jul 2023. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v202/shrivastava23a.html) [v202/shrivastava23a.html](https://proceedings.mlr.press/v202/shrivastava23a.html).
- <span id="page-24-13"></span>Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023.
- <span id="page-24-10"></span>Tao Sun, Linzheng Chai, Jian Yang, Yuwei Yin, Hongcheng Guo, Jiaheng Liu, Bing Wang, Liqun Yang, and Zhoujun Li. Unicoder: Scaling code large language model via universal code. *arXiv preprint arXiv:2406.16441*, 2024.
- <span id="page-24-3"></span>Tianhua Tao, Junbo Li, Bowen Tan, Hongyi Wang, William Marshall, Bhargav M Kanakiya, Joel Hestness, Natalia Vassilieva, Zhiqiang Shen, Eric P Xing, et al. Crystal: Illuminating llm abilities on language and code. In *First Conference on Language Modeling*.
- <span id="page-24-9"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.
- <span id="page-24-2"></span>Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Opendevin: An open platform for ai software developers as generalist agents. *arXiv preprint arXiv:2407.16741*, 2024.
- <span id="page-24-0"></span>Zekun Moore Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Man Zhang, Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhu Chen, Jie Fu, and Junran Peng. Rolellm: Benchmarking, eliciting, and enhancing roleplaying abilities of large language models. *arXiv preprint arXiv: 2310.00746*, 2023.
- <span id="page-24-4"></span>Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei Lü, Rui Hu, et al. Skywork: A more open bilingual foundation model. *arXiv preprint arXiv:2310.19341*, 2023a.
- <span id="page-24-7"></span>Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Source code is all you need. *arXiv preprint arXiv:2312.02120*, 2023b.
- <span id="page-24-1"></span>Yanan Wu, Jie Liu, Xingyuan Bu, Jiaheng Liu, Zhanhui Zhou, Yuanxing Zhang, Chenchen Zhang, Zhiqi Bai, Haibin Chen, Tiezheng Ge, et al. Conceptmath: A bilingual concept-wise benchmark for measuring mathematical reasoning of large language models. *arXiv preprint arXiv:2402.14660*, 2024.
- <span id="page-24-11"></span>Weixiang Yan, Yuchen Tian, Yunzhe Li, Qian Chen, and Wen Wang. Codetransocean: A comprehensive multilingual benchmark for code translation. 2023.
- <span id="page-24-5"></span>An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024.
- <span id="page-24-6"></span>Jian Yang, Shuming Ma, Haoyang Huang, Dongdong Zhang, Li Dong, Shaohan Huang, Alexandre Muzio, Saksham Singhal, Hany Hassan, Xia Song, and Furu Wei. Multilingual machine translation systems from microsoft for WMT21 shared task. In Loïc Barrault, Ondrej Bojar, Fethi Bougares, Rajen Chatterjee, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Alexander Fraser, Markus Freitag, Yvette Graham, Roman Grundkiewicz, Paco Guzman, Barry Haddow, Matthias Huck, Antonio Jimeno-Yepes, Philipp Koehn, Tom Kocmi, André Martins, Makoto Morishita, and Christof Monz (eds.), *Proceedings of the Sixth Conference on Machine Translation, WMT@EMNLP 2021, Online Event, November 10-11, 2021*, pp. 446–455. Association for Computational Linguistics, 2021. URL <https://aclanthology.org/2021.wmt-1.54>.
- <span id="page-24-8"></span>Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. *arXiv preprint arXiv:2405.03548*, 2024.

- <span id="page-25-6"></span>Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. Repocoder: Repository-level code completion through iterative retrieval and generation. *arXiv preprint arXiv:2303.12570*, 2023. URL <https://arxiv.org/abs/2303.12570>.
- <span id="page-25-0"></span>Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan, Esther Cheng, Jie Liu, Qunshu Lin, et al. Map-neo: Highly capable and transparent bilingual large language model series. *arXiv preprint arXiv:2405.19327*, 2024a.
- <span id="page-25-3"></span>Yifan Zhang, Yifan Luo, Yang Yuan, and Andrew Chi-Chih Yao. Automathtext: Autonomous data selection with language models for mathematical texts. *arXiv preprint arXiv:2402.07625*, 2024b.
- <span id="page-25-4"></span>Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In *The Twelfth International Conference on Learning Representations*, 2024.
- <span id="page-25-7"></span>Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023.
- <span id="page-25-2"></span>Fan Zhou, Zengzhi Wang, Qian Liu, Junlong Li, and Pengfei Liu. Programming every example: Lifting pre-training data quality like experts at scale. *arXiv preprint arXiv:2409.17115*, 2024.
- <span id="page-25-1"></span>Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. *arXiv preprint arXiv:2406.11931*, 2024.
- <span id="page-25-5"></span>Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, et al. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. *arXiv preprint arXiv:2406.15877*, 2024.

## <span id="page-26-0"></span>A FILTERING RULES

### <span id="page-26-1"></span>A.1 DESIGN OF FILTERING RULES

Designing heuristic filtering rules is inherently challenging, often requiring iterative refinement and experimentation to ultimately develop an effective set of rules. Given this complexity, in addition to providing detailed explanations of our designed rules, we will also share the general insights and methodologies we have accumulated throughout the designing process. We believe that this section will offer valuable guidance for designing heuristic filtering rules applicable to any dataset, thereby significantly enhancing the efficiency of constructing an effective data cleaning pipeline.

Heuristic rules filter data based on specific characteristics of a file, which, for each file, are ultimately expressed as a score representing the file's attribute and a corresponding threshold set by the rule. During the rule design process, we found that understanding the distribution of scores and the impact of different threshold settings on data filtering is critical to creating effective rules. Therefore, based on the approach used in RedPajama [\(Computer,](#page-20-1) [2023\)](#page-20-1), we decompose the heuristic filtering process into two steps: quality signal computation and filtering execution. The quality signal computation calculates the scores for all rules for each file, while the filtering execution module decides whether a file is retained based on its quality signal scores and the corresponding thresholds.

Additionally, we recommend placing the heuristic filtering process as late as possible in the overall data pipeline. Unlike other, more fixed stages of the data processing pipeline, this stage requires frequent adjustments based on the final quality of the data. Placing it later in the process allows for more precise control over the data and minimizes the need to repeat subsequent steps after this filtering module.

The specific steps for designing our heuristic filtering rules are as follows:

- 1. Quality Signals Designing: Based on the definition of low-quality data and the attributes of the dataset, we firstly design a series of quality signals that describe the attributes contributing to file quality.
- 2. Coarse Threshold Tuning: Referring to the definition of low-quality data and the distribution of quality signal scores, we roughly set filtering thresholds for all rules at once. We then apply the filters to obtain an initial version of the filtered dataset.
- 3. Fine-grained Threshold Tuning: For each rule, we focus on the data that was exclusively affected by that specific rule, meaning it did not trigger other filters. This part of the data is directly influenced by the current rule, so we can examine whether the retention or removal of this data under different threshold settings aligns with the intended purpose of the rule. If a rule is effective in improving data quality based on its target attribute, we select the optimal threshold; otherwise, the rule is discarded. After evaluating each rule, we apply the filters again to obtain a more refined filtered dataset.
- 4. Data Quality Inspection: We then assess whether the filtered dataset meets our expectations for the quality of pretraining data. In addition to traditional manual inspection, we introduce a perplexity (PPL)-based method for data quality evaluation. Specifically, we randomly sample a set of data from the filtered dataset and use a high-performing LLM to compute the PPL on these samples. We then examine the top-N and bottom-N samples based on PPL. Generally, extremely low PPL suggests that the data is overly simplistic, containing limited valuable knowledge, while extremely high PPL indicates that the data may lack learnable patterns. Both of them are advisable to be filtered out. We closely inspect both sets of samples and, based on their characteristics, decide whether to add new rules or adjust existing thresholds. This process can be repeated until the dataset reaches the desired quality.

#### <span id="page-26-2"></span>A.2 EXAMPLES OF FILTERING RULES

We elaborate several representative examples about general code filtering rules in Table [11](#page-27-0) and language-specific filtering rules in Table [12](#page-27-1) and explain their rationale. It is essential to note that for general code filtering rules, the threshold values may be slightly adjusted depending on the programming language of the file. For specific threshold values, please refer to our implementation details of the data processing pipeline.

Table 11: Examples of general code filtering rules.

<span id="page-27-0"></span>

| Description                                                                                    | Explanation                                                                                                                                                                                                                                                             | Filtering Quota |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| The proportion of lines in strings<br>with a word count exceeding.                             | Files with too many long<br>strings indicate a lack of<br>code logic.                                                                                                                                                                                                   | score > 0.2     |
| The proportion of characters in<br>words from strings with a charac<br>ter count exceeding 20. | String variables containing<br>long sequences of charac<br>ters are often indicative of<br>meaningless content such as<br>base64 data, Hash encoding,<br>url, etc.                                                                                                      | score > 0.4     |
| The proportion of hexadecimal<br>characters.                                                   | Files with two many hex<br>adecimal characters indicate<br>a lack of code logic.                                                                                                                                                                                        | score > 0.4     |
| The proportion of lines like<br>"you code here", "TODO" or<br>"FIXME".                         | We found that these ele<br>ments tend to be excessively<br>repeated in the dataset,<br>which increases the likeli<br>hood that the model, during<br>code completion, will output<br>placeholders like the ones<br>mentioned above instead of<br>generating actual code. | score > 0.01    |
| The proportion of lines containing<br>an "assert" statement.                                   | Files containing a large<br>number of 'assert' state<br>ments are often test files,<br>which tend to have relatively<br>simple and repetitive code<br>patterns.                                                                                                         | score > 0.4     |

Table 12: Examples of python-specific filtering rules.

<span id="page-27-1"></span>

| Description                                                                          | Explanation                                                                                                                                                          | Filtering Quota |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| The proportion of the number<br>of python functions to the total<br>number of lines. | A higher number of Python<br>functions in a file may in<br>dicate that the functions are<br>overly simple, with limited<br>code logic, or have a bad<br>code format. | score > 0.2     |
| Whether the file can be parsed<br>into an python abstract syntax<br>tree (AST).      | Files that cannot be parsed<br>into an AST contain syntax<br>errors and should be filtered<br>out.                                                                   | score == False  |
| The proportion of lines that are<br>"import" statements.                             | A file with exceeding prpor<br>tion of "import" statements<br>indicates to have sparse<br>code logic.                                                                | score > 0.3     |

## <span id="page-28-0"></span>B ANALYSIS ON CHUNK-LEVEL DEDUPLICATION

During pretraining, data is first randomly concatenated and segmented into chunks of context length, followed by full-attention computation within each chunk. We further explored chunk-level deduplication. Specifically, the pretraining data was randomly concatenated and segmented into chunks of 4096 tokens, followed by MinHash and LSH deduplication on these chunks. Additionally, we applied chunk-level deduplication after file-level and repo-level deduplication.

<span id="page-28-1"></span>Table 13: Comparison of deduplication strategies on Python data. At the File level, "Lines" refers to the number of lines in individual files; at the Repo level, it indicates the line count of aggregated strings; Note that for all deduplication strategies involving the Chunk level, "Lines" specifically refers to 4096-token chunks.

|                          | # Total Lines | # Retained Lines | # Retained Tokens |
|--------------------------|---------------|------------------|-------------------|
| Chunk-level              | 333,007,812   | 79,272,460       | 324.70 B          |
| File-level               | 485,817,123   | 30,488,834       | 32.74 B           |
| File-level + Chunk-level | 333,007,812   | 7,993,164        | 32.70 B           |
| Repo-level               | 11,037,352    | 7,480,488        | 99.47 B           |
| Repo-level + Chunk-level | 333,007,812   | 17,675,781       | 72.40 B           |

<span id="page-28-2"></span>![](_page_28_Figure_5.jpeg)

Figure 12: Comparison of Pass@1 performance on HumanEval & MBPP for different dedup strategies (File-Level, Repo-Level, and Repo-level + Chunk-Level) across RefineCode Python corpus.

From the results in table [13,](#page-28-1) We observe that chunk-level deduplication alone was even less effective than repo-level deduplication, and applying chunk-level deduplication after file-level removed only an additional 0.04B of data. This indicates that chunk-level deduplication is not an effective approach. We pre-trained three 1.5B models on the data retained under file-level, repo-level, and repo-level + chunk-level deduplication strategies. The benchmark results are shown in Figure [12.](#page-28-2) It is evident that file-level deduplication achieves the highest training efficiency, while repo-level + chunk-level deduplication outperforms repo-level alone. We attribute the superior performance of file-level deduplication to its higher degree of data removal. Overall, we conclude that file-level deduplication is the most suitable method for GitHub data.

## <span id="page-29-0"></span>C EXTRA DATA PROCESSING

### <span id="page-29-1"></span>C.1 CHINESE CODE-LIKE DOMAINS ANNOTATION

The manual annotation of the URLs of the website is presented as shown in the table [14.](#page-29-3) For future new CC datasets, we can sample pages in these domains as initial seed corpus.

<span id="page-29-3"></span>Table 14: We manually annotate code-like and math-like Chinese domains, utilizing the '%' symbol as a wildcard in our pattern matching. For example, the URL 'https://my.oschina.net/u/4/blog/11' is matched by the pattern '%my.oschina.net%blog%'.

| Domain               | Prefix                                    | Tag  |
|----------------------|-------------------------------------------|------|
| cloud.tencent.com    | %cloud.tencent.com/developer/article%     | Code |
| cloud.tencent.com    | %cloud.tencent.com/ask%                   | Code |
| cloud.tencent.com    | %cloud.tencent.com/developer/information% | Code |
| cloud.tencent.com    | %cloud.tencent.com/document%              | Code |
| my.oschina.net       | %my.oschina.net%blog%                     | Code |
| ask.csdn.net         | %ask.csdn.net/questions%                  | Code |
| www.cnblogs.com      | %www.cnblogs.com%                         | Code |
| forum.ubuntu.org.cn  | %forum.ubuntu.org.cn%                     | Code |
| q.cnblogs.com        | %q.cnblogs.com/q%                         | Code |
| segmentfault.com     | %segmentfault.com/q%                      | Code |
| segmentfault.com     | %segmentfault.com/a%                      | Code |
| woshipm.com          | %woshipm.com/data-analysis%               | Code |
| zgserver.com         | %zgserver.com/server%                     | Code |
| zgserver.com         | %zgserver.com/linux%                      | Code |
| zgserver.com         | %zgserver.com/ubuntu%                     | Code |
| juejin.cn            | %juejin.cn/post%                          | Code |
| jiqizhixin.com       | %jiqizhixin.com/articles%                 | Code |
| help.aliyun.com      | %help.aliyun.com/zh%                      | Code |
| jyeoo.com            | %jyeoo.com%                               | Math |
| www.haihongyuan.com  | %haihongyuan.com%shuxue%                  | Math |
| www.03964.com        | %www.03964.com%                           | Math |
| www.nbhkdz.com       | %www.nbhkdz.com%                          | Math |
| 9512.net             | %9512.net%                                | Math |
| lanxicy.com          | %lanxicy.com%                             | Math |
| bbs.emath.ac.cn      | %bbs.emath.ac.cn%                         | Math |
| math.pro             | %math.pro%                                | Math |
| mathschina.com       | %mathschina.com%                          | Math |
| shuxue.chazidian.com | %shuxue.chazidian.com%                    | Math |
| shuxue.ht88.com      | %shuxue.ht88.com%                         | Math |

#### <span id="page-29-2"></span>C.2 CODE-RELATED DATA FROM GITHUB TEXT FILES

Github Text files primarily consist of content written in natural languages, which includes abundant code-related knowledge. However, we observed that a substantial portion of the dataset is unrelated to code, which is detrimental to the model's ability to learn code-related knowledge. Therefore, we employed the following strategies to extract and retain the code-relevant portions before our filtering module. Firstly, following the strategy used in starcoder [\(Li et al.,](#page-22-2) [2023\)](#page-22-2), we retained the files with "requirement" in the lowercased filename, or if the filename without the extension is one of "readme", "notes", "todo", "description", "cmakelists", in order to ensure that only text files pertinent to coding contexts are preserved. This strategy recalled 3% volume of the whole text part. Additionally, we trained a fasttext model to recall code-related text files and recalled extra 7% file volume from the original text data.

#### <span id="page-30-0"></span>C.3 JUPYTER NOTEBOOKS

Our Jupyter notebook data is sourced from GitHub and Meta Kaggle code (Plotts & Risdal, 2023). We converted this type of data into the *Jupyter-structured* format used in StarCoder (Li et al., 2023), which consists of a triplet of consecutive markdown, code, and code execution results. However, we discarded the *Jupyter-script* format mentioned in StarCoder. Because the code files generated from Jupyter notebook conversions tend to have poor overall code writing standards, and the content in *Jupyter-script* and *Jupyter-structured* formats is highly redundant, making it sufficient to retain only one format.

#### <span id="page-30-1"></span>D COMPARISON OF REFINECODE WITH THE STACK SERIES

Table 15 compares RefineCode with two versions of The Stack. RefineCode not only includes more tokens (960 billion) but also incorporates over 130 rules, significantly more than the 15 rules used in previous versions. Additionally, RefineCode leverages 75 billion web data tokens and introduces language-specific (LS) rules, providing more precise and fine-tuned handling across a wide range of programming languages.

<span id="page-30-4"></span>Table 15: The Comparison of training data between **RefineCode** and series of The Stack. "LS" denotes "Language Specific".

|              | # Tokens | # Languages | # Web Data Tokens | # Rules | LS Rules |
|--------------|----------|-------------|-------------------|---------|----------|
| The Stack v1 | 200 B    | 88          | \                 | ~15     | X        |
| The Stack v2 | 900 B    | 619         | ~30 B             | ~15     | X        |
| RefineCode   | 960 B    | 607         | ~75 B             | ~130    | ✓        |

#### <span id="page-30-2"></span>E PROGRAMMING LANGUAGES CATEGORIES

#### <span id="page-30-3"></span>E.1 INCLUDED PROGRAMMING LANGUAGES

Included programming languages can be categoried into three classes: code, data and text. Among them, the "code" category represents files rich in code logic, while the "data" category primarily consists of files with structured data, and the "text" category refers to files dominated by natural language content. The threshold settings for the filtering rules vary slightly depending on the data type.

Code(470 types): 1C Enterprise, 4D, ABAP, ABAP CDS, AIDL, AL, AMPL, ANTLR, API Blueprint, APL, ASL, ASP.NET, ATS, ActionScript, Ada, Agda, Alloy, Alpine Abuild, AngelScript, Apex, Apollo Guidance Computer, AppleScript, Arc, AspectJ, Assembly, Astro, Asymptote, Augeas, AutoHotkey, AutoIt, Awk, BASIC, BON, Ballerina, Batchfile, Beef, Befunge, Berry, Bikeshed, Bison, BitBake, Blade, BlitzBasic, BlitzMax, Bluespec, Boo, Boogie, Brainfuck, Brightscript, C, C#, C++, C2hs Haskell, CAP CDS, CLIPS, CMake, COBOL, CUE, Cadence, Cairo, CameLIGO, Cap'n Proto, Ceylon, Chapel, Charity, ChucK, Circom, Cirru, Clarion, Clarity, Classic ASP, Clean, Click, Clojure, Closure Templates, CodeQL, CoffeeScript, ColdFusion, ColdFusion CFC, Common Lisp, Common Workflow Language, Component Pascal, Coq, Crystal, Csound, Csound Document, Csound Score, Cuda, Curry, Cycript, Cypher, Cython, D, D2, DIGITAL Command Language, DM, Dafny, Dart, DataWeave, Dhall, Diff, Dockerfile, Dogescript, Dylan, E, ECL, EJS, EQ, Earthly, Edge, EdgeQL, Elixir, Elm, Elvish, Emacs Lisp, EmberScript, Erlang, F#, F\*, FIRRTL, FLUX, Factor, Fancy, Fantom, Faust, Fennel, Filebench WML, Fluent, Forth, Fortran, Fortran Free Form, FreeBasic, Futhark, GAML, GAMS, GAP, GDB, GLSL, GSC, Game Maker Language, Genero 4gl, Genero per, Genshi, Gentoo Ebuild, Gentoo Eclass, Gherkin, Gleam, Glimmer JS, Glyph, Go, Golo, Gosu, Grace, Grammatical Framework, Groovy, Groovy Server Pages, HCL, HLSL, HTML, HTML+ECR, HTML+EEX, HTML+ERB, HTML+PHP, HTML+Razor, Hack, Haml, Handlebars, Harbour, Haskell, Haxe, HiveQL, HolyC, Hy, IDL, IGOR Pro, Idris, ImageJ Macro, Imba, Inform 7, Ink, Inno Setup, Io, Ioke, Isabelle, Isabelle ROOT, J, JCL, JFlex, JSONiq, Janet, Jasmin, Java, Java Server Pages, JavaScript, JetBrains

MPS, Jinja, Jison, Jison Lex, Jolie, Jsonnet, Julia, Just, KRL, Kaitai Struct, KakouneScript, Kerbo-Script, Kit, Kotlin, LFE, LLVM, LOLCODE, LSL, LabVIEW, Latte, Lean, Less, Lex, LigoLANG, LilyPond, Limbo, Liquid, Literate Agda, Literate CoffeeScript, Literate Haskell, LiveScript, Logos, Logtalk, LookML, Lua, Luau, M, M4, M4Sugar, MATLAB, MAXScript, MLIR, MQL4, MQL5, MTML, MUF, Macaulay2, Makefile, Mako, Marko, Mask, Mathematica, Mercury, Mermaid, Meson, Metal, MiniD, Mint, Mirah, Modelica, Modula-3, Module Management System, Mojo, Monkey, MoonScript, Motorola 68K Assembly, Move, Mustache, Myghty, NASL, NSIS, NWScript, Nearley, Nemerle, NetLinx, NetLogo, Nextflow, Nim, Nit, Nix, Nu, NumPy, Nunjucks, OCaml, Oberon, Objective-C++, Objective-J, Omgrofl, Opa, Opal, Open Policy Agent, OpenCL, Open-QASM, OpenSCAD, Ox, Oxygene, Oz, P4, PDDL, PEG.js, PHP, PLSQL, PLpgSQL, Pact, Pan, Papyrus, Parrot, Parrot Assembly, Parrot Internal Representation, Pascal, Pawn, Pep8, Perl, PigLatin, Pike, PogoScript, Polar, Pony, Portugol, PowerBuilder, PowerShell, Praat, Processing, Procfile, Prolog, Promela, Propeller Spin, Pug, Puppet, PureScript, Prover9, Pyret, Python, Q#, QML, QMake, Qt Script, Quake, R, RAML, REALbasic, REXX, RPGLE, RUNOFF, Racket, Ragel, Raku, Rascal, ReScript, Reason, ReasonLIGO, Rebol, Red, Redcode, RenderScript, Ring, Riot, RobotFramework, Roc, Rouge, Ruby, Rust, SAS, SMT, SQF, SQL, Sage, SaltStack, Sass, Scala, Scaml, Scenic, Scheme, Scilab, Self, Shell, ShellSession, Shen, Sieve, Singularity, Slash, Slim, Slint, SmPL, Smali, Smalltalk, Smarty, Smithy, Snakemake, SourcePawn, Squirrel, Stan, Standard ML, Starlark, Stata, Stylus, SugarSS, Svelte, Sway, Swift, SystemVerilog, TI Program, TL-Verilog, TLA, TSX, TXL, Talon, Tcl, Tcsh, Tea, Terraform Template, Thrift, Toit, Turing, Twig, TypeScript, Typst, Unified Parallel C, Uno, UnrealScript, UrWeb, V, VBA, VBScript, VCL, VHDL, Vala, Velocity Template Language, Verilog, Vim Script, Vim Snippet, Visual Basic .NET, Visual Basic 6.0, Volt, Vue, Vyper, WDL, WGSL, WebAssembly, WebIDL, Whiley, Witcher Script, Wollok, Wren, X10, XC, XProc, XQuery, XS, XSLT, Xojo, Xonsh, Xtend, YARA, YASnippet, Yacc, Yul, ZAP, ZIL, Zeek, Zen-Script, Zephir, Zig, Zimpl, eC, fish, hoon, kvlang, mIRC Script, mcfunction, mupad, nesC, ooc, templ, wisp, xBase

Data(115 types): ABNF, ASN.1, Adobe Font Metrics, Altium Designer, Ant Build System, ApacheConf, Avro IDL, BibTeX, Browserslist, CIL, CODEOWNERS, CSON, CSS, Cabal Config, Caddyfile, CartoCSS, Cloud Firestore Security Rules, CoNLL-U, DNS Zone, Darcs Patch, Debian Package Control File, Dotenv, EBNF, Eagle, Easybuild, Ecere Projects, EditorConfig, Edje Data Collection, FIGlet Font, Formatted, GEDCOM, GN, Gemfile.lock, Gerber Image, Git Attributes, Git Config, Glyph Bitmap Distribution Format, Go Checksums, Go Module, Go Workspace, Godot Resource, Gradle, Gradle Kotlin DSL, GraphQL, Graphviz (DOT), HAProxy, HOCON, HTTP, HXML, INI, Ignore List, JAR Manifest, JSON, JSON with Comments, Jest Snapshot, Kusto, Lark, Linker Script, Maven POM, NEON, NL, NPM Config, Nginx, Ninja, ObjDump, Object Data Instance Notation, OpenStep Property List, OpenType Feature File, Option List, PlantUML, PostCSS, Prisma, Protocol Buffer, Protocol Buffer Text Format, Python traceback, RBS, RON, Readline Config, Record Jar, Redirect Rules, Regular Expression, SCSS, SELinux Policy, SPARQL, SSH Config, STAR, STON, ShellCheck Config, Simple File Verification, Soong, Spline Font Database, TOML, TextMate Properties, Turtle, Type Language, Valve Data Format, Wavefront Material, Web Ontology Language, WebAssembly Interface Type, Wget Config, Windows Registry Entries, X BitMap, X Font Directory Index, XCompose, XML, XML Property List, XPages, YAML, YANG, cURL Config, crontab, desktop, dircolors, edn, nanorc

Text(22 types): AsciiDoc, Creole, Gemini, Gettext Catalog, MDX, Markdown, Muse, Org, Pod, Pod 6, RDoc, RMarkdown, Rich Text Format, Roff, SRecode Template, Sweave, TeX, Texinfo, Text, Textile, Wikitext, reStructuredText

#### <span id="page-31-0"></span>E.2 EXCLUDED PROGRAMMING LANGUAGES

2-Dimensional Array, AGS Script, Adblock Filter List, Bicep, COLLADA, CSV, Checksums, DirectX 3D File, E-mail, G-code, Git Revision List, Gnuplot, IRC log, KiCad Layout, KiCad Legacy Layout, KiCad Schematic, Lasso, Linux Kernel Module, Max, Microsoft Developer Studio Project, Microsoft Visual Studio Solution, POV-Ray SDL, Pic, Pickle, PostScript, Public Key, Pure Data, PureBasic, Raw token data, Roff Manpage, STL, SVG, SubRip Text, TSV, Unity3D Asset, Wavefront Object, WebVTT, X PixMap, robots.txt

### <span id="page-32-0"></span>F RAW CODE DATA COMPOSITION

Figure [16](#page-32-1) shows the composition of raw code data for top 85 programming languages in the RefineCode dataset, both after deduplication and filtering process. It can be observed that, after filtering, the proportion of data for different programming languages has shifted significantly, with a notable increase in the representation of commonly used programming languages.

<span id="page-32-1"></span>Table 16: Overview of the data composition of in RefineCode. The items in the table are sorted in descending order according to the file volume after filtering.

|                 |             | After deduplication |          | After filtering |         |          |
|-----------------|-------------|---------------------|----------|-----------------|---------|----------|
| Language        | # Files     | Vol(GB)             | Ratio(%) | # Files         | Vol(GB) | Ratio(%) |
| html            | 141,081,897 | 3,175.4             | 8.56     | 45,100,466      | 582.4   | 18.08    |
| java            | 215,177,833 | 706.8               | 1.90     | 124,751,295     | 474.3   | 14.72    |
| python          | 109,725,362 | 493.3               | 1.33     | 58,640,346      | 271.1   | 8.41     |
| csharp          | 88,825,202  | 364.2               | 0.98     | 57,910,485      | 232.4   | 7.21     |
| javascript      | 190,670,421 | 1,925.0             | 5.19     | 69,579,517      | 226.9   | 7.04     |
| php             | 84,378,361  | 374.4               | 1.01     | 60,089,397      | 222.7   | 6.91     |
| cpp             | 51,362,503  | 375.2               | 1.01     | 38,037,406      | 176.9   | 5.49     |
| go              | 35,649,865  | 301.1               | 0.81     | 26,723,829      | 153.7   | 4.77     |
| typescript      | 40,211,985  | 287.4               | 0.77     | 20,621,755      | 140.4   | 4.35     |
| ruby            | 15,735,042  | 244.5               | 0.66     | 8,285,561       | 122.7   | 3.81     |
| perl            | 16,354,543  | 121.7               | 0.33     | 9,532,620       | 65.6    | 2.04     |
| rust            | 10,605,421  | 63.6                | 0.17     | 6,086,150       | 39.9    | 1.24     |
| r               | 6,132,978   | 92.5                | 0.25     | 4,803,109       | 34.7    | 1.08     |
| swift           | 4,238,754   | 47.9                | 0.13     | 2,938,498       | 31.8    | 0.99     |
| kotlin          | 4,493,548   | 56.4                | 0.15     | 3,123,156       | 29.8    | 0.94     |
| dart            | 4,087,329   | 33.0                | 0.09     | 2,161,462       | 18.5    | 0.57     |
| java-pages      | 6,174,654   | 31.0                | 0.08     | 4,145,336       | 15.4    | 0.48     |
| css             | 39,822,744  | 241.5               | 0.65     | 15,771,061      | 15.3    | 0.47     |
| lua             | 4,027,221   | 116.0               | 0.31     | 2,538,234       | 14.4    | 0.45     |
| xml             | 61,171,289  | 1,934.2             | 5.21     | 3,173,128       | 12.8    | 0.40     |
| scala           | 5,897,567   | 19.7                | 0.05     | 4,204,979       | 11.7    | 0.36     |
| shell           | 12,054,632  | 23.0                | 0.06     | 6,043,070       | 11.2    | 0.35     |
| pascal          | 1,306,130   | 27.8                | 0.07     | 960,497         | 9.5     | 0.29     |
| fortran         | 2,274,663   | 39.7                | 0.10     | 1,218,491       | 8.6     | 0.27     |
| perl6           | 1,943,430   | 16.4                | 0.04     | 1,034,748       | 8.6     | 0.27     |
| rmarkdown       | 1,317,760   | 14.0                | 0.04     | 827,951         | 7.9     | 0.25     |
| html+erb        | 7,618,377   | 11.4                | 0.03     | 4,452,355       | 7.8     | 0.24     |
| smali           | 3,457,531   | 37.9                | 0.10     | 1,408,274       | 7.4     | 0.23     |
| scss            | 18,061,278  | 35.6                | 0.10     | 7,705,822       | 7.4     | 0.23     |
| gettext catalog | 1,100,044   | 51.3                | 0.14     | 442,385         | 6.3     | 0.19     |
| haskell         | 1,746,444   | 24.0                | 0.06     | 1,218,491       | 6.8     | 0.27     |
| tcl             | 253,345     | 4.2                 | 0.01     | 136,171         | 1.0     | 0.03     |
| gradle          | 2,431,985   | 2.9                 | 0.01     | 724,609         | 1.0     | 0.03     |
| scheme          | 357,909     | 4.7                 | 0.01     | 201,170         | 1.0     | 0.03     |
| qml             | 354,756     | 1.8                 | 0.01     | 252,621         | 1.0     | 0.03     |
| mdx             | 795,525     | 6.4                 | 0.17     | 222,013         | 1.0     | 0.03     |
| classic asp     | 220,344     | 2.8                 | 0.08     | 141,236         | 0.9     | 0.03     |
| xbase           | 192,780     | 2.5                 | 0.07     | 80,396          | 0.9     | 0.03     |
| ini             | 7,232,136   | 19.1                | 0.05     | 1,517,099       | 1.3     | 0.04     |
| objective-c++   | 197,416     | 2.4                 | 0.01     | 149,223         | 1.3     | 0.04     |
| motorola68k     | 1,066,095   | 26.5                | 0.07     | 220,218         | 1.2     | 0.04     |
| gap             | 752,261     | 2.6                 | 0.01     | 510,420         | 1.2     | 0.04     |
|                 |             |                     |          |                 |         |          |

## <span id="page-33-0"></span>G PROMPTS FOR SFT SYNTHETIC DATA

#### Prompt for Educational Instruction Synthesis

You are a teaching assistant helping to create a Python programming task from a given code snippet. You must provide the best response to the Python programming task, including reasoning thought, reference solutions, explanation of test cases, and test code.

#### [Code Snippet]

{Code}

Your response must have these parts:

### [Task]

{Create an independent and detailed Python programming task}

#### [Analysis]

{Analyze the task and reason about the given task step by step}

### [Solution]

{Write a high-quality reference solution in a self-contained script that solves the task}

#### [Test]

{Provide ten assert statements to check the correctness of your solution}

#### Prompt for Package-related Instruction Synthesis

You are exceptionally skilled at crafting high-educational level problems and offering precise solutions. Please gain inspiration from the following code snippet to create a highquality programming problem, which is beneficial for learning the use of corresponding libraries. Present your output in two distinct sections: [Problem Description] and [Solution].

#### [Code Snippet]

{Code}

#### [Library Api Requirements]

{Api Requirements}

#### [Library Api Doc]

{Api Doc}

#### Guidelines for each section:

- 1. [Problem Description]: This should be \*\*completely self-contained\*\*, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included. This problem should be \*\*educational for learning the provided Library api, and please explicitly request the use of the relevant package in the question. This question should only concern the writing of \*\*one function\*\*, and you need to be clear about the function name and role of this function.
- 2. [Solution]: Offer a comprehensive, \*\*correct\*\* solution that addresses the [Problem Description] you provided. This solution should follow the standard of corresponding Library Api doc. Please ensure that the Solution only involves answering the Problem, \*\*without addressing the requirements I provided!\*\* Please provide essential explanation abouth this solution, especially the use of requiremed Library Api.

#### Prompt for Large-scale Diverse Instruction Synthesis

You are an expert in designing high-quality programming questions based on the given text.

#### [Guidelines]

- You can draw inspiration from the given text to create the programming questions.
- The created question should be a self-contained question, which does not depend on any external context.
- The created response must contain the complete code snippet.

#### [Given Text]

{Given Text}

### [Created Question]

{Created Question}