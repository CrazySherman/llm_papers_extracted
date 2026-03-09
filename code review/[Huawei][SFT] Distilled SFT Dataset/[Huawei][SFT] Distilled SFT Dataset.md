# Distilling Desired Comments for Enhanced Code Review with Large Language Models

Yongda Yu *Software Institute Nanjing University* Nanjing, China yuyongda@smail.nju.edu.cn

Lei Zhang *Software Institute Nanjing University* Nanjing, China 522023320200@smail.nju.edu.cn

Guoping Rong\* *Software Institute Nanjing University* Nanjing, China ronggp@nju.edu.cn

Haifeng Shen *Faculty of Science and Engineering Southern Cross University* Bilinga, Queensland, Australia haifeng.shen@scu.edu.au

Jiahao Zhang *Software Institute Nanjing University* Nanjing, China 211250031@smail.nju.edu.cn

Haoxiang Yan *Software Institute Nanjing University* Nanjing, China 211250009@smail.nju.edu.cn

Guohao Shi *Software Institute Nanjing University* Nanjing, China 211250033@smail.nju.edu.cn

Dong Shao *Software Institute Nanjing University* Nanjing, China dongshao@nju.edu.cn

Ruiqi Pan *Huawei Technologies Co., Ltd.* Shenzhen, China panruiqi@huawei.com

Yuan Li *Huawei Technologies Co., Ltd.* Shenzhen, China liyuan50@huawei.com

Qiushi Wang *Huawei Technologies Co., Ltd.* Shenzhen, China wangqiushi6@huawei.com

Zhao Tian *Huawei Technologies Co., Ltd.* Shenzhen, China tianzhao@huawei.com

*Abstract*—There has been a growing interest in using Large Language Models (LLMs) for code review thanks to their proven proficiency in code comprehension. The primary objective of most review scenarios is to generate desired review comments (*DRC*s) that explicitly identify issues to trigger code fixes. However, existing LLM-based solutions are not so effective in generating *DRC*s for various reasons such as hallucination. To enhance their code review ability, they need to be fine-tuned with a customized dataset that is ideally full of *DRC*s. Nevertheless, such a dataset is not yet available, while manual annotation of *DRC*s is too laborious to be practical. In this paper, we propose a dataset distillation method, *Desiview*, which can automatically construct a distilled dataset by identifying *DRC*s from a code review dataset. Experiments on the CodeReviewer dataset comprising more than 150K review entries show that *Desiview* achieves an impressive performance of 88.93%, 80.37%, 86.67%, and 84.44% in terms of Precision, Recall, Accuracy, and F1, respectively, surpassing state-of-the-art methods. To validate the effect of such a distilled dataset on enhancing LLMs' code review ability, we first fine-tune the latest LLaMA series (i.e., LLaMA 3 and LLaMA 3.1) to build model *Desiview4FT*. We then enhance the model training effect through KTO alignment by feeding those review comments identified as non-*DRC*s to the LLMs, resulting in model *Desiview4FA*. Verification results indicate that *Desiview4FA* slightly outperforms *Desiview4FT*, while both models have significantly improved against the base models in terms of generating *DRC*s. Human evaluation confirms that both models identify issues more accurately and tend to generate review comments that better describe the issues contained in the code than the base LLMs do.

*Index Terms*—LLM, Automated Code Review, Fine-tuning,

Alignment

# I. INTRODUCTION

Code review is a crucial component of modern software development and has been widely applied in the development of software systems [\[1\]](#page-10-0). The primary objective of most review scenarios is to generate review comments that explicitly identify issues in a code to trigger code fixes before it is executed for quality assurance [\[2\]](#page-10-1), [\[3\]](#page-10-2). We refer to these comments as *DRC*s (Desired Review Comments). Typically, a *DRC* should accurately pinpoint the locations of the issues in the code, correctly describe the nature of the issues, and/or lead to meaningful subsequent repairs to the code. However, as code review is generally a lengthy and costly process [\[3\]](#page-10-2), [\[4\]](#page-10-3), considerable efforts have been made to automate the process by adopting machine learning or deep learning techniques [\[5\]](#page-10-4). In recent years, the emergence of Large Language Models (LLMs) has introduced new possibilities to automated code review [\[2\]](#page-10-1). Owing to their stronger semantic understanding capabilities than traditional machine learning methods and general language models, they have the potential to enable more accurate identification of subtle issues in the code. Additionally, their inherent content generation capabilities allow them to generate better review comments [\[2\]](#page-10-1), [\[5\]](#page-10-4).

However, existing LLM-based solutions may not be able to effectively generate *DRC*s for various reasons such as

|   | Desired review comments                                                                                                                                                                                                                    |                                                                                            |                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                      |  |  |  |  |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|
|   | Original Code Commit                                                                                                                                                                                                                       | Original Review Comment                                                                    | Subsequent code fixes                                                                                                                                                                                                                                         | Observation                                                                                                                                                                                                                                          |  |  |  |  |
| A | if err != nil {<br>-<br>log.G(h.ctx).Warn("cannot get GPU devices list",<br>zap.Error(err))<br>-<br>gpuInfo = make([]gpu.Device, 0)<br>}<br>hardware := &Hardware{                                                                         | Return error instead.                                                                      | if err != nil {<br>+<br>return nil, err<br>}<br>hardware := &Hardware{                                                                                                                                                                                        | The review comment required<br>returning error message, which is<br>implemented in the subsequent fix.                                                                                                                                               |  |  |  |  |
|   |                                                                                                                                                                                                                                            | Undesired review comments                                                                  |                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                      |  |  |  |  |
|   | Original Code Commit                                                                                                                                                                                                                       | Original Review Comment                                                                    | Subsequent code fixes                                                                                                                                                                                                                                         | Observation                                                                                                                                                                                                                                          |  |  |  |  |
| B | }<br>static void serialICConfig(TIM_TypeDef *tim, uint8_t<br>channel, uint16_t polarity)<br>{<br>TIM_ICInitTypeDef TIM_ICInitStructure;                                                                                                    | BTW: It would be great to remove the mhz<br>nonsense - just pass the value around in<br>Hz | }<br>+<br>+// XXX This is almost identical to timerChConfigIC.<br>+// XXX Expensive? Direct register manipulation?<br>+<br>static void serialICConfig(TIM_TypeDef *tim, uint8_t<br>channel, uint16_t polarity)<br>{<br>TIM_ICInitTypeDef TIM_ICInitStructure; | The review comment suggests that it<br>would be better to remove 'mhz' and<br>pass 'hz' value", yet the subsequent<br>code fixes only adding some code<br>comment, which has nothing to do<br>with 'mhz' or 'hz'.                                    |  |  |  |  |
| C | client = None<br>try:<br>- headers = {<br>- 'Authorization': f'Bearer 23498534098845934865984'<br>-<br>}<br>client = Client(<br>base_url="https://us-central1-bynextmonday<br>4ffc3.cloudfunctions.net/securehealth/",<br>headers=headers) | i suggest the removing header here                                                         | client = None<br>try:<br>client = Client(<br>base_url="https://us-central1-bynextmonday<br>4ffc3.cloudfunctions.net/securehealth/",<br>headers=headers)                                                                                                       | The review comment suggests<br>deleting the request header, which<br>has already been implemented by the<br>Diff under review. Such review<br>comments, while also of some value,<br>do not provide information for<br>subsequent code improvements. |  |  |  |  |

<span id="page-1-0"></span>Fig. 1. Examples of desired and undesired review comments in CodeReviewer [\[5\]](#page-10-4) dataset

their inherent characteristic of hallucination [\[6\]](#page-10-5). Among these reasons, a critical one is that they are not effectively finetuned for the code review task [\[7\]](#page-10-6) and a common cause is the lack of an adequate fine-tuning dataset comprising of *DRC*s [\[2\]](#page-10-1). For example, the dataset may contain a considerable proportion of non-*DRC* data (i.e., undesired review comments, cf. Table [I\)](#page-4-0). Figure [1](#page-1-0) illustrates examples of both desired and undesired review comments, which were drawn from one of the commonly used datasets in code review research [\[5\]](#page-10-4). Example A represents a *DRC* as it identifies an issue in the Diff to be reviewed, which has been fixed in the subsequent code. Conversely, Examples B and C depict undesired review comments, as the subsequent code changes indicate that they do not seem to be triggered by these comments. It has been commonly acknowledged that the adequacy of datasets impacts the training effect of LLMs [\[8\]](#page-10-7)–[\[10\]](#page-10-8). As such, to enhance an LLM's code review ability, it needs to be finetuned with a customized dataset that is ideally full of *DRC*s.

An intuitive way to obtain such a dataset is through manual annotation [\[11\]](#page-10-9), [\[12\]](#page-10-10). However, the enormous labor cost (e.g., the dataset [\[5\]](#page-10-4) contains more than 150000 review entries) behind manual annotation [\[13\]](#page-10-11), [\[14\]](#page-10-12) makes it rather impractical. On top of that is its varying quality, which has been repeatedly raised in multiple studies [\[13\]](#page-10-11), [\[15\]](#page-10-13), [\[16\]](#page-10-14). Code review researchers thereby have made various attempts to construct such a customized dataset automatically, most of which have relied on simple keyword or rule-based filtering methods [\[5\]](#page-10-4), [\[17\]](#page-10-15). For instance, some studies employ the 10-line rule, which deems a review desired if it results in modifications within 10 lines in the new version of the code [\[17\]](#page-10-15). Other studies consider the first record of all review comments as desired by excluding the original author's comments [\[5\]](#page-10-4). As these methods lack semantic understanding and analysis of both the review comments and the relevant code, their effectiveness is suboptimal such that there is no guarantee that the customized dataset contains a high proportion of *DRC*s.

This paper aims to fill this gap by proposing a dataset distillation method, *Desiview*, which can automatically construct a distilled dataset that fine-tunes LLMs for code review tasks by identifying *DRC*s from a code review dataset. By employing this method to distinguish between desired and undesired review comments and subsequently constructing a distilled dataset with a high proportion of *DRC*s, we first finetune the latest LLaMA series (i.e., LLaMA 3 and LLaMA 3.1) to build a model *Desiview4FT* and then KTO-align the model to build an enhanced model *Desiview4FA*. The main contributions of this paper are summarized below.

- We propose the *Desiview* method for automatically distilling *DRC*s from a code review dataset. It achieves an accuracy of 86.67% on the CodeReviewer dataset [\[5\]](#page-10-4), surpassing previous methods including the GPT-4o's 76.50%.
- We develop two code review models *Desiview4FT* and *Desiview4FA* by fine-tuning and KTO-aligning the latest LLaMA series with the distilled dataset. Both models have significantly improved against the base models in terms of generating *DRC*s on the CodeReviewer dataset.
- We conduct a human evaluation of the generated review comments. The results indicate that both *Desiview4FT* and *Desiview4FA* identify issues more accurately and tend to generate review comments that better describe the issues contained in the code than the base LLMs do.

The rest of the paper is organized as follows. Section [II](#page-2-0) introduces some related work. Section [III](#page-3-0) describes the research methodology followed by the evaluation process in Section [IV.](#page-5-0) Section [V](#page-7-0) discusses the implications, followed by the validity risks in Section [VI.](#page-9-0) Section [VII](#page-10-16) concludes the paper with a summary of contributions and future work.

### II. RELATED WORK

<span id="page-2-0"></span>In this section, we describe related work to our study, including automated code review and applications of LLMs in software engineering.

## *A. Automated code review*

Code review, as an essential process in software development, has garnered widespread attention from researchers [\[4\]](#page-10-3), [\[18\]](#page-10-17). Given that code review may consume a significant amount of reviewers' effort and time [\[3\]](#page-10-2), [\[18\]](#page-10-17), researchers have increasingly focused on building automated review systems to assist reviewers. An automated review system typically comprises two components: defect detection and review comment recommendation/generation.

Defect detection is used to find potential issues contained in the code snippets under review. For example, DACE [\[19\]](#page-10-18) uses CNN and LSTM techniques to extract Diff features from the code, thereby predicting the quality of code Diff patches. Some pre-trained models also have been used to assess code quality, such as CodeBert [\[20\]](#page-10-19) and CodeT5 [\[21\]](#page-10-20). CodeBert [\[20\]](#page-10-19) is a bimodal pre-training model designed for programming languages and natural language. It performs well in tasks such as natural language-based code search and code documentation generation. CodeT5 [\[21\]](#page-10-20) leverages a unified framework to support both code understanding and generation tasks, thereby facilitating multi-task learning. This method exhibits superior performance compared to previous techniques in several relevant tasks such as code understanding [\[20\]](#page-10-19) and generation [\[22\]](#page-10-21).

Review comment recommendation/generation produces review comments through retrieval or generation methods. For example, CommentFinder [\[23\]](#page-10-22) uses deep learning techniques to retrieve relevant code review comments, thereby reducing the time reviewers spend writing review comments. DCR [\[24\]](#page-10-23) learns the similarity between code commit Diffs and review comments to retrieve review comments related to a specific code commit. CodeReviewer [\[5\]](#page-10-4) achieves notable results in code defect detection, code review comment generation, and code repair tasks by constructing pre-training tasks targeted at code review in an end-to-end manner. LLaMA-Reviewer [\[2\]](#page-10-1) introduces LLMs into code review tasks, using low-parameter fine-tuning techniques to fine-tune LLaMA, achieving impressive results in review comment generation. It is worth noting that the above two studies use the same dataset [\[5\]](#page-10-4) for model training and verification. They assume the existence of review comments indicates ground truth without considering whether the review comments actually pertain to the code fixes.

#### *B. Large language models for software engineering*

Recent years have witnessed widespread applications of LLMs in various software engineering tasks, especially in those related to code. For example, CodeLLaMA [\[25\]](#page-10-24), an LLM by fine-tuning LLaMA2 [\[26\]](#page-10-25) with a large amount of source code, achieves good performance on various code tasks. DeepSeek Coder [\[27\]](#page-10-26) is pre-trained on 2 trillion tokens across more than 80 programming languages, surpassing CodeL-LaMA in code tasks. StarCoder 2 [\[28\]](#page-11-0), trained on 3.3 to 4.3 trillion tokens with carefully selected data, outperforms the 33B parameter DeepSeek Coder using 15.5B parameters. LLaMA3 [\[29\]](#page-11-1), one of the latest versions of the most widely used LLM architecture in the open-source community, has achieved state-of-the-art in multiple tasks. In general, there are three main technical routes for applying LLMs in software engineering – prompt engineering, fine-tuning, and alignment.

Prompt engineering focuses on leveraging the inherent capabilities of large models by carefully constructing prompts and implementing processes to achieve better performance. For example, CodeT [\[30\]](#page-11-2) uses prompt engineering to first guide the large model to generate test code corresponding to the code generation task and then continuously verifies the accuracy of the generated code using the test code, thereby achieving higher code generation accuracy. MapCoder [\[31\]](#page-11-3) employs prompt engineering to construct multi-agent prompting, simulating the cycle of recalling relevant examples, planning, code generation, and debugging in the human development process, achieving state-of-the-art in multiple evaluation sets.

Fine-tuning involves training LLMs with data so that they can solve problems based on the given information without providing examples. Magicoder [\[32\]](#page-11-4) enhances the instruction code generation capability of LLMs through fine-tuning by constructing diverse instruction data for code generation, surpassing ChatGPT in code generation performance on the humaneval dataset using a 7B model. LLaMA-Reviewer [\[2\]](#page-10-1) enhances the review capability of LLMs by fine-tuning them with the CodeReviewer [\[5\]](#page-10-4) dataset, achieving state-of-theart in code review tasks. RepairLLaMA [\[33\]](#page-11-5) fine-tunes the LLaMA series models to endow them with automatic repair capabilities, achieving state-of-the-art on two code repair datasets. Research has shown that the quality of fine-tuning datasets significantly affects the performance of LLMs [\[8\]](#page-10-7). Obtaining higher quality datasets has become an important research direction in fine-tuning LLMs [\[32\]](#page-11-4), [\[34\]](#page-11-6).

Alignment enhances the ability of LLMs to generate valid answers while reducing the probability of generating invalid answers by training them with both desired and undesired datasets [\[35\]](#page-11-7). Large model alignment algorithms are mainly divided into online alignment and offline alignment. Online alignment algorithms involve online sampling, online scoring, and using the scores to optimize the model. Offline alignment algorithms, on the other hand, optimize model performance using only given desired and undesired data. Online alignment algorithms typically consume a lot of resources but usually perform better, while offline alignment algorithms are the opposite [\[36\]](#page-11-8). RLHF [\[11\]](#page-10-9) is the most representative online alignment method, successfully learning human preferences through a reward model and teaching these preferences to LLMs using the PPO algorithm [\[37\]](#page-11-9). Due to the high resource consumption of online alignment algorithms, researchers have turned their attention to offline alignment algorithms. DPO [\[38\]](#page-11-10) is the first proposed offline alignment algorithm, using the LLM itself as the reward model, achieving low-cost alignment of LLMs. However, DPO requires paired data, meaning that a single effective alignment data entry must contain both desired and undesired data under one instruction, which is difficult to obtain in practice. To solve this problem, researchers proposed the KTO [\[39\]](#page-11-11) alignment method, which does not require paired data for alignment and can also perform alignment in situations where the ratio of desired to undesired data is unbalanced.

Many researchers have started using alignment algorithms to improve the performance of software engineering tasks. For example, RLSQM (Reinforcement Learning from Static Quality) [\[40\]](#page-11-12) proposes a novel technique to construct a quality model based on human analysis and optimize the LLM using PPO, surpassing GPT-4 in test code generation tasks. Step-Coder [\[41\]](#page-11-13) scores code based on feedback from the compiler and uses alignment algorithms to enhance the code generation capability of LLMs, achieving state-of-the-art results on test data. PanGu-Coder2 [\[42\]](#page-11-14) proposes a Rank Responses to align Test & Teacher Feedback framework based on alignment technology, effectively improving the performance of LLMs in code generation tasks. Similarly, alignment technology usually requires high-quality data by distinguishing between desired and undesired data to improve model performance. This data is often manually annotated [\[40\]](#page-11-12) or generated based on relevant standards [\[41\]](#page-11-13), [\[42\]](#page-11-14). Construction of high-quality alignment data remains one of the important research directions in the research of alignment techniques [\[43\]](#page-11-15), and, to the best of our knowledge, there is no such work for code review tasks.

## III. METHODOLOGY

<span id="page-3-0"></span>The primary objective of this research is to develop an LLM-based solution that is effective in generating *DRC*s for code review tasks. The pivotal component of our research methodology is to construct a customized dataset that contains a high proportion of *DRC*s with a novel dataset distillation method. Subsequently, with such a dataset, we first fine-tune the base model of LLaMA-3 and LLaMA-3.1 to develop the code review model of *Desiview4FT* and then align *Desiview4FT* to develop an enhanced model of *Desiview4FA*.

## <span id="page-3-1"></span>*A. Desiview: Constructing a distilled dataset*

The proposed *Desiview* dataset distillation method comprises two main steps: (1) identification of *DRC*s, and (2) dataset preparation and pre-processing.

*1) Identification of DRCs:* In theory, during a generation process, an LLM gradually generates content by continuously sampling data from the probability distribution of the next token, in which tokens with higher probabilities are more likely to be selected. When the average probability of the given answer is higher, the model is considered to be more certain about that answer. Based on this principle, researchers [\[44\]](#page-11-16), [\[45\]](#page-11-17) have proposed the concept of 'perplexity' and used it to evaluate models and guide the selection of hyperparameters [\[45\]](#page-11-17). Generally, the definition of 'perplexity' is as follows:

$$\mathbf{PPL}(X) = exp\{-\frac{1}{N} \sum_{i=1}^{N} log P(x_i|x_{< i})\}$$
 (1)

where X = (x0, x1, ..., x<sup>N</sup> ) is the answer to be evaluated, x<sup>i</sup> is the i-th token, logP(x<sup>i</sup> |x<i) is the log-likelihood of the i-th token given the preceding tokens, and N is the total number of tokens to be calculated. Perplexity is used to evaluate the model's ability to uniformly predict a specified set of tokens for a given content. The higher the perplexity, the lower the probability that the model successfully generates the given content, and vice versa.

For a code review task, the reviewer first writes review comments R based on the original code commit Co, denoted by P(R|Co). Subsequently, the developer writes code fixes C<sup>r</sup> based on the review comments R, denoted by P(Cr|Co, R), as shown in the upper left of Fig. [2.](#page-4-1) Since *DRC*s should lead to code fixes, as pointed out in several studies [\[3\]](#page-10-2), we can calculate the *desiredness score* of review comments DS according to the following formula:

$$DS = -(\mathbf{PPL}(P(C_r|C_o, R)) - \mathbf{PPL}(P(C_r|C_o)))$$
 (2)

The formula represents the difference in the perplexity of the code fix with and without the review comments, using a negative sign to align with the human preference that higher scores indicate more desired comments. Generally, when DS > 0, it is considered that the review comments have had a positive impact on the code fix, making them desired. When DS ≤ 0, it is considered that the review comments have not contributed to the code fix or have introduced noises, making them undesired.

*2) Dataset preparation and pre-processing:* We select the CodeReviewer dataset [\[5\]](#page-10-4), one of the most widely adopted datasets in code review research, as the base dataset to construct a distilled dataset. To the best of our knowledge, this is the only public multi-programming language dataset in code review research field that contains the original code submissions (Co), code review comments (R), and subsequent code fixes (Cr), thereby meeting all the requirements for identifying *DRC*s elaborated above. To perform the perplexity calculation, we use a straightforward prompt, as shown in Fig. [3,](#page-4-2) to generate code fixes with and without the review comments. The desiredness score of *DRC*s is then calculated as the difference between the perplexity for both code fixes with and without the review comments. An example of perplexity calculation is shown in Fig. [4.](#page-5-1) We construct the dialogue input using the chat templates of different models and calculate the perplexity of the standard answers to obtain the required perplexity. Note that the perplexity calculation does not involve a content generation process, so it is not affected by errors due to LLM hallucinations. When this score is greater than 0, the review comment is judged to be desired; otherwise, it is considered undesired.

We use four commonly available LLMs to construct a consensus mechanism(i.e., voting) so as to enhance the accuracy of the resultant judgment, including: CodeLlama-13B [\[25\]](#page-10-24), starchat2-15B [\[28\]](#page-11-0), Meta-Llama-3-8B [\[29\]](#page-11-1), and deepseekcoder-6.7B [\[27\]](#page-10-26). The median of the results from the four LLMs is used as the final score to determine the desired review

![](_page_4_Figure_0.jpeg)

<span id="page-4-1"></span>Fig. 2. The process of developing Desiview4FT and Desiview4FA

```
Refine the given code based on the provided code review comment.

The comment is: '{comment}'
The code is: '{code}'
```

<span id="page-4-2"></span>Fig. 3. Code refine template

comments. The identification results of the desired review comments of the training set and testing set in CodeReviewer dataset are shown in Table I. We can observe that less than half of the review comments are DRCs, which have a positive effect on subsequent code fixes. In addition, the proportions of DRCs in the training and test sets are also close to each other, somewhat indicating the reliability of Desiview.

TABLE I
ANALYSIS RESULTS OF CODEREVIEWER DATASET

<span id="page-4-0"></span>

| Dataset Type | Total         | Desired        | Undesired      |
|--------------|---------------|----------------|----------------|
| Training     | 150406 (100%) | 64934 (43.17%) | 85472 (56.83%) |
| Testing      | 13103 (100%)  | 5727 (43.71%)  | 7376 (56.29%)  |

## B. Desiview4FT: Fine-tuning Large Language Models

With the distilled dataset, we fine-tune base LLMs to develop our first code review model *Desiview4FT*. We choose LLaMA series as the base LLMs since they are among the most commonly used models in the open-source community [14]. To be specific, we use both LLaMA-3 [29] and the most recently released LLaMA-3.1 since both LLMs represent the latest models in the LLaMA series. In particular, we use the smallest version of these models, namely LLaMA-3-8B and LLaMA-3.1-8B, due to our GPU resource limitations.

In terms of training methods, we opt to use LoRA [46] for fine-tuning the LLaMA series, thereby reducing the resource requirements. LoRA assumes that the parameter changes during the fine-tuning phase have a low intrinsic rank, allowing

the parameter changes to be decomposed into the product of low-rank matrices, i.e.,  $W' = W_0 + \Delta W = W_0 + BA$ . Here, W' represents the fine-tuned model parameters,  $W_0$  is the set of pre-trained model parameters,  $\Delta W$  is the change in model parameters after fine-tuning,  $B \in \mathbb{R}^{d \times r}$ ,  $A \in \mathbb{R}^{r \times k}$ , with d and k being the dimensions of the model parameters, and satisfying  $r \ll \min(d, k)$ . During training, the original pretrained parameter set  $W_0$  is frozen and does not participate in gradient updates; only B and A are updated. Since the number of parameters in the low-rank matrices is much smaller than that of the original model matrix, it allows for fine-tuning the large model with a minimal number of parameters. The finetuning was conducted using 2 Nvidia A100 40GB GPUs, with the fine-tuning parameters shown in Table II. The prompts used for fine-tuning were based on the LLaMA-Reviewer prompts to facilitate subsequent comparisons, and the code review task is illustrated in Fig. 5.

TABLE II TRAINING HYPERPARAMETERS

<span id="page-4-3"></span>

| Method      | Mathad anasha hatah lu | cutoff | 1    | lora  | lora   |       |         |
|-------------|------------------------|--------|------|-------|--------|-------|---------|
| Method      | epochs                 | batch  | lr   | cuton | lora r | alpha | dropout |
| Fine-tuning | 10                     | 32     | 1e-5 | 2048  | 16     | 32    | 0.05    |
| Alignment   | 5                      | 64     | 1e-5 | 2048  | 16     | 32    | 0.05    |

## C. Desiview4FA: Aligning Large Language Models

While LLM fine-tuning with task-specific data can improve its task performance, LLM alignment goes a step further by ensuring the LLM behaves in accordance with human intentions and values. Therefore, we align model *Desiview4FT* to develop an enhanced code review model *Desiview4FA* by encouraging LLMs to generate desired review comments. LLM alignment typically requires paired data, i.e., a desired answer and an undesired answer under the same prompt. However, in code review, there can be usually only one review comment within a piece of code, making it difficult

```
Code Fix PPL with Review Comment
                                                                                  Code Fix PPL without Review Comment
[INST] <<SYS>>></SYS>> Chat template (Using LLaMA template as an example)
                                                                                  [INST] <<SYS>>></SYS>> Chat template (Using LLaMA template as an example)
Refine the given code based on the provided code review comment.
                                                                                   Refine the given code based on the provided code review comment.
The comment is: 'Should we perhaps disallow reading the 'ObjectHandle' once the
                                                                                  The comment is: "
'Writer' has been made?'
The code is: ' if opts.BeforeWrite != nil {
                                                                                  The code is: ' if opts.BeforeWrite != nil {
     asFunc := func(i interface{}) bool {
                                                                                       asFunc := func(i interface{}) bool {
                                                                                             p = obip
          p = objp
          return true
                                                                                             return true
                                                                                                                                                         Inputs
[/INST]
                                                                                  [/INST]
                                                                                     if opts.BeforeWrite != nil {
   if opts.BeforeWrite != nil {
     asFunc := func(i interface {}) bool {
                                                                                       asFunc := func(i interface{}) bool {
        if p, ok := i.(***storage.ObjectHandle); ok && w == nil {
                                                                                          if p, ok := i.(***storage.ObjectHandle); ok && w == nil {
          p = objp
                                                                                             p = objp
          return true
                                                                                            return true
                                                                                                                                        Calculate Perplexity
```

<span id="page-5-1"></span>Fig. 4. A perplexity calculation example

```
Review the given code and provide a constructive code review comment.

The code/(diff hunk) is: '{} '
```

<span id="page-5-2"></span>Fig. 5. Code Review template

to construct reasonable paired data. Therefore, we choose the KTO algorithm [39], which does not require paired data. The optimization objective of KTO is as follows:

$$L_{KTO}(\pi_{\theta}, \pi_{ref}) = \mathbb{E}_{x,y \sim D}[\lambda_y - v(x, y)]$$

where

$$v(x,y) = \begin{cases} \lambda_D \sigma(\beta(r_\theta(x,y) - z_0)) & \text{if } y \sim y_{desired} | x \\ \lambda_U \sigma(\beta(z_0 - r_\theta(x,y))) & \text{if } y \sim y_{undesired} | x \end{cases}$$

$$r_\theta(x,y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

$$z_0 = \mathbb{E}_{x' \sim D}[\mathbf{KL}(\pi_\theta(y'|x') || \pi_{ref}(y'|x'))]$$

$$\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \frac{4}{3}]$$

 $\pi_{\theta}$  is the model to be optimized, which in this work is the fine-tuned model with the LoRA model superimposed, where LoRA is the trainable part.  $\pi_{ref}$  is the reference model, which in this work is the fine-tuned model. A KL (Kullback–Leibler) divergence penalty is introduced to restrict how far the language model can drift from  $\pi_{ref}$ .  $\lambda_u$  is usually set to 1, and  $\lambda_D$  and  $\lambda_U$  are set according to the ratio of desirable data  $n_D$  and undesirable data  $n_U$  and the constraint  $\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \frac{4}{3}]$ , with  $\lambda_D = 1.7$  and  $\lambda_U = 1.0$ .  $\sigma$  is a nonlinear function, here taken as sigmoid, and  $\beta$  is used to control the degree of risk aversion. The larger the value, the more quickly the value saturates, meaning the model is simultaneously more risk-averse in gains and more risk-seeking in losses. This value is set to 0.1, consistent with the original paper [39]. To reduce the GPU resource requirements for training, alignment also uses LoRA for training. Other training hyperparameters are also shown in Table II.

#### IV. EVALUATION

<span id="page-5-0"></span>In this section, we validate the performance of the *Desiview* dataset distillation method for identifying *DRC*s and examine the effect of using the distilled dataset to fine-tune and align LLMs on their ability to perform code review tasks. Specifically, we aim to answer the following research questions:

- RQ1: How accurately can the dataset distillation method identify *DRCs*?
- RQ2: How much performance enhancement can LLMs gain by being fine-tuned and aligned with the distilled dataset?

RQ1 aims to gauge the effectiveness of the proposed *Desiview* dataset distillation method in identifying desired review comments and subsequently constructing a high-quality distilled dataset compared to that of existing alternative methods. RQ2 aims to test the hypothesis that LLMs fine-tuned and aligned with the distilled dataset (specifically *Desiview4FT* and *Desiview4FA*) can generate more desired review comments than those fine-tuned and aligned with the original dataset (specifically LLaMA-Reviewer).

## A. Experimental settings

<u>Dataset.</u> The base dataset is CodeReviewer dataset [5], which is the only public multi-programming language dataset for code review research in the open-source community and has been widely used in several studies [2], [5]. Besides, as pointed out in Section III-A, it is so far the only publicly available dataset that contains code snippets before and after the review as well as the review comments that meet the needs of this study.

Benchmark approaches. For RQ1, we aim to compare the effectiveness of different methods for identifying *DRCs*. We choose the 10-line rule [17], GPT-3.5, and GPT-40 as the benchmark methods. The first method is one of the few rule-based approaches and has been adopted by several studies [17], [47]. Meanwhile, the GPT family has been widely used in numerous studies as a benchmark method for text comprehension and analysis. The latter, in particular, has been confirmed by many studies as one of the strongest LLMs

![](_page_6_Figure_0.jpeg)

Fig. 6. The evaluation process

available for accomplishing such tasks. For RQ2, we select LLaMA-Reviewer [\[2\]](#page-10-1) as the benchmark method because it uses the same dataset, and, additionally, our study also uses LLaMA as the base model. Choosing LLaMA-Reviewer as the baseline approach facilitates a fair comparison.

*1) The experiment for RQ1:* First of all, we need to construct a test set containing explicit annotations of *DRC* and non-*DRC* for each entry. As shown in Table [I,](#page-4-0) the CodeReviewer training set contains a total of 150,406 entries. We randomly selected 600 of these entries for manual annotation, achieving a margin of error of less than 4% at a 95% confidence level. The annotation was performed by two software engineering graduate students, each annotating 450 data entries with 300 duplicated entries to check for consistency. To be specific, when a review comment triggers a fix that pertains to the review comment, the review comment is labeled "*desired*." Otherwise, it is labeled "*undesired*." We used the Chi-Squared test [\[48\]](#page-11-20) to check the determination consistency of the duplicate annotations and obtained a p-value of 0.965, thereby rejecting the hypothesis of inconsistency in the annotations.

To demonstrate the effectiveness of the *Desiview* dataset distillation method, we compare it against the 10-line rule for change-triggering review comments [\[17\]](#page-10-15), GPT-3.5-turbo, and GPT-4o using prompt engineering. Different treatments are required for these benchmark approaches.

The 10-line rule determines whether changes were made within 10 lines of the given review comment. As the CodeReviewer dataset only contains information on changes at the corresponding locations, the rule was simplified to whether modifications were made subsequently.

GPT-3.5-turbo and GPT-4o require prompt engineering to detect *DRC*s, as shown in Fig. [7.](#page-6-0) We experimented with different phrasing methods and selected a relatively betterperforming prompt as the final prompt. To avoid the impact of sampling generation by large models, we set the sampling parameter temperature to 0, ensuring the model uses greedy search generation for result stability. As other methods did not provide examples, to ensure a fair comparison, we did not use examples in the prompt engineering method either, i.e., we adopted a zero-shot prompt [\[49\]](#page-11-21) strategy. Common metrics such as 'accuracy', 'precision', 'recall', and 'F1-score' were used for evaluation, thereby determining the effectiveness of different methods in identifying *DRC*(s).

```
Your t a s k i s t o d e t e r m i n e whether t h e changes i n t h e g i v e n
o r i g i n a l code and t h e m o d i f i e d code p e r t a i n t o t h e p r o v i d e d
r e v i e w comment . I f t h e y p e r t a i n , o u t p u t True ; i f t h e y do n o t
p e r t a i n , o u t p u t F a l s e . Only p r o v i d e True or F a l s e , w i t h o u t
any a d d i t i o n a l c o n t e n t .
' ' ' o r i g i n a l code
{}
' ' '
' ' ' m o d i f i e d code
{}
' ' '
' ' ' r e v i e w comment
{}
' ' '
```

<span id="page-6-0"></span>Fig. 7. The prompt used to detect *DRC*s

*2) The experiment for RQ2:* To evaluate the quality of *DRC*s generated by LLMs trained with the original and the distilled datasets, we compare three LLMs: (1) LLaMA (LLaMA-3 and LLaMA-3.1) fine-tuned with the original dataset, i.e., LLaMA-Reviewer [\[2\]](#page-10-1), (2) LLaMA (LLaMA-3 and LLaMA-3.1) fine-tuned with the distilled data, i.e., *Desiview4FT*, and LLaMA (LLaMA-3 and LLaMA-3.1) aligned with both distilled data (*DRC*s) and dropped data (non-*DRC*s), i.e., *Desiview4FA*. For a fair comparison, the fine-tuning process applies the same settings as study [\[2\]](#page-10-1) with different datasets. The evaluation process consists of two parts: automated evaluation and human evaluation.

Automated evaluation uses a test set of 5,727 entries, as shown in Table [I.](#page-4-0) As the distilled dataset contain a high proportion of review comments that can lead to effective code fixes, it is fair enough to regard the ground truth as the correct answer. With the trained LLMs generating review comments for a given code commit, the generated review comments are compared against the existing ones contained in the test set using the BLEU-4 metric [\[50\]](#page-11-22) to calculate text similarity.

Human evaluation is conducted by two software engineering graduate students. The test set contains a total of 5,727 *DRC* entries (as shown in Table [I\)](#page-4-0), from which we randomly selected 300 entries for human evaluation, ensuring a margin of error of less than 6% at a 95% confidence level. Each student evaluates 180 pieces of data, including 60 duplicated evaluations, to check for consistency. We applied the Chi-Squared test [\[48\]](#page-11-20) to check consistency, obtaining a p-value of 0.887, thereby rejecting the hypothesis of evaluation inconsistency and proving the consistency of the evaluations. The human evaluation involved observing the original code commit under review and the LLM-generated review comments to determine whether the provided review comments correctly identify and describe the issues. In this sense, we divided the evaluation into two tasks: accurately locating code issues and accurately describing the issues. The criteria we adopted to determine the results of these two tasks are as follows:

- 1) *Human Position:* To be considered correct, it requires the LLM-generated review comments to pinpoint the same location of the code issues as in the answer, regardless of whether the description of the code issues is correct or not.
- 2) *Human Perfect:* To be considered correct, it requires the LLM-generated review comments to describe the same issues and/or solutions as in the answer. It is clear that the second task builds upon the first one.

## *B. Results analysis*

TABLE III PERFORMANCE OF EACH METHOD IN IDENTIFYING *DRC*S

<span id="page-7-1"></span>

| Method            | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| 10-line rule      | 58.33    | 51.92     | 100.00 | 68.35    |
| gpt3.5-turbo-0125 | 68.00    | 60.71     | 81.85  | 69.72    |
| gpt-4o-0513       | 76.50    | 79.72     | 64.07  | 71.05    |
| Desiview          | 86.67    | 88.93     | 80.37  | 84.44    |

*a) RQ1: Accuracy of automated identification of DRCs:* Table [III](#page-7-1) presents the performance of different methods in identifying *DRC*s. As all entries in the test set contain code changes, the 10-line rule can always identify all *DRC*s, achieving 100% recall. However, this method only determines changes and cannot assess whether the changes align with the review comments, resulting in poor performance in comprehensive metrics such as accuracy and F1-score. The GPT-3.5-turbo and GPT-4o methods can somewhat understand the relationship between changes and review comments, but their performance is still inferior to our method, which significantly outperforms existing methods in both comprehensive metrics of accuracy and F1-score. Fig. [8](#page-8-0) illustrates examples of desired and undesired review comments. From these examples, it is evident that the determination of *DRC*s cannot be based solely on the review's phrasing or keywords to ascertain whether they can lead to effective code fixes. It requires a deeper understanding of both the code and the review comments.

<span id="page-7-2"></span>TABLE IV PERFORMANCE OF THE CODE REVIEW COMMENT GENERATION TASK

| Method                               | BLEU-4    | Human Position | Human Perfect |  |
|--------------------------------------|-----------|----------------|---------------|--|
| LLaMA-Reviewer<br>(LLaMA-3 Origin)   | 8.33      | 70.33          | 16.67         |  |
| Desiview4FT                          | 11.87     | 76.67          | 18.33         |  |
| (LLaMA-3 Based)                      | (+42.50%) | (+9.01%)       | (+9.96%)      |  |
| Desiview4FA                          | 13.13     | 80.00          | 18.67         |  |
| (LLaMA-3 Based)                      | (+57.62%) | (+13.75%)      | (+12.00%)     |  |
| LLaMA-Reviewer<br>(LLaMA-3.1 Origin) | 6.86      | 68.67          | 12.67         |  |
| Desiview4FT                          | 12.48     | 78.67          | 16.00         |  |
| (LLaMA-3.1 Based)                    | (+81.92%) | (+14.56%)      | (+26.28%)     |  |
| Desiview4FA                          | 13.57     | 79.00          | 16.67         |  |
| (LLaMA-3.1 Based)                    | (+97.81%) | (+15.04%)      | (+31.57%)     |  |

*b) RQ2: Effect of dataset distillation on the task performance of LLMs:* Table [IV](#page-7-2) shows the performance of the code review comment generation task under both automated and human evaluation. It is evident that the distilled dataset significantly improves the performance of the LLMs in generating *DRC*s that present more accurate and useful information to users. Notably, LLMs fine-tuned with the distilled dataset that contains a high proportion of *DRC*s and whose size is less than half of that of the original dataset significantly outperform those fine-tuned with the original full dataset in terms of both localizing and describing code issues. Alignment further extends this advantage. It is worth noting that LLaMA-3.1 does not show a clear advantage over LLaMA-3, but it appears to be more sensitive to the fine-tuning and alignment of the distilled dataset. Nevertheless, the distilled dataset improves both versions.

Fig. [9](#page-8-1) presents some intuitive examples of the review comments generated by the three methods (based on LLaMA-3 only). It is apparent that training with the distilled dataset significantly enhances LLaMA's ability to identify key issues. Using alignment techniques can further improve the model's ability to generate accurate information and reduce the occurrence of irrelevant information.

## V. DISCUSSION

<span id="page-7-0"></span>The primary contribution of this work is the dataset distillation method, *Desiview*, which can be used to construct a distilled dataset for fine-tuning LLMs to enhance their performance in code review tasks. This contribution has a profound impact on LLM-based code review research and can be generally extended to other LLM-based software engineering tasks. In this section, we discuss the implications of dataset distillation in code review research.

## *A. Distilled dataset for training code review models*

Training data is fundamental to generating effective automated code reviews. The results of our study partially demonstrate the value a distilled dataset brings to LLM-based code review, confirming that distilled data often leads to better LLM performance [\[8\]](#page-10-7), [\[32\]](#page-11-4). However, acquiring distilled datasets is generally challenging. Specifically, in the field of code review, previous studies have not effectively addressed this issue, aside from the extremely costly manual annotation methods [\[11\]](#page-10-9), [\[12\]](#page-10-10). By leveraging the relationship between the content of review comments and the code fixes as a criterion, we introduce the perplexity metric to the quality assessment of code review data in terms of the desiredness of review comments. The proposed dataset distillation method enables automated and reliable acquisition of large-scale, high-quality review data at a low cost, thereby effectively addressing the scarcity of high-quality code review datasets. More importantly, we believe this method has the potential to be customized and generalized to support other code review objectives, such as performance bottlenecks [\[51\]](#page-11-23) and security risks [\[52\]](#page-11-24) and even other software engineering objectives, such as vulnerability detection [\[53\]](#page-11-25). Fig. [2](#page-4-1) (upper left corner) illustrates a typical pull-request development mode widely used in the open-source community. Drawing on the principles behind *Desiview*, it

```
Desired review comment Undesired review comment
Original Code Commit: 
return errors.Wrap(err, "[ Phase1Packet.DeserializeWithoutHeader ] Can't read 
Section 2") 
} 
- data.Read(read) 
} 
p1p.Signature = make([]byte, SignatureLength)
                                                                             Original Code Commit: 
                                                                             private String getName(PublicizeConnection connection) { 
                                                                             - String name = connection.getExternalDisplayName(); 
                                                                             if (name.isEmpty()) { 
                                                                             - name = connection.getExternalName(); 
                                                                             } 
                                                                             return name;
Review Comment: 
Error return value of `data.Read` is not checked
                                                                             Review Comment: 
                                                                             Let's use `TextUtils.isEmpty(name)` here so we also have `null` as a possibility.
Subsequent code fixes: 
return errors.Wrap(err, "[ Phase1Packet.DeserializeWithoutHeader ] Can't read 
Section 2") 
} 
+ _, err = data.Read(read) 
+ if err != nil { 
+ return errors.Wrap(err, "failed to read raw data") 
+ } 
} 
p1p.Signature = make([]byte, SignatureLength)
                                                                             Subsequent code fixes: 
                                                                             private String getName(PublicizeConnection connection) { 
                                                                             + String name = connection.getExternalName(); 
                                                                             if (name.isEmpty()) { 
                                                                             + name = connection.getExternalDisplayName(); 
                                                                             } 
                                                                             return name;
Observation: 
The review comment points out that the file was read without checking for reported 
errors, and the subsequent fix added a check for the raised errors.
                                                                             Observation: 
                                                                             The review comment suggests using `TextUtils.isEmpty(name)` to check if `name` is 
                                                                             null, but the subsequent code fix does not seem to accept this suggestion.
```

<span id="page-8-0"></span>Fig. 8. Examples of *DRC*s and non-*DRC*s identified by *Desiview*

```
Case 1 
Original code commit: 
 auto r = get_if<record_type>(&t); 
- return r ? is_flat(*r) : true; 
 } 
 size_t flat_size(const record_type& rec) { 
Ground truth comment (reflected in the subsequent code): 
Simpler: `return r && is_flat(*r)`.
LLaMA-Reviewer: 
I think this is a bug. `is_flat` returns `false` for `record_type` that are not flat, but `get_if` returns `nullptr` for `record_type` that are not flat. So this will return `true` for non-
flat `record_type`s.
Desiview4FT: 
```suggestion return r && is_flat(*r); ```
Desiview4FA: 
I think you want `return r && is_flat(*r)`
Observation: 
Training with distilled data significantly improves the locating and describing issues in the code commit by LLMs. 
Case 2
Original code commit: 
 path + " Irohad config top element must be an object."); 
 const auto obj = src.GetObject(); 
- getValByKey(path, dest.blok_store_path, obj, config_members::BlockStorePath); 
 getValByKey(path, dest.torii_port, obj, config_members::ToriiPort); 
 getValByKey(path, dest.internal_port, obj, config_members::InternalPort); getValByKey(path, dest.pg_opt, obj, config_members::PgOpt); 
Ground comment (reflected in the subsequent code): 
```suggestion getValByKey(path, dest.block_store_path, obj, config_members::BlockStorePath); ```
LLaMA-Reviewer: 
I think it's better to use `getValByKey` for all the keys.
Desiview4FT: 
```suggestion getValByKey(path, dest.blok_store_path, obj, config_members::BlockStorePath); getValByKey(path, dest.torii_port, obj, config_members::ToriiPort); 
getValByKey(path, dest.internal_port, obj, config_members::InternalPort); getValByKey(path, dest.pg_opt, obj, config_members::PgOpt); ```
Desiview4FA: 
```suggestion getValByKey(path, dest.blok_store_path_, obj, config_members::BlockStorePath); ```
Observation: 
The ground truth comment implies that the code, initially marked for deletion, should be reinstated to its original position. This is increasingly reflected in the outputs from 
LLaMA-Reviewer, Desiview4FT, and Desiview4FA models. Notably, Desiview4FA generates a review comment that is identical to the ground truth.
```

<span id="page-8-1"></span>Fig. 9. Examples of review comments generated by LLaMA-Reviewer, *Desiview4FT* and *Desiview4FA*

should be feasible to construct high-quality datasets applicable to various LLM-enabled software engineering scenarios by adjusting the desiredness criterion of review comments to other relevant criteria, which necessitates further exploration.

## *B. Distilled dataset for training review quality prediction models*

The distilled dataset obtained from this study can also be used to train models to predict the quality of generated review comments — specifically, predicting whether they can trigger code fixes. One of the pain points in applying LLMs to automated code review is the uncontrollable quality of generated review comments. In extreme cases, developers have to check not only the code but also the generated review comments, which can even increase their workload. This defeats the purpose of automated code review, which is to reduce their workload. By applying the proposed dataset distillation method, which divides the original dataset into high-quality and low-quality datasets, we can train a binary classification model on traditional Bert series [\[20\]](#page-10-19), [\[54\]](#page-11-26) models with smaller parameter sizes to predict the quality of LLMgenerated review comments. The training hyperparameters are shown in Table [V.](#page-9-1)

<span id="page-9-1"></span>TABLE V TRAINING HYPERPARAMETERS FOR PREDICTING DESIRED REVIEW COMMENTS

| Pattern | epochs | batch | lr   | label smoothing | weight decay |
|---------|--------|-------|------|-----------------|--------------|
| Value   | 5      | 32    | 1e-5 | 0.1             | 0            |

TABLE VI PERFORMANCE OF THE *DRC* PREDICTION TASK

<span id="page-9-2"></span>

| Method        | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Robert-Base   | 79.82    | 84.28     | 80.71  | 82.46    |
| CodeBert-base | 79.11    | 86.82     | 78.39  | 82.39    |

The review comment quality prediction task involves predicting whether a given review comment will lead to a modification based on the original code submission and the generated review comment. The input consists of a code Diff from the original submission and a corresponding review comment, and the output indicates whether it will lead to a code fix. We selected two commonly used Bert series models, Roberta [\[54\]](#page-11-26) and CodeBERT [\[20\]](#page-10-19), as our base models. Evaluation metrics include accuracy, precision, recall, and F1-score. The results in Table [VI](#page-9-2) show a great potential to predict the quality of generated review comments. One possible application scenario is to integrate an LLM-based code review solution with a review comment quality assessment mechanism to provide a corresponding quality evaluation for each review comment, which can assist developers in making better decisions on whether to accept a particular review comment. A more aggressive approach could be to embed this quality assessment mechanism within the LLM-based code review system, ensuring it only outputs high-quality review comments and drops low-quality ones.

## VI. THREATS TO VALIDITY

<span id="page-9-0"></span>In this section, we discuss several validity risks.

*Definition of quality in terms of desiredness:* In this paper, we define a high-quality code review dataset as one composed of only *DRC*s. This approach inevitably carries some validity risks. The concept of *DRC*s in this paper specifically refers to review comments that can trigger subsequent code fixes and improvements. This concept is based on multiple code review studies, all of which regard the detection of issues and triggering of subsequent code fixes as their primary purpose [\[3\]](#page-10-2). Our study does not intend to diminish other purposes or the unique meaning of *DRC*s in these scenarios. For instance, as shown in Fig. 1 (example C), acknowledging a developer's fix can also be meaningful to the developer. In fact, during our exploration process, we found that the highquality dataset distilled according to our definition of *DRC* is large enough (close to 65,000 entries of *DRC*s) to meet the needs for fine-tuning LLaMA. Therefore, we did not expand the concept of *DRC*s to cover other code review purposes. Nevertheless, this leaves some room for future research.

*Noise in the distilled dataset:* Based on the results in Table [III,](#page-7-1) it is reasonable to assume that a small number of non-*DRC*s remain in the distilled dataset when using *Desiview* to distil the CodeReviewer dataset, including the test set. This may introduce some bias in addressing RQ2. However, the amount of such data is minimal and unlikely to have a significant impact. Moreover, the results of the human evaluation fully corroborate the validity and efficacy of the *Desiview* method, making this risk controllable.

*Pre-trained models:* Different pre-trained models exhibit varying performance in the task of code review. During the phase of assessing *DRC*s, we employed a voting mechanism using multiple models (without any preference) to reduce the impact of a single LLM on the results. Although the results show a positive effect of this strategy, it is possible that using different LLMs may impact the results.

*Model parameter size:* In general, the performance of a model can be significantly influenced by the size of its parameters within the same series. However, due to limitations in GPU capacity, we restricted our study to models with parameters under 16B for all computations. For tasks such as fine-tuning and alignment, we utilized models of 8B size. It is worth noting that employing larger models could potentially enhance performance, particularly in terms of assessing the quality of review comments and generating desired review comments more accurately. The size of the model parameters also contributes to a substantial disparity in the quality of the review comments we generate compared to those generated by GPT4o. We believe one of the reasons for this difference is on the scale of three orders of magnitude in terms of parameter numbers. Despite this, our distilled dataset has proven highly effective in enhancing the original LLaMA series. Therefore, it would be intriguing to explore and compare the effectiveness of LLaMA-3 and GPT4o in conducting code review with a comparable number of parameters (e.g., LLaMA-3.1-405B).

*Model training methods:* Typically, large model training methods are divided into low-parameter training and full-parameter training. Low-parameter training involves finetuning only a small portion of the LLM, making it possible to train it with fewer resources. Full-parameter training involves training all parameters of the LLM, which can lead to better performance but at a significantly higher cost compared to low-parameter training [\[55\]](#page-11-27). Research has shown that fullparameter training outperforms low-parameter training. Due to computational resource constraints, this work employed lowparameter training methods for training LLMs. Using fullparameter training methods might result in better performance.

*Dataset:* As far as we know, the only public multiprogramming language dataset in the open-source community that includes original submissions, subsequent fixes, and review comments is the CodeReviewer dataset [\[5\]](#page-10-4). Therefore, we could only use this dataset for our training and testing. This factor creates some risk in terms of the generalization of results, and we encourage the community to construct other datasets using our method to validate this work.

*Errors from human evaluation:* Despite using standardized methods and consistency checks to ensure the accuracy of manual evaluations, errors in manual evaluation are still possible, which may affect the results. We mitigated this risk by employing evaluators with a background in software engineering, ensuring they have the necessary expertise to determine the relationship between source code and review comments during the evaluation. In addition, allowing some degree of duplication across multiple evaluators, in addition to conducting chi-square tests, further reduces this risk.

## VII. CONCLUSIONS

<span id="page-10-16"></span>In this work, we propose a method for analyzing, assessing and automatically identifying *DRC*s. This solves one critical problem of automated construction of high-quality datasets for code review research. Empirical experiments reveal that this method surpasses all other methods in terms of identifying *DRC*s, including GPT-4o, in terms of accuracy. Using this method, we constructed a distilled dataset containing a high proportion of *DRC*s, which not only can be used to train a model to predict whether a new review comment is *DRC* but also support fine-tuning and aligning LLMs to perform better code review tasks in terms of generating *DRC*s. Both automated evaluation and human evaluation reveal that LLMs trained with the distilled dataset outperform those trained with the original dataset. Future work includes applying the proposed dataset distillation method to construct datasets suitable for different code review objectives, during which better LLMs can be leveraged to improve the accuracy of high-quality review comment identification. Additionally, using newer and stronger LLMs as the base models, along with new techniques for fine-tuning and alignment can also be explored to further enhance the application efficacy of distilled datasets.

## REFERENCES

- <span id="page-10-0"></span>[1] G. Gousios, M. Pinzger, and A. v. Deursen, "An exploratory study of the pull-based software development model," in *Proceedings of the 36th international conference on software engineering*, 2014, pp. 345–355.
- <span id="page-10-1"></span>[2] J. Lu, L. Yu, X. Li, L. Yang, and C. Zuo, "Llama-reviewer: Advancing code review automation with large language models through parameterefficient fine-tuning," in *2023 IEEE 34th International Symposium on Software Reliability Engineering (ISSRE)*. IEEE, 2023, pp. 647–658.
- <span id="page-10-2"></span>[3] O. Kononenko, O. Baysal, and M. W. Godfrey, "Code review quality: How developers see it," in *Proceedings of the 38th international conference on software engineering*, 2016, pp. 1028–1038.
- <span id="page-10-3"></span>[4] A. Bosu and J. C. Carver, "Impact of peer code review on peer impression formation: A survey," in *2013 ACM/IEEE International Symposium on Empirical Software Engineering and Measurement*. IEEE, 2013, pp. 133–142.
- <span id="page-10-4"></span>[5] Z. Li, S. Lu, D. Guo, N. Duan, S. Jannu, G. Jenks, D. Majumder, J. Green, A. Svyatkovskiy, S. Fu *et al.*, "Automating code review activities by large-scale pre-training," in *Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering*, 2022, pp. 1035–1047.
- <span id="page-10-5"></span>[6] X. Hou, Y. Zhao, Y. Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J. Grundy, and H. Wang, "Large language models for software engineering: A systematic literature review," *arXiv preprint arXiv:2308.10620*, 2023.

- <span id="page-10-6"></span>[7] A. Fan, B. Gokkaya, M. Harman, M. Lyubarskiy, S. Sengupta, S. Yoo, and J. M. Zhang, "Large language models for software engineering: Survey and open problems," in *2023 IEEE/ACM International Conference on Software Engineering: Future of Software Engineering (ICSE-FoSE)*. IEEE, 2023, pp. 31–53.
- <span id="page-10-7"></span>[8] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu *et al.*, "Lima: Less is more for alignment," *Advances in Neural Information Processing Systems*, vol. 36, 2024.
- [9] Y. Liu, S. Tao, X. Zhao, M. Zhu, W. Ma, J. Zhu, C. Su, Y. Hou, M. Zhang, M. Zhang *et al.*, "Coachlm: Automatic instruction revisions improve the data quality in llm instruction tuning," in *2024 IEEE 40th International Conference on Data Engineering (ICDE)*. IEEE, 2024, pp. 5184–5197.
- <span id="page-10-8"></span>[10] R. Rejeleene, X. Xu, and J. Talburt, "Towards trustable language models: Investigating information quality of large language models," *arXiv preprint arXiv:2401.13086*, 2024.
- <span id="page-10-9"></span>[11] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray *et al.*, "Training language models to follow instructions with human feedback," *Advances in neural information processing systems*, vol. 35, pp. 27 730–27 744, 2022.
- <span id="page-10-10"></span>[12] N. McAleese, R. M. Pokorny, J. F. C. Uribe, E. Nitishinskaya, M. Trebacz, and J. Leike, "Llm critics help catch llm bugs," *arXiv preprint arXiv:2407.00215*, 2024.
- <span id="page-10-11"></span>[13] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, "Self-instruct: Aligning language models with selfgenerated instructions," *arXiv preprint arXiv:2212.10560*, 2022.
- <span id="page-10-12"></span>[14] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong *et al.*, "A survey of large language models," *arXiv preprint arXiv:2303.18223*, 2023.
- <span id="page-10-13"></span>[15] X. Luo, Q. Zhu, Z. Zhang, X. Wang, Q. Yang, D. Xu, and W. Che, "Semi-instruct: Bridging natural-instruct and self-instruct for code large language models," *arXiv preprint arXiv:2403.00338*, 2024.
- <span id="page-10-14"></span>[16] B. Plank, "The'problem'of human label variation: On ground truth in data, modeling and evaluation," *arXiv preprint arXiv:2211.02570*, 2022.
- <span id="page-10-15"></span>[17] A. Bosu, M. Greiler, and C. Bird, "Characteristics of useful code reviews: An empirical study at microsoft," in *2015 IEEE/ACM 12th Working Conference on Mining Software Repositories*. IEEE, 2015, pp. 146–156.
- <span id="page-10-17"></span>[18] C. Sadowski, E. Soderberg, L. Church, M. Sipko, and A. Bacchelli, ¨ "Modern code review: a case study at google," in *Proceedings of the 40th international conference on software engineering: Software engineering in practice*, 2018, pp. 181–190.
- <span id="page-10-18"></span>[19] S.-T. Shi, M. Li, D. Lo, F. Thung, and X. Huo, "Automatic code review by learning the revision of source code," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 33, no. 01, 2019, pp. 4910– 4917.
- <span id="page-10-19"></span>[20] Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang *et al.*, "Codebert: A pre-trained model for programming and natural languages," *arXiv preprint arXiv:2002.08155*, 2020.
- <span id="page-10-20"></span>[21] Y. Wang, W. Wang, S. Joty, and S. C. Hoi, "Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation," *arXiv preprint arXiv:2109.00859*, 2021.
- <span id="page-10-21"></span>[22] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*, "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.
- <span id="page-10-22"></span>[23] Y. Hong, C. Tantithamthavorn, P. Thongtanunam, and A. Aleti, "Commentfinder: a simpler, faster, more accurate code review comments recommendation," in *Proceedings of the 30th ACM joint European software engineering conference and symposium on the foundations of software engineering*, 2022, pp. 507–519.
- <span id="page-10-23"></span>[24] A. Gupta and N. Sundaresan, "Intelligent code reviews using deep learning," in *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'18) Deep Learning Day*, 2018.
- <span id="page-10-24"></span>[25] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, T. Remez, J. Rapin *et al.*, "Code llama: Open foundation models for code," *arXiv preprint arXiv:2308.12950*, 2023.
- <span id="page-10-25"></span>[26] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale *et al.*, "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.
- <span id="page-10-26"></span>[27] D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. Li *et al.*, "Deepseek-coder: When the large language model meets programming–the rise of code intelligence," *arXiv preprint arXiv:2401.14196*, 2024.

- <span id="page-11-0"></span>[28] A. Lozhkov, R. Li, L. B. Allal, F. Cassano, J. Lamy-Poirier, N. Tazi, A. Tang, D. Pykhtar, J. Liu, Y. Wei *et al.*, "Starcoder 2 and the stack v2: The next generation," *arXiv preprint arXiv:2402.19173*, 2024.
- <span id="page-11-1"></span>[29] AI@Meta, "Llama 3 model card," 2024. [Online]. Available: [https://github.com/meta-llama/llama3/blob/main/MODEL](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) CARD.md
- <span id="page-11-2"></span>[30] B. Chen, F. Zhang, A. Nguyen, D. Zan, Z. Lin, J.-G. Lou, and W. Chen, "Codet: Code generation with generated tests," *arXiv preprint arXiv:2207.10397*, 2022.
- <span id="page-11-3"></span>[31] M. A. Islam, M. E. Ali, and M. R. Parvez, "Mapcoder: Multiagent code generation for competitive problem solving," *arXiv preprint arXiv:2405.11403*, 2024.
- <span id="page-11-4"></span>[32] Y. Wei, Z. Wang, J. Liu, Y. Ding, and L. Zhang, "Magicoder: Empowering code generation with oss-instruct," in *Forty-first International Conference on Machine Learning*, 2024.
- <span id="page-11-5"></span>[33] A. Silva, S. Fang, and M. Monperrus, "Repairllama: Efficient representations and fine-tuned adapters for program repair," *arXiv preprint arXiv:2312.15698*, 2023.
- <span id="page-11-6"></span>[34] Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and D. Jiang, "Wizardcoder: Empowering code large language models with evol-instruct," *arXiv preprint arXiv:2306.08568*, 2023.
- <span id="page-11-7"></span>[35] J. Ji, T. Qiu, B. Chen, B. Zhang, H. Lou, K. Wang, Y. Duan, Z. He, J. Zhou, Z. Zhang *et al.*, "Ai alignment: A comprehensive survey," *arXiv preprint arXiv:2310.19852*, 2023.
- <span id="page-11-8"></span>[36] Y. Tang, D. Z. Guo, Z. Zheng, D. Calandriello, Y. Cao, E. Tarassov, R. Munos, B. A. Pires, M. Valko, Y. Cheng ´ *et al.*, "Understanding the performance gap between online and offline alignment algorithms," *arXiv preprint arXiv:2405.08448*, 2024.
- <span id="page-11-9"></span>[37] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
- <span id="page-11-10"></span>[38] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, "Direct preference optimization: Your language model is secretly a reward model," *Advances in Neural Information Processing Systems*, vol. 36, 2024.
- <span id="page-11-11"></span>[39] K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela, "Kto: Model alignment as prospect theoretic optimization," *arXiv preprint arXiv:2402.01306*, 2024.
- <span id="page-11-12"></span>[40] B. Steenhoek, M. Tufano, N. Sundaresan, and A. Svyatkovskiy, "Reinforcement learning from automatic feedback for high-quality unit test generation," *arXiv preprint arXiv:2310.02368*, 2023.
- <span id="page-11-13"></span>[41] S. Dou, Y. Liu, H. Jia, L. Xiong, E. Zhou, J. Shan, C. Huang, W. Shen, X. Fan, Z. Xi *et al.*, "Stepcoder: Improve code generation with reinforcement learning from compiler feedback," *arXiv preprint arXiv:2402.01391*, 2024.
- <span id="page-11-14"></span>[42] B. Shen, J. Zhang, T. Chen, D. Zan, B. Geng, A. Fu, M. Zeng, A. Yu, J. Ji, J. Zhao *et al.*, "Pangu-coder2: Boosting large language models for code with ranking feedback," *arXiv preprint arXiv:2307.14936*, 2023.
- <span id="page-11-15"></span>[43] T. Shen, R. Jin, Y. Huang, C. Liu, W. Dong, Z. Guo, X. Wu, Y. Liu, and

- D. Xiong, "Large language model alignment: A survey," *arXiv preprint arXiv:2309.15025*, 2023.
- <span id="page-11-16"></span>[44] F. Jelinek, R. L. Mercer, L. R. Bahl, and J. K. Baker, "Perplexity—a measure of the difficulty of speech recognition tasks," *The Journal of the Acoustical Society of America*, vol. 62, no. S1, pp. S63–S63, 1977.
- <span id="page-11-17"></span>[45] A. Miaschi, D. Brunato, F. Dell'Orletta, and G. Venturi, "What makes my model perplexed? a linguistic investigation on neural language models perplexity," in *Proceedings of Deep Learning Inside Out (DeeLIO): The 2nd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures*, 2021, pp. 40–47.
- <span id="page-11-18"></span>[46] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "Lora: Low-rank adaptation of large language models," *arXiv preprint arXiv:2106.09685*, 2021.
- <span id="page-11-19"></span>[47] G. Rong, Y. Yu, Y. Zhang, H. Zhang, H. Shen, D. Shao, H. Kuang, M. Wang, Z. Wei, Y. Xu *et al.*, "Distilling quality enhancing comments from code reviews to underpin reviewer recommendation," *IEEE Transactions on Software Engineering*, 2024.
- <span id="page-11-20"></span>[48] K. Pearson, "X. on the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling," *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, vol. 50, no. 302, pp. 157–175, 1900.
- <span id="page-11-21"></span>[49] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell *et al.*, "Language models are few-shot learners," *Advances in neural information processing systems*, vol. 33, pp. 1877–1901, 2020.
- <span id="page-11-22"></span>[50] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "Bleu: a method for automatic evaluation of machine translation," in *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, 2002, pp. 311–318.
- <span id="page-11-23"></span>[51] A. Shypula, A. Madaan, Y. Zeng, U. Alon, J. Gardner, M. Hashemi, G. Neubig, P. Ranganathan, O. Bastani, and A. Yazdanbakhsh, "Learning performance-improving code edits," *arXiv preprint arXiv:2302.07867*, 2023.
- <span id="page-11-24"></span>[52] N. Tihanyi, T. Bisztray, R. Jain, M. A. Ferrag, L. C. Cordeiro, and V. Mavroeidis, "The formai dataset: Generative ai in software security through the lens of formal verification," in *Proceedings of the 19th International Conference on Predictive Models and Data Analytics in Software Engineering*, 2023, pp. 33–43.
- <span id="page-11-25"></span>[53] R. Croft, M. A. Babar, and M. M. Kholoosi, "Data quality for software vulnerability datasets," in *2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)*. IEEE, 2023, pp. 121–133.
- <span id="page-11-26"></span>[54] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "Roberta: A robustly optimized bert pretraining approach," *arXiv preprint arXiv:1907.11692*, 2019.
- <span id="page-11-27"></span>[55] D. Biderman, J. G. Ortiz, J. Portes, M. Paul, P. Greengard, C. Jennings, D. King, S. Havens, V. Chiley, J. Frankle *et al.*, "Lora learns less and forgets less," *arXiv preprint arXiv:2405.09673*, 2024.