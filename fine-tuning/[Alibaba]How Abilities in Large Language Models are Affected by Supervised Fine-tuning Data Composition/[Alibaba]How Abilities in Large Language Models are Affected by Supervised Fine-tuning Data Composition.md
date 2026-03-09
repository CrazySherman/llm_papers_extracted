# How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition

#### Guanting Dong\*, Hongyi Yuan\* , Keming Lu, Chengpeng Li\* , Mingfeng Xue Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, Jingren Zhou

Alibaba Group

{dongguanting.dgt,yuanzheng.yuanzhen,ericzhou.zc}@alibaba-inc.com

### Abstract

Large language models (LLMs) with enormous pre-training tokens and parameters emerge diverse abilities, including math reasoning, code generation, and instruction following. These abilities are further enhanced by supervised fine-tuning (SFT). While the open-source community has explored ad-hoc SFT for enhancing individual capabilities, proprietary LLMs exhibit versatility across various skills. Therefore, understanding the facilitation of multiple abilities via SFT is paramount. In this study, we specificially focuses on the interplay of data composition between mathematical reasoning, code generation, and general human-aligning abilities during SFT. We propose four intriguing research questions to explore the association between model performance and various factors including data amount, composition ratio, model size and SFT strategies. Our experiments reveal that distinct capabilities scale differently and larger models generally show superior performance with same amount of data. Mathematical reasoning and code generation consistently improve with increasing data amount, whereas general abilities plateau after roughly a thousand samples. Moreover, we observe data composition appears to enhance various abilities under limited data conditions, yet can lead to performance conflicts when data is plentiful. Our findings also suggest the amount of composition data influences performance more than the composition ratio. In analysis of SFT strategies, we find that sequentially learning multiple skills risks catastrophic forgetting. Our proposed Dual-stage Mixed Fine-tuning (DMT) strategy offers a promising solution to learn multiple abilities with different scaling patterns.

# 1 Introduction

Recent research has demonstrated the remarkable and versatile proficiency of large language models (LLMs) in dealing with a variety of real-world

tasks expressed in natural languages [\(Ouyang et al.,](#page-11-0) [2022a;](#page-11-0) [Anil et al.,](#page-8-0) [2023;](#page-8-0) [OpenAI,](#page-11-1) [2023;](#page-11-1) [Luo et al.,](#page-10-0) [2023a\)](#page-10-0), especially Information Extraction (IE) [\(Lu](#page-10-1) [et al.,](#page-10-1) [2022;](#page-10-1) [Xu et al.,](#page-12-0) [2023b;](#page-12-0) [Zhao et al.,](#page-12-1) [2023;](#page-12-1) [Cheng et al.,](#page-9-0) [2023d;](#page-9-0) [Wang et al.,](#page-11-2) [2023b;](#page-11-2) [Zhang](#page-12-2) [et al.,](#page-12-2) [2024b;](#page-12-2) [Li et al.,](#page-10-2) [2024\)](#page-10-2), Information Retrieval (IR) [\(Zhu et al.,](#page-13-0) [2024;](#page-13-0) [Liu et al.,](#page-10-3) [2024c\)](#page-10-3) and Spoken Language Understanding (SLU) [\(Hoscilowicz](#page-10-4) [et al.,](#page-10-4) [2024;](#page-10-4) [Yin et al.,](#page-12-3) [2024;](#page-12-3) [Cheng et al.,](#page-9-1) [2023a,](#page-9-1) [2024;](#page-9-2) [Dong et al.,](#page-10-5) [2023a\)](#page-10-5). Among the tasks, LLMs especially emerge with three outstanding abilities in reasoning [\(Cobbe et al.,](#page-9-3) [2021;](#page-9-3) [Wei et al.,](#page-12-4) [2022\)](#page-12-4), coding [\(Chen et al.,](#page-9-4) [2021\)](#page-9-4), and aligning general human intentions [\(Ouyang et al.,](#page-11-0) [2022a\)](#page-11-0), which have drawn much attention from the LLM research community. In order to further incentivize such abilities, it necessitates supervised fine-tuning (SFT) stages on annotated task data.

However, existing research has mostly conducted separate SFT investigations on each of the three tasks, where reasoning and coding abilities require SFT on in-domain human-annotated or augmented data [\(Yuan et al.,](#page-12-5) [2023b;](#page-12-5) [Luo et al.,](#page-11-3) [2023b;](#page-11-3) [Yu et al.,](#page-12-6) [2024\)](#page-12-6) while diverse and complex human instructions are applauded for aligning human intentions [\(Wang et al.,](#page-12-7) [2023d;](#page-12-7) [Taori et al.,](#page-11-4) [2023;](#page-11-4) [Cheng et al.,](#page-9-5) [2023c;](#page-9-5) [Xu et al.,](#page-12-8) [2023a;](#page-12-8) [Zhou et al.,](#page-13-1) [2023a;](#page-13-1) [Wang et al.,](#page-11-5) [2023a;](#page-11-5) [Lu et al.,](#page-10-6) [2023\)](#page-10-6). As shown by the strong performance of proprietary LLMs such as GPT-4 [\(OpenAI,](#page-11-1) [2023\)](#page-11-1) and Claude, LLMs have the potential to master all the tasks in one model. Therefore, it is of paramount importance to investigate the versatile performance of SFT with composite task data, and understanding and addressing the challenges posed by the data composition problem in the SFT stage is crucial for further enhancing the capabilities of LLMs in a comprehensive manner.

In essence, the tasks of reasoning, coding, and aligning human intentions are of different characteristics. Reasoning and coding tasks require ad-

<sup>\*</sup> Work done during internships at Alibaba Group.

hoc abilities of complex and detailed logic in decomposing task instructions and dealing with nonlinguistic and symbolic features [\(Chen et al.,](#page-9-4) [2021;](#page-9-4) [Huang and Chang,](#page-10-7) [2023\)](#page-10-7), whereas aligning human intentions requires versatility and understanding obscure intentions expressed in human instructions [\(Lu et al.,](#page-10-6) [2023\)](#page-10-6). Given the fundamental difference among the tasks, multi-task learning with composite data fine-tuning for small-scaled pre-trained language models is prone to catastrophic forgetting [\(De Lange et al.,](#page-9-6) [2022\)](#page-9-6), hindering the fine-tuned performance of one model on separate tasks. Many efforts have been made to compensate for the phenomenon [\(Liang et al.,](#page-10-8) [2021;](#page-10-8) [Xu et al.,](#page-12-9) [2021;](#page-12-9) [Yuan](#page-12-10) [et al.,](#page-12-10) [2023a\)](#page-12-10). There has also been research discovering that scaling up the pre-trained language model scale and the fine-tuning data scale are beneficial for zero-shot out-of-domain generalization on various linguistic tasks while leaving out the assessment of in-domain performance [\(Sanh et al.,](#page-11-6) [2022;](#page-11-6) [Chung et al.,](#page-9-7) [2022a;](#page-9-7) [Longpre et al.,](#page-10-9) [2023\)](#page-10-9). Given the increased capacity of LLMs, the multi-task performance by SFT on composite data of essentially different downstream tasks is less studied. Understanding the SFT performance with composite data and corresponding scaling patterns is of great utility in practice.

In this study, we focus on the data composition problem among mathematical reasoning, code generation, and general human-aligning abilities in SFT. We aim to comprehensively investigate the relationship between model performance and different factors including data amount, data composition ratio, model scales, and SFT training strategies. We also investigate how the relationship varies under different scales. Specifically, we focus on the following four research questions:

- 1. *How do math reasoning, coding, and general abilities scale with SFT data amounts?*
- 2. *Are there performance conflicts when combining these three abilities in SFT?*
- 3. *What are the key factors that induce the performance conflicts?*
- 4. *What are the impacts of different SFT strategies for composite data?*

To answer these questions, we conduct experiments on three benchmarks, which are GSM8K [\(Cobbe et al.,](#page-9-3) [2021\)](#page-9-3) for mathematical reasoning, HumanEval [\(Chen et al.,](#page-9-4) [2021\)](#page-9-4) for coding, and MT-Bench [\(Zheng et al.,](#page-13-2) [2023\)](#page-13-2) for general human alignment. We fine-tune LLMs on the related training data to activate these abilities. Furthermore, we

conduct extensive analysis regarding model parameter scales ranging from LLaMA 7B to 33B [\(Tou](#page-11-7)[vron et al.,](#page-11-7) [2023\)](#page-11-7) and explore four different SFT strategies shown in Figure [1:](#page-2-0) multi-task learning, sequential training, mixed sequential training, and dual-stage mixing fine-tuning (DMT), providing empirical guidance for learning a versatile LLM with composite SFT. The key findings of this paper can be summarized as follows:

- Different SFT abilities exhibit distinct scaling patterns, while larger models show better performances with the same data amount generally.
- Compared to single ability learning, multitask learning multiple abilities exhibits improvement in low-resource and decline in high-resource. Additionally, as the model size increases, there is a greater performance gain in low-resource settings for math and general abilities.
- Data amounts directly influence each ability, while the data ratio is insignificant.
- Multi-task learning lead to conflicts, while sequential training results in catastrophic forgetting. Our proposed DMT effectively alleviates both performance conflicts and catastrophic forgetting in the SFT phrase, achieving a balance between general and specialized abilities.

#### 2 Related Works

Supervised Fine-Tuning of Large Language Models Large Language Models (LLMs) have shown notable zero-shot performance in various domains [\(Brown et al.,](#page-8-1) [2020;](#page-8-1) [Wu et al.,](#page-12-11) [2021;](#page-12-11) [Hou](#page-10-10) [et al.,](#page-10-10) [2024;](#page-10-10) [Dong et al.,](#page-10-11) [2023b;](#page-10-11) [Zhou et al.,](#page-13-3) [2024;](#page-13-3) [Wu et al.,](#page-12-12) [2023;](#page-12-12) [Song et al.,](#page-11-8) [2023\)](#page-11-8), prompting further development to push the boundaries of these models. To delve deeper to their potential, LLMs are subjected to a Supervised Fine-Tuning (SFT) phase, enhancing their ability to solve tasks and align better with human instructions. Here, we extend the conventional definition of SFT to include various forms of sequence-to-sequence fine-tuning, such as fine-tuning for human alignment, instruction following, and domain-specific task optimization [\(Zhou et al.,](#page-13-4) [2023b;](#page-13-4) [Yuan et al.,](#page-12-13) [2023c;](#page-12-13) [Cheng](#page-9-8) [et al.,](#page-9-8) [2023b;](#page-9-8) [Zhang et al.,](#page-12-14) [2024a\)](#page-12-14).

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1: The illustration of four different training strategies in this paper.

Recent research has delved into multi-task instruction fine-tuning of pre-trained LLMs to bolster their zero-shot performance across numerous downstream NLP tasks (Sanh et al., 2022). In an effort to encompass existing NLP tasks comprehensively, Chung et al.; Longpre et al. curated the expansive FLAN dataset specifically for instruction-based fine-tuning. LLMs, both open-source (Chung et al., 2022b) and proprietary (Singhal et al., 2022), fine-tuned with FLAN, have demonstrated enhanced zero-shot performance on a variety of unseen tasks.

While research has probed into the generalization capabilities of LLMs within out-of-distribution domains (Liu et al., 2024a; Yuan et al., 2024; Wang et al., 2024a), the effects of multi-task training on in-domain performance remain under-explored. With the ascent of proprietary models like Chat-GPT, the focus on SFT for aligning LLMs with human intent has intensified (Ouyang et al., 2022b). Moving away from crowd-sourced SFT data, recent initiatives have generated SFT datasets from user logs within proprietary LLM platforms (Chiang et al., 2023; Wang et al., 2023a), employing the models themselves to assist in the data generation process (Wang et al., 2023d; Taori et al., 2023; Cheng et al., 2023d; Lei et al., 2023; Xu et al., 2023a; Xue et al., 2023). Additionally, methods to improve SFT data quality have been proposed, targeting more accurate alignment with human interactions (Zhou et al., 2023a; Wang et al., 2023c; Lu et al., 2023: Liu et al., 2024b).

Furthermore, SFT has proven beneficial for

LLMs in specialized areas such as mathematical reasoning (Cobbe et al., 2021; Hendrycks et al., 2021; Yuan et al., 2023b; Chen et al., 2024; Yue et al., 2023; Gou et al., 2024; Li et al., 2023; Yue et al., 2024) and code generation tasks (Chaudhary, 2023; Luo et al., 2023b; Wang et al., 2024b; Wei et al., 2023). Taking advantage of their advanced interactive capabilities, some researchers have leveraged supervised fine-tuned LLMs to compose commands that interface with external tools, thus enhancing the handling of assorted downstream applications (Shen et al., 2023; Yao et al., 2023b,a; Song et al., 2024; Fu et al., 2024). This paper examines the SFT performance using composite datasets, considering different model sizes and data amounts.

Scaling Laws in Large Language Models The exceptional performance of LLMs comes from scaling up model sizes, data amounts, and computational costs to massive scales. Therefore, it is crucial to explore the model performance across an exponential range of scales. Many endeavors have been made to discuss the scaling laws for pretraining (Anil et al., 2023; Hoffmann et al., 2022), transfer learning (Chronopoulou et al., 2019), preference modeling (Gao et al., 2022) and mathematical reasoning (Yuan et al., 2023b). In this paper, we also explore the SFT performance with composite data from the perspective of different scales of model sizes and data amounts.

#### 3 Experiments

We have SFT datasets  $\{D_1, D_2, ..., D_k\}$  where each  $D_i = \{q_{i,j}, r_{i,j}\}_j$  contains queries and responses from one source. We consider each SFT dataset to correspond to one ability and we also have k in-domain metrics to measure them. We investigate the performances of in-domain metrics with different dataset compositions  $(D \subset \bigcup_{1 \leq i \leq k} D_i)$  and training strategies on different sizes of LLMs.

#### 3.1 Experiment Setup

We collect three SFT datasets  $\{D_1, D_2, D_3\}$  including GSM8K RFT (Yuan et al., 2023b), Code Alpaca (Chaudhary, 2023), and ShareGPT (Chiang et al., 2023) to represent math reasoning, coding, and general human-aligning ability SFT dataset respectively. We will integrate a new SFT dataset Dby these three datasets to investigate how data composition affects the model performances. We use GSM8K test set (Cobbe et al., 2021), HumanEval (Chen et al., 2021), and MT-Bench (Zheng et al., 2023) to measure abilities including math reasoning, coding, and general human-aligning. We use LLaMA (Touvron et al., 2023) series as our pretrained language models and use FastChat framework (Zheng et al., 2023) for fine-tuning. We finetune models with 3 epochs and a peak of 2e-5 learning rate. The batch size during SFT is 16. More details about SFT datasets, evaluation metrics, implementations and Training FLOPs can be found in Appendix A, B, C and D.

# 3.2 RQ1. Individual Ability Performance vs. Data Amount

The instruction following ability can be activated via SFT on datasets like ShareGPT which contain around 100 thousand samples. However, (Zhou et al., 2023a) demonstrates that strong base models can achieve human alignment with just 1000 samples. Specialized abilities such as math reasoning require a large amount of data (Cobbe et al., 2021; Yuan et al., 2023b), unlike general abilities. Therefore, it is crucial to investigate how each ability improves as the data amount increases.

**Experimental Design:** We conduct SFT on LLaMA of various sizes using {1, 1/4, 1/16, 1/64, 1/256} proportions of the training set obtained from GSM8K RFT, Code Alpaca, and ShareGPT seperately. This allowed us to evaluate each ability with various data sizes and model sizes.

<span id="page-3-0"></span>![](_page_3_Figure_7.jpeg)

Figure 2: The scaling curve of different sizes of LLaMA in three individual domains.

**Results and Analysis.** Figure 2 shows the individual data scaling curves for different abilities after SFT. We find that: Different abilities exhibit different scaling curves. To be more specific, mathematical reasoning capability shows a positive correlation with the data amount across various model sizes which is consistent with (Yuan et al., 2023b). Similarly, general human-aligning ability demonstrates an almost monotonically increasing scaling curve. However, it is noteworthy that general ability emerges with only around 1k data samples (ranging from 1/256 to 1/64), and after reaching a certain threshold (1/64), their performances improve slowly. This further supports (Zhou et al., 2023a), indicating that a small amount of high-quality SFT data is possible for the emergence of general human-aligning ability in LLMs. On the other hand, code ability exhibits an irregular scaling curve when the model's parameter count is small (7B & 13B). However, when the parameter count increases to 33B, its coding performance shows an approximately log-linear trend with the data amount. One possible explanation is that Code Alpaca and the samples in HumanEval have different distributions. Larger models can capture shared knowledge across code data distributions in the in-domain samples, which enables them to exhibit some level of generalization to out-of-distribution (OOD) samples. Another observation is larger models show better performances with the same data amount generally. The outlier is with very little data (1/256), smaller models may outperform larger models. If there is enough data, larger models have stable better performances.

# 3.3 RQ2. Performance Difference vs. Mixed Data Amount

We should deliver a versatile model that requires us to mix various SFT datasets and apply SFT. We want to ask how each ability varies due to SFT dataset mixtures. We investigate it with different amounts of mixed data and compare them with individual ability performance.

Experimental Design: For the individual source

<span id="page-4-1"></span>![](_page_4_Figure_0.jpeg)

Figure 3: Comparative experiments between mix domains and individual domains for LLaMA.

setting, consistent with the setup in RQ1, we performed fine-tuning on LLaMA models of different sizes using {1, 1/4, 1/16, 1/64, 1/256} amounts of training data from GSM8K, Code Alpaca, and ShareGPT separately. For the mixed source setting, we sampled {1, 1/4, 1/16, 1/64, 1/256} amounts of training data from GSM8K, Code Alpaca, and ShareGPT, and directly mixed them according to the corresponding proportions. In this way, we constructed datasets with fixed proportions of different ability domains, while varying the total data amount. These datasets are then used for fine-tuning the LLaMA models <sup>1</sup>.

**Results and Analysis.** Figure 3 presents results of LLaMA of different sizes on three benchmarks under the individual source and mixed source settings. The following observations are made: Abilities are improved with low-resource and are decreased with high-resource compared to individual source abilities. In the case of LLaMA-7B, compared to the data scaling curve of the individual source setting, the models fine-tuned with mixed source data consistently demonstrated performance conflicts among the three ability domains at high resources (100%). However, as the data volume decreased, a turning point in performance is observed between the two settings in the data range of 1/64 to 1/16. Notably, the models fine-tuned with mixed source data exhibited performance gains at low resources (1/256), indicating that SFT data from different sources benefit each other in a low-resource setting. However, when there is enough data, data from other sources could be viewed as noise for

<span id="page-4-3"></span>![](_page_4_Figure_5.jpeg)

Figure 4: Different data ratio (k) between specific abilities and general abilities on three benchmarks.

in-domain generalization. As the model size increases, the performance gain in low-resource settings also increases for math and general abilities. In the case of the 13B and 33B models, it is obvious that the scaling curve for the mix source setting follows a similar trend observed in previous analyses, with the presence of performance intersection points as the data volume scales. However, a crucial distinction arises, whereby larger models exhibit more pronounced performance gains under low resources as the size of model parameters increases. The outlier is the LLaMA-7B (code only, 1/256). A possible reason is the introduction of a small amount of unseen code data easily disrupts the original code ability of the pretrained model, as supported by its low HumanEval score (less than 6). In conclusion, our finding implies that larger language models excel in acquiring general and specialized abilities from diverse data sources under low-resource conditions<sup>2</sup>.

# 3.4 RQ3. Performance Difference vs. Data Composition Ratio

We observe ability conflicts in high-resource settings, and we want to investigate the reasons why the conflicts occur. Two possible factors are the **data amount** of other abilities is too high or the **data ratio** of other abilities is too high. Here we conduct experiments to investigate the data ratio factor.

**Experimental Design:** We consider coding and mathematics as a combined specialized data source, and the ShareGPT as the general data source. We designed three setups as follows which control the amount of one source of data and vary the ratio between general and specialized data.

#### 1. Fixed general data, scaling specialized data:

<span id="page-4-0"></span><sup>&</sup>lt;sup>1</sup>We also conduct "Equal Data Amount VS. Equal Data Proportion" experiments in Appendix H

<span id="page-4-2"></span><sup>&</sup>lt;sup>2</sup>To validate the generalizability of our conclusions, we further conduct the more experiments on **World Knowledge**, **Language Understanding** and **Translation** in Appendix E.

We use a full training set of ShareGPT and sampled different proportions {1, 1/4, 1/16, 1/64, 1/256} of GSM8K RFT and Code Alpaca as a mixture.

- **2. Fixed specialized data, scaling general data:** We use a full training set of GSM8K RFT and Code Alpaca and sample different proportions of ShareGPT as a mixture.
- **3. Fixed 1/64 general data, scaling specialized data**: Motivated by LIMA's setup (Zhou et al., 2023a), we used a 1/64 ShareGPT set (about 1500 examples) and sampled different proportions of GSM8K RFT and Code Alpaca as a mixture.

Results and Analysis. Q1: Does the performance of the model vary with different ratios of general and specialized data? As illustrated in the top three graphs of Figure 4, we conduct ablation studies of the data ratio (k) between specialized and general abilities. To be noticed ratio is normalized by data amount, for example, k=1 means  $\frac{\text{specialized use data amount}}{\text{general all data amount}} = \frac{\text{specialized all data amount}}{\text{general all data amount}}$ . We utilize a fixed special

specialized all data amount general all data amount. We utilize a fixed specialized data setting (directly mixing 100% code & math data for training) and a fixed general data setting (100% general data for training) as the baseline and observe:

- (1) With the increase in the ratio of general data from 1/256 to 1/1, Fixed specialized data, scaling general data setup exhibits similar performance to the setup that *Fixed specialized abilities* in terms of math reasoning. This suggests that variations in the data ratio k have minimal impact on math ability. We consider the reason that math and general abilities are non-conflict since they are too different in the semantic space. However, when considering HumanEval, the Fixed specialized data, scaling general data setup displays noticeable fluctuations compared to the baseline. We attribute this to the inclusion of a certain proportion of code data in ShareGPT. Due to the differences in data format and distribution, the presence of similar data features exacerbates the performance conflicts between abilities when the data ratio k increases. Further analysis of the distribution of different abilities is discussed in Section 4.1.
- (2) With the increase in the ratio of specialized data from 1/256 to 1/1, the setup that *Fixed general data*, *scaling specialized data* displayed no significant performance changes compared to the baseline. This echoes our hypothesis that when there are significant differences in task formats and data distributions between different SFT

abilities, the impact of data ratio is minimal. However, when there is some degree of similarities, the data ratio can lead to noticeable performance fluctuations.

Q2: Under extremely limited general data resources, does the ratio of specialized data have an impact on the model's performance? We further explore the impact of different ratios of specialized data when the model has just acquired a certain level of general human-aligning ability (k = 1/64). The bottom 3 graphs of Figure 4 present comparative experiments between two settings. We observe that regardless of whether the data amount for general capabilities is abundant (k = 1) or scarce (k = 1/64), the performance on MT-Bench shows no significant fluctuations with varying proportions of specialized data. Furthermore, in mathematical reasoning, 1/64 general data setup exhibited a scaling trend that is almost identical to the full general data setup. However, for coding ability, with the same amount of code data and different ratios, code abilities are different in the two settings. We still consider the reason is code data are partly related to ShareGPT data and cause the performance difference and provide an analysis in Discussion 4.2.

# 3.5 RQ4. Performance Difference vs. Training Strategies

We could feed these SFT datasets into models with different training strategies. In this section, We experiment with these settings and investigate how they influence each ability's performance.

**Experimental Design:** Firstly, we introduce three kinds of naive training strategies as follows:

- **1. Multi-task learning:** We directly mix different SFT data sources  $D = \bigcup_{1 \leq i \leq k} D_i$  and applying SFT. If we view each data source as a different task, this can be viewed as multi-task learning.
- **2. Sequential Training:** We sequentially apply SFT on each dataset. Specifically, we sequentially trained on coding, math reasoning, and the general ability dataset. Since the general ability is the most important one for human alignment, we put ShareGPT as our last dataset.
- **3. Mixed Sequential Training:** We apply multitask learning on specialized datasets(code, math) first and apply SFT on the general ability dataset. These three approaches are presented in Figure 1.

**Results and Analysis:** Table 1 presents performances under different training strategies in terms

of mathematical reasoning, code generation, and general human-aligning ability. Multi-task learning preserves specialized abilities among these strategies while hurting the general ability most among them. Sequential training and mixed sequential training preserve general ability while losing too many specialized abilities. The observed outcome is in accordance with expectations, as during the final fine-tuning phase, the mixed sequential training strategy remains unaffected by specialized data, thereby effectively preserving its generalization capability. However, an inherent drawback of multistage training is the occurrence of catastrophic forgetting of prior knowledge, which motivates us to further explore methods that can alleviate catastrophic forgetting of specialized abilities while maximizing the preservation of general capability.

4. Dual-stage Mixed Fine-tuning (DMT): Based on our observation from RQ1 to RQ4, we propose a new training strategy that can reduce the ability conflict during multi-task learning and relieve the issue of catastrophic forgetting during sequential training. From RQ1, the model needs large data amounts to activate specialized abilities. From RQ2, multi-task learning with all amounts of specialized data and general data will hurt each ability. From RQ3, a small amount of specialized data will not affect the general ability performance. From RQ4, (mixed) sequential training forgets specialized abilities. So the model needs to learn large amounts of specialized data and should not forget them during learning general ability. A natural choice is to learn full amounts of specialized data first and add a small amount of specialized data to general data during the last stage of sequential training to prevent forgetting. As shown in Figure [1,](#page-2-0) we first apply SFT on the specialized dataset which is same as the first stage of the mixed sequential training strategy. For the second stage, we perform SFT with a mixed data source comprising a combination of the general data and varying proportions k (1, 1/2, 1/4, 1/8, 1/16, 1/32) of code and math data. Adding code and math data in the second stage helps models to recall the specialized ability. The results of DMT (k = 1/256) are presented in Table [1](#page-7-1) and the detailed scaling analysis of proportion k can be found in the discussion.

Model Accuracy vs. DMT Strategies. In Table [1,](#page-7-1) LLaMA-7B with DMT (k = 1/256) strategy perform significant improvement in mathematical reasoning (32.6 to 41.92) and code generation (15.24 to 17.68) compared to the mixed sequential training

strategy, which indicates a significant alleviating effect of mixing specialized capability data in the last fine-tuning stage on catastrophic forgetting. Surprisingly, DMT (k = 1/256) even exhibits a slight improvement on MT-Bench, further highlighting its ability to alleviate catastrophic forgetting while effectively preserving general capability.

Regarding the 13B and 33B models, DMT (k = 1/256) demonstrates noticeable alleviation of catastrophic forgetting in mathematical reasoning (13B: 40.48 to 46.47 / 33B: 44.24 to 56.36) and code generation (13B: 18.3 to 19.5 / 33B: 24.4 to 25.5) compared to the mixed sequential training strategy. Additionally, it significantly retains its general capability (13B: 5.93 to 6.03 / 33B 6.43 to 6.69). Therefore, these results serve as additional validation of the efficacy of DMT in mitigating catastrophic forgetting while maintaining general capability [3](#page-6-1) .

### 4 Discussion

#### <span id="page-6-0"></span>4.1 Visualization of Different SFT Abilities

In the aforementioned analysis of data composition, we observed a significant performance degradation when different data sources are directly mixed. In this section, our aim is to explore the potential mutual influence of semantic representation distributions among different data sources. Specifically, we randomly sampled 100 queries from CodeAlpaca, GSM8k RFT, and ShareGPT datasets and extracted the hidden layer representations located in the Middle layer (15th) of the model. Subsequently, we employed the t-SNE toolkit [\(Van der](#page-11-16) [Maaten and Hinton,](#page-11-16) [2008\)](#page-11-16) to visualize the representations of the three types of capabilities. The results in Figure [5](#page-7-2) illustrate a notable collapse phenomenon in the semantic representations of both the original LLaMA-13b and LLaMA-13b with DMT (k=1/256). While both models exhibit a certain level of separation in the mathematical data representations, there remains a certain degree of overlap between the representations of code and general samples. In Appendix [G,](#page-16-1) we further discuss the visualization of semantic spaces at different layers of LLaMA 7B & 13B.

<span id="page-6-1"></span><sup>3</sup>To verify the effectiveness of DMT strategy on relatively OOD benchmarks, we further evaluate it on MBPP and MATH in Appendix [F.](#page-16-2)

<span id="page-7-1"></span>

| Methods                       | LLaMA -7B |              | LLaMA -13B |       |              | LLaMA -33B |              |           |          |
|-------------------------------|-----------|--------------|------------|-------|--------------|------------|--------------|-----------|----------|
|                               | GSM8K     | HumanEval    | MT-Bench   | GSM8K | HumanEval    | MT-Bench   | GSM8K        | HumanEval | MT-Bench |
| Individual domain             |           |              |            |       |              |            |              |           |          |
| General only                  | 11.10     | 10.42        | 5.88       | 14.02 | 16.40        | 6.13       | 26.06        | 24.30     | 6.63     |
| Math only                     | 49.10     | 6.71         | 2.53       | 51.40 | 12.8         | 2.54       | 57.91        | 15.5      | 3.18     |
| Code only                     | 4.51      | 18.40        | 4.30       | 5.15  | 17.1         | 3.53       | 6.06         | 26.82     | 4.18     |
| Different Training Strategies |           |              |            |       |              |            |              |           |          |
| Multi-task learning           | 47.53     | 14.63        | 5.76       | 50.94 | 19.50        | 5.73       | 56.69        | 18.9      | 6.07     |
| Sequential Training           | 31.39     | <u>15.85</u> | 5.72       | 39.12 | 20.12        | 5.93       | 47.27        | 24.80     | 6.73     |
| Mixed Sequential Training     | 32.60     | 15.24        | 6.02       | 40.48 | 18.30        | 5.93       | 44.24        | 24.4      | 6.43     |
| DMT(k=1/256)                  | 41.92     | 17.68        | 6.08       | 46.47 | <u>19.50</u> | 6.03       | <u>56.36</u> | 25.00     | 6.73     |

Table 1: The results of LLaMA-7B, 13B, 33B under different training strategies on three benchmarks. The top two results across different strategies are marked with **bold** and <u>underlined</u>.

<span id="page-7-2"></span>![](_page_7_Figure_2.jpeg)

Figure 5: The left two figures show the t-SNE plots of LLaMA-13B and LLaMA-13B with the DMT strategy. The two right figures show the performance scaling of LLaMA-7B & 13B with DMT under different *k* values.

# <span id="page-7-0"></span>4.2 Ablation of the Specialized Domains in ShareGPT

In RQ2, we observe using mixed data sources resulted in improved abilities under low-resource conditions but diminished abilities under high-resource conditions when compared to single data sources. However, the presence of coding and mathematical samples within the ShareGPT introduces uncertainty regarding whether the performance gain under low resources is solely attributed to these specific coding & mathematical data or other orthogonal samples in the general dataset (e.g., translation or extraction). Hence, the objective of this section is to investigate whether the conclusions drawn in Section 3.3 remain valid after removing the code and math samples within ShareGPT.

Experimental Design: We employed an openset tagger InsTag (Lu et al., 2023) to annotate samples in ShareGPT. To filter out data related to coding and mathematical abilities, we conduct regular expression matching to eliminate instances where the tags contain keywords "code" or "math". Finally, we obtain a ShareGPT dataset devoid of any code or math-related information (reducing from 86K to 63K). In alignment with the settings in Section 3.3, we sampled different proportions of training data (1, 1/4, 1/16, 1/64, 1/256) from GSM8K,

Code Alpaca, and the modified ShareGPT dataset (without code math). These samples were directly mixed according to the corresponding proportions. Subsequently, the LLaMA models were fine-tuned by using this mixed dataset.

Results and Analysis. Figure 6 shows the results of our experiment. Removing the code and math from ShareGPT not only mitigates the performance conflicts among different abilities to some extent under high-resource conditions but also maintains stable gains in low-resource settings. We propose that the potential reason behind these findings lies in the differences in the distribution of code and math data between ShareGPT, CodeAlpaca, and GSM8K RFT datasets. This distribution gap introduces an extra noise during the SFT phrase, while its removal enables the model to better generalize coding and mathematical abilities. Furthermore, in low-resource scenarios, this phenomenon indicates that the code and math samples in ShareGPT are not the key factor contributing to performance improvements, but rather the diversity and variability of the data (Longpre et al., 2023). In summary, the presence of code math data within ShareGPT does not emerge as a key factor impacting the performance gains identified in Section 3.3, highlighting the generalization of our conclusions.

<span id="page-8-2"></span>![](_page_8_Figure_0.jpeg)

Figure 6: The scaling curve after ablating code and math-related samples from ShareGPT.

#### 4.3 Specialized Data Amount in DMT

We investigate how different values of k influence model performance and results shown in Figure 5. When we adjust k from 0 to 1/256 (k = 0 is equal to mixed sequential training), the SFT models show significant improvements in both specialized ability and general human-aligning ability. On the contrary, as k increased from 1/4 to 1, the model exhibited a decline in general ability. We believe this is in line with the findings in RQ2, which concluded that high-resource settings lead to conflicts while low-resource settings lead to gains in mixed sources. Furthermore, as k increased from 1/256 to 1/4, we observe a linear inverse trend between general ability and specialized ability, especially an increase in general ability coincided with a decrease in specialized ability. This suggests k needs to be tuned based on specific requirements in order to achieve a balance between multiple abilities.

#### 5 Conclusion

We explore the data composition in the SFT phase, focusing on mathematical reasoning, code generation, and general human-aligning abilities. We formulate four research questions to guide our investigation and analyze the scaling trends between different abilities and factors (e.g. data amount, data ratio, model parameters, and training strategies). Our findings reveal distinct scaling patterns among different abilities, with larger models demonstrating superior performance when trained with the same amount of data. Moreover, mixing data sources in the SFT phase improves performance in low-resource scenarios but diminishes in highresource scenarios. Interestingly, the phenomenon of low-resource gain becomes more prominent as the model parameter size increases. Furthermore, our observations indicate that data amount directly influences performance conflicts, whereas the impact of data ratio is insignificant within our experimental setup. Finally, regarding the SFT strategies, we demonstrate our proposed DMT strategy effectively alleviates performance conflicts, offering a

promising solution to activate multiple abilities.

#### Limitations

Due to our use of the large language model LLaMA-33B, the extensive computational resources and time required for both training and inference may limit its applicability. The datasets used in this article are all open source, so there are no ethical or moral issues; However, inappropriate prompts and noisy training corpora can potentially lead to privacy and bias issues with LLMs. Furthermore, the evaluation benchmark MT-Bench relies on GPT-4 for scoring, which may result in some variability in the results, and these may not always align perfectly with human judgment standards. In this paper, we primarily focus on three SFT capabilities that are of great interest in the LLMs community, including mathematical reasoning, code generation, and general human-aligned ability. To verify the generality of our conclusions, we further explore three additional SFT capabilities in the appendix. Nevertheless, there are still many other SFT capabilities (such as creative generation) within the LLMs community that have data composition issues waiting to be explored by researchers, which will also be the focus of our future research efforts

#### References

<span id="page-8-0"></span>Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. *arXiv* preprint arXiv:2305.10403.

<span id="page-8-4"></span>Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. 2021. Program synthesis with large language models. *arXiv* preprint arXiv:2108.07732.

<span id="page-8-3"></span>Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. 2022. A framework for the evaluation of code generation models. https://github.com/bigcode-project/bigcode-evaluation-harness.

<span id="page-8-1"></span>Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess,

- Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. [Language models are few-shot learners.](http://arxiv.org/abs/2005.14165)
- <span id="page-9-14"></span>Mauro Cettolo, Jan Niehues, Sebastian Stüker, Luisa Bentivogli, and Marcello Federico. 2014. [Report on](https://aclanthology.org/2014.iwslt-evaluation.1) [the 11th IWSLT evaluation campaign.](https://aclanthology.org/2014.iwslt-evaluation.1) In *Proceedings of the 11th International Workshop on Spoken Language Translation: Evaluation Campaign*, pages 2–17, Lake Tahoe, California.
- <span id="page-9-12"></span>Sahil Chaudhary. 2023. Code alpaca: An instructionfollowing llama model for code generation. [https:](https://github.com/sahil280114/codealpaca) [//github.com/sahil280114/codealpaca](https://github.com/sahil280114/codealpaca).
- <span id="page-9-4"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. [Evaluating](http://arxiv.org/abs/2107.03374) [large language models trained on code.](http://arxiv.org/abs/2107.03374)
- <span id="page-9-11"></span>Zhaorun Chen, Zhuokai Zhao, Zhihong Zhu, Ruiqi Zhang, Xiang Li, Bhiksha Raj, and Huaxiu Yao. 2024. Autoprm: Automating procedural supervision for multi-step reasoning via controllable question decomposition. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*.
- <span id="page-9-1"></span>Xuxin Cheng, Bowen Cao, Qichen Ye, Zhihong Zhu, Hongxiang Li, and Yuexian Zou. 2023a. Ml-lmcl: Mutual learning and large-margin contrastive learning for improving asr robustness in spoken language understanding. In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 6492– 6505.
- <span id="page-9-8"></span>Xuxin Cheng, Qianqian Dong, Fengpeng Yue, Tom Ko, Mingxuan Wang, and Yuexian Zou. 2023b. M 3 st: Mix at three levels for speech translation. In *ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 1–5. IEEE.
- <span id="page-9-5"></span>Xuxin Cheng, Zhihong Zhu, Bowen Cao, Qichen Ye, and Yuexian Zou. 2023c. Mrrl: Modifying the reference via reinforcement learning for nonautoregressive joint multiple intent detection and slot

- filling. In *The 2023 Conference on Empirical Methods in Natural Language Processing*, pages 10495– 10505.
- <span id="page-9-2"></span>Xuxin Cheng, Zhihong Zhu, Hongxiang Li, Yaowei Li, Xianwei Zhuang, and Yuexian Zou. 2024. Towards multi-intent spoken language understanding via hierarchical attention and optimal transport. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 17844–17852.
- <span id="page-9-0"></span>Xuxin Cheng, Zhihong Zhu, Wanshi Xu, Yaowei Li, Hongxiang Li, and Yuexian Zou. 2023d. Accelerating multiple intent detection and slot filling via targeted knowledge distillation. In *The 2023 Conference on Empirical Methods in Natural Language Processing*.
- <span id="page-9-10"></span>Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. [Vicuna: An open](https://lmsys.org/blog/2023-03-30-vicuna/)[source chatbot impressing gpt-4 with 90%\\* chatgpt](https://lmsys.org/blog/2023-03-30-vicuna/) [quality.](https://lmsys.org/blog/2023-03-30-vicuna/)
- <span id="page-9-13"></span>Alexandra Chronopoulou, Christos Baziotis, and Alexandros Potamianos. 2019. An embarrassingly simple approach for transfer learning from pretrained language models. *arXiv preprint arXiv:1902.10547*.
- <span id="page-9-7"></span>Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022a. [Scaling instruction-finetuned](http://arxiv.org/abs/2210.11416) [language models.](http://arxiv.org/abs/2210.11416)
- <span id="page-9-9"></span>Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022b. [Scaling instruction-finetuned](http://arxiv.org/abs/2210.11416) [language models.](http://arxiv.org/abs/2210.11416)
- <span id="page-9-3"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*.
- <span id="page-9-6"></span>Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. 2022. [A continual learning sur](https://doi.org/10.1109/TPAMI.2021.3057446)[vey: Defying forgetting in classification tasks.](https://doi.org/10.1109/TPAMI.2021.3057446) *IEEE*

- *Transactions on Pattern Analysis and Machine Intelligence*, 44(7):3366–3385.
- <span id="page-10-5"></span>Guanting Dong, Tingfeng Hui, Zhuoma GongQue, Jinxu Zhao, Daichi Guo, Gang Zhao, Keqing He, and Weiran Xu. 2023a. [Demonsf: A multi-task](http://arxiv.org/abs/2310.10169) [demonstration-based generative framework for noisy](http://arxiv.org/abs/2310.10169) [slot filling task.](http://arxiv.org/abs/2310.10169)
- <span id="page-10-11"></span>Guanting Dong, Jinxu Zhao, Tingfeng Hui, Daichi Guo, Wenlong Wan, Boqi Feng, Yueyan Qiu, Zhuoma Gongque, Keqing He, Zechen Wang, et al. 2023b. Revisit input perturbation problems for llms: A unified robustness evaluation framework for noisy slot filling task. *arXiv preprint arXiv:2310.06504*.
- <span id="page-10-18"></span>Dayuan Fu, Jianzhao Huang, Siyuan Lu, Guanting Dong, Yejie Wang, Keqing He, and Weiran Xu. 2024. [Preact: Predicting future in react enhances agent's](http://arxiv.org/abs/2402.11534) [planning ability.](http://arxiv.org/abs/2402.11534)
- <span id="page-10-20"></span>Leo Gao, John Schulman, and Jacob Hilton. 2022. [Scal](http://arxiv.org/abs/2210.10760)[ing laws for reward model overoptimization.](http://arxiv.org/abs/2210.10760)
- <span id="page-10-16"></span>Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. 2024. [Tora: A tool-integrated reasoning agent](http://arxiv.org/abs/2309.17452) [for mathematical problem solving.](http://arxiv.org/abs/2309.17452)
- <span id="page-10-15"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*.
- <span id="page-10-19"></span>Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. 2022. [Training compute-optimal](http://arxiv.org/abs/2203.15556) [large language models.](http://arxiv.org/abs/2203.15556)
- <span id="page-10-4"></span>Jakub Hoscilowicz, Pawel Pawlowski, Marcin Skorupa, Marcin Sowanski, and Artur Janicki. 2024. ´ [Large](http://arxiv.org/abs/2404.02588) [language models for expansion of spoken language](http://arxiv.org/abs/2404.02588) [understanding systems to new languages.](http://arxiv.org/abs/2404.02588)
- <span id="page-10-10"></span>Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large language models are zero-shot rankers for recommender systems. In *European Conference on Information Retrieval*, pages 364–381. Springer.
- <span id="page-10-7"></span>Jie Huang and Kevin Chen-Chuan Chang. 2023. [To](https://doi.org/10.18653/v1/2023.findings-acl.67)[wards reasoning in large language models: A survey.](https://doi.org/10.18653/v1/2023.findings-acl.67) In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 1049–1065, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-10-22"></span>Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.

- <span id="page-10-13"></span>Shanglin Lei, Guanting Dong, Xiaoping Wang, Keheng Wang, and Sirui Wang. 2023. [Instructerc: Reforming](http://arxiv.org/abs/2309.11911) [emotion recognition in conversation with a retrieval](http://arxiv.org/abs/2309.11911) [multi-task llms framework.](http://arxiv.org/abs/2309.11911)
- <span id="page-10-17"></span>Chengpeng Li, Zheng Yuan, Hongyi Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, and Chang Zhou. 2023. [Query and response](http://arxiv.org/abs/2310.05506) [augmentation cannot help out-of-domain math rea](http://arxiv.org/abs/2310.05506)[soning generalization.](http://arxiv.org/abs/2310.05506)
- <span id="page-10-2"></span>Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and Zhicheng Dou. 2024. [From](http://arxiv.org/abs/2404.14851) [matching to generation: A survey on generative in](http://arxiv.org/abs/2404.14851)[formation retrieval.](http://arxiv.org/abs/2404.14851)
- <span id="page-10-8"></span>Xiaobo Liang, Lijun Wu, Juntao Li, Yue Wang, Qi Meng, Tao Qin, Wei Chen, Min Zhang, and Tie-Yan Liu. 2021. [R-drop: Regularized dropout for](http://arxiv.org/abs/2106.14448) [neural networks.](http://arxiv.org/abs/2106.14448)
- <span id="page-10-21"></span>Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let's verify step by step. *arXiv preprint arXiv:2305.20050*.
- <span id="page-10-12"></span>Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, and Xiao-Ming Wu. 2024a. [How good are llms at](http://arxiv.org/abs/2308.10261) [out-of-distribution detection?](http://arxiv.org/abs/2308.10261)
- <span id="page-10-14"></span>Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. 2024b. [What makes good data for align](http://arxiv.org/abs/2312.15685)[ment? a comprehensive study of automatic data se](http://arxiv.org/abs/2312.15685)[lection in instruction tuning.](http://arxiv.org/abs/2312.15685)
- <span id="page-10-3"></span>Zheng Liu, Yujia Zhou, Yutao Zhu, Jianxun Lian, Chaozhuo Li, Zhicheng Dou, Defu Lian, and Jian-Yun Nie. 2024c. [Information retrieval meets large](https://doi.org/10.1145/3589335.3641299) [language models.](https://doi.org/10.1145/3589335.3641299) In *Companion Proceedings of the ACM on Web Conference 2024*, WWW '24, page 1586–1589, New York, NY, USA. Association for Computing Machinery.
- <span id="page-10-9"></span>Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. 2023. The flan collection: Designing data and methods for effective instruction tuning. *arXiv preprint arXiv:2301.13688*.
- <span id="page-10-6"></span>Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, and Chang Zhou. 2023. # instag: Instruction tagging for diversity and complexity analysis. *arXiv preprint arXiv:2308.07074*.
- <span id="page-10-1"></span>Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. 2022. Unified structure generation for universal information extraction. *arXiv preprint arXiv:2203.12277*.
- <span id="page-10-0"></span>Haoran Luo, Haihong E, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guanting Dong, Meina Song, and Wei Lin. 2023a. [Chatkbqa:](http://arxiv.org/abs/2310.08975) [A generate-then-retrieve framework for knowledge](http://arxiv.org/abs/2310.08975) [base question answering with fine-tuned large lan](http://arxiv.org/abs/2310.08975)[guage models.](http://arxiv.org/abs/2310.08975)

<span id="page-11-3"></span>Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023b. Wizardcoder: Empowering code large language models with evolinstruct. *arXiv preprint arXiv:2306.08568*.

#### <span id="page-11-1"></span>OpenAI. 2023. [Gpt-4 technical report.](http://arxiv.org/abs/2303.08774)

- <span id="page-11-0"></span>Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022a. [Training language models to follow instructions with](http://arxiv.org/abs/2203.02155) [human feedback.](http://arxiv.org/abs/2203.02155)
- <span id="page-11-11"></span>Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022b. [Training language models to follow instructions with](http://arxiv.org/abs/2203.02155) [human feedback.](http://arxiv.org/abs/2203.02155)
- <span id="page-11-17"></span>Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. [Deepspeed: System opti](https://doi.org/10.1145/3394486.3406703)[mizations enable training deep learning models with](https://doi.org/10.1145/3394486.3406703) [over 100 billion parameters.](https://doi.org/10.1145/3394486.3406703) In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, KDD '20, page 3505–3506, New York, NY, USA. Association for Computing Machinery.
- <span id="page-11-6"></span>Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Tali Bers, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. 2022. [Multi](http://arxiv.org/abs/2110.08207)[task prompted training enables zero-shot task gener](http://arxiv.org/abs/2110.08207)[alization.](http://arxiv.org/abs/2110.08207)
- <span id="page-11-14"></span>Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. [Hugging](http://arxiv.org/abs/2303.17580)[gpt: Solving ai tasks with chatgpt and its friends in](http://arxiv.org/abs/2303.17580) [hugging face.](http://arxiv.org/abs/2303.17580)
- <span id="page-11-9"></span>Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Nathaneal Scharli, Aakanksha Chowdhery, Philip Mansfield, Blaise Aguera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan

- Karthikesalingam, and Vivek Natarajan. 2022. [Large](http://arxiv.org/abs/2212.13138) [language models encode clinical knowledge.](http://arxiv.org/abs/2212.13138)
- <span id="page-11-8"></span>Xiaoshuai Song, Keqing He, Pei Wang, Guanting Dong, Yutao Mou, Jingang Wang, Yunsen Xian, Xunliang Cai, and Weiran Xu. 2023. [Large language models](http://arxiv.org/abs/2310.10176) [meet open-world intent discovery and recognition:](http://arxiv.org/abs/2310.10176) [An evaluation of chatgpt.](http://arxiv.org/abs/2310.10176)
- <span id="page-11-15"></span>Xiaoshuai Song, Zhengyang Wang, Keqing He, Guanting Dong, Yutao Mou, Jinxu Zhao, and Weiran Xu. 2024. [Knowledge editing on black-box large lan](http://arxiv.org/abs/2402.08631)[guage models.](http://arxiv.org/abs/2402.08631)
- <span id="page-11-4"></span>Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. [https://](https://github.com/tatsu-lab/stanford_alpaca) [github.com/tatsu-lab/stanford\\_alpaca](https://github.com/tatsu-lab/stanford_alpaca).
- <span id="page-11-18"></span>Erik F. Tjong Kim Sang and Fien De Meulder. 2003. [Introduction to the CoNLL-2003 shared task:](https://aclanthology.org/W03-0419) [Language-independent named entity recognition.](https://aclanthology.org/W03-0419) In *Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003*, pages 142– 147.
- <span id="page-11-7"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. [Llama: Open](http://arxiv.org/abs/2302.13971) [and efficient foundation language models.](http://arxiv.org/abs/2302.13971)
- <span id="page-11-16"></span>Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-sne. *Journal of machine learning research*, 9(11).
- <span id="page-11-5"></span>Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. 2023a. [Openchat: Advanc](http://arxiv.org/abs/2309.11235)[ing open-source language models with mixed-quality](http://arxiv.org/abs/2309.11235) [data.](http://arxiv.org/abs/2309.11235)
- <span id="page-11-10"></span>Pei Wang, Yejie Wang, Muxi Diao, Keqing He, Guanting Dong, and Weiran Xu. 2024a. [Multi-perspective](http://arxiv.org/abs/2402.11279) [consistency enhances confidence estimation in large](http://arxiv.org/abs/2402.11279) [language models.](http://arxiv.org/abs/2402.11279)
- <span id="page-11-2"></span>Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang. 2023b. [Gpt-ner: Named entity recognition via large](http://arxiv.org/abs/2304.10428) [language models.](http://arxiv.org/abs/2304.10428)
- <span id="page-11-13"></span>Yejie Wang, Keqing He, Guanting Dong, Pei Wang, Weihao Zeng, Muxi Diao, Yutao Mou, Mengdi Zhang, Jingang Wang, Xunliang Cai, and Weiran Xu. 2024b. [Dolphcoder: Echo-locating code large language mod](http://arxiv.org/abs/2402.09136)[els with diverse and multi-objective instruction tun](http://arxiv.org/abs/2402.09136)[ing.](http://arxiv.org/abs/2402.09136)
- <span id="page-11-12"></span>Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. 2023c. How far can camels go? exploring the state of instruction tuning on open resources. *arXiv preprint arXiv:2306.04751*.

- <span id="page-12-7"></span>Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023d. [Self-instruct: Aligning language](http://arxiv.org/abs/2212.10560) [models with self-generated instructions.](http://arxiv.org/abs/2212.10560)
- <span id="page-12-4"></span>Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35:24824–24837.
- <span id="page-12-19"></span>Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. 2023. [Magicoder: Source code is](http://arxiv.org/abs/2312.02120) [all you need.](http://arxiv.org/abs/2312.02120)
- <span id="page-12-11"></span>Shaohua Wu, Xudong Zhao, Tong Yu, Rongguo Zhang, Chong Shen, Hongli Liu, Feng Li, Hong Zhu, Jiangang Luo, Liang Xu, et al. 2021. Yuan 1.0: Largescale pre-trained language model in zero-shot and few-shot learning. *arXiv preprint arXiv:2110.04725*.
- <span id="page-12-12"></span>Yuxiang Wu, Guanting Dong, and Weiran Xu. 2023. [Semantic parsing by large language models for intri](http://arxiv.org/abs/2310.10520)[cate updating strategies of zero-shot dialogue state](http://arxiv.org/abs/2310.10520) [tracking.](http://arxiv.org/abs/2310.10520)
- <span id="page-12-8"></span>Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023a. [Wizardlm: Empowering large lan](http://arxiv.org/abs/2304.12244)[guage models to follow complex instructions.](http://arxiv.org/abs/2304.12244)
- <span id="page-12-0"></span>Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, and Enhong Chen. 2023b. [Large language models for](http://arxiv.org/abs/2312.17617) [generative information extraction: A survey.](http://arxiv.org/abs/2312.17617)
- <span id="page-12-9"></span>Runxin Xu, Fuli Luo, Zhiyuan Zhang, Chuanqi Tan, Baobao Chang, Songfang Huang, and Fei Huang. 2021. [Raise a child in large language model: To](https://doi.org/10.18653/v1/2021.emnlp-main.749)[wards effective and generalizable fine-tuning.](https://doi.org/10.18653/v1/2021.emnlp-main.749) In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 9514– 9528, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- <span id="page-12-16"></span>Mingfeng Xue, Dayiheng Liu, Kexin Yang, Guanting Dong, Wenqiang Lei, Zheng Yuan, Chang Zhou, and Jingren Zhou. 2023. [Occuquest: Mitigating occupa](http://arxiv.org/abs/2310.16517)[tional bias for inclusive large language models.](http://arxiv.org/abs/2310.16517)
- <span id="page-12-21"></span>Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a. [Tree of thoughts: Deliberate](http://arxiv.org/abs/2305.10601) [problem solving with large language models.](http://arxiv.org/abs/2305.10601)
- <span id="page-12-20"></span>Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023b. [React: Synergizing reasoning and acting in language](http://arxiv.org/abs/2210.03629) [models.](http://arxiv.org/abs/2210.03629)
- <span id="page-12-22"></span>Wen-tau Yih, Matthew Richardson, Chris Meek, Ming-Wei Chang, and Jina Suh. 2016. [The value of se](https://doi.org/10.18653/v1/P16-2033)[mantic parse labeling for knowledge base question](https://doi.org/10.18653/v1/P16-2033) [answering.](https://doi.org/10.18653/v1/P16-2033) In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 201–206, Berlin,

- Germany. Association for Computational Linguistics.
- <span id="page-12-3"></span>Shangjian Yin, Peijie Huang, Yuhong Xu, Haojing Huang, and Jiatian Chen. 2024. [Do large language](http://arxiv.org/abs/2403.04481) [model understand multi-intent spoken language ?](http://arxiv.org/abs/2403.04481)
- <span id="page-12-6"></span>Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. [Meta](http://arxiv.org/abs/2309.12284)[math: Bootstrap your own mathematical questions](http://arxiv.org/abs/2309.12284) [for large language models.](http://arxiv.org/abs/2309.12284)
- <span id="page-12-10"></span>Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Fei Huang, and Songfang Huang. 2023a. [HyPe: Better pre](https://doi.org/10.18653/v1/2023.acl-long.182)[trained language model fine-tuning with hidden rep](https://doi.org/10.18653/v1/2023.acl-long.182)[resentation perturbation.](https://doi.org/10.18653/v1/2023.acl-long.182) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3246– 3264, Toronto, Canada. Association for Computational Linguistics.
- <span id="page-12-15"></span>Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Fangyuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, and Maosong Sun. 2024. Revisiting out-ofdistribution robustness in nlp: Benchmarks, analysis, and llms evaluations. *Advances in Neural Information Processing Systems*, 36.
- <span id="page-12-5"></span>Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. 2023b. [Scal](http://arxiv.org/abs/2308.01825)[ing relationship on learning mathematical reasoning](http://arxiv.org/abs/2308.01825) [with large language models.](http://arxiv.org/abs/2308.01825)
- <span id="page-12-13"></span>Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. 2023c. [Rrhf: Rank](http://arxiv.org/abs/2304.05302) [responses to align language models with human feed](http://arxiv.org/abs/2304.05302)[back without tears.](http://arxiv.org/abs/2304.05302)
- <span id="page-12-17"></span>Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. 2023. Mammoth: Building math generalist models through hybrid instruction tuning. *arXiv preprint arXiv:2309.05653*.
- <span id="page-12-18"></span>Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. 2024. [Mammoth2: Scaling instructions from the](http://arxiv.org/abs/2405.03548) [web.](http://arxiv.org/abs/2405.03548)
- <span id="page-12-14"></span>Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin Wang. 2024a. [In](http://arxiv.org/abs/2308.10792)[struction tuning for large language models: A survey.](http://arxiv.org/abs/2308.10792)
- <span id="page-12-2"></span>Zhen Zhang, Yuhua Zhao, Hang Gao, and Mengting Hu. 2024b. [Linkner: Linking local named entity](http://arxiv.org/abs/2402.10573) [recognition models to large language models using](http://arxiv.org/abs/2402.10573) [uncertainty.](http://arxiv.org/abs/2402.10573)
- <span id="page-12-1"></span>Gang Zhao, Xiaocheng Gong, Xinjie Yang, Guanting Dong, Shudong Lu, and Si Li. 2023. [Demosg:](http://arxiv.org/abs/2310.10481) [Demonstration-enhanced schema-guided generation](http://arxiv.org/abs/2310.10481) [for low-resource event extraction.](http://arxiv.org/abs/2310.10481)

- <span id="page-13-2"></span>Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. [Judging](http://arxiv.org/abs/2306.05685) [llm-as-a-judge with mt-bench and chatbot arena.](http://arxiv.org/abs/2306.05685)
- <span id="page-13-1"></span>Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2023a. Lima: Less is more for alignment. *arXiv preprint arXiv:2305.11206*.
- <span id="page-13-4"></span>Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023b. [Instruction-following evaluation for](http://arxiv.org/abs/2311.07911) [large language models.](http://arxiv.org/abs/2311.07911)
- <span id="page-13-3"></span>Sizhe Zhou, Yu Meng, Bowen Jin, and Jiawei Han. 2024. Grasping the essentials: Tailoring large language models for zero-shot relation extraction. *arXiv preprint arXiv:2402.11142*.
- <span id="page-13-0"></span>Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zhicheng Dou, and Ji-Rong Wen. 2024. [Large lan](http://arxiv.org/abs/2308.07107)[guage models for information retrieval: A survey.](http://arxiv.org/abs/2308.07107)

#### <span id="page-13-5"></span>A SFT Datasets

We investigate the data composition issues of mathematical reasoning, coding, and general capabilities in the SFT stage from the following SFT datasets.

- Code Alpaca [\(Chaudhary,](#page-9-12) [2023\)](#page-9-12) aims to build and share an instruction-following LLaMA model for code generation. which is fully based on Stanford Alpaca and contains 20K data used for fine-tuning the model. The Code Alpaca dataset has been open-sourced[4](#page-13-7) .
- GSM8K RFT [\(Yuan et al.,](#page-12-5) [2023b\)](#page-12-5) is a mathematical dataset enhanced by integrating multiple reasoning paths based on the original GSM8K dataset [\(Cobbe et al.,](#page-9-3) [2021\)](#page-9-3) through the rejection sampling. It contains 7.5K questions and 110K responses in the training set. The GSM8k RFT dataset has been opensourced [5](#page-13-8) .
- ShareGPT refers to the multi-turn chatting histories used by Vicuna [\(Chiang et al.,](#page-9-10) [2023\)](#page-9-10). ShareGPT includes 86K human queries and responses from ChatGPT and other chatbots. The GSM8k RFT dataset has been opensourced [6](#page-13-9) .

The following table [2](#page-13-10) presents the statistics of three datasets at different subset proportion (k).

<span id="page-13-10"></span>

| Data Ratio | GSM8K RFT | CodeAlpaca | ShareGPT |
|------------|-----------|------------|----------|
| K=1/1      | 110142    | 20022      | 86060    |
| K=1/4      | 27535     | 5005       | 21515    |
| K=1/16     | 6883      | 1251       | 5378     |
| K=1/64     | 1720      | 312        | 1344     |
| K=1/256    | 430       | 78         | 336      |

Table 2: Data statistics of three datasets at different subset proportion (k).

#### <span id="page-13-6"></span>B Evaluation metrics

We use the following metrics to measure the aligned large language models.

<span id="page-13-7"></span><sup>4</sup> <https://github.com/sahil280114/codealpaca>

<span id="page-13-9"></span><span id="page-13-8"></span><sup>5</sup> <https://github.com/OFA-Sys/gsm8k-ScRel>

<sup>6</sup>Exact dataset of ShareGPT (<https://sharegpt.com/>) has not been released. We instead use a reproduced version from [https://huggingface.co/datasets/](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) [anon8231489123/ShareGPT\\_Vicuna\\_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) cleaned raw dataset, and follow Vicuna preprocess.

- HumanEval [\(Chen et al.,](#page-9-4) [2021\)](#page-9-4) consists of 164 original programming problems, with an average of 9.6 test cases allocated to each problem. To ensure a thorough assessment of the functional correctness of LLMsynthesized code, HumanEval+ extends the number of test cases significantly, averaging at 774.8 test cases per problem. We use the same method as [\(Chen et al.,](#page-9-4) [2021\)](#page-9-4)to obtain unbiased estimates of Pass@k under greedy decoding. To facilitate the reproducibility of our results, we use the open-source github repository BigCode [\(Ben Allal et al.,](#page-8-3) [2022\)](#page-8-3) to evaluate all the HumanEval scores in this paper [7](#page-14-2) .
- GSM8K [\(Cobbe et al.,](#page-9-3) [2021\)](#page-9-3) is a math word problem dataset used to measure large language model math reasoning ability. We use the default test set to measure the model. We calculate the score based on greedy decoding accuracy (maj@1). In this paper, we use the open-source github repository gsm8k-ScRel [8](#page-14-3) to evaluate all the GSM8k scores.
- MT-Bench [\(Zheng et al.,](#page-13-2) [2023\)](#page-13-2) is a significant benchmark that contribute to the evaluation and advancement of chatbot models and LLMs in different contexts. MT-Bench[9](#page-14-4) evaluates LLMs on multi-turn dialogues using comprehensive questions tailored to handling conversations. It provides a comprehensive set of questions specifically designed for assessing the capabilities of models in handling multi-turn dialogues.

We also supplement more benchmark evaluation results in the appendix F to verify the generalization of our conclusions:

• MATH [\(Hendrycks et al.,](#page-10-15) [2021\)](#page-10-15) is a dataset with challenging high-school math problems. Problems are classified into the following topics: Prealgebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, and Precalculus. Problems in MATH are harder and more diverse than in GSM8K. In this paper, we use the open-source github repository gsm8k-ScRel to evaluate all

- the MATH scores. We use 500 test problems from [\(Lightman et al.,](#page-10-21) [2023\)](#page-10-21) as out-of-domain math benchmark.
- MBPP [\(Austin et al.,](#page-8-4) [2021\)](#page-8-4) consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entrylevel programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases. To facilitate the reproducibility of our results, we use the open-source github repository BigCode [\(Ben Allal et al.,](#page-8-3) [2022\)](#page-8-3) to evaluate all the MBPP scores in this paper.

# <span id="page-14-0"></span>C Implementation Details

We fine-tune all the SFT datasets with 3 epochs and a batch size of 128 on NVIDIA A100 GPUs. We use 8 GPUs for 7B and 13B models, 16 GPUs for 33B models during fine-tuning. We use a peak learning rate of 2e-5 with a 3% learning rate warmup. We evaluate the results on the final epoch. We use greedy decode to calculate Pass@1 and maj@1. Since the scores of MT-bench will fluctuate, we conducted three experiments and took the average.

All experiments are conducted using the default template of the FastChat framework [\(Zheng et al.,](#page-13-2) [2023\)](#page-13-2), as shown in the figure below:

#### Prompt Template

A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {Query} ASSISTANT:

To facilitate the replication of our results, all datasets and evaluation benchmarks used in our experiments have been open-sourced and their detailed sources are indicated. We will also opensource our code after the blind review process.

### <span id="page-14-1"></span>D Estimating FLOPs of SFT

Training FLOPs. We mainly follow the notations of [\(Kaplan et al.,](#page-10-22) [2020\)](#page-10-22) here.

For each input sample of length nctx in SFT dataset (GSM8K, CodeAlpaca, ShareGPT), we can split it into two parts:

$$n_{ctx} = n_Q + n_R \tag{1}$$

<span id="page-14-2"></span><sup>7</sup> [https://github.com/bigcode-project/](https://github.com/bigcode-project/bigcode-evaluation-harness) [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)

<span id="page-14-3"></span><sup>8</sup> <https://github.com/OFA-Sys/gsm8k-ScRel>

<span id="page-14-4"></span><sup>9</sup> <https://huggingface.co/spaces/lmsys/mt-bench>

<span id="page-15-3"></span>![](_page_15_Figure_0.jpeg)

Figure 7: The scaling curve of LLaMA-7B in WebQSP, CoNLL 2003, IWSLT14(de-en), IWSLT14(de-en).

<span id="page-15-1"></span>

| Model size               | 7B                                                                    | 13B                         | 33B                         |
|--------------------------|-----------------------------------------------------------------------|-----------------------------|-----------------------------|
| GSM8k RFT                |                                                                       |                             |                             |
| SFT FLOPs<br>SFT GPI hrs | $\begin{array}{ c c c } 2.4 \times 10^{18} \\ \hline 6.1 \end{array}$ | $4.3 \times 10^{18}$ 12.1   | $1.1 \times 10^{19}$ 37.4   |
| Code Alpaca              |                                                                       |                             |                             |
| SFT FLOPs<br>SFT GPI hrs | $\begin{array}{ c c c c } 4.7 \times 10^{17} \\ 1.2 \end{array}$      | $7.8 \times 10^{17}$ $2.5$  | $2.0 \times 10^{18}$<br>8.2 |
| ShareGPT                 |                                                                       |                             |                             |
| SFT FLOPs<br>SFT GPI hrs | $2.2 \times 10^{18}$ 5.4                                              | $3.9 \times 10^{18}$ $10.9$ | $9.7 \times 10^{19}$ 34.0   |

Table 3: The statistics of FLOPs and GPU hours required for SFT. For 33B, we use DeepSpeed ZeRO3 (Rasley et al., 2020) for distributed training. All the GPU hours are based on NVIDIA A100 80GB GPU. Note we use non-embedding parameters to compute FLOPs in our experiments.

$$C_{\text{train}} \approx 6Nn_{ctx}N_s$$
 (2)

where  $n_Q, n_R$  denotes the length of question and generated answers respectively.  $N, N_s$  denotes the non-embedding parameters and the numbers of samples.

Therefore, We estimate the SFT FLOPs following (Kaplan et al., 2020) and GPU times in Table 3.

#### <span id="page-15-0"></span>E Validation Experiments in More SFT Abilities

To validate the generalization of our conclusions, we selected representative datasets to evaluate the capabilities of large models across different dimensions. These dimensions include **World Knowledge**: WebQuestionsSP (Yih et al., 2016), **Language Understanding**: CoNLL 2003 (Tjong Kim Sang and De Meulder, 2003), and **Translation**: IWSLT14 (Cettolo et al., 2014)

**Experimental Design:** Align the settings of RQ1 and RQ2, we introduce two settings as follows:

**1. Individual Domain:** We conduct SFT on LLaMA of various sizes using {1, 1/2, 1/4, 1/8} proportions <sup>10</sup> of the training set obtained from WebQSP, CoNLL 2003, and IWSLT14 seperately. This allowed us to evaluate each ability with various data sizes and model sizes.

2. Mixed Domain: We sampled {1, 1/2, 1/4, 1/8} amounts of training data from WebQSP, CoNLL 2003, and IWSLT14, and directly mixed them according to the corresponding proportions. In this way, we constructed datasets with fixed proportions of different ability domains, while varying the total data amount. These datasets are then used for fine-tuning the LLaMA models.

**Analysis.** As shown in Figure 7 and Table 4, we have following observations.

For the **individual domain**, the performance (P, R, F1) of the model in the language understanding (NER) task shows a positive correlation with the scaling curve of data volume. These two abilities exhibit similar scaling curve trends as the mathematical ability performance in RQ1. In the case of world knowledge (WebQSP), a similar positive correlation trend is observed in terms of F1 and Hits@1. However, when the data ratio is reduced from 1/4 to 1/8, there is a significant performance fluctuation, particularly in the performance of translation ability, which shows a relatively irregular trend. These conclusions further support the core conclusion of RQ1 that different data exhibit different scaling curves.

For the **mixed domains**, the findings align with the conclusions in RQ2, where abilities are improved with low-resource and decreased with high-resource compared to individual source abilities. This consistent conclusion holds for world knowledge, language understanding, and translation abilities.

<span id="page-15-2"></span><sup>&</sup>lt;sup>10</sup>Because these three datasets have relatively small amounts of data (a few thousand), the scaling range is from 1/1 of the data volume to 1/8 of the data volume.

<span id="page-16-3"></span>

| Datasets           | P     | CONIL03<br>R | F1    | F1    | WebQSP<br>Hits@1 | IWSLT14<br>de-en | en-de |
|--------------------|-------|--------------|-------|-------|------------------|------------------|-------|
| Single Domain(1/1) | 91.89 | 89.33        | 90.59 | 33.51 | 64.12            | 50               | 52    |
| Single Domain(1/2) | 90.59 | 87.15        | 88.83 | 27.10 | 61.87            | 46               | 43    |
| Single Domain(1/4) | 85.24 | 79.46        | 82.25 | 22.56 | 61.38            | 42               | 40    |
| Single Domain(1/8) | 83.22 | 80.42        | 81.79 | 13.63 | 49.05            | 41               | 40    |
| Mixed Domains(1/1) | 91.74 | 87.79        | 89.72 | 32.10 | 63.70            | 46               | 49    |
| Mixed Domains(1/2) | 90.69 | 86.93        | 88.77 | 29.98 | 62.29            | 45               | 45    |
| Mixed Domains(1/4) | 88.81 | 85.62        | 87.18 | 25.42 | 58.02            | 43               | 43    |
| Mixed Domains(1/8) | 86.47 | 81.18        | 83.74 | 21.36 | 56.86            | 45               | 45    |

Table 4: Results in other domains for single and mixed source settings based on LLaMA-7B.

<span id="page-16-4"></span>![](_page_16_Figure_2.jpeg)

Figure 8: The scaling curve of LLaMA-7B on MATH and MBPP benchmarks.

# <span id="page-16-2"></span>F Results on OOD Benchmarks in Math and Code

To validate the generalization of our findings on other benchmarks, we utilized GSM8K and Code Alpaca as the training sets. We further evaluated the results on the individual domain, mixed domain, and different training strategies on other specialized ability benchmark, including MATH and MBPP, which is illustrated in Table [5](#page-17-0) and Figure [8.](#page-16-4) We have the following findings:

- (1) In the individual domain, LLaMA shows a positive correlation between performance in MATH and MBPP and the data volume (consistent with RQ1).
- (2) Comparing the individual and mixed domains, LLaMA-7B exhibits a trade-off between high-resource performance conflict and lowresource performance gain in both MATH and MBPP (consistent with RQ2).
- (3) Considering the general ability results shown in Table 1, we can observe that DMT maintains competitive results in MATH and MBPP while prioritizing general abilities. This further validates the effectiveness of DMT (consistent with RQ4).

### <span id="page-16-1"></span>G Visualization of Different Layers

In this section, we compared the visualization results of the baseline model of LLaMA-13B and DMT (k=1/256) in the starting layer (Layer1), middle layer (Layer15), and ending layer (Layer31) in Figure [9](#page-18-0) and [10.](#page-18-1)

The visualization result of the starting layer are relatively chaotic, while the visualization results of the middle layer and the ending layer are clearer. And the results of the middle layer and the last layer are consistent in pointing out that both base model and model with DMT strategy exhibit a certain level of separation in the mathematical data representations, there remains a certain degree of overlap between the representations of code and general samples.

# <span id="page-16-0"></span>H Equal Data Amount VS. Equal Data Proportion

In a realistic SFT phrase for training general LLM, the data amount for different abilities is likely to differ. Therefore, instead of controlling the same amount of data, we select to mix datasets with the same proportion of subsets to better simulate realworld scenarios in above experiments. In addition, We further supplement the experimental results using different abilities mixed with the equal data amount and compare them with the results using the equal subset proportion in Table [6.](#page-17-1)

Equal Data amount Setting: we utilize the data amount of GSM8k RFT as the baseline. We sampled data with proportions of 1/16, 1/64, 1/256, and mixed samples of the same data amount from Code alpaca and ShareGPT.

Equal Proportion Setting: we sampled data with proportions of 1/16, 1/64, 1/256 according to the subset proportions of each dataset and mixed them, which is aligned with the setup in RQ2.

It can be observed that there is not a significant difference in the results of the three benchmark tests between the two settings. Therefore, these findings do not significantly impact the main experimental conclusions presented in the paper.

<span id="page-17-0"></span>

| Methods                       | Math Benchmarks |      | Code Benchmarks |      |  |
|-------------------------------|-----------------|------|-----------------|------|--|
|                               | GSM8K           | MATH | HumanEval       | MBPP |  |
| Individual domain (Scaling)   |                 |      |                 |      |  |
| Single Domain(k=1/1)          | 49.10           | 4.4  | 18.4            | 21.6 |  |
| Single Domain(k=1/4)          | 43.37           | 3.9  | 11.58           | 18.8 |  |
| Single Domain(k=1/16)         | 35.90           | 3.2  | 12.19           | 16.6 |  |
| Single Domain(k=1/64)         | 22.71           | 3.2  | 9.14            | 15.8 |  |
| Single Domain(k=1/256)        | 12.7            | 2.0  | 5.48            | 15.8 |  |
| Mixed domain (Scaling)        |                 |      |                 |      |  |
| Mixed Domain(k=1/1)           | 47.53           | 3.6  | 14.63           | 19.4 |  |
| Mixed Domain(k=1/4)           | 41.98           | 3.2  | 9.14            | 18.8 |  |
| Mixed Domain(k=1/16)          | 32.97           | 2.4  | 9.16            | 18.4 |  |
| Mixed Domain(k=1/64)          | 25.77           | 2.4  | 14.63           | 17.2 |  |
| Mixed Domain(k=1/256)         | 14.78           | 3.0  | 11.37           | 16.6 |  |
| Individual domain             |                 |      |                 |      |  |
| General only                  | 11.1            | 2.9  | 10.4            | 1.0  |  |
| Math only                     | 49.10           | 4.4  | 6.71            | 9.0  |  |
| Code only                     | 4.51            | 1.0  | 18.40           | 21.6 |  |
| Different Training Strategies |                 |      |                 |      |  |
| Multi-task learning           | 47.53           | 3.6  | 14.63           | 19.4 |  |
| Sequential Training           | 31.39           | 2.0  | 15.85           | 15.8 |  |
| Mixed Sequential Training     | 32.6            | 2.5  | 15.24           | 16.6 |  |
| DMT (k=1/256)                 | 41.92           | 3.6  | 17.68           | 19.8 |  |

<span id="page-17-1"></span>Table 5: The detailed results of LLaMA-7B, 13B with different training strategies on OOD benchmarks.

| Methods                                 | GSM8K | HumanEval | MT-Bench |
|-----------------------------------------|-------|-----------|----------|
| Mixed Domain(k=1/16, Equal Amount)      | 34.49 | 9.14      | 5.49     |
| Mixed Domain(k=1/64, Equal Amount)      | 25.02 | 13.54     | 5.21     |
| Mixed Domain(k=1/256, Equal Amount)     | 16.7  | 11.54     | 4.63     |
| Mixed Domain(k=1/16, Equal Proportion)  | 32.97 | 9.16      | 5.52     |
| Mixed Domain(k=1/64, Equal Proportion)  | 25.77 | 14.63     | 5.24     |
| Mixed Domain(k=1/256, Equal Proportion) | 14.78 | 11.37     | 4.41     |

Table 6: Comparative experiment between equal data amounts and equal subset proportions of different SFT abilities on LLaMA-7B

# I Comparison Experiment of Different Training Sequences

To investigate the impact of training order on different SFT abilities, we have conducted additional experiments with six different training orders. The results and analysis of these experiments are provided in Table [7.](#page-19-0) Based on our findings, we conclude the following:

- 1. The SFT ability trained in the final stage tend to retain relatively good performance.
- 2. If general and code abilities are trained in the first two stages, there is a noticeable performance decrease in code capability, while math capability does not show significant impact. One possible reason is that the task format of code generation

and general ability exhibits similar data distributions (as discussed in RQ3 and Discussion1). This can result in a more severe catastrophic forgetting phenomenon during continuous fine-tuning.

#### J Detailed Results of experiments

#### J.1 Results of Different Random Seeds

For each dataset, we employed random selection by utilizing a random function with three distinct seeds for sampling. Subsequently, we conducted a comparative analysis of the results obtained from different subsets on the three benchmark tests. The specific details are presented in Table [8.](#page-19-1) It can be observed that DMT maintains its superiority under three different random seed settings. The influence

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

![](_page_18_Figure_1.jpeg)

![](_page_18_Figure_2.jpeg)

Figure 9: From left to right are the visualization results of starting layer (Layer1), middle layer (Layer15), and ending layer (Layer31) on LLaMA-7B.

<span id="page-18-1"></span>![](_page_18_Figure_4.jpeg)

![](_page_18_Figure_5.jpeg)

![](_page_18_Figure_6.jpeg)

Figure 10: From left to right are the visualization results of starting layer (Layer1), middle layer (Layer15), and ending layer (Layer31) on LLaMA-7B with DMT(k=1/256) strategy.

of different subsets on experimental results is not a key factor and does not affect the overall trend.

### J.2 Results of Single Source and Mixed Source

In Table [9](#page-19-2) and Table [10,](#page-19-3) we report the detailed comparative results between mix domains and individual domains for LLaMA-7B, 3B and 33B, as the supplemental results in RQ2.

#### J.3 Results of Data Ratio (k)

In Table [11,](#page-20-0) we report The detailed results of the data ratio (k) between specific abilities and general abilities on three benchmarks, as the supplemental results in RQ3.

# J.4 Results of Specialized Data Amount of DMT

In Table [12,](#page-20-1) we report The detailed results of LLaMA-7B, 13B, 33B with different training strategies on three benchmarks, as the supplemental results in RQ4.

# J.5 Results of MT-Bench

In Figure [11,](#page-21-0) we report detailed results of LLaMA-7B, 13B, 33B with different training strategies on MT-Bench, which include coding, extraction, humanities, math, reasoning, roleplay, stem and writing abilities.

# J.6 Supplemental Results for Discussion

In Figure [12,](#page-21-1) we report the t-SNE visualizations of LLaMA-7B and LLaMA-7B with DMT(k=1/256) strategy. What's more, the bottom figure represents the scaling relationship of LLaMA-7B with DMT(k=1/256) under different values of K.

Moreover, in Table [13,](#page-20-2) we report The detailed results of LLaMA-7B, 13B, 33B with different training strategies on three benchmarks, as the supplemental results in RQ4.

<span id="page-19-0"></span>

| Methods               | GSM8K | HumanEval | MT-Bench |
|-----------------------|-------|-----------|----------|
| Code → Math → General | 31.39 | 15.85     | 5.72     |
| Math → Code → General | 29.71 | 15.85     | 5.65     |
| Code → General → Math | 48.21 | 9.75      | 4.7      |
| General → Code → Math | 48.21 | 7.9       | 4.59     |
| General → Math → Code | 37.60 | 15.85     | 3.79     |
| Math → General → Code | 26.45 | 16.46     | 3.68     |

Table 7: Results of different sequential training for LLaMA-7B

<span id="page-19-1"></span>

| Methods                       | LLaMA -7B |           | LLaMA -13B |       |           | LLaMA -33B |       |           |          |
|-------------------------------|-----------|-----------|------------|-------|-----------|------------|-------|-----------|----------|
|                               | GSM8K     | HumanEval | MT-Bench   | GSM8K | HumanEval | MT-Bench   | GSM8K | HumanEval | MT-Bench |
| Different Training Strategies |           |           |            |       |           |            |       |           |          |
| Multi-task learning           | 47.53     | 14.63     | 5.76       | 50.94 | 19.50     | 5.73       | 56.69 | 18.9      | 6.07     |
| Sequential Training           | 31.39     | 15.85     | 5.72       | 39.12 | 20.12     | 5.93       | 47.27 | 24.80     | 6.73     |
| Mixed Sequential Training     | 32.60     | 15.24     | 6.02       | 40.48 | 18.30     | 5.93       | 44.24 | 24.4      | 6.43     |
| DMT(k=1/256,random seed=1)    | 41.92     | 17.68     | 6.08       | 46.47 | 19.50     | 6.03       | 56.36 | 25.00     | 6.73     |
| DMT(k=1/256,random seed=2)    | 41.31     | 17.68     | 6.02       | 45.85 | 18.90     | 6.08       | 55.64 | 24.80     | 6.71     |
| DMT(k=1/256,random seed=3)    | 42.03     | 18.21     | 6.13       | 46.22 | 20.52     | 6.10       | 56.12 | 25.30     | 6.73     |

Table 8: The results of LLaMA-7B, 13B, 33B under different training strategies on three benchmarks. We tested the results of DMT on randomly sampling k proportion of specified data under three random seeds.

<span id="page-19-2"></span>

| Methods         | LLaMA-7B |           |          | LLaMA-13B |           |          |  |
|-----------------|----------|-----------|----------|-----------|-----------|----------|--|
|                 | GSM8K    | HumanEval | MT-Bench | GSM8K     | HumanEval | MT-Bench |  |
| Single(k=1)     | 49.10    | 18.4      | 5.88     | 51.4      | 18.4      | 6.13     |  |
| Single(k=1/4)   | 43.37    | 11.58     | 5.85     | 48.59     | 13.41     | 6.03     |  |
| Single(k=1/16)  | 35.90    | 12.19     | 5.61     | 43.00     | 12.80     | 5.66     |  |
| Single(k=1/64)  | 22.71    | 9.14      | 5.11     | 27.40     | 12.20     | 5.24     |  |
| Single(k=1/256) | 12.70    | 5.48      | 4.00     | 18.40     | 10.36     | 2.95     |  |
| Mix(k=1)        | 47.53    | 14.63     | 5.76     | 50.49     | 17.10     | 5.73     |  |
| Mix(k=1/4)      | 41.98    | 9.14      | 5.48     | 48.52     | 14.00     | 5.61     |  |
| Mix(k=1/16)     | 32.97    | 9.16      | 5.22     | 40.63     | 14.60     | 5.52     |  |
| Mix(k=1/64)     | 25.77    | 14.63     | 5.27     | 33.2      | 17.68     | 5.24     |  |
| Mix(k=1/256)    | 14.78    | 11.37     | 4.11     | 24.94     | 12.19     | 4.4      |  |

<span id="page-19-3"></span>Table 9: Comparative experiments between mix domains and individual domains for LLaMA-7B, 13B.

| Methods         | GSM8K | HumanEval | MT-Bench |
|-----------------|-------|-----------|----------|
| Single(k=1)     | 57.91 | 26.82     | 6.63     |
| Single(k=1/4)   | 56.10 | 25.61     | 6.66     |
| Single(k=1/16)  | 54.60 | 21.95     | 6.17     |
| Single(k=1/64)  | 44.60 | 18.59     | 5.99     |
| Single(k=1/256) | 29.21 | 14.02     | 2.3      |
| Mix(k=1)        | 56.69 | 18.9      | 6.07     |
| Mix(k=1/4)      | 54.54 | 22.56     | 5.92     |
| Mix(k=1/16)     | 53.33 | 26.82     | 6.26     |
| Mix(k=1/64)     | 46.66 | 18.6      | 5.73     |
| Mix(k=1/256)    | 36.54 | 17.68     | 4.58     |

Table 10: Comparative experiments between mix domains and individual domains for LLaMA-33B.

<span id="page-20-0"></span>

| Model size                        | GSM8K | HumanEval | MT-Bench |
|-----------------------------------|-------|-----------|----------|
| Mix[(code,math),1 general]        | 47.53 | 14.63     | 5.76     |
| Mix[(code,math),1/4 general]      | 48.44 | 15.85     | 5.73     |
| Mix[(code,math),1/16 general]     | 47.99 | 15.24     | 5.27     |
| Mix[(code,math),1/64 general]     | 47.23 | 14.63     | 5.16     |
| Mix[(code,math),1/256 general]    | 48.52 | 16.46     | 4.69     |
| Mix[1(code,math),general]         | 47.53 | 14.63     | 5.76     |
| Mix[1/4(code,math),general]       | 41.31 | 10.97     | 5.81     |
| Mix[1/16(code,math),general]      | 33.20 | 11.58     | 5.76     |
| Mix[1/64(code,math),general]      | 25.17 | 12.19     | 5.84     |
| Mix[1/256(code,math),general]     | 16.52 | 9.14      | 5.82     |
| Mix[1(code,math),1/64general]     | 47.68 | 14.63     | 5.09     |
| Mix[1/4(code,math),1/64general]   | 43.29 | 12.19     | 5.07     |
| Mix[1/16(code,math),1/64general]  | 33.81 | 12.19     | 5.17     |
| Mix[1/64(code,math),1/64general]  | 26.23 | 12.19     | 5.12     |
| Mix[1/256(code,math),1/64general] | 18.27 | 10.36     | 5.12     |

Table 11: The detailed results of the data ratio (k) between specific abilities and general abilities on three benchmarks.

<span id="page-20-1"></span>

| Methods                       |       | LLaMA-7B  |          | LLaMA-13B |           |          |  |
|-------------------------------|-------|-----------|----------|-----------|-----------|----------|--|
|                               | GSM8K | HumanEval | MT-Bench | GSM8K     | HumanEval | MT-Bench |  |
| Individual domain             |       |           |          |           |           |          |  |
| General only                  | 11.10 | 10.42     | 5.88     | 14.02     | 16.40     | 6.13     |  |
| Math only                     | 49.10 | 6.71      | 2.53     | 51.40     | 12.8      | 2.54     |  |
| Code only                     | 4.51  | 18.40     | 4.30     | 5.15      | 17.1      | 3.53     |  |
| Different Training Strategies |       |           |          |           |           |          |  |
| Multi-task learning           | 47.53 | 14.63     | 5.76     | 50.94     | 19.5      | 5.73     |  |
| Sequential Training           | 31.39 | 15.85     | 5.72     | 39.12     | 20.12     | 5.93     |  |
| Mixed Sequential Training     | 32.6  | 15.24     | 6.02     | 40.48     | 18.30     | 5.93     |  |
| DMT (k=1)                     | 45.79 | 14.02     | 5.63     | 50.49     | 16.46     | 5.76     |  |
| DMT (k=1/4)                   | 48.37 | 13.41     | 5.69     | 50.18     | 18.9      | 5.83     |  |
| DMT (k=1/16)                  | 43.3  | 15.24     | 5.78     | 48.59     | 18.9      | 5.96     |  |
| DMT (k=1/64)                  | 42.53 | 15.85     | 6.01     | 47.61     | 15.24     | 6.03     |  |
| DMT (k=1/256)                 | 41.92 | 17.68     | 6.08     | 46.47     | 19.5      | 6.03     |  |

<span id="page-20-2"></span>Table 12: The detailed results of LLaMA-7B, 13B with different training strategies on three benchmarks.

| Model size                                  | GSM8K | HumanEval | MT-Bench |
|---------------------------------------------|-------|-----------|----------|
| 1/1 Mix(code,math,general(w/o code math))   | 49.05 | 17.68     | 5.80     |
| 1/4 Mix(code,math,general(w/o code math))   | 43.13 | 15.85     | 5.71     |
| 1/16 Mix(code,math,general(w/o code math))  | 36.23 | 10.36     | 5.38     |
| 1/64 Mix(code,math,general(w/o code math))  | 25.62 | 10.97     | 5.21     |
| 1/256 Mix(code,math,general(w/o code math)) | 15.31 | 11.37     | 4.38     |

Table 13: The scaling curve after ablating code and math-related samples from ShareGPT

<span id="page-21-0"></span>![](_page_21_Figure_0.jpeg)

Figure 11: The detailed results of LLaMA-7B, 13B, 33B with different training strategies on MT-Bench.

<span id="page-21-1"></span>![](_page_21_Figure_2.jpeg)

Figure 12: Figures show the t-SNE visualizations of LLaMA-7B and LLaMA-7B with DMT(k=1/256) stategy.