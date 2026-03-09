# Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale

Hritik Bansal<sup>1</sup><sup>∗</sup> Karthik Gopalakrishnan<sup>2</sup>† Saket Dingliwal<sup>2</sup> Sravan Bodapati<sup>2</sup> Katrin Kirchhoff<sup>2</sup> Dan Roth<sup>2</sup>

<sup>1</sup>University of California, Los Angeles <sup>2</sup>AWS AI Labs

hbansal@cs.ucla.edu

{karthgop, skdin, sravanb, katrinki, drot}@amazon.com

# Abstract

Language models have been shown to perform better with an increase in scale on a wide variety of tasks via the in-context learning paradigm. In this paper, we investigate the hypothesis that the ability of a large language model to in-context learn-perform a task is not uniformly spread across all of its underlying components. Using a 66 billion parameter language model (OPT-66B) across a diverse set of 14 downstream tasks, we find this is indeed the case: ∼70% of attention heads and ∼20% of feed forward networks can be removed with minimal decline in task performance. We find substantial overlap in the set of attention heads (un)important for incontext learning across tasks and number of incontext examples. We also address our hypothesis through a task-agnostic lens, finding that a small set of attention heads in OPT-66B score highly on their ability to perform primitive *induction* operations associated with in-context learning, namely, prefix matching and copying. These *induction* heads overlap with taskspecific important heads, suggesting that induction heads are among the heads capable of more sophisticated behaviors associated with in-context learning. Overall, our study provides several insights that indicate large language models may be under-trained to perform in-context learning and opens up questions on how to pre-train language models to more effectively perform in-context learning.

### 1 Introduction

In recent years, large language models (LLMs) [\(Brown et al.,](#page-11-0) [2020;](#page-11-0) [Rae et al.,](#page-12-0) [2021;](#page-12-0) [Lieber et al.,](#page-11-1) [2021;](#page-11-1) [Black et al.,](#page-11-2) [2022;](#page-11-2) [Zhang et al.,](#page-13-0) [2022;](#page-13-0) [Chowd](#page-11-3)[hery et al.,](#page-11-3) [2022;](#page-11-3) [Hoffmann et al.,](#page-11-4) [2022;](#page-11-4) [Smith](#page-12-1) [et al.,](#page-12-1) [2022\)](#page-12-1) based on the Transformer architecture [\(Vaswani et al.,](#page-12-2) [2017\)](#page-12-2) pre-trained using self-

Code: [github.com/amazon-science/llm-interpret](https://github.com/amazon-science/llm-interpret)

supervision on web-scale textual corpora have revolutionized the field of natural language processing (NLP). At larger scales, these models demonstrate remarkable *emergent* [\(Wei et al.,](#page-12-3) [2022\)](#page-12-3) prowess in performing a wide variety of tasks without any form of fine-tuning, via the zero/few-shot incontext learning paradigm [\(Brown et al.,](#page-11-0) [2020\)](#page-11-0). In this paradigm (depicted in Figure [1\)](#page-0-0), LLMs are prompted to generate output text conditioned on a few (or zero) in-context examples that form solved "input-output" pairs along with a query input.

<span id="page-0-0"></span>
$$\underbrace{\frac{\text{dogs} \rightarrow \text{bark}, \text{lions} \rightarrow \text{roar}, \underbrace{\text{ducks}}_{\text{generation}}}_{\text{input prompt}} \rightarrow \underbrace{\text{quack}}_{\text{generation}}$$

Figure 1: A sample input prompt for in-context learning and the model output.

How in-context learning works has been an open question since its advent and recent studies [\(Xie](#page-12-4) [et al.,](#page-12-4) [2021;](#page-12-4) [Garg et al.,](#page-11-5) [2022;](#page-11-5) [Olsson et al.,](#page-12-5) [2022;](#page-12-5) [Min et al.,](#page-12-6) [2022b\)](#page-12-6) have begun scratching the surface toward better understanding the paradigm. In this paper, we empirically address the following key question:

*Are all LLM components really needed to perform in-context learning?*

The first way we address the aforementioned question is through the lens of task-specific importance scores and structured pruning [\(Li et al.,](#page-11-6) [2016;](#page-11-6) [Molchanov et al.,](#page-12-7) [2016;](#page-12-7) [Anwar et al.,](#page-11-7) [2017\)](#page-11-7) of components underlying modern LLMs, which are primarily stacks composed of multiple highdimensional self-attention blocks that form *multiheaded attention* and densely activated *feed forward networks* (FFNs). We pick the Open Pretrained Transformer (OPT) [\(Zhang et al.,](#page-13-0) [2022\)](#page-13-0) model with 66B parameters for our analyses, which yield several surprising observations. We find that

<sup>∗</sup>Work done as an intern at AWS AI Labs.

<sup>†</sup>Corresponding author.

important attention heads are primarily clustered in the intermediate layers and important FFNs are primarily in later (31+) layers ([§4\)](#page-3-0). We find that the ability to perform zero/few-shot in-context learning on almost all of a variety of 14 NLP datasets/tasks stays nearly intact when up to 70% (∼15.7B parameters in OPT-66B) of the attention heads are removed ([§5.1\)](#page-4-0). The attention heads that are (un)important for in-context learning also seem to overlap across tasks ([§6.1\)](#page-5-0) and shots ([§6.2\)](#page-7-0), and pruning of attention heads based on a "universal" importance order computed using all 14 datasets generalizes to varying degrees on out-ofdistribution datasets ([§6.1.2\)](#page-6-0). These observations indicate that a common task-agnostic subset of the attention heads are responsible for in-context learning. We also find that only up to 20% of the FFNs (∼8.5B parameters) can be removed with minimal decline in zero/few-shot in-context learning performance ([§5.2\)](#page-4-1), indicating the importance of FFNs toward in-context learning. In addition, when accounting for the interplay between attention heads and FFNs by jointly pruning both based on their individual importance orders, the newly obtained inflection points to in-context learning performance do not deviate much from the inflection points obtained via standalone pruning ([§5.3\)](#page-5-1).

The second way we address the aforementioned question is by quantifying the capacity of all attention heads in OPT-66B to perform a subset of task-agnostic primitive operations associated with in-context learning, namely, *prefix matching* and *copying*: explicitly searching for a prior occurrence of the current token in-context and copying over its suffix. [Elhage et al.](#page-11-8) [\(2021\)](#page-11-8) and [Olsson et al.](#page-12-5) [\(2022\)](#page-12-5) developed a mathematical framework to reverse-engineer a Transformer and also find such heads, termed *induction heads*, and explored the hypothesis that such heads drive in-context learning with model sizes up to 13B parameters in a mostly task-agnostic fashion. Using this framework, we compute task-agnostic scores for prefix matching and copying for each attention head and find that a small set of heads in OPT-66B have nontrivial scores for both primitives ([§6.3\)](#page-8-0). Qualitative inspection and quantitative analyses show that these heads overlap (to varying degrees) with the ones identified earlier to be important for in-context learning via our set of 14 NLP datasets/tasks, suggesting that induction heads are capable of more sophisticated behaviors associated with in-context

learning such as latent concept matching but are not the only heads with such capabilities ([§6.3.1\)](#page-8-1).

Overall, our study provides several insights about in-context learning at massive scale using both task-specific and task-agnostic settings. In a world of ever increasing language model sizes, we believe these insights serve as a strong foundation for researchers and practitioners in language modeling to build and leverage compact language models that can also demonstrate emergent abilities.

# 2 Background & Methods

In this section, we briefly describe the Open Pretrained Transformer (OPT) [\(Zhang et al.,](#page-13-0) [2022\)](#page-13-0) model used for our experiments, provide background on in-context learning and the mathematical formulation of induction heads by [Olsson et al.](#page-12-5) [\(2022\)](#page-12-5) that we leverage, and describe our adaptation of oracle and gradient-based importance score formulations for in-context learning.

# <span id="page-1-0"></span>2.1 Open Pre-trained Transformer (OPT)

OPT is a suite of language models of varying sizes that are aimed at serving as open replicas of GPT-3. The largest openly accessible model from this suite is OPT-66B with 66 billion parameters, which was also the largest publicly available dense decoderonly language model at the time of our experiments. Hence, we chose OPT-66B for our study.

Architecture: Consider a tokenized input sentence to OPT, X ∈ R N×d<sup>e</sup> , where N is the number of tokens in the sentence and d<sup>e</sup> is the embedding dimension. The input is processed by multiple decoder layers consisting of *multi-headed attention* (MHA) blocks, *layer norm* (LN) and *feed forward networks* (FFN), followed by a final FFN to produce logits over the output vocabulary. The decoder layers can be formally expressed as follows:

$$\mathbf{t}^{(\ell+1)} = \mathbf{z}^{\ell} + \text{MHA}^{\ell}(\text{LN}^{\ell}(\mathbf{z}^{\ell})) \tag{1}$$

$$\mathbf{z}^{(\ell+1)} = \mathbf{t}^{(\ell+1)} + FFN^{\ell}(\mathbf{t}^{(\ell+1)})$$
 (2)

where z <sup>1</sup> = X, and [\(1\)](#page-1-0) & [\(2\)](#page-1-0) are the residual connections corresponding to the MHA and FFN in layer ` >= 1 respectively. OPT-66B was pre-trained with a maximum sequence length of 2048 and embedding dimension d<sup>e</sup> = 9216.

MHA: In an MHA block, H attention heads are applied in parallel to the input and their outputs are concatenated. In OPT-66B, there are H = 72 attention heads of dimension d<sup>h</sup> = 128 in every layer `. An individual attention head h in layer  $\ell$  consists of three learnable matrices,  $\mathbf{W}_k^h, \mathbf{W}_q^h, \mathbf{W}_v^h \in \mathbb{R}^{d_e \times d_h}$ , all unique to the head, such that it applies self-attention  $A^h(.)$  on the input, where  $d_h = d_e/H$ . Formally, for input  $\mathbf{M}$  in layer  $\ell$ :

$$MHA^{\ell}(\mathbf{M}) = [A^{1}(\mathbf{M}); \cdots; A^{H}(\mathbf{M})]\mathbf{W}_{o}^{\ell}$$
 (3)

$$A^{h}(\mathbf{M}) = s^{h}(\mathbf{M})\mathbf{M}\mathbf{W}_{v}^{h} \tag{4}$$

$$s^{h}(\mathbf{M}) = \sigma\left(\frac{\mathbf{M}\mathbf{W}_{q}^{h}(\mathbf{W}_{k}^{h})^{T}\mathbf{M}^{T}}{\sqrt{d_{h}}}\right)$$
 (5)

where  $\sigma$  is the softmax function and  $\mathbf{W}_o^\ell \in \mathbb{R}^{d_e \times d_e}$  is a learnable output matrix unique to the MHA block in layer  $\ell$ . To ensure OPT is auto-regressive, the output of  $s^h(.)$  is masked to prevent the dependence of the hidden state of the token  $i, z_i^\ell \in \mathbb{R}^{d_e}$ , on the tokens that lie to the right of it in indices  $\{i+1,\ldots,N\}$ .

To remove a head h in layer  $\ell$  in practice, we set  $A^h(\mathbf{M})$  to be the zero matrix in Equation (3). This implies that  $\mathbf{W}_k^h, \mathbf{W}_q^h, \mathbf{W}_v^h$  can be entirely removed, and the corresponding  $d_h$  rows in  $\mathbf{W}_o^\ell$  can also be removed. In total, there are 4608 attention heads across 64 layers in OPT-66B that constitute 21.7B of the total 66B parameters.

**FFN**: Each layer  $\ell$  consists of a feed forward network (FFN) parameterized by a high-dimensional projection matrix,  $\mathbf{W}_1^{\ell} \in \mathbb{R}^{d_e \times d}$  followed by a low-dimensional projection matrix,  $\mathbf{W}_2^{\ell} \in \mathbb{R}^{d \times d_e}$  where d=36864 for OPT-66B. Formally, for input  $\mathbf{M}$  in layer  $\ell$ :

$$FFN^{\ell}(\mathbf{M}) = ReLU(LN^{\ell}(\mathbf{M})\mathbf{W}_1^{\ell})\mathbf{W}_2^{\ell} \qquad (6)$$

where ReLU is the rectified linear unit activation function and LN is the layer norm.

To remove an FFN in layer  $\ell$  in practice, we set  $FFN^\ell(\mathbf{M})$  to be the zero matrix in Equation (6). This implies  $\mathbf{W}_1^\ell, \mathbf{W}_2^\ell$  and the layer norm  $LN^\ell(.)$  for the FFN can be entirely removed. In total, FFNs constitute 43.4B parameters in OPT-66B.

### <span id="page-2-0"></span>2.2 In-Context Learning & Induction Heads

With increasingly larger language models being trained in recent years, a new paradigm of learning termed *in-context learning* (Brown et al., 2020) has become popular. In this paradigm, language models perform tasks by being prompted to generate output text conditioned on a few (or zero) incontext training examples that form solved "inputoutput" pairs for the task along with a query in-

put. There are no gradient-based model updates involved as this is a purely inference-time paradigm. Figure 1 illustrates the paradigm for the task of identifying the sound that an animal makes. In some cases, tasks can also be accompanied by task descriptions/templates to help prime the language model better, e.g., zero-shot translating from English to German using the prompt:

English phrase: I like dogs. German phrase:

While these examples involve learning and relying on latent concepts during inference, few-shot in-context learning can additionally involve explicit primitive interactions between the in-context examples. For example, with the prompt:

English phrase: I like dogs. German phrase: ich mag Hunde. English phrase: I like ducks. German phrase:

the model may rely on prior in-context translations of the tokens I and like when performing the task for the query input. Olsson et al. (2022) develop a mathematical framework toward better understanding such mechanics, starting off with a task-agnostic formulation of in-context learning as the ability of a model to better predict tokens later in the context than the tokens earlier (Kaplan et al., 2020) and defining a heuristic for it. They define a set of task-agnostic primitive operations that reflect the kind of interactions we refer to in the above example, namely, prefix matching and copying. These operations are defined in a simplistic fashion on a repeated sequence of randomly generated tokens: explicitly searching for a prior occurrence of the current token in-context and copying over its suffix. The heads that are capable of performing these operations are termed induction heads. Figure 2 depicts these operations for a repeated sequence of tokens. While these operations are intertwined in practice, the capacity of attention heads to independently perform them is computed with the scoring algorithms described in detail in Appendix A.4.

### 2.3 Importance Scores

Consider a model  $\mathcal{M}$  and a dataset  $\mathcal{D} = \{\mathcal{X}, \mathcal{Y}\}$ , where  $\mathcal{X} = \{\mathbf{x}_1, \cdots, \mathbf{x}_L\}$  and  $\mathcal{Y} = \{\mathbf{y}_1, \cdots, \mathbf{y}_L\}$  such that  $\mathbf{x}_i$  represents a prompt with few (or zero) in-context training examples along with a query input and  $\mathbf{y}_i$  represents the corresponding target output sequence. We define and compute importance

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Prefix matching and copying depicted at a given time-step for a repeated sequence of tokens.

scores for model components using such datasets to quantify their relative contributions to the model's ability to perform in-context learning.

### <span id="page-3-3"></span>**2.3.1** Oracle

Let  $\mathcal{P}_{\mathcal{M}}(\mathcal{D})$  denote a dataset/task-specific performance metric, e.g., accuracy. Given dataset  $\mathcal{D}$ , the oracle importance score of a component  $\mathcal{C}$  in  $\mathcal{M}$  is computed as follows:

$$IS_{\mathcal{C}}(\mathcal{D}) = \mathcal{P}_{\mathcal{M}}(\mathcal{D}) - \mathcal{P}_{\mathcal{M}_{\setminus \mathcal{C}}}(\mathcal{D})$$
 (7)

where  $\mathcal{M}_{\setminus \mathcal{C}}$  denotes the resultant model when  $\mathcal{C}$  is pruned from  $\mathcal{M}$ . Clearly, if pruning a component leads to poor model performance on the task, it must be important for the task. Similarly, if there is no difference or an improvement in performance upon pruning a component, it must be unimportant.

Computing oracle importance scores for K model components requires us to perform  $\mathcal{O}(K)$  evaluations for each dataset  $\mathcal{D}$ . Given that OPT-66B only has 64 FFNs, this is computationally feasible. However, it becomes infeasible to score the 4608 attention heads in OPT-66B in this manner, so we adopt the second method described below to compute importance scores for attention heads.

#### <span id="page-3-2"></span>2.3.2 Gradient-based

Given dataset  $\mathcal{D}$ , the gradient-based importance score (Molchanov et al., 2016; Michel et al., 2019) of an attention head h captures the expected sensitivity of the model to h and is computed as follows:

$$IS_h(\mathcal{D}) = \mathbb{E}_{(\mathbf{x}, \mathbf{y})} \left| A^h([\mathbf{x}; \mathbf{y}])^T \frac{\partial \mathcal{L}(\mathbf{y}|\mathbf{x})}{\partial A^h([\mathbf{x}; \mathbf{y}])} \right|$$
(8)

where; is the concatenation operator,  $(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}$  such that  $\mathbf{x}$  is a sequence of  $T_x$  tokens  $x_{1:T_x}$ ,  $\mathbf{y}$  is a sequence of  $T_y$  tokens  $y_{1:T_y}$ ,  $A^h$  is defined in (4) and the loss term in (8) is computed using the autoregressive decomposition of the log-likelihood:

$$\mathcal{L}(\mathbf{y}|\mathbf{x}) = -\frac{1}{T_y} \sum_{j=1}^{j=T_y} \log(p(y_j|\mathbf{x}, y_{1:j-1})) \quad (9)$$

These importance scores can be efficiently computed for all heads by simply performing a single forward and backward pass over the model with  $\mathcal{D}$ .

We also define the aggregated importance score of an attention head on a set of datasets  $\mathbb{S}=\{\mathcal{D}_1,\cdots,\mathcal{D}_K\}$  as follows:

$$IS_h(\mathbb{S}) = \mathbb{E}_{\mathcal{D} \sim \mathbb{S}} [IS_h(\mathcal{D})]$$
 (10)

# 3 Experimental Setup

We perform our experiments with OPT-66B on a variety of 14 NLP datasets/tasks. For consistency in the evaluation metric, we report accuracy on all tasks. Our choice of datasets and metric is in line with that of Meta AI in the OPT paper (Zhang et al., 2022). The datasets include ARC Easy and Challenge (Clark et al., 2018) and OpenBookQA (Mihaylov et al., 2018) for advanced questionanswering, HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020) and Winogrande (Sakaguchi et al., 2021) for various forms of commonsense reasoning, and the following datasets from the standard SuperGLUE benchmark (Wang et al., 2019): BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, and WSC. For a subset of experiments involving evaluation for out-of-distribution generalization, we also use 2 additional datasets: MathQA (Amini et al., 2019) and LAMBADA (Paperno et al., 2016).

We use a modified version of Eleuther AI's *lm-evaluation-harness* framework (Gao et al., 2021) for our experiments. The default framework samples in-context examples at random, which we use as-is without modification.

### <span id="page-3-0"></span>4 Importance Scores for OPT-66B

In this section, we present importance scores for all attention heads and feed forward networks in OPT-66B toward performing zero-shot and few-shot (1-shot and 5-shot) in-context learning.

#### <span id="page-3-4"></span>4.1 Attention Heads

Figure 3 depicts heatmaps of the head importance scores averaged across all tasks (as described in §2.3.2) in the zero-shot, one-shot and five-shot settings. Task-specific heatmaps are provided in Appendix A.1. We observe that the important attention heads are primarily clustered in the intermediate layers of OPT-66B in both the task-averaged and task-specific cases. We also observe some overlap in the most important attention heads across the

<span id="page-4-2"></span>![](_page_4_Figure_0.jpeg)

Figure 3: Attention head aggregate importance score heatmaps for in-context learning with OPT-66B.

different zero/few-shot settings. This is further confirmed in follow-up analysis in [§6.2.](#page-7-0)

### <span id="page-4-4"></span>4.2 Feed Forward Networks

We compute oracle importance scores (both taskspecific and averaged across tasks) for each of the 64 FFNs in OPT-66B as described in [§2.3.1](#page-3-3) in the zero-shot and few-shot settings and present them in Figure [4.](#page-5-2) Note that if the importance score, i.e., accuracy difference, for an FFN f is near zero, f is redundant. If it is negative, the model is better off without f, and if it is positive, f is important.

We observe that in the zero-shot and one-shot settings, the removal of any FFN in the early (1-30) layers of OPT-66B either gives comparable or better performance for a vast majority of tasks. In the five-shot setting however, both the early and later layers seem to have important FFNs for most tasks. We also generally observe high variance in FFN importance scores in later layers. We particularly note high variance for WSC and MultiRC, observing that removal of some individual FFNs can lead to absolute accuracy improvements/degradation of up to 20%! One hypothesis for the cause for such variance could be the more challenging nature of some of these tasks such as MultiRC, which has fairly large sequence lengths, and such FFNs might be relied upon to be able to process challenging task instances. We leave further investigation into the cause for this variance for future work.

# 5 Iterative Pruning

The observations in the previous section ([§4\)](#page-3-0) indicated that the ability to in-context learn-perform a task is not uniformly spread across all of OPT-66B's underlying components. In this section, we build on these observations and assess to what extent we can remove *multiple* attention heads and

FFNs with minimal decline in task performance.

### <span id="page-4-0"></span>5.1 Removing Attention Heads

For each task in each (zero-shot, one-shot and fiveshot) in-context learning setting, we sort all attention heads in OPT-66B in ascending order by importance score ([§4.1\)](#page-3-4). We then remove attention heads in an iterative fashion from the start of this order, 10% at a time, and re-evaluate task performance after each removal.[1](#page-4-3) Figure [5](#page-5-3) depicts the resulting accuracy trends for each task and the trends when averaged across all tasks.

We observe that the average accuracy across tasks does not change much up until ∼70% of the attention heads are removed. A fine-grained look at the individual tasks also mostly shows similar trends, with accuracy staying fairly intact until a large proportion of the attention heads are removed. Some oddities include tasks such as WSC and CB, wherein we observe that the zero-shot accuracy actually increases after the removal of 70% of the attention heads. As noted by [Zhang et al.](#page-13-0) [\(2022\)](#page-13-0), we also observe low absolute accuracy values on MultiRC for the full model and upon pruning.

# <span id="page-4-1"></span>5.2 Removing FFNs

For each task in each (zero-shot, one-shot and five-shot) in-context learning setting, we sort all FFNs in OPT-66B in ascending order by importance score ([§4.2\)](#page-4-4). We then remove FFNs in an iterative fashion from the start of this order, 10% at a time, and re-evaluate task performance after each removal. Figure [6](#page-6-1) depicts the resulting accuracy trends for each task and the trends when averaged across all tasks. We observe that in the zero-shot setting, the average accuracy across tasks does not change up until ∼20% of the FFNs are removed.

<span id="page-4-3"></span><sup>1</sup>We do not remove attention heads one at a time and reevaluate given the number of heads and evaluation cost.

<span id="page-5-2"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Feed forward network (FFN) oracle importance scores for in-context learning with OPT-66B. Each FFN is knocked off independently to compute these scores, i.e., the curves are discrete and **not** cumulative.

<span id="page-5-3"></span>![](_page_5_Figure_2.jpeg)

Figure 5: Effect on in-context learning accuracy when removing attention heads in OPT-66B in an iterative manner based on task-specific and shot-specific importance scores.

For some tasks such as PIQA, Winogrande and RTE, the accuracy does not change even if 30% of the FFNs ( $\sim$ 13B of the 66B parameters) are removed. We also observe that the inflection point after which we observe a sharp decline in accuracy changes to 10% for the few-shot settings. Overall, these observations indicate that FFNs play a critical role toward in-context learning.

#### <span id="page-5-1"></span>5.3 Combined Removal of Heads & FFNs

We now investigate whether the inflection points to in-context learning performance when removing either attention heads or FFNs in an iterative fashion still hold when removing them in *tandem*. Figure 7 depicts the average accuracy of all tasks on joint iterative removal of attention heads and FFNs, 10% at a time, based on their respective importance scores in the zero and few-shot settings.

We observe that the removal of 70% of the attention heads ( $\sim 15.7 B$  parameters) and 20% of the FFNs ( $\sim 8.5 B$  parameters) leads to a mere 5% absolute drop in the average zero-shot accuracy. In the one-shot setting, the absolute drop in accuracy is 6% on removing 70% of the attention heads and 10% of the FFNs. In the five-shot setting, the ab-

solute drop in accuracy is 4% on removing 60% of the attention heads and 20% of the FFNs. Overall, these new inflection points have deviated by at most a factor of 10% absolute, which may be attributed to the interplay between heads and FFNs.

### 6 Detailed Analysis of Attention Heads

In this section, we perform a detailed analysis of the attention heads in OPT-66B, given that incontext learning is auto-regressive in nature and attention heads explicitly encode cross-token interactions. We perform cross-task analyses to understand whether various tasks share (un)important attention heads. We also perform cross-shot analyses to study whether the (un)important attention heads for a task are shared across the zero-shot and few-shot settings. We finally quantify the capacity of the attention heads to perform task-agnostic induction operations (defined in §2.2) and study correlations with task-specific importance scores.

### <span id="page-5-0"></span>6.1 Cross-Task Analysis

Michel et al. (2019) found preliminary empirical evidence of the existence of "universally" important

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 6: Effect on in-context learning accuracy when removing feed forward networks (FFNs) in OPT-66B in an iterative manner based on task-specific and shot-specific importance scores.

<span id="page-6-2"></span>![](_page_6_Figure_2.jpeg)

Figure 7: Effect on average in-context learning accuracy when removing **both** attention heads and feed forward networks (FFNs) in OPT-66B in an iterative manner based on shot-specific task-aggregate importance scores.

attention heads in trained task-specific Transformer and BERT models via evaluating on out-of-domain test sets for machine translation and natural language inference respectively. With similar motivation, we study if the (un)important attention heads identified in various in-context learning settings for OPT-66B are shared across tasks.

#### 6.1.1 Spearman's Rank Correlation

We assess overlap in (un)important attention heads across tasks by sorting task-specific head importance scores to get head importance rankings and computing the Spearman's rank correlation coefficient (SRCC) between the rankings for every pair of tasks in the zero-shot and few-shot settings. We also sort the task-aggregate head importance scores to get the aggregate ranking and compute the SRCC against the ranking for every constituent task. All pairwise correlations for the zero and one-shot settings are depicted in Figure 8, and depicted for the five-shot setting in Appendix A.2.

In both zero and few-shot settings, we observe statistically significant (p < 0.01) positive correlations in the head importance rankings for every pair of tasks, as well as between every task's ranking

and the aggregate ranking. This indicates that the set of (un)important attention heads are clustered together across tasks. We also observe seemingly lower magnitude SRCC values between every task and ReCoRD, a long reading comprehension task which requires commonsense reasoning, indicating the amount of head overlap is proportionally lower.

#### <span id="page-6-0"></span>**6.1.2** Generalization Trends

We now investigate how well head importance rankings generalize across tasks by studying accuracy trends for a task when pruning using its own head importance ranking as well as using different head importance rankings. We perform this investigation for two sets of tasks.

The first set of tasks we investigate were used to compute the aggregate ranking: COPA, Winogrande and ReCoRD. For each of these 3 tasks, we use the self-ranking, aggregate ranking and the rankings from the tasks which share the *highest* and *lowest* SRCC with them based on Figure 8. For instance, we see that ReCoRD has the highest and lowest SRCCs with RTE and OpenBookQA respectively in the zero-shot setting, so we consider the impact of pruning based on their head

<span id="page-7-2"></span><span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 8: Spearman's rank correlation coefficients between the attention head importance rankings for various tasks. All p-values < 0.01. Coefficients in the five-shot setting are provided in Appendix A.2.

importance rankings on ReCoRD accuracy. Figures 9a, 9b and 9c depict the accuracy trends for these 3 tasks in the zero-shot setting. We observe that the accuracy on all 3 tasks when pruning using the rankings described is almost unaffected up to the 50% mark. We then observe a sharp decline in accuracy on COPA and Winogrande when the model is pruned to the 70% mark using the ranking identified via ReCoRD, the task with the lowest SRCC (0.13) with both COPA and Winogrande in Figure 8a. This indicates that even if the rankings vary between ReCoRD and COPA/Winogrande (as reflected in the low magnitude of the SRCC score), the set of attention heads important for zero-shot learning with ReCoRD are important for COPA/Winogrande too. To further verify this, we calculated and found 71% and 76% overlap between the top 30% important attention heads for ReCoRD-COPA and ReCoRD-Winogrande respectively. We also surprisingly observe that the accuracy on ReCoRD at the 70% pruning mark is better using the aggregate ranking than using the ranking for ReCoRD itself. We observe similar trends in the few-shot settings as well, depicted in Appendix A.3. A key distinction we observe relative to the zero-shot setting is that the decline/divergence in accuracy beyond the 50% pruning mark using the ReCoRD ranking is less sharp for COPA and Winogrande in the one-shot setting and fades away in the five-shot setting, indicating a convergence of important attention heads across tasks.

The second set of tasks we investigate were unseen, i.e., not used to compute the aggregate ranking: MathQA and LAMBADA. For these tasks, we

analyze accuracy trends when pruning using the self-ranking and aggregate ranking. Figures 9d and 9e depict their accuracy trends in the zero-shot setting. As expected, we observe that the self-ranking accuracy curves are somewhat higher than the aggregate ranking accuracy curves in general across both tasks. For MathQA, we also observe that the absolute difference in accuracy for both cases is within 1-2%. These indicate that the aggregate rankings generalize well to MathQA but not as much to LAMBADA. We observe similar trends in the few-shot settings, depicted in Appendix A.3.

<span id="page-7-3"></span>

|        | 0-shot                | 1-shot                | 5-shot         |
|--------|-----------------------|-----------------------|----------------|
| 0-shot | 1                     | 0.39 <sub>0.001</sub> | $0.37_{0.001}$ |
| 1-shot | 0.39 <sub>0.001</sub> | 1                     | $0.41_{0.001}$ |
| 5-shot | 0.37 <sub>0.001</sub> | 0.41 <sub>0.001</sub> | 1              |

Table 1: Spearman's rank correlation coefficient (SRCC) between attention head importance rankings for different in-context learning settings. Each cell depicts the mean and variance (subscript) in SRCC across 14 tasks. p-value < 0.01 for every correlation pair.

### <span id="page-7-0"></span>6.2 Cross-Shot Analysis

To see if the attention heads that are identified to be (un)important for a task are shared across the different zero and few-shot settings, we compute Spearman's rank correlation coefficient (SRCC) between the cross-shot head importance rankings for each task and compute the mean and variance across all 14 tasks. Table 1 depicts these metrics. We observe that the SRCC is higher for rankings within the few-shot setting (0.41 between 1-shot and 5-shot) than for rankings across the zero and

<span id="page-8-3"></span><span id="page-8-2"></span>![](_page_8_Figure_0.jpeg)

<span id="page-8-5"></span>Figure 9: Cross-task transfer of attention head importance rankings as measured by impact of pruning on accuracy in the zero-shot setting.

few-shot settings. This matches the intuition that a similar set of heads must be important within the different few-shot settings than across the zero-shot and any of the few-shot settings. However, we also see that the SRCC magnitudes for the latter (0.39 and 0.37) are not very far off. In totality, these indicate non-trivial overlap in the (un)important attention heads for tasks across shots.

### <span id="page-8-0"></span>6.3 Induction Heads in OPT-66B

We look for induction heads in OPT-66B by quantifying the capacity of all attention heads to perform prefix matching and copying using random input sequences in a task-agnostic fashion, following the definition and algorithms by Olsson et al. (2022) discussed in §2.2 and Appendix A.4.

Figures 10a and 10b depict the prefix matching and copying score heatmaps respectively for OPT-66B. We observe that a small subset of attention heads in OPT-66B have high prefix matching scores, located in the upper layers (31+) of the model. On the other hand, there are a relatively larger number of attention heads with high copying scores, although the vast majority of these are also located in the upper layers (41+). When seen in conjunction, these observations indicate that there is a sparse set of attention heads that are capable of

<span id="page-8-6"></span><span id="page-8-4"></span>performing both primitive operations and thus can be deemed plausible induction heads.

### <span id="page-8-1"></span>**6.3.1** Are Induction Heads Important?

We previously (§4.1) computed task-specific and task-aggregated importance scores for attention heads in the zero-shot and few-shot in-context learning settings. By virtue of being important for the challenging downstream tasks we consider, these attention heads are capable of sophisticated and latent behaviors associated with in-context learning that go well beyond the basic primitives of explicit prefix matching and copying. We now study whether induction heads overlap with attention heads deemed important for our chosen tasks.

A qualitative comparison of the heatmaps in Figure 10 against the heatmaps in Figure 3 indicate that induction heads do overlap with task-aggregated important attention heads. To better facilitate this comparison, we first formalize the total capacity of a model to perform prefix matching (or copying) to be the sum of the respective scores for individual attention heads in the model. We then investigate how much of this capacity is retained when attention heads are pruned in the order of least important heads first. Figure 11 depicts this comparison. We observe that much of the total prefix matching

<span id="page-9-2"></span><span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

![](_page_9_Figure_1.jpeg)

![](_page_9_Figure_2.jpeg)

<span id="page-9-1"></span>Figure 10: Attention head prefix matching and copying score heatmaps for OPT-66B.

score is retained when 20% of the least important heads are removed, with the slope of decline becoming sharp only after the 40% pruning mark. This indicates that unimportant heads also have low prefix matching scores. We also observe that the prefix matching scores are generally higher for heads important for few-shot in-context learning than for heads important for zero-shot learning. On the other hand, we observe across the zero-shot and few-shot settings that the total copying score retained on pruning attention heads rapidly and consistently declines, indicating that even unimportant heads have a non-trivial capacity to perform copying. When seen in conjunction, these observations indicate that induction heads in OPT-66B are capable of sophisticated behaviors associated with in-context learning popular downstream NLP tasks and reinforce the induction head generality arguments Olsson et al. (2022) make in the context of smaller models with stylized and synthetic tasks. We provide further evidence of this in Appendix A.5 with per-task plots which showcase that some tasks rely on induction heads more than other tasks.

<span id="page-9-3"></span>![](_page_9_Figure_5.jpeg)

(a) Prefix Matching

![](_page_9_Figure_7.jpeg)

Figure 11: Total inductive (prefix matching / copying) capacity retained as a function of percentage of attention heads pruned, where heads are pruned based on task-aggregate importance score rankings in the order of least important first.

#### 7 Related Work

There has been an interest in effectively leveraging the in-context learning paradigm (Zhao et al., 2021; Holtzman et al., 2021; Min et al., 2022a; Liu et al., 2022; Lu et al., 2022; Rubin et al., 2022; Mishra et al., 2022) ever since its introduction by Brown et al. (2020), but there have been relatively fewer studies toward better *understanding* the paradigm itself. Xie et al. (2021) cast in-context learning as implicit Bayesian inference where the language model implicitly infers a shared concept among in-context examples when making a prediction, showing that this occurs when the pre-training distribution is a mixture of Hidden Markov Models (HMMs) over latent concepts despite a mismatch with the prompt distribution. Min et al. (2022b) study the role of the in-context examples themselves, finding that the ground-truth labels are not needed in the examples and that the more important drivers are provision of the label space, the distribution of the input text and the overall format of the sequence. [Garg et al.](#page-11-5) [\(2022\)](#page-11-5) showcase that Transformer models trained from scratch can in-context learn the class of linear functions with performance comparable to the optimal least squares estimator even under distribution shifts. [Razeghi et al.](#page-12-17) [\(2022\)](#page-12-17) showcase that in-context learning performance is correlated strongly with term frequencies in the pre-training corpora used. [Olsson et al.](#page-12-5) [\(2022\)](#page-12-5) consider an alternate framing of in-context learning as the ability of a language model to better predict tokens later in the context than tokens earlier and hypothesize the existence of induction heads that are responsible for in-context learning. [Chan](#page-11-17) [et al.](#page-11-17) [\(2022\)](#page-11-17) explore training Transformers separately on controlled stimuli and natural language to showcase that Transformers exhibit striking differences in their generalization from in-context vs. in-weights information.

Several works have also focused on analyzing and interpreting how attention works. [Vig and](#page-12-18) [Belinkov](#page-12-18) [\(2019\)](#page-12-18) performed a study on GPT-2 *small* with Wikipedia sentences, finding that attention targets different parts of speech at different layer depths and aligns with dependency relations most strongly in the middle layers. [Tenney et al.](#page-12-19) [\(2019\)](#page-12-19) showcase that BERT encodes the classical NLP pipeline in an interpretable way across layers. There are works relying on different formulations for head importance, such as layer-wise relevance propagation [\(Voita et al.,](#page-12-20) [2019\)](#page-12-20), gradient-based importance and oracle knock-off importance [\(Michel](#page-12-8) [et al.,](#page-12-8) [2019\)](#page-12-8), with small task-specific trained models and report the existence of specialized heads. Given the recent trend of increasing model scale [\(Lieber et al.,](#page-11-1) [2021;](#page-11-1) [Chowdhery et al.,](#page-11-3) [2022;](#page-11-3) [Smith](#page-12-1) [et al.,](#page-12-1) [2022;](#page-12-1) [Rae et al.,](#page-12-0) [2021;](#page-12-0) [Hoffmann et al.,](#page-11-4) [2022\)](#page-11-4) toward tuning-free general-purpose language models that exhibit emergent in-context learning abilities, we draw and build on prior work to understand just how much scale is really needed and/or used for in-context learning *downstream*, an aspect somewhat eclipsed by the focus on the pre-training loss curve in scaling laws [\(Hoffmann et al.,](#page-11-4) [2022\)](#page-11-4). It is also worth noting that some of our empirical observations rely on a simple greedy approach to training-free pruning since our focus was not to optimally prune a language model with respect to performing in-context learning. [Li et al.](#page-11-18) [\(2021\)](#page-11-18) show the greedy approach is sub-optimal and produces under-estimates and [Halabi et al.](#page-11-19) [\(2022\)](#page-11-19) account

for the need to re-compute importance scores after removal of each attention head or FFN by formulating pruning as weakly sub-modular maximization.

### 8 Conclusion & Future Work

In this paper, we studied the efficacy of attention heads and feed forward networks (FFNs) in a large language model (OPT-66B) in performing in-context learning in both task-specific and task-agnostic settings. We observed that while in-context learning may have emerged via selfsupervised pre-training at scale, only a core nucleus of attention heads and FFNs seem to be important for in-context learning across a wide variety of downstream tasks. We observed that a small set of attention heads have the capacity to perform taskagnostic primitive induction operations associated with in-context learning, namely prefix matching and copying. We also saw that these induction heads overlap with task-specific important attention heads to varying degrees, indicating that induction heads are capable of more sophisticated forms of in-context learning and reinforcing claims [\(Olsson et al.,](#page-12-5) [2022\)](#page-12-5) about the generality of induction heads. Overall, our in-context learning-centric observations complement recent work [\(Hoffmann](#page-11-4) [et al.,](#page-11-4) [2022\)](#page-11-4) in indicating that large language models may be under-trained and motivate several interesting directions for future work. While induction heads are formed naturally during self-supervised pre-training in its current form, we believe it may be possible to increase the number and strength of induction heads formed by defining auxiliary pre-training objectives for primitives like prefix matching and copying. This could, in turn, translate to improved in-context learning of downstream tasks, especially tasks we observed to currently rely highly on induction heads. More generally, it may also be prudent to investigate and improve pre-training regimes to increase the number of important model components (including both attention heads and FFNs) to in-context learn-perform a wide variety of downstream tasks. We hope our work spurs research toward better understanding the behavior of large language models and consequently building compact ones afresh that can also demonstrate emergent abilities.

# References

- <span id="page-11-12"></span>Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. *arXiv preprint arXiv:1905.13319*.
- <span id="page-11-7"></span>Sajid Anwar, Kyuyeon Hwang, and Wonyong Sung. 2017. [Structured pruning of deep convolutional neu](https://arxiv.org/pdf/1512.08571.pdf)[ral networks.](https://arxiv.org/pdf/1512.08571.pdf) *ACM Journal on Emerging Technologies in Computing Systems (JETC)*, 13(3):1–18.
- <span id="page-11-11"></span>Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. [Piqa: Reasoning about physical](https://arxiv.org/abs/1911.11641) [commonsense in natural language.](https://arxiv.org/abs/1911.11641) In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439.
- <span id="page-11-2"></span>Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. 2022. [Gpt-neox-20b: An open-source autoregressive lan](https://arxiv.org/abs/2204.06745)[guage model.](https://arxiv.org/abs/2204.06745) *arXiv preprint arXiv:2204.06745*.
- <span id="page-11-0"></span>Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. [Language models are few-shot](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [learners.](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) *Advances in neural information processing systems*, 33:1877–1901.
- <span id="page-11-17"></span>Stephanie CY Chan, Ishita Dasgupta, Junkyung Kim, Dharshan Kumaran, Andrew K Lampinen, and Felix Hill. 2022. Transformers generalize differently from information stored in context vs in weights. *arXiv preprint arXiv:2210.05675*.
- <span id="page-11-3"></span>Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. [Palm: Scaling](https://arxiv.org/abs/2204.02311) [language modeling with pathways.](https://arxiv.org/abs/2204.02311) *arXiv preprint arXiv:2204.02311*.
- <span id="page-11-10"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. [Think you have solved question an](https://arxiv.org/abs/1803.05457)[swering? try arc, the ai2 reasoning challenge.](https://arxiv.org/abs/1803.05457) *arXiv preprint arXiv:1803.05457*.
- <span id="page-11-8"></span>Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. [A](https://transformer-circuits.pub/2021/framework/index.html) [mathematical framework for transformer circuits.](https://transformer-circuits.pub/2021/framework/index.html) *Transformer Circuits Thread*.
- <span id="page-11-13"></span>Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish

- Thite, Ben Wang, Kevin Wang, and Andy Zou. 2021. [A framework for few-shot language model evalua](https://doi.org/10.5281/zenodo.5371628)[tion.](https://doi.org/10.5281/zenodo.5371628)
- <span id="page-11-5"></span>Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. What can transformers learn in-context? a case study of simple function classes. *arXiv preprint arXiv:2208.01066*.
- <span id="page-11-19"></span>Marwa El Halabi, Suraj Srinivas, and Simon Lacoste-Julien. 2022. [Data-efficient structured pruning](https://arxiv.org/pdf/2203.04940.pdf) [via submodular optimization.](https://arxiv.org/pdf/2203.04940.pdf) *arXiv preprint arXiv:2203.04940*.
- <span id="page-11-4"></span>Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. [Train](https://arxiv.org/pdf/2203.15556.pdf)[ing compute-optimal large language models.](https://arxiv.org/pdf/2203.15556.pdf) *arXiv preprint arXiv:2203.15556*.
- <span id="page-11-14"></span>Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. 2021. [Surface form compe](https://doi.org/10.18653/v1/2021.emnlp-main.564)[tition: Why the highest probability answer isn't al](https://doi.org/10.18653/v1/2021.emnlp-main.564)[ways right.](https://doi.org/10.18653/v1/2021.emnlp-main.564) In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 7038–7051, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- <span id="page-11-9"></span>Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.
- <span id="page-11-6"></span>Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. 2016. Pruning filters for efficient convnets. *arXiv preprint arXiv:1608.08710*.
- <span id="page-11-18"></span>Jiaoda Li, Ryan Cotterell, and Mrinmaya Sachan. 2021. [Differentiable subset pruning of transformer heads.](https://aclanthology.org/2021.tacl-1.86/) *Transactions of the Association for Computational Linguistics*, 9:1442–1459.
- <span id="page-11-1"></span>Opher Lieber, Or Sharir, Barak Lenz, and Yoav Shoham. 2021. [Jurassic-1: Technical details and](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) [evaluation.](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) *White Paper. AI21 Labs*.
- <span id="page-11-15"></span>Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. [What](https://doi.org/10.18653/v1/2022.deelio-1.10) [makes good in-context examples for GPT-3?](https://doi.org/10.18653/v1/2022.deelio-1.10) In *Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures*, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics.
- <span id="page-11-16"></span>Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. [Fantastically or](https://doi.org/10.18653/v1/2022.acl-long.556)[dered prompts and where to find them: Overcoming](https://doi.org/10.18653/v1/2022.acl-long.556) [few-shot prompt order sensitivity.](https://doi.org/10.18653/v1/2022.acl-long.556) In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics.

- <span id="page-12-8"></span>Paul Michel, Omer Levy, and Graham Neubig. 2019. [Are sixteen heads really better than one?](https://arxiv.org/abs/1905.10650) *Advances in neural information processing systems*, 32.
- <span id="page-12-9"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. [Can a suit of armor conduct elec](https://www.semanticscholar.org/paper/Can-a-Suit-of-Armor-Conduct-Electricity-A-New-for-Mihaylov-Clark/1536e8958697c5364f68b2e2448905dbbeb3a0ca)[tricity? a new dataset for open book question answer](https://www.semanticscholar.org/paper/Can-a-Suit-of-Armor-Conduct-Electricity-A-New-for-Mihaylov-Clark/1536e8958697c5364f68b2e2448905dbbeb3a0ca)[ing.](https://www.semanticscholar.org/paper/Can-a-Suit-of-Armor-Conduct-Electricity-A-New-for-Mihaylov-Clark/1536e8958697c5364f68b2e2448905dbbeb3a0ca) *arXiv preprint arXiv:1809.02789*.
- <span id="page-12-14"></span>Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022a. [Noisy channel language](https://doi.org/10.18653/v1/2022.acl-long.365) [model prompting for few-shot text classification.](https://doi.org/10.18653/v1/2022.acl-long.365) In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5316–5330, Dublin, Ireland. Association for Computational Linguistics.
- <span id="page-12-6"></span>Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022b. [Rethinking the role of demonstra](https://arxiv.org/abs/2202.12837)[tions: What makes in-context learning work?](https://arxiv.org/abs/2202.12837) *arXiv preprint arXiv:2202.12837*.
- <span id="page-12-16"></span>Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. 2022. [Reframing](https://doi.org/10.18653/v1/2022.findings-acl.50) [instructional prompts to GPTk's language.](https://doi.org/10.18653/v1/2022.findings-acl.50) In *Findings of the Association for Computational Linguistics: ACL 2022*, pages 589–612, Dublin, Ireland. Association for Computational Linguistics.
- <span id="page-12-7"></span>Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. 2016. [Pruning convolutional](https://arxiv.org/abs/1611.06440) [neural networks for resource efficient inference.](https://arxiv.org/abs/1611.06440) *arXiv preprint arXiv:1611.06440*.
- <span id="page-12-5"></span>Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2022. [In-context learning and induction heads.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) *Transformer Circuits Thread*.
- <span id="page-12-13"></span>Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The lambada dataset: Word prediction requiring a broad discourse context. *arXiv preprint arXiv:1606.06031*.
- <span id="page-12-0"></span>Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. [Scaling language models:](https://arxiv.org/pdf/2112.11446.pdf) [Methods, analysis & insights from training gopher.](https://arxiv.org/pdf/2112.11446.pdf) *arXiv preprint arXiv:2112.11446*.
- <span id="page-12-17"></span>Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. 2022. Impact of pretraining term frequencies on few-shot reasoning. *arXiv preprint arXiv:2202.07206*.

- <span id="page-12-15"></span>Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. [Learning to retrieve prompts for in-context](https://doi.org/10.18653/v1/2022.naacl-main.191) [learning.](https://doi.org/10.18653/v1/2022.naacl-main.191) In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 2655–2671, Seattle, United States. Association for Computational Linguistics.
- <span id="page-12-11"></span>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. [Winogrande: An adver](https://arxiv.org/abs/1907.10641)[sarial winograd schema challenge at scale.](https://arxiv.org/abs/1907.10641) *Communications of the ACM*, 64(9):99–106.
- <span id="page-12-1"></span>Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022. [Using](https://arxiv.org/abs/2201.11990) [deepspeed and megatron to train megatron-turing](https://arxiv.org/abs/2201.11990) [nlg 530b, a large-scale generative language model.](https://arxiv.org/abs/2201.11990) *arXiv preprint arXiv:2201.11990*.
- <span id="page-12-19"></span>Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019. [BERT rediscovers the classical NLP pipeline.](https://doi.org/10.18653/v1/P19-1452) In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4593– 4601, Florence, Italy. Association for Computational Linguistics.
- <span id="page-12-2"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. [Attention is all](https://arxiv.org/abs/1706.03762?context=cs) [you need.](https://arxiv.org/abs/1706.03762?context=cs) *Advances in neural information processing systems*, 30.
- <span id="page-12-18"></span>Jesse Vig and Yonatan Belinkov. 2019. Analyzing the structure of attention in a transformer language model. *arXiv preprint arXiv:1906.04284*.
- <span id="page-12-20"></span>Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. 2019. [Analyzing multi](https://arxiv.org/abs/1905.09418)[head self-attention: Specialized heads do the heavy](https://arxiv.org/abs/1905.09418) [lifting, the rest can be pruned.](https://arxiv.org/abs/1905.09418) *arXiv preprint arXiv:1905.09418*.
- <span id="page-12-12"></span>Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. [Superglue: A](https://arxiv.org/abs/1905.00537) [stickier benchmark for general-purpose language un](https://arxiv.org/abs/1905.00537)[derstanding systems.](https://arxiv.org/abs/1905.00537) *Advances in neural information processing systems*, 32.
- <span id="page-12-3"></span>Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*.
- <span id="page-12-4"></span>Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. *arXiv preprint arXiv:2111.02080*.
- <span id="page-12-10"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. [Hellaswag: Can a](https://arxiv.org/abs/1803.05457) [machine really finish your sentence?](https://arxiv.org/abs/1803.05457) *arXiv preprint arXiv:1905.07830*.

<span id="page-13-0"></span>Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. [Opt: Open pre-trained transformer language](https://arxiv.org/abs/2205.01068) [models.](https://arxiv.org/abs/2205.01068) *arXiv preprint arXiv:2205.01068*.

<span id="page-13-6"></span>Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In *International Conference on Machine Learning*, pages 12697–12706. PMLR.

# A Appendix

### <span id="page-13-2"></span>A.1 Task-Specific Head Importance Scores

Figures [12,](#page-16-0) [13](#page-17-0) and [14](#page-18-0) depict the attention head importance scores for each task in the zero-shot, one-shot and five-shot settings respectively.

# <span id="page-13-3"></span>A.2 Cross-Task Analysis: Spearman's Rank Correlation

Figure [15](#page-19-0) depicts the Spearman's rank correlation coefficients (SRCC) between the attention head importance rankings for every pair of tasks in the five-shot setting. It also depicts the SRCC between the aggregate ranking and the ranking for each constituent task.

# <span id="page-13-4"></span>A.3 Cross-Task Analysis: Generalization Trends

Figures [16](#page-19-1) and [17](#page-20-0) depict the cross-task head importance ranking generalization plots in the one-shot and five-shot settings.

# <span id="page-13-1"></span>A.4 Details of Prefix Matching and Copying Scores

Algorithms [1](#page-14-0) and [2](#page-15-0) contain pseudo-code to compute prefix matching and copying scores respectively for each attention head in OPT-66B. We follow the approach described by [Olsson et al.](#page-12-5) [\(2022\)](#page-12-5), but instead of computing scores using 10 sequences with fixed length of 25, we compute these scores using 100 sequences with varying lengths to account for OPT-66B's large maximum sequence length. As in [Olsson et al.](#page-12-5) [\(2022\)](#page-12-5), we exclude a small fraction of the most and least common tokens from the model's vocabulary and randomly sample tokens for these sequences to strip out the effects of pretraining corpora memorization from our scores and inductive behavior analyses.

For prefix matching, the high-level approach is the following: take a random sequence, repeat it 4 times, perform a forward pass and then for each head, compute the attention pattern and take the average of all attention pattern entries attending from a given token back to tokens that succeeded the same token in earlier repeats.

For copying, the high-level approach is the following: take a random sequence, directly feed the sequence through each head and compute the contribution of the head to the output logits, and then measure how much the head increased the logit of the maximally attended to token over increasing the logits of other attendable tokens at each timestep. Unlike [Olsson et al.](#page-12-5) [\(2022\)](#page-12-5), we do not scale the raw scores to be in the range of -1 to 1.

# <span id="page-13-5"></span>A.5 Importance of Induction Heads to Each Task

Figures [18](#page-20-1) and [19](#page-20-2) showcase the importance of induction heads to each task via measuring the percentage of the total prefix matching and copying capacities retained as a function of percentage of attention heads pruned, where heads are pruned based on each task's head importance ranking for each in-context learning setting (zero-shot, oneshot and five-shot) in the order of least important first. A small initial slope of decline implies that unimportant heads also have low prefix matching or copying scores while a steep initial slope of decline implies unimportant heads also have high prefix matching or copying scores. We observe differences in the slopes of decline across different tasks, with tasks like HellaSwag and ReCoRD (which have high accuracies in Figure [5\)](#page-5-3) having smaller initial slopes than a task like OpenBookQA (which has relatively lower accuracy in Figure [5\)](#page-5-3). When seen in conjunction, these plots not only point to the generality of induction heads to more sophisticated behaviors associated with in-context learning but also indicate that some tasks rely on induction heads more than others.

### <span id="page-14-0"></span>Algorithm 1 Prefix Matching Scores for Attention Heads

```
Arguments: Model M
model ← Pretrained(M)
layers, heads ← model.num_layers, model.num_heads_per_layer
ranked_vocab_list ← model.tokenizer.vocab . term-frequency based vocabulary list of model
exclude_vocab_size ← 0.04 × len(ranked_vocab_list) . remove 4% most & least common tokens
ranked_vocab_list ← ranked_vocab_list[exclude_vocab_size : −exclude_vocab_size]
pref ix_matching ← [ ]
for seed in {1 · · · 100} do
  L ← 2 × seed + 23 . ensure 4L ∈ [100, 892]
  X ← random.choice(ranked_vocab_list,size = L,seed = seed, replace = F alse) . L length
random sequence with all unique tokens
  X ← repeat(X, 4) . Repeat it four times
  Y ← model.forward(X) . Forward pass the repeated sequence
  score ← zeros(layers, heads) . Zero matrix of shape layers × heads
  for layer in layers do
     for head in heads do
        att ← model.get_attention(layer, head) . Shape: 4L × 4L
        for token in {L + 1 · · · 4L} do . Repetition starts from token L + 1
           att_token ← att[token] . Shape: 4L
           for every prev_token == token do . Look at the previous repetitions of the token
              pref ix_score = att_token[prev_token + 1] . Attention given to token whose
prefix is current token
              score[layer][head] ← score[layer][head] + pref ix_score
           end for
        end for
        score[layer][head] ← score[layer][head]/3L . Normalizing by length of for loop
     end for
  end for
  pref ix_matching.append(score) . Prefix matching scores via one randomly generated example
end for
pref ix_matching ← average(pref ix_matching) . Attention head-wise average over all examples
return pref ix_matching
```

### <span id="page-15-0"></span>Algorithm 2 Copying Scores for Attention Heads

```
Arguments: Model M
Definitions: Dimension per Head D, Vocabulary Size V
model ← Pretrained(M)
layers, heads ← model.num_layers, model.num_heads_per_layer
ranked_vocab_list ← model.tokenizer.vocab . term-frequency based vocabulary list of model
exclude_vocab_size ← 0.04 × len(ranked_vocab_list) . remove 4% most & least common tokens
ranked_vocab_list ← ranked_vocab_list[exclude_vocab_size : −exclude_vocab_size]
copying_score ← [ ]
for seed in {1 · · · 100} do
   L ← 4 × (2 × seed + 23) . L ∈ [100, 892]
   X ← random.choice(ranked_vocab_list,size = L,seed = seed, replace = F alse) . L length
random sequence with all unique tokens
   score ← zeros(layers, heads) . Zero matrix of shape layers × heads
   for layer in layers do
      for head in heads do
         attn_layer_head ← model.get_attention_head(layer, head)
         out ← attn_layer_head(X) . Shape: L × D
         attention ← model.get_attention(layer, head) . Shape: L × L
         logits ← model.hidden_to_vocab(out) . Shape: L × V
         logits ← softmax(logits, dim = 1)
         for token in {1 · · ·L} do
            max_ind ← argmax(attention[token]) . Index of the token being max attended to
            attendable_input ← X[1 : token] . Attendable input tokens
            attendable_logits ← logits[token][attendable_input] . Logits of attendable tokens
            mean_of_logits ← average(attendable_logits)
            raised_logits ← attendable_logits − mean_of_logits
            relu_raised_logits ← ReLU(raised_logits) . Computing raise in logit values
            relu_raised_logit_max_ind ← relu_raised_logits[X[max_ind]]
            temp_score ← relu_raised_logit_max_ind/sum(relu_raised_logits)
            score[layer][head] ← score[layer][head] + temp_score
         end for
         score[layer][head] ← score[layer][head]/L . Normalizing by length of for loop
      end for
   end for
   copying_score.append(score) . Copying scores via one randomly generated example
end for
copying_score ← average(copying_score) . Attention head-wise average over all examples
return copying_score
```

<span id="page-16-0"></span>![](_page_16_Figure_0.jpeg)

Figure 12: Attention head importance score heatmaps for zero-shot in-context learning with OPT-66B for each task.

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

Figure 13: Attention head importance score heatmaps for one-shot in-context learning with OPT-66B for each task.

<span id="page-18-0"></span>![](_page_18_Figure_0.jpeg)

Figure 14: Attention head importance score heatmaps for five-shot in-context learning with OPT-66B for each task.

<span id="page-19-0"></span>![](_page_19_Figure_0.jpeg)

Figure 15: Spearman's rank correlation coefficients between the attention head importance rankings for various tasks in the five-shot setting. All p-values < 0.01.

<span id="page-19-1"></span>![](_page_19_Figure_2.jpeg)

Figure 16: Cross-task transfer of attention head importance rankings as measured by impact of pruning on accuracy in the one-shot setting.

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 17: Cross-task transfer of attention head importance rankings as measured by impact of pruning on accuracy in the five-shot setting.

<span id="page-20-1"></span>![](_page_20_Figure_2.jpeg)

Figure 18: Total prefix matching capacity retained as a function of percentage of attention heads pruned, where heads are pruned based on task-specific importance score rankings in the order of least important first.

<span id="page-20-2"></span>![](_page_20_Figure_4.jpeg)

Figure 19: Total copying capacity retained as a function of percentage of attention heads pruned, where heads are pruned based on task-specific importance score rankings in the order of least important first.