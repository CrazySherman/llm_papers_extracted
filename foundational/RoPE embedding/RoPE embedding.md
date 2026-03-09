# ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING

### Jianlin Su

Zhuiyi Technology Co., Ltd. Shenzhen bojonesu@wezhuiyi.com

### Ahmed Murtadha

Zhuiyi Technology Co., Ltd. Shenzhen mengjiayi@wezhuiyi.com

### Yu Lu

Zhuiyi Technology Co., Ltd. Shenzhen julianlu@wezhuiyi.com

### Bo Wen

Zhuiyi Technology Co., Ltd. Shenzhen brucewen@wezhuiyi.com

# Shengfeng Pan

Zhuiyi Technology Co., Ltd. Shenzhen nickpan@wezhuiyi.com

### Yunfeng Liu

Zhuiyi Technology Co., Ltd. Shenzhen glenliu@wezhuiyi.com

August 10, 2022

# ABSTRACT

Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: [https://huggingface.co/docs/transformers/model\\_doc/roformer](https://huggingface.co/docs/transformers/model_doc/roformer).

*Keywords* Pre-trained Language Models · Position Information Encoding · Pre-training · Natural Language Processing.

# 1 Introduction

The sequential order of words is of great value to natural language understanding. Recurrent neural networks (RRNs) based models encode tokens' order by recursively computing a hidden state along the time dimension. Convolution neural networks (CNNs) based models (CNNs) [Gehring et al.](#page-11-0) [\[2017\]](#page-11-0) were typically considered position-agnostic, but recent work [Islam et al.](#page-11-1) [\[2020\]](#page-11-1) has shown that the commonly used padding operation can implicitly learn position information. Recently, the pre-trained language models (PLMs), which were built upon the transformer [Vaswani et al.](#page-11-2) [\[2017\]](#page-11-2), have achieved the state-of-the-art performance of various natural language processing (NLP) tasks, including context representation learning [Devlin et al.](#page-11-3) [\[2019\]](#page-11-3), machine translation [Vaswani et al.](#page-11-2) [\[2017\]](#page-11-2), and language modeling [Radford et al.](#page-11-4) [\[2019\]](#page-11-4), to name a few. Unlike, RRNs and CNNs-based models, PLMs utilize the self-attention mechanism to semantically capture the contextual representation of a given corpus. As a consequence, PLMs achieve a significant improvement in terms of parallelization over RNNs and improve the modeling ability of longer intra-token relations compared to CNNs[1](#page-0-0) .

<span id="page-0-0"></span><sup>1</sup>A stack of multiple CNN layers can also capture longer intra-token relation, here we only consider single layer setting.

It is noteworthy that the self-attention architecture of the current PLMs has shown to be position-agnostic Yun et al. [2020]. Following this claim, various approaches have been proposed to encode the position information into the learning process. On one side, generated absolute position encoding through a pre-defined function Vaswani et al. [2017] was added to the contextual representations, while a trainable absolute position encoding Gehring et al. [2017], Devlin et al. [2019], Lan et al. [2020], Clark et al. [2020], Radford et al. [2019], Radford and Narasimhan [2018]. On the other side, the previous work Parikh et al. [2016], Shaw et al. [2018], Huang et al. [2018], Dai et al. [2019], Yang et al. [2019], Raffel et al. [2020], Ke et al. [2020], He et al. [2020], Huang et al. [2020] focuses on relative position encoding, which typically encodes the relative position information into the attention mechanism. In addition to these approaches, the authors of Liu et al. [2020] have proposed to model the dependency of position encoding from the perspective of Neural ODE Chen et al. [2018a], and the authors of Wang et al. [2020] have proposed to model the position information in complex space. Despite the effectiveness of these approaches, they commonly add the position information to the context representation and thus render them unsuitable for the linear self-attention architecture.

In this paper, we introduce a novel method, namely Rotary Position Embedding(RoPE), to leverage the positional information into the learning process of PLMS. Specifically, RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Note that the proposed RoPE is prioritized over the existing methods through valuable properties, including the sequence length flexibility, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Experimental results on various long text classification benchmark datasets show that the enhanced transformer with rotary position embedding, namely RoFormer, can give better performance compared to baseline alternatives and thus demonstrates the efficacy of the proposed RoPE.

In brief, our contributions are three-folds as follows:

- We investigated the existing approaches to the relative position encoding and found that they are mostly built based on the idea of the decomposition of adding position encoding to the context representations. We introduce a novel method, namely Rotary Position Embedding(RoPE), to leverage the positional information into the learning process of PLMS. The key idea is to encode relative position by multiplying the context representations with a rotation matrix with a clear theoretical interpretation.
- We study the properties of RoPE and show that it decays with the relative distance increased, which is desired for natural language encoding. We kindly argue that previous relative position encoding-based approaches are not compatible with linear self-attention.
- We evaluate the proposed RoFormer on various long text benchmark datasets. Our experiments show that it consistently achieves better performance compared to its alternatives. Some experiments with pre-trained language models are available on GitHub: https://github.com/ZhuiyiTechnology/roformer.

The remaining of the paper is organized as follows. We establish a formal description of the position encoding problem in self-attention architecture and revisit previous works in Section (2). We then describe the rotary position encoding (RoPE) and study its properties in Section (3). We report experiments in Section (4). Finally, we conclude this paper in Section (5).

### <span id="page-1-0"></span>2 Background and Related Work

#### 2.1 Preliminary

Let  $\mathbb{S}_N = \{w_i\}_{i=1}^N$  be a sequence of N input tokens with  $w_i$  being the  $i^{th}$  element. The corresponding word embedding of  $\mathbb{S}_N$  is denoted as  $\mathbb{E}_N = \{x_i\}_{i=1}^N$ , where  $x_i \in \mathbb{R}^d$  is the d-dimensional word embedding vector of token  $w_i$  without position information. The self-attention first incorporates position information to the word embeddings and transforms them into queries, keys, and value representations.

<span id="page-1-1"></span>
$$q_m = f_q(\boldsymbol{x}_m, m)$$

$$\boldsymbol{k}_n = f_k(\boldsymbol{x}_n, n)$$

$$\boldsymbol{v}_n = f_v(\boldsymbol{x}_n, n),$$
(1)

where  $q_m$ ,  $k_n$  and  $v_n$  incorporate the  $m^{th}$  and  $n^{th}$  positions through  $f_q$ ,  $f_k$  and  $f_v$ , respectively. The query and key values are then used to compute the attention weights, while the output is computed as the weighted sum over the value

<span id="page-2-1"></span>representation.

$$a_{m,n} = \frac{\exp(\frac{\mathbf{q}_{m}^{\mathsf{T}} \mathbf{k}_{n}}{\sqrt{d}})}{\sum_{j=1}^{N} \exp(\frac{\mathbf{q}_{m}^{\mathsf{T}} \mathbf{k}_{j}}{\sqrt{d}})}$$

$$\mathbf{o}_{m} = \sum_{n=1}^{N} a_{m,n} \mathbf{v}_{n}$$
(2)

The existing approaches of transformer-based position encoding mainly focus on choosing a suitable function to form Equation (1).

#### 2.2 Absolute position embedding

A typical choice of Equation (1) is

<span id="page-2-0"></span>
$$f_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i,i) := \boldsymbol{W}_{t:t\in\{q,k,v\}}(\boldsymbol{x}_i + \boldsymbol{p}_i), \tag{3}$$

where  $p_i \in \mathbb{R}^d$  is a d-dimensional vector depending of the position of token  $x_i$ . Previous work Devlin et al. [2019], Lan et al. [2020], Clark et al. [2020], Radford et al. [2019], Radford and Narasimhan [2018] introduced the use of a set of trainable vectors  $p_i \in \{p_t\}_{t=1}^L$ , where L is the maximum sequence length. The authors of Vaswani et al. [2017] have proposed to generate  $p_i$  using the sinusoidal function.

$$\begin{cases} \boldsymbol{p}_{i,2t} &= \sin(k/10000^{2t/d}) \\ \boldsymbol{p}_{i,2t+1} &= \cos(k/10000^{2t/d}) \end{cases}$$
(4)

in which  $p_{i,2t}$  is the  $2t^{th}$  element of the d-dimensional vector  $p_i$ . In the next section, we show that our proposed RoPE is related to this intuition from the sinusoidal function perspective. However, instead of directly adding the position to the context representation, RoPE proposes to incorporate the relative position information by multiplying with the sinusoidal functions.

#### 2.3 Relative position embedding

The authors of Shaw et al. [2018] applied different settings of Equation (1) as following:

<span id="page-2-2"></span>
$$f_q(\boldsymbol{x}_m) := \boldsymbol{W}_q \boldsymbol{x}_m$$

$$f_k(\boldsymbol{x}_n, n) := \boldsymbol{W}_k(\boldsymbol{x}_n + \tilde{\boldsymbol{p}}_r^k)$$

$$f_v(\boldsymbol{x}_n, n) := \boldsymbol{W}_v(\boldsymbol{x}_n + \tilde{\boldsymbol{p}}_r^v)$$
(5)

where  $\tilde{\boldsymbol{p}}_r^k, \tilde{\boldsymbol{p}}_r^v \in \mathbb{R}^d$  are trainable relative position embeddings. Note that  $r = \text{clip}(m-n, r_{\min}, r_{\max})$  represents the relative distance between position m and n. They clipped the relative distance with the hypothesis that precise relative position information is not useful beyond a certain distance. Keeping the form of Equation (3), the authors Dai et al. [2019] have proposed to decompose  $q_n^{\mathsf{T}} k_n$  of Equation (2) as

$$q_m^{\dagger} k_n = x_m^{\dagger} W_q^{\dagger} W_k x_n + x_m^{\dagger} W_q^{\dagger} W_k p_n + p_m^{\dagger} W_q^{\dagger} W_k x_n + p_m^{\dagger} W_q^{\dagger} W_k p_n,$$
 (6)

the key idea is to replace the absolute position embedding  $p_n$  with its sinusoid-encoded relative counterpart  $\tilde{p}_{m-n}$ , while the absolute position  $p_m$  in the third and fourth term with two trainable vectors  $\mathbf{u}$  and  $\mathbf{v}$  independent of the query positions. Further,  $\mathbf{W}_k$  is distinguished for the content-based and location-based key vectors  $\mathbf{x}_n$  and  $\mathbf{p}_n$ , denoted as  $\mathbf{W}_k$  and  $\widetilde{\mathbf{W}}_k$ , resulting in:

$$\boldsymbol{q}_{m}^{\mathsf{T}}\boldsymbol{k}_{n} = \boldsymbol{x}_{m}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + \boldsymbol{x}_{m}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\widetilde{\boldsymbol{W}}_{k}\tilde{\boldsymbol{p}}_{m-n} + \mathbf{u}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + \mathbf{v}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\widetilde{\boldsymbol{W}}_{k}\tilde{\boldsymbol{p}}_{m-n}$$
(7)

It is noteworthy that the position information in the value term is removed by setting  $f_v(x_j) := W_v x_j$ . Later work Raffel et al. [2020], He et al. [2020], Ke et al. [2020], Huang et al. [2020] followed these settings by only encoding the relative position information into the attention weights. However, the authors of Raffel et al. [2020] reformed Equation (6) as:

$$\boldsymbol{q}_{m}^{\mathsf{T}}\boldsymbol{k}_{n} = \boldsymbol{x}_{m}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + b_{i,j} \tag{8}$$

where  $b_{i,j}$  is a trainable bias. The authors of Ke et al. [2020] investigated the middle two terms of Equation (6) and found little correlations between absolute positions and words. The authors of Raffel et al. [2020] proposed to model a pair of words or positions using different projection matrices.

$$\boldsymbol{q}_{m}^{\mathsf{T}}\boldsymbol{k}_{n} = \boldsymbol{x}_{m}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + \boldsymbol{p}_{m}^{\mathsf{T}}\boldsymbol{\mathbf{U}}_{q}^{\mathsf{T}}\boldsymbol{\mathbf{U}}_{k}\boldsymbol{p}_{n} + b_{i,j}$$

$$\tag{9}$$

The authors of He et al. [2020] argued that the relative positions of two tokens could only be fully modeled using the middle two terms of Equation (6). As a consequence, the absolute position embeddings  $p_m$  and  $p_n$  were simply replaced with the relative position embeddings  $\tilde{p}_{m-n}$ :

<span id="page-3-1"></span>
$$\boldsymbol{q}_{m}^{\mathsf{T}}\boldsymbol{k}_{n} = \boldsymbol{x}_{m}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + \boldsymbol{x}_{m}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\tilde{\boldsymbol{p}}_{m-n} + \tilde{\boldsymbol{p}}_{m-n}^{\mathsf{T}}\boldsymbol{W}_{q}^{\mathsf{T}}\boldsymbol{W}_{k}\boldsymbol{x}_{n} \tag{10}$$

A comparison of the four variants of the relative position embeddings Radford and Narasimhan [2018] has shown that the variant similar to Equation (10) is the most efficient among the other three. Generally speaking, all these approaches attempt to modify Equation (6) based on the decomposition of Equation (3) under the self-attention settings in Equation (2), which was originally proposed in Vaswani et al. [2017]. They commonly introduced to directly add the position information to the context representations. Unlikely, our approach aims to derive the relative position encoding from Equation (1) under some constraints. Next, we show that the derived approach is more interpretable by incorporating relative position information with the rotation of context representations.

# <span id="page-3-0"></span>3 Proposed approach

In this section, we discuss the proposed rotary position embedding (RoPE). We first formulate the relative position encoding problem in Section (3.1), we then derive the RoPE in Section (3.2) and investigate its properties in Section (3.3).

#### <span id="page-3-2"></span>3.1 Formulation

Transformer-based language modeling usually leverages the position information of individual tokens through a self-attention mechanism. As can be observed in Equation (2),  $q_m^{\mathsf{T}} k_n$  typically enables knowledge conveyance between tokens at different positions. In order to incorporate relative position information, we require the inner product of query  $q_m$  and key  $k_n$  to be formulated by a function g, which takes only the word embeddings  $x_m$ ,  $x_n$ , and their relative position m-n as input variables. In other words, we hope that the inner product encodes position information only in the relative form:

$$\langle f_q(\mathbf{x}_m, m), f_k(\mathbf{x}_n, n) \rangle = g(\mathbf{x}_m, \mathbf{x}_n, m - n). \tag{11}$$

<span id="page-3-4"></span>The ultimate goal is to find an equivalent encoding mechanism to solve the functions  $f_q(x_m, m)$  and  $f_k(x_n, n)$  to conform the aforementioned relation.

#### <span id="page-3-3"></span>3.2 Rotary position embedding

#### 3.2.1 A 2D case

We begin with a simple case with a dimension d=2. Under these settings, we make use of the geometric property of vectors on a 2D plane and its complex form to prove (refer Section (3.4.1) for more details) that a solution to our formulation Equation (11) is:

$$f_{q}(\boldsymbol{x}_{m}, m) = (\boldsymbol{W}_{q}\boldsymbol{x}_{m})e^{im\theta}$$

$$f_{k}(\boldsymbol{x}_{n}, n) = (\boldsymbol{W}_{k}\boldsymbol{x}_{n})e^{in\theta}$$

$$g(\boldsymbol{x}_{m}, \boldsymbol{x}_{n}, m - n) = \operatorname{Re}[(\boldsymbol{W}_{q}\boldsymbol{x}_{m})(\boldsymbol{W}_{k}\boldsymbol{x}_{n})^{*}e^{i(m-n)\theta}]$$
(12)

where  $\text{Re}[\cdot]$  is the real part of a complex number and  $(\boldsymbol{W}_k \boldsymbol{x}_n)^*$  represents the conjugate complex number of  $(\boldsymbol{W}_k \boldsymbol{x}_n)$ .  $\theta \in \mathbb{R}$  is a preset non-zero constant. We can further write  $f_{\{q,k\}}$  in a multiplication matrix:

$$f_{\{q,k\}}(\boldsymbol{x}_{m},m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} W_{\{q,k\}}^{(11)} & W_{\{q,k\}}^{(12)} \\ W_{\{q,k\}}^{(21)} & W_{\{q,k\}}^{(22)} \end{pmatrix} \begin{pmatrix} x_{m}^{(1)} \\ x_{m}^{(2)} \end{pmatrix}$$
(13)

where  $(x_m^{(1)}, x_m^{(2)})$  is  $\boldsymbol{x}_m$  expressed in the 2D coordinates. Similarly, g can be viewed as a matrix and thus enables the solution of formulation in Section (3.1) under the 2D case. Specifically, incorporating the relative position embedding is straightforward: simply rotate the affine-transformed word embedding vector by amount of angle multiples of its position index and thus interprets the intuition behind *Rotary Position Embedding*.

#### 3.2.2 General form

In order to generalize our results in 2D to any  $x_i \in \mathbb{R}^d$  where d is even, we divide the d-dimension space into d/2 sub-spaces and combine them in the merit of the linearity of the inner product, turning  $f_{\{q,k\}}$  into:

<span id="page-4-2"></span>
$$f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m$$
(14)

where

<span id="page-4-4"></span>
$$\mathbf{R}_{\Theta,m}^{d} = \begin{pmatrix} \cos m\theta_{1} & -\sin m\theta_{1} & 0 & 0 & \cdots & 0 & 0\\ \sin m\theta_{1} & \cos m\theta_{1} & 0 & 0 & \cdots & 0 & 0\\ 0 & 0 & \cos m\theta_{2} & -\sin m\theta_{2} & \cdots & 0 & 0\\ 0 & 0 & \sin m\theta_{2} & \cos m\theta_{2} & \cdots & 0 & 0\\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2} \end{pmatrix}$$

$$(15)$$

is the rotary matrix with pre-defined parameters  $\Theta = \{\theta_i = 10000^{-2(i-1)/d}, i \in [1,2,...,d/2]\}$ . A graphic illustration of RoPE is shown in Figure (1). Applying our RoPE to self-attention in Equation (2), we obtain:

$$\boldsymbol{q}_{m}^{\mathsf{T}}\boldsymbol{k}_{n} = (\boldsymbol{R}_{\Theta,m}^{d}\boldsymbol{W}_{q}\boldsymbol{x}_{m})^{\mathsf{T}}(\boldsymbol{R}_{\Theta,n}^{d}\boldsymbol{W}_{k}\boldsymbol{x}_{n}) = \boldsymbol{x}^{\mathsf{T}}\boldsymbol{W}_{q}\boldsymbol{R}_{\Theta,n-m}^{d}\boldsymbol{W}_{k}\boldsymbol{x}_{n}$$
(16)

where  $R_{\Theta,n-m}^d = (R_{\Theta,m}^d)^{\mathsf{T}} R_{\Theta,n}^d$ . Note that  $R_{\Theta}^d$  is an orthogonal matrix, which ensures stability during the process of encoding position information. In addition, due to the sparsity of  $R_{\Theta}^d$ , applying matrix multiplication directly as in Equation (16) is not computationally efficient; we provide another realization in theoretical explanation.

In contrast to the additive nature of position embedding method adopted in the previous works, i.e., Equations (3) to (10), our approach is multiplicative. Moreover, RoPE naturally incorporates relative position information through rotation matrix product instead of altering terms in the expanded formulation of additive position encoding when applied with self-attention.

![](_page_4_Figure_10.jpeg)

<span id="page-4-1"></span>Figure 1: Implementation of Rotary Position Embedding(RoPE).

#### <span id="page-4-0"></span>3.3 Properties of RoPE

**Long-term decay:** Following Vaswani et al. [2017], we set  $\theta_i = 10000^{-2i/d}$ . One can prove that this setting provides a long-term decay property (refer to Section (3.4.3) for more details), which means the inner-product will decay when the relative position increase. This property coincides with the intuition that a pair of tokens with a long relative distance should have less connection.

<span id="page-4-3"></span>**RoPE** with linear attention: The self-attention can be rewritten in a more general form.

Attention(**Q**, **K**, **V**)<sub>m</sub> = 
$$\frac{\sum_{n=1}^{N} \sin(\boldsymbol{q}_{m}, \boldsymbol{k}_{n}) \boldsymbol{v}_{n}}{\sum_{n=1}^{N} \sin(\boldsymbol{q}_{m}, \boldsymbol{k}_{n})}.$$
(17)

<span id="page-5-1"></span>The original self-attention chooses  $sim(\boldsymbol{q}_m, \boldsymbol{k}_n) = \exp(\boldsymbol{q}_m^{\mathsf{T}} \boldsymbol{k}_n / \sqrt{d})$ . Note that the original self-attention should compute the inner product of query and key for every pair of tokens, which has a quadratic complexity  $\mathbb{O}(N^2)$ . Follow Katharopoulos et al. [2020], the linear attentions reformulate Equation (17) as

Attention
$$(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})_m = \frac{\sum_{n=1}^{N} \phi(\boldsymbol{q}_m)^{\mathsf{T}} \varphi(\boldsymbol{k}_n) \boldsymbol{v}_n}{\sum_{n=1}^{N} \phi(\boldsymbol{q}_m)^{\mathsf{T}} \varphi(\boldsymbol{k}_n)},$$
 (18)

where  $\phi(\cdot), \varphi(\cdot)$  are usually non-negative functions. The authors of Katharopoulos et al. [2020] have proposed  $\phi(x) = \varphi(x) = \operatorname{elu}(x) + 1$  and first computed the multiplication between keys and values using the associative property of matrix multiplication. A softmax function is used in Shen et al. [2021] to normalize queries and keys separately before the inner product, which is equivalent to  $\phi(q_i) = \operatorname{softmax}(q_i)$  and  $\phi(k_j) = \exp(k_j)$ . For more details about linear attention, we encourage readers to refer to original papers. In this section, we focus on discussing incorporating RoPE with Equation (18). Since RoPE injects position information by rotation, which keeps the norm of hidden representations unchanged, we can combine RoPE with linear attention by multiplying the rotation matrix with the outputs of the non-negative functions.

Attention(Q, K, V)<sub>m</sub> = 
$$\frac{\sum_{n=1}^{N} (\mathbf{R}_{\Theta,m}^{d} \phi(\mathbf{q}_{m}))^{\mathsf{T}} (\mathbf{R}_{\Theta,n}^{d} \varphi(\mathbf{k}_{n})) \mathbf{v}_{n}}{\sum_{n=1}^{N} \phi(\mathbf{q}_{m})^{\mathsf{T}} \varphi(\mathbf{k}_{n})}.$$
 (19)

<span id="page-5-2"></span>It is noteworthy that we keep the denominator unchanged to avoid the risk of dividing zero, and the summation in the numerator could contain negative terms. Although the weights for each value  $v_i$  in Equation (19) are not strictly probabilistic normalized, we kindly argue that the computation can still model the importance of values.

#### 3.4 Theoretical Explanation

#### <span id="page-5-0"></span>3.4.1 Derivation of RoPE under 2D

Under the case of d = 2, we consider two-word embedding vectors  $x_q$ ,  $x_k$  corresponds to query and key and their position m and n, respectively. According to eq. (1), their position-encoded counterparts are:

<span id="page-5-4"></span><span id="page-5-3"></span>
$$\mathbf{q}_{m} = f_{q}(\mathbf{x}_{q}, m),$$

$$\mathbf{k}_{n} = f_{k}(\mathbf{x}_{k}, n),$$
(20)

where the subscripts of  $q_m$  and  $k_n$  indicate the encoded positions information. Assume that there exists a function g that defines the inner product between vectors produced by  $f_{\{q,k\}}$ :

$$\boldsymbol{q}_{m}^{\mathsf{T}}\boldsymbol{k}_{n} = \langle f_{q}(\boldsymbol{x}_{m}, m), f_{k}(\boldsymbol{x}_{n}, n) \rangle = g(\boldsymbol{x}_{m}, \boldsymbol{x}_{n}, n - m), \tag{21}$$

we further require below initial condition to be satisfied:

<span id="page-5-7"></span>
$$q = f_q(\mathbf{x}_q, 0),$$
  

$$\mathbf{k} = f_k(\mathbf{x}_k, 0),$$
(22)

which can be read as the vectors with empty position information encoded. Given these settings, we attempt to find a solution of  $f_q$ ,  $f_k$ . First, we take advantage of the geometric meaning of vector in 2D and its complex counter part, decompose functions in Equations (20) and (21) into:

$$f_{q}(\boldsymbol{x}_{q}, m) = R_{q}(\boldsymbol{x}_{q}, m)e^{i\Theta_{q}(\boldsymbol{x}_{q}, m)},$$

$$f_{k}(\boldsymbol{x}_{k}, n) = R_{k}(\boldsymbol{x}_{k}, n)e^{i\Theta_{k}(\boldsymbol{x}_{k}, n)},$$

$$g(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) = R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m)e^{i\Theta_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m)},$$
(23)

<span id="page-5-5"></span>where  $R_f$ ,  $R_g$  and  $\Theta_f$ ,  $\Theta_g$  are the radical and angular components for  $f_{\{q,k\}}$  and g, respectively. Plug them into Equation (21), we get the relation:

$$R_q(\mathbf{x}_q, m)R_k(\mathbf{x}_k, n) = R_g(\mathbf{x}_q, \mathbf{x}_k, n - m),$$
  

$$\Theta_k(\mathbf{x}_k, n) - \Theta_q(\mathbf{x}_q, m) = \Theta_g(\mathbf{x}_q, \mathbf{x}_k, n - m),$$
(24)

with the corresponding initial condition as:

<span id="page-5-6"></span>
$$\mathbf{q} = \|\mathbf{q}\|e^{i\theta_q} = R_q(\mathbf{x}_q, 0)e^{i\Theta_q(\mathbf{x}_q, 0)},$$

$$\mathbf{k} = \|\mathbf{k}\|e^{i\theta_k} = R_k(\mathbf{x}_k, 0)e^{i\Theta_k(\mathbf{x}_k, 0)},$$
(25)

where  $\|q\|$ ,  $\|k\|$  and  $\theta_q$ ,  $\theta_k$  are the radial and angular part of q and k on the 2D plane.

Next, we set m = n in Equation (24) and take into account initial conditions in Equation (25):

$$R_q(\mathbf{x}_q, m)R_k(\mathbf{x}_k, m) = R_q(\mathbf{x}_q, \mathbf{x}_k, 0) = R_k(\mathbf{x}_q, 0)R_k(\mathbf{x}_k, 0) = \|\mathbf{q}\| \|\mathbf{k}\|,$$
(26a)

$$\Theta_k(\boldsymbol{x}_k, m) - \Theta_q(\boldsymbol{x}_q, m) = \Theta_g(\boldsymbol{x}_q, \boldsymbol{x}_k, 0) = \|\Theta_k(\boldsymbol{x}_k, 0) - \Theta_q(\boldsymbol{x}_q, 0)\| = \|\theta_k - \theta_q\|.$$
 (26b)

On one hand, from, a straightforward solution of  $R_f$  could be formed from Equation (26a):

<span id="page-6-1"></span>
$$R_{q}(\boldsymbol{x}_{q}, m) = R_{q}(\boldsymbol{x}_{q}, 0) = \|\boldsymbol{q}\|$$

$$R_{k}(\boldsymbol{x}_{k}, n) = R_{k}(\boldsymbol{x}_{k}, 0) = \|\boldsymbol{k}\|$$

$$R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, n - m) = R_{g}(\boldsymbol{x}_{q}, \boldsymbol{x}_{k}, 0) = \|\boldsymbol{q}\|\|\boldsymbol{k}\|$$

$$(27)$$

which interprets the radial functions  $R_q$ ,  $R_k$  and  $R_g$  are independent from the position information. On the other hand, as can be noticed in Equation (26b),  $\Theta_q(\boldsymbol{x}_q,m) - \theta_q = \Theta_k(\boldsymbol{x}_k,m) - \theta_k$  indicates that the angular functions does not dependent on query and key, we set them to  $\Theta_f := \Theta_q = \Theta_k$  and term  $\Theta_f(\boldsymbol{x}_{\{q,k\}},m) - \theta_{\{q,k\}}$  is a function of position m and is independent of word embedding  $\boldsymbol{x}_{\{q,k\}}$ , we denote it as  $\phi(m)$ , yielding:

<span id="page-6-3"></span><span id="page-6-2"></span>
$$\Theta_f(\mathbf{x}_{\{q,k\}}, m) = \phi(m) + \theta_{\{q,k\}},$$
(28)

Further, by plugging n = m + 1 to Equation (24) and consider the above equation, we can get:

$$\phi(m+1) - \phi(m) = \Theta_q(\mathbf{x}_q, \mathbf{x}_k, 1) + \theta_q - \theta_k, \tag{29}$$

Since RHS is a constant irrelevant to m,  $\phi(m)$  with continuous integer inputs produce an arithmetic progression:

<span id="page-6-4"></span>
$$\phi(m) = m\theta + \gamma,\tag{30}$$

<span id="page-6-5"></span>where  $\theta, \gamma \in \mathbb{R}$  are constants and  $\theta$  is non-zero. To summarize our solutions from Equations (27) to (30):

$$f_{q}(\boldsymbol{x}_{q}, m) = \|\boldsymbol{q}\|e^{i\theta_{q}+m\theta+\gamma} = \boldsymbol{q}e^{i(m\theta+\gamma)},$$
  

$$f_{k}(\boldsymbol{x}_{k}, n) = \|\boldsymbol{k}\|e^{i\theta_{k}+n\theta+\gamma} = \boldsymbol{k}e^{i(n\theta+\gamma)}.$$
(31)

Note that we do not apply any constrains to  $f_q$  and  $f_k$  of Equation (22), thus  $f_q(x_m, 0)$  and  $f_k(x_n, 0)$  are left to choose freely. To make our results comparable to Equation (3), we define:

$$q = f_q(\mathbf{x}_m, 0) = \mathbf{W}_q \mathbf{x}_n,$$
  

$$\mathbf{k} = f_k(\mathbf{x}_n, 0) = \mathbf{W}_k \mathbf{x}_n.$$
(32)

Then, we simply set  $\gamma=0$  in Equation (31) of the final solution:

$$f_q(\mathbf{x}_m, m) = (\mathbf{W}_q \mathbf{x}_m) e^{im\theta},$$
  

$$f_k(\mathbf{x}_n, n) = (\mathbf{W}_k \mathbf{x}_n) e^{in\theta}.$$
(33)

#### 3.4.2 Computational efficient realization of rotary matrix multiplication

Taking the advantage of the sparsity of  $\mathbf{R}_{\Theta,m}^d$  in Equation (15), a more computational efficient realization of a multiplication of  $R_{\Theta}^d$  and  $\mathbf{x} \in \mathbb{R}^d$  is:

$$\boldsymbol{R}_{\Theta,m}^{d}\boldsymbol{x} = \begin{pmatrix} x_{1} \\ x_{2} \\ x_{3} \\ x_{4} \\ \vdots \\ x_{d-1} \\ x_{d} \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_{1} \\ \cos m\theta_{1} \\ \cos m\theta_{2} \\ \cos m\theta_{2} \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix} + \begin{pmatrix} -x_{2} \\ x_{1} \\ -x_{4} \\ x_{3} \\ \vdots \\ -x_{d-1} \\ x_{d} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_{1} \\ \sin m\theta_{1} \\ \sin m\theta_{2} \\ \sin m\theta_{2} \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}$$
(34)

#### <span id="page-6-0"></span>3.4.3 Long-term decay of RoPE

We can group entries of vectors  $\mathbf{q} = \mathbf{W}_q \mathbf{x}_m$  and  $\mathbf{k} = \mathbf{W}_k \mathbf{x}_n$  in pairs, and the inner product of RoPE in Equation (16) can be written as a complex number multiplication.

$$(\boldsymbol{R}_{\Theta,m}^{d}\boldsymbol{W}_{q}\boldsymbol{x}_{m})^{\intercal}(\boldsymbol{R}_{\Theta,n}^{d}\boldsymbol{W}_{k}\boldsymbol{x}_{n}) = \operatorname{Re}\left[\sum_{i=0}^{d/2-1} \boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^{*}\boldsymbol{e}^{i(m-n)\theta_{i}}\right]$$
(35)

![](_page_7_Figure_1.jpeg)

<span id="page-7-1"></span>Figure 2: Long-term decay of RoPE.

where  $\mathbf{q}_{[2i:2i+1]}$  represents the  $2i^{th}$  to  $(2i+1)^{th}$  entries of  $\mathbf{q}$ . Denote  $h_i = \mathbf{q}_{[2i:2i+1]}\mathbf{k}_{[2i:2i+1]}^*$  and  $S_j = \sum_{i=0}^{j-1} e^{i(m-n)\theta_i}$ , and let  $h_{d/2} = 0$  and  $S_0 = 0$ , we can rewrite the summation using Abel transformation

$$\sum_{i=0}^{d/2-1} \mathbf{q}_{[2i:2i+1]} \mathbf{k}_{[2i:2i+1]}^* e^{i(m-n)\theta_i} = \sum_{i=0}^{d/2-1} h_i (S_{i+1} - S_i) = -\sum_{i=0}^{d/2-1} S_{i+1} (h_{i+1} - h_i).$$
 (36)

Thus,

$$\left| \sum_{i=0}^{d/2-1} \mathbf{q}_{[2i:2i+1]} \mathbf{k}_{[2i:2i+1]}^* e^{i(m-n)\theta_i} \right| = \left| \sum_{i=0}^{d/2-1} S_{i+1} (h_{i+1} - h_i) \right|$$

$$\leq \sum_{i=0}^{d/2-1} |S_{i+1}| |(h_{i+1} - h_i)|$$

$$\leq \left( \max_{i} |h_{i+1} - h_i| \right) \sum_{i=0}^{d/2-1} |S_{i+1}|$$
(37)

Note that the value of  $\frac{1}{d/2} \sum_{i=1}^{d/2} |S_i|$  decay with the relative distance m-n increases by setting  $\theta_i = 10000^{-2i/d}$ , as shown in Figure (2).

#### <span id="page-7-0"></span>4 Experiments and Evaluation

We evaluate the proposed RoFormer on various NLP tasks as follows. We validate the performance of the proposed solution on machine translation task Section (4.1). Then, we compare our RoPE implementation with BERTDevlin et al. [2019] during the pre-training stage in Section (4.2). Based on the pre-trained model, in Section (4.3), we further carry out evaluations across different downstream tasks from GLUE benchmarksSingh et al. [2018]. In Addition, we conduct experiments using the proposed RoPE with the linear attention of PerFormer Choromanski et al. [2020] in Section (4.4). By the end, additional tests on Chinese data are included in Section (4.5). All the experiments were run on two cloud severs with 4 x V100 GPUs.

#### <span id="page-7-2"></span>4.1 Machine Translation

We first demonstrate the performance of RoFormer on sequence-to-sequence language translation tasks.

<span id="page-8-2"></span>Table 1: The proposed RoFormer gives better BLEU scores compared to its baseline alternative [Vaswani et al.](#page-11-2) [\[2017\]](#page-11-2) on the WMT 2014 English-to-German translation tas[kBojar et al.](#page-12-19) [\[2014\]](#page-12-19).

| Model                                 | BLEU |
|---------------------------------------|------|
| Transformer-baseVaswani et al. [2017] | 27.3 |
| RoFormer                              | 27.5 |

### 4.1.1 Experimental Settings

We choose the standard WMT 2014 English-German datase[tBojar et al.](#page-12-19) [\[2014\]](#page-12-19), which consists of approximately 4.5 million sentence pairs. We compare to the transformer-based baseline alternative [Vaswani et al.](#page-11-2) [\[2017\]](#page-11-2).

### 4.1.2 Implementation details

We carry out some modifications on self-attention layer of the baseline model [Vaswani et al.](#page-11-2) [\[2017\]](#page-11-2) to enable RoPE to its learning process. We replicate the setup for English-to-German translation with a vocabulary of 37k based on a joint source and target byte pair encoding(BPE[\)Sennrich et al.](#page-13-0) [\[2015\]](#page-13-0). During the evaluation, a single model is obtained by averaging the last 5 checkpoints. The result uses beam search with a beam size of 4 and length penalty 0.6. We implement the experiment in PyTorch in the fairseq toolkit (MIT License[\)Ott et al.](#page-13-1) [\[2019\]](#page-13-1). Our model is optimized with the Adam optimizer using β<sup>1</sup> = 0.9, β<sup>2</sup> = 0.98, learning rate is increased linearly from 1e − 7 to 5e − 4 and then decayed proportionally to the inverse square root of the step number. Label smoothing with 0.1 is also adopted. We report the BLE[UPapineni et al.](#page-13-2) [\[2002\]](#page-13-2) score on the test set as the final metric.

# 4.1.3 Results

We train the baseline model and our RoFormer under the same settings and report the results in Table [\(1\)](#page-8-2). As can be seen, our model gives better BLEU scores compared to the baseline Transformer.

#### <span id="page-8-0"></span>4.2 Pre-training Language Modeling

The second experiment is to validate the performance of our proposal in terms of learning contextual representations. To achieve this, we replace the original sinusoidal position encoding of BERT with our RoPE during the pre-training step.

### 4.2.1 Experimental Settings

We use the BookCorpus [Zhu et al.](#page-13-3) [\[2015\]](#page-13-3) and the Wikipedia Corpus [Foundation](#page-13-4) [\[2021\]](#page-13-4) from Huggingface Datasets library (Apache License 2.0) for pre-training. The corpus is further split into train and validation sets at 8:2 ratio. We use the masked language-modeling (MLM) loss values of the training process as an evaluation metric. The well-known BERT [Devlin et al.](#page-11-3) [\[2019\]](#page-11-3) is adopted as our baseline model. Note that we use bert-base-uncased in our experiments.

#### 4.2.2 Implementation details

For RoFormer, we replace the sinusoidal position encoding in the self-attention block of the baseline model with our proposed RoPE and realizes self-attention according to Equation [\(16\)](#page-4-2). We train both BERT and RoFormer with batch size 64 and maximum sequence length of 512 for 100k steps. AdamW [Loshchilov and Hutter](#page-13-5) [\[2017\]](#page-13-5) is used as the optimizer with learning rate 1e-5.

### 4.2.3 Results

The MLM loss during pre-training is shown on the left plot of Figure [\(3\)](#page-9-1). Compare to the vanilla BERT, RoFormer experiences faster convergence.

#### <span id="page-8-1"></span>4.3 Fine-tuning on GLUE tasks

Consistent with the previous experiments, we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks in order to evaluate its generalization ability on the downstream NLP tasks.

![](_page_9_Figure_1.jpeg)

![](_page_9_Figure_2.jpeg)

<span id="page-9-1"></span>Figure 3: Evaluation of RoPE in language modeling pre-training. Left: training loss for BERT and RoFormer. Right: training loss for PerFormer with and without RoPE.

### 4.3.1 Experimental Settings

We look at several datasets from GLUE, i.e. MRPC [Dolan and Brockett](#page-13-6) [\[2005\]](#page-13-6), SST-2 [Socher et al.](#page-13-7) [\[2013\]](#page-13-7), QNLI [Rajpurkar et al.](#page-13-8) [\[2016\]](#page-13-8), STS-B [Al-Natsheh](#page-13-9) [\[2017\]](#page-13-9), QQP [Chen et al.](#page-13-10) [\[2018b\]](#page-13-10) and MNLI [Williams et al.](#page-13-11) [\[2018\]](#page-13-11). We use F1-score for MRPC and QQP dataset, spearman correlation for STS-B, and accuracy for the remaining as the evaluation metrics.

### 4.3.2 Implementation details

We use Huggingface Transformers library (Apache License 2.0[\)Wolf et al.](#page-13-12) [\[2020\]](#page-13-12) to fine-tune each of the aforementioned downstream tasks for 3 epochs, with a maximum sequence length of 512, batch size of 32 and learning rates 2,3,4,5e-5. Following [Devlin et al.](#page-11-3) [\[2019\]](#page-11-3), we report the best-averaged results on the validation set.

<span id="page-9-2"></span>Table 2: Comparing RoFormer and BERT by fine tuning on downstream GLEU tasks.

| Model                    | MRPC | SST-2 | QNLI | STS-B | QQP  | MNLI(m/mm) |
|--------------------------|------|-------|------|-------|------|------------|
| BERTDevlin et al. [2019] | 88.9 | 93.5  | 90.5 | 85.8  | 71.2 | 84.6/83.4  |
| RoFormer                 | 89.5 | 90.7  | 88.0 | 87.0  | 86.4 | 80.2/79.8  |

### 4.3.3 Results

The evaluation results of the fine-tuning tasks are reported in Table [\(2\)](#page-9-2). As can be seen, RoFormer can significantly outperform BERT in three out of six datasets, and the improvements are considerable.

### <span id="page-9-0"></span>4.4 Performer with RoPE

Performer [Choromanski et al.](#page-12-18) [\[2020\]](#page-12-18) introduces an alternative attention mechanism, linear attention, which is designed to avoid quadratic computation cost that scales with input sequence length. As discussed in Section [\(3.3\)](#page-4-0), the proposed RoPE can be easily implemented in the PerFormer model to realize the relative position encoding while keeping its linearly scaled complexity in self-attention. We demonstrate its performance with the pre-training task of language modeling.

#### 4.4.1 Implementation details

We carry out tests on the Enwik8 dataset [Mahoney](#page-13-13) [\[2006\]](#page-13-13), which is from English Wikipedia that includes markup, special characters and text in other languages in addition to English text. We incorporate RoPE into the 12 layer char-based PerFormer with 768 dimensions and 12 heads[2](#page-9-3) . To better illustrate the efficacy of RoPE, we report the loss curves of the pre-training process with and without RoPE under the same settings, i.e., learning rate 1e-4, batch size 128 and a fixed maximum sequence length of 1024, etc.

<span id="page-9-3"></span>For this experiment, we adopt code (MIT License) from <https://github.com/lucidrains/performer-pytorch>

### 4.4.2 Results

As shown on the right plot of Figure [\(3\)](#page-9-1), substituting RoPE into Performer leads to rapid convergence and lower loss under the same amount of training steps. These improvements, in addition to the linear complexity, make Performer more attractive.

### <span id="page-10-0"></span>4.5 Evaluation on Chinese Data

In addition to experiments on English data, we show additional results on Chinese data. To validate the performance of RoFormer on long texts, we conduct experiments on long documents whose length exceeds 512 characters.

### 4.5.1 Implementation

In these experiments, we carried out some modifications on WoBERT [Su](#page-13-14) [\[2020\]](#page-13-14) by replacing the absolute position embedding with our proposed RoPE. As a cross-comparison with other pre-trained Transformer-based models in Chinese, i.e. BERT [Devlin et al.](#page-11-3) [\[2019\]](#page-11-3), WoBERT [Su](#page-13-14) [\[2020\]](#page-13-14), and NEZHA [Wei et al.](#page-13-15) [\[2019\]](#page-13-15), we tabulate their tokenization level and position embedding information in Table [\(3\)](#page-10-1).

<span id="page-10-1"></span>Table 3: Cross-comparison between our RoFormer and other pre-trained models on Chinese data. 'abs' and 'rel' annotates absolute position embedding and relative position embedding, respectively.

| Model              | BERTDevlin et al. [2019] | WoBERTSu [2020] | NEZHAWei et al. [2019] | RoFormer |
|--------------------|--------------------------|-----------------|------------------------|----------|
| Tokenization level | char                     | word            | char                   | word     |
| Position embedding | abs.                     | abs.            | rel.                   | RoPE     |

# 4.5.2 Pre-training

We pre-train RoFormer on approximately 34GB of data collected from Chinese Wikipedia, news and forums. The pre-training is carried out in multiple stages with changing batch size and maximum input sequence length in order to adapt the model to various scenarios. As shown in Table [\(4\)](#page-10-2), the accuracy of RoFormer elevates with an increasing upper bound of sequence length, which demonstrates the ability of RoFormer in dealing with long texts. We claim that this is the attribute to the excellent generalizability of the proposed RoPE.

<span id="page-10-2"></span>Table 4: Pre-training strategy of RoFormer on Chinese dataset. The training procedure is divided into various consecutive stages. In each stage, we train the model with a specific combination of maximum sequence length and batch size.

| Stage | Max seq length | Batch size | Training steps | Loss | Accuracy |
|-------|----------------|------------|----------------|------|----------|
| 1     | 512            | 256        | 200k           | 1.73 | 65.0%    |
| 2     | 1536           | 256        | 12.5k          | 1.61 | 66.8%    |
| 3     | 256            | 256        | 120k           | 1.75 | 64.6%    |
| 4     | 128            | 512        | 80k            | 1.83 | 63.4%    |
| 5     | 1536           | 256        | 10k            | 1.58 | 67.4%    |
| 6     | 512            | 512        | 30k            | 1.66 | 66.2%    |

#### 4.5.3 Downstream Tasks & Dataset

We choose Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM[\)Xiao et al.](#page-13-16) [\[2019\]](#page-13-16) dataset to illustrate the ability of RoFormer in dealing with long texts, i.e., semantic text matching. CAIL2019-SCM contains 8964 triplets of cases published by the Supreme People's Court of China. The input triplet, denoted as (A, B and C), are fact descriptions of three cases. The task is to predict whether the pair (A, B) is closer than (A, C) under a predefined similarity measure. Note that existing methods mostly cannot perform significantly on CAIL2019-SCM dataset due to the length of documents (i.e., mostly more than 512 characters). We split train, validation and test sets based on the well-known ratio 6:2:2.

### 4.5.4 Results

We apply the pre-trained RoFormer model to CAIL2019-SCM with different input lengths. The model is compared with the pre-trained BERT and WoBERT model on the same pre-training data, as shown in Table [\(5\)](#page-11-7). With short text cut-offs, i.e., 512, the result from RoFormer is comparable to WoBERT and is slightly better than the BERT implementation. However, when increasing the maximum input text length to 1024, RoFormer outperforms WoBERT by an absolute improvement of 1.5%.

<span id="page-11-7"></span>Table 5: Experiment results on CAIL2019-SCM task. Numbers in the first column denote the maximum cut-off sequence length. The results are presented in terms of percent accuracy.

| Model         | Validation | Test   |
|---------------|------------|--------|
| BERT-512      | 64.13%     | 67.77% |
| WoBERT-512    | 64.07%     | 68.10% |
| RoFormer-512  | 64.13%     | 68.29% |
| RoFormer-1024 | 66.07%     | 69.79% |

### 4.5.5 Limitations of the work

Although we provide theoretical groundings as well as promising experimental justifications, our method is limited by following facts:

- Despite the fact that we mathematically format the relative position relations as rotations under 2D sub-spaces, there lacks of thorough explanations on why it converges faster than baseline models that incorporates other position encoding strategies.
- Although we have proved that our model has favourable property of long-term decay for intern-token products, Section [\(3.3\)](#page-4-0), which is similar to the existing position encoding mechanisms, our model shows superior performance on long texts than peer models, we have not come up with a faithful explanation.

Our proposed RoFormer is built upon the Transformer-based infrastructure, which requires hardware resources for pre-training purpose.

# <span id="page-11-6"></span>5 Conclusions

In this work, we proposed a new position embedding method that incorporates explicit relative position dependency in self-attention to enhance the performance of transformer architectures. Our theoretical analysis indicates that relative position can be naturally formulated using vector production in self-attention, with absolution position information being encoded through a rotation matrix. In addition, we mathematically illustrated the advantageous properties of the proposed method when applied to the Transformer. Finally, experiments on both English and Chinese benchmark datasets demonstrate that our method encourages faster convergence in pre-training. The experimental results also show that our proposed RoFormer can achieve better performance on long texts task.

# References

<span id="page-11-0"></span>Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. Convolutional sequence to sequence learning. In *International Conference on Machine Learning*, pages 1243–1252. PMLR, 2017.

<span id="page-11-1"></span>Md. Amirul Islam, Sen Jia, and Neil D. B. Bruce. How much position information do convolutional neural networks encode? *ArXiv*, abs/2001.08248, 2020.

<span id="page-11-2"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, L ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.cc/paper/2017/file/](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

- <span id="page-11-3"></span>J. Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *NAACL-HLT*, 2019.
- <span id="page-11-4"></span>A. Radford, Jeffrey Wu, R. Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

<span id="page-11-5"></span>Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=ByxRM0Ntvr>.

- <span id="page-12-0"></span>Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=H1eA7AEtvS>.
- <span id="page-12-1"></span>Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. ELECTRA: Pre-training text encoders as discriminators rather than generators. In *ICLR*, 2020. URL <https://openreview.net/pdf?id=r1xMH1BtvB>.
- <span id="page-12-2"></span>A. Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018.
- <span id="page-12-3"></span>Ankur P. Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. In *EMNLP*, 2016.
- <span id="page-12-4"></span>Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In *NAACL-HLT*, 2018.
- <span id="page-12-5"></span>Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Noam Shazeer, I. Simon, C. Hawthorne, Andrew M. Dai, M. Hoffman, M. Dinculescu, and D. Eck. Music transformer. *arXiv: Learning*, 2018.
- <span id="page-12-6"></span>Zihang Dai, Z. Yang, Yiming Yang, J. Carbonell, Quoc V. Le, and R. Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In *ACL*, 2019.
- <span id="page-12-7"></span>Z. Yang, Zihang Dai, Yiming Yang, J. Carbonell, R. Salakhutdinov, and Quoc V. Le. Xlnet: Generalized autoregressive pretraining for language understanding. In *NeurIPS*, 2019.
- <span id="page-12-8"></span>Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, W. Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *J. Mach. Learn. Res.*, 21: 140:1–140:67, 2020.
- <span id="page-12-9"></span>Guolin Ke, Di He, and T. Liu. Rethinking positional encoding in language pre-training. *ArXiv*, abs/2006.15595, 2020.
- <span id="page-12-10"></span>Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled attention. *ArXiv*, abs/2006.03654, 2020.
- <span id="page-12-11"></span>Zhiheng Huang, Davis Liang, Peng Xu, and Bing Xiang. Improve transformer models with better relative position embeddings. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 3327–3335, Online, November 2020. Association for Computational Linguistics. doi[:10.18653/v1/2020.findings-emnlp.298.](https://doi.org/10.18653/v1/2020.findings-emnlp.298) URL <https://www.aclweb.org/anthology/2020.findings-emnlp.298>.
- <span id="page-12-12"></span>Xuanqing Liu, Hsiang-Fu Yu, Inderjit S. Dhillon, and Cho-Jui Hsieh. Learning to encode position for transformer with continuous dynamical model. In *Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of *Proceedings of Machine Learning Research*, pages 6327–6335. PMLR, 2020. URL <http://proceedings.mlr.press/v119/liu20n.html>.
- <span id="page-12-13"></span>Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and Roman Garnett, editors, *Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada*, pages 6572–6583, 2018a. URL [https:](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html) [//proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html).
- <span id="page-12-14"></span>Benyou Wang, Donghao Zhao, Christina Lioma, Qiuchi Li, Peng Zhang, and Jakob Grue Simonsen. Encoding word order in complex embeddings. In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=Hke-WTVtwr>.
- <span id="page-12-15"></span>Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In *International Conference on Machine Learning*, pages 5156–5165. PMLR, 2020.
- <span id="page-12-16"></span>Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 3531–3539, 2021.
- <span id="page-12-17"></span>Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. 04 2018.
- <span id="page-12-18"></span>Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, A. Gane, Tamás Sarlós, Peter Hawkins, J. Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with performers. *ArXiv*, abs/2009.14794, 2020.
- <span id="page-12-19"></span>Ondrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, Radu Soricut, Lucia Specia, and Alevs Tamchyna. Findings of the 2014 workshop on statistical machine translation. pages 12–58, 06 2014. doi[:10.3115/v1/W14-3302.](https://doi.org/10.3115/v1/W14-3302)

- <span id="page-13-0"></span>Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. 08 2015.
- <span id="page-13-1"></span>Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. pages 48–53, 01 2019. doi[:10.18653/v1/N19-4009.](https://doi.org/10.18653/v1/N19-4009)
- <span id="page-13-2"></span>Kishore Papineni, Salim Roukos, Todd Ward, and Wei Jing Zhu. Bleu: a method for automatic evaluation of machine translation. 10 2002. doi[:10.3115/1073083.1073135.](https://doi.org/10.3115/1073083.1073135)
- <span id="page-13-3"></span>Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In *arXiv preprint arXiv:1506.06724*, 2015.
- <span id="page-13-4"></span>Wikimedia Foundation. Wikimedia downloads, <https://dumps.wikimedia.org>, 2021.
- <span id="page-13-5"></span>Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. *arXiv e-prints*, art. arXiv:1711.05101, November 2017.
- <span id="page-13-6"></span>William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*, 2005. URL [https://www.aclweb.org/](https://www.aclweb.org/anthology/I05-5002) [anthology/I05-5002](https://www.aclweb.org/anthology/I05-5002).
- <span id="page-13-7"></span>Richard Socher, A. Perelygin, J.Y. Wu, J. Chuang, C.D. Manning, A.Y. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. *EMNLP*, 1631:1631–1642, 01 2013.
- <span id="page-13-8"></span>Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. pages 2383–2392, 01 2016. doi[:10.18653/v1/D16-1264.](https://doi.org/10.18653/v1/D16-1264)
- <span id="page-13-9"></span>Hussein Al-Natsheh. Udl at semeval-2017 task 1: Semantic textual similarity estimation of english sentence pairs using regression model over pairwise features. 08 2017.
- <span id="page-13-10"></span>Z. Chen, H. Zhang, and L. Zhang, X.and Zhao. Quora question pairs., 2018b. URL [https://www.kaggle.com/c/](https://www.kaggle.com/c/quora-question-pairs) [quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs).
- <span id="page-13-11"></span>Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. pages 1112–1122, 01 2018. doi[:10.18653/v1/N18-1101.](https://doi.org/10.18653/v1/N18-1101)
- <span id="page-13-12"></span>Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online, October 2020. Association for Computational Linguistics. URL <https://www.aclweb.org/anthology/2020.emnlp-demos.6>.
- <span id="page-13-13"></span>Matt Mahoney. Large text compression benchmark, <http://www.mattmahoney.net/dc/text.html>, 2006.
- <span id="page-13-14"></span>Jianlin Su. Wobert: Word-based chinese bert model - zhuiyiai. Technical report, 2020. URL [https://github.com/](https://github.com/ZhuiyiTechnology/WoBERT) [ZhuiyiTechnology/WoBERT](https://github.com/ZhuiyiTechnology/WoBERT).
- <span id="page-13-15"></span>Victor Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, and Qun Liu. Nezha: Neural contextualized representation for chinese language understanding. 08 2019.
- <span id="page-13-16"></span>Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Tianyang Zhang, Xianpei Han, Zhen hu, Heng Wang, and Jianfeng Xu. Cail2019-scm: A dataset of similar case matching in legal domain. 11 2019.