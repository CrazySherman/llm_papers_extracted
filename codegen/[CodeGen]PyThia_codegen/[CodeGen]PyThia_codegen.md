# PYTHIA: AI-ASSISTED CODE COMPLETION SYSTEM

Alexey Svyatkovskiy Microsoft One Microsoft Way Redmond, WA 98052 alsvyatk@microsoft.com

Ying Zhao Microsoft One Microsoft Way Redmond, WA 98052

Shengyu Fu Microsoft One Microsoft Way Redmond, WA 98052

Neel Sundaresan Microsoft One Microsoft Way Redmond, WA 98052

# ABSTRACT

In this paper, we propose a novel end-to-end approach for AI-assisted code completion called Pythia. It generates ranked lists of method and API recommendations which can be used by software developers at edit time. The system is currently deployed as part of Intellicode extension in Visual Studio Code IDE. Pythia exploits state-of-the-art large-scale deep learning models trained on code contexts extracted from abstract syntax trees. It is designed to work at a high throughput predicting the best matching code completions on the order of 100 ms.

We describe the architecture of the system, perform comparisons to frequency-based approach and invocation-based Markov Chain language model, and discuss challenges serving Pythia models on lightweight client devices.

The offline evaluation results obtained on 2700 Python open source software GitHub repositories show a top-5 accuracy of 92%, surpassing the baseline models by 20% averaged over classes, for both intra and cross-project settings.

*K*eywords Code completion, neural networks, naturalness of software

# 1 Introduction

In software development through Integrated Development Environments (IDEs) [\[1\]](#page-11-0), code completion is one of the most widely used features. Intelligent code completion [\[2\]](#page-11-1) assists developers by reducing typographic and other common errors, effectively improving developer productivity. These features may include pop-ups when typing, calling methods and APIs on typed classes and objects, querying parameters of functions, variable or function name disambiguation, and program structure completion that use code context to reliably predict following code.

Traditional code completion tools in IDEs typically list out all possible attributes or methods that can be invoked when a user types a "." or an "=" trigger character. However, not ranking the suggested results requires users to scroll through a long alphabetically ordered list, which is often even slower than typing the full name of a method directly. Studies show that developers tend to rely heavily on prefix filtering to reduce the number of choices [\[3\]](#page-11-2).

In this paper, we introduce Pythia – a neural code completion system, trained on code snippets extracted from a large-scale open source code dataset. The fundamental task of the system is to find the most likely method given a code snippet. In other words, given original code snippet C, the vocabulary V , and the set of all possible methods M ⊂ V , we would like to determine:

$$m^* = argmax(P(m|C)), \forall m \in M.$$
(1)

In order to find that method, we construct a model capable of scoring responses and then determining the most likely method.

A large number of intelligent code completion systems for both statically and dynamically typed languages have been proposed in the literature [\[4,](#page-11-3) [5,](#page-11-4) [2,](#page-11-1) [6,](#page-11-5) [7,](#page-11-6) [8\]](#page-11-7). Best Matching Neighbor (BMN) and statistical language models such as n-grams, as well as recurrent neural network (RNN) based approaches leveraging sequential nature of the source code have been particularly effective at creating such systems. The majority of the past approaches did not leverage the long-range sequential nature of the source code or tried to transfer natural language methods without taking advantage of unique opportunities offered by code's abstract syntax trees (AST).

<span id="page-1-1"></span>Figure 1: Code completion system providing suggestions in alphabetical order. Left: "socket" Python library, right: "PyTorch".

The nature of the problem of code completion makes long short-term memory (LSTM) networks [\[9\]](#page-11-8) a promising candidate. Pythia consumes partial ASTs corresponding to code snippets containing member access expressions and module function invocations as an input for model training, aiming to capture semantics carried by distant nodes.

The main contributions of the paper are: (i) we introduce and implement several baselines of code completions systems, including frequency-based and Markov Chain models, (cf. section [2\)](#page-1-0), (ii) we propose and deploy a novel end-to-end code completion system based on LSTM trained on source code snippets (cf. sections [4](#page-3-0) and [5\)](#page-4-0), (iii) we evaluate our model on a dataset of 15.8 million method calls extracted from real-world source code, showing that our best model achieves 92% accuracy, beating simpler baselines (cf. section [7\)](#page-8-0), (iv) we discuss and document practical challenges of training deep neural networks and hyperparameter tuning on high-performance computing clusters, and model deployment on lightweight client devices at high throughput (cf. section [8\)](#page-8-1).

# <span id="page-1-0"></span>2 Baseline code completion systems

The most basic code completion system in an IDE would list out all possible attributes or methods in an alphabetically ordered list. Fig. [1](#page-1-1) shows example completions that such a system could serve. As seen, this approach is capable of yielding accurate results for classes with a relatively small number of members (e.g. "socket" Python library), otherwise requiring a developer to scroll through a long list (e.g. in the case of "PyTorch").

#### 2.1 Frequency models

Instead of ranking methods alphabetically, popularity ranking can be adopted to increase recommendation relevance. The simplest frequency-based model one could think of would order methods belonging to each class based on the occurrence count in the training corpus.

Looking for a stronger baseline, we built an improved frequency model incorporating context information about if-conditions (referred to as "frequency-if" throughout the paper). In what follows, we subdivide all the invocations into two groups based on whether they are inside an if-condition and creating two corresponding popularity lists. As seen in Tab. [1,](#page-2-0) the most frequent method invocations outside if-statements are related to variable or placeholder declarations, setting up a neural network or training routines; inside if-conditions, however, a typical call would be to file input-output manipulation wrappers, context managers (e.g. eager execution mode), or specifying tensor properties (sparse or dense). As a result, the "frequency-if" model would provide more accurate code suggestions inside if-statements increasing overall accuracy.

| Invocation        | Frequency |
|-------------------|-----------|
| tf.float32        | 10508     |
| tf.nn             | 8954      |
| tf.constant       | 7754      |
| tf.train          | 6869      |
| tf.variable_scope | 6330      |
| tf.placeholder    | 5622      |

| Invocation                    | Frequency |  |
|-------------------------------|-----------|--|
| tf.context.in_graph_mode      | 905       |  |
| tf.gfile.Exists               | 640       |  |
| tf.gfile                      | 624       |  |
| tf.ops.Tensor                 | 448       |  |
| tf.context.executing_eagerly  | 322       |  |
| tf.sparse_tensor.SparseTensor | 271       |  |

<span id="page-2-0"></span>Table 1: Frequencies of the most popular method calls from the TensorFlow library. Left: outside if-statements, right: inside if-statements.

| Method                 | Transition probability |  |
|------------------------|------------------------|--|
| os.rename, shutil.move | 0.22                   |  |
| os.write               | 0.15                   |  |
| shutil.copy            | 0.13                   |  |
| os.path.isfile         | 0.11                   |  |
| os.remove              | 0.10                   |  |
| All others methods     | 0.29                   |  |

<span id="page-2-1"></span>Table 2: Transition probabilities to the third state of the three-state Markov chain: os.path.isfile → os.remove → ?.

### 2.2 Invocation-based Markov Chain model

Markov Chain models have demonstrated great strength at modeling stochastic transitions, from uncovering sequential patterns [\[10,](#page-11-9) [11\]](#page-11-10) to modeling decision processes [\[12\]](#page-11-11). When analyzing invocation occurrences in the source code, we discovered that some sequences of invocations are always followed by the same methods.

For example, looking at all the function invocations of "os" and "shutil" Python modules, related to file manipulation, which come after an invocation chain os.path.isfile → os.remove, we have observed that more than 20% of the time it will be "os.rename" or "shutil.move", even though those two modules have over 30 methods related to file manipulation.

Considering following three-state Markov process:

$$\texttt{os.path.isfile} \rightarrow \texttt{os.remove} \rightarrow \texttt{?},$$

where the third state can be any function from "os" or "shutil" modules, we compute the corresponding transition probabilities in Tab. [2.](#page-2-1) Using previous invocations occurrences to predict what invocation will be used next is at the core of our Markov Chain model of code completion:

$$P(m_i|h_i) = P(m_i|m_{i-n}, ..., m_{i-1}),$$
(2)

here, sequence h<sup>i</sup> = m1, m2, ..., mi−<sup>1</sup> denotes a history of previous method invocations in the same document scope for a given class. In an n-th order Markov chain model the probability of next invocation will only depend on the previous n − 1 invocations, which is estimated by counting the number of invocation sequence occurrences in the training corpus.

The abundance of high-quality source code data available on GitHub and the tools allowing to extract and label code snippets are making it possible to train and deploy advanced machine learning models achieving more accurate predictions (cf. section [5\)](#page-4-0).

### 3 Dataset

We collected a dataset to train and evaluate code completion models from open source Python repositories on GitHub. A total of 2700 top-starred (non-fork) Python projects have been selected, containing libraries from a diverse set of domains including scientific computing, machine learning, dataflow programming, and web development, with over 15.8 million method calls. Fig. [2](#page-3-1) shows Python libraries with the most method call occurrences in the dataset.

We split the dataset into development and test sets in the proportion 70-30 on the repository level. The development set is then split at random into training and validation sets in the proportion 80-20. To serve predictions online, the model is retrained using the entire dataset.

![](_page_3_Figure_0.jpeg)

<span id="page-3-1"></span>Figure 2: Number of method calls per library, for top 10 libraries in the training dataset.

# <span id="page-3-0"></span>4 Representing code snippets

Any programming language has an explicit context-free grammar (CFG), which can be used to parse source code into an AST. An AST is a rooted n-ary tree, where each non-leaf node corresponds to a non-terminal in the CFG specifying structural information. Each leaf node corresponds to a syntax token encoding program text.

The Pythia model consumes partial file-level ASTs corresponding to code snippets containing member access expressions and module function invocations as an input for training. Before feeding to LSTM, ASTs are serialized to sequences according to in-order depth-first traversal. We retain up to T lookback tokens preceding each method call when extracting training sequences, where T is a tunable parameter of the model provided in Tab. [3.](#page-7-0)

Further, in order for an AST to be consumable by LSTM, its nodes and tokens have to be mapped to numeric vectors. Like the weights of the LSTM, the look-up tables mapping discrete values of tokens to real vectors can be learned via backpropagation. The Word2Vec approach [\[13\]](#page-11-12) has been shown to work well in Natural Language Processing, and is employed in the Pythia system to represent code snippets as dense low-dimensional vectors.

Each input data file is parsed using the PTVS parser [1](#page-3-2) to extract its AST. The method invocations and associated metadata, including invocation spans, serialized sequences of syntax nodes and tokens from the AST, and the runtime types of method call receiver tokens are extracted. The data processing workflow is summarized in Fig. [3.](#page-4-1)

Syntax node and token names are mapped to integers from 1 to V to obtain token indices. Infrequent tokens are removed to reduce vocabulary size. During training, the out-of-vocabulary (OoV) tokens in a code snippet are mapped to integers greater than vocabulary size V , in a way that the occurrences of the same OoV token in the training code snippet would be represented by the same integer. Each training sample is an AST serialized into sequence terminated with the "." end-of-sequence character. A name of the method call token serves as a label and is one-hot encoded. We train a word embedding on the corpus of the source code repositories to map token indices to low-dimensional dense vectors preserving semantic relationships in the code.

#### 4.1 Leveraging type information

Python is a dynamically typed language. While PEP 484[2](#page-3-3) introduces type annotations for various common types starting Python 3.6, most of the libraries have not yet adopted it. As such, Python interpreter does variable type checking only as code runs. We infer types of the method call receiver tokens and local variables at runtime based on static analysis of the object usage patterns and include this information into training sequences.

<span id="page-3-2"></span><sup>1</sup> https://github.com/Microsoft/PTVS

<span id="page-3-3"></span>https://www.python.org/dev/peps/pep-0484

![](_page_4_Figure_0.jpeg)

<span id="page-4-1"></span>Figure 3: Pythia workflow: from raw source code files to serving recommendations online.

In Python programming language, a user has freedom of aliasing imported modules. For instance, a commonly used "import numpy as np" may well be "import numpy as an\_unusual\_alias". Embedding vectors for "np" and "numpy" tokens are nearby in embedding space. However, an infrequent alias like "an\_unusual\_alias" may be an out-of-vocabulary token and not represented in the embedding. Including type information during training means an embedding vector corresponding to "numpy" token would be used in all of the cases described above.

Variable names are nouns chosen by a developer, which may differ from one code implementation to another. Keeping them in the original format increases the vocabulary size, making a code completion system dependent on spelling of a variable name. To avoid this, variable names are normalized according to var :< variabletype > convention.

# <span id="page-4-0"></span>5 Neural code completion model

Recurrent Neural Networks are a family of neural networks capable of processing sequential data of arbitrary length. Long short-term memory networks [\[9\]](#page-11-8), a particularly successful form of RNNs, trained on large datasets can obtain remarkable performance results across a wide variety of domains – from image captioning [\[14\]](#page-11-13), to sentiment analysis and machine translation [\[15,](#page-11-14) [16\]](#page-11-15). An LSTM unit uses no activation function within its recurrent components, controlling information flow using gates implemented using the logistic function. Thus, the stored values are not iteratively expanded or squeezed over time, and the gradient does not tend to explode or vanish when trained.

The task of method completion is to predict a token m<sup>∗</sup> , conditional on a sequence of syntax tokens ct, t = 0...T, corresponding to the terminal nodes of the AST of a code snippet C, plus the special end-of-sequence token ".". This problem is a natural fit for sequence learning with LSTM:

$$x_t = Lc_t, (3)$$

$$h_t = f(x_t, h_{t-1}), \tag{4}$$

$$P(m|C) = y_t = softmax(Wh_t + b), (5)$$

$$m^* = argmax(P(m|C)). (6)$$

Here, the matrix L ∈ R<sup>d</sup>x×|<sup>V</sup> <sup>|</sup> is the word embedding matrix, d<sup>x</sup> is the word embedding dimension and |V | is the size of the vocabulary. In case of Pythia, function f(., .) represents a stacked LSTM taking the current input and the previous hidden state and producing the hidden state at the next temporal step. W ∈ R<sup>|</sup><sup>V</sup> |×d<sup>h</sup> and b ∈ R<sup>|</sup><sup>V</sup> <sup>|</sup> are the output projection matrix and the bias. The d<sup>h</sup> is the size of the hidden state of LSTM.

Inspired by [\[17\]](#page-11-16), we are reusing the input word embedding matrix as the output classification matrix as shown in Fig. [4,](#page-5-0) which allows to remove the large fully connected layer and significantly reduce the number of trainable parameters and the model size on disk. More specifically, we introduce a projection matrix A = (a)ij ∈ R<sup>d</sup>h×d<sup>x</sup> initialized according to a random uniform distribution. Given an LSTM encoded snippet and a hidden state at the last temporal

![](_page_5_Picture_0.jpeg)

Figure 4: Architecture of the neural network deployed in the Pythia code completion system.

<span id="page-5-0"></span>step h<sup>T</sup> ∈ Rd<sup>h</sup> , by multiplying the two together we obtain the predicted embedding vector L pred = (l pred)<sup>j</sup> ∈ Rd<sup>x</sup> as:

$$l_j^{pred} = \sum_i h_{Ti} a_{ij}. \tag{7}$$

Subsequently, the unnormalized predictions of the neural network are obtained as:

$$y_k = \sum_j l_{kj} l_j^{pred} + b_k \tag{8}$$

where bk, k = 0...|V | − 1 is the bias vector initialized to zeros before backpropagation.

### 6 Model training

Neural networks are trained iteratively, making multiple passes over an entire dataset before converging to a minimum. Backpropagation through time (BPTT) is a gradient-based neural network training algorithm we apply to train the LSTMs statefully.

Training deep neural networks is a computationally intensive problem that requires the engagement of high-performance computing clusters. Pythia uses a data-parallel distributed training algorithm with Adam optimizer, keeping a copy of an entire neural model on each worker, processing different mini-batches of the training dataset in parallel lockstep.

The offline training module of the Pythia system is implemented as a Python library integrating TensorFlow and CUDA-aware MPI for distributed training. The software stack makes use of CUDA 9, GPU accelerated deep learning primitives from CuDNN 7, and TensorFlow 1.10. We use BatchAI [3](#page-5-1) – an Azure cloud service providing on-demand GPU clusters with Kubernetes resource manager – for model training and hyperparameter optimization. The online module is implemented in C#, making use of ML.NET library and ONNX data format [4](#page-5-2) .

#### 6.1 Batching and Learning rate schedule

The number of nodes in a Python file-level abstract syntax tree ranges from O(10<sup>2</sup> ) to O(10<sup>4</sup> ). To leverage long-range dependencies in the source code we consider sequence lengths in the range of 100 − 1000. To overcome the gradient

<span id="page-5-1"></span><sup>3</sup> https://docs.microsoft.com/en-us/azure/batch-ai/overview

<span id="page-5-2"></span><sup>4</sup> https://github.com/onnx/onnx

![](_page_6_Figure_0.jpeg)

<span id="page-6-0"></span>Figure 5: Snapshot of the training buffer.

vanishing problem for long sequences, we approximate the computation of the gradient of the loss with respect to the model parameters by truncated backpropagation through time [\[18\]](#page-11-17).

Efficient batching is important to fully utilize parallelism of modern GPUs. In general, training sequences will have variable lengths, with a majority having the maximum length of T, as shown in Fig. [5.](#page-6-0) Training samples are presorted and split into three buckets based on their lengths. Within each bucket, sequences are padded to the maximum length using the special padding token, which is then excluded from loss calculation by means of masking. A training buffer to maintain sequences belonging to d<sup>b</sup> distinct ASTs is allocated. At every training step, the first TRNN timesteps of the buffer are fed as the next batch. The buffer is then shifted by TRNN . Every time a sequence is finished in the buffer, a new set of sequences is loaded, and the LSTM internal states are reset for training.

The learning rate controlling the magnitude of the weight update during gradient optimization is lowered upon completion of each epoch according to the exponential decay. In a distributed regime, the learning rate is scaled up during the first few epochs to facilitate reliable model convergence. Closely following [\[19\]](#page-11-18), we linearly scale the learning rate up proportionally to the number of workers Nworker during the first 4 epochs ("warm-up" period):

$$\lambda_0(N_{worker}) = \lambda_0 \cdot \gamma^i \cdot \frac{N_{worker}}{\alpha} \tag{9}$$

here, λ<sup>0</sup> is the base learning rate, γ is the learning rate decay constant, i is the epoch number, and parameter α is the scaling fraction controlling the learning rate at the end of the warm-up period, α = 4 was found to work best via hyperparameter search.

Fig. [6](#page-7-1) shows the reduction in training time due to distributed training and the effect of learning rate schedule.

#### 6.2 Hyperparameters

Overall, the model architecture, training procedure, and data normalization produce a large number of hyperparameters that must be tuned to maximize predictive performance. These hyperparameters include numerical values such as the learning rate and number of LSTM layers, dimension of embedding space, but also abstract categorical variables such as the precise model architecture or the normalization algorithm. These parameters are summarized in Tab. [3.](#page-7-0)

![](_page_7_Figure_0.jpeg)

<span id="page-7-1"></span>Figure 6: Accuracy on the validation set for various training regimes. Left: serial and distributed training with 8 worker GPUs; right: distributed training with 8 GPUs for various learning rate schedules. The horizontal green line indicates the target model accuracy of 92%.

| Hyperparameter            | Explanation                                     | Best value                |
|---------------------------|-------------------------------------------------|---------------------------|
| λ0                        | Base learning rate                              | 0.002                     |
| γ                         | Learning rate decay per epoch                   | 0.97                      |
| N                         | Number of recurrent neural network layers       | 2                         |
| dh                        | Number of hidden units in LSTM, per layer       | 100                       |
| T                         | Number of lookback tokens,                      | 1000                      |
|                           | timesteps through which backpropagation is run  |                           |
| TRNN                      | Number of timesteps through which               | 100                       |
|                           | backpropagation is run                          |                           |
| RNN type                  | Type of RNN                                     | LSTM                      |
| db                        | Batch size                                      | 256                       |
| Loss function             | Type of loss function                           | Categorical cross-entropy |
| dx                        | Embedded vector dimension                       | 150                       |
| Optimizer                 | Stochastic optimization scheme                  | Adam                      |
| Dropout                   | Dropout keep probability                        | 0.8                       |
| L2 Regularization         | Weight regularization of all layers             | 10                        |
| Clip norm                 | Maximum norm of gradients                       | 10                        |
| Token frequency threshold | Minimum frequency of syntax token in the corpus | 500                       |
|                           | for inclusion in vocabulary                     |                           |

<span id="page-7-0"></span>Table 3: Hyperparameters to be optimized together with explanations and well-performing values.

Throughout this work, the "best" model is determined by hyperparameter tuning. This is done via random search in the respective hyperparameter space of each method, i.e. by training a number of models with random hyperparameters on the training set and choosing the one with the highest performance on validation set.

#### 6.3 Tuning neural network architecture

Architecture of the neural network is a hyperparameter. Besides LSTM, we consider gated recurrent units (GRU), LSTM with attention mechanism for temporal data [\[20\]](#page-12-0), and variations of classification layer following the LSTM.

Selecting the best model for serving online is a trade-off between accuracy and model size. Tab. [4](#page-8-2) shows validation level accuracy for various neural model architectures and associated sizes on disk. As seen, removing a large fully connected classification layer in favor of predicted embedding reduces the number of trainable parameters and the model size on disk by 25%. The best top-5 accuracy is achieved when the attention mechanism is employed, however, this model is 8% larger and has slower inference speeds. Consequently, we have chosen the LSTM with predicted embedding for deployment.

| Model architecture       | Top-5 accuracy | Model size (quantized size), MB |
|--------------------------|----------------|---------------------------------|
| LSTM+fully connected     | 0.91           | 202 (51)                        |
| GRU+predicted embedding  | 0.91           | 152 (38)                        |
| LSTM+predicted embedding | 0.92           | 152 (38)                        |
| LSTM+attention           | 0.93           | 164 (41)                        |

<span id="page-8-2"></span>Table 4: Accuracy on the validation set for various neural model architectures and associated model sizes.

| Method       | Top-1 accuracy | Top-5 accuracy | MRR   |
|--------------|----------------|----------------|-------|
| Alphabetic   | 0.36           | 0.47           | 0.372 |
| Frequency    | 0.38           | 0.64           | 0.495 |
| Frequency-if | 0.40           | 0.67           | 0.521 |
| Markov Chain | 0.58           | 0.83           | 0.704 |
| Pythia       | 0.71           | 0.92           | 0.814 |

<span id="page-8-3"></span>Table 5: Accuracy and mean reciprocal ranks on the test set for the Pythia neural model and various baselines.

# <span id="page-8-0"></span>7 Evaluation

#### 7.1 Evaluation Metrics

Top-k accuracy and the mean reciprocal rank (MRR) [\[21\]](#page-12-1) are used to measure the quality of recommendations, defined as:

$$Acc(k) = \frac{N_{top-k}}{Q}, (10)$$

$$MRR = \frac{1}{Q} \cdot \sum_{i=1}^{Q} \frac{1}{rank_i}, \tag{11}$$

where Ntop−<sup>k</sup> denotes the number of relevant recommendation in top k suggestions, Q represents the total number of test data samples and rank<sup>i</sup> is the prediction rank of a recommendation.

Accuracy in top-1 tells how often the top recommendation is correct, while top-5 accuracy provides an idea of how often the top five recommendation list contains the suggestion a user is looking for. The MRR captures the overall rank of the result, thus providing information of how far outside of the list of top suggestions the model prediction was. MRR values closer to one indicate overall smaller recommendation ranks, corresponding to a better performing model.

#### 7.2 Evaluation results

Tab. [5](#page-8-3) shows the performance comparison of the Pythia neural model and various simpler baselines, including basic alphabetic ordering, frequency models, and invocation-based Markov Chain model. As seen, our model significantly outperforms all the baselines, especially for the accuracy of top-1 suggestions. Tab. [6](#page-9-0) summarizes the performance of the Pythia system for 10 most popular completion classes. Fig. [7](#page-9-1) illustrates a performance improvement achieved by the Pythia neural model as compared to the Markov Chain baseline on the test set. As seen from the histogram, Pythia yields over 50% accuracy improvement for nearly 6000 completion classes. Markov Chain model relies on the type inference, which results in a lower coverage than the Pythia neural model. The classes not covered by the Markov Chain model are included in the overflow bin.

# <span id="page-8-1"></span>8 Model deployment

The Pythia code completion system is currently deployed as a part of Intellicode extension [5](#page-8-4) in Visual Studio Code IDE. The main challenges in designing and implementing an online system capable of serving models on lightweight client devices are the prediction latency and memory footprint. We apply neural network quantization into an 8-bit unsigned integer numeric format to reduce memory footprint and model size on disk.

<span id="page-8-4"></span><sup>5</sup> https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode

| Class name | Top-5 accuracy, MC | Top-5 accuracy, Pythia | $\frac{\Delta Acc}{Acc}$ [%] | $N_{recommend}$ |
|------------|--------------------|------------------------|------------------------------|-----------------|
| os         | 0.863              | 0.950                  | 10.0                         | 125673          |
| numpy      | 0.575              | 0.697                  | 21.2                         | 113227          |
| list       | 0.978              | 0.989                  | 1.1                          | 79406           |
| str        | 0.974              | 0.988                  | 1.4                          | 77507           |
| os.path    | 0.895              | 0.957                  | 6.8                          | 75270           |
| sys        | 0.821              | 0.959                  | 16.8                         | 51231           |
| WX         | 0.272              | 0.533                  | 95.9                         | 47821           |
| logging    | 0.846              | 0.914                  | 8.01                         | 40871           |
| time       | 0.951              | 0.980                  | 3.1                          | 27874           |
| tensorflow | 0.511              | 0.754                  | 47.6                         | 16620           |

<span id="page-9-0"></span>Table 6: Performance summary on the test set for the Pythia neural model and the Markov Chain baseline. Relative percentage accuracy change is defined as:  $\frac{\Delta Acc}{Acc} = 100 \cdot (Acc_{Pythia} - Acc_{MC})/Acc_{MC} \ [\%]$ 

![](_page_9_Figure_2.jpeg)

<span id="page-9-1"></span>Figure 7: Number of completion classes with a given relative percentage accuracy difference for the Pythia neural model as compared to the invocation-based Markov Chain baseline.  $\frac{\Delta Acc}{Acc} = 100 \cdot (Acc_{Pythia} - Acc_{MC})/Acc_{MC} \ [\%]$ 

#### 8.1 Neural network quantization

Neural network quantization is a process of reducing the number of bits to store weights. The numerical format used during training of the Pythia neural model is IEEE 754 32-bit float. Quantization is performed layer-by-layer, extracting minimum and maximum values of weights and activations in the floating point format, zero shifting, and scaling. Given a weight matrix  $W=w_{ij}$  for a layer, the quantized weight matrix  $W^q$  is obtained as:

$$\beta = \frac{max(W) - min(W)}{2^8}, \quad w_{ij}^q = \frac{w_{ij} - min(W)}{\beta}.$$
 (12)

In the case of Pythia, post-training quantization into 8-bit integer results in model size reduction to quarter the size – from 152 MB to 38 MB – reducing top-5 accuracy from 92% to an acceptable 89%.

An example online recommendation served by the Pythia neural model is shown in Fig. 8. As seen, Pythia accurately suggests the most likely method call should involve an optimizer, with two out of the top five recommendations being Adam and SGD optimizers. Other suggestions include "Saver" – a method adding ops to save and restore variables to

<span id="page-10-0"></span>Figure 8: Example code completion served by Pythia.

and from checkpoints, "exponential\_decay" – controlling learning rate decay, and "Feature" – a data wrapper specific to TensorFlow, all of which are likely to be called to before creating a session and running the training ops.

# 9 Conclusions

We have introduced a novel end-to-end approach for AI-assisted code completion called Pythia, generating ranked lists of method and API recommendations that can be used by software developers. We have deployed the system as part of Intellicode extension in Visual Studio Code IDE. Pythia makes use of long short-term memory networks trained on long-range code contexts extracted from abstract syntax trees, capturing semantics carried by distant nodes. Evaluation results on a large dataset of 15.8 million method calls extracted from real-world source code showed that our best model achieves 92% top-5 accuracy, beating simpler baselines.

We have overcome and documented several practical challenges of training deep neural networks and hyperparameter tuning on high-performance computing clusters, and model deployment on lightweight client devices to predict the best matching code completions at edit time.

Besides Python, Intellicode extension is providing AI-assisted code completions for a variety of programming languages including C#, Java, C++, and XAML based on our Markov Chain model. In the future, advanced deep learning approaches will be explored in application to programming languages other than Python, and for more sophisticated code completion scenarios.

# 10 Acknowledgement

We are thankful to Miltiadis Allamanis of Microsoft Research for valuable discussions on the neural network architectures and reduction of number of trainable parameters. We also thank Microsoft AI Frameworks and ML.NET teams for helping deploying the model, as well as to Christian Bird of Microsoft Research for reading the manuscript.

# References

- <span id="page-11-0"></span>[1] G. C. Murphy, M. Kersten, and L. Findlater. How are java software developers using the eclipse ide? *IEEE*, 23(4):76–83, 2006.
- <span id="page-11-1"></span>[2] Sebastian Proksch, Johannes Lerch, and Mira Mezini. Intelligent code completion with bayesian networks. *ACM Transactions on Software Engineering and Methodology (TOSEM)*, 25(1):3, 2015.
- <span id="page-11-2"></span>[3] M. Maraoiu, L. Church, and A. Blackwell. An empirical investigation of code completion usage by professional software developers. 2015.
- <span id="page-11-3"></span>[4] D'Souza Andrea Renika, Di Yang, and Cristina V Lopes. Collective intelligence for smarter api recommendations in python. In *Source Code Analysis and Manipulation (SCAM), 2016 IEEE 16th International Working Conference on*, pages 51–60. IEEE, 2016.
- <span id="page-11-4"></span>[5] Veselin Raychev, Martin Vechev, and Eran Yahav. Code completion with statistical language models. In *ACM SIGPLAN Notices*, volume 49, pages 419–428. ACM, 2014.
- <span id="page-11-5"></span>[6] Marcel Bruch, Martin Monperrus, and Mira Mezini. Learning from examples to improve code completion systems. In *Proceedings of the the 7th joint meeting of the European software engineering conference and the ACM SIGSOFT symposium on The foundations of software engineering*, pages 213–222. ACM, 2009.
- <span id="page-11-6"></span>[7] A. Alnusair, T. Zhao, and E. Bodden. Effective api navigation and reuse. *IRI*, pages 7–12, 2010.
- <span id="page-11-7"></span>[8] T. Gvero, V. Kuncak, I. Kuraj, and R. Piskac. Complete completion using types and weights. In *PLDI'13*. ACM, 2013.
- <span id="page-11-8"></span>[9] Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. Learning to forget: Continual prediction with lstm. 1999.
- <span id="page-11-9"></span>[10] A. Zimdars, D. M. Chickering, and C. Meek. Using temporal data for making recommendations. In *Proceedings of 7th Conference on Uncertainty in Artificial Intelligence*, pages 580–588. UAI, 2001.
- <span id="page-11-10"></span>[11] B. Mobasher, H. Dai, T. Luo, and M. Nakagawa. Using sequential and non-sequential patterns in predictive web usage mining tasks. In *International Conference on Data Mining*. IEEE, 2002.
- <span id="page-11-11"></span>[12] R. I. Brafman G. Shani and D. Heckerman. An mdp-based recommender system. In *Proceedings of 8th Conference on Uncertainty in Artificial Intelligence*. UAI, 2002.
- <span id="page-11-12"></span>[13] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, *Advances in Neural Information Processing Systems 26*, pages 3111–3119. Curran Associates, Inc., 2013.
- <span id="page-11-13"></span>[14] Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan Yuille. Deep captioning with multimodal recurrent neural networks (m-rnn). *ICLR*, 2015.
- <span id="page-11-14"></span>[15] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 1631–1642, Stroudsburg, PA, October 2013. Association for Computational Linguistics.
- <span id="page-11-15"></span>[16] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, *Advances in Neural Information Processing Systems 27*, pages 3104–3112. Curran Associates, Inc., 2014.
- <span id="page-11-16"></span>[17] Hakan Inan, Khashayar Khosravi, and Richard Socher. Tying word vectors and word classifiers: A loss framework for language modeling. *CoRR*, abs/1611.01462, 2016.
- <span id="page-11-17"></span>[18] Alex Graves. Generating sequences with recurrent neural networks. *CoRR*, abs/1308.0850, 2013.
- <span id="page-11-18"></span>[19] Priya Goyal, Piotr Dollár, Ross B. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: training imagenet in 1 hour. *CoRR*, abs/1706.02677, 2017.

- <span id="page-12-0"></span>[20] Colin Raffel and Daniel P. W. Ellis. Feed-forward networks with attention can solve some long-term memory problems. *CoRR*, abs/1512.08756, 2015.
- <span id="page-12-1"></span>[21] Dragomir R. Radev, Hong Qi, Harris Wu, and Weiguo Fan. Evaluating web-based question answering systems. *Language Resources and Evaluation*, 2002.