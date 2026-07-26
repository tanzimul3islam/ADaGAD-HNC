Data Mining and Knowledge Discovery (2025) 39:44
https://doi.org/10.1007/s10618-025-01115-5
Towards automated self-supervised learning for truly
unsupervised graph anomaly detection
Zhong Li1 · Yuhang Wang1 · Matthijs van Leeuwen1
Received: 22 May 2024 / Accepted: 8 June 2025 / Published online: 5 July 2025
© The Author(s) 2025
Abstract
Self-supervised learning (SSL) is an emerging paradigm that exploits supervisory
signals generated from the data itself, and many recent studies have leveraged SSL
to conduct graph anomaly detection. However, we empirically found that three
important factors can substantially impact detection performance across datasets:
(1) the specific SSL strategy employed; (2) the tuning of the strategy’s hyperpa-
rameters; and (3) the allocation of combination weights when using multiple strate-
gies. Most SSL-based graph anomaly detection methods circumvent these issues by
arbitrarily or selectively (i.e., guided by label information) choosing SSL strategies,
hyperparameter settings, and combination weights. While an arbitrary choice may
lead to subpar performance, using label information in an unsupervised setting is
label information leakage and leads to severe overestimation of a method’s perfor-
mance. Leakage has been criticized as “one of the top ten data mining mistakes",
yet many recent studies on SSL-based graph anomaly detection have been using la-
bel information to select hyperparameters. To mitigate this issue, we propose to use
an internal evaluation strategy (with theoretical analysis) to select hyperparameters
in SSL for unsupervised anomaly detection. We perform extensive experiments us-
ing 10 recent SSL-based graph anomaly detection algorithms on various benchmark
datasets, demonstrating both the prior issues with hyperparameter selection and the
effectiveness of our proposed strategy.
Keywords Graph anomaly detection · Self-supervised learning · Automated
machine learning · Graph neural networks · Label leakage
Zhong Li
z.li@liacs.leidenuniv.nl
Yuhang Wang
yuhang.wang@umail.leidenuniv.nl
Matthijs van Leeuwen
m.van.leeuwen@liacs.leidenuniv.nl
1 Leiden Institute of Advanced Computer Science (LIACS), Leiden University, Leiden, The
Netherlands
1 3

44 Page 2 of 43 Z. Li et al.
1 Introduction
Graph anomaly detection (GAD) refers to the tasks of identifying anomalous graph
objects—such as nodes, edges or sub-graphs—in an individual graph (Akoglu et al.
2015; Ma et al. 2021), or identifying anomalous graphs from a set of graphs (Ma et al.
2022; Li et al. 2024a). GAD has numerous successful applications, e.g., in finance
fraud detection (Motie and Raahemi 2023), fake news detection (Xu et al. 2022a),
system fault diagnosis (Li et al. 2024b), and network intrusion detection (Garcia-Teo-
doro et al. 2009). In this paper, we focus on unsupervised node anomaly detection on
static attributed graphs, namely identifying which nodes in a static attributed graph
are anomalous. Recently, Graph Neural Networks (GNNs) have become prevalent in
detecting node anomalies in graphs and have shown promising performance (Kim
et al. 2022). Specifically, GNNs can learn an embedding for each node by consider-
ing both the node attributes and the graph topological information, enabling them to
capture and exploit complex patterns for anomaly detection.
Like with other neural networks, the high performance of GNNs is typically
achieved at the cost of a substantial volume of labeled data. However, the process of
labeling graphs is often a laborious and time-consuming effort, necessitating domain-
specific expertise. For these reasons, GAD is preferably tackled in an unsupervised
manner, without relying on any ground-truth labels. Self-supervised learning (SSL)
has emerged as a promising unsupervised learning technique on graphs (Liu et al.
2022c), and recent studies have shown its usefulness for node anomaly detection
(Fan et al. 2020; Zheng et al. 2021; Jin et al. 2021a; Liu et al. 2021; Yuan et al. 2021;
Xu et al. 2022b; Liu et al. 2022a; Chen et al. 2022).
Graph SSL can be roughly divided into generative, contrastive, and predictive
methods (Wu et al. 2021). First, generative methods such as DOMINANT (Ding
et al. 2019), GUIDE (Yuan et al. 2021), and AnomalyDAE (Fan et al. 2020) aim to
detect graph anomalies by reconstructing (‘generating’) the adjacency matrix and/or
the node attribute matrix. Next, contrastive methods such as CoLA (Liu et al. 2021),
ANEMONE (Jin et al. 2021a), GRADATE (Duan et al. 2023), and Sub-CR (Zhang
et al. 2022) train a graph encoder to pull positive pairs closer while pushing negative
pairs away in the embedding space. The nodes with relatively large contrastive loss
values are deemed anomalies. Finally, predictive methods such as SL-GAD (Zheng
et al. 2021) try to predict node properties using its local context (e.g., a subgraph),
and nodes with large prediction errors are considered anomalies.
Contrastive learning is arguably the most successful SSL strategy for graphs (Xie
et al. 2022). Most contrastive graph learning methods consist of two main modules:
(1) a data augmentation module that generates augmented data by operations such
as edge dropping, node attribute masking, node addition, subgraph sampling, and/
or graph diffusion. The augmented view of an instance is generally regarded as a
positive pair with the original instance; and (2) a contrastive learning module that
contrasts positive pairs (and often involves negative pairs) at different levels, such as
node-node contrast, node-subgraph contrast, and subgraph-subgraph contrast.
Although SSL-based graph anomaly detection has been successful, using it in
practice is often not straightforward. The most important reason for this is that most
methods require a large number of choices to be made, leading to three challenges:
1 3

Towards automated self-supervised learning for truly unsupervised… Page 3 of 43 44
C1. How should we select appropriate data augmentation functions?
C2. How should we choose appropriate values for hyperparameters (HPs) of a
given augmentation function? (e.g., subgraph size in a subgraph sampling function,
or the proportion of edges to drop in an edge dropping function)
C3. How to combine the contrast losses at different levels? (i.e., how to set their
combination weights?)
Further, a recent study (Zheng et al. 2021) shows that combining multiple SSL strate-
gies for GAD can achieve better performance than using a single SSL strategy. This
leads to the fourth challenge:
C4. How should we combine different SSL strategies? (i.e., how to set the com-
bination weights of different SSL loss functions?)
Previous work (Chen et al. 2020a; You et al. 2020; Yoo et al. 2023) showed that
the choice of SSL strategies and hyperparameter values can strongly impact perfor-
mance. In a supervised setting, these choices can be systematically and rigorously
made by using separate labeled data for validation. In an unsupervised setting such
as anomaly detection, however, one should assume that no labels are available even
for hyperparameter tuning. In our extensive literature study, we found that existing
SSL-based GAD methods typically either (1) arbitrarily choose settings or (2) do use
labeled data, corroborating the findings in Yoo et al. (2023).
In the former case, practitioners typically heuristically select an augmentation func-
tion (C1) and fix its associated HPs (C2) across all datasets, and set the combination
weights all equal to 1 or other fixed values (for C3 and C4). Although this approach
is not flawed, it is likely to result in suboptimal detection performance: graphs from
different domains usually have different properties (Zhao et al. 2022a), implying that
the optimal SSL strategy is in general data-dependent (Chen et al. 2020a; You et al.
2020). Therefore, utilizing a unified and pre-fixed combination weights and/or HPs in
SSL strategies for all graphs can result in sub-optimal performance.
In the latter case, practitioners pick the optimal combination weights and other
hyperparameter values following a ‘hyperparameters sensitivity analysis’ using
labeled data. By using ground-truth labels on test data to check model performance
with different hyperparameter values and using that to select the best model, however,
label leakage occurs. That is, information about the target of a data mining problem is
used for learning/selecting model, while this information should not be legitimately
accessible for learning purposes (Nisbet et al. 2009; Kaufman et al. 2012). Specifi-
cally, label information should never be used (whether implicitly or explicitly) in
an unsupervised learning scenario. As shown in Fig. 1, label leakage leads to huge
overestimation of the model’s performance, which is also corroborated in Liu et al.
(2022b) by comparing the max and average performance with different hyperparam-
eter configurations (cf. Appendix C for more details).
The reason that hyperparameter values are often chosen either arbitrarily or using
label information is probably that it is challenging to construct an internal evalua-
tion strategy for anomaly detection without using any labels. There have been some
research efforts aimed at automating graph SSL though. For instance, JOAO (You
et al. 2021) aims to automatically combine several predefined graph augmentations
via learning a sampling distribution, where the augmentations themselves are not
learnable. Meanwhile, AD-GCL (Suresh et al. 2021) uses learnable edge dropping
1 3

44 Page 4 of 43 Z. Li et al.
Fig. 1 Large performance variations (here measured by AUC) over different hyperparameter configura-
tions for ANEMONE (Jin et al. 2021a) on various benchmark datasets. Using labeled data and only
reporting the best possible performance leads to severe overestimation of model performance. For
instance, the green squares on Cora, CiteSeer, and PubMed are reported by Jin et al. (2021a) (the other
datasets were not used). Similar results are observed for other algorithms (see Appendix B for details).
The red triangles represent the results obtained by our internal evaluation strategy, showing its poten-
tial for automating truly unsupervised anomaly detection
augmentation and AutoGCL (Yin et al. 2022) proposes a learnable graph view gen-
erator that learns a probability distribution over node-level augmentations, which
can well preserve the semantic labels of graphs for graph-level tasks. However, all
these automated graph augmentation methods are agnostic to the downstream tasks,
making the learned graph embeddings sub-optimal for a specific downstream task,
namely anomaly detection in our case. Additionally, these methods are specifically
designed for certain SSL frameworks, and it is non-trivial (if at all possible) to extend
them to the general SSL framework. Moreover, these automated SSL strategies are
computationally expensive, rendering them impractical in real-world applications.
As an initial step towards mitigating this long-standing but neglected issue, we
propose a lightweight and plug-and-play approach dubbed AutoGAD, to automate
SSL for truly unsupervised graph anomaly detection. Specifically, AutoGAD lever-
ages a so-called internal evaluation strategy (Ma et al. 2023), without relying on
any ground-truth labels (whether explicitly or implicitly), to select optimal combina-
tion weights and/or SSL-specific hyperparameter values. Moreover, we theoretically
analyze the internal evaluation strategy to prove why it is effective and empirically
demonstrate this.
Overall, our main contributions can be summarized as follows:
● We raise renewed awareness to the label information leakage issue, which is criti-
cal but often overlooked in the unsupervised GAD field;
● Although there exists a plethora of graph SSL methods and GAD approaches,
we are the first to investigate automated SSL specifically for unsupervised GAD;
● We propose a lightweight, plug-and-play approach to automate SSL for truly un-
supervised GAD and provide a theoretical analysis;
● Extensive experiments are conducted using 10 state-of-the-art SSL-based GAD
1 3

Towards automated self-supervised learning for truly unsupervised… Page 5 of 43 44
algorithms on 10 benchmark datasets, demonstrating the effectiveness of our ap-
proach.
2 Related work
Our work is related to node anomaly detection on static attributed graphs, self-super-
vised learning for graph anomaly detection, automated self-supervised learning, and
automated anomaly detection.
2.1 Anomaly detection on attributed graphs
Early methods for node anomaly detection in static attributed graphs, such as AMEN
(Perozzi and Akoglu 2016), Radar (Li et al. 2017a), and Anomalous (Peng et al.
2018), are not based on deep learning. These methods work well on low-dimensional
attributed graphs, but their performance is limited on complex graphs with high-
dimensional node attributes.
Recently, deep learning-based methods, including DOMINANT (Ding et al. 2019),
AnomalyDAE (Fan et al. 2020), and GUIDE (Yuan et al. 2021), have been proposed
for GAD. These methods usually employ graph autoencoders to encode nodes fol-
lowed by decoders to reconstruct the adjacency matrix and/or node attributes. As
a result, nodes with large reconstruction errors are considered anomalies. Despite
their superior performance to non-deep learning methods, these reconstruction-based
methods still suffer from sub-optimal performance, as reconstruction is a generic
unsupervised learning objective. Besides, these methods require the full attribute and
adjacency matrices as model input, making them unsuitable or even impossible for
large graphs.
2.2 Self-supervised learning for graph anomaly detection
Graph SSL aims to learn a model by using supervision signals generated from the
graph itself, without relying on human-annotated labels (Liu et al. 2022c). It has
achieved promising performance on typical graph mining tasks such as representa-
tion learning (Jiao et al. 2020) and graph classification (Zeng and Xie 2021). Liu
et al. (2021) first applied SSL to the GAD problem. Their proposed method CoLA
performs single scale comparison (node-subgraph) for anomaly detection. However,
ANEMONE (Jin et al. 2021a) argues that modeling the relationships in a single con-
trastive perspective leads to limited capability of capturing complex anomalous pat-
terns. Hence, they propose additional node-node contrast. Additionally, GRADATE
(Duan et al. 2023) and M-MAG (Liu et al. 2023) combines various multi-contrast
objectives, namely node-node, node-subgraph, and subgraph-subgraph contrasts for
node anomaly detection. To achieve better performance, SL-GAD (Zheng et al. 2021)
combines multi-view contrastive learning and generative attribute regression, while
Sub-CR (Zhang et al. 2022) combines multi-view contrastive learning and graph
autoencoder. Finally, CONAD (Xu et al. 2022b) considers both contrastive learning
and generative reconstruction for better node anomaly detection.
1 3

44 Page 6 of 43 Z. Li et al.
2.3 Automated self-supervised learning
Seminal work on automated data augmentation for images (Ratner et al. 2017; Cubuk
et al. 2018) was followed by work improving (Cubuk et al. 2018) via faster searching
mechanisms (Ho et al. 2019; Lim et al. 2019; Cubuk et al. 2020) or advanced optimi-
zation methods (Hataya et al. 2020; Li et al. 2020a; Zhang et al. 2019).
In the context of automated data augmentation for graphs, related work exists on
graph representation learning (Hassani and Khasahmadi 2022; Suresh et al. 2021;
Jin et al. 2021b; Xie et al. 2022; Yin et al. 2022; You et al. 2021), node classification
(Zhao et al. 2021a; Sun et al. 2021), and graph-level classification (Luo et al. 2022;
Yue et al. 2022; Yin et al. 2022). For example, JOAO (You et al. 2021) learns the
sampling distribution of a set of predefined graph augmentations. AD-GCL (Suresh
et al. 2021) designs a learnable edge dropping augmentation and employs adversarial
training strategy, and AutoGCL (Yin et al. 2022) proposes a learnable graph view
generator that learns a probability distribution over the node-level augmentations.
Further, Luo et al. (2022) augment graph data samples, while Yue et al. (2022) per-
turb the representation vector. However, these methods focus on other typical graph
learning tasks and it is unclear how to use them for unsupervised GAD.
2.4 Automated anomaly detection
Recent studies (Zhao et al. 2021b; Bahri et al. 2022; Ding et al. 2022; Zhao and Ako-
glu 2022) pointed out that unsupervised anomaly detection methods tend to be highly
sensitive to the values of their hyperparameters (HPs). For example, Zhao et al.
(2021b) show that a 10x performance difference is observed for LOF (Breunig et al.
2000) by changing the number of nearest neighbors. Even more, Ding et al. (2022)
indicate that deep anomaly detection methods suffer more from such HP sensitivity
issues. Concretely, Zhao and Akoglu (2022) demonstrate that RAE (Zhou and Paffen-
roth 2017) exhibits a 37x performance difference with different HPs configurations.
To tackle this issue, automated HP tuning and model selection for unsupervised
anomaly detection has received increasing but insufficient attention; Bahri et al.
(2022) present an overview. Inspired by Bahri et al. (2022); Zhao and Akoglu (2022),
we subdivide existing approaches into two main categories:
● Supervised evaluation methods which require ground-truth labels although
anomaly detection algorithms are unsupervised. Methods include PyODDS (Li
et al. 2020b), TODS (Lai et al. 2021), AutoOD (Li et al. 2021b), and AutoAD (Li
et al. 2021a);
● Unsupervised evaluation methods which do not require ground-truth labels. They
include
● Randomly selecting an HP configuration;
● Selecting an HP configuration via an internal evaluation strategy (Goix 2016;
Zhao et al. 2019; Marques et al. 2020; Putina et al. 2022);
● Averaging the outputs of a set of randomly selected HP configurations (Wen-
zel et al. 2020);
1 3

Towards automated self-supervised learning for truly unsupervised… Page 7 of 43  44
●  Meta-learning based methods (Zhao et al. 2020; Zha et al. 2020; Zhao and
Akoglu 2022).
However, existing automated anomaly detection methods are primarily designed for
non-graph data.
3  Problem statement
We utilize lowercase letters, bold lowercase letters, uppercase letters, and calligraphic
fonts to represent scalars (x), vectors (x), matrices (X), and sets ( ), respectively.
X
Definition 1 (Attributed Graph) We denote an attributed graph as  , ,X
|     |     |     |     |     |     |     |     | =    | ,   |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- |
|     |     |     |     |     |     |     |     | G {V | E } |
where  = v ,...,v  is the set of nodes. Besides,  = e  is the set
|     | V   | { 1 | n } |     |     | E { | ij } i,j | 1,...,n |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | ------- | --- |
of edges, where e =1 if there exists an edge between v  and v  and e ∈{ } =0 other-
|     |     |     | ij  |     |     | i   | j   | ij  |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
n d represents the node attribute matrix, where the i-th row
| wise. Moreover, X |     |     | R × |     |     |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
∈
| vector x |  means the node attribute of v |     |     | .   |     |     |     |     |     |
| -------- | ------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
|          | i                              |     |     | i   |     |     |     |     |     |
Formally, we consider unsupervised node anomaly detection on attributed graphs
(dubbed GAD hereafter), which is defined as follows:
Problem 1 (Node Anomaly Detection on Attributed Graph) Given an attributed graph
, ,X , we aim to learn an anomaly scoring function f() that assigns an
as  =
|     | G {V | E   | }   |     |     |     |     | ·   |     |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
anomaly score s=f(v ) to each node v , with a higher score representing a higher
|     |     |     | i   | i   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
degree of being anomalous. Next, the anomaly scores are used to rank the nodes such
that the top-k nodes can be considered as anomalies.
In this paper, we consider the transductive unsupervised anomaly detection setting:
the graph containing both normal and abnormal nodes is given at the training stage.
Node labels are not accessible during the training stage and they are only used for
performance evaluation. Importantly, the labels of nodes are not (and should not be)
used for HP tuning under this unsupervised setting.
Formally, we consider the hyperparameter optimization problem for unsupervised
graph anomaly detection (dubbed HPO for GAD):
Problem 2 (HPO for GAD) Given a graph   without labels and a graph anomaly
G
detection algorithm f() with hyperparameter space Λ, we aim to identify a hyper-
·
parameter configuration λ Λ such that the resulting model f(λ) can achieve
∈
the best performance on  . I.e., suppose λ consists of K different hyperparameters
G
λ ,...,λ ,...,λ , where λ Λ  can be discrete or continuous, we then aim
|     | 1   | k   | K   | k k |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| {   |     |     | }   | ∈   |     |     |     |     |     |
to find
|     |     |                | argmax     | Metric[f(λ | ,...,λ | ,...,λ | ;   | )], |     |
| --- | --- | -------------- | ---------- | ---------- | ------ | ------ | --- | --- | --- |
|     |     |                |            |            | 1      | k      | K   | G   | (1) |
|     |     | λ1∈ Λ1,...,λk∈ | Λk,...,λK∈ | ΛK         |        |        |     |     |     |
where Metric[] is a given performance metric.
·
1 3

44 Page 8 of 43 Z. Li et al.
4 SSL for unsupervised GAD
In this section, we first revisit existing self-supervised learning methods for “unsu-
pervised" graph anomaly detection, followed by an analysis and experiments to
showcase pitfalls in existing studies.
4.1 Existing SSL for “Unsupervised" GAD
Figure 2 shows how existing SSL based GAD methods can be divided into generative
methods and contrastive methods.
That is, a generative method usually consists of two individual SSL tasks, namely
1.1) structure reconstruction that aims to reconstruct the adjacency matrix, and 1.2)
attribute reconstruction that aims to reconstruct the node attribute matrix. On this
basis, the attribute reconstruction error and the structure reconstruction error are
combined to obtain an anomaly score, where higher reconstruction error indicates a
higher degree of anomalousness.
Meanwhile, a contrastive method often consists of two modules: 2.1) data aug-
mentation module, and 2.2) contrastive learning module. First, for each target node,
the data augmentation module utilizes one augmentation function f(δ) to produce
augmented samples, which usually include positive samples and negative samples.
The scenario of using multiple augmentation functions can be obtained in a similar
way. Second, three contrastive perspectives can be applied to contrast positive pairs
and negative pairs: 2.2.1) node-node contrast that contrasts node embedding with
node embedding, and 2.2.2) node-subgraph contrast that contrasts node embedding
with subgraph embedding, and 2.2.3) subgraph-subgraph contrast that contrasts sub-
graph embedding with subgraph embedding.
4.2 Pitfalls in existing methods
In this subsection, we revisit existing SSL-based unsupervised GAD methods by
checking the following three aspects for each method:
Fig. 2 Self-supervised learning based graph anomaly detection methods can be subdivided into gen-
erative based methods and contrastive based methods. A generative based method generally involves
graph structure reconstruction and node attributes reconstruction. A contrastive based method usually
consists of a graph augmentation module and a contrastive learning module
1 3

Towards automated self-supervised learning for truly unsupervised… Page 9 of 43 44
● Which SSL framework does the method employ: generative, contrastive, or both?
● How many SSL-specific hyperparameters are involved? (E.g., combination
weights and others.)
● How are values for key SSL hyperparameters chosen? (E.g., the ratio of node at-
tribute masking or dropping edges, and the combination weights of multiple loss
functions?)
By doing so, we point out that these studies have noticeable pitfalls. More impor-
tantly, we perform experiments to show that the high performance that these meth-
ods claim to achieve is often strongly overestimated due to label leakage issues (cf.
Table 1).
Due to space constraints and to enhance readability, we revisit three representa-
tive SSL-based GAD algorithms in the main paper, including a contrastive method:
ANEMONE (Jin et al. 2021a), a generative method: AnomalyDAE (Fan et al. 2020),
and a combined contrastive and generative method: SL-GAD (Zheng et al. 2021).
4.2.1 Revisiting ANEMONE
ANEMONE (Jin et al. 2021a) is a contrastive method for unsupervised GAD.
Graph Augmentation Module. A single graph augmentation operation is used,
namely Random Ego-Nets generation with a fixed size K. Specifically, taking the tar-
get node as the center, they employ RWR (Tong et al. 2006) to generate two different
subgraphs as ego-nets with a fixed size K. This results in one critical HP, namely K.
Contrast Learning Module. Two contrast perspectives are considered: (1) node-
node contrast between the embedding of a masked target node within the ego-net and
the embedding of the original node, leading to loss term , and (2) node-subgraph
NN
L
contrast within each view, leading to loss term . These loss terms are combined
NS
L
as =(1 α) +α , where α [0,1] is the trade-off HP, giving one more
NN NS
L − L L ∈
critical HP, namely α.
Table 1 Performance variation, quantified as max(AUC) − min(AUC) across different hyperparameter
max(AUC)
settings on ten benchmark datasets
Results are averaged over five independent runs, each initialized with a unique random seed. ‘OOM’
indicates out-of-memory errors, while ‘OOR’ signifies that runtime exceeded the 7-day limit for a
single trial. Cells marked as ‘UNF’ denote persistent underfitting of algorithms, even after reaching the
maximum allowed training epochs (e.g., loss values change by less than 10− 2 after 400 epochs). ‘NAN’
indicates execution errors caused by excessive NaN values; these cases are excluded from further
analysis. Refer to Sect. 6 for details on the experimental setup
1 3

44 Page 10 of 43 Z. Li et al.
HPs Sensitivity & Tuning. By using ground-truth label information, they heuristi-
cally set α to 0.8, 0.6, 0.8 on Cora, CiterSeer, and PubMed respectively, and report the
corresponding results. The setting of K is not studied, and is set to 4 for all datasets.
4.2.2 Revisiting AnomalyDAE
AnomalyDAE (Fan et al. 2020) is a generative method using autoencoders (based on
GNNs) for unsupervised GAD.
Generative Framework. AnomalyDAE consists of two components: (1) an attri-
bute autoencoder to reconstruct the node attributes, where the encoder consists of two
non-linear feature transform layers and the decoder is simply a dot product operation.
This leads to the loss term , and is associated with a penalty HP η; and (2)
A A
L L
a structure autoencoder to reconstruct the structure, where the encoder is based on
GAT (Veličković et al. 2017) and the decoder is a dot product operation followed
by a sigmoid function. This leads to the loss term , and is associated with a
S S
L L
penalty HP θ.
Their overall optimization objective is then defined as =α +(1 α) ,
S A
L L − L
where α (0,1) balances the two objectives.
∈
HPs Sensitivity & Tuning. The paper finds that the AUC usually increases first
and then decreases with the increase of α. However, the specific value of α on each
dataset is selected using label information. The HPs (α,η,θ) are heuristically set as
(0.7, 5, 40), (0.9, 8, 90), (0.7, 8, 10) on BlogCatalog, Flickr, and ACM respectively.
4.2.3 Revisiting SL-GAD
SL-GAD (Zheng et al. 2021) is an unsupervised GAD method that combines both
contrastive and generative objectives.
Contrastive Framework—Data Augmentation Module. The method uses a single
graph augmentation operation, namely Random Ego-Nets generation with a fixed
size K. Specifically, taking the target node as the center, RWR (Tong et al. 2006) is
used to generate two different subgraphs as ego-nets with a fixed size K, where K
controls the radius of the surrounding contexts. This gives one critical HP for graph
augmentation, namely K.
Contrastive Framework—Contrast Learning Module. The Multi-View Contras-
tive Learning module compares the similarity between a node embedding and the
embedding of sampled sub-graphs in augmented views (namely node-subgraph con-
trast), leading to loss terms and . Combining those leads to contrastive
con,1 con,2
L L
objective = 1( + ).
L con 2 L con,1 L con,2
Generative Framework. The Generative Attribute Regression module reconstructs
node attributes, with the aim to achieve node-level discrimination. Specifically, they
minimize the Mean Square Error between the target node’s original and reconstructed
attributes in augmented views, leading to loss terms and . Combining
gen,1 gen,2
L L
those with equal weights leads to generative objective = 1( + ).
L gen 2 L gen,1 L gen,2
The overall optimization objective is then defined as =α +β , where
con gen
L L L
α,β (0,1] are trade-off HPs to balance the importance of the two SSL objectives.
∈
1 3

Towards automated self-supervised learning for truly unsupervised… Page 11 of 43 44
HPs Sensitivity & Tuning. The authors conducted a sensitive analysis and found
that: (1) the performance first increases and then decreases with the increase of K.
For efficiency considerations, they heuristically set the sampled subgraph size K =4
for all datasets; (2) they heuristically fix α=1 for all datasets as they found that this
achieves good performance on most datasets (with the help of label information); and
(3) the selection of β is highly dependent on the specific dataset. Hence, they “fine-
tune" the value of β for each dataset via selecting β from 0.2,0.4,0.6,0.8,1.0
{ }
using labels.
4.2.4 Other SSL-based GAD methods
Due to space constraints, the analyses of other SSL-based GAD methods, includ-
ing CoLA (Liu et al. 2021), GRADATE (Duan et al. 2023), Sub-CR (Zhang et al.
2022), CONAD (Xu et al. 2022b), DOMINANT (Ding et al. 2019), GUIDE (Yuan
et al. 2021), and GAAN (Chen et al. 2020b), are given in Appendix A. These meth-
ods are all representatives of recent advancements in using SSL to conduct unsuper-
vised graph anomaly detection, and have yielded outstanding detection performance.
Likewise, however, these methods also exhibit pitfalls with regard to hyperparameter
tuning, similar to those of ANEMONE (Jin et al. 2021a), AnomalyDAE (Fan et al.
2020), and SL-GAD (Zheng et al. 2021).
4.3 Sensitivity analysis
After revisiting recent SSL-based unsupervised GAD methods, we now empirically
investigate their sensitivity to SSL-related HPs in a systematic way. More concretely,
we report their performance variations in terms of ROC-AUC values under different
hyperparameter configurations (see Sect. 6 for experiment settings).
As shown in Fig. 1, for a typical run with different hyperparameter configurations,
the performance of ANEMONE (Jin et al. 2021a) can vary strongly on each of the
ten datasets. Other SSL-based GAD algorithms exhibit similar behavior; extensive
results and analysis are deferred to Appendix B for space reasons.
For an in-depth yet compact analysis, Table 1 presents average results over five
independent runs when varying SSL-related hyperparameter values. Specifically,
CoLA (Liu et al. 2021), GUIDE (Yuan et al. 2021), DOMINANT (Ding et al. 2019),
GRADATE (Duan et al. 2023), and Sub-CR (Zhang et al. 2022) demonstrate moder-
ate performance variations (namely between 7.3% and 14.7% on average). Mean-
while, CONAD (Xu et al. 2022b), ANEMONE (Jin et al. 2021a), SL-GAD (Zheng
et al. 2021), GAAN (Chen et al. 2020b), and AnomalyDAE (Fan et al. 2020) suffer
from large performance variations (namely ranging from 15.7% to 30.0% on aver-
age). From Sect. 4.2 and Appendix A, we see that the results reported in existing
papers are often obtained by manually tuned HPs (in a post-hoc way with label infor-
mation), thereby leading to strongly overestimated performance for real-world appli-
cations where labels are not accessible. To mitigate this severe issue, we propose
AutoGAD, a method for automating hyperparameter selection in SSL for GAD and
achieving truly unsupervised graph anomaly detection. Importantly, AutoGAD does
not need any ground-truth labels.
1 3

44 Page 12 of 43 Z. Li et al.
5 AutoGAD: using internal evaluation to automate SSL for GAD
Our proposed approach, called AutoGAD, consists of two parts: (1) an unsupervised
performance metric, and (2) an effective search method. Importantly, and as men-
tioned before, the chosen performance metric—denoted Metric[] in Eq. 1—should
·
not use any ground-truth label information, simply because this is not available in
a truly unsupervised setting. We therefore propose to utilize an internal evaluation
strategy, which will be elucidated later. Next, given the impracticality of evaluating
an infinite number of configurations for continuous hyperparameter domains, another
challenge is the efficient exploration of the search space. Section 5.2 describes a
straightforward approach using discretization and grid search that works well in prac-
tice, as shown in the next section.
5.1 Internal evaluation strategy
The intuition behind the internal evaluation strategy that we use is to measure the
similarity of anomaly scores within the same predicted anomaly class and the dis-
similarity between anomaly scores across different predicted classes (i.e., ‘anomaly’
or ‘no anomaly’). As we will prove later, optimizing the resulting measure is equiva-
lent to simultaneously minimizing the false positive rate and the false negative rate.
In this way, we aim to evaluate and optimize the performance of the anomaly detector
under different SSL configurations without having to rely on any ground-truth labels.
5.1.1 Contrast score margin
The metric that we use is Contrast Score Margin (Xu et al. 2019), which was intro-
duced before but not for graph anomaly detection, and is defined as
µˆ µˆ
T(f)= O− I ,
(2)
1(δˆ2 +δˆ2)
k O I
√
where µˆ and δˆ2 denote the average and variance of the anomaly scores of the k
O O
predicted anomalous objects (Oˆ), respectively. Moreover, µˆ and δˆ2 represent the
I I
average and variance of the anomaly scores of the k predicted normal objects (ˆI)
with the highest scores, respectively. Intuitively, the metric focuses on the k predicted
normal objects that are most similar to the k predicted anomalous objects, and aims
to measure the margin of the anomaly scores between them. It only takes linear time
with respect to n to compute.
5.1.2 Analysis
We now analyze why the internal evaluation metric Contrast Score Margin should
work for our purposes.
1 3

Towards automated self-supervised learning for truly unsupervised… Page 13 of 43 44
Theorem 1 (Minimizing False Positives and Negatives) For an anomaly detector
f() on dataset X, assume the anomaly scores of the top k true anomalies (O) have
·
the expected value µ and variance δ2, and the anomaly scores of the top k true
O O
normal objects with the highest anomaly scores (I) have the expected value µ and
I
variance δ2, then maximizing T is equal to simultaneously minimizing the false posi-
I
tive rate and the false negative rate.
Proof According to Cantelli’s inequality, which makes no assumptions on specific prob-
ability distributions, on the one hand, for x O we have P(f(x) µ α) δ O 2 ,
∈ ≤ O− ≤ δ2+α2
O
where α 0 is a small constant chosen based on a desired bound on the false nega-
≥
tive. By replacing α=aδ , we have P(f(x) µ aδ ) 1 , which is the
O ≤ O− O ≤ 1+a2
False Negative Bound. In other words, f(x) has a maximum probability of 1 to be
1+a2
less than µ aδ .
O− O
On the other hand, for y I we have P(f(y) µ +β) δ I 2 , where β 0 is
∈ ≥ I ≤ δ2+β2 ≥
I
a small constant chosen based on a desired bound on the false positive. By replacing
β =bδ , we have P(f(y) µ +bδ ) 1 , which is the False Positive Bound.
I ≥ I I ≤ 1+b2
In other words, f(y) has a maximum probability of 1 to be larger than µ +bδ .
1+b2 I I
Furthermore, (µ aδ ) (µ +bδ ) = (µ µ ) (bδ +aδ ). Hence, to
O− O − I I O− I − I O
ensure a small false positive rate and a small false negative rate, we want µ µ to
O− I
be as large as possible while bδ +aδ as small as possible. In fact, this is equivalent
O I
to optimize the Contrast Score Margin, i.e.,
µ µ
T(f)= O− I
1(δ2 +δ2)
k O I
√
Note that if an anomaly detector f() produces a perfect anomaly detection result,
·
i.e., for any x O and any y X O, we have f(x)>f(y), then we will obtain
∈ ∈ \
µ µ >0. In another extreme, if f() produces a poor anomaly detection result,
O− I ·
i.e., for all x O and any y X O, we have f(x)<f(y), then we will obtain
∈ ∈ \
µ µ <0. Meanwhile, if an anomaly detector f() produces a random result,
O− I ·
i.e., for some x O and any y X O, we have f(x)<f(y), then we may obtain
∈ ∈ \
µ O− µ I <0 or µ O− µ I ≈ 0. □
5.1.3 Improvements and remarks
In practice we observed that Eq. 2 is not always stable. Possible reasons are that
(1) the proportion of anomalies is usually very small (namely less than 5% in most
datasets); and (2) the exact number of anomalies is generally not known (even for
a dataset with injected anomalies, there may exist some natural samples that exhibit
similar behaviors as anomalies). Therefore, we propose to modify Eq. 2 as follows:
1 3

44 Page 14 of 43 Z. Li et al.
µˆ µ˜
T(f)= O− I ,
(3)
δˆ2 +δ˜2
O I
√
where µˆ and δˆ2 denote the average and variance of the anomaly scores of the k pre-
O O
dicted anomalous objects, respectively. Importantly, µ˜ and δ˜2 represent the average
I I
and variance of the anomaly scores of the remaining n k objects, respectively. This
−
change should lead to more stable performance compared to using anomaly scores
of the top-k predicted normal objects in Eq. 2. This is because the true labels are not
accessible, and thus we utilize the pseudo-labels to identify the top-k anomalous and
the top-k normal objects. However, the pseudo-labels of the top-k “pseudo-normal"
objects may not be reliable due to the two facts stated above.
Moreover, to ensure the effectiveness of this internal evaluation strategy, we have
to make sure that: (1) we use the same algorithm with different hyperparameter con-
figurations; and (2) the scales of the loss values are approximately the same when
combining multiple loss functions in the same algorithm. In other words, we should
not directly use the strategy to select among different heterogeneous anomaly detec-
tion algorithms (please refer to Appendix F for empirical evidence of this).
5.2 Discretization and grid search
Algorithm 1 Grid Search for Anomaly Detector Hyperparameter Optimization
To find the optimal hyperparameter configuration, we first perform discretization of
the continuous search space and then conduct grid search. The corresponding pseudo-
code is provided in Algorithm 1, with a detailed explanation presented below.
Discretization of Continuous Search Space (Lines 1–2). To make the over-
all search process feasible, we discretize the hyperparameter space. Assume we
are given a GAD algorithm f() with its set of hyperparameters λ Λ. With-
· ∈
out loss of generality, we assume there are L different hyperparameters and let
λ= λ(1),λ(2),...,λ(L) , where each λ(l) Λ(l) for l=1,2,...,L. If a hyperpa-
{ } ∈
1 3

Towards automated self-supervised learning for truly unsupervised… Page 15 of 43  44
rameter domain Λ(l) is continuous, we discretize it into a finite set of values (with car-
dinality Λ(l)
). This results in M possible hyperparameter configurations, represented
|              | | |      |          |        |           |       |           |        |
| ------------ | -------- | -------- | ------ | --------- | ----- | --------- | ------ |
| by the set λ |          | λ ,...,λ | ,...,λ | , where λ | λ(    | 1),λ( 2), | ,λ( L) |
|              | search   | = 1      | m      | M         | m = m | m         | m      |
|              |          | {        |        | }         | {     | ···       | }      |
| and M        | = L Λ(l) |          |        |           |       |           |        |
|              | l=1|     | .        |        |           |       |           |        |
|
Grid Se∏arch (Lines 3–11). Once the hyperparameter search space is discretized,
we apply grid search to evaluate each configuration. For each hyperparameter con-
| figuration λ | λ   | , we run the GAD algorithm f(λ |     |     |                         |     |      |
| ------------ | --- | ------------------------------ | --- | --- | ----------------------- | --- | ---- |
|              | m   | search                         |     |     | m ) on the given graph  |     |  to  |
|              | ∈   |                                |     |     |                         |     | G    |
produce a vector of anomaly scores s ( )=f(λ ; ). These scores are evaluated
|     |     |     | m   | G m | G   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
using an internal unsupervised performance metric T()( with Eq. 3) to yield a final
·
score t m ( )=T(s m ( )). The configuration that maximizes T() is selected as the
|     | G   | G   |     |     | ·   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
optimal values of hyperparameters.
Note that more advanced strategies than grid search, such as SMBO-based opti-
mization (Jones et al. 1998), could be employed (see Appendix E for an example).
However, these methods often introduce additional hyperparameters (whose tuning
may be non-trivial), which contradicts our goal of automated anomaly detection.
6  Experiments
We aim to answer the following research questions (RQ):
RQ1  How sensitive are existing SSL-based GAD methods to the values of their
hyperparameters?
RQ2  How effective is AutoGAD in tuning SSL-related hyperparameter values for
these methods?
We describe the experiment settings, including the datasets, baselines, evaluation
metrics, and software and hardware used, which is followed by the experiment results
and their interpretation.
6.1  Datasets
We use three popular citation networks, namely Cora, Citeseer, and Pubmed (Sen
et al. 2008) with injected anomalies, one social network Flickr (Zeng et al. 2019) (less
homophily) with injected anomalies, ACM (Tang et al. 2008) as well as BlogCataLog
(Zeng et al. 2019) with injected anomalies. Particularly, we follow the methods used
by ANEMONE (Jin et al. 2021a) and CoLA (Liu et al. 2021) to inject structural and
contextual anomalies. Note that Liu et al. (2022b) have slightly modified this injec-
tion procedure. Following (Qiao and Pang 2024), we also consider four commonly-
used graph datasets with real anomalies: Amazon (Sánchez et al. 2013), Facebook
(Leskovec and Mcauley 2012), Reddit (Kumar et al. 2019), and YelpChi (Rayana and
Akoglu 2015). The resulting datasets are summarized in Table 2.
6.2  Baselines
We study the performance of the following SSL-based graph anomaly detection
methods:
1 3

| 44  Page 16 of 43 |     |     |     | Z. Li et al. |
| ----------------- | --- | --- | --- | ------------ |

Table 2 Summary of datasets:  Dataset #Nodes #Edges #Attributes #Anomalies
anomalies in Cora, CiteSeer,
|     | Cora (Sen et al.  | 2708 | 11,060 1433 | 138 (5.1%) |
| --- | ----------------- | ---- | ----------- | ---------- |
PubWeb, ACM, BlogCatalog,
2008)
and Flickr are synthetically
injected following established  CiteSeer Sen  3327 4732 3703 150 (4.5%)
| methods (Jin et al. 2021a; Liu  | et al. (2008) |     |     |     |
| ------------------------------- | ------------- | --- | --- | --- |
et al. 2021), while Amazon,  PubMed (Sen  19,717 44,338 500 150 (2.5%)
| Facebook, Reddit, and YelpChi  | et al. 2008) |     |     |     |
| ------------------------------ | ------------ | --- | --- | --- |
contain real-world anomalies ACM (Tang et al.  16,484 71,980 8337 600 (3.6%)
2008)
|     | BlogCataLog  | 5196 | 171,743 8189 | 300 (5.8%) |
| --- | ------------ | ---- | ------------ | ---------- |
(Zeng et al. 2019)
|     | Flickr (Zeng et al.  | 7575 | 239,738 12,407 | 450 (5.9%) |
| --- | -------------------- | ---- | -------------- | ---------- |
2019)
|     | Amazon (Sánchez  | 10,244 | 175,608 25 | 693 (6.7%) |
| --- | ---------------- | ------ | ---------- | ---------- |
et al. 2013)
|     | Facebook (Lesk- | 1081 | 55,104 576 | 27 (2.5%) |
| --- | --------------- | ---- | ---------- | --------- |
ovec and Mcauley
2012)
|     | Reddit (Kumar  | 10,984 | 168,016 64 | 366 (3.3%) |
| --- | -------------- | ------ | ---------- | ---------- |
et al. 2019)
|     | YelpChi (Rayana  | 24,741 | 49,315 32 | 1217   |
| --- | ---------------- | ------ | --------- | ------ |
|     | and Akoglu 2015) |        |           | (4.9%) |
●  Generative methods: DOMINANT (Ding et al. 2019), AnomalyDAE (Fan et al.
2020), GUIDE (Yuan et al. 2021), GAAN (Chen et al. 2020b);
●  Contrastive methods (and some also generative): CoLA (Liu et al. 2021), ANEM-
ONE (Jin et al. 2021a), GRADATE (Duan et al. 2023), SL-GAD (Zheng et al.
2021), Sub-CR (Zhang et al. 2022), CONAD (Xu et al. 2022b).
Particularly, the SSL-related HPs for each GAD algorithm and their discretized
search spaces are given in Table 6 in the Appendix. These GAD methods are further
summarized in Table 7 in the Appendix.
6.3  Evaluation metrics
To evaluate the effectiveness of various GAD algorithms, we utilize the ROC-
AUC metric (Hanley and McNeil 1982) (AUC for short hereinafter), where a value
approaching 1 denotes the best possible performance.
Moreover, to quantify the performance variation of an individual GAD method
under different SSL-related HP configurations, we define the following performance
variation metric:
|           | max(AUC)  | min(AUC)   |                    |              |
| --------- | --------- | ---------- | ------------------ | ------------ |
|           |           | −          | ,                  | (4)          |
|           |           | max(AUC)   |                    |              |
| max(AUC)  | min(AUC)  |            |                    |              |
| where     | and       | represent  | the  maximum  and  | minimum  of  |
achieved AUC values for the evaluated GAD algorithm with different configurations,
respectively. Hence, the smaller this value is, the less sensitive the algorithm is to
SSL-related HPs.
1 3

Towards automated self-supervised learning for truly unsupervised… Page 17 of 43 44
Further, we define the performance gain over minimal AUC as
CSM(AUC) min(AUC)
− , (5)
min(AUC)
where CSM(AUC) indicates the AUC value obtained for the evaluated GAD algo-
rithm when configured with the HPs selected using the Contrast Score Margin. This
metric can quantify the effectiveness of our strategy relative to the worst case hyper-
parameter setting. Next, we define performance gain over median AUC as
CSM(AUC) median(AUC)
− , (6)
median(AUC)
where median(AUC) represents the median of the obtained AUC values for the
GAD algorithm with different configurations. Thus, if the value of this metric is posi-
tive, the GAD algorithm configured with our selected HPs can at least outperform its
counterparts configured with 50% of the other sampled hyperparameter values.
Furthermore, we define performance gain over maximal AUC as
CSM(AUC) max(AUC)
− , (7)
max(AUC)
where max(AUC) represents the maximum of the obtained AUC values for the
GAD algorithm with different configurations. Thus, if the value of this metric is close
to zero, the GAD algorithm configured with our selected HPs can approximately
achieve the best possible performance.
6.4 Software and hardware
All algorithms are implemented in Python 3.8 (using PyTorch (Paszke et al. 2019)
and PyTorch Geometric (Fey and Lenssen 2019) libraries when applicable) and ran
on workstations equipped with AMD EPYC7453 CPUs (with 64GB RAM) and/or
Nvidia RTX4090 GPUs (with 24.0 GB video memory). All code and datasets are
available on GitHub.1
6.5 Results and analysis
We answer the research questions as follows:
6.5.1 RQ1: sensitivity of SSL-based GAD methods to HPs
The results are summarized in Table 1 for five independent runs. Typical runs are
depicted in Fig. 1 and in Figs. 5, 6, 7, 8, 9, 10, 11, 12 and 13 in Appendix B. We
briefly analyzed the results in Sect. 4.3; more detailed analyses are given in Appendix
1 https://github.com/ZhongLIFR/AutoGAD2024.
1 3

44 Page 18 of 43 Z. Li et al.
Table 3 Performance gain over minimal AUC defined as CSM(AUC) − min(AUC)
min(AUC)
Results are averaged on five independent runs. CSM is contrast score margin defined in Equation 3,
while OOM, OOR, UNF, and NAN convey the same meanings as in Table 1
Table 4 Performance gain over median AUC defined as CSM(AUC) − median(AUC)
median(AUC)
Results are averaged on five independent runs. CSM is contrast score margin defined in Equation 3,
while OOM, OOR, UNF, and NAN convey the same meanings as in Table 1. For enhanced readability,
cells are color-coded based on their values, as specified in the legend
B. To recall, five out of ten algorithms show moderate performance variations, while
the remaining five algorithms demonstrate large performance variations when the
values of SSL-related HPs are varied. In other words, SSL-based GAD methods are
(sometimes highly) sensitive to hyperparameter values.
6.5.2 RQ2: effectiveness of AutoGAD in tuning SSL-related HPs
The results are summarized in Tables 3, 4 and 5 for five independent runs, while Figs.
1, 5, 6, 7, 8, 9, 10, 11, 12 and 13 depict typical runs. We have the following main
observations:
(1) From Table 3, one can see that AutoGAD can result in moderate performance
gain over minimal AUC (namely between 4.1% and 13.1% on average) for
CoLA, GUIDE, CONAD, DOMINANT, Sub-CR, and GRADATE. Recall that
five of these algorithms (including CoLA, GUIDE, DOMINANT, GRADATE,
and Sub-CR) exhibit moderate performance variations, ranging from 7.3% to
14.7% on average. Moreover, AutoGAD leads to large performance gain over
minimal AUC (namely between 15.2% and 34.8% on average) for the remain-
ing four algorithms, which suffer from large performance variations (namely
between 17.0% and 30.0% on average). Overall, AutoGAD is substantially better
1 3

Towards automated self-supervised learning for truly unsupervised… Page 19 of 43 44
Table 5 Performance gain over maximal AUC defined as CSM(AUC) − max(AUC)
max(AUC)
Results are averaged on five independent runs. CSM is contrast score margin defined in Equation 3,
while OOM, OOR, and NAN convey the same meanings as in Table 1
than the worst case, i.e., when one happens to select the HP values that give the
smallest AUC value.
(2) From Table 4, one can see that AutoGAD can result in positive performance gain
over median AUC in 8 out 10 algorithms (ranging from 0.6% to 5.6% on aver-
age), implying that the HP values selected by AutoGAD are better than at least
50% of randomly selected HP values. Particularly, the performance gains over
median AUC for GRADATE (Duan et al. 2023), ANEMONE (Jin et al. 2021a),
and GAAN (Chen et al. 2020b) are 5.6%,4.0%, and 2.5% respectively, which
shows that AutoGAD is highly effective for these methods.
(3) From Table 5, one can see that AutoGAD can result in performance gain over
max AUC larger than 10% in 8 out 10 algorithms, implying that the HP values
−
selected by AutoGAD can achieve performances that are comparable to optimal
performances. For instance, the performance gains over max AUC for GRA-
DATE and SL-GAD are 1.7% and 7.5% respectively, which shows that
− −
AutoGAD is highly effective for these methods while they show moderate or
large performance variations (12.5% and 22.8% respectively).
(4) Following the above observations, we check the details in Fig. 7 for SL-GAD,
Fig. 6 for GRADATE, and Fig. 1 for ANEMONE. For SL-GAD and GRADATE,
AutoGAD often selects HP values better than 90% of randomly selected HPs val-
ues on most datasets. For ANEMONE, the HP values selected by AutoGAD often
outperform 75% of randomly selected HP values.
6.5.3 Sensitivity analysis
Sensitivity to k. The selection of the value of k in our experiments acknowledges the
varying anomaly ratios across different datasets, implying that k should ideally differ
to reflect the unique characteristics of each dataset. We operated under the assump-
tion that the anomaly ratio within a dataset is approximately known, a premise that
aligns with real-world anomaly detection tasks where some prior knowledge about
the frequency of anomalies is often available.
As shown in Fig. 3, we conducted a sensitivity analysis on k to assess the stabil-
ity of AutoGAD against deviations from the true anomaly ratio. The findings from
this analysis indicate that the effectiveness of AutoGAD remains stable as long as
k is not drastically distant from the actual anomaly ratio, reinforcing the practical
1 3

44 Page 20 of 43 Z. Li et al.
Fig. 3 Sensitivity analysis of k (for our proposed AutoGAD) on dataset CiteSeer with all investigated
SSL-based GAD algorithms. It can be seen that AutoGAD remains stable as long as k is not drastically
distant from the actual anomaly ratio (namely 4.5%) for all SSL-based GAD algorithms
Fig. 4 Performance of AutoGAD across different granularity levels of search grids using ANEMONE
on the Cora, ACM, and Facebook datasets. Similar trends were observed for other anomaly detectors
and datasets, which are omitted for brevity
applicability of our approach even when exact anomaly proportions are not precisely
determined.
Sensitivity to the Granularity of the Search Grid. Acknowledging the significance
of search space granularity in the performance of AutoGAD, we conduct a sensi-
tivity analysis by varying the granularity levels of the search grids in grid search.
Figure 4 presents representative results using ANEMONE (Jin et al. 2021a) on the
Cora, ACM, and Facebook datasets with four levels of search granularity, as follows:
● Granularity Level 1: α 0, 0.2, 0.4, 0.6, 0.8, 1 , K 2, 4 ;
∈{ } ∈{ }
● Granularity Level 2: α 0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99,
∈{
1 , K = 2, 3, 4, 5 ;
} { }
1 3

Towards automated self-supervised learning for truly unsupervised… Page 21 of 43 44
● Granularity Level 3: α 0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
∈{
0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1 ,K 2, 3, 4, 5 ;
} ∈{ }
● Granularity Level 4: α 0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
∈{
0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525,
0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85,
0.875, 0.9, 0.925, 0.95, 0.975, 0.99, 1 ,K 2, 3, 4, 5, 6, 7 .
} ∈{ }
The results indicate that finer search grids tend to improve the performance of
AutoGAD. This is expected, as the optimal value achievable in a finer search grid
cannot be worse than that in a coarser grid. Similar observations were made for other
anomaly detection methods and datasets, which are omitted here for brevity.
7 Alternative strategies and discussion
Internal evaluation strategies aim to assess the quality of a model based solely on
internal information, without relying on external information such as ground-truth
labels. Internal information can typically be derived from two sources: (1) the input
samples, such as feature values of instances in tabular data or node attributes in graph
data; or (2) the anomaly scores generated by an anomaly detection model. Beyond
the Contrast Score Margin (Xu et al. 2019) discussed in this paper, additional internal
evaluation strategies exist for unsupervised model selection in anomaly detection.
According to Ma et al. (2023), these strategies can be categorized as stand-alone or
consensus-based internal evaluation strategies; we will next discuss each category.
7.1 Stand-alone internal evaluation strategies
Stand-alone strategies rely solely on input samples or individual anomaly detection
methods (or models with specific HP configurations in our setting) and their output
anomaly scores. Key methods include:
● IROES (Marques et al. 2015, 2020) quantifies the separability of each input sam-
ple, assuming that a good anomaly detection model assigns high anomaly scores
to highly separable samples. However, separability scores are defined only for
tabular data, making extension to graph data non-trivial. Additionally, comput-
ing separability scores is computationally expensive, posing challenges for large
datasets.
● Mass-Volume and Excess-Mass (Goix 2016) use statistical tools to measure the
quality of an anomaly scoring function. These methods operate on the raw input
samples rather than anomaly scores and assume that anomalies occur in the dis-
tribution’s tail. However, they are restricted to tabular data and are not applicable
to graph data.
● Clustering Validation Metrics (Nguyen et al. 2016) assume that an anomaly de-
tector divides input samples into two clusters: abnormal and normal. Clustering
validation metrics, such as the Xie-Beni index (Xie and Beni 1991), are then used
to evaluate performance. While clustering coefficients on graphs could be analo-
1 3

44 Page 22 of 43 Z. Li et al.
gous (Li et al. 2017b), these metrics are computationally expensive, particularly
for large datasets.
7.2 Consensus-based internal evaluation strategies
Consensus-based strategies assess the agreement among multiple anomaly detection
models (or the same model with varying HP configurations in our setting). Key meth-
ods include:
● UDR (Duan et al. 2019) assumes that good HP configurations yield consistent re-
sults under different random initializations, while poor configurations do not. Ma
et al. (2023) repurposed UDR to select among heterogeneous anomaly detectors,
assuming that good detectors produce consistent results across HP configurations.
● Model Centrality (Lin et al. 2020) hypothesizes that good models are close to the
optimal model and thus to each other.
● Model Centrality by HITS (Kleinberg 1999) follows a similar hypothesis but
employs a different computation approach.
● Unsupervised Anomaly Detection Ensembling (Ma et al. 2023) infers pseudo
anomaly labels by aggregating outputs from a predefined subset of good models.
However, this method is less feasible in our setting as there is no such pre-defined
good models.
Two challenges remain when utilizing these strategies in our setting: (1) validat-
ing the underlying assumptions, which often lack theoretical justification, and (2)
addressing their computational expenses, as consensus-based methods require pair-
wise comparisons. In contrast, Contrast Score Margin is computationally efficient, as
it operates on anomaly scores rather than on raw data points and it avoids pairwise
comparisons.
7.3 Discussion and future work
Although Ma et al. (2023) demonstrated that many internal evaluation strategies per-
form suboptimally for selecting heterogeneous anomaly detectors, we hypothesize
that some can be valuable for hyperparameter tuning within a single anomaly detec-
tion model. However, this is beyond the scope of this paper and is left for future
work. The primary objectives of this paper are twofold:
● We highlight flaws in existing studies on using SSL for unsupervised graph
anomaly detection. Specifically, we:
1. Review these studies, showing that most tune HPs arbitrarily or selectively.
2. Demonstrate empirically, through extensive experiments, that these methods
are highly sensitive to HP settings. Consequently, we argue that these meth-
ods may suffer from label information leakage under unsupervised learn-
ing settings, leading to overstated performance in practical scenarios where
label-based tuning is inaccessible.
1 3

Towards automated self-supervised learning for truly unsupervised… Page 23 of 43 44
● We propose an initial solution to these issues by utilizing and improving the Con-
trast Score Margin. This internal evaluation metric was selected for two reasons:
1. It operates on anomaly scores rather than on raw data points and avoids pair-
wise computations, making it computationally efficient and suitable for large
datasets.
2. Theoretical guarantees for its properties are provided by Theorem 1, which
may not hold for other internal evaluation strategies.
This paper does not aim to provide a perfect solution to the issues mentioned above.
Instead, our goal is to spark interest in the research community to address these chal-
lenges. Unlike Ma et al. (2023), we do not aim to conduct a comprehensive review and
evaluation of internal evaluation strategies for SSL-based graph anomaly detection,
as this requires significant computational resources and in-depth analysis. Neverthe-
less, we aim to explore this direction in future work by considering and potentially
repurposing the internal evaluation strategies reviewed in Ma et al. (2023). We have
described a more advanced search strategy than grid search, namely SMBO-based
optimization (Jones et al. 1998), in Appendix E, without experimental evaluation.
This is because this method introduces additional hyperparameters and their tuning is
non-trivial, contradicting our goal of automated anomaly detection. Other advanced
hyper-parameter tuning methods (Yang and Shami 2020; Bischl et al. 2023; Zhao and
Akoglu 2024) to speed up the search are possible, and we leave their explorations for
future work.
8 Conclusions
SSL has received much attention in recent years, and many recent studies have
explored SSL to perform unsupervised GAD. However, we found that most existing
studies tune hyperparameters arbitrarily or selectively (i.e., guided by labels), and
our empirical findings reveal that most methods are highly sensitive to hyperparam-
eter settings. Using label information to tune hyperparameters in an unsupervised
setting, however, is label information leakage and leads to severe overestimation of
model performance. To mitigate this issue, we introduce AutoGAD, the first auto-
mated hyperparameter selection method for SSL-based unsupervised GAD. Exten-
sive experiments demonstrate the effectiveness of our proposed strategy. Overall,
we aim to raise awareness to the label information leakage issue in the unsupervised
GAD field, and AutoGAD provides a first step towards achieving truly unsupervised
SSL-based GAD.
1 3

44 Page 24 of 43 Z. Li et al.
Pitfalls in existing methods (full analysis)
CoLA
Particularly, CoLA (Liu et al. 2021) is the first contrastive-based framework for
unsupervised GAD. The design of its data augmentation module and contrast learn-
ing module is as follows.
Data Augmentation Module They consider one type of data augmentation, sub-
graph sampling, to obtain local augmented view for each node. Particularly, they
employ RWR (Tong et al. 2006) to generate a sub-graph with a fixed size K in sub-
graph sampling, resulting in one critical HP in graph augmentation, namely K.
Contrast Learning Module They consider a single contrast aspect, namely node-
subgraph contrast between the embedding of the target node and the aggregated
embedding of its local sug-graph, without resulting in any HPs.
HPs Sensitivity & Tuning They conducted sensitive analysis and found that the
selection of subgraph size K is dependent on the specific dataset. The AUC perfor-
mance usually increases first and then decreases with the increasing of K. However,
for efficiency and robustness consideration, they heuristically set the sampled sub-
graph size K =4 for all datasets.
ANEMONE
ANEMONE (Jin et al. 2021a) is a contrastive-based framework for unsupervised
GAD. They argue that modeling the relationships in a single contrastive perspective
leads to limited capability of capturing complex anomalous patterns, and thus pro-
pose additional contrast perspectives as follows.
Data Augmentation Module They consider a single graph augmentation operation,
namely Random Ego-Nets generation with a fixed size K. Specifically, taking the tar-
get node as the center, they employ RWR (Tong et al. 2006) to generate two different
subgraphs as ego-nets with a fixed size K. Overall, they result in one critical HP in
graph augmentation, namely K.
Contrast Learning Module They consider two contrast perspectives: (1) node-
node contrast between the embedding of masked target node within ego-net and the
embedding of the original node, leading to loss term , and (2) node-subgraph
NN
L
contrast within each view, leading to loss term . On this basis, they combine
NS
L
these loss terms as
=(1 α)
NN
+α
NS
L − L L
where α [0,1] is the trade-off HP. Hence, they result in one critical HP in graph
∈
contrast, namely α.
HPs Sensitivity & Tuning In their ablation studies: (1) by using ground-truth label
information, they heuristically set α as 0.8, 0.6, 0.8 on Cora, CiterSeer and PubMed
respectively, and report the corresponding results; and (2) the setting of K was not
studied, and it is set to 4 for all datasets.
1 3

Towards automated self-supervised learning for truly unsupervised… Page 25 of 43 44
GRADATE
GRADATE (Duan et al. 2023) is also a contrastive-based framework. They argue
that subgraph-subgraph contrast is also critical in detecting graph anomalies, and
design it as follows.
Data Augmentation Module They consider a single graph augmentation operation,
namely Edge Modification that removes edges in the adjacency matrix as well as add
the same number of edges. Concretely, they fix a proportion P, and then uniformly
and randomly sample P ·2 M edges from a total of M edges to remove. Meanwhile,
P ·2 M edges are added into the adjacency matrix. Overall, they result in one critical
HP in graph augmentation, namely P.
Contrast Learning Module They consider three contrast aspects: (1) node-node
contrast within each view, leading to loss term ), (2) node-subgraph contrast
NN
L
within each view, leading to loss term , and (3) subgraph-subgraph contrast
NS
L
between original view and augmented view, leading to loss term . On this basis,
SS
L
they combine these loss terms as
=(1 β)
NN
+β
NS
+γ
SS
,
L − L L L
where β,γ (0,1) are trade-off HPs. More, =α +(1 α) , and
NN NN,1 NN,2
∈ L L − L
=α +(1 α) , with and being the loss term in the
NS NS,1 NS,2 NN,1 NN,2
L L − L L L
first and second views respectively. Overall, they result in three critical HPs in graph
contrast, namely the combination weights α,β,γ.
HPs Sensitivity & Tuning In their ablation studies, (1) they compared four differ-
ent graph augmentation strategies, including Gaussian Noise Feature, Feature Mask-
ing, Graph Diffusion, and Edge Modification, and they found that Edge Modification
performs the best across different datasets (with ground-truth labels on test data to
measure the performance); (2) with the help of ground-truth label information on test
data, they heuristically set (α,β) as (0.9, 0.3), (0.1, 0.7), (0.7, 0.1), (0.9, 0.3), (0.7,
0.5), (0.5, 0.5) on EAT, WebKB, UAT, Cora, UAI2010, and Citation respectively; 3)
similarly, they set γ =1 for all datasets; and 4) they also heuristically set P =0.2
for all datasets.
SL-GAD
Different from CoLA, ANEMONE and GRADATE, SL-GAD (Zheng et al. 2021)
combines the contrastive-based framework and the generative-based framework for
unsupervised GAD.
First, the design of the contrastive-based framework is as follows.
Contrastive Framework—Data Augmentation Module They consider a single
graph augmentation operation, namely Random Ego-Nets generation with a fixed
size K. Specifically, taking the target node as the center, they employ RWR (Tong
et al. 2006) to generate two different subgraphs as ego-nets with a fixed size K, where
K controls the radius of the surrounding contexts. Overall, they result in one critical
HP in graph augmentation, namely K. Particularly, they indicate that other augmenta-
1 3

44 Page 26 of 43 Z. Li et al.
tion strategies such as attribute masking and edge modification may introduce extra
anomalies, while random ego-nets and graph diffusion can augment data without
changing the underlying graph semantic information.
Contrastive Framework—Contrast Learning Module They introduce a Multi-
View Contrastive Learning module that compare the similarity between node embed-
ding and embedding of sampled sub-graphs in augmented views (namely node-sub-
graph contrast), leading to two loss terms and corresponding to two
con,1 con,2
L L
augmented views, respectively. On this basis, they obtain the contrastive objective
= 1( + ), which combines the two loss terms with equal weights.
L con 2 L con,1 L con,2
Second, the generative-based framework is designed as follows.
Generative Framework They introduce a Generative Attribute Regression module
that reconstructs node attributes, with the aim to achieve node-level discrimination,
where the encoder is a GCN and the decoder is another GCN. Specifically, they
minimize the Mean Square Error between the target node’s original and reconstructed
attributes in augmented views, leading to two loss terms and corre-
gen,1 gen,2
L L
sponding to two augmented views, respectively. Then they combine them with equal
weights, leading to the generative objective = 1( + ).
L gen 2 L gen,1 L gen,2
At last, their final optimization objective is defined as follows:
=α +β ,
con gen
L L L
where α,β (0,1] are trade-off HPs to balance the importance of two SSL objectives.
∈
HPs Sensitivity & Tuning They conducted sensitive analysis and found that: (1)
the performance first increases and then decreases with the increasing of K. For effi-
ciency consideration, they heuristically set the sampled subgraph size K =4 for
all datasets; (2) they heuristically fix α=1 for all datasets as they found that this
achieves good performance on most datasets (with the help of label information);
and (3) the selection of β is high dependent on the specific dataset. Hence, they “fine-
tune" the value of β for each dataset via selecting β from 0.2,0.4,0.6,0.8,1.0 with
{ }
labels.
Sub-CR
Similar to SL-GAD, Sub-CR (Zhang et al. 2022) also combines the contrastive-based
framework and the generative-based framework for unsupervised GAD.
First, the design of the contrastive-based framework is as follows.
Contrastive Framework—Data Augmentation Module They consider two types of
data augmentation: (1) subgraph sampling to obtain local augmented views for each
node (so-called local view subgraph), (2) graph diffusion plus subgraph sampling (in
a sequential order) to obtain global augmented views for each node (so-called global
view subgraph). Particularly, they employ RWR (Tong et al. 2006) to generate a sub-
graph with a fixed size K in subgraph sampling. Besides, they apply Persnonalized
PageRank to power the graph diffusion (Zhang et al. 2023), wherein the teleport
1 3

Towards automated self-supervised learning for truly unsupervised… Page 27 of 43 44
probability α needs to be determined. Overall, they result in two critical HPs in graph
augmentation, namely K and α.
Contrastive Framework—Contrast Learning Module This module consists
of: (1) intra-view contrastive learning that maximizes the agreement between
the node and its sub-graph level representations in the local view (with loss term
), and the agreement between the node and its sub-graph level representa-
intra,1
L
tions in the global view (with loss term ), where they combine the local view
intra,2
L
and global view loss terms with equal weights to obtain the intra-view loss term
= + ; and (2) inter-view contrastive learning that makes
intra intra,1 intra,2
L L L
closer the discriminative scores of node-subgraph pairs in local view and global
view, leading to the loss term . On this basis, they combine the intra-view loss
inter
L
term and inter-view loss term with equal weights to obtain the multi-view contrastive
learning loss term = + .
con intra inter
L L L
Second, the generative-based framework is designed as follows.
Generative Framework They introduce a masked Autoencoder-based Reconstruc-
tion module, where the encoder is a GCN and the decoder is a multilayer perceptron
with PReLU activation function, aiming to reconstruct the attributes of the target
node based on the attributes of neighboring nodes in the local view (with loss term
), and in the global view (with loss term ). Next, they combine the local
res,1 res,2
L L
view and global view loss terms with equal weights to obtain the overall reconstruc-
tion loss term = + for each node.
res res,1 res,2
L L L
At last, their final optimisation objective is defined as follows:
=
con
+γ
res
,
L L L
where γ (0,1] is the trade-off HP to balance the importance of two different SSL
∈
objectives.
HPs Sensitivity & Tuning They conducted sensitive analysis and found that: (1)
the selection of K is dependent on the specific dataset. However, for efficiency and
performance consideration, they heuristically set the sampled subgraph size K =4
for all datasets; (2) they did not discuss the setting of teleport probability α; and (3)
they claim that most datasets are not sensitive to the value of γ when γ >0.4. Hence,
they heuristically set γ =0.6 for Cora, Citeseer, Flickr, and BlogCatalog while
γ =0.4 for PubMed with the help of label information.
CONAD
Similar to SL-GAD and Sub-CR, CONAD (Xu et al. 2022b) also combines the
contrastive-based framework and the generative-based framework for unsupervised
GAD.
First, the design of the contrastive-based framework is as follows.
Contrastive Framework—Data Augmentation Module They consider four differ-
ent types of data augmentations, with each type of data augmentation operation cor-
responding to a specific type of node anomaly. They include (1) edge adding augmen-
tation that connects a node with many other non-connected nodes (structure - high
degree), (2) edge removing augmentation that removes most edges of a node (struc-
1 3

| 44  Page 28 of 43 |     |     |     |     |     |     | Z. Li et al. |
| ----------------- | --- | --- | --- | --- | --- | --- | ------------ |
ture - outlying); (3) attribute replacement augmentation that replaces the target node’s
attributes with another dissimilar node’s attributes (attribute - deviated), and (4) attri-
bute scaling augmentation that scales the target node’s attributes to much larger or
smaller values (attribute - disproportionate); This leads to four HPs p ,p ,p ,p ,
1 2 3 4
which represent the sampling probability of each augmentation strategy. Moreover,
the rate r of augmented anomalies (namely modified nodes) is also a HP.
Contrastive  Framework—Contrast  Learning  Module  They  con-
sider  two  different  contrast  strategies:  (1)  Siamese  contrast
| =    |      | d(z ,ˆz )+ |     | max | 0,m | d(z ,ˆz )  where d(z | ,ˆz ) is the  |
| ---- | ---- | ---------- | --- | --- | --- | -------------------- | ------------- |
| L SC | i NM | i i        | j   | MM  | {   | − j j }              | i i           |
|      | ∈    |            | ∈   |     |     |                      |               |
distance between embeddings of node i in the original view and in the augmented
|     | ∑   |     | ∑   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
view. MM and NM mean the node is modified or non-modified, respectively; (2)
Triplet contrast  TC = max 0,m [d(z i ,ˆz j ) d(z i ,z j )]  where d(z i ,z j ) is
|     |     | L   |     | { − |     | − } |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
the distance between embeddings of node i and its neighbor j in the original view,
∑
and d(z ,ˆz ) is the distance between embeddings of node i in the original view
|     | i j |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
and its neighbor j in the augmented view. Particularly, the contrastive loss term
|       | =  or  |       | =   | . This module contains a HP, namely the margin m. |     |     |     |
| ----- | ------ | ----- | --- | ------------------------------------------------- | --- | --- | --- |
| Contr | SC     | Contr | TC  |                                                   |     |     |     |
| L     | L      | L     | L   |                                                   |     |     |     |
Second, the generative-based framework is designed as follows.
Generative Framework This framework consists of two components: (1) an attri-
bute autoencoder to reconstruct the node attributes, where the encoder is a GAT
(Veličković et al. 2017) and the decoder is another GAT. This leads to the loss term
; and (2) a structure autoencoder to reconstruct the structure, where the encoder
L A
is a GAT and the decoder is a dot product operation followed by a sigmoid func-
tion (namely sigmoid(ztz)). This leads to the loss term  . Combining these two
L S
loss terms leads to a loss term  =λ +(1 λ) , where λ (0,1) is a
|     |     |     |     | Recon | A   | S   |     |
| --- | --- | --- | --- | ----- | --- | --- | --- |
|     |     |     |     | L     | L   | − L | ∈   |
trade-off HP to balance the two reconstruction errors. Unlike SL-GAD and Sub-CR,
CONAD requires the whole adjacency matrix and node attribute matrix as input, and
thus it can reconstruct the graph structure, making it unsuitable to large graphs. In
contrast, SL-GAD and Sub-CR only require subgraphs as inputs, and thus are unable
to perform structure reconstruction while being scalable.
At last, the final optimization objective is defined as follows:
|     |     |     | =η  | +(1   |     | η) ,    |     |
| --- | --- | --- | --- | ----- | --- | ------- | --- |
|     |     | L   | L   | Contr | −   | L Recon |     |
where η (0,1) is the trade-off HP to balance the importance of two SSL objectives.
∈
HPs Sensitivity & Tuning They did not perform sensitivity analysis over the HPs.
Instead, (1) They heuristically set the ration of augmented anomalies r =0.1 and
r =0.2 for small and large datasets, respectively; (2) The sampling probability of
each augmentation strategy is set to p =0.25 for i 1,2,3,4 ; (3) They heuristi-
|     |     |     |     | i   |     | ∈{ } |     |
| --- | --- | --- | --- | --- | --- | ---- | --- |
cally set the margin m=0.5 for all datasets; and (4) They heuristically set the trade-
| off hyper-parameters λ=0.9 and η |     |     |     | =0.7 for all datasets |     |     |     |
| -------------------------------- | --- | --- | --- | --------------------- | --- | --- | --- |
1 3

Towards automated self-supervised learning for truly unsupervised… Page 29 of 43 44
Fig. 5 Performance variations over different HP configurations for CoLA (Liu et al. 2021) on different
benchmark datasets
Fig. 6 Performance variations over different HP configurations for GRADATE (Duan et al. 2023) on
different benchmark datasets
DOMINANT
DOMINANT (Ding et al. 2019) is arguably the first work that utilizes generative-
based framework and GNNs to perform unsupervised anomaly detection on attribute
graphs.
Generative Framework They first employ GCN (Kipf and Welling 2016) to obtain
node embeddings. Next, they construct two decoders: (1) an attribute decoder, which
consists of another GCN, to reconstruct the node attributes, leading to the loss term
, and (2) a structure decoder, which is a dot product operation followed by a sig-
A
L
moid function (namely sigmoid(ztz)), to reconstruct topological structures, leading
to the loss term .
S
L
At last, their final optimization objective is defined as follows:
1 3

44 Page 30 of 43 Z. Li et al.
Fig. 7 Performance variations over different HP configurations for SL-GAD (Zheng et al. 2021) on
different benchmark datasets
Fig. 8 Performance variations over different HP configurations for Sub-CR (Zhang et al. 2022) on dif-
ferent benchmark datasets
=α
A
+(1 α)
S
,
L L − L
where α (0,1) is the trade-off HP to balance the importance of two objectives.
∈
HPs Sensitivity & Tuning Specifically, they found that the AUC performance usu-
ally increases first and then decreases with the increasing of α. However, the specific
value of α on each dataset is heuristically selected with the help of labels. The HP
α is selected from [0.4, 0.7], [0.4, 0.7], [0.5, 0.8] on BlogCatalog, Flickr, and ACM
respectively.
AnomalyDAE
Similar to DOMINANT, AnomalyDAE (Fan et al. 2020) leverages generative-based
framework and autoencoders (based on GNNs) to perform unsupervised GAD.
1 3

Towards automated self-supervised learning for truly unsupervised… Page 31 of 43 44
Fig. 9 Performance variations over different HP configurations for CONAD (Xu et al. 2022b) on dif-
ferent benchmark datasets
Fig. 10 Performance variations over different HP configurations for DOMINANT (Ding et al. 2019)
on different benchmark datasets
Generative Framework AnomalyDAE consists of two components: (1) an attri-
bute autoencoder to reconstruct the node attributes, where the encoder consists of two
non-linear feature transform layers and the decoder is simply a dot product operation.
This leads to the loss term , and is associated with a penalty HP η >1); and
A A
L L
(2) a structure autoencoder to reconstruct the structures, where the encoder is based
GAT (Veličković et al. 2017) and the decoder is a dot product operation followed by
a sigmoid function (namely sigmoid(ztz)). This leads to the loss term , and is
S S
L L
associated with a penalty HP θ >1.
At last, their final optimization objective is defined as follows:
=α
S
+(1 α)
A
,
L L − L
where α (0,1) is the trade-off HP to balance the importance of two objectives.
∈
1 3

44 Page 32 of 43 Z. Li et al.
40
Fig. 11 Performance variations over different HP configurations for AnomalyDAE (Fan et al. 2020) on
different benchmark datasets
Fig. 12 Performance variations over different HP configurations for GUIDE (Yuan et al. 2021) on dif-
ferent benchmark datasets
HPs Sensitivity & Tuning Specifically, they found that the AUC performance usu-
ally increases first and then decreases with the increasing of α. However, the specific
value of α on each dataset is selected using label information. The HPs (α,η,θ) are
heuristically set as (0.7, 5, 40), (0.9, 8, 90), (0.7, 8, 10) on BlogCatalog, Flickr, and
ACM respectively.
GUIDE
Similar to AnomalyDAE, GUIDE (Yuan et al. 2021) leverages generative-based
framework and autoencoders (based on GNNs) to perform unsupervised GAD. Par-
ticularly, they consider reconstructing the high-order structures.
Generative Framework GUIDE consists of two components: (1) an attribute
autoencoder to reconstruct the node attributes, where the encoder is a GCN and the
1 3

Towards automated self-supervised learning for truly unsupervised… Page 33 of 43 44
Fig. 13 Performance variations over different HP configurations for GAAN (Chen et al. 2020b) on
different benchmark datasets
decoder is another GCN. This leads to the loss term ; and (2) a structure auto-
A
L
encoder to reconstruct the high-order structures, where the encoder is a graph node
attention network based on (Ding et al. 2021) and the decoder is another graph node
attention layer. This leads to the loss term . Moreover, structure matrix is com-
S
L
posed of node motif degrees, which leads to a HP, namely the degree of motifs D.
At last, their final optimization objective is defined as follows:
=α
A
+(1 α)
S
,
L L − L
where α (0,1) is the trade-off HP to balance the importance of two SSL objectives.
∈
HPs Sensitivity & Tuning They mention that the HPs are optimised via a parameter
sensitivity analysis experiment for each dataset. Specifically, they found that: (1) the
AUC performance usually increases first and then decreases with the increasing of α,
and most datasets can achieve a good performance when 0.1<α<0.3. However,
the specific value of α on each dataset is selected using labels; and (2) they heuristi-
cally set the degree of motifs as D =4.
GAAN
GAAN (Chen et al. 2020b) combines the generative-based framework and GAN
(Goodfellow et al. 2014) for unsupervised GAD. Particularly, GAN can be consid-
ered as a special case of contrastive-based framework.
Contrastive Framework—Data Augmentation Module GAAN employs GAN,
which consists of a generator and a discriminator, to generate adversarial samples as
augmented views, without involving any HPs.
Contrastive Framework—Contrastive Learning Module For each target node,
GAAN computes the sum of cross-entropy losses of its 1-hop neighboring nodes
(where the edge is considered as from real distribution by the discriminator) as anom-
aly score, leading to a loss term . In particular, this discriminator loss can be
D
L
regarded as contrastive loss, and it considers both node attributes and graph structures.
1 3

44  Page 34 of 43 Z. Li et al.

| Table 6 SSL-related HPs for  | Algo | HPs Range |
| ---------------------------- | ---- | --------- |
different algorithms, where
|     | ANEMONE (Jin  | K {2, 3, 4, 5} |
| --- | ------------- | -------------- |
“Range" indicates the tested
|     | et al. 2021a) | α {0, 0.01, 0.1, 0.2, 0.3, 0.4,  |
| --- | ------------- | -------------------------------- |
values in grid search
0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1}
|     | AnomalyDAE Fan  | α {0.01, 0.1, 0.2, 0.3, 0.4, 0.5,  |
| --- | --------------- | ---------------------------------- |
et al. (2020) 0.6, 0.7, 0.8, 0.9, 0.99, 1}
η {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
θ {10}
|     | CoLA (Liu et al.  | K {2, 3, 4, 5} |
| --- | ----------------- | -------------- |
2021)
|     | CONAD (Zhang  | r {0.10}  |
| --- | ------------- | --------- |
|     | et al. 2022)  | p1 {0.25} |
p2 {0.25}
p3 {0.25}
p4 {0.25}
m {0.5}
λ {0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.9, 0.99, 1}
η
{0.01, 0.5, 0.99, 1}
|     | DOMINANT (Ding  | α {0.01, 0.1, 0.2, 0.3, 0.4, 0.5,  |
| --- | --------------- | ---------------------------------- |
et al. 2019) 0.6, 0.7, 0.8, 0.9, 0.99, 1}
|     | GAAN (Chen et al.  | α {0, 0.01, 0.1, 0.2, 0.3, 0.4,  |
| --- | ------------------ | -------------------------------- |
2020b) 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1}
|     | GRADATE (Duan  | P {0.20} |
| --- | -------------- | -------- |
|     | et al. 2023)   | α {0.9}  |
β
{0, 0.01, 0.1, 0.2, 0.3, 0.4,
0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1}
γ {0, 0.01, 0.1, 0.2, 0.3, 0.4,
0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1}
|     | GUIDE (Yuan et al.  | D {4}                              |
| --- | ------------------- | ---------------------------------- |
|     | 2021)               | α {0.01, 0.1, 0.2, 0.3, 0.4, 0.5,  |
0.6, 0.7, 0.8, 0.9, 0.99}
|     | SL-GAD (Zheng  | K {2, 3, 4, 5, 6, 7, 8, 9}         |
| --- | -------------- | ---------------------------------- |
|     | et al. 2021)   | α {0.01, 0.1, 0.2, 0.3, 0.4, 0.5,  |
0.6, 0.7, 0.8, 0.9, 0.99, 1}
β {0.6}
|     | Sub-CR (Zhang  | K {2, 3, 4, 5, 6, 7, 8, 9} |
| --- | -------------- | -------------------------- |
|     | et al. 2022)   | α {0.01}                   |
γ {0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.9, 0.99, 1}
Generative Framework GAAN utilizes the generator to reconstruct the node attri-
bute, and employs the reconstruction error to compute anomaly score, leading to a
loss term  .
G
L
At last, their final optimisation objective is defined as
|     | =α +(1 | α) ,  |
| --- | ------ | ----- |
|     | L L G  | − L D |
where α [0,1] is the trade-off HP to balance the importance of two objectives
∈
1 3

Towards automated self-supervised learning for truly unsupervised… Page 35 of 43  44
Table 7 Summary of existing SSL-based graph anomaly detection methods
| Method | Venue Datasets | SSL Methods | Hyperparameters | Code |
| ------ | -------------- | ----------- | --------------- | ---- |
ANEMONE  CIKM’21 Cora, Citeseer,  Node-Node CL,  Ego-Net size (K),  Github
| (Jin et al.  | PubMed | Node-Sub CL | Combination  |     |
| ------------ | ------ | ----------- | ------------ | --- |
| 2021a)       |        |             | weights      |     |
AnomalyDAE  ICASSP’20 ACM, Flickr,  Attribute Recon,  Penalty HPs, Com- PyGOD
| (Fan et al.  | BlogCatalog | Structure Recon | bination weights |     |
| ------------ | ----------- | --------------- | ---------------- | --- |
2020)
CoLA (Liu  TNNLS’21 Cora, Citeseer,  Node-Sub CL Random walk length  Github,
| et al. 2021) | Pubmed, BlogCata- |     | (K) | PyGOD |
| ------------ | ----------------- | --- | --- | ----- |
log, Flickr, ACM,
ogbn-arxiv
CONAD (Xu  PAKDD’22 Amazon, Flickr,  Node-Sub CL,  Augmentation sam- PyGOD,
et al. 2022b) Enron, Facebook,  Attribute Recon,  pling probabilities,  Github
|     | Twitter | Structure Recon | combination weights |     |
| --- | ------- | --------------- | ------------------- | --- |
DOMINANT  ICDM’19 ACM, Flickr,  Attribute Recon,  Combination weight PyGOD
| (Ding et al.  | BlogCatalog | Structure Recon |     |     |
| ------------- | ----------- | --------------- | --- | --- |
2019)
GAAN (Chen  CIKM’20 ACM, Flickr,  Attribute Recon,  Combination weight PyGOD
| et al. 2020b) | BlogCatalog | Discr Loss |     |     |
| ------------- | ----------- | ---------- | --- | --- |
GRADATE  AAAI’23 EAT, WebKB, UAT,  Node-Node CL,  Proportion of modi- Github
(Duan et al.  Cora, UAI2010,  Node-Sub CL,  fied edges (P), Com-
| 2023) | Citation | Sub-Sub CL | bination weights |     |
| ----- | -------- | ---------- | ---------------- | --- |
GUIDE (Yuan  BigData’21 Cora, Citation,  Attribute Recon,  Combination weight PyGOD
| et al. 2021) | PubMed, ACM,  | Structure Recon |     |     |
| ------------ | ------------- | --------------- | --- | --- |
DBLP
SL-GAD  TKDE’21 Cora, Citeseer,  Node-Sub  Random walk length  Github
(Zheng et al.  PubMed, ACM,  CL,    Attribute  (K), Combination
| 2021) | Flickr, BlogCatalog | Recon | weights |     |
| ----- | ------------------- | ----- | ------- | --- |
Sub-CR  IJCAI’22 Cora, Citeseer,  Node-Sub  Random walk length  Github
(Zhang et al.  PubMed, Flickr,  CL,    Attribute  (K), Teleport prob-
| 2022) | BlogCatalog | Recon | ability α, Combina- |     |
| ----- | ----------- | ----- | ------------------- | --- |
tion weights

Fig. 14 Performance of AutoGAD in selecting heterogeneous anomaly detectors on selected datasets
(results on other datasets are similar and thus omitted)
1 3

44 Page 36 of 43 Z. Li et al.
HPs Sensitivity & Tuning Specifically, they found that the AUC performance usu-
ally increases first and then decreases with the increasing of α. However, the specific
value of α on each dataset is selected using label information. The HP α is heuristi-
cally set as 0.2, 0.3, 0.1 on BlogCatalog, Flickr, and ACM respectively.
Performance variations under different HP settings
In this section, we present a comprehensive analysis of the performance exhibited by
various semi-supervised learning (SSL) based graph anomaly detection techniques.
This evaluation encompasses an extensive array of hyperparameter (HP) configura-
tions and is conducted across multiple benchmark datasets.
Specifically, the results for GAAN (Chen et al. 2020b) is provided in Fig. 13,
from which we can see huge performance variations under different HP settings. For
example, the AUC value can vary from 0.474 to 0.747 if one utilizes different HP
configurations on dataset CiteSeer (namely by changing the HP α from 0.5 to 0).
Moreover, the results for CoLA (Liu et al. 2021) is provided in Fig. 5. Compared to
GAAN, CoLA is less sensitive to the setting of HPs, while we can still see moderate
performance variations on some datasets (e.g., from 0.693 to 0.733 on Flickr, and
from 0.767 to 0.795 on ACM). Besides, Fig. 10 shows that DOMINANT is also sen-
sitive to HPs except for the cases where the algorithm is largely underfitted (i.e., on
ACM, Flickr and BlogCatalog the loss values change only by 10 − 2 after 400 epochs
of training).
Particularly, AnomalyDAE and SL-GAD are very sensitive to HPs as shown in
Figs. 11 and 7. For example, the performance of AnomalyDAE ranges from 0.702
to 0.941 on CiteSeer, and the performance of SL-GAD vary from 0.787 to 0.920.
As shown in Fig. 9, CONAD shows similar behaviors except for the cases where
CONAD is largely underfitted (namely on ACM) or suffers from OOM errors
(namely on Flickr and BlogCatalog). The analysis for GUIDE in Fig. 12, GRADATE
in Fig. 6, and Sub-CR in Fig. 8 is similar and conveys the same issues (Figs. 5, 6, 7,
8, 9, 10, 11).
Similar observations in other papers
Liu et al. (2022b) conduct a comprehensive benchmark for unsupervised graph
anomaly detection. From their results (note that their experiment setting is slightly
different from ours), we can have similar observations as follows by comparing the
average AUC vs max AUC (Figs. 12 and 13):
● Radar (Li et al. 2017a) is not sensitive to hyper-parameters (0.65 VS 0.66 on
Cora, 0.99 VS 0.99 on Weibo, 0.55 VS 0.57 on Reddit, 0.52 VS 0.52 on Disney,
0.53 VS 0.53 on Books), but it will suffer from OOM errors for large graphs;
● ANOMALOUS (Peng et al. 2018) is very sensitive to hyper-parameters on some
datasets (0.55 VS 0.68 on Cora, 0.99 VS 0.99 on Weibo, 0.55 VS 0.60 on Reddit,
0.52 VS 0.52 on Disney, 0.53 VS 0.53 on Books), and it will suffer from OOM
1 3

Towards automated self-supervised learning for truly unsupervised… Page 37 of 43 44
errors for large graphs;
● DOMINANT (Ding et al. 2019) is very sensitive to hyper-parameters on some
datasets (0.83 VS 0.84 on Cora, 0.76 VS 0.85 on Flickr, 0.85 VS 0.93 on Weibo,
0.50 VS 0.58 on Books, 0.56 VS 0.56 on Reddit, 0.47 VS 0.55 on Disney)
● AnomalyDAE (Fan et al. 2020) is very sensitive to hyper-parameters on some
datasets (0.83 VS 0.85 on Cora, 0.86 VS 0.91 on Amazon, 0.66 VS 0.70 on Flickr,
0.91 VS 0.93 on Weibo, 0.56 VS 0.56 on Reddit, 0.49 VS 0.55 on Disney, 0.54
VS 0.69 on Books);
● GUIDE (Yuan et al. 2021) is very sensitive to hyper-parameters on some datasets
(0.39 VS 0.53 on Disney, 0.52 VS 0.63 on Books, 0.75 VS 0.78 on Cora), and it
will suffer from OOM errors on large graph (including Amazon, Flickr, Weibo,
Reddit). It needs much time and memory for training as it employs a graph motif
counting algorithm to extract structural information;
● CONAD (Xu et al. 2022b) is very sensitive to hyper-parameters on some datasets
(0.79 VS 0.84 on Cora, 0.81 VS 0.82 on Amazon, 0.65 VS 0.67 on Flickr, 0.85
VS 0.93 on Weibo, 0.56 VS 0.56 on Reddit, 0.48 VS 0.53 on Disney, 0.52 VS
0.63 on Books).
Summary of existing SSL-based graph anomaly detection methods
Existing SSL-based graph anomaly detection methods are summarized in Table 7,
which includes the datasets used to test, the core principles of SSL techniques, the
involved hyper-parameters (only SSL related ones), and their public implementations.
Search space approximation based on SMBO
Performance surrogate functions
Although discretization of continuous domains can largely reduce the search space,
it is still computationally prohibitive to search the full discretized HP space when the
number of HPs is large. Therefore, we learn a regressor g() which aims to to learn
·
the mapping from HP settings onto the performance metric (namely the domain of
T()). Note that g() should be different for different combinations of graph and graph
· ·
anomaly detector [ ,f()], and we call these functions performance surrogate func-
G ·
tions. Gaussian Process (GP) (Williams and Rasmussen 1995) is one popular choice
for g(). Based on these performance surrogate functions, we can identify promising
·
HPs without running experiments on all possible HPs, which will be illustrated in
next subsection.
SMBO-based optimization
Particularly, we leverage Sequential Model-based Optimization (SMBO) (Jones et al.
1998) to iteratively and efficiently identify promising HP configurations to evaluate,
1 3

| 44  Page 38 of 43 |     |     |     |     |     |     |     | Z. Li et al. |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | ------------ |
and finally output the optimal one as follows. Similar idea is also explored in Zhao
et al. (2022b).
Initialization Specifically, we first randomly sample a small number of HPs
λ = λ ,λ ,...,λ  with J M. Second, for each HP, we compute its unsu-
| eval { | 1 2 | J   | }   | ≪   |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- |
pervised performance metric score t( ), leading to pairs  (λ ,t ( )),(λ ,t ( )),...,
|     |     |     |     |     |     | 1 1 | 2   | 2   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | G   |     | {   | G   | G   |
(λ J ,t J ( )) . Third, we employ these pairs to train a specific performance surrogate
| G   | }   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
function g().
·
Iteration For each iteration, we leverage g() to predict the performance for a
·
sampled HP λ , denoted as η =g(λ ). Moreover, we also utilize g() to predict
|     | j   |     | j   | j   |     |     | ·   |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
the uncertainty around the prediction of λ , denoted as σ =σ[g(λ λ λ )].
|     |     |     |     | j   |     | j   | l l | sample |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ |
| ∈
Note that λ sample  is different from λ eval , and it is a finite number of HPs that is
randomly sampled from the full HP space before discretization. Next, we utilize a
so-called acquisition function h(), which can make a trade-off between predicted
·
performance and uncertainty, to select the most promising HP to evaluate. Particu-
larly, we leverage Expected Improvement (EI) (Jones et al. 1998) as the acquisition
function since it has shown prominent performances in many studies (Zhao and Ako-
glu 2022). Under the mild Gaussian assumption, the EI value of HP setting λ  has the
j
following closed-form expression:
|     |     | EI(g(λ | ))=[ϕ(ηˆ | )+ηˆ | Φ(ηˆ | )]σ ,  |     | (8) |
| --- | --- | ------ | -------- | ---- | ---- | ------ | --- | --- |
|     |     |        | j        | j    | j ·  | j j    |     |     |
ηj− η e∗val if σ
where ηˆ j = j >0 and ηˆ j =0 otherwise. Moreover, ϕ( ) and Φ( ) denote
|     | σ   | j   |     |     |     |     | ·   | ·   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
the probability density function and the cumulative distribution function of standard
Gaussian distribution, respectively. In addition, η  is the highest prediction perfor-
e∗val
mance on λ eval  so far. For each iteration, the most promising HP can be obtained as
follows:
|     |     |     | λ = | argmax | h(g(λ | )), |     |     |
| --- | --- | --- | --- | ------ | ----- | --- | --- | --- |
|     |     |     | ∗   |        |       | j   |     | (9) |
λj∈ λsample

where g()=g(current)() is the surrogate function in the current iteration, which
| ·   |     |     | ·   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
can output the most promising HP λ ∗ to evaluate. On this basis, we apply f(λ )
∗
 to obtain a vector of anomaly scores s ∗, followed by inputting s
| on graph  |     |     |     |     |     |     |     | ∗   |
| --------- | --- | --- | --- | --- | --- | --- | --- | --- |
G
into Eq. 3 to obtain the performance metric score t ∗. At last, we update the eval-
uation  HP  set  as  λ =λ λ ∗,  and  retrain  g()  with  the  updated  pairs
|     |     | eval | eval |     |     |     |     |     |
| --- | --- | ---- | ---- | --- | --- | --- | --- | --- |
|     |     |      |      | ∪   |     | ·   |     |     |
(λ 1 ,t 1 ( )),(λ 2 ,t 2 ( )),...,  (λ J ,t J ( )) ...,(λ ∗ ,t ∗ ) . Additionally,  we  update
| { G                    |     | G   |      | G } |     | }   |     |     |
| ---------------------- | --- | --- | ---- | --- | --- | --- | --- | --- |
| η  using the updated λ |     |     | .    |     |     |     |     |     |
| e∗val                  |     |     | eval |     |     |     |     |     |
AutoGAD for selecting heterogeneous anomaly detectors
To evaluate the effectiveness of AutoGAD in selecting heterogeneous anomaly
detectors, we compute the Pearson Correlation Coefficient between the highest
improved CSM scores (based on Eq. 3) and the corresponding AUC scores for all
anomaly detectors on each individual dataset.
1 3

Towards automated self-supervised learning for truly unsupervised… Page 39 of 43 44
As shown in Fig. 14, the results reveal that AutoGAD’s CSM score does not effec-
tively predict the true performance (AUC) of heterogeneous anomaly detectors. Spe-
cifically, on the Cora dataset, the Pearson correlation is very weak (0.070), indicating
almost no relationship between the CSM score and AUC. On the Amazon dataset, the
correlation is negative ( 0.488), suggesting that higher CSM scores are, in fact, asso-
−
ciated with lower AUC values in many cases. This weak or inverse correlation dem-
onstrates that AutoGAD’s scoring mechanism may not be suitable for selecting the
best-performing anomaly detectors, as it fails to consistently align with true detector
performance. Notably, detectors such as SL-GAD, which achieve high AUC, do not
consistently receive high CSM scores, further underscoring the discrepancy. In sum-
mary, these findings suggest that AutoGAD’s current approach to ranking anomaly
detectors is unreliable and may require significant revisions to improve its predictive
accuracy.
Acknowledgements Zhong Li and Matthijs van Leeuwen: this publication is part of the project Digital
Twin with project number P18-03 of the research programme TTW Perspective, which is (partly) financed
by the Dutch Research Council (NWO).
Author contributions Zhong Li: Conceptualization, Methodology, Validation, Investigation, Software,
Writing, Visualisation, Project Administration. Yuhang Wang: Methodology, Investigation, Software,
Writing. Matthijs van Leeuwen: Methodology, Validation, Writing, Funding acquisition, Supervision.
Funding This work is supported by Project 4 of the Digital Twin research programme, a TTW Perspectief
programme with project number P18-03 that is primarily financed by the Dutch Research Council (NWO).
All opinions, findings, conclusions and recommendations in this paper are those of the authors and do not
necessarily reflect the views of the funding agencies.
Data availability For reproducibility, all code and datasets are provided online via the following link:
https://github.com/ZhongLIFR/AutoGAD2024.
Declarations
Conflict of interest The author(s) declared no potential Conflict of interest with respect to the research,
authorship and/or publication of this article.
Ethical approval This study does not involve human and animal data, and thus the need for approval was
waived.
Open Access This article is licensed under a Creative Commons Attribution 4.0 International License,
which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long
as you give appropriate credit to the original author(s) and the source, provide a link to the Creative
Commons licence, and indicate if changes were made. The images or other third party material in this
article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line
to the material. If material is not included in the article’s Creative Commons licence and your intended use
is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission
directly from the copyright holder. To view a copy of this licence, visit h t t p : / / c r e a t i v e c o m m o n s . o r g / l i c e n
s e s / b y / 4 . 0 / .
1 3

44 Page 40 of 43 Z. Li et al.
References
Akoglu L, Tong H, Koutra D (2015) Graph based anomaly detection and description: a survey. Data Min
Knowl Disc 29:626–688
Bahri M, Salutari F, Putina A et al (2022) Automl: state of the art with a focus on anomaly detection, chal-
lenges, and research directions. Int J Data Sci Anal 14(2):113–126
Bischl B, Binder M, Lang M et al (2023) Hyperparameter optimization: foundations, algorithms, best
practices, and open challenges. Wiley Interdiscip Rev Data Min Knowl Discov 13(2):e1484
Breunig MM, Kriegel HP, Ng RT, et al (2000) Lof: identifying density-based local outliers. In: Proceed-
ings of the 2000 ACM SIGMOD international conference on Management of data, pp 93–104
Chen B, Zhang J, Zhang X, et al (2022) Gccad: graph contrastive learning for anomaly detection. IEEE
Trans Knowl Data Eng
Chen T, Kornblith S, Norouzi M, et al (2020a) A simple framework for contrastive learning of visual rep-
resentations. In: International conference on machine learning, PMLR, pp 1597–1607
Chen Z, Liu B, Wang M, et al (2020b) Generative adversarial attributed network anomaly detection. In:
Proceedings of the 29th ACM international conference on information & knowledge management,
pp 1989–1992
Cubuk ED, Zoph B, Mane D, et al (2018) Autoaugment: Learning augmentation policies from data. arXiv
preprint arXiv:1805.09501
Cubuk ED, Zoph B, Shlens J, et al (2020) Randaugment: practical automated data augmentation with a
reduced search space. In: Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition workshops, pp 702–703
Ding K, Li J, Bhanushali R, et al (2019) Deep anomaly detection on attributed networks. In: Proceedings
of the 2019 SIAM international conference on data mining, SIAM, pp 594–602
Ding K, Li J, Agarwal N, et al (2021) Inductive anomaly detection on attributed networks. In: Proceedings
of the twenty-ninth international conference on international joint conferences on artificial intel-
ligence, pp 1288–1294
Ding X, Zhao L, Akoglu L (2022) Hyperparameter sensitivity in deep outlier detection: analysis and a
scalable hyper-ensemble solution. Adv Neural Inf Process Syst 35:9603–9616
Duan J, Wang S, Zhang P, et al (2023) Graph anomaly detection via multi-scale contrastive learning net-
works with augmented view. In: Proceedings of the AAAI conference on artificial intelligence, pp
7459–7467
Duan S, Matthey L, Saraiva A, et al (2019) Unsupervised model selection for variational disentangled
representation learning. arXiv preprint arXiv:1905.12614
Fan H, Zhang F, Li Z (2020) Anomalydae: Dual autoencoder for anomaly detection on attributed networks.
In: ICASSP 2020–2020 IEEE international conference on acoustics. IEEE, Speech and Signal Pro-
cessing (ICASSP), pp 5685–5689
Fey M, Lenssen JE (2019) Fast graph representation learning with pytorch geometric. arXiv preprint
arXiv:1903.02428
Garcia-Teodoro P, Diaz-Verdejo J, Maciá-Fernández G et al (2009) Anomaly-based network intrusion
detection: techniques, systems and challenges. Comput Secur 28(1–2):18–28
Goix N (2016) How to evaluate the quality of unsupervised anomaly detection algorithms? arXiv preprint
arXiv:1607.01152
Goodfellow I, Pouget-Abadie J, Mirza M, et al (2014) Generative adversarial nets. Adv Neural Inf Process
Syst 27
Hanley JA, McNeil BJ (1982) The meaning and use of the area under a receiver operating characteristic
(roc) curve. Radiology 143(1):29–36
Hassani K, Khasahmadi AH (2022) Learning graph augmentations to learn graph representations. arXiv
preprint arXiv:2201.09830
Hataya R, Zdenek J, Yoshizoe K, et al (2020) Faster autoaugment: learning augmentation strategies using
backpropagation. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK,
August 23–28, 2020, Proceedings, Part XXV 16, Springer, pp 1–16
Ho D, Liang E, Chen X, et al (2019) Population based augmentation: efficient learning of augmentation
policy schedules. In: International conference on machine learning, PMLR, pp 2731–2741
Jiao Y, Xiong Y, Zhang J, et al (2020) Sub-graph contrast for scalable self-supervised graph representation
learning. In: 2020 IEEE international conference on data mining (ICDM), IEEE, pp 222–231
1 3

Towards automated self-supervised learning for truly unsupervised… Page 41 of 43 44
Jin M, Liu Y, Zheng Y, et al (2021a) Anemone: graph anomaly detection with multi-scale contrastive
learning. In: Proceedings of the 30th ACM international conference on information & knowledge
management, pp 3122–3126
Jin W, Liu X, Zhao X, et al (2021b) Automated self-supervised learning for graphs. arXiv preprint
arXiv:2106.05470
Jones DR, Schonlau M, Welch WJ (1998) Efficient global optimization of expensive black-box functions.
J Global Optim 13:455–492
Kaufman S, Rosset S, Perlich C et al (2012) Leakage in data mining: formulation, detection, and avoid-
ance. ACM Trans Knowl Discov Data (TKDD) 6(4):1–21
Kim H, Lee BS, Shin WY, et al (2022) Graph anomaly detection with graph neural networks: Current
status and challenges. IEEE Access
Kipf TN, Welling M (2016) Semi-supervised classification with graph convolutional networks. arXiv pre-
print arXiv:1609.02907
Kleinberg JM (1999) Authoritative sources in a hyperlinked environment. J ACM (JACM) 46(5):604–632
Kumar S, Zhang X, Leskovec J (2019) Predicting dynamic embedding trajectory in temporal interaction
networks. In: Proceedings of the 25th ACM SIGKDD international conference on knowledge discov-
ery & data mining, pp 1269–1278
Lai KH, Zha D, Wang G, et al (2021) Tods: An automated time series outlier detection system. In: Proceed-
ings of the aaai conference on artificial intelligence, pp 16060–16062
Leskovec J, Mcauley J (2012) Learning to discover social circles in ego networks. Adv Neural Inf Process
Syst 25
Li J, Dani H, Hu X, et al (2017a) Radar: Residual analysis for anomaly detection in attributed networks.
In: IJCAI, pp 2152–2158
Li Y, Shang Y, Yang Y (2017) Clustering coefficients of large networks. Inf Sci 382:350–358
Li Y, Hu G, Wang Y, et al (2020a) Differentiable automatic data augmentation. In: Computer Vision–
ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII
16, Springer, pp 580–595
Li Y, Zha D, Venugopal P et al (2020) Pyodds: an end-to-end outlier detection system with automated
machine learning. Compan Proc Web Conf 2020:153–157
Li Y, Chen Z, Zha D et al (2021) Automated anomaly detection via curiosity-guided search and self-
imitation learning. IEEE Trans Neural Netw Learn Syst 33(6):2365–2377
Li Y, Chen Z, Zha D, et al (2021b) Autood: Neural architecture search for outlier detection. In: 2021 IEEE
37th international conference on data engineering (ICDE), IEEE, pp 2117–2122
Li Z, Liang S, Shi J et al (2024) Cross-domain graph level anomaly detection. IEEE Trans Knowl Data
Eng 36(12):7839–7850
Li Z, Shi J, Van Leeuwen M (2024b) Graph neural networks based log anomaly detection and explana-
tion. In: Proceedings of the 2024 IEEE/ACM 46th international conference on software engineering:
companion proceedings, pp 306–307
Lim S, Kim I, Kim T, et al (2019) Fast autoaugment. Adv Neural Inf Process Syst 32
Lin Z, Thekumparampil K, Fanti G, et al (2020) Infogan-cr and modelcentrality: self-supervised model
training and selection for disentangling gans. In: International conference on machine learning,
PMLR, pp 6127–6139
Liu F, Ma X, Wu J, et al (2022a) Dagad: Data augmentation for graph anomaly detection. In: 2022 IEEE
international conference on data mining (ICDM), IEEE, pp 259–268
Liu K, Dou Y, Zhao Y et al (2022) Bond: benchmarking unsupervised outlier node detection on static
attributed graphs. Adv Neural Inf Process Syst 35:27021–27035
Liu Y, Li Z, Pan S et al (2021) Anomaly detection on attributed networks via contrastive self-supervised
learning. IEEE Trans Neural Netwo Learn Syst 33(6):2378–2392
Liu Y, Jin M, Pan S et al (2022) Graph self-supervised learning: a survey. IEEE Trans Knowl Data Eng
35(6):5879–5900
Liu Z, Cao C, Tao F, et al (2023) Revisiting graph contrastive learning for anomaly detection. arXiv pre-
print arXiv:2305.02496
Luo Y, McThrow M, Au WY, et al (2022) Automated data augmentations for graph classification. arXiv
preprint arXiv:2202.13248
Ma MQ, Zhao Y, Zhang X, et al (2023) The need for unsupervised outlier model selection: a review and
evaluation of internal evaluation strategies. ACM SIGKDD Explor Newslett 25(1)
1 3

44 Page 42 of 43 Z. Li et al.
Ma R, Pang G, Chen L, et al (2022) Deep graph-level anomaly detection by glocal knowledge distillation.
In: Proceedings of the fifteenth ACM international conference on web search and data mining, pp
704–714
Ma X, Wu J, Xue S, et al (2021) A comprehensive survey on graph anomaly detection with deep learning.
IEEE Trans Knowl Data Eng
Marques HO, Campello RJ, Zimek A, et al (2015) On the internal evaluation of unsupervised outlier
detection. In: Proceedings of the 27th international conference on scientific and statistical database
management, pp 1–12
Marques HO, Campello RJ, Sander J et al (2020) Internal evaluation of unsupervised outlier detection.
ACM Trans Knowl Discov Data (TKDD) 14(4):1–42
Motie S, Raahemi B (2023) Financial fraud detection using graph neural networks: a systematic review.
Expert Syst Appl 122156
Nguyen TT, Nguyen UQ et al (2016) An evaluation method for unsupervised anomaly detection algo-
rithms. J Comput Sci Cybern 32(3):259–272
Nisbet R, Elder J, Miner GD (2009) Handbook of statistical analysis and data mining applications. Aca-
demic press, Cambridge
Paszke A, Gross S, Massa F, et al (2019) Pytorch: an imperative style, high-performance deep learning
library. Adv Neural Inf Process Syst 32
Peng Z, Luo M, Li J, et al (2018) Anomalous: a joint modeling approach for anomaly detection on attrib-
uted networks. In: IJCAI, pp 3513–3519
Perozzi B, Akoglu L (2016) Scalable anomaly ranking of attributed neighborhoods. In: Proceedings of the
2016 SIAM international conference on data mining, SIAM, pp 207–215
Putina A, Bahri M, Salutari F, et al (2022) Autoad: an automated framework for unsupervised anomaly
detectio. In: 2022 IEEE 9th international conference on data science and advanced analytics (DSAA).
IEEE, pp 1–10
Qiao H, Pang G (2024) Truncated affinity maximization: one-class homophily modeling for graph anom-
aly detection. Adv Neural Inf Process Syst 36
Ratner AJ, Ehrenberg H, Hussain Z, et al (2017) Learning to compose domain-specific transformations for
data augmentation. Adv Neural Inf Process Syst 30
Rayana S, Akoglu L (2015) Collective opinion spam detection: Bridging review networks and metadata.
In: Proceedings of the 21th acm sigkdd international conference on knowledge discovery and data
mining, pp 985–994
Sánchez PI, Müller E, Laforet F, et al (2013) Statistical selection of congruent subspaces for mining attrib-
uted graphs. In: 2013 IEEE 13th international conference on data mining, IEEE, pp 647–656
Sen P, Namata G, Bilgic M et al (2008) Collective classification in network data. AI Mag 29(3):93–93
Sun J, Wang B, Wu B (2021) Automated graph representation learning for node classification. In: 2021
international joint conference on neural networks (IJCNN). IEEE, pp 1–7
Suresh S, Li P, Hao C et al (2021) Adversarial graph augmentation to improve graph contrastive learning.
Adv Neural Inf Process Syst 34:15920–15933
Tang J, Zhang J, Yao L, et al (2008) Arnetminer: extraction and mining of academic social networks. In:
Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data
mining, pp 990–998
Tong H, Faloutsos C, Pan JY (2006) Fast random walk with restart and its applications. In: Sixth interna-
tional conference on data mining (ICDM’06). IEEE, pp 613–622
Veličković P, Cucurull G, Casanova A, et al (2017) Graph attention networks. arXiv preprint
arXiv:1710.10903
Wenzel F, Snoek J, Tran D et al (2020) Hyperparameter ensembles for robustness and uncertainty quanti-
fication. Adv Neural Inf Process Syst 33:6514–6527
Williams C, Rasmussen C (1995) Gaussian processes for regression. Adv Neural Inf Process Syst 8
Wu L, Lin H, Tan C, et al (2021) Self-supervised learning on graphs: contrastive, generative, or predictive.
IEEE Trans Knowl Data Eng
Xie XL, Beni G (1991) A validity measure for fuzzy clustering. IEEE Trans Pattern Anal Mach Intell
13(08):841–847
Xie Y, Xu Z, Zhang J et al (2022) Self-supervised learning of graph neural networks: a unified review.
IEEE Trans Pattern Anal Mach Intell 45(2):2412–2429
Xu W, Wu J, Liu Q et al (2022) Evidence-aware fake news detection with graph neural networks. Proc
ACM Web Conf 2022:2501–2510
1 3

Towards automated self-supervised learning for truly unsupervised… Page 43 of 43 44
Xu Z, Kakde D, Chaudhuri A (2019) Automatic hyperparameter tuning method for local outlier factor,
with applications to anomaly detection. In: 2019 IEEE international conference on big data (big
data). IEEE, pp 4201–4207
Xu Z, Huang X, Zhao Y, et al (2022b) Contrastive attributed network anomaly detection with data augmen-
tation. In: Pacific-Asia conference on knowledge discovery and data mining. Springer, pp 444–457
Yang L, Shami A (2020) On hyperparameter optimization of machine learning algorithms: theory and
practice. Neurocomputing 415:295–316
Yin Y, Wang Q, Huang S, et al (2022) Autogcl: Automated graph contrastive learning via learnable view
generators. In: Proceedings of the AAAI conference on artificial intelligence, pp 8892–8900
Yoo J, Zhao Y, Zhao L, et al (2023) Dsv: An alignment validation loss for self-supervised outlier model
selection. In: Joint European conference on machine learning and knowledge discovery in databases.
Springer, pp 254–269
You Y, Chen T, Sui Y et al (2020) Graph contrastive learning with augmentations. Adv Neural Inf Process
Syst 33:5812–5823
You Y, Chen T, Shen Y, et al (2021) Graph contrastive learning automated. In: International conference on
machine learning. PMLR, pp 12121–12132
Yuan X, Zhou N, Yu S, et al (2021) Higher-order structure based anomaly detection on attributed networks.
In: 2021 IEEE international conference on big data (Big Data). IEEE, pp 2691–2700
Yue H, Zhang C, Zhang C et al (2022) Label-invariant augmentation for semi-supervised graph classifica-
tion. Adv Neural Inf Process Syst 35:29350–29361
Zeng H, Zhou H, Srivastava A, et al (2019) Graphsaint: Graph sampling based inductive learning method.
arXiv preprint arXiv:1907.04931
Zeng J, Xie P (2021) Contrastive self-supervised learning for graph classification. In: Proceedings of the
AAAI conference on Artificial Intelligence, pp 10824–10832
Zha D, Lai KH, Wan M, et al (2020) Meta-aad: active anomaly detection with deep reinforcement learning.
In: 2020 IEEE international conference on data mining (ICDM). IEEE, pp 771–780
Zhang J, Wang S, Chen S (2022) Reconstruction enhanced multi-view contrastive learning for anomaly
detection on attributed networks. arXiv preprint arXiv:2205.04816
Zhang M, Qamar M, Kang T, et al (2023) A survey on graph diffusion models: Generative ai in science for
molecule, protein and material. arXiv preprint arXiv:2304.01565
Zhang X, Wang Q, Zhang J, et al (2019) Adversarial autoaugment. arXiv preprint arXiv:1912.11188
Zhao T, Liu Y, Neves L, et al (2021a) Data augmentation for graph neural networks. In: Proceedings of the
aaai conference on artificial intelligence, pp 11015–11023
Zhao T, Tang X, Zhang D, et al (2022a) Autogda: automated graph data augmentation for node classifica-
tion. In: Learning on graphs conference, PMLR, pp 32–1
Zhao Y, Akoglu L (2022) Towards unsupervised hpo for outlier detection. arXiv preprint arXiv:2208.11727
Zhao Y, Akoglu L (2024) Hpod: Hyperparameter optimization for unsupervised outlier detection. In:
AutoML 2024 Methods Track
Zhao Y, Nasrullah Z, Hryniewicki MK, et al (2019) Lscp: locally selective combination in parallel outlier
ensembles. In: Proceedings of the 2019 SIAM international conference on data mining. SIAM, pp
585–593
Zhao Y, Rossi RA, Akoglu L (2020) Automating outlier detection via meta-learning. arXiv preprint
arXiv:2009.10606
Zhao Y, Rossi R, Akoglu L (2021) Automatic unsupervised outlier model selection. Adv Neural Inf Pro-
cess Syst 34:4489–4502
Zhao Y, Zhang S, Akoglu L (2022b) Toward unsupervised outlier model selection. In: 2022 IEEE interna-
tional conference on data mining (ICDM). IEEE, pp 773–782
Zheng Y, Jin M, Liu Y, et al (2021) Generative and contrastive self-supervised learning for graph anomaly
detection. IEEE Trans Knowl Data Eng
Zhou C, Paffenroth RC (2017) Anomaly detection with robust deep autoencoders. In: Proceedings of the
23rd ACM SIGKDD international conference on knowledge discovery and data mining, pp 665–674
Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims in published maps
and institutional affiliations.
1 3