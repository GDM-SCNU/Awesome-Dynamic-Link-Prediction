<h1 align="center"> Awesome-Dynamic-Link-Prediction</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

</h5>

# Introduction

Dynamic graphs have recently become a prominent topic in graph learning. Link prediction in dynamic graphs, or dynamic link prediction (DLP), is particularly well-suited for real-world applications, but it also presents significant challenges. DLP focuses on capturing temporal changes in graph features and predicting future link states in graph structures. Due to its practical relevance, DLP has garnered increasing attention, with numerous methods—especially those leveraging deep learning techniques—being continuously proposed. This repository compiles recent advancements in DLP methods and related resources.



  

## 1. Related Survey

* (_2020._)  **Temporal Link Prediction: A Survey** [[Paper](https://link.springer.com/article/10.1007/s00354-019-00065-z)]
* (_2020._) **Link Prediction in Dynamic Social Networks: A Literature Review** [[Paper](https://ieeexplore.ieee.org/abstract/document/8596511)]
* (_2023._)  **A Survey of Dynamic Network Link Prediction** [[Paper](https://ieeexplore.ieee.org/document/10297326)]
* (_2023._)  **Temporal link prediction: A unified framework, taxonomy, and review** [[Paper](https://dl.acm.org/doi/abs/10.1145/3625820)]
* (2023.) **Dynamic Graph Representation Learning with Neural Networks: A Survey**  [[Paper](https://arxiv.org/pdf/2304.05729)]

## 2. Non-Deep Learning Methods

### 2.1 Matrix Factorization Based Methods

* (_2018._) **Adaptive Multiple Non-negative Matrix Factorization for Temporal Link Prediction in Dynamic Networks** [[Paper](https://dl.acm.org/doi/10.1145/3229543.3229546)]
* (_2018._) **Graph regularized nonnegative matrix factorization for temporal link prediction in dynamic networks** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0378437117313316)]
* (_2022._) **Graph regularized nonnegative matrix factorization for link prediction in directed temporal networks using PageRank centrality** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0960077922003174)]
* (_2023._) **LUSTC: A Novel Approach for Predicting Link States on Dynamic Attributed Networks** [[paper](https://ieeexplore.ieee.org/document/10091545/)]

### 2.2 Similarity Based Methods

* (_2022._) **CFLP: A new cost based feature for link prediction in dynamic networks** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1877750322001193)]
* (_2019._) **Link prediction in dynamic networks based on the attraction force between nodes** [[paper](https://www.researchgate.net/publication/333627112_Link_prediction_in_dynamic_networks_based_on_the_attraction_force_between_nodes)]
* (_2020._) **Link prediction of time-evolving network based on node ranking** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S095070512030157X)]
* (_2021._) **PILHNB: Popularity, interests, location used hidden Naive Bayesian-based model for link prediction in dynamic social networks** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231221009620)]
* (_2023._) **Temporal link prediction based on node dynamics** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S096007792300303X)]
* (_2022._) **Dynamic Network Embedding via Temporal Path Adjacency Matrix Factorization** [[paper](https://dl.acm.org/doi/10.1145/3511808.3557302)]

### 2.3 Probabilistic Inference Based Methods

* (_2017._) **A probabilistic link prediction model in time-varying social networks** [[paper](https://ieeexplore.ieee.org/document/7996909)]
* (_2019._) **Opportunistic Networks Link Prediction Method Based on Bayesian Recurrent Neural Network** [[paper](https://ieeexplore.ieee.org/document/8937548)]
* (_2024._) **Anomalous Link Detection in Dynamically Evolving Scale-Free-Like Networked Systems** [[paper](https://ieeexplore.ieee.org/document/10553458)]


### 2.4 Random Walk Based Methods

* (_2018._) **Combining Temporal Aspects of Dynamic Networks with Node2Vec for a more Efficient Dynamic Link Prediction** [[paper](https://ieeexplore.ieee.org/document/8508272)]
* (_2019._) **Dynamic Graph Embedding via LSTM History Tracking** [[paper](https://ieeexplore.ieee.org/document/8964233)]
* (_2018._) **dynnode2vec: Scalable Dynamic Network Embedding** [[paper](https://www.computer.org/csdl/proceedings-article/big-data/2018/08621910/17D45XDIXOz) | [code](https://github.com/pedugnat/dynnode2vec)]
* (_2022._) **Link Prediction and Unlink Prediction on Dynamic Networks** [[paper](https://ieeexplore.ieee.org/document/9757817)]
* (_2018._) **Continuous-Time Dynamic Network Embeddings** [[paper](https://dl.acm.org/doi/fullHtml/10.1145/3184558.3191526)]
* (_2022._) **Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks** [[paper](https://arxiv.org/abs/2101.05974) |[code](https://github.com/snap-stanford/CAW)]
* (_2022._) **Neural temporal walks: motif-aware representation learning on continuous-time dynamic graphs** [[paper](https://dl.acm.org/doi/abs/10.5555/3600270.3601715) |[code](https://github.com/KimMeen/Neural-Temporal-Walks)]

## 3. Deep Learning Methods

### 3.1 GNNs Based Methods for DTDGs

* (_2020._) **EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs** [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5984/5840) | [Code](https://github.com/IBM/EvolveGCN)]
* (_2020._) **ComGCN: Community-Driven Graph Convolutional Network for Link Prediction in Dynamic Networks** [[Paper](https://ieeexplore.ieee.org/document/9634845)]
* (_2022._) **A Novel Representation Learning for Dynamic Graphs Based on Graph Convolutional Networks** [[Paper](https://ieeexplore.ieee.org/document/9743367)| [code](https://github.com/cgao-comp/DGCN)]
* (_2024._) **Dynamic network link prediction with node representation learning from graph convolutional networks** [[Paper](https://www.nature.com/articles/s41598-023-50977-6)]
* (_2022._) **GC-LSTM: graph convolution embedded LSTM for dynamic network link prediction** [[Paper](https://dl.acm.org/doi/abs/10.1007/s10489-021-02518-9)]
* (_2022._) **Gated graph convolutional network based on spatio-temporal semi-variogram for link prediction in dynamic complex network** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231222008840)]
* (_2024._) **BehaviorNet: A Fine-grained Behavior-aware Network for Dynamic Link Prediction** [[Paper](https://dl.acm.org/doi/10.1145/3580514)]
* (_2024._) **A deep contrastive framework for unsupervised temporal link prediction in dynamic networks** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025524004122)]
* (_2023._) **HGWaveNet: A Hyperbolic Graph Neural Network for Temporal Link Prediction** [[Paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583455) | [code](https://github.com/TaiLvYuanLiang/HGWaveNet)]
* (_2023._) **Euler: Detecting Network Lateral Movement via Scalable Temporal Link Prediction** [[Paper](https://dl.acm.org/doi/10.1145/3588771) | [code](https://github.com/iHeartGraph/Euler)]
* (_2019._) **GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks** [[Paper](https://ieeexplore.ieee.org/document/8737631) | [code](https://github.com/jiangqn/GCN-GAN-pytorch)]
* (_2019._) **An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks** [[Paper](https://ieeexplore.ieee.org/document/8737631) | [code](https://github.com/jiangqn/GCN-GAN-pytorch)]
* (_2023._) **High-Quality Temporal Link Prediction for Weighted Dynamic Graphs via Inductive Embedding Aggregation** [[Paper](https://ieeexplore.ieee.org/document/10026343) | [code](https://github.com/KuroginQin/IDEA)]
* (_2019._) **Graph WaveNet for Deep Spatial-Temporal Graph Modeling** [[Paper](https://arxiv.org/abs/1906.00121) |[code](https://github.com/nnzhan/Graph-WaveNet)]
* (_2021._) **Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting** [[Paper](https://arxiv.org/abs/2012.09641) |[code](https://github.com/MengzhangLI/STFGNN)]
* (_2018._) **Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting** [[Paper](https://arxiv.org/abs/1709.04875) | [code](https://github.com/VeritasYin/STGCN_IJCAI-18)]
* (_2022._) **ROLAND:GraphLearning Framework for Dynamic Graphs** [[Paper](https://arxiv.org/abs/2208.07239) |[code](https://github.com/snap-stanford/roland)]
* (_2023._) **WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window** [[Paper](https://dl.acm.org/doi/10.1145/3580305.3599551) | [code](https://github.com/THUDM/WinGNN/tree/main))]
* (_2024._) **Temporal graph learning for dynamic link prediction with text in online social networks** [[Paper](https://link.springer.com/article/10.1007/s10994-023-06475-x) ]
* (_2024._) **Dynamic link prediction by learning the representation of node-pair via graph neural networks** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423031871) | [code](https://github.com/ljlilzu/DLP-LRN)]
* (_2023._) **Towards Time-Variant-Aware Link Prediction in Dynamic Graph Through Self-supervised Learning** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-46674-8_33)]

### 3.2 GNNs Based Methods for CTDGs

* (_2018._) **HTNE: Embedding temporal network via neighborhood formation** [[Paper](https://dl.acm.org/doi/10.1145/3219819.3220054) | [Code](http://zuoyuan.github.io/files/htne.zip)]
* (_2019._) **MMDNE: Temporal Network Embedding with Micro- and Macro-dynamics** [[Paper](https://dl.acm.org/doi/10.1145/3357384.3357943) | [Code](https://github.com/rootlu/MMDNE)]
* (_2023._) **Dynamic Representation Learning with Temporal Point Processes for Higher-Order Interaction Forecasting** [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25939/25711)]
* (_2021._) **Dynamic heterogeneous graph embedding via heterogeneous Hawkes process** [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-86486-6_24) | [code](https://github.com/BUPT-GAMMA/HPGE)]
* (_2022._) **TREND: TempoRal Event and Node Dynamics for Graph Representation Learning** [[Paper](https://dl.acm.org/doi/10.1145/3485447.3512164) | [code](https://github.com/WenZhihao666/TREND)]
* (_2022._) **Self-Supervised Temporal Graph Learning With Temporal and Structural Intensity Alignment** [[Paper](https://ieeexplore.ieee.org/iel7/5962385/6104215/10506201.pdf)]
* (_2022._) **Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network** [[Paper](https://dl.acm.org/doi/abs/10.1145/3366423.3380073) | [code](https://github.com/Leo-Q-316/TDGNN)]
* (_2023._) **Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs** [[Paper](https://dl.acm.org/doi/10.1145/3539597.3570465) | [code](https://github.com/RManLuo/GSNOP)]
* (_2021._) **Self-supervised Representation Learning on Dynamic Graphs** [[Paper](https://dl.acm.org/doi/10.1145/3459637.3482389) | [code](https://github.com/ckldan520/DDGCL)]
* (_2020._) **Temporal Graph Networks for Deep Learning on Dynamic Graphs** [[Paper](https://arxiv.org/abs/2006.10637) | [code](https://github.com/twitter-research/tgn)]
* (_2020._) **Streaming Graph Neural Networks** [[Paper](https://dl.acm.org/doi/10.1145/3397271.3401092) | [code](https://github.com/wyd1502/DGNN)]

### 3.3 Attention Mechanism

* (_2020._) **DySAT: Deep Neural Representation Learning on Dynamic
Graphs via Self-Attention Networks** [[Paper](https://dl.acm.org/doi/10.1145/3336191.3371845) | [code](https://github.com/aravindsankar28/DySAT)]
* (_2020._) **Inductive Representation Learning on Temporal Graphs** [[Paper](https://arxiv.org/abs/2002.07962) | [code](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)]
* (_2024._) **DyFormer: A Scalable Dynamic Graph Transformer with Provable Benefits on Generalization Ability** [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523014664) | [code](https://github.com/CongWeilin/DyFormer)]
* (_2023._) **Temporal group-aware graph diffusion networks for dynamic link prediction** [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523014664) | [code](https://github.com/CongWeilin/DyFormer)]
* (_2023._) **Dynamic Link Prediction Using Graph Representation Learning with Enhanced Structure and Temporal Information** [[Paper](https://arxiv.org/abs/2306.14157)]
* (_2022._) **TSAM: Temporal Link Prediction in Directed
Networks based on Self-Attention Mechanism** [[Paper](https://dl.acm.org/doi/abs/10.3233/IDA-205524)]

### 3.4 Autoencoder

* (_2018._) **DynGEM: Deep Embedding Method for Dynamic Graphs** [[Paper](https://arxiv.org/abs/1805.11273) | [code](https://github.com/palash1992/DynamicGEM)]
* (_2020._) **dyngraph2vec: Capturing network dynamics using dynamic graph representation learning** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705119302916) | [code](https://github.com/palash1992/DynamicGEM)]
* (_2019._) **E-LSTM-D: A Deep Learning Framework for Dynamic Network Link Prediction** [[Paper](https://arxiv.org/abs/1902.08329) | [code](https://github.com/jianz94/e-lstm-d)]
* (_2019._) **Variational Graph Recurrent Neural Networks** [[Paper](https://arxiv.org/abs/1908.09710) | [code](https://github.com/VGraphRNN/VGRNN)]
* (_2023._) **DyVGRNN: DYnamic mixture Variational Graph Recurrent Neural Networks** [[Paper](https://www.sciencedirect.com/science/article/pii/S0893608023002927) | [code](https://github.com/GhazalehNiknam/DyVGRNN)]
* (_2021._) **Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs** [[Paper](https://arxiv.org/abs/2104.02228)]

## 4. DLP METHODS FOR COMPLEX GRAPHS

### 4.1 Dynamic Directed Graph

* (_2022._) **GC-LSTM: graph convolution embedded LSTM for dynamic network link prediction** [[Paper](https://dl.acm.org/doi/abs/10.1007/s10489-021-02518-9)]
* (_2022._) **Graph regularized nonnegative matrix factorization for link prediction in directed temporal networks using PageRank centrality** [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0960077922003174)]

### 4.2 Dynamic Weighted Graph

* (_2019._) **GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks** [[Paper](https://ieeexplore.ieee.org/document/8737631) | [code](https://github.com/jiangqn/GCN-GAN-pytorch)]
* (_2023._) **High-Quality Temporal Link Prediction for Weighted Dynamic Graphs via Inductive Embedding Aggregation** [[Paper](https://ieeexplore.ieee.org/document/10026343) | [code](https://github.com/KuroginQin/IDEA)]

### 4.3 Dynamic Heterogeneous Graph

* (_2020._) **Modeling Dynamic Heterogeneous Network for Link Prediction using Hierarchical Attention with Temporal RNN** [[Paper](https://arxiv.org/abs/2004.01024) | [code](https://github.com/skx300/DyHATR)]
* (_2020._) **Dynamic Heterogeneous Information Network Embedding With Meta-Path Based Proximity** [[Paper](https://ieeexplore.ieee.org/document/9091208) | [code](https://github.com/rootlu/DyHNE)]
* (_2021._) **Temporal Heterogeneous Information Network Embedding** [[Paper](https://www.ijcai.org/proceedings/2021/203) | [code](https://github.com/S-rz/THINE)]
* (_2023._) **Link Prediction for Temporal Heterogeneous Networks Based on the Information Lifecycle** [[Paper](https://www.mdpi.com/2227-7390/11/16/3541)]
* (_2022._) **H2TNE:Temporal heterogeneous information network embedding in hyperbolic
spaces** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_11)]
* (_2019._) **JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks** [[Paper](https://snap.stanford.edu/jodie/) | [code](https://github.com/srijankr/jodie)]
* (_2023._) **Dynamic Heterogeneous Graph Attention Neural Architecture Search** [[Paper](https://zw-zhang.github.io/files/2023_AAAI_DHGAS.pdf) | [code](https://github.com/wondergo2017/DHGAS)]

### 4.4 Dynamic Signed Graph

* (_2023._) **Representation Learning in Continuous-Time Dynamic Signed Networks** [[Paper](https://dl.acm.org/doi/10.1145/3583780.3615032) | [code](https://github.com/claws-lab/semba)]

* (_2024._) **DynamiSE: dynamic signed network embedding for link prediction** [[Paper](https://dl.acm.org/doi/10.1007/s10994-023-06473-z) | [code](https://github.com/claws-lab/semba)]

### 4.4 Dynamic HyperGraph

* (_2022._) **Dynamic Hypergraph Convolutional Network** [[Paper](https://ieeexplore.ieee.org/document/9835240)]
* (_2022._) **Temporal Edge-Aware Hypergraph Convolutional Network for Dynamic Graph Embedding** [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20862-1_32)]
* (_2023._) **# HyperDNE: Enhanced hypergraph neural network for dynamic network embedding** [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231223000504) | [code](https://github.com/qhgz2013/HyperDNE)]

## 5. Dynamic Graph Datasets

| **Dataset**            | **Graph type**            | **URL**                                                                 |
|------------------------|---------------------------|-------------------------------------------------------------------------|
| Alibaba                | -                         | [Link](https://tianchi.aliyun.com/competition/entrance/231719)          |
| Aminer                 | -                         | [Link](https://www.aminer.cn/citation)                                  |
| AS-733                 | -                         | [Link](https://snap.stanford.edu/data/as-733.html)                      |
| Askubuntu              | -                         | [Link](https://west.uni-koblenz.de/konect/networks)                     |
| Bitcoin-Alpha          | Weighted                  | [Link](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)     |
| Bitcoin-OTC            | Weighted, signed          | [Link](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)       |
| Brain                  | -                         | [Link](https://www.dropbox.com/sh/33p0gk4etgdjfvz)                     |
| Cellphone call         | -                         | [Link](http://www.sociopatterns.org/datasets)                           |
| Colab                  | -                         | [Link](https://arnetminer.org)                                          |
| CollegeMsg             | Directed                  | [Link](https://snap.stanford.edu/data/CollegeMsg.html)                 |
| Cora                   | -                         | [Link](https://graphsandnetworks.com/the-cora-dataset)                 |
| COVID2019              | Heterogeneous             | [Link](https://coronavirus.1point3acres.com)                            |
| DBLP                   | Heterogeneous             | [Link](https://www.aminer.org/citation)                                 |
| Email                  | Directed                  | [Link](http://networkrepository.com/dynamic.php)                        |
| Enron                  | -                         | [Link](https://networkrepository.com/ia-enron-employees.php)           |
| Epinions               | Signed                    | [Link](http://www.trustlet.org/wiki/Extended_Epinions_dataset)          |
| FB-Messages            | -                         | [Link](http://networkrepository.com/fb-messages.php)                   |
| Face-to-face interaction | Directed, weighted        | [Link](https://snap.stanford.edu/data/comm-f2f-Resistance.html)        |
| Hep-Ph                 | Directed                  | [Link](https://snap.stanford.edu/data/cit-HepPh.html)                   |
| Math                   | -                         | [Link](http://snap.stanford.edu/data/sx-mathoverflow.html)              |
| MathOverflow           | Heterogeneous             | [Link](https://snap.stanford.edu/data/sx-mathoverflow.html)             |
| MOOC                   | Directed, heterogeneous   | [Link](https://snap.stanford.edu/data/act-mooc.html)                    |
| NBA                    | Heterogeneous             | [Link](https://www.basketball-reference.com)                            |
| Reddit-Hyperlink       | Directed, signed, heterogeneous | [Link](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)      |
| Wikipedia              | -                         | [Link](http://snap.stanford.edu/jodie/wikipedia.csv)                    |
| Yelp                   | Heterogeneous             | [Link](https://www.yelp.com/dataset)                                    |
| Twitter                | Directed                  | [Link](https://snap.stanford.edu/data/higgs-twitter.html)              |
| Taobao                 | Heterogeneous             | [Link](https://tianchi.aliyun.com/dataset/9716)                        |

