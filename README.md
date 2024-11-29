<h1 align="center"> Awesome-Dynamic-Link-Prediction</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align="center">

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

</h5>

# introduction



## Table of Contents

- Dynamic Link Prediction
  - [Related survey](#Related-Survey)
  - [Non-Deep Learning Methods](#Matrix-Factorization-Based-Methods)
  - [Deep Learning Methods](#Deep-Learning-Methods)

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
* (_2022._) **A Novel Representation Learning for Dynamic Graphs Based on Graph Convolutional Networks** [[Paper](https://ieeexplore.ieee.org/document/9743367)|[code](https://github.com/cgao-comp/DGCN)]
* (_2024._) **Dynamic network link prediction with node representation learning from graph convolutional networks** [[Paper](https://www.nature.com/articles/s41598-023-50977-6)]
* (_2022._) **GC-LSTM: graph convolution embedded LSTM for dynamic network link prediction** [[Paper](https://dl.acm.org/doi/abs/10.1007/s10489-021-02518-9)]
* (_2022._) **Gated graph convolutional network based on spatio-temporal semi-variogram for link prediction in dynamic complex network** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231222008840)]
* (_2024._) **BehaviorNet: A Fine-grained Behavior-aware Network for Dynamic Link Prediction** [[Paper](https://dl.acm.org/doi/10.1145/3580514)]
* (_2024._) **A deep contrastive framework for unsupervised temporal link prediction in dynamic networks** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025524004122)]
* (_2023._) **HGWaveNet: A Hyperbolic Graph Neural Network for Temporal Link Prediction** [[Paper](https://dl.acm.org/doi/abs/10.1145/3543507.3583455) |[code](https://github.com/TaiLvYuanLiang/HGWaveNet)]
* (_2023._) **Euler: Detecting Network Lateral Movement via Scalable Temporal Link Prediction** [[Paper](https://dl.acm.org/doi/10.1145/3588771) |[code](https://github.com/iHeartGraph/Euler)]
* (_2019._) **GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks** [[Paper](https://ieeexplore.ieee.org/document/8737631) |[code](https://github.com/jiangqn/GCN-GAN-pytorch)]
* (_2019._) **An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks** [[Paper](https://ieeexplore.ieee.org/document/8737631) |[code](https://github.com/jiangqn/GCN-GAN-pytorch)]
* (_2023._) **High-Quality Temporal Link Prediction for Weighted Dynamic Graphs via Inductive Embedding Aggregation** [[Paper](https://ieeexplore.ieee.org/document/10026343) |[code](https://github.com/KuroginQin/IDEA)]
* (_2019._) **Graph WaveNet for Deep Spatial-Temporal Graph Modeling** [[Paper](https://arxiv.org/abs/1906.00121) |[code](https://github.com/nnzhan/Graph-WaveNet)]
* (_2021._) **Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting** [[Paper](https://arxiv.org/abs/2012.09641) |[code](https://github.com/MengzhangLI/STFGNN)]
* (_2018._) **Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting** [[Paper](https://arxiv.org/abs/1709.04875) |[code](https://github.com/VeritasYin/STGCN_IJCAI-18)]
* (_2022._) **ROLAND:GraphLearning Framework for Dynamic Graphs** [[Paper](https://arxiv.org/abs/2208.07239) |[code](https://github.com/snap-stanford/roland)]
* (_2023._) **WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window** [[Paper](https://dl.acm.org/doi/10.1145/3580305.3599551) |[code](https://github.com/THUDM/WinGNN/tree/main))]
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
* (_2023._) **Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs** [[Paper](https://dl.acm.org/doi/10.1145/3539597.3570465) |  [code](https://github.com/RManLuo/GSNOP)]
* (_2021._) **Self-supervised Representation Learning on Dynamic Graphs** [[Paper](https://dl.acm.org/doi/10.1145/3459637.3482389) | [code](https://github.com/ckldan520/DDGCL)]
* (_2020._) **Temporal Graph Networks for Deep Learning on Dynamic Graphs** [[Paper](https://arxiv.org/abs/2006.10637) | [code](https://github.com/twitter-research/tgn)]
* (_2020._) **Streaming Graph Neural Networks** [[Paper](https://dl.acm.org/doi/10.1145/3397271.3401092) | [code](https://github.com/wyd1502/DGNN)]
* (_2020._) **DySAT: Deep Neural Representation Learning on Dynamic
Graphs via Self-Attention Networks** [[Paper](https://dl.acm.org/doi/10.1145/3336191.3371845) | [code](https://github.com/aravindsankar28/DySAT)]
* (_2020._) **Inductive Representation Learning on Temporal Graphs** [[Paper](https://arxiv.org/abs/2002.07962) | [code](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)]
* (_2024._) **DyFormer: A Scalable Dynamic Graph Transformer with Provable Benefits on Generalization Ability** [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523014664) | [code](https://github.com/CongWeilin/DyFormer)]
* (_2023._) **Temporal group-aware graph diffusion networks for dynamic link prediction** [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523014664) | [code](https://github.com/CongWeilin/DyFormer)]
* (_2023._) **Dynamic Link Prediction Using Graph Representation Learning with Enhanced Structure and Temporal Information** [[Paper](https://arxiv.org/abs/2306.14157)]
* (_2022._) **TSAM: Temporal Link Prediction in Directed
Networks based on Self-Attention Mechanism** [[Paper](https://dl.acm.org/doi/abs/10.3233/IDA-205524)]
* (_2018._) **DynGEM: Deep Embedding Method for Dynamic Graphs** [[Paper](https://arxiv.org/abs/1805.11273) | [code](https://github.com/palash1992/DynamicGEM)]
* (_2020._) **dyngraph2vec: Capturing network dynamics using dynamic graph representation learning** [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705119302916) | [code](https://github.com/palash1992/DynamicGEM)]
* (_2019._) **E-LSTM-D: A Deep Learning Framework for Dynamic Network Link Prediction** [[Paper](https://arxiv.org/abs/1902.08329) | [code](https://github.com/jianz94/e-lstm-d)]
* (_2019._) **Variational Graph Recurrent Neural Networks** [[Paper](https://arxiv.org/abs/1908.09710) | [code](https://github.com/VGraphRNN/VGRNN)]
* (_2023._) ** DyVGRNN: DYnamic mixture Variational Graph Recurrent Neural Networks** [[Paper](https://www.sciencedirect.com/science/article/pii/S0893608023002927) | [code](https://github.com/GhazalehNiknam/DyVGRNN)]
* (_2021._) **Hyperbolic Variational Graph Neural Network for Modeling Dynamic Graphs** [[Paper](https://arxiv.org/abs/2104.02228)]





### Attention Mechanism

### Autoencoder



* (_2023.05_) [NeurIPS' 2023] **Can language models solve graph problems in natural language?** [[Paper](https://arxiv.org/abs/2305.10037) | [Code](https://github.com/Arthur-Heng/NLGraph)]
