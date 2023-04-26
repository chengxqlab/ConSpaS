## ConSpaS

### Overview

con-intS is a GNN with contrastive learning strategy that identifies spatial domains from ST data. Generally, con-intS considers both local and global similarities through a GAE and contrastive learning， respectively.Specifically, it mainly consists of two modules: GAE and CL modules.The GAE module aims to capture the local spatial similarity and makes the spatial neighbors get closer in the learned feature space, thereby ensuring that the cluster assignments can be spatially continuous.The CL module is designed to capture the global semantic similarity and refine the spot feature learned by the GAE. In particular, by considering global semantic information, we propose an augmentation-free mechanism to define the global positive sample and simultaneously use a semi-easy sampling strategy to construct negative samples.Finally, the learned features are took $L_2$ normalization and then used to perform $k$-means or Leiden to obtain the cluster assignments. The learned feature of spots can be also well generalized to other downstream tasks, including clustering, UMAP, trajectory inference and denoising.

![](https://github.com/chengxqlab/ConSpaS/blob/main/pic/overview.jpg)

### Installation

1. create conda environment

   ```
   conda create -n ConSpaS python=3.8
   conda activate ConSpaS
   ```

2. install pytorch

   ```
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

3. install pyG

   ```
   pip install torch_geometric
   
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
   ```

4. install required package

   ```
   pip install -r requirements.txt
   ```

### Getting started

see DLPFC_151672_tutorial.ipynb
