# Graph Model Merging

## Environment set up

```bash
# create a new environment
conda create -n merging-graph -y python=3.10
conda activate merging-graph

# install pytorch
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# install pytorch-geometric
pip install numpy==1.26.1 torch-geometric==2.5.1

# install pytorch-geometric dependencies (if needed)
# install torch-scatter (recommended)
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
# install torch-sparse (recommended)
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
# install torch-cluster (not necessary)
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.1.1+cu118.html

# other requirements see requirements.txt
```

## Data Preparation

- Default dataset root: `data/datasets` (override with `data_preparation.dataset.root`).
- Node- and edge-level dataset list: `data/available_node_datasets.tsv`.
- Graph-level dataset list: `data/available_graph_datasets.tsv`.
- Total available datasets: `50`.
- `run_data_preparation.py` handles download, preprocessing, split generation, feature SVD, induced subgraphs, and subgraph SVD (based on config).
- If `data_preparation.target_datasets` is empty, both TSV lists are prepared automatically.

```bash
# prepare selected datasets (task level inferred from dataset registry)
python run_data_preparation.py \
  data_preparation.target_datasets "[cora,actor]"

# prepare selected graph datasets
python run_data_preparation.py \
  data_preparation.target_datasets "[bace,bbbp]"

# prepare all available node/edge datasets
python run_data_preparation.py \
  data_preparation.target_datasets data/available_node_datasets.tsv

# prepare all available graph datasets
python run_data_preparation.py \
  data_preparation.target_datasets data/available_graph_datasets.tsv

# prepare all available datasets (both TSV files)
python run_data_preparation.py

# HPC execution (array job, one dataset per task)
sbatch slurm/run_all_data_preparation.slurm

# generate dataset summary at data/data_summary.tsv
python run_data_summary.py
```

## Expert Models

We use 9 expert models across 4 categories:
- [`MLP`](https://www.sciencedirect.com/science/article/pii/S1352231097004470): pure feature-based model.
- [`GCN`](https://arxiv.org/abs/1609.02907), [`GAT`](https://arxiv.org/abs/1710.10903), [`GIN`](https://arxiv.org/abs/1810.00826): message passing GNNs (homophilous settings).
- [`H2GCN`](https://arxiv.org/abs/2006.11468), [`FAGCN`](https://arxiv.org/abs/2101.00797): message passing GNNs (heterophilous settings).
- [`Transformer`](https://arxiv.org/abs/1706.03762), [`NodeFormer`](https://arxiv.org/abs/2306.08385), [`GPS`](https://arxiv.org/abs/2205.12454): transformer-style graph models.

## Training

```bash
# supervised: node dataset, node-level task
python run_train.py \
  model.name gcn \
  train.dataset.name cora \
  train.dataset.task_level node \
  train.dataset.induced True \
  train.dataset.fixed_split "(0.8,0.1,0.1)" \
  train.num_runs 5 \
  device 0

# supervised: node dataset, edge-level task
python run_train.py \
  model.name gcn \
  train.dataset.name cora \
  train.dataset.task_level edge \
  train.dataset.induced True \
  train.dataset.fixed_split "(0.1,0.05,0.1)" \
  train.num_runs 5 \
  device 0

# supervised: graph-level task
python run_train.py \
  model.name gcn \
  train.dataset.name bace \
  train.dataset.task_level graph \
  train.dataset.fixed_split "(0.8,0.1,0.1)" \
  train.num_runs 5 \
  device 0

# few-shot: node classification
python run_train.py \
  model.name gcn \
  train.dataset.name cora \
  train.dataset.task_level node \
  train.dataset.induced True \
  train.dataset.fixed_split "(100,0.0,1.0)" \
  train.num_runs 5 \
  device 0

# few-shot: graph classification
python run_train.py \
  model.name gcn \
  train.dataset.name bace \
  train.dataset.task_level graph \
  train.dataset.fixed_split "(5,0.0,1.0)" \
  train.num_runs 5 \
  device 0
```

Note: few-shot splits are not supported for regression datasets.

## Pre-training 

Supported 7 pretraining methods:
- [`attr_masking`](https://arxiv.org/abs/1905.12265): mask and reconstruct node/edge attributes.
- [`context_pred`](https://arxiv.org/abs/1905.12265): predict sampled subgraph context.
- [`dgi`](https://arxiv.org/abs/1809.10341): maximize mutual information for node embeddings.
- [`edge_pred`](https://arxiv.org/abs/1905.12265): link prediction with negative sampling.
- [`graphcl`](https://arxiv.org/abs/2010.13902): graph contrastive learning.
- [`infograph`](https://arxiv.org/abs/1908.01000): maximize mutual information between graph and patch embeddings.
- [`supervised`](https://arxiv.org/abs/1905.12265): standard label supervision on provided splits.

Run a single pre-training job from the project root:
```bash
# attr_masking / context_pred / dgi / edge_pred / graphcl / infograph
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  device 0

# supervised
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method supervised \
  pretrain.dataset.fixed_split "(0.8,0.1,0.1)" \
  device 0
```

Submit an HPC job for large-scale pre-training:
```bash
sbatch slurm/run_all_pretrain.slurm
```

## Fine-tuning

Supported 9 fine-tuning methods:
- [`supervised`](https://arxiv.org/abs/1905.12265): standard label supervision on provided splits.
- [`all_in_one`](https://arxiv.org/abs/2307.01504): multi-task graph prompting with a single promptable backbone; uses task-specific prompts/templates to adapt one pretrained model across many graph tasks.
- [`edgeprompt`](https://arxiv.org/abs/2503.00750): edge-level prompting that injects learnable prompt parameters on edges (or messages) to steer the model for downstream tasks.
- [`edgeprompt+`](https://arxiv.org/abs/2503.00750): the stronger variant with an enhanced prompt design/training recipe (more prompt capacity and/or improved optimization/regularization). 
- [`gpf`](https://arxiv.org/abs/2209.15240): prepend/add a small set of learnable “prompt” node features to the input and tune only these prompts (backbone mostly frozen).
- [`gpf+`](https://arxiv.org/abs/2209.15240): adds an extra prompt module (for example, a feature-wise transformation / gating) to improve adaptation.
- [`gppt`](https://dl.acm.org/doi/abs/10.1145/3534678.3539249): prompt-based tuning for graph pretrained transformers by learning a small number of prompt tokens/vectors while keeping most pretrained parameters fixed.
- [`graphprompt`](https://arxiv.org/abs/2302.08043): prompt tuning for graphs by learning task prompts that condition a pretrained GNN/GFM for downstream prediction. 
- [`graphprompt+`](https://arxiv.org/abs/2311.15317): extends the base method with a stronger prompt formulation and/or training strategy for better transfer.

```bash
# supervised
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  device 0

# all_in_one
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  finetune.batch_size 128 \
  device 0

# edgeprompt
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0

# edgeprompt+
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0

# gpf
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0

# gpf+
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0

# gppt
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.method gppt \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0

# graphprompt
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0

# graphprompt+
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(5,0.0,1.0)" \
  device 0
```
