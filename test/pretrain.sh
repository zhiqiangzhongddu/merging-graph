# attr_masking: node dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method attr_masking \
  device 0

# attr_masking: graph dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method attr_masking \
  device 0

# context_pred: node dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method context_pred \
  device 0

# context_pred: graph dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method context_pred \
  device 0

# dgi: node dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method dgi \
  device 0

# dgi: graph dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method dgi \
  device 0

# edge_pred: node dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  device 0

# edge_pred: graph dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  device 0

# graphcl: node dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method graphcl \
  device 0

# graphcl: graph dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method graphcl \
  device 0

# infograph: node dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method infograph \
  device 0

# infograph: graph dataset
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method infograph \
  device 0

# supervised: node dataset, node-level classification task
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.dataset.task_type classification \
  pretrain.method supervised \
  pretrain.dataset.fixed_split "(0.8,0.1,0.1)" \
  device 0

# supervised: graph-level single-label classification task
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.dataset.task_type classification \
  pretrain.method supervised \
  pretrain.dataset.fixed_split "(0.8,0.1,0.1)" \
  device 0

# supervised: graph-level multiple-label classification task
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name tox21 \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.dataset.task_type classification \
  pretrain.method supervised \
  pretrain.dataset.fixed_split "(0.8,0.1,0.1)" \
  device 0

# supervised: graph-level regression task
python run_pretrain.py \
  model.name gcn \
  pretrain.dataset.name esol \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.dataset.task_type regression \
  pretrain.method supervised \
  pretrain.dataset.fixed_split "(0.8,0.1,0.1)" \
  device 0
