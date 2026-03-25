# supervised: node dataset, node-level classification task
python run_train.py \
  model.name gcn \
  train.dataset.name cora \
  train.dataset.task_level node \
  train.dataset.induced True \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(0.8,0.1,0.1)" \
  train.num_runs 1 \
  device 0

# supervised: node dataset, edge-level classification task
python run_train.py \
  model.name gcn \
  train.dataset.name cora \
  train.dataset.task_level edge \
  train.dataset.induced True \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(0.1,0.05,0.1)" \
  train.num_runs 1 \
  device 0

# supervised: graph-level single-label classification task
python run_train.py \
  model.name gcn \
  train.dataset.name bace \
  train.dataset.task_level graph \
  train.dataset.induced False \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(0.8,0.1,0.1)" \
  train.num_runs 1 \
  device 0

# supervised: graph-level multiple-label classification task
python run_train.py \
  model.name gcn \
  train.dataset.name tox21 \
  train.dataset.task_level graph \
  train.dataset.induced False \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(0.8,0.1,0.1)" \
  train.num_runs 1 \
  device 0

# supervised: graph-level regression task
python run_train.py \
  model.name gcn \
  train.dataset.name esol \
  train.dataset.task_level graph \
  train.dataset.induced False \
  train.dataset.task_type regression \
  train.dataset.fixed_split "(0.1,0.1,0.8)" \
  train.num_runs 1 \
  device 0

# few-shot: node dataset, node-level classification task
python run_train.py \
  model.name gcn \
  train.dataset.name cora \
  train.dataset.task_level node \
  train.dataset.induced True \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(100,0.0,1.0)" \
  train.num_runs 1 \
  device 0

# few-shot: graph-level single-label classification task
python run_train.py \
  model.name gcn \
  train.dataset.name bace \
  train.dataset.task_level graph \
  train.dataset.induced False \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(5,0.0,1.0)" \
  train.num_runs 1 \
  device 0

# few-shot: graph-level multiple-label classification task
python run_train.py \
  model.name gcn \
  train.dataset.name tox21 \
  train.dataset.task_level graph \
  train.dataset.induced False \
  train.dataset.task_type classification \
  train.dataset.fixed_split "(5,0.0,1.0)" \
  train.num_runs 1 \
  device 0
  