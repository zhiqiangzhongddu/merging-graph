# supervised: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# supervised: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# supervised: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# supervised: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# supervised: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method supervised \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# supervised: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# supervised: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# supervised: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# supervised: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method supervised \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# supervised: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method supervised \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# all_in_one: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# all_in_one: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method all_in_one \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.batch_size 128 \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# edgeprompt+: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method edgeprompt \
  finetune.edgeprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# gpf: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# gpf: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# gpf: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# gpf: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method gpf \
  finetune.gpf.plus False \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gpf+: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method gpf \
  finetune.gpf.plus True \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# gppt: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gppt: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# gppt: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gppt: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gppt: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method gppt \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# gppt: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gppt: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# gppt: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gppt: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method gppt \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# gppt: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method gppt \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method graphprompt \
  finetune.graphprompt.plus False \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from node dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from node dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name citeseer \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from node dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from node dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from node dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name cora \
  pretrain.dataset.task_level node \
  pretrain.dataset.induced True \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from graph dataset to node dataset, node-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level node \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from graph dataset to node dataset, edge-level classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name cora \
  finetune.dataset.task_level edge \
  finetune.dataset.induced True \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.05,0.1)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from graph dataset to graph dataset, graph-level single-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name bace \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from graph dataset to graph dataset, graph-level multiple-label classification task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name tox21 \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type classification \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(100,0.0,1.0)" \
  finetune.num_runs 1 \
  device 0

# graphprompt+: from graph dataset to graph dataset, graph-level regression task
python run_finetune.py \
  model.name gcn \
  pretrain.dataset.name bace \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method edge_pred \
  finetune.dataset.name esol \
  finetune.dataset.task_level graph \
  finetune.dataset.induced False \
  finetune.dataset.task_type regression \
  finetune.method graphprompt \
  finetune.graphprompt.plus True \
  finetune.dataset.fixed_split "(0.1,0.1,0.8)" \
  finetune.num_runs 1 \
  device 0
