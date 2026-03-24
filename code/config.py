import os
import argparse
from yacs.config import CfgNode as CN


def _default_dataset_cfg() -> CN:
    """Build a dataset config block with common defaults."""
    ds = CN()
    ds.name = "cora" # dataset name, None means to iterate over all available datasets
    ds.root = "data/datasets" # root directory for datasets
    ds.available_node_datasets = "data/available_node_datasets.tsv" # path to available node- and edge-level datasets list
    ds.available_graph_datasets = "data/available_graph_datasets.tsv" # path to available graph-level datasets list
    ds.task_type = "none"  # classification or regression or none (none is used for pretraining runs on large datasets without available labels, e.g., ZINC15)
    ds.task_level = "none"  # node or graph or edge or none (none is used for pretraining runs on large datasets without available labels, e.g., ZINC15)
    ds.num_classes = None  # filled automatically if available
    ds.label_dim = None  # number of target dimensions when available (e.g., multi-task graph labels)
    ds.fixed_split = None  # when set, use fixed train/val/test split ratios
    ds.num_splits = 5 # number of random splits to generate (overridden by fixed_split if set)
    ds.split_root = "data/splits" # root directory for dataset splits
    ds.feat_reduction = True  # SVD feature reduction on node features
    ds.feat_reduction_svd_dim = 100 # target dimension for SVD feature reduction
    ds.feature_svd_dir = "data/feature_svd" # output directory for feature SVD files
    ds.induced = True  # when True, operate on induced subgraphs
    ds.induced_min_size = 10 # min number of nodes for induced subgraphs
    ds.induced_max_size = 30 # max number of nodes for induced subgraphs
    ds.induced_max_hops = 5 # max hops to consider when building induced subgraphs
    ds.edge_level_max_num_nodes = 10000 # skip edge-level data preparation for node datasets above this node-count threshold; set <=0 to disable
    ds.induced_root = "data/induced_subgraphs" # output directory for induced subgraphs
    ds.subgraph_svd = True  # generate subgraph SVD features during data prep
    ds.subgraph_svd_feat_dim = 100 # subgraph SVD feature dimension
    ds.subgraph_svd_struct_dim = 100 # subgraph SVD structure dimension
    ds.subgraph_svd_matrix = "adjacency"  # adjacency or laplacian
    ds.subgraph_svd_dir = "data/subgraph_svd"  # output directory for subgraph SVD files
    return ds

def set_cfg(cfg: CN) -> CN:

    # ------------------------------------------------------------------------ #
    # General settings
    # ------------------------------------------------------------------------ #
    cfg.device = 0  # CUDA device index
    cfg.seed = 42   # global seed for reproducibility
    cfg.seeds = [0, 42, 100, 123, 2024]  # seeds used by multi-run workflows (e.g., training/data prep/finetuning)

    # ------------------------------------------------------------------------ #
    # Dataset preparation options
    # ------------------------------------------------------------------------ #
    cfg.data_preparation = CN()
    cfg.data_preparation.dataset = _default_dataset_cfg()
    cfg.data_preparation.target_datasets = None # required: "data_name" -> one dataset; "data_name1,data_name2" -> multiple datasets; existing/path-like file -> read names from file
    cfg.data_preparation.node_task_splits = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (100, 0.0, 1.0), (5, 0.0, 1.0)]
    cfg.data_preparation.graph_task_splits = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (100, 0.0, 1.0), (5, 0.0, 1.0)]
    cfg.data_preparation.edge_task_splits = [(0.1, 0.05, 0.1)]
    cfg.data_preparation.generate_edge_level = True  # whether to generate edge-level information for node-level datasets
    cfg.data_preparation.summary_file = "data/summary.tsv" # output file for dataset summary information
    
    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    # General GNN options
    cfg.model = CN()
    cfg.model.name = "gcn"  # gcn, gin, gat, mlp, h2gcn, fagcn, transformer, gps, nodeformer
    cfg.model.in_dim = 100  # inferred from dataset when 0 or None
    cfg.model.hidden_dim = 128 # hidden dimension
    cfg.model.out_dim = 128 # output dimension
    cfg.model.num_layers = 2 # number of GNN layers
    cfg.model.dropout = 0.5 # dropout rate
    cfg.model.activation = "relu" # relu, gelu, prelu, elu, leaky_relu
    cfg.model.graph_pooling = "mean" # mean, max, sum, target (only output the target node embedding)
    cfg.model.use_batchnorm = False # apply BatchNorm1d after hidden layers (and layerwise InfoGraph cache)
    # FAGCN specific options
    cfg.model.fagcn = CN()
    cfg.model.fagcn.eps = 0.1 # initial epsilon value
    cfg.model.fagcn.use_batchnorm = False # whether to use batchnorm
    # GAT specific options
    cfg.model.gat = CN()
    cfg.model.gat.heads = 8 # number of attention heads
    # GTN specific options
    cfg.model.gtn = CN()
    cfg.model.gtn.channels = 2 # number of channels
    cfg.model.gtn.filters = 2 # number of filters
    # H2GCN specific options
    cfg.model.h2gcn = CN()
    cfg.model.h2gcn.use_batchnorm = False # whether to use batchnorm
    # GPS specific options
    cfg.model.gps = CN()
    cfg.model.gps.heads = 4 # number of attention heads
    cfg.model.gps.dropout = 0.1 # attention dropout rate
    cfg.model.gps.attn_type = "multihead"  # or performer
    # NodeFormer specific options
    cfg.model.nodeformer = CN()
    cfg.model.nodeformer.heads = 4 # number of attention heads
    cfg.model.nodeformer.num_random_features = 30 # number of random features
    cfg.model.nodeformer.tau = 1.0 # softmax temperature
    cfg.model.nodeformer.use_layernorm = True # whether to use layernorm
    cfg.model.nodeformer.use_gumbel = True # whether to use gumbel softmax
    cfg.model.nodeformer.use_residual = True # whether to use residual connections
    cfg.model.nodeformer.use_activation = True # whether to use activation function
    cfg.model.nodeformer.use_jk = False # whether to use jumpy knowledge connections
    cfg.model.nodeformer.num_gumbel_samples = 10 # number of gumbel samples
    cfg.model.nodeformer.rb_order = 0 # random basis order
    cfg.model.nodeformer.rb_trans = "sigmoid" # transformation for random basis: sigmoid, softplus, relu
    cfg.model.nodeformer.use_edge_loss = False  # disable edge loss for encoder use cases

    # ------------------------------------------------------------------------ #
    # Train options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    cfg.train.num_runs = 5  # number of training runs with different seeds
    cfg.train.epochs = 500 # maximum number of training epochs
    cfg.train.early_stopping = 50 # early stopping patience
    cfg.train.lr = 1e-3 # learning rate
    cfg.train.weight_decay = 0.0 # weight decay
    cfg.train.batch_size = 128  # used for induced tasks and graph-level tasks
    cfg.train.monitor_metric = "auto" # canonical train monitor setting; auto is resolved by the shared monitoring policy
    cfg.train.num_workers = 0 # number of data loading workers
    cfg.train.checkpoint_dir = "trained_models" # directory to save checkpoints
    cfg.train.log_dir = "logs/training_models" # directory to save training logs
    cfg.train.skip_if_exists = True  # skip training run if checkpoint already exists
    # train dataset options
    cfg.train.dataset = _default_dataset_cfg()
    cfg.train.dataset.fixed_split = (1, 0.0, 1.0) # fixed train/val/test split ratios (overridden by specific pretrain/finetune settings)

    # ------------------------------------------------------------------------ #
    # Pretrain options
    # ------------------------------------------------------------------------ #
    cfg.pretrain = CN()
    cfg.pretrain.method = "edge_pred" # pretraining method: supervised, attr_masking, context_pred, dgi, edge_pred,  graphcl, infograph
    cfg.pretrain.epochs = 500 # maximum number of pretraining epochs
    cfg.pretrain.early_stopping = 50 # early stopping patience
    cfg.pretrain.lr = 1e-3 # learning rate
    cfg.pretrain.weight_decay = 0.0 # weight decay
    cfg.pretrain.batch_size = 128  # used for induced tasks and graph-level tasks
    cfg.pretrain.monitor_metric = "auto" # pretrain monitor setting; auto is resolved by the shared monitoring policy
    cfg.pretrain.num_workers = 0 # number of data loading workers
    cfg.pretrain.checkpoint_dir = "pretrained_models" # directory to save checkpoints
    cfg.pretrain.skip_if_exists = True  # skip pretrain run if checkpoint already exists
    cfg.pretrain.tasks_tsv = "slurm/run_all_pretrain_experiments.tsv"  # dataset/task definitions for run_all
    # pretrain dataset options
    cfg.pretrain.dataset = _default_dataset_cfg()
    cfg.pretrain.dataset.fixed_split = (0.8, 0.1, 0.1) # train, val, test split ratios
    # attr_masking specific options
    cfg.pretrain.attr_masking = CN()
    cfg.pretrain.attr_masking.mask_ratio = 0.15 # ratio of node features to mask
    cfg.pretrain.attr_masking.mask_edge = False # when True, add edge-type masking loss for edges attached to masked nodes
    cfg.pretrain.attr_masking.edge_loss_weight = 1.0 # weighting factor for optional edge masking loss
    cfg.pretrain.attr_masking.node_vocab_size = 0 # >0 enables CE node-type prediction using x[:,0] as label
    cfg.pretrain.attr_masking.edge_vocab_size = 0 # >0 enables CE edge-type prediction using edge_attr[:,0] as label
    # context_pred specific options
    cfg.pretrain.context_pred = CN()
    cfg.pretrain.context_pred.mode = "cbow" # cbow or skipgram
    cfg.pretrain.context_pred.context_pooling = "mean" # mean, sum, max
    cfg.pretrain.context_pred.neg_samples = 1 # number of negatives per positive pair
    cfg.pretrain.context_pred.context_size = 3 # l2 = (k-1) + context_size
    cfg.pretrain.context_pred.substruct_hops = 0 # 0 means use model.num_layers
    # edge_pred specific options
    cfg.pretrain.edge_pred = CN()
    cfg.pretrain.edge_pred.pos_edge_ratio = 1.0  # <1.0 to sample a subset of positive edges
    cfg.pretrain.edge_pred.pos_edge_max = 0  # >0 to cap the number of positive edges
    cfg.pretrain.edge_pred.neg_ratio = 1.0  # negatives per positive edge
    cfg.pretrain.edge_pred.forward_edge_ratio = 1.0  # <1.0 to sample edges used in message passing
    cfg.pretrain.edge_pred.forward_edge_max = 0  # >0 to cap edges used in message passing
    cfg.pretrain.edge_pred.use_mlp_scorer = False  # when True, use an MLP edge scorer instead of dot product
    cfg.pretrain.edge_pred.use_neighbor_sampling = False  # edge-level mini-batch with neighbor sampling
    cfg.pretrain.edge_pred.neighbor_sizes = [15, 10]  # per-layer neighbor sample sizes
    cfg.pretrain.edge_pred.edge_batch_size = 4096  # number of edges per batch
    # graphcl specific options
    cfg.pretrain.graphcl = CN()
    cfg.pretrain.graphcl.edge_remove_prob = 0.1 # edge removal probability
    cfg.pretrain.graphcl.permE_add_edges = False # when True, permE also adds random edges (chem GraphCL variant)
    cfg.pretrain.graphcl.node_drop_prob = 0.1 # node dropping probability
    cfg.pretrain.graphcl.feature_mask_prob = 0.1 # feature masking probability
    cfg.pretrain.graphcl.use_subgraph_aug = True # whether to use subgraph augmentation in addition to node/edge/feature perturbations
    cfg.pretrain.graphcl.subgraph_ratio = 0.2 # node-keep ratio for GraphCL subgraph augmentation
    cfg.pretrain.graphcl.temperature = 0.1 # temperature for contrastive loss (GraphCL default)
    cfg.pretrain.graphcl.proj_hidden = 256 # projection head hidden dimension
    # infograph specific options
    cfg.pretrain.infograph = CN()
    cfg.pretrain.infograph.measure = "JSD" # f-divergence measure (official InfoGraph uses JSD)
    cfg.pretrain.infograph.graph_pooling = "add" # pooling used for InfoGraph global embedding
    cfg.pretrain.infograph.use_layerwise = True # use concatenated per-layer node/graph reps (InfoGraph paper setting)
    cfg.pretrain.infograph.prior = False # enable prior matching regularization
    cfg.pretrain.infograph.gamma = 0.1 # prior regularization coefficient
    
    # ------------------------------------------------------------------------ #
    # Fine-tune options
    # ------------------------------------------------------------------------ #
    cfg.finetune = CN()
    cfg.finetune.method = "supervised"  # supervised, all_in_one, edgeprompt, gpf, gppt, graphprompt
    cfg.finetune.num_runs = 5  # number of finetuning runs with different seeds
    # finetune dataset options
    cfg.finetune.dataset = _default_dataset_cfg()
    cfg.finetune.dataset.fixed_split = (0.1, 0.1, 0.8) # train, val, test split ratios
    # general finetuning options
    cfg.finetune.epochs = 200 # maximum number of finetuning epochs
    cfg.finetune.early_stopping = 20 # early stopping patience (0 to disable)
    cfg.finetune.lr = 1e-3 # learning rate
    cfg.finetune.weight_decay = 0.0 # weight decay
    cfg.finetune.batch_size = 32  # used for induced tasks and graph-level tasks
    cfg.finetune.num_workers = 0 # number of data loading workers
    cfg.finetune.freeze_pretrained = True  # when True, freeze pretrained model parameters during finetuning
    cfg.finetune.checkpoint_dir = "finetuned_models" # directory to save checkpoints
    cfg.finetune.monitor_metric = "auto" # finetune monitor setting; auto is resolved by the shared monitoring policy
    cfg.finetune.skip_if_exists = True  # skip finetune run if checkpoint already exists
    cfg.finetune.run_all = False  # when True, finetune all checkpoints on all datasets except their pretraining dataset
    cfg.finetune.tasks_tsv = "slurm/run_all_finetune_experiments.tsv"  # dataset/task definitions for run_all
    cfg.finetune.pretrained_checkpoint = ""  # explicit checkpoint path override (optional)
    # supervised finetuning method-specific options
    cfg.finetune.supervised = CN()
    # all_in_one finetuning method-specific options
    cfg.finetune.all_in_one = CN()
    cfg.finetune.all_in_one.token_num = 10 # number of prompt tokens
    cfg.finetune.all_in_one.cross_prune = 0.1 # threshold for prompt->graph cross edges
    cfg.finetune.all_in_one.inner_prune = 0.3 # threshold for prompt inner edges
    cfg.finetune.all_in_one.total_epochs = 1000 # official-style total epoch budget for all_in_one (<=0 uses finetune.epochs)
    cfg.finetune.all_in_one.cache_answer_embeddings = True # speed-up: cache frozen-encoder embeddings during answering phase
    cfg.finetune.all_in_one.answer_with_softmax = False # optional strict parity mode: reference head is Linear+Softmax with CE
    cfg.finetune.all_in_one.bidirectional_cross_edges = None # None auto-selects by task (node-induced=True, graph=False); set bool to force behavior
    cfg.finetune.all_in_one.exclude_prompt_from_pooling = None # None auto-selects by task (node-induced=True, graph=False); set bool to force behavior
    cfg.finetune.all_in_one.answer_epoch = -1 # <=0 uses default auto schedule: node-induced/few-shot=50, graph-standard=5
    cfg.finetune.all_in_one.prompt_epoch = -1 # <=0 uses default auto schedule: node-induced/few-shot=50, graph-standard=1
    cfg.finetune.all_in_one.prompt_lr = 1e-6 # prompt optimizer learning rate
    cfg.finetune.all_in_one.prompt_weight_decay = None # None auto-selects by task (node-induced=5e-4, graph=finetune.weight_decay)
    cfg.finetune.all_in_one.answer_lr = 1e-3 # answering head optimizer learning rate
    cfg.finetune.all_in_one.answer_weight_decay = None # None auto-selects by task (node-induced=5e-4, graph=finetune.weight_decay)
    # edgeprompt finetuning method-specific options
    cfg.finetune.edgeprompt = CN()
    cfg.finetune.edgeprompt.plus = True # prefer anchor-conditioned prompts; set False to recover vanilla EdgePrompt
    cfg.finetune.edgeprompt.num_anchors = None # None uses baseline-style defaults: 10 for node tasks, 5 for graph tasks
    cfg.finetune.edgeprompt.add_self_loops = True # None keeps task-aware defaults (graph off; node resolved by backbone)
    cfg.finetune.edgeprompt.lr = cfg.finetune.lr # override when edgeprompt head needs a different optimizer step size
    cfg.finetune.edgeprompt.weight_decay = cfg.finetune.weight_decay # override when edgeprompt head needs different regularization
    cfg.finetune.edgeprompt.force_mean_pooling = True # use mean readout to match the released EdgePrompt downstream setup
    cfg.finetune.edgeprompt.pin_bn_eval_when_frozen = False # keep BN in train mode unless strict frozen-BN behavior is required
    cfg.finetune.edgeprompt.gin_message_relu = True # keep GIN message nonlinearity aligned with the baseline operator
    cfg.finetune.edgeprompt.gin_train_eps = True # allow eps to adapt during prompt tuning for baseline-like GIN behavior
    cfg.finetune.edgeprompt.use_official_node_subgraphs = True # use 2-hop node-induced subgraphs without aggressive size clipping
    cfg.finetune.edgeprompt.node_subgraph_hops = 2 # official node downstream extracts fixed 2-hop subgraphs
    cfg.finetune.edgeprompt.node_subgraph_min_size = 1 # avoid hop-expansion by minimum-size heuristics
    cfg.finetune.edgeprompt.node_subgraph_max_size = 100000 # keep almost all sampled nodes to avoid random truncation
    cfg.finetune.edgeprompt.disable_early_stopping = True # official scripts run a fixed epoch budget
    # gpf finetuning method-specific options
    cfg.finetune.gpf = CN()
    cfg.finetune.gpf.plus = False # when True use GPF+, otherwise GPF
    cfg.finetune.gpf.p_num = 5  # number of prompt bases for GPF+
    cfg.finetune.gpf.lr = cfg.finetune.lr  # base optimizer lr
    cfg.finetune.gpf.weight_decay = cfg.finetune.weight_decay  # base optimizer weight decay
    cfg.finetune.gpf.prompt_lr = cfg.finetune.gpf.lr  # prompt lr override
    cfg.finetune.gpf.prompt_weight_decay = cfg.finetune.gpf.weight_decay  # prompt weight decay override
    cfg.finetune.gpf.head_lr_scale = 1.0  # scale applied to base lr for prediction head
    cfg.finetune.gpf.head_lr = cfg.finetune.gpf.lr  # prediction head lr override
    cfg.finetune.gpf.head_weight_decay = cfg.finetune.gpf.weight_decay  # prediction head weight decay override
    cfg.finetune.gpf.head_layers = 2  # official GPF head depth knob (`num_layers` in upstream scripts)
    cfg.finetune.gpf.head_hidden_dim = 0  # <=0 uses input representation dim for hidden layers
    cfg.finetune.gpf.head_dropout = 0.0  # optional dropout between MLP head layers
    cfg.finetune.gpf.update_pretrained = False  # None -> auto(not freeze_pretrained), otherwise bool
    cfg.finetune.gpf.encoder_lr = cfg.finetune.gpf.lr  # encoder lr when update_pretrained=True
    cfg.finetune.gpf.encoder_weight_decay = cfg.finetune.gpf.weight_decay  # encoder weight decay when update_pretrained=True
    cfg.finetune.gpf.freeze_encoder_bn_when_frozen = True  # keep frozen encoder BN stats fixed during prompt tuning
    cfg.finetune.gpf.prefer_non_induced_node = True  # official GPF-style node tuning usually runs on original graph (non-induced)
    cfg.finetune.gpf.monitor_train_loss = False  # when True, auto monitor train_loss even when val split exists
    cfg.finetune.gpf.disable_early_stopping = True  # official GPF scripts run fixed epochs
    # gppt finetuning method-specific options
    cfg.finetune.gppt = CN()
    cfg.finetune.gppt.center_num = 0  # number of structure centers (0 -> auto: num_classes)
    cfg.finetune.gppt.lr = 2e-3  # official GPPT prompt optimizer learning rate
    cfg.finetune.gppt.weight_decay = 5e-4  # official GPPT prompt optimizer weight decay
    cfg.finetune.gppt.constraint_weight = 1e-2  # orthogonality regularization on task tokens
    cfg.finetune.gppt.force_freeze_encoder = False  # when True, tune prompts only and freeze encoder
    cfg.finetune.gppt.structure_mode = "concat"  # feature mode for StructureToken: node, neighbor, concat
    cfg.finetune.gppt.task_mode = "concat"  # feature mode for TaskToken heads: node, neighbor, concat
    cfg.finetune.gppt.add_self_loops = False  # add self-loops in mean-neighbor aggregation
    cfg.finetune.gppt.update_structure_every_step = True  # refresh structure token centers after each step
    cfg.finetune.gppt.update_structure_from_mask = True  # update structure centers from supervised nodes when masks exist
    cfg.finetune.gppt.kmeans_max_iter = 100  # max kmeans iterations for center updates
    cfg.finetune.gppt.kmeans_tol = 1e-4  # convergence tolerance for center updates
    cfg.finetune.gppt.kmeans_restarts = 1  # number of kmeans restarts per update
    cfg.finetune.gppt.use_sklearn_kmeans = True  # align with official implementation when sklearn is available
    cfg.finetune.gppt.kmeans_random_state = 0  # random_state for sklearn KMeans
    cfg.finetune.gppt.kmeans_n_init = 10  # n_init for sklearn KMeans
    # graphprompt finetuning method-specific options
    cfg.finetune.graphprompt = CN()
    cfg.finetune.graphprompt.plus = False  # when True use GraphPrompt+ stage-wise prompt parameterization
    cfg.finetune.graphprompt.p_num = 4  # number of prompt masks for official GraphPrompt+
    cfg.finetune.graphprompt.init = "xavier"  # xavier or identity
    cfg.finetune.graphprompt.init_std = 0.02  # std for identity init noise
    cfg.finetune.graphprompt.tau = 0.1  # temperature for prototype contrastive classification
    cfg.finetune.graphprompt.score_mode = "neg_distance"  # neg_distance, distance, or cosine
    cfg.finetune.graphprompt.loss_reduction = "mean"  # mean or sum
    cfg.finetune.graphprompt.train_center_mode = "batch"  # batch, train, or ema
    cfg.finetune.graphprompt.eval_center_mode = "train"  # train or batch
    cfg.finetune.graphprompt.graph_pooling = "sum"  # encoder, sum/add, mean, max, or target
    cfg.finetune.graphprompt.prompt_dropout = 0.0  # dropout on prompted embeddings during train (official uses 0)
    cfg.finetune.graphprompt.embedding_scalar = 1e3  # official GraphPrompt commonly scales prompted embeddings by 1e3
    cfg.finetune.graphprompt.update_pretrained = False  # official GraphPrompt/extension default keeps encoder frozen
    cfg.finetune.graphprompt.center_momentum = 0.9  # EMA momentum for prototype bank when train_center_mode=ema
    cfg.finetune.graphprompt.prompt_lr = 1e-3  # prompt optimizer learning rate
    cfg.finetune.graphprompt.prompt_weight_decay = 1e-5  # official GraphPrompt default weight decay
    cfg.finetune.graphprompt.encoder_lr = 1e-3  # encoder optimizer learning rate when not frozen
    cfg.finetune.graphprompt.encoder_weight_decay = 0.0  # encoder optimizer weight decay when not frozen
    
    # ------------------------------------------------------------------------ #
    # Other baseline options
    # ------------------------------------------------------------------------ #
    cfg.baseline = CN()
    cfg.baseline.method = ""  # baseline method name (dispatch key); set explicitly in run_baseline.py
    # anygraph baseline options (grouped by stage)
    cfg.baseline.anygraph = CN()
    # execution control
    cfg.baseline.anygraph.execution = CN()
    cfg.baseline.anygraph.execution.step = "all"  # conversion, train, eval, or all
    # shared paths/resources
    cfg.baseline.anygraph.paths = CN()
    cfg.baseline.anygraph.paths.dataset_root = "data/datasets"  # AGAE dataset root
    cfg.baseline.anygraph.paths.split_root = "data/splits"  # AGAE split root
    cfg.baseline.anygraph.paths.out_root = "data/anygraph_data"  # converted AnyGraph-format root
    # conversion stage
    cfg.baseline.anygraph.conversion = CN()
    cfg.baseline.anygraph.conversion.skip = False  # when True, reuse existing converted AnyGraph data
    cfg.baseline.anygraph.conversion.task = "auto"  # auto, node, link, or all
    cfg.baseline.anygraph.conversion.dataset = ""  # comma-separated dataset names, list, or empty
    cfg.baseline.anygraph.conversion.dataset_file = "data/available_node_datasets.tsv"  # used when dataset is empty
    cfg.baseline.anygraph.conversion.mask_col = 0  # split mask column index
    cfg.baseline.anygraph.conversion.feat_reduction = False  # apply feature reduction during conversion
    cfg.baseline.anygraph.conversion.l1_normalize_features = True  # row-wise L1 normalize link-task features
    cfg.baseline.anygraph.conversion.feat_dim = 100  # feature SVD output dimension for link conversion
    cfg.baseline.anygraph.conversion.seeds = list(cfg.seeds)  # split seeds processed during AnyGraph conversion
    cfg.baseline.anygraph.conversion.edge_splits = [tuple(split) for split in cfg.data_preparation.edge_task_splits]  # edge splits to convert
    cfg.baseline.anygraph.conversion.node_splits = [tuple(split) for split in cfg.data_preparation.node_task_splits]  # node splits to convert
    cfg.baseline.anygraph.conversion.edge_eval_payload_name = "agae_edge_eval_payload.pt"  # fixed val/test edge payload name
    cfg.baseline.anygraph.conversion.node_output_feat_dim = 128  # output feature dim for node conversion
    cfg.baseline.anygraph.conversion.emit_node_val = True  # emit val_mat.pkl for node conversion
    # anygraph train stage
    cfg.baseline.anygraph.train = CN()
    cfg.baseline.anygraph.train.mode = "both"  # link, node, or both
    cfg.baseline.anygraph.train.epoch = 100  # AnyGraph epoch budget for the train step
    cfg.baseline.anygraph.train.skip_if_exists = True  # skip training routes whose checkpoints already exist
    cfg.baseline.anygraph.train.load_link_model = ""  # optional link checkpoint for resume training
    cfg.baseline.anygraph.train.load_node_model = ""  # optional node checkpoint for resume training
    cfg.baseline.anygraph.train.save_link_path = "anygraph_link_run"  # link save-path tag
    cfg.baseline.anygraph.train.save_node_path = "anygraph_node_run"  # node save-path tag
    cfg.baseline.anygraph.train.dataset = _default_dataset_cfg()
    cfg.baseline.anygraph.train.dataset.name = None  # source dataset name(s); None means all converted datasets
    cfg.baseline.anygraph.train.dataset.fixed_split = None  # fixed split used to resolve converted datasets across cfg.seeds
    cfg.baseline.anygraph.train.link_dataset_setting = ""  # explicit AnyGraph link dataset_setting for training
    cfg.baseline.anygraph.train.node_dataset_setting = ""  # explicit AnyGraph node dataset_setting for training
    cfg.baseline.anygraph.train.mixed_dataset_setting = ""  # shared dataset_setting split into node/link via conversion index
    # anygraph evaluation stage
    cfg.baseline.anygraph.eval = CN()
    cfg.baseline.anygraph.eval.mode = "both"  # link, node, or both
    cfg.baseline.anygraph.eval.load_link_model = ""  # explicit link checkpoint tag to evaluate
    cfg.baseline.anygraph.eval.load_node_model = ""  # explicit node checkpoint tag to evaluate
    cfg.baseline.anygraph.eval.dataset = _default_dataset_cfg()
    cfg.baseline.anygraph.eval.dataset.name = None  # source dataset name(s); None falls back to train.dataset.name or all converted datasets
    cfg.baseline.anygraph.eval.dataset.fixed_split = None  # fixed split used to resolve target evaluation datasets across cfg.seeds
    cfg.baseline.anygraph.eval.link_dataset_setting = ""  # explicit AnyGraph link dataset_setting for evaluation
    cfg.baseline.anygraph.eval.node_dataset_setting = ""  # explicit AnyGraph node dataset_setting for evaluation
    cfg.baseline.anygraph.eval.mixed_dataset_setting = ""  # shared dataset_setting split into node/link via conversion index
    # prediction/evaluation stage
    cfg.baseline.anygraph.prediction = CN()
    cfg.baseline.anygraph.prediction.link = CN()
    cfg.baseline.anygraph.prediction.link.eval_protocol = "agae"  # AnyGraph link eval protocol
    cfg.baseline.anygraph.prediction.link.edge_eval_threshold_mode = "val_best_acc"  # val_best_acc or zero
    cfg.baseline.anygraph.prediction.link.edge_eval_repeat_times = 5  # repeated edge eval rounds
    cfg.baseline.anygraph.prediction.link.tst_epoch = 1  # passthrough to link runner when mode=link
    cfg.baseline.anygraph.prediction.link.topk = 20  # passthrough to link runner when mode=link
    cfg.baseline.anygraph.prediction.node = CN()
    cfg.baseline.anygraph.prediction.node.tst_epoch = 1  # passthrough to node runner when mode=node
    cfg.baseline.anygraph.prediction.node.assignment = "top1"  # passthrough to node runner when mode=node
    # output/report stage
    cfg.baseline.anygraph.output = CN()
    cfg.baseline.anygraph.output.link_csv = "outputs/anygraph/anygraph_link_eval.csv"
    cfg.baseline.anygraph.output.node_csv = "outputs/anygraph/anygraph_node_eval.csv"
    cfg.baseline.anygraph.output.report_csv = "outputs/anygraph/anygraph_baseline_report.csv"
    # optional passthrough args appended to both link/node runner scripts
    cfg.baseline.anygraph.extra_args = []

    # ------------------------------------------------------------------------ #
    # Context generation options
    # ------------------------------------------------------------------------ #
    cfg.context = CN()
    cfg.context.method = "llm" # llm, bonsai, sgdc
    cfg.context.output_dir = "generated_context" # directory to save generated context
    # dataset selection shared by context generation stages
    cfg.context.dataset = _default_dataset_cfg()
    cfg.context.dataset.name = None # single-dataset target when run_all is False
    cfg.context.dataset.task_level = "node" # node, graph, or edge
    cfg.context.dataset.induced = True # use induced subgraphs for node/edge datasets when available
    cfg.context.dataset.fixed_split = (0.8, 0.1, 0.1) # shared split for condensation stages
    cfg.context.dataset.available_datasets = "" # optional combined dataset list (one per line or TSV with extra columns)
    # context options for LLM-based generation
    cfg.context.llm = CN()
    cfg.context.llm.target = "all" # dataset, model, all
    cfg.context.llm.model_name = "gpt-5.2" # language model to use for context generation: gpt-4o-mini, gpt-5.2
    cfg.context.llm.temperature = 0.2 # sampling temperature
    cfg.context.llm.run_all = False # when True, iterate through available datasets/architectures for LLM generation
    cfg.context.llm.skip_if_exists = True # skip writing context files that already exist
    cfg.context.llm.skip_induced_generation = True # avoid induced subgraph materialization during dataset context generation
    cfg.context.llm.arch_root = "" # optional architecture root override (defaults to pretrain.checkpoint_dir)
    cfg.context.llm.run_name = "" # optional pretrained run-name override for model context generation
    cfg.context.llm.available_datasets_tsv = "" # optional dataset/task list for LLM run_all
    # context options for Bonsai graph condensation-based generation
    cfg.context.bonsai = CN()
    cfg.context.bonsai.stage = "condense" # condense, eval, all
    cfg.context.bonsai.run_all = False # iterate all node-level datasets from context.bonsai.dataset.available_node_datasets
    cfg.context.bonsai.dataset = _default_dataset_cfg() # shared dataset config for bonsai stages
    cfg.context.bonsai.dataset.name = None # single-dataset target when run_all is False
    cfg.context.bonsai.dataset.task_level = "node" # bonsai only supports node-level datasets
    cfg.context.bonsai.dataset.induced = False # bonsai uses the original full graph
    cfg.context.bonsai.dataset.feat_reduction = False # official Bonsai uses raw node features by default
    cfg.context.bonsai.dataset.fixed_split = (0.8, 0.1, 0.1) # split ratios when masks are missing
    cfg.context.bonsai.out_root = "" # default: <context.output_dir>/graph_condensation/bonsai
    cfg.context.bonsai.target_size_frac = 0.01 # condensation budget ratio
    cfg.context.bonsai.budget_mb = 0 # optional byte-budget in MB (overrides target_size_frac when >0)
    cfg.context.bonsai.k = 5 # kNN parameter for Bonsai node ranking
    cfg.context.bonsai.neighbor_hops = 1 # neighborhood hops to include around seed nodes
    cfg.context.bonsai.max_train_nodes = 0 # cap train nodes when computing pairwise distances (0 disables cap)
    cfg.context.bonsai.knn_block_q = 1024 # query block size for blockwise kNN
    cfg.context.bonsai.knn_block_c = 8192 # candidate block size for blockwise kNN
    cfg.context.bonsai.knn_device = "cpu" # cpu or cuda for blockwise kNN
    cfg.context.bonsai.budget_mode = "node_frac" # node_frac, bonsai, bytes
    cfg.context.bonsai.min_nodes = 0 # minimum nodes to keep when using node_frac mode
    cfg.context.bonsai.min_seed_nodes = 0 # minimum number of seed nodes before neighborhood expansion
    cfg.context.bonsai.csv_out = "" # optional CSV path to append per-dataset condensation summary
    cfg.context.bonsai.split_mode = "pretrain" # pretrain, auto, or bonsai split handling
    cfg.context.bonsai.mask_col = 0 # split column index for 2D masks [N, S]
    # context options for Bonsai evaluation of condensed datasets
    cfg.context.bonsai.eval = CN()
    cfg.context.bonsai.eval.run_all = False # evaluate all node datasets from context.bonsai.dataset.available_node_datasets
    cfg.context.bonsai.eval.dataset = _default_dataset_cfg() # explicit dataset override for bonsai evaluation
    cfg.context.bonsai.eval.dataset.name = None # single-dataset target when run_all is False
    cfg.context.bonsai.eval.dataset.task_level = "node" # bonsai evaluation only supports node-level datasets
    cfg.context.bonsai.eval.dataset.induced = False # bonsai evaluation uses the original full graph
    cfg.context.bonsai.eval.dataset.feat_reduction = False # match condensation feature space unless explicitly overridden
    cfg.context.bonsai.eval.dataset.fixed_split = None # when unset, reuse context.bonsai.dataset.fixed_split
    cfg.context.bonsai.eval.condensed_root = "" # default: <context.output_dir>/graph_condensation/bonsai
    cfg.context.bonsai.eval.condensed_dir = "" # explicit condensed directory override
    cfg.context.bonsai.eval.target_size_frac = 0.01 # condensed ratio tag used to locate outputs
    cfg.context.bonsai.eval.budget_mb = 0 # budget tag used to locate outputs when non-zero
    # cfg.seeds and cfg.device are reused for Bonsai evaluation runs
    cfg.context.bonsai.eval.backend = "sparse_gcn" # gcnconv or sparse_gcn
    cfg.context.bonsai.eval.mode = "neighbor_exact" # full or neighbor_exact for inference
    cfg.context.bonsai.eval.batch_size = 4096 # evaluation batch size for neighbor loaders
    cfg.context.bonsai.eval.num_workers = 0 # workers for evaluation loaders
    cfg.context.bonsai.eval.baseline_train_mode = "neighbor_sampled" # full, neighbor_exact, neighbor_sampled
    cfg.context.bonsai.eval.baseline_train_batch_size = 4096 # baseline train batch size
    cfg.context.bonsai.eval.baseline_train_num_workers = 0 # baseline train loader workers
    cfg.context.bonsai.eval.baseline_num_neighbors = "15,10" # fanout list for sampled baseline training
    cfg.context.bonsai.eval.condensed_train_mode = "neighbor_sampled" # full, neighbor_exact, neighbor_sampled
    cfg.context.bonsai.eval.condensed_train_batch_size = 4096 # condensed train batch size
    cfg.context.bonsai.eval.condensed_train_num_workers = 0 # condensed train loader workers
    cfg.context.bonsai.eval.condensed_num_neighbors = "15,10" # fanout list for sampled condensed training
    cfg.context.bonsai.eval.model = "gcn" # gcn, gat, gin, mlp
    cfg.context.bonsai.eval.num_layers = 2 # encoder layer count for evaluation
    cfg.context.bonsai.eval.hidden_dim = 256 # hidden dimension for evaluation models
    cfg.context.bonsai.eval.out_dim = 256 # output embedding dimension for evaluation models
    cfg.context.bonsai.eval.dropout = 0.5 # dropout in evaluation models
    cfg.context.bonsai.eval.lr = 1e-3 # learning rate for evaluation optimization
    cfg.context.bonsai.eval.weight_decay = 0.0 # weight decay for evaluation optimization
    cfg.context.bonsai.eval.epochs = 200 # epochs for baseline/condensed training
    cfg.context.bonsai.eval.mask_col = 0 # split column index for 2D masks [N, S]
    cfg.context.bonsai.eval.split_mode = "pretrain" # pretrain, auto, bonsai
    cfg.context.bonsai.eval.preset = False # enable original Bonsai-style preset run
    cfg.context.bonsai.eval.runs = 5 # repeated runs when bonsai_preset is enabled
    cfg.context.bonsai.eval.csv_out = "" # optional CSV path for evaluation summary
    # context options for SGDC graph condensation-based generation
    cfg.context.sgdc = CN()
    cfg.context.sgdc.stage = "condense" # pretrain, condense, all
    cfg.context.sgdc.run_all = False # iterate all graph datasets from context.sgdc.dataset.available_graph_datasets
    cfg.context.sgdc.dataset = _default_dataset_cfg() # shared dataset config for SGDC stages
    cfg.context.sgdc.dataset.name = None # single-dataset target when run_all is False
    cfg.context.sgdc.dataset.task_level = "graph" # SGDC only supports graph-level datasets
    cfg.context.sgdc.dataset.induced = False # SGDC uses full input graphs
    cfg.context.sgdc.dataset.feat_reduction = False # official SGDC uses raw node features by default
    cfg.context.sgdc.dataset.fixed_split = (0.8, 0.1, 0.1) # train, val, test split
    # SGDC teacher pretraining options
    cfg.context.sgdc.pretrain = CN()
    cfg.context.sgdc.pretrain.run_all = False # pretrain teachers for all datasets from context.sgdc.dataset.available_graph_datasets
    cfg.context.sgdc.pretrain.dataset = _default_dataset_cfg() # explicit dataset override for SGDC pretrain
    cfg.context.sgdc.pretrain.dataset.name = None # single-dataset target when run_all is False
    cfg.context.sgdc.pretrain.dataset.task_level = "graph" # SGDC pretraining only supports graph-level datasets
    cfg.context.sgdc.pretrain.dataset.induced = False # SGDC pretraining uses full graphs
    cfg.context.sgdc.pretrain.dataset.feat_reduction = False # match official SGDC teacher pretrain on raw graph features
    cfg.context.sgdc.pretrain.dataset.fixed_split = None # when unset, reuse context.sgdc.dataset.fixed_split
    cfg.context.sgdc.pretrain.epochs = 50 # teacher pretrain epochs
    cfg.context.sgdc.pretrain.batch_size = 128 # teacher pretrain batch size
    cfg.context.sgdc.pretrain.lr = 1e-3 # teacher pretrain learning rate
    cfg.context.sgdc.pretrain.weight_decay = 2e-5 # teacher pretrain weight decay
    cfg.context.sgdc.pretrain.train_model = "gin" # teacher encoder family for checkpoint naming; current pretrain implementation supports gin
    cfg.context.sgdc.pretrain.nhid = 128 # SGDC teacher hidden dimension
    cfg.context.sgdc.pretrain.layers = 3 # SGDC teacher number of layers
    cfg.context.sgdc.pretrain.pooling = "mean" # mean or sum pooling for teacher pretrain
    cfg.context.sgdc.pretrain.aug = "dnodes" # augmentation mode for contrastive teacher pretrain
    cfg.context.sgdc.pretrain.save_ckpt = "" # optional explicit checkpoint path (default: <dataset>_<train_model>_pretrain.cpt)
    cfg.context.sgdc.pretrain.default_ckpt_dir = "" # default: outputs/sgdc/pretrain
    # SGDC condensation options
    cfg.context.sgdc.condense = CN()
    cfg.context.sgdc.condense.model = "gin" # SGDC model family shared by teacher/evaluator: gin or gin_var
    cfg.context.sgdc.condense.nhid = 128 # SGDC hidden dimension
    cfg.context.sgdc.condense.layers = 3 # SGDC encoder layers
    cfg.context.sgdc.condense.drop_ratio = 0.5 # SGDC dropout ratio
    cfg.context.sgdc.condense.outer_bs = 64 # SGDC outer-loop batch size
    cfg.context.sgdc.condense.outer_wd = 0.0 # weight decay for outer optimizer
    cfg.context.sgdc.condense.run_all = False # condense all datasets from context.sgdc.dataset.available_graph_datasets
    cfg.context.sgdc.condense.dataset = _default_dataset_cfg() # explicit dataset override for SGDC condensation
    cfg.context.sgdc.condense.dataset.name = None # single-dataset target when run_all is False
    cfg.context.sgdc.condense.dataset.task_level = "graph" # SGDC condensation only supports graph-level datasets
    cfg.context.sgdc.condense.dataset.induced = False # SGDC condensation uses full graphs
    cfg.context.sgdc.condense.dataset.feat_reduction = False # match official SGDC condensation on raw graph features
    cfg.context.sgdc.condense.dataset.fixed_split = None # when unset, reuse context.sgdc.dataset.fixed_split
    cfg.context.sgdc.condense.out_root = "" # default: <context.output_dir>/graph_condensation/sgdc
    cfg.context.sgdc.condense.budget = 10 # task-adaptive synthetic graph budget base: per class for classification, total graphs for regression when budget_total=0
    cfg.context.sgdc.condense.budget_total = 0 # total synthetic graphs budget (>0 overrides the task-dependent default budget from budget)
    cfg.context.sgdc.condense.initialize = "Herding" # Herding, KCenter, Random
    cfg.context.sgdc.condense.pretrained_ckpt = "" # optional explicit teacher checkpoint override; default is inferred from dataset + model under outputs/sgdc/pretrain
    cfg.context.sgdc.condense.outer_lr = 1e-4 # learning rate for outer optimizer
    cfg.context.sgdc.condense.outer_epoch = 50 # outer optimization epochs
    cfg.context.sgdc.condense.eval_every = 1 # evaluate synthetic set every N outer epochs
    cfg.context.sgdc.condense.outer_grad_norm = 0.1 # gradient clipping for outer loop
    cfg.context.sgdc.condense.online_iteration = 50 # online update iterations per outer step
    cfg.context.sgdc.condense.online_lr = 0.01 # online learning rate
    cfg.context.sgdc.condense.online_wd = 0.0 # online weight decay
    cfg.context.sgdc.condense.online_batch_size = 128 # online batch size
    cfg.context.sgdc.condense.online_min_iteration = 20 # minimum online iterations before early stop
    cfg.context.sgdc.condense.num_models = 1 # number of models in the online pool
    cfg.context.sgdc.condense.pre_epoch = 100 # warmup pretrain epochs
    cfg.context.sgdc.condense.pre_lr = 0.01 # warmup pretrain learning rate
    cfg.context.sgdc.condense.pre_wd = 2e-5 # warmup pretrain weight decay
    cfg.context.sgdc.condense.test_epoch = 300 # downstream evaluation epochs
    cfg.context.sgdc.condense.test_bs = 128 # downstream evaluation batch size
    cfg.context.sgdc.condense.test_lr = 1e-3 # downstream evaluation learning rate
    cfg.context.sgdc.condense.test_wd = 2e-5 # downstream evaluation weight decay
    cfg.context.sgdc.condense.metric = "acc" # acc, rocauc, rmse, mae
    cfg.context.sgdc.condense.scale = "uniform" # GNTK scaling: uniform or degree
    cfg.context.sgdc.condense.reg_lambda = 1e-6 # GNTK regularization
    cfg.context.sgdc.condense.orth_reg = 1e-3 # orthogonality regularization
    
    # ------------------------------------------------------------------------ #
    # Router dataset generation options
    # ------------------------------------------------------------------------ #
    cfg.router_dataset = CN()
    cfg.router_dataset.freeze_pretrained = True  # when True, freeze pretrained model during router dataset generation
    cfg.router_dataset.epochs = 500 # maximum number of training epochs
    cfg.router_dataset.early_stopping = 50 # early stopping patience
    cfg.router_dataset.lr = 1e-3 # learning rate
    cfg.router_dataset.weight_decay = 0.0 # weight decay
    cfg.router_dataset.batch_size = 128 # used for induced tasks and graph-level tasks
    cfg.router_dataset.num_workers = 0 # number of data loading workers
    cfg.router_dataset.output_dir = "router_datasets/records/" # directory to save router datasets
    cfg.router_dataset.skip_if_exists = True  # skip run if dataset already exists
    # router target dataset options
    cfg.router_dataset.target_dataset = _default_dataset_cfg()
    cfg.router_dataset.target_dataset.name = None  # when None, iterate over all available datasets
    cfg.router_dataset.target_dataset.fixed_split = (0.1, 0.1, 0.8) # train, val, test split ratios

    # ------------------------------------------------------------------------ #
    # Router graph construction options
    # ------------------------------------------------------------------------ #
    cfg.router_graph = CN()
    cfg.router_graph.output_dir = "router_datasets/graph" # directory to save router graphs
    cfg.router_graph.strict_duplicates = False  # when True, raise on duplicate edges instead of reporting them
    cfg.router_graph.topk_expert = None  # when set, keep only top-k experts per subgraph by prob_correct
    cfg.router_graph.topk_neg_expert = False  # when True, keep topk_expert lowest-probability experts per subgraph by prob_correct

    # ------------------------------------------------------------------------ #
    # Router expert selection options
    # ------------------------------------------------------------------------ #
    cfg.router_selection = CN()
    cfg.router_selection.graph_dir = "router_datasets/graph" # directory for router graphs (split tag auto-appended)
    cfg.router_selection.graph_split = (0.8, 0.1, 0.1) # split tag for router graph
    cfg.router_selection.graph_topk_expert = 5 # when set, uses graph with top-k experts per subgraph
    cfg.router_selection.graph_topk_neg_expert = True # when True, uses graph with top-k negative experts per subgraph
    cfg.router_selection.output_dir = "router_selection/outputs" # directory to save outputs
    cfg.router_selection.checkpoint_dir = "router_selection/checkpoints" # directory to save checkpoints
    cfg.router_selection.run_name = "" # name for the current run
    cfg.router_selection.predict_only = False # skip training and only run routing prediction
    cfg.router_selection.checkpoint_path = "" # optional checkpoint to load when predict_only is True
    cfg.router_selection.predict_all_subgraphs = False # when True, route all subgraphs instead of split-only
    cfg.router_selection.train_datasets = "" # comma-separated list of datasets for training
    cfg.router_selection.test_dataset = "" # test dataset
    cfg.router_selection.valid_edge_ratio = 0.1 # ratio of train edges held out for validation
    cfg.router_selection.embed_dim = 128 # dimension of node features used for router training
    # context options
    cfg.router_selection.context_mode = "text" # text, condensation, concat
    cfg.router_selection.context_source = "gpt-5.2" # context subfolder (gpt-4o or gpt-5.2)
    cfg.router_selection.text_embed_model = "bert-base-uncased" # text encoder model name
    cfg.router_selection.text_embed_cache_dir = "router_selection/cache/text_embeddings" # text embedding cache
    cfg.router_selection.text_max_length = 256 # max tokens for text encoder
    cfg.router_selection.text_batch_size = 128 # batch size for text encoder
    cfg.router_selection.condensation = CN()
    cfg.router_selection.condensation.method = "bonsai" # bonsai or sgdc
    cfg.router_selection.condensation.root_dir = "data/datasets/bonsai" # condensation output root
    cfg.router_selection.condensation.tag = "" # optional tag folder (auto-pick latest if empty)
    cfg.router_selection.condensation.use_processed = False # use processed/data.pt when raw is missing
    cfg.router_selection.condensation.input_dim = 0 # 0 means infer from first graph
    cfg.router_selection.condensation.hidden_dim = 128 # hidden size for graph encoder
    cfg.router_selection.condensation.embed_dim = 128 # output embedding dim for dataset graphs
    cfg.router_selection.condensation.num_layers = 2 # GCN layers for graph encoder
    cfg.router_selection.condensation.dropout = 0.1 # dropout for graph encoder
    cfg.router_selection.condensation.pool = "mean" # mean, sum, or max pooling
    cfg.router_selection.condensation.max_graphs = 0 # limit graphs per dataset (0 = all)
    cfg.router_selection.query_feat_svd_dim = 100 # feature SVD dim for query subgraphs
    cfg.router_selection.query_struct_svd_dim = 100 # structure SVD dim for query subgraphs
    cfg.router_selection.structure_svd_matrix = "adjacency" # adjacency or laplacian
    cfg.router_selection.structure_svd_cache_dir = "router_selection/cache/svd" # SVD cache directory
    # router selection model options
    cfg.router_selection.gnn_layers = 2 # GNN layers for router graph
    cfg.router_selection.gnn_dropout = 0.1 # dropout for router GNN
    cfg.router_selection.scorer = "mlp"  # dot or mlp
    cfg.router_selection.mlp_hidden_dim = 128 # only used if scorer is mlp
    # router selection training options
    cfg.router_selection.monitor_metric = "mae" # mae or loss_reg
    cfg.router_selection.epochs = 500 # number of training epochs
    cfg.router_selection.early_stopping = 50 # early stopping patience
    cfg.router_selection.lr = 3e-4 # learning rate
    cfg.router_selection.weight_decay = 1e-4 # weight decay
    cfg.router_selection.batch_size = 128 # training batch size / mask iterations for BCE
    cfg.router_selection.num_workers = 0 # number of data loading workers
    cfg.router_selection.topk = 5 # number of experts to recommend

    # ------------------------------------------------------------------------ #
    # Router prediction options
    # ------------------------------------------------------------------------ #
    cfg.router_prediction = CN()
    cfg.router_prediction.mode = "cluster"  # cluster or head
    cfg.router_prediction.router_output_path = "" # path to router selection output
    cfg.router_prediction.graph_dir = "router_datasets/graph" # directory for router graph metadata
    cfg.router_prediction.output_dir = "router_prediction" # directory to save predictions
    cfg.router_prediction.run_name = "" # name for the current run
    cfg.router_prediction.pretrained_dir = "pretrained_models" # directory containing expert checkpoints
    cfg.router_prediction.embed_agg = "concat" # concat or mean
    cfg.router_prediction.cache_features = True # whether to cache query features
    cfg.router_prediction.train_dataset = "" # training dataset names (comma-separated)
    # router prediction test dataset options
    cfg.router_prediction.test_dataset = _default_dataset_cfg()
    cfg.router_prediction.test_dataset.name = None # name of the test dataset
    cfg.router_prediction.test_dataset.task_level = "node" # node, graph, or edge
    cfg.router_prediction.test_dataset.induced = True # whether to use induced subgraphs
    cfg.router_prediction.test_dataset.fixed_split = (0.1, 0.1, 0.8) # train, val, test split ratios
    cfg.router_prediction.test_dataset.task_type = "classification" # classification or regression
    cfg.router_prediction.test_dataset.k_expert_feat = None  # None means use all selected features, 0 disables expert features
    cfg.router_prediction.test_dataset.keep_query_features = False  # when True, append raw query features to expert embeddings
    # clustering configuration
    cfg.router_prediction.cluster_method = "kmeans" # kmeans
    cfg.router_prediction.calibration_ratio = 0.0 # ratio of calibration data from training set
    cfg.router_prediction.weak_labels_path = "" # path to weak labels file
    # head configuration
    cfg.router_prediction.head = CN() # head configuration
    cfg.router_prediction.head.type = "mlp" # mlp
    cfg.router_prediction.head.hidden_dim = 128 # MLP hidden size
    cfg.router_prediction.head.num_layers = 2 # number of MLP layers
    cfg.router_prediction.head.dropout = 0.2 # dropout rate for MLP
    # head training options
    cfg.router_prediction.lr = 1e-3 # learning rate for MLP
    cfg.router_prediction.weight_decay = 0.0 # weight decay
    cfg.router_prediction.epochs = 500 # number of training epochs
    cfg.router_prediction.early_stop_patience = 50 # early stopping patience
    cfg.router_prediction.batch_size = 128 # training batch size

    return cfg


def update_cfg(cfg: CN, args_str: str=None)-> CN:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        default="",
        metavar="FILE", 
        help="Path to config file"
    )
    # opts arg needs to match set_cfg
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(cfg=CN())
