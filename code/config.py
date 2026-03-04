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
    ds.task_type = "classification"  # classification or regression
    ds.task_level = "node"  # node or graph or edge
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
    ds.induced_root = "data/induced_subgraphs" # output directory for induced subgraphs
    ds.force_reload_raw = False  # when True, force reload raw data (reprocess) even if processed cache exists
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
    cfg.data_preparation.target_datasets = [] # when set as a list, only prepare these datasets (comma-separated names); when set to a file path, read dataset names from the file (one per line)
    cfg.data_preparation.node_task_splits = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (100, 0.0, 1.0), (20, 0.0, 1.0), (5, 0.0, 1.0), (1, 0.0, 1.0)]
    cfg.data_preparation.graph_task_splits = [(0.8, 0.1, 0.1), (0.1, 0.1, 0.8), (100, 0.0, 1.0), (20, 0.0, 1.0), (5, 0.0, 1.0), (1, 0.0, 1.0)]
    cfg.data_preparation.edge_task_splits = [(0.1, 0.05, 0.1)]
    cfg.data_preparation.generate_edge_level = True  # whether to generate edge-level information for node-level datasets
    
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
    cfg.pretrain.context_pred.temperature = 0.2 # legacy option (unused in cbow/skipgram mode)
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
    cfg.pretrain.graphcl.rw_restart_prob = 0.15 # random walk restart probability
    cfg.pretrain.graphcl.rw_num_seeds_ratio = 0.3 # ratio of seed nodes for random walk
    cfg.pretrain.graphcl.temperature = 0.1 # temperature for contrastive loss (GraphCL default)
    cfg.pretrain.graphcl.proj_hidden = 256 # projection head hidden dimension
    # infograph specific options
    cfg.pretrain.infograph = CN()
    cfg.pretrain.infograph.measure = "JSD" # f-divergence measure (official InfoGraph uses JSD)
    cfg.pretrain.infograph.graph_pooling = "add" # pooling used for InfoGraph global embedding
    cfg.pretrain.infograph.use_layerwise = True # use concatenated per-layer node/graph reps (InfoGraph paper setting)
    cfg.pretrain.infograph.prior = False # enable prior matching regularization
    cfg.pretrain.infograph.gamma = 0.1 # prior regularization coefficient
    cfg.pretrain.infograph.temperature = 0.2 # retained for backward compatibility (unused by JSD objective)
    
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
    cfg.finetune.monitor_metric = "auto"
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
    cfg.finetune.gpf.optimizer = "adam"  # adam or adamw
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
    cfg.finetune.gppt.concat_neighbor = True  # legacy switch: task_mode=concat when True, otherwise neighbor
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
    cfg.finetune.graphprompt.amsgrad = True  # official GraphPrompt uses AdamW with amsgrad=True
    cfg.finetune.graphprompt.encoder_lr = 1e-3  # encoder optimizer learning rate when not frozen
    cfg.finetune.graphprompt.encoder_weight_decay = 0.0  # encoder optimizer weight decay when not frozen
    
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
