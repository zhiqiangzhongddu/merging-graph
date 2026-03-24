import torch
import torch.nn.functional as F

from ..model.wrapper import get_model


def run(
        args, device,
        model_name,
        syn_loader
):

    """PRETRAIN"""
    # model and opt
    model = get_model(model_name, args, args.num_target_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.pre_lr, weight_decay=args.pre_wd)

    # pretrain
    # print("Pretrain")
    model.train()
    for _ in range(1, args.pre_epoch + 1):
        for data_syn in syn_loader:
            x, edge_index = data_syn.x.to(device), data_syn.edge_index.to(device)
            batch, y = data_syn.batch.to(device), data_syn.y.to(device)
            if model_name == 'gin':
                loss = F.mse_loss(model(edge_index, x, batch), y)
            elif model_name == 'gin_var':
                edge_attr = data_syn.edge_attr.to(device)
                edge_weight = data_syn.edge_weight if hasattr(data_syn, 'edge_weight') else None
                x, _ = model(batch, x, edge_index, edge_attr, edge_weight)
                loss = F.mse_loss(x, y)
            else:
                raise NotImplementedError
        # print(loss)
        # update
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model



