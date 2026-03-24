import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class G_GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nconvs=3, dropout=0, pooling='mean', **kwargs):
        super().__init__()

        self.convs = torch.nn.ModuleList([])
        self.convs.append(GINConv(torch.nn.Linear(input_dim, hidden_dim), train_eps=True))

        for _ in range(nconvs - 1):
            self.convs.append(GINConv(torch.nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.project = Linear(hidden_dim, output_dim)
        self.norms = torch.nn.ModuleList([])
        for _ in range(nconvs):
            if nconvs == 1:
                norm = torch.nn.Identity()
            else:
                norm = torch.nn.BatchNorm1d(hidden_dim)
            self.norms.append(norm)

        self.dropout = dropout
        self.pooling = pooling

    def forward(self, edge_index, x, batch, edge_weight=None):

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.norms[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight)

        if self.pooling == 'mean':
            x = global_mean_pool(x, batch=batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch=batch)

        return x

    def get_emb(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if isinstance(data, list):
                    data = data[0].to(device)
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x = self.forward(edge_index, x, batch, None)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    # only for SSL!!!
    def loss_cal(self, x, x_aug):
        # NT-Xent-style loss with small epsilon to avoid NaN when norms are zero.
        T = 0.2
        eps = 1e-7
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1) + eps
        x_aug_abs = x_aug.norm(dim=1) + eps

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / torch.clamp(sim_matrix.sum(dim=1) - pos_sim, min=eps)
        loss = - torch.log(loss + eps).mean()
        return loss
