import numpy as np

import torch

from torch_geometric.utils import to_dense_batch, to_dense_adj

from ..utils import dense2batch


def discretize(adj, device, tau=1.0, train=True, epoch=0):
    if train:
        """ Gumbel sigmoid sampling """
        num_graphs, num_nodes = adj.shape[0], adj.shape[1]
        delta = torch.zeros_like(adj).to(device)
        vals = torch.rand(num_graphs * num_nodes * (num_nodes+1) // 2).to(device)
        vals = vals.view(num_graphs, -1)
        i, j = torch.triu_indices(num_nodes, num_nodes)
        delta[:, i, j] = vals
        delta.transpose(1,2)[:, i, j] = vals
        adj = torch.log(delta) - torch.log(1-delta) + adj

        """ an annealing schedule for the temperature """
        temp_start = tau
        temp_end = 0.01
        t = max(temp_start*(temp_end/temp_start)**(epoch/200), temp_end)

        adj = torch.sigmoid(adj/t)

    else:
        """ Using the threshold 0.5 to discretize the adjacency matrix """
        adj = torch.sigmoid(adj)
        adj[adj> 0.5] = 1
        adj[adj<= 0.5] = 0
    return adj

def run(args, device, real_loader, model_pool, data_syn, outer_opt, gntk):

    # prepare model
    idx = np.random.randint(args.num_models)
    model = model_pool.models[idx]

    num_iter = loss = 0.0
    delta = 0.999
    syn_loader = None
    batch_X_S, batch_A_S, batch_y_S = data_syn

    for data in real_loader:
        outer_opt.zero_grad()
        num_iter += 1
        real_batch, y_real = data
        y_real = torch.squeeze(y_real, dim=0)
        real_batch, y_real = real_batch.to(device), y_real.to(device)
        x_t, edge_index_t, batch_t = real_batch.x, real_batch.edge_index, real_batch.batch
        batch_X_T, batch_mask = to_dense_batch(x_t, batch=batch_t, max_num_nodes=None)
        batch_A_T = to_dense_adj(edge_index_t, batch=batch_t, edge_attr=None, max_num_nodes=None)

        pred, K_SS = gntk(batch_A_S, batch_X_S, batch_y_S,
                          batch_A_T, batch_X_T, device)
        # print(pred.shape, K_SS.shape)
        outer_loss = args.loss_func(pred, y_real)

        diag = torch.sqrt(K_SS.diag())
        tmp = diag[:, None] * diag[None, :]
        K_SS = K_SS / tmp

        orthogonal_loss = args.orth_reg * torch.norm(K_SS - torch.eye(K_SS.shape[0]).to(device))
        final_loss = outer_loss + orthogonal_loss
        loss += float(final_loss.item())
        final_loss.backward()
        # outer_grad = torch.autograd.grad(final_loss, [batch_X_S, batch_y_S, batch_A_S])
        # outer_grad = torch.autograd.grad(final_loss, [batch_X_S, batch_y_S])
        #
        #
        # update syn data

        # if batch_X_S.grad is None:
        #     batch_X_S.grad = outer_grad[0]
        # else:
        #     batch_X_S.grad.data.copy_(outer_grad[0].data)
        # if batch_y_S.grad is None:
        #     batch_y_S.grad = outer_grad[1]
        # else:
        #     batch_y_S.grad.data.copy_(outer_grad[1].data)
        # # if batch_A_S.grad is None:
        # #     batch_A_S.grad = outer_grad[2]
        # # else:
        # #     batch_A_S.grad.data.copy_(outer_grad[2].data)
        #
        # if args.outer_grad_norm > 0.:
        #     # batch_A_S.data.clamp_(min=0, max=1)
        #     # batch_X_S.data.clamp_(min=0, max=1)
        #     nn.utils.clip_grad_norm_(batch_X_S, args.outer_grad_norm)
        #     nn.utils.clip_grad_norm_(batch_y_S, args.outer_grad_norm)
        #     # nn.utils.clip_grad_norm_(batch_A_S, args.outer_grad_norm)
        outer_opt.step()

        # new syn data for inner loop
        syn_loader = dense2batch(batch_X_S, batch_A_S, batch_y_S)
        model_pool.update(idx, syn_loader)

    avg_loss = loss / num_iter
    return avg_loss, syn_loader, (batch_X_S, batch_A_S)
