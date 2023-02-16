"""ccl"""


import torch
import torch.nn as nn
from subfunc.showdata import *


# =============================================================
# =============================================================
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        m, _ = torch.max(x.reshape([*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size]), dim=-1)
        return m


# =============================================================
# =============================================================
class Net(nn.Module):
    def __init__(self, h_sizes, num_node, conn=None, phi_type='maxout', pool_size=2, z_order=2):
        """ Network model
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_node: number of nodes
             conn: list of edges
             phi_type: model type of phi ('maxout')
             pool_size: pool size of max-out
             z_order: model order of z
         """
        super(Net, self).__init__()

        if conn is None:
            conn = np.triu_indices(num_node, k=1)
            conn_inv = conn[::-1]
            conn = (np.concatenate([conn[0], conn_inv[0]]), np.concatenate([conn[1], conn_inv[1]]))
        num_conn = len(conn[0])
        num_dim = h_sizes[-1]
        self.conn = conn
        self.num_conn = num_conn
        self.num_node = num_node
        self.num_dim = num_dim
        self.w = None

        # h
        if len(h_sizes) > 1:
            h = [nn.Linear(h_sizes[k-1], h_sizes[k]*pool_size) for k in range(1, len(h_sizes)-1)]
            h.append(nn.Linear(h_sizes[-2], h_sizes[-1]))
            self.h = nn.ModuleList(h)
        else:
            self.h = []
        self.bn = nn.BatchNorm1d(num_features=num_dim)
        self.maxout = Maxout(pool_size)
        self.phi_type = phi_type
        # phi
        if self.phi_type == 'maxout':
            self.phiw = nn.Parameter(torch.ones([num_dim, 2, 2]))
            self.phib = nn.Parameter(torch.zeros([num_dim, 2]))
        else:
            raise ValueError
        # mlr
        self.wu = nn.Linear(num_dim, int(num_conn / 2))
        self.wl = nn.Linear(num_dim, int(num_conn / 2))
        # mlrz
        self.zwu = nn.Parameter(torch.ones([num_dim, z_order, int(num_conn / 2)]))
        self.zbu = nn.Parameter(torch.zeros([num_dim, z_order, int(num_conn / 2)]))
        self.zwl = nn.Parameter(torch.ones([num_dim, z_order, int(num_conn / 2)]))
        self.zbl = nn.Parameter(torch.zeros([num_dim, z_order, int(num_conn / 2)]))

        # initialize
        for k in range(len(self.h)):
            torch.nn.init.xavier_uniform_(self.h[k].weight)
        torch.nn.init.xavier_uniform_(self.wu.weight)
        torch.nn.init.xavier_uniform_(self.wl.weight)

    def forward(self, x, calc_logit=True):
        """ forward
         Args:
             x: input [batch, node, dim]
             calc_logit: obtain logits additionally to h, or not
         """
        batch_size, _, num_dim = x.size()

        # h
        h = x
        for k in range(len(self.h)):
            h = self.h[k](h)
            if k != len(self.h)-1:
                h = self.maxout(h)
        h = self.bn(h.reshape([-1, h.shape[-1]])).reshape(h.shape)

        if calc_logit:
            # phi
            if self.phi_type == 'maxout':
                ha_mo, _ = torch.max(self.phiw[None, :, :, :] * (h[:, 0, :, None, None] - self.phib[None, :, :, None]), dim=3)
                hb_mo, _ = torch.max(self.phiw[None, :, :, :] * (h[:, 1, :, None, None] - self.phib[None, :, :, None]), dim=3)
                phi_ab = hb_mo[:, :, 1] * ha_mo[:, :, 0]
                phi_ba = ha_mo[:, :, 1] * hb_mo[:, :, 0]
                #
                wu_ab = torch.nn.functional.linear(phi_ab, self.wu.weight, torch.zeros(1, device=x.device))
                wl_ab = torch.nn.functional.linear(phi_ab, self.wl.weight, torch.zeros(1, device=x.device))
                w_ab = torch.cat([wu_ab, wl_ab], dim=1)
                wl_ba = torch.nn.functional.linear(phi_ba, self.wl.weight, torch.zeros(1, device=x.device))
                wu_ba = torch.nn.functional.linear(phi_ba, self.wu.weight, torch.zeros(1, device=x.device))
                wt_ba = torch.cat([wl_ba, wu_ba], dim=1)
                logits = w_ab + wt_ba + torch.cat([self.wu.bias, self.wu.bias])

            # phi_bar
            ha = h[:, 0, :]
            zu_a, _ = torch.max(self.zwu[None, :, :, :] * ha[:, :, None, None] + self.zbu[None, :, :, :], dim=2)  # [batch, dim, conn/2]
            zl_a, _ = torch.max(self.zwl[None, :, :, :] * ha[:, :, None, None] + self.zbl[None, :, :, :], dim=2)  # [batch, dim, conn/2]
            zu_a = torch.sum(zu_a**2, dim=1)  # [batch, conn/2]
            zl_a = torch.sum(zl_a**2, dim=1)  # [batch, conn/2]
            z_a = torch.cat([zu_a, zl_a], dim=1)  # [batch, conn]
            hb = h[:, 1, :]
            zl_b, _ = torch.max(self.zwl[None, :, :, :] * hb[:, :, None, None] + self.zbl[None, :, :, :], dim=2)  # [batch, dim, conn/2]
            zu_b, _ = torch.max(self.zwu[None, :, :, :] * hb[:, :, None, None] + self.zbu[None, :, :, :], dim=2)  # [batch, dim, conn/2]
            zl_b = torch.sum(zl_b**2, dim=1)  # [batch, conn/2]
            zu_b = torch.sum(zu_b**2, dim=1)  # [batch, conn/2]
            z_b = torch.cat([zl_b, zu_b], dim=1)  # [batch, conn]
            logitsz = z_a + z_b
            logits = logits - logitsz
        else:
            logits = None

        return logits, h

    def adjacency_matrix(self):
        """ Obtain adjacency matrix from wu, wl.
        Args:
        Returns:
            adjacency matrix: [node x node x dim]
        """
        num_conn_half = int(self.num_conn / 2)
        w = np.zeros([self.num_node, self.num_node, self.num_dim])
        w[self.conn[0][:num_conn_half], self.conn[1][:num_conn_half], :] = self.wu.weight.to('cpu').detach().numpy()
        w[self.conn[0][num_conn_half:], self.conn[1][num_conn_half:], :] = self.wl.weight.to('cpu').detach().numpy()
        self.w = w

        return self.w
