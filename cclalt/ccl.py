"""cclalt"""


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
    def __init__(self, h_sizes, num_node, phi_type='maxout', pool_size=2, z_order=2):
        """ Network model
         Args:
             h_sizes: number of channels for each layer [num_layer+1] (first size is input-dim)
             num_node: number of nodes
             phi_type: model type of phi ('maxout')
             pool_size: pool size of max-out
             z_order: model order of z
         """
        super(Net, self).__init__()

        num_dim = h_sizes[-1]
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
        # mlrz
        self.zw = nn.Parameter(torch.ones([num_dim, z_order, 2]))
        self.zw2 = nn.Parameter(torch.zeros([num_dim, 2]))
        self.zb = nn.Parameter(torch.zeros([num_dim, z_order, 2]))
        self.b = nn.Parameter(torch.zeros([1]))

        # initialize
        for k in range(len(self.h)):
            torch.nn.init.xavier_uniform_(self.h[k].weight)

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
                ha_mo, _ = torch.max(self.phiw[None, :, 0, :] * (h[:, 0, :, None] - self.phib[None, :, 0, None]), dim=-1)
                hb_mo, _ = torch.max(self.phiw[None, :, 1, :] * (h[:, 1, :, None] - self.phib[None, :, 1, None]), dim=-1)
                phi_ab = hb_mo * ha_mo
                logits = torch.sum(phi_ab, dim=1)

            # phi_bar
            za, _ = torch.max(self.zw[None, :, 0, :] * (h[:, 0, :, None] - self.zb[None, :, 0, :]), dim=-1)
            zb, _ = torch.max(self.zw[None, :, 1, :] * (h[:, 1, :, None] - self.zb[None, :, 1, :]), dim=-1)
            logitsz = torch.sum(self.zw2[:, 0][None, :] * za**2, dim=-1) + torch.sum(self.zw2[:, 1][None, :] * zb**2, dim=-1)
            logits = - logits + logitsz + self.b
        else:
            logits = None

        return logits, h
