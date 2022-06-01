import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class D_constraint1(torch.nn.Module):

    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).cuda()
        loss_d1_constraint = torch.norm(torch.mm(d.t(),d) * I - I)
        return 	1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim,n_clusters):
        S = torch.ones(d.shape[1],d.shape[1]).cuda()
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(),d) * S)
        return 1e-3 * loss_d2_constraint


