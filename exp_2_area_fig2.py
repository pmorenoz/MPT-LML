# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import time
from tqdm import tqdm
from tqdm import trange
from torch.autograd import Variable
from datetime import datetime
import sys
import argparse
from math import nan, isnan
from scipy.special import binom

import numpy as np
import random
import torch
from  torch.distributions import MultivariateNormal as Normal
import argparse
import matplotlib.pyplot as plt

font = {'family' : 'serif',
        'size'   : 16}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']

palette_red = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
palette_blue = ["#012a4a","#013a63","#01497c","#014f86","#2a6f97","#2c7da0","#468faf","#61a5c2","#89c2d9","#a9d6e5"]
palette_green = ['#99e2b4','#88d4ab','#78c6a3','#67b99a','#56ab91','#469d89','#358f80','#248277','#14746f','#036666']
palette_pink = ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf","#6d23b6","#6411ad","#571089","#47126b"]
palette_super_red = ["#641220","#6e1423","#85182a","#a11d33","#a71e34","#b21e35","#bd1f36","#c71f37","#da1e37","#e01e37"]

palette = color_palette_1

parser = argparse.ArgumentParser()
parser.add_argument('--n', '-n', type=int, default=100)
parser.add_argument('--dim', '-d', type=int, default=3)
parser.add_argument('--precision', '-b', type=float, default=5)
parser.add_argument('--max_perm', '-p', type=int, default=100)
parser.add_argument('--latent_dim', '-k', type=int, default=2)
parser.add_argument('--fix_mask', '-fm', type=bool, default=False)
args = parser.parse_args()

# Command:
# Fig. 2: python exp_2_area_fig2.py -n 1000 -d 200 -k 50 -p 5

print('Total num. of permutations=', binom(512,76))

torch.manual_seed(1991)
np.random.seed(1991)

# Dimensions
N = args.n
D = args.dim
K = args.latent_dim
beta = args.precision
max_permutations = args.max_perm
range_max_perm = [*range(max_permutations)]
[x+1 for x in range_max_perm]

############################################
# PPCA: GENERATIVE PROCESS
###########################################

# true weights
W = torch.randn(D,K)
Z = torch.randn(N,K)
epsilon = (1/beta) * torch.randn(N,1)

# how data is generated
x = Z @ W.T + epsilon

############################################
# EXACT LOG-MARGINAL LIKELIHOOD
############################################
S_lml = W @ W.T + (1/beta)*torch.eye(D)
lml_dist = Normal(torch.zeros(D), S_lml)
lml = lml_dist.log_prob(x).sum()
print('Mean Log-marginal Likelihood (MLML) =', lml.item()/N)

############################################
# MASKED PRE-TRAINING LOSS
############################################
def active_set_permutation(x, W):
    """ Description:    Does a random permutation of data and selects a subset
    Input:          Data observations X (NxD)
    Return:         Active Set X_A and X_rest / X_A U X_rest = X
    """
    permutation = torch.randperm(x.size()[1])

    W_perm = W[permutation]
    x_perm = x[:, permutation]

    return x_perm, W_perm

list_of_seeds = [4,44,444,4444,44444]
list_of_areas = []


for seed in list_of_seeds:

    torch.manual_seed(seed)
    np.random.seed(seed)

    masked_loss_m = torch.zeros(D)
    # sum over sizes of masked tokens --
    masked_loss = 0.0
    for a in range(D):
        # m = a+1
        m = 171
        # average over permutations --
        loss_pred = torch.zeros_like(x[:,0])
        for p in range(max_permutations):
            x_p, W_p = active_set_permutation(x, W)
            S_p = W_p @ W_p.T + (1/beta)*torch.eye(D)

            # Computation // forgetting other extra elements
            x_r = x_p[:,m:]
            iS_rr = torch.inverse(S_p[m:,m:])

            m_pred = S_p[:m,m:] @ iS_rr @ x_r.T
            v_pred = torch.diagonal(S_p[:m,:m] - S_p[:m,m:] @ iS_rr @ S_p[:m,m:].T)

            m_pred = m_pred.T
            v_pred = torch.tile(v_pred.unsqueeze(1),(1,N)).T

            log_p_masked = -0.5*torch.log(v_pred) - 0.5*np.log(2*np.pi) - (0.5*(x_p[:,:m] - m_pred)**2 / v_pred)
            loss_pred += log_p_masked.sum(1)

        loss_pred = loss_pred/(max_permutations * m)
        masked_loss += loss_pred.sum()
        masked_loss_m[a] = loss_pred.sum()/N

    list_of_areas.append(masked_loss_m)
    print(max_permutations, ' -- Masked pre-training loss (MPTL) =', masked_loss.item()/N)
    loss_diff = lml.item()/N - masked_loss.item()

fig, ax = plt.subplots(figsize=(8, 6))
for i, masked_loss_m in enumerate(list_of_areas):
    ax.bar(torch.arange(D), masked_loss_m, color=palette_pink[i], alpha=0.1)
    ax.plot(torch.arange(D), masked_loss_m, lw=2.0, color=palette_pink[i], alpha=0.8)

ax.set_xlim(0.0, D-1)
ax.set_xticks([2,30,60,100,140,170,199])
ax.set_xticklabels([r'1{\%}', r'15{\%}', r'30{\%}', r'50{\%}', r'70{\%}', r'85{\%}', r'100{\%}'])
plt.title(r'{\#} \textsc{tokens} $= 200$ / $K = 50$ / Max. permutations $= '+str(args.max_perm)+'$')
plt.ylabel(r'\textsc{mpt} loss')
plt.xlabel(r'{\%} of masked tokens')
plt.savefig('./plots/area_fig2.pdf')
plt.show()
