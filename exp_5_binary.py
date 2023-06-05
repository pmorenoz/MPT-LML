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
from torch import nn
from  torch.distributions import MultivariateNormal as Normal
from torch.distributions import Bernoulli
import argparse
import matplotlib.pyplot as plt
from utils import data_loaders

# Command: python exp_5_binary.py -ni 4 -e 800 -n 1000 -data mnist
# Command: python exp_5_binary.py -ni 4 -e 800 -n 1000 -data fmnist

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
parser.add_argument('--num_integrator', '-ni', type=int, default=20)
parser.add_argument('--standarize', '-st', type=bool, default=True)
parser.add_argument('--nof_observations', '-n', type=int, default=2000)
parser.add_argument('--nof_test', '-ntest', type=int, default=100)
parser.add_argument('--dim', '-d', type=int, default=3)
parser.add_argument('--precision', '-b', type=float, default=10)
parser.add_argument('--max_perm', '-p', type=int, default=1)
parser.add_argument('--latent_dim', '-k', type=int, default=2)
parser.add_argument('--fix_mask', '-fm', type=bool, default=False)
parser.add_argument('--num_epochs', '-e', type=int, default=100)
parser.add_argument('--dataset', '-data', type=str, default='mnist')
parser.add_argument('--batch_size', '-bs', type=int, default=100)
args = parser.parse_args()

# Reproducibility
torch.manual_seed(1991)
np.random.seed(1991)

# Dimensions
L = args.num_integrator
N = args.nof_observations
D = args.dim
K = args.latent_dim
beta = args.precision
max_permutations = args.max_perm
range_max_perm = [*range(max_permutations)]
[x+1 for x in range_max_perm]

############################################
# DATA & CONSTANTS
###########################################

data_loader, test_loader, data_dimension = data_loaders.load_dataset(args)

# Dimensions
N = args.nof_observations
D = data_dimension
K = args.latent_dim
beta = args.precision
max_permutations = args.max_perm
range_max_perm = [*range(max_permutations)]
[x+1 for x in range_max_perm]

dataset = data_loader.dataset
x = torch.zeros(N, D)
for i in range(args.nof_observations):
    input_x, label = dataset[i]
    input_x = input_x.view(D)
    input_x[input_x >= 0.5] = 1.0
    input_x[input_x < 0.5] = 0.0
    x[i,:] = input_x

############################################
# LINEAR VAE MODEL
############################################
class BernoulliLinearVAE(nn.Module):
    def __init__(self, K=2, S=10, learning_rate=1e-2, bins=20, device="cpu"):
        super(BernoulliLinearVAE, self).__init__()

        self.K = K
        self.S = S      # Posterior samples expectation
        self.L = bins   # bins for num. integration

        # Linear Encoder ###
        self.encoder_w = torch.nn.Parameter(torch.randn(K, D), requires_grad=True)
        self.encoder_log_v = torch.nn.Parameter(torch.randn(K), requires_grad=True)

        # Linear Decoder ###
        self.decoder_w = torch.nn.Parameter(torch.randn(D, K), requires_grad=True)
        self.decoder_b = torch.nn.Parameter(torch.randn(D), requires_grad=True)
        self.decoder_log_v = torch.nn.Parameter(torch.log(torch.tensor([0.5])), requires_grad=False)

        # Grid for Numerical Integration ###
        zx = torch.linspace(-5, 5, steps=bins)
        zy = torch.linspace(-5, 5, steps=bins)
        z_1, z_2 = torch.meshgrid(zx, zy, indexing='xy')
        self.z_int = torch.zeros(bins**2, 2)
        self.z_int[:,0] = z_1.reshape(-1,1).flatten()
        self.z_int[:,1] = z_2.reshape(-1,1).flatten()

         # Optimization setup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if device == "gpu":
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def approximation_lml(self, x):
        log_w = torch.tensor([2.0*np.log(10.0/self.L)])
        log_p_z = Normal(torch.zeros(self.K), torch.eye(self.K)).log_prob(self.z_int)

        b = torch.tile(self.decoder_b.unsqueeze(1),(1,self.L**2)).T
        probs = torch.sigmoid(self.decoder_w @ self.z_int.T + b.T)
        lik = Bernoulli(probs)

        log_prob_x = lik.log_prob(torch.tile(x.unsqueeze(2),(1,1,self.L**2))).sum(1)
        log_integrand = log_w + torch.tile(log_p_z.unsqueeze(0), (x.shape[0],1)) + log_prob_x
        lml = torch.logsumexp(log_integrand,1).sum()

        return lml

    def forward(self, x):
        N_x = x.shape[0]
        W = self.decoder_w
        mu = self.decoder_b

        V = self.encoder_w
        Diag = torch.diag(torch.exp(self.encoder_log_v))
        scaled_x = (x - mu)
        A = scaled_x @ W @ V
        Q = W @ V @ scaled_x.T

        m_z = V @ scaled_x.T
        v_z = Diag
        q_z = Normal(m_z.T, v_z)

        z_s = q_z.rsample()
        probs = torch.sigmoid(W @ z_s.T + torch.tile(mu.unsqueeze(1),(1,z_s.shape[0]))).T
        lik = Bernoulli(probs)
        expectation = lik.log_prob(x).sum()

        # expectation = - N_x*torch.trace(W @ Diag @ W.T)
        # expectation -= torch.pow(Q, 2).sum()
        # expectation += 2*(A * scaled_x).sum()
        # expectation -= torch.pow(scaled_x, 2).sum()
        # expectation *= 0.5/sigma_var
        # expectation -= 0.5*D*N_x*torch.log(torch.tensor(2*np.pi)*sigma_var)

        kl = 0.5*(- N_x*torch.logdet(Diag) + torch.pow(scaled_x @ V.T, 2).sum() + N_x*torch.trace(Diag) - N_x*K)
        elbo = expectation - kl

        return -elbo

############################################
# MASKED PRE-TRAINING MODEL
############################################
class BernoulliMaskedPPCA(nn.Module):
    def __init__(self, K=2, learning_rate=1e-2, bins=20, device="cpu"):
        super(BernoulliMaskedPPCA, self).__init__()

        self.L = bins
        self.W = torch.nn.Parameter(torch.randn(D, K), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(D), requires_grad=True)
        self.log_var = torch.nn.Parameter(torch.log(torch.tensor([0.5])), requires_grad=False)

        self.register_parameter('PPCA weights', self.W)
        self.register_parameter('PPCA bias', self.b)
        self.register_parameter('PPCA noise', self.log_var)

        zx = torch.linspace(-5, 5, steps=bins)
        zy = torch.linspace(-5, 5, steps=bins)
        z_1, z_2 = torch.meshgrid(zx, zy, indexing='xy')
        self.z_int = torch.zeros(bins**2, 2)
        self.z_int[:,0] = z_1.reshape(-1,1).flatten()
        self.z_int[:,1] = z_2.reshape(-1,1).flatten()

        # Optimization setup
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if device == "gpu":
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def active_set_permutation(self, x, W, b):
        """ Description:    Does a random permutation of data and selects a subset
        Input:          Data observations X (NxD)
        Return:         Active Set X_A and X_rest / X_A U X_rest = X
        """
        permutation = torch.randperm(x.size()[1])

        W_perm = W[permutation]
        b_perm = b[permutation]
        x_perm = x[:, permutation]

        return x_perm, W_perm, b_perm

    def approximation_lml(self, x):
        # lml = torch.zeros_like(x[:,0])
        log_w = torch.tensor([2.0*np.log(10.0/self.L)])
        log_p_z = Normal(torch.zeros(K), torch.eye(K)).log_prob(self.z_int)

        b = torch.tile(self.b.unsqueeze(1),(1,self.L**2)).T
        probs = torch.sigmoid(self.W @ self.z_int.T + b.T)
        lik = Bernoulli(probs)

        log_prob_x = lik.log_prob(torch.tile(x.unsqueeze(2),(1,1,L**2))).sum(1)
        log_integrand = log_w + torch.tile(log_p_z.unsqueeze(0), (x.shape[0],1)) + log_prob_x
        lml = torch.logsumexp(log_integrand,1).sum()

        return lml

    def forward(self, x):
        # sum over sizes of masked tokens --
        masked_loss = 0.0
        for a in range(1):
            m = int(D*0.15) # 15% of masked tokens
            # average over permutations --
            loss_pred = torch.zeros_like(x[:,0])
            for p in range(max_permutations):
                x_p, W_p, b_p = self.active_set_permutation(x, self.W, self.b)

                # Numerical integration for predictive conditionals
                W_m = W_p[:m,:]
                b_m = torch.tile(b_p[:m].unsqueeze(1),(1,self.L**2)).T
                probs_m = torch.sigmoid(W_m @ self.z_int.T + b_m.T)

                log_w = torch.tensor([2.0*np.log(10.0/L)])
                cond_ber= Bernoulli(probs_m)
                log_p_z = Normal(torch.zeros(K), torch.eye(K)).log_prob(self.z_int)

                log_prob_x = cond_ber.log_prob(torch.tile(x_p[:,:m].unsqueeze(2),(1,1,L**2))).sum(1)
                log_integrand = log_w + torch.tile(log_p_z.unsqueeze(0), (x_p.shape[0],1)) + log_prob_x
                loss_pred[:] = torch.logsumexp(log_integrand,1)

            loss_pred = D*loss_pred/(max_permutations * m)
            masked_loss += loss_pred.sum()/N

        return -masked_loss # minimization

list_of_seeds = [4,44,444,4444,44444]
list_of_loss = []
list_of_lml = []

list_of_loss_mpt = []
list_of_lml_mpt = []

for seed in list_of_seeds:

    torch.manual_seed(seed)
    np.random.seed(seed)

    ############################################
    # VAE MODEL TRAINING
    ############################################
    initial_time = time.time()
    loss_time = []
    vae_lml_curve = []
    loss_curve = []

    model = BernoulliLinearVAE(K=K, bins=L)
    with tqdm(range(args.num_epochs)) as pbar:
        for epoch in range(args.num_epochs):

            loss = model(x)
            model.optimizer.zero_grad()
            loss.backward()  # Backward pass <- computes gradients
            model.optimizer.step()

            loss_time.append(time.time() - initial_time)  # in seconds
            loss_curve.append(loss.item()/N)

            lml_vae = model.approximation_lml(x)
            vae_lml_curve.append(lml_vae.item()/N)

            pbar.update()
            pbar.set_description("loss: %.3f" % loss)

    loss_curve = -torch.tensor(loss_curve)
    list_of_loss.append(loss_curve)
    list_of_lml.append(vae_lml_curve)

    ############################################
    # MODEL LOG-MARGINAL LIKELIHOOD
    ############################################
    lml = model.approximation_lml(x)
    print('(VAE) Mean Log-marginal Likelihood (MLML) =', lml.item()/N)

    ############################################
    # MaskedPPCA MODEL TRAINING
    ############################################
    initial_time = time.time()
    loss_time = []
    mpt_ppca_curve = []
    mpt_lml_curve = []

    model_ppca = BernoulliMaskedPPCA(K=K, bins=L)
    with tqdm(range(args.num_epochs)) as pbar:
        for epoch in range(args.num_epochs):

            loss = model_ppca(x)
            model_ppca.optimizer.zero_grad()
            loss.backward()  # Backward pass <- computes gradients
            model_ppca.optimizer.step()

            #loss_curve.append(np.nansum(loss_epoch) / len(loss_epoch))
            loss_time.append(time.time() - initial_time)  # in seconds
            mpt_ppca_curve.append(loss.item())

            lml_mpt = model_ppca.approximation_lml(x)
            mpt_lml_curve.append(lml_mpt.item()/N)

            pbar.update()
            pbar.set_description("loss: %.3f" % loss)

    mpt_ppca_curve = - torch.tensor(mpt_ppca_curve)
    list_of_loss_mpt.append(mpt_ppca_curve)
    list_of_lml_mpt.append(mpt_lml_curve)

    ############################################
    # MODEL LOG-MARGINAL LIKELIHOOD
    ############################################
    lml_mpt = model_ppca.approximation_lml(x)
    print('(PPCA) Mean Log-marginal Likelihood (MLML) =', lml_mpt.item()/N)

############################################
# PLOTTING
############################################
fig, ax = plt.subplots(figsize=(8, 6))

for i, _ in enumerate(list_of_lml):
    if i == 0:
        loss_line, = ax.plot(list_of_loss[i], lw=3.0, alpha=0.2, color=palette_pink[i])
        lml_line, = ax.plot(list_of_lml[i], lw=2.0, alpha=1.0, color=palette_pink[i])

        mpt_lml_line, = ax.plot(list_of_lml_mpt[i], lw=2.0, alpha=1.0, color=palette_blue[i])
        loss_line, = ax.plot(list_of_loss_mpt[i], lw=3.0, alpha=0.2, color=palette_blue[i])
    else:
        ax.plot(list_of_loss[i], lw=3.0, alpha=0.2, color=palette_pink[i])
        ax.plot(list_of_lml[i], lw=2.0, alpha=1.0, color=palette_pink[i])

        ax.plot(list_of_lml_mpt[i], lw=2.0, alpha=1.0, color=palette_blue[i])
        ax.plot(list_of_loss_mpt[i], lw=3.0, alpha=0.2, color=palette_blue[i])

ax.legend([loss_line, lml_line, mpt_lml_line, loss_line],[r'\textsc{elbo}', r'\textsc{lml-elbo}', r'\textsc{mpt}', r'\textsc{lml-mpt}'])
ax.set_ylim(-3000.0, 0.0)
ax.set_xlim(0.0, args.num_epochs-1)
plt.title(r'\textsc{Bernoulli Linear vae} / \textsc{mnist}')
plt.ylabel(r'\textsc{Negative Loss}')
plt.xlabel(r'Epochs')
plt.savefig('./plots/exp5_fig1.pdf')
plt.show()
