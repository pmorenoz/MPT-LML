# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pablo Moreno-Munoz
# CogSys Section  ---  (pabmo@dtu.dk)
# Technical University of Denmark (DTU)

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import argparse

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
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
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def load_file(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-data', type=str, default='ax')
args = parser.parse_args()


file_name_pretrained = "./data/results/{}_pretrainedBERT".format(args.dataset)
file_name_random = "./data/results/{}_randomBERT".format(args.dataset)
loss_pretrained = load_file(file_name_pretrained)
loss_random = load_file(file_name_random)
probs = np.arange(0,1,0.01)



fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(probs, loss_pretrained, width=0.0075, color=palette_blue[6], alpha=0.1)
ax.plot(probs, loss_pretrained, lw=2.0, color=palette_blue[0], alpha=1.0)

ax.bar(probs, loss_random, width=0.0075, color=palette_blue[0], alpha=0.1)
ax.plot(probs, loss_random, lw=2.0, color=palette_blue[0], alpha=1.0)

ax.set_xticks([0.01,0.15,0.30,0.50,0.7,0.85,0.98])
ax.set_xticklabels([r'1{\%}', r'15{\%}', r'30{\%}', r'50{\%}', r'70{\%}', r'85{\%}', r'100{\%}'])
ax.set_xlim(0.01, 0.99)
plt.title(r'\textsc{bert} --- \textsc{'+ args.dataset +'}')
plt.legend([r'\textsc{pre-trained}', r'\textsc{random}'])

plt.ylabel(r'\textsc{mpt} loss')
plt.xlabel(r'{\%} of masked tokens')
plt.savefig('./plots/area_fig4_llm_' + args.dataset + '.pdf')
plt.show()
