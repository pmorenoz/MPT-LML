
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Pol G. Recasens

import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
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


def integrate_area(losses, dimensions=512):
    prob = len([l for l in losses if math.isnan(l)])/len(losses)
    s = 0
    for l in losses:
        if not math.isnan(l):
            scaled_l = -l*prob*dimensions
            s += scaled_l

    return s


dataset = "mrpc"
file_name_pretrained = "./results/{}_pretrainedBERT".format(dataset)
file_name_random = "./results/{}_randomBERT".format(dataset)
loss_pretrained = load_file(file_name_pretrained)
loss_random = load_file(file_name_random)
probs = np.arange(0,1,0.01)

area_pretrained = integrate_area(loss_pretrained)
area_random = integrate_area(loss_random)

print("Area pretrained BERT for the {} dataset: {}".format(dataset, area_pretrained))
print("Area random BERT for the {} dataset: {}".format(dataset, area_random))
