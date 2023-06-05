import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
import time
from data.cola import losses_cola
from data.ax import losses_ax
from data.mrpc import losses_mrpc
from data.qnli import losses_qnli
from data.sst2 import losses_sst2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
os.environ["WANDB_DISABLED"] = "True"

palette = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
meta_color = 'r'

# This makes the figures look like LATEX / comment this if latex not installed
font = {'family' : 'serif',
        'size'   : 12}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def store_file(losses, dataset, bert):
    file_name = "./results/{}_{}".format(dataset, bert)
    with open(file_name, "wb") as f:
        pickle.dump(losses, f)


def plot_masking_losses(losses, probs):
    plt.plot(probs, losses)
    plt.xlabel('Masking percentage')
    plt.ylabel('Test loss')
    plt.show()


def get_masked_losses(model, lm_dataset, dataset):
    if dataset == "cola":
        return losses_cola(model, lm_dataset)
    elif dataset == "ax":
        return losses_ax(model, lm_dataset)
    elif dataset == "qnli":
        return losses_qnli(model, lm_dataset)
    elif dataset == "mrpc":
        return losses_mrpc(model, lm_dataset)
    elif dataset == "sst2":
        return losses_sst2(model, lm_dataset)


datasets = ["cola"]
# subsets: "ax","mnli","mnli_matched","mnli_mismatched","mrpc","qnli","qqp"


for dataset in datasets:
    start_time = time.time()
    lm_dataset = load_dataset("glue", dataset)

    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    losses, probs = get_masked_losses(model, lm_dataset, dataset)
    store_file(losses, dataset, "pretrainedBERT")
    # plot_masking_losses(losses, probs)

    end_time = time.time() # end tracking time
    elapsed_time = end_time - start_time # calculate elapsed time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    print(f"Time taken to compute {dataset}, pretrained BERT: {time_str}")

    start_time = time.time()
    lm_dataset = load_dataset("glue", dataset)

    configuration = BertConfig()
    model = BertForMaskedLM(configuration)
    losses, probs = get_masked_losses(model, lm_dataset, dataset)
    store_file(losses, dataset, "randomBERT")

    end_time = time.time() # end tracking time
    elapsed_time = end_time - start_time # calculate elapsed time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

    print(f"Time taken to compute {dataset}, random BERT: {time_str}")
