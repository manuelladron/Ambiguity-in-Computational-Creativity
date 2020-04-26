import numpy as np
import torch
import time
import datetime
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import * # for pad_sequence and whatnot
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler

from torch.utils import data
from torchvision import transforms

import matplotlib.pyplot as plt
import time

import json

from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import logging
logging.basicConfig(level=logging.INFO)

from train_test import train, test
from preprocessing import PreprocessedData_wordlevel


cuda = torch.cuda.is_available()
print(cuda)
device = torch.device("cuda" if cuda else "cpu")
torch.cuda.empty_cache()
    
def run_epochs(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs):
    start_time = time.time()

    train_losses, train_accs = [], []
    test_losses, test_accs = [] , []
    
    for epoch in range(epochs):
        print('======== Epoch %d ========' % epoch)

        train_loss, train_acc = train(model, train_dataloader, optimizer, scheduler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_loss, test_acc = test(model, validation_dataloader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
    print("Total training took %.2f seconds" % (time.time()-start_time))
    
    return train_losses, train_accs, test_losses, test_accs
import pickle
def main():
    torch.cuda.empty_cache()
    MODEL_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    batches = [64, 32, 16, 8]
    max_lengths = [64, 32, 16]
    epoch_list = list(range(0, 4))
    equal = [True, False]
    for eqs in equal:
        eq_name = 'equal' if eqs else 'unequal'
        for b in batches:
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []
            ms = []
            for m in max_lengths:
                # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
                batch_size = b
                max_length = m

                dataset = PreprocessedData_wordlevel(["./data/architecture_dz-cleaned-tagged.json",
                                "./data/design_dz-cleaned-tagged.json",
                               "./data/technology_dz-cleaned-tagged.json", ], 
                               ["./data/architecture_dz-cleaned.json", 
                                "./data/design_dz-cleaned.json",
                               "./data/technology_dz-cleaned.json"], tokenizer, max_length, eqs)

                train_dataset = dataset.train_dataset
                dev_dataset = dataset.dev_dataset

                # We'll take training samples in random order. 
                train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

                validation_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size)

                # Load BertForSequenceClassification, the pretrained BERT model with a single 
                # linear classification layer on top. 
                model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",          # Use the 12-layer BERT model, with an uncased vocab.
                    num_labels = 2,               # The number of output labels--2 for binary classification. 
                    output_attentions = False,    # Whether the model returns attentions weights.
                    output_hidden_states = False, # Whether the model returns all hidden-states.
                )

                model.to(device)

                optimizer = optim.AdamW(model.parameters(), 
                              lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                              eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                            )

                # The BERT authors recommend between 2 and 4. 
                epochs = 4

                # Total number of training steps is [number of batches] x [number of epochs]. 
                # (Note that this is not the same as the number of training samples).
                total_steps = len(train_dataloader) * epochs

                # Create the learning rate scheduler.
                scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                            num_warmup_steps = 0,
                                                            num_training_steps = total_steps)

                data = run_epochs(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs)

                train_l, train_a, test_l, test_a = data
                train_losses.append(train_l)
                train_accs.append(train_a)
                test_losses.append(test_l)
                test_accs.append(test_a)
                ms.append(m)

                torch.save(model.state_dict(), 'b%d-%s.pt' % (b, eq_name))

                del model
                torch.cuda.empty_cache()

            # Make plot data
            plots = []
            for i in range(len(ms)):
                data = train_losses[i]
                name = 'Training, Max Length: %d' % ms[i]
                plots.append([data, name])
                data = test_losses[i]
                name = 'Testing, Max Length: %d' % ms[i]
                plots.append([data, name])
            name = './graphs/loss-b%d-%s.pkl' % (b, eq_name)
            with open(name, 'wb') as f:
                pickle.dump(plots, f)

            plots = []
            for i in range(len(ms)):
                data = train_accs[i]
                name = 'Training, Max Length: %d' % ms[i]
                plots.append([data, name])
                data = test_accs[i]
                name = 'Testing, Max Length: %d' % ms[i]
                plots.append([data, name])
            name = './graphs/accs-b%d-%s.pkl' % (b, eq_name)
            with open(name, 'wb') as f:
                pickle.dump(plots, f)
    
main()