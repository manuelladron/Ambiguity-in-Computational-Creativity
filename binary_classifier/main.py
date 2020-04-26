import numpy as np
import time, os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import PreprocessedData
from dataset import getDataLoader
from model import classifier
from train_test import train, test
import pickle

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print(CUDA)

def make_graph(epochs, train, test, train_name, val_name, name_long, name_short):
    plt.plot(epochs, train, 'g', label=train_name, c="mediumvioletred")
    plt.plot(epochs, test, 'b', label=val_name, c="darkturquoise")
    plt.title(name_long)
    plt.xlabel('Epochs')
    plt.ylabel(name_short)
    plt.legend()
    plt.show()

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def makePlotData(ms, train_losses, test_losses, train_accs, test_accs, eq_name, name_starter):
    plots = []
    for i in range(len(ms)):
        data = train_losses[i]
        name = 'Training, Max Length: %d' % ms[i]
        plots.append([data, name])
        data = test_losses[i]
        name = 'Testing, Max Length: %d' % ms[i]
        plots.append([data, name])
    name = './graphs/loss-%s.pkl' % (name_starter)
    with open(name, 'wb') as f:
        pickle.dump(plots, f)
        
#     with open(name, 'rb') as f:
#         plots = pickle.load(f)
#     print(plots)

    plots = []
    for i in range(len(ms)):
        data = train_accs[i]
        name = 'Training, Max Length: %d' % ms[i]
        plots.append([data, name])
        data = test_accs[i]
        name = 'Testing, Max Length: %d' % ms[i]
        plots.append([data, name])
    name = './graphs/accs-%s.pkl' % (name_starter)
    with open(name, 'wb') as f:
        pickle.dump(plots, f)
    
#     with open(name, 'rb') as f:
#         plots = pickle.load(f)
#     print(plots)
    
#     pdb.set_trace()

def run(model, optimizer, criterion, train_loader, dev_loader, nepochs):
    train_losses, train_accs = [], []
    test_losses, test_accs = [] , []
    epochs = []

    for e in range(nepochs):
        print('----- EPOCH %d ------- \n' % e)
        start_time = time.time()

        # Train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test
        test_loss, test_acc = test(dev_loader, model, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    print(train_loss, train_acc, test_loss, test_acc)
    return train_losses, train_accs, test_losses, test_accs

    # Make final graphs
#     make_graph(epochs, train_accs, test_accs, 'Training Acc', 'Testing Acc',
#                'Training and Testing Accuracy', 'Accuracy')
#     make_graph(epochs, train_losses, test_losses, 'Training loss', 'Testing loss',
#                'Training and Testing loss', 'Loss')

def main():
    num_workers = 8 if CUDA else 0
    nepochs = 20
    lr = 1e-4
    
    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    num_layers = [2, 4, 6]
    embedding_dims = [128, 256, 512]
    drop_outs = [.2, .4]

    # Run training and testing loop
    batches = [64, 16]
    max_lengths = [64, 32]
    equal = [True, False]
    for eqs in equal:
        eq_name = 'equal' if eqs else 'unequal'
        for d in drop_outs:
            for embedding_dim in embedding_dims:
                for nLayer in num_layers:
                    for b in batches:
                        train_losses = []
                        train_accs = []
                        test_losses = []
                        test_accs = []
                        ms = []
                        for m in max_lengths:
                            max_sentence_length = m
                            print('Batchsize: %d, %s, Length: %d, nLayers:%d, EmbedDim:%d Dropout:%f' % (b, eq_name, m, nLayer, embedding_dim, d))
                            dataset = PreprocessedData(["./data/architecture_dz-cleaned-tagged.json",
                                                        "./data/design_dz-cleaned-tagged.json",
                                                       "./data/technology_dz-cleaned-tagged.json"],
                                                       ["./data/architecture_dz-cleaned.json",
                                                        "./data/design_dz-cleaned.json",
                                                       "./data/technology_dz-cleaned.json"],
                                                          max_sentence_length, eqs)

                            # Hyperparameters
                            batch_size = b

                            vocab_size = dataset.vocab_size

                            num_hidden_nodes = embedding_dim
                            num_output_nodes = 2

                            bidirection = True
                            dropout =d

                            train_loader = getDataLoader(batch_size, num_workers, dataset, CUDA, True)
                            dev_loader = getDataLoader(batch_size, num_workers, dataset, CUDA, False)

                            # Instantiate
                            model = classifier(vocab_size, embedding_dim, num_hidden_nodes, num_output_nodes,
                                               nLayer, bidirectional=bidirection, dropout=dropout)
                            model.to(DEVICE)
                            optimizer = optim.Adam(model.parameters(), lr=lr)

                            data = run(model, optimizer, criterion, train_loader, dev_loader, nepochs)

                            train_l, train_a, test_l, test_a = data
                            train_losses.append(train_l)
                            train_accs.append(train_a)
                            test_losses.append(test_l)
                            test_accs.append(test_a)
                            ms.append(m)

                            del model
                            torch.cuda.empty_cache()

                        # Make plot data
                        name_starter = 'Batch:%d-%s-Length:%d-LR:%f-nLayers:%d-EmbedDim:%d-Drop%.2f' % (b, eq_name, m, lr, nLayer, embedding_dim, d)
                        makePlotData(ms, train_losses, test_losses, train_accs, test_accs, eq_name, name_starter)



main()