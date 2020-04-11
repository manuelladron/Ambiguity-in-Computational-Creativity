import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import PreprocessedData
from dataset import getDataLoader
from model import classifier
from train_test import train, test

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

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

def run(model, optimizer, criterion, train_loader, dev_loader, nepochs):
    train_losses, train_accs = [], []
    test_losses, test_accs = [] , []
    epochs = []

    for e in range(nepochs):
        print('----- EPOCH %d ------- \n' % e)

        # Train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test
        test_loss, test_acc = test(dev_loader, model, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Epochs
        epochs.append(e)
        if e % 2 == 0 and e != 0:
            print('Training Loss: ', train_loss)
            print('Training Accuracy: ', train_acc)

    # Make final graphs
    make_graph(epochs, train_accs, test_accs, 'Training Acc', 'Testing Acc',
               'Training and Testing Accuracy', 'Accuracy')
    make_graph(epochs, train_losses, test_losses, 'Training loss', 'Testing loss',
               'Training and Testing loss', 'Loss')

    # Save model
    checkDirectory('./saved_models/')
    torch.save(model.state_dict(), "./saved_models/v4_%d.pth" % e)

def main():
    num_workers = 8 if CUDA else 0

    dataset = PreprocessedData(["./data/architecture_dz-cleaned-tagged.json",
                            "./data/design_dz-cleaned-tagged.json",
                           "./data/technology_dz-cleaned-tagged.json"],
                           ["./data/architecture_dz-cleaned.json",
                            "./data/design_dz-cleaned.json",
                           "./data/technology_dz-cleaned.json"])

    # Hyperparameters
    batch_size_gpu = 64
    batch_size_cpu = 64

    vocab_size = dataset.VOCAB_SIZE
    embedding_dim = 100
    num_hidden_nodes = 32
    num_output_nodes = 2
    num_layers = 2
    bidirection = True
    dropout = 0.2
    nepochs = 20
    lr = 1e-4

    # Training
    train_loader = getDataLoader(batch_size_gpu, batch_size_cpu, num_workers,
                                 dataset, CUDA, True)
    dev_loader = getDataLoader(batch_size_gpu, batch_size_cpu, num_workers,
                               dataset, CUDA, False)

    # Instantiate
    model = classifier(vocab_size, embedding_dim, num_hidden_nodes, num_output_nodes,
                       num_layers, bidirectional=bidirection, dropout=dropout)
    model.to(DEVICE)

    # Criterion & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Run training and testing loop
    run(model, optimizer, criterion, train_loader, dev_loader, nepochs)


if __name__ == '__main__':
    main()
