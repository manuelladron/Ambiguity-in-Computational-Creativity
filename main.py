import numpy as np
import time, os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import utils
import vocabulary
from vocabulary import Vocabulary
import preprocessing
import dataset
import model
import train_test
import generate
import spacy

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
print(CUDA)


nlp = spacy.load('en_core_web_lg')

############################### PATHS #################################################################
im_path_arch = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/architecture/thumbs/big"
im_path_des = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/design/design/thumbs/big"
im_path_tech = "/Users/manuelladron/phd_cd/DL_11785/project/data_collection/technology/thumbs/big"

tagged_path_arch = "../data/architecture_dz-cleaned-tagged.json"
tagged_path_des = "../data/design_dz-cleaned-tagged.json"
tagged_path_tech = "../data/technology_dz-cleaned-tagged.json"

vocab_path_arch = "../data/architecture_dz-cleaned.json"
vocab_path_des = "../data/design_dz-cleaned.json"
vocab_path_tech = "../data/technology_dz-cleaned.json"

im_paths = [im_path_arch, im_path_des, im_path_tech]
paths_for_vocab = [vocab_path_arch, vocab_path_des, vocab_path_tech]
tagged_files_paths = [tagged_path_arch, tagged_path_des, tagged_path_tech]

im_toy = [im_path_tech]
tagg_toy = [tagged_path_tech]

dataset_new_folder = "/Users/manuelladron/phd_cd/DL_11785/project/Github_Ambiguity-in-Computational-Creativity-master/multilabel_classifier/dataset/images"
vocab_dict_path = "/Users/manuelladron/phd_cd/DL_11785/project/Github_Ambiguity-in-Computational-Creativity-master/data/vocab/vocab-dict.json"


#####################################################################################################

def main():
    VOCAB = vocabulary.load_vocab(vocab_dict_path)
    print(len(VOCAB))
    dataset_raw = preprocessing.PreprocessedData(tagg_toy, im_toy, dataset_new_folder, nlp, VOCAB)
    # dataset_raw = preprocessing.PreprocessedData(tagged_files_paths, im_paths, dataset_new_folder, nlp, VOCAB)
    print("\nDataset lengths.... data / labels")
    print(len(dataset_raw.train_data), len(dataset_raw.labels))

    dataset_processed = dataset.ImageTextDataset(dataset_raw)
    print('\nLength of dataset')
    print(len(dataset_processed))

    U = utils.Utils()
    train_n, val_n = U.partition_numbers(.8, len(dataset_processed))
    train_set, val_set = torch.utils.data.random_split(dataset_processed, [train_n, val_n])
    print('\nTrainset {}, valset {}'.format(train_set, val_set))

    ########################### HYPERPARAMETERS ##########################################################
    num_workers = 8 if CUDA else 0
    batch_size = 32
    embedding_dim = 64
    num_hidden_nodes = 512
    size_of_vocab = len(VOCAB)
    num_output_nodes = size_of_vocab
    num_layers = 3
    bidirection = True
    dropout = 0
    nepochs = 20
    lr = 0.001
    weight_decay = 0.00001
    #####################################################################################################

    train_dataloader = DataLoader(train_set, batch_size=batch_size, collate_fn=dataset.collate, shuffle=True,
                                  num_workers=num_workers, drop_last=False)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, collate_fn=dataset.collate, shuffle=True,
                                num_workers=num_workers, drop_last=True)


    # Instantiate
    encoder = model.EncoderCNN(embedding_dim)
    decoder = model.DECODER(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes,
                       num_layers, bidirectional=bidirection, dropout=dropout)

    # Criterion & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    train_losses, test_losses, train_perplexities, test_perplexities = train_test.run_epochs(encoder, decoder,
                                                                                             optimizer, criterion,
                                                                                            train_dataloader, val_dataloader,
                                                                                             nepochs)

    # Generate
    generate.generate_labels(dataset_raw.train_data[0:20])

main()