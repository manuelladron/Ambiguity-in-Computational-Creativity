from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pdb

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        X = self.data[i]
        Y = self.labels[i]
        return X, Y

def collate(seq_list):
    # Get inputs shapes and sequences
    x = pad_sequence([torch.tensor(s[0]) for s in seq_list])
    lengths = torch.LongTensor([len(s[0]) for s in seq_list])

    # Assign binary classification
    targets = torch.tensor([s[1] for s in seq_list])
    return x, lengths, targets

def getDataLoader(batch_size, num_workers, dataset, cuda, isTrain):
    if cuda:
        loader_args = dict(shuffle=isTrain, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate)
    else:
        loader_args =  dict(shuffle=isTrain, batch_size=batch_size, collate_fn=collate)
    if isTrain:
        cur_dataset = TextDataset(dataset.train_data, dataset.train_labels)
    else:
        cur_dataset = TextDataset(dataset.dev_data, dataset.dev_labels)
    loader = DataLoader(cur_dataset, **loader_args)
    return loader
