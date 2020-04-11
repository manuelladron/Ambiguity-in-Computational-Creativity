from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

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
    x = pad_sequence([torch.tensor(s[0]).to(DEVICE) for s in seq_list])
    lengths = torch.LongTensor([len(s[0]) for s in seq_list])

    # Assign binary classification
    targets = torch.tensor([[s[1][0].item(), s[1][0].item()] for s in seq_list])

    return x, lengths, targets
