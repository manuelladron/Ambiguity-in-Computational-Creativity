import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Graphs and Utils
def make_graph(epochs, train, test, train_name, val_name, name_long, name_short):
    plt.plot(epochs, train, 'g', label=train_name)
    plt.plot(epochs, test, 'b', label=val_name)
    plt.title(name_long)
    plt.xlabel('Epochs')
    plt.ylabel(name_short)
    plt.legend()
    plt.show()



# Dataset work and preprocessing
class PreprocessedData(object):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.train_dataset = None
        self.test_dataset = None
        self.vocab = []
        self.word_2_index = dict()

    def make_train_test(self):
        pass


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




# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
class classifier(nn.Module):

    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super(classifier).__init__()

        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        #activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):

        #text = [batch size, sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]

        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]

        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        #hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)

        return outputs



def binary_accuracy(outs, target):
    max_index = outs.max(dim = 1).indices
    target_index = target.max(dim=1).indices
    num_correct = (max_index == target_index).sum().item()
    return num_correct / len(target_index)


def train_lstm(loader, model, criterion, optimizer):
    # Place model into mode and onto correct device
    model.train()
    model.to(DEVICE)

    running_loss = 0.0
    running_acc = 0.0

    for (data, lengths, target) in loader:
        # Zero gradients
        optimizer.zero_grad()

        # Use correct types for data
        data = data.to(DEVICE).long()
        lengths = lengths.to(DEVICE)
        target = target.to(DEVICE).float()

        # Get model outputs
        outputs = model(data, lengths)

        # Calculate loss
        loss = criterion(outputs, target)
        running_loss += loss.item()

        accuracy = binary_accuracy(outs, target)
        running_acc += accuracy

        # Compute gradients and take step
        loss.backward()
        optimizer.step()

    running_loss /= len(loader)
    running_acc /= len(loader)

    return running_loss, running_acc


def test(loader, model, criterion, epoch):
    with torch.no_grad():
        # Place into eval mode
        model.eval()
        model.to(DEVICE)
        running_loss = 0.0
        running_acc = 0.0

        for (data, lengths, target) in loader:
            # Use correct types for data
            data = data.to(DEVICE).long()
            lengths = lengths.to(DEVICE)
            target = target.to(DEVICE).float()

            # Get model outputs
            outputs = model(data, lengths)

            # Calculate loss
            loss = criterion(outs, target)
            running_loss += loss.item()

            accuracy = binary_accuracy(outs, target)
            running_acc += accuracy

    running_loss /= len(loader)
    running_acc /= len(loader)

    return running_loss, running_acc



def main():
    dataset = PreprocessedData(folder_path)

    # Define hyperparameters
    size_of_vocab = len(dataset.vocab)
    embedding_dim = 100
    num_hidden_nodes = 32
    num_output_nodes = 1
    num_layers = 2
    bidirection = True
    dropout = 0.2
    nepochs = 10
    lr = 1e-4

    train_loader = DataLoader(dataset.train_dataset, shuffle=True,
                              batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False,
                             batch_size=batch_size, collate_fn=collate)

    # Instantiate
    model = classifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes,
                       num_layers, bidirectional=bidirection, dropout=dropout)

    # Criterion & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    epochs = []
    for e in range(nepochs):
        # Train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # Test
        test_loss, test_acc = test(test_loader, model, criterion)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        # Epochs
        epochs.append(e)
        if e % 10 == 0 and e != 0:
            print('Epoch: ', e)
            print('Training Loss: ', train_loss)
            print('Training Accuracy: ', train_acc)

    make_graph(epochs, train_accs, test_accs, 'Training Acc', 'Testing Acc',
               'Training and Testing Accuracy', 'Accuracy')
    make_graph(epochs, train_losses, test_losses, 'Training loss', 'Testing loss',
               'Training and Testing loss', 'Loss')



if __name__ == '__main__':
    main()
