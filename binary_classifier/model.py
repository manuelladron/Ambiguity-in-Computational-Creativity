import torch.nn as nn
import torch, pdb

class classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super(classifier, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # Linear layer
        multiplier = 2 if bidirectional else 1
        input_size = hidden_dim * (n_layers * multiplier)
        self.fc = nn.Linear(input_size, output_dim)

        # Activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [sent_length, batchsize]
        embedded = self.embedding(text)
        # embedded = [sent_length, batchsize, embeddingsize]

        #packed sequence
#         print(embedded, text_lengths)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        #packed_embdded = [XXXX, emb dimension]
        
#         print(packed_embedded)
#         pdb.set_trace()
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [num_layers * num_directions, batch, hidden_size]

        # Concat the final forward and backward hidden state
        hiddens = [hidden[-i, :, :] for i in range(hidden.shape[0])]
        hidden = torch.cat(hiddens, dim=1)
        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
#         outputs = self.act(dense_outputs)

        return dense_outputs
