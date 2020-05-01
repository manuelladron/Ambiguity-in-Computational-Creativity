import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DECODER(nn.Module):
    """
    The only combination I get to work is using the output of the LSTM, not the hiddens.
    Can be either 1 direction or 2.

    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, max_seq=5):
        super(DECODER, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq = max_seq

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # linear layer
        multiplier = 2 if bidirectional else 1
        #         input_size = hidden_dim * (n_layers * multiplier) # ---> if using hiddens (n_layers important)
        input_size = hidden_dim * (multiplier)  # --------------> nlayers is not correct here! think of diagram!
        self.fc = nn.Linear(input_size, output_dim)


    def forward(self, image_feature, labels, labels_lengths):
        # embedding_size = 64

        # --------------------------------   SHAPES -----------------------------------------------------
        # labels ----------------------> [batch size, max_length]  they come padded to go through the data loader
        # image_feature is ------------> [batch_size, embedding size]
        # image feature unsqueeze(1) --> [batch_size, 1, embedding_size]
        # embedded_label --------------> [batch size, max_length, emb dim] (64 dimensions for each of the 5 labels)
        # embeddings (img + labels) ---> [batch_size, max_length + 1 (dim of image), embedding_shape]
        # packed_embdded --------------> [XXXX, emb dimension]
        # out lstm shape --------------> [seq_len, batch_size, hidden_dim * directions (2 if bidirectional else 1)])
        # Hidden lstm shape -----------> [batch_size, hidden_size*num_layers])
        # Cell lstm shape -------------> [directions * layers, batch_size, hidden_size])

        embedded_label = self.embedding(labels)

        # Concatenate image_feature + label_feature
        embeddings = torch.cat((image_feature.unsqueeze(1), embedded_label), 1)

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeddings, labels_lengths,
                                                            enforce_sorted=False, batch_first=True)

        # IMPORTANT: The lstm input is the packed version of the concatenation of image + label embedding!
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # concat the final forward and backward hidden state --> NOT USING THIS ONE
        hiddens = [hidden[-i, :, :] for i in range(hidden.shape[0])]
        hidden = torch.cat(hiddens, dim=1)

        dense_outputs = self.fc(packed_output[0])

        return dense_outputs

    def sample_topk(self, features, k=3):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        hiddens, states = self.lstm(inputs)  # hiddens: (batch_size, 1, hidden_size)
        outputs = self.fc(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.topk(k)  # predicted: (batch_size)
        sampled_ids.append(predicted.tolist()[0])
        return sampled_ids


    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(self.max_seq):  # MAX SEQ LENGTH TO GENERATE LABELS
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.fc(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            print("predicted: ", predicted)
            sampled_ids.append(predicted)

            # Update inputs
            inputs = self.embedding(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
