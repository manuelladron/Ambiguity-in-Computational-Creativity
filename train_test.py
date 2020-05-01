
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

def train(loader, encoder, decoder, criterion, optimizer, epoch, num_epochs):
    # Place model into mode and onto correct device
    decoder.train()
    decoder.to(device=DEVICE)

    running_loss = 0.0
    total_step = len(loader)
    for i, (images, captions, lengths) in enumerate(loader):
        # Zero gradients
        optimizer.zero_grad()

        images = images.to(DEVICE)
        captions = captions.to(DEVICE)  # --> they come padded, so we need to pack it to get an array of labels
        targets = nn.utils.rnn.pack_padded_sequence(captions, lengths,
                                                    batch_first=True, enforce_sorted=False)[0]

        # Econde image with CNN
        features = encoder(images)

        # Get model outputs
        outputs = decoder(features, captions, lengths)  # <---- main function of the whole training

        # Calculate loss & accuracy
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()

        # Calculate loss & accuracy
        running_loss += loss.item()

        # Compute gradients and take step
        loss.backward()
        optimizer.step()

        # Print log info
        if i % 10 == 0 and i != 0:
            perplexity = np.exp(loss.item() / i)
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch + 1, num_epochs, i, total_step, loss.item(), perplexity))

    running_loss /= len(loader)

    return running_loss, perplexity


def test(loader, encoder, decoder, criterion, epoch, num_epochs):
    with torch.no_grad():
        # Place into eval mode
        decoder.eval()
        decoder.to(device=DEVICE)

        running_loss = 0.0
        perplexity = 10000
        total_step = len(loader)
        for i, (images, captions, lengths) in enumerate(loader):

            images = images.to(DEVICE)
            captions = captions.to(DEVICE)  # --> they come padded, so we need to pack it to get an array of labels
            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths,
                                                        batch_first=True, enforce_sorted=False)[0]

            # Econde image with CNN
            features = encoder(images)

            # Get model outputs
            outputs = decoder(features, captions, lengths)  # <---- main function of the whole training

            # Calculate loss & accuracy
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Print log info
            if i % 10 == 0 and i != 0:
                perplexity = np.exp(loss.item() / i)
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch + 1, num_epochs, i, total_step, loss.item(), perplexity))

    running_loss /= len(loader)

    return running_loss, perplexity


def make_graph(epochs, train, test, train_name, val_name, name_long, name_short):
    plt.plot(epochs, train, 'g', label=train_name, c="mediumvioletred")
    plt.plot(epochs, test, 'b', label=val_name, c="darkturquoise")
    plt.title(name_long)
    plt.xlabel('Epochs')
    plt.ylabel(name_short)
    plt.legend()
    plt.show()


def run_epochs(encoder, decoder, optimizer, criterion, train_loader, dev_loader, n_epochs):
    train_losses, train_perplexities = [], []
    test_losses, test_perplexities = [], []

    epochs = []

    for e in range(n_epochs):
        print('----- EPOCH ------- \n', e + 1)

        # Train
        train_loss, train_perplexity = train(train_loader, encoder, decoder, criterion, optimizer, e, n_epochs)
        train_losses.append(train_loss)
        train_perplexities.append(train_perplexity)

        # Test
        test_loss, test_perplexity = test(dev_loader, encoder, decoder, criterion, e, n_epochs)
        test_losses.append(test_loss)
        test_perplexities.append(test_perplexity)

        # Epochs
        epochs.append(e)
        if e % 20 == 0 and e != 0:
            print('Training Loss: ', train_loss)
            print('Training Accuracy: ', train_acc)
            print("Train losses:\n{}\nTest losses:\n{}\n".format(train_losses, test_losses))

        # Make graph after each epoch
        make_graph(epochs, train_perplexities, test_perplexities, 'Training Perp', 'Testing Perp',
                   'Training and Testing Perplexity', 'Perplexity')
        make_graph(epochs, train_losses, test_losses, 'Training loss', 'Testing loss',
                   'Training and Testing loss', 'Loss')

        # save model
        torch.save(encoder.state_dict(), "./saved_models/encoder-{}.pth".format(e))
        torch.save(decoder.state_dict(), "./saved_models/decoder-{}.pth".format(e))

    return train_losses, test_losses, train_perplexities, test_perplexities

# path_to_load = './saved_models/v3_7.pth'
# model.load_state_dict(torch.load(path_to_load, map_location=DEVICE))
