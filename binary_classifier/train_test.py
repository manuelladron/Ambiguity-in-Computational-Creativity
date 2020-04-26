import time
import torch
import pdb

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

def binary_accuracy(outs, target):
    max_index = outs.max(dim = 1).indices
    target_index = target
    num_correct = (max_index == target_index).sum().item()
    return num_correct / len(target_index)

def train(loader, model, criterion, optimizer):
    # Place model into mode and onto correct device
    model.train()
    model.to(device=DEVICE)

    running_loss = 0.0
    running_acc = 0.0

    for (data, lengths, target) in loader:
        # Zero gradients
        optimizer.zero_grad()
        
        # Use correct types for data
        data = data.to(device=DEVICE).long()
#         lengths = lengths.to(DEVICE)
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
        target = target.to(device=DEVICE).long().view(target.shape[0],)

        # Get model outputs
        outputs = model(data, lengths)

        # Calculate loss & accuracy
        loss = criterion(outputs, target)
        running_loss += loss.item()
        accuracy = binary_accuracy(outputs, target)
        running_acc += accuracy

        # Compute gradients and take step
        loss.backward()
        optimizer.step()

    running_loss /= len(loader)
    running_acc /= len(loader)
    
    return running_loss, running_acc


def test(loader, model, criterion):
    with torch.no_grad():
        # Place into eval mode
        model.eval()
        model.to(device=DEVICE)
        
        running_loss = 0.0
        running_acc = 0.0

        for (data, lengths, target) in loader:
            # Use correct types for data
            data = data.to(device=DEVICE).long()
            lengths = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
            target = target.to(device=DEVICE).long().view(target.shape[0],)

            # Get model outputs
            outputs = model(data, lengths)

            # Calculate loss & accuracy
            loss = criterion(outputs, target)
            running_loss += loss.item()
            accuracy = binary_accuracy(outputs, target)
            running_acc += accuracy

    running_loss /= len(loader)
    running_acc /= len(loader)

    return running_loss, running_acc
