import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, criterion=nn.CrossEntropyLoss(), optimizer=optim.SGD, lr=0.001, epochs=200):
    model.train()
    optimizer = optimizer(model.parameters(), lr=lr)
    training_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_losses.append(epoch_loss / len(train_loader))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {training_losses[-1]}')

    return model, training_losses
