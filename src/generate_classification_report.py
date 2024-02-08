from sklearn.metrics import classification_report
import torch


def generate_classification_report(model, dataloader, input_size):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.view(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    return classification_report(y_true, y_pred)