import torch.nn as nn
import torch.nn.functional as F

class SoftmaxClassifier(nn.Module):
    def __init__(self, n_input, n_output):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        # Apply linear transformation
        x = self.linear(x)
        # Apply ReLU activation
        x = F.relu(x)
        return x
