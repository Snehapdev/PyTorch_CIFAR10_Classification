import torch.nn as nn

class SoftmaxClassifier(nn.Module):
    def __init__(self, n_input, n_output):
        super(SoftmaxClassifier, self).__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear(x)
