from torch import nn
import torch.nn.functional as nnF


class ClassificationHead(nn.Module):
    '''Classification head atop BERT. Expects the CLS token.'''

    def __init__(self, input_size=768, hidden_state_size=512, n_classes=5):
        super(ClassificationHead, self).__init__()
        self.n_classes = n_classes
        self.hidden_state_size = hidden_state_size
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, hidden_state_size)
        self.layer2 = nn.Linear(hidden_state_size, n_classes)

    def forward(self, X):
        out = nnF.relu(self.layer1(X))
        out = self.layer2(out)
        return out
