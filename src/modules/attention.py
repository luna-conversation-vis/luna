import torch
from torch import nn
import torch.nn.functional as nnF
import math

"""Implementations of Attention and Multi-headed attention."""


class Attention(torch.nn.Module):
    def __init__(self, hidden_state_size=768):
        super(Attention, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.key_transform = nn.Linear(hidden_state_size, hidden_state_size)
        self.query_transform = nn.Linear(hidden_state_size, hidden_state_size)
        self.value_transform = nn.Linear(hidden_state_size, hidden_state_size)
        self.attn_scale = math.sqrt(hidden_state_size)

    def forward(self, q, k, v, need_weights=True):
        # B x L x d
        k = self.key_transform(k)
        # B x L x d
        v = self.value_transform(v)
        # B x n x d
        q = self.query_transform(q)

        # B x d x L
        k_t = torch.transpose(k, 1, 2)

        # Q * K^T == B x n x d * B x d x L -> B x n x L
        scores = torch.matmul(q, k_t)
        scores = scores / self.attn_scale
        # B x n x L
        probabilities = nnF.softmax(scores, dim=-1)

        # B x n x d
        outputs = torch.matmul(probabilities, v)
        if need_weights:
            # TODO I'm not actually sure scores is the correct value to return
            # but it is just a placeholder to follow torch's multiheaded API
            return outputs, scores
        return (outputs, None)
