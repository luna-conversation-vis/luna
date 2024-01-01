import torch
from torch import nn
from src.modules.attention import Attention


class ColumnAttention(nn.Module):
    '''wrapper over attention to form an attention header.'''

    def __init__(self, hidden_state_size=768, dimension_out=1,
                 return_attention=False, multiheaded=True, num_heads=12):
        super(ColumnAttention, self).__init__()
        if multiheaded is True:
            self.attn = nn.MultiheadAttention(
                hidden_state_size, num_heads, batch_first=True)
        else:
            self.attn = Attention(hidden_state_size=hidden_state_size)
        self.hidden_state_size = hidden_state_size
        self.dimension_collapser = nn.Linear(hidden_state_size, dimension_out)
        self.return_attention = return_attention
        self.multiheaded = multiheaded

    def forward(self, NL_embedding, header_embedding):
        # NL B x L x d
        # Header B x n x d

        # B x n x d
        attn_out = self.attn(
            header_embedding,
            NL_embedding,
            NL_embedding,
            need_weights=False)[0]
        # B x n x 1
        probabilities = self.dimension_collapser(attn_out)
        probabilities = torch.squeeze(probabilities, dim=-1)

        if self.return_attention:
            return probabilities, attn_out
        return probabilities
