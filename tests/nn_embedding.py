from torch import nn
import torch

"""
Construct an embedding module. 
Inputs:
    num_embeddings: int  Size of the vocabulary
    embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
    device: torch.device | None = None  Device to store the parameters on
    dtype: torch.dtype | None = None  Data type of the para
"""
class MyEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        w = torch.empty(vocab_size, d_model, device = device, dtype = dtype)
        nn.init.trunc_normal_(w, 0, 1, -3, 3)
        self.weight = nn.Parameter(w)


    def forward(self, token_ids: torch.Tensor):
        """
        Lookup the embedding vectors for the given token_ids (which are in reality idx of rows) shape: (batch_size, seq_len)
        """
        # Check if inputs are valid
        assert torch.all(token_ids < self.vocab_size), f'Some tokens are out of vocab range {self.vocab_size}'
        assert torch.all(token_ids >= 0), 'Some tokens are negative'
        
        return self.weight[token_ids]
