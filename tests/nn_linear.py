"""
Construct a linear transformation Module
input.
Make sure to:
• subclass nn.Module
• call the superclass constructor
• construct and store your parameter as 𝑊 (not 𝑊⊤), putting it in an nn.Parameter
• of course, don’t use nn.Linear or nn.functional.linear
For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to 
initialize the weights.
To test your Linear module, implement the test adapter at [adapters.run_linear] . The adapter 
should load the given weights into your Linear module. You can use Module.load_state_dict for 
this purpose. Then, run uv run pytest -k test_linear
"""

"""
Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear 
transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear 
module, except for not having a bias argument or parameter. We recommend the following 
interface:
"""
from einops import einsum
import torch
from torch import nn


class MyModule(nn.Module):
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_features (int): The size of the input dimension
        out_features (int): The size of the output dimension
        weights (Float[Tensor, "out_features in_features"]): The linear weights to use
        in_features (Float[Tensor, "... in_features"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None) :
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty(out_features, in_features, device = device, dtype = dtype)
        std_dev = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(w, 0, std_dev, -3 * std_dev, 3 * std_dev)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor):
        assert x.shape[-1] == self.in_features, f"expected last dim {self.in_features}, got {x.shape[-1]}"
        y = einsum(x, self.weight, '... in, out in -> ... out')
        assert y.shape[-1] == self.out_features
        return y