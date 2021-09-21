from typing import Union

import torch

def first_element(x: torch.Tensor, element: Union[int, float], dim: int = 1) -> torch.Tensor:
nonz = x == element
ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
ind[ind == 0] = x.shape[dim]
return ind