"""Run after installing the optional integrations/torch-tensorcx package."""
import torch
import torch_tensorcx as bridge

x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
shared = bridge.from_torch(x)
assert bridge.to_torch(shared).data_ptr() == x.data_ptr()
weight = torch.ones((4, 3))
result = bridge.linear(x, weight)
torch.testing.assert_close(result, torch.nn.functional.linear(x, weight))
print(result)
