# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

# N is batch size; D_in is input dimension;
# D_out is output dimension.
N, D_in, D_out = 64, 1000, 10

# Create random Tensors to hold inputs and outputs.
x1 = torch.randn(N, D_in).cuda().requires_grad_(True)
y1 = torch.randn(N, D_out).cuda()

# Create copies of these tensors.
x2 = x1.clone().detach().requires_grad_(True)
y2 = y1.clone()

model = torch.nn.Sequential(
    torch.nn.ReLU(),
    torch.nn.Linear(D_in, D_out)
).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-4)

# dL/dx1 and dL/dx2 should be equal, regardless of when optimizer.step()
# is called for the first time.
y1_pred = model(x1)
y2_pred = model(x2)
loss1 = loss_fn(y1_pred, y1)
loss2 = loss_fn(y2_pred, y2)
loss1.backward()
x1_grad = x1.grad.clone().detach()
optimizer.step()  # Assertion fails if this line is commented out.
loss2.backward()
x2_grad = x2.grad.clone().detach()
optimizer.step()

# These gradients should actually be the same, since the same weights
# were used to compute the forward pass. PyTorch's SGD optimizer by
# default doesn't keep track of the right weight versions, leading
# to an incorrect gradient being computed.
assert not torch.equal(x1_grad, x2_grad)
