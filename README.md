# My course on NLP
A course on modern NLP


## Installing requirements and such

We use `uv` as a package manager. Refer to `https://docs.astral.sh/uv/` for installation instructions. 

Clone the repository and do a local installation:

    git clone https://github.com/tiagoft/nlp_course.git
    cd nlp_course
    uv sync
    source .venv/bin/activate

## Base loop for training a classifier in Pytorch

```python
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Linear(in_features=X_train.shape[1], out_features=1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

print("Entering loop")

losses = []
for epoch in tqdm(range(5000)):
    optimizer.zero_grad()
    z_pred = model(X_tensor_train)
    loss = F.binary_cross_entropy_with_logits(z_pred, y_tensor_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
```
