# My course on NLP
A course on modern NLP


## Installing requirements and such

We use `uv` as a package manager. Refer to `https://docs.astral.sh/uv/` for installation instructions. 

Clone the repository and do a local installation:

    git clone https://github.com/tiagoft/nlp_course.git
    cd nlp_course
    uv sync
    source .venv/bin/activate

## How to make a pull request

Before contributing, read the rules in [AGENTS.md](AGENTS.md) (they apply to people who use agents such as Claude Code or Codex and also to people who contribute without an agent). The step by step with an agent is on the [How to contribute](courseware/docs/how_to_contribute.md) page of the material.

To contribute without an agent, fork the original repository on GitHub and make your changes on a branch of the fork:

1. Go to the original repository on GitHub and click **Fork**.
2. Clone the fork and enter the project directory:

    git clone <YOUR_FORK_URL>
    cd nlp_course

3. Create a branch for your contribution:

    git switch -c my-contribution

4. Make the necessary changes.
5. Add and commit your changes:

    git add .
    git commit -m "Describe your changes"

6. Push the branch to your fork:

    git push origin my-contribution

7. Go to your fork on GitHub and open the **Pull requests** tab.
8. Click **New pull request**, check that the original repository is the target, and create the pull request, filling in the template (type, justification written by you, and course content used).

The pull request will be available in the original repository for review.

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
