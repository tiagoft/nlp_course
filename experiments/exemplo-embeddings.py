import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_data(
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    splits = {
        'train': 'plain_text/train-00000-of-00001.parquet',
        'test': 'plain_text/test-00000-of-00001.parquet',
        'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'
    }
    df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" +
                         splits["train"]).sample(1000)
    df_test = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" +
                              splits["test"]).sample(1000)

    X_train = df['text']
    y_train = df['label']
    X_test = df_test['text']
    y_test = df_test['label']

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).reshape((-1, 1)).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).reshape((-1, 1)).float()

    return X_train, y_train, X_test, y_test


def get_fictional_data():
    X_train = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [4, 3, 5, 6],
        [6, 5, 4, 3],
    ])
    y_train = torch.tensor([[0, 0, 1, 1]]).T.float()
    return X_train, y_train


def train_one_batch(model, X, y, optimizer, loss_fn):
    optimizer.zero_grad()
    z_pred = model(X)
    loss = loss_fn(z_pred, y)
    loss.backward()
    optimizer.step()
    return model, loss.item()


def train(model, X, y, optimizer, loss_fn, n_epochs):
    losses = []
    for _ in range(n_epochs):
        model, loss = train_one_batch(model, X, y, optimizer, loss_fn)
        losses.append(loss)
    return model, losses


class MyModel(nn.Module):

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.linear_layer = nn.Linear(
            in_features=embedding_dim,
            out_features=1,
        )

    def forward(self, X):
        Xe = self.embedding_layer(X)
        Xe_pooled = Xe.mean(dim=1)
        y_pred = self.linear_layer(Xe_pooled)
        return y_pred


def run():
    X_train, y_train, X_test, y_test = get_data()
    #X_train, y_train = get_fictional_data()

    model = MyModel(7, 2)
    y_pred = model(X_train)

    X_val = torch.tensor ( [ [0, 1, 2, 3, 4, 5, 6]]).T
    embeddings_iniciais = model.embedding_layer(X_val)
    

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-1,
    )
    loss_fn = F.binary_cross_entropy_with_logits


    model, losses = train(
        model=model,
        X=X_train,
        y=y_train,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=100,
    )
    
    embeddings_finais = model.embedding_layer(X_val)
    
    print(embeddings_finais)

if __name__ == "__main__":
    run()
