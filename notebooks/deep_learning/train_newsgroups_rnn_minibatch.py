# Train_newsgroups

from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

newsgroups_train = fetch_20newsgroups(subset='train')
X = newsgroups_train.data
y = newsgroups_train.target
print(X[0:5])
print(y[0:5])
n_classes = len(set(y))
print(n_classes)

from torch.utils.data import Dataset, DataLoader

sp = spm.SentencePieceProcessor()
sp.load('fakenews_tokenizer.model')
padding_idx = sp.piece_to_id('<PAD>')


def pad_to_len(sequences, pad_idx, max_len):
    padded = []
    for s in sequences:
        if len(s) >= max_len:
            padded.append(s[:max_len])
        else:
            padded.append(s + [pad_idx] * (max_len - len(s)))
    return padded


class MulticlassClassifierRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)
        self.rnnlayer = nn.RNN(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )
        self.clf = nn.Linear(embedding_dim, n_classes)

    def summarize(self, x):
        x, _ = self.rnnlayer(x)
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self.summarize(x)
        x = self.clf(x)
        return x


tokens = sp.encode_as_ids(list(X))
tokens = pad_to_len(tokens, padding_idx, 500)
tokens = torch.tensor(tokens)
y = torch.tensor(y)


class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data  # If data does not fit memory, change this to e.g. indexes or file pointers
        self.labels = labels

    def __len__(self):  # Returns the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.labels[idx]
        return sample


dataset = MyDataset(tokens, y)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
)

model = MulticlassClassifierRNN(vocab_size=5000,
                                embedding_dim=60,
                                n_classes=n_classes)
model.load_state_dict(torch.load('newsgroups_rnn.pth', weights_only=True))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-5,
)  # lr is the learning rate - this is our alpha

# And now, this is the training loop:
losses = []

model = model.cuda()


def train_one_batch(X, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = torch.mean(criterion(y_pred, y))
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X = X.cuda()
        y = y.cuda()
        loss = train_one_batch(X, y, model, optimizer, criterion)
        total_loss += loss
    return total_loss / len(dataloader)


print("Entering loop")
losses = []
for epoch in tqdm(range(300)):
    epoch_loss = train_one_epoch(model, dataloader, optimizer, F.cross_entropy)
    losses.append(epoch_loss)

model = model.cpu()

torch.save(model.state_dict(), 'newsgroups_rnn.pth')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()
