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


class MulticlassClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

        self.clf = nn.Linear(embedding_dim, n_classes)

    def summarize(self, x):
        #x, _ = self.rnnlayer(x)
        #x = x[:, -1, :]
        return x.mean(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.summarize(x)
        x = self.clf(x)
        return x


tokens = sp.encode_as_ids(list(X))
tokens = pad_to_len(tokens, padding_idx, 500)
tokens = torch.tensor(tokens)
y = torch.tensor(y)

model = MulticlassClassifier(vocab_size=5000,
                             embedding_dim=60,
                             n_classes=n_classes)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)  # lr is the learning rate - this is our alpha

# And now, this is the training loop:
losses = []

model.train()
model = model.cuda()
tokens = tokens.cuda()
y = y.cuda()
print("Entering loop")
for epoch in tqdm(range(10000)):
    optimizer.zero_grad()
    output = model(tokens)
    loss = torch.mean(F.cross_entropy(
        output,
        y,
    ))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

model = model.cpu()

torch.save(model.state_dict(), 'newsgroups_mean.pth')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()
