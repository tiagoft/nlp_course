# Train_newsgroups

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re

phrases = [
    "I like dogs yay",
    "I like cats yay",
    "I like chicken yay",
    "I like spongebob yay",
    "I hate broccoli aff",
    "I hate sand aff",
    "I hate sadness aff",
    "I hate airports aff",
]

idx_to_word = list(set(' '.join(phrases).split()))
word_to_idx = {}
for i in range(len(idx_to_word)):
    word_to_idx[idx_to_word[i]] = i


def tokenize(list_of_phrases):
    tokenized_phrases = []
    for i in range(len(list_of_phrases)):
        tokens = [
            word_to_idx[w] for w in re.findall(r'\w+', list_of_phrases[i])
        ]
        tokenized_phrases.append(tokens)
    return torch.tensor(tokenized_phrases)


token_tensor = tokenize(phrases)


class MyLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.rnnlayer = nn.RNN(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

        self.clf = nn.Linear(embedding_dim, vocab_size)

    def sequence_model(self, x):
        y, _ = self.rnnlayer(x)
        #y = y + x
        return y

    def forward(self, x):
        x = self.embedding(x)
        x = self.sequence_model(x)
        x = self.clf(x)
        return x


class MyDataset(Dataset):

    def __init__(self, data, window_length):
        self.data = data  # If data does not fit memory, change this to e.g. indexes or file pointers
        self.window_length = window_length

    def __len__(self):  # Returns the number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        t_start = torch.randint(low=0,
                                high=self.data.shape[1] - self.window_length,
                                size=(1, )).item()

        x = self.data[idx, t_start:t_start + self.window_length]
        y = self.data[idx, t_start + 1:t_start + self.window_length + 1]

        return x, y


dataset = MyDataset(token_tensor, 3)
x, y = dataset[0]
print("---")
print(x)
print(y)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

model = MyLanguageModel(
    vocab_size=len(idx_to_word),
    embedding_dim=2,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-2,
)  # lr is the learning rate - this is our alpha

# And now, this is the training loop:
losses = []

model = model.cuda()


def train_one_batch(X, y, model, optimizer, criterion):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = torch.mean(criterion(torch.transpose(y_pred, 1, 2), y))
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
for epoch in tqdm(range(1000)):
    epoch_loss = train_one_epoch(model, dataloader, optimizer, F.cross_entropy)
    losses.append(epoch_loss)

model = model.cpu()

torch.save(model.state_dict(), 'language_model.pth')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

model.eval()
word_tokens = tokenize(idx_to_word)
word_embeddings = model.embedding(word_tokens).cpu().detach().numpy()[:, 0, :]
print(word_embeddings.shape)
plt.figure()
plt.scatter(word_embeddings[:, 0], word_embeddings[:, 1])
for i in range(len(idx_to_word)):
    plt.text(x=word_embeddings[i, 0],
             y=word_embeddings[i, 1],
             s=idx_to_word[i])
plt.savefig('word_embeddings.png')
plt.show()
