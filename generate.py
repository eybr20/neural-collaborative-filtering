import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import torch
import torch.nn as nn

hyperparams = dict(factors=8, layers=(16, 8), share=True, biases=True, dropout=0.5, 
                   weight_decay=0.0, epochs=5, lr=3e-3, batch_size=256, seed=20)

class GMF(nn.Module):
    def __init__(self, n_users, n_items, mu, factors):
        super().__init__()
        self.P = nn.Embedding(n_users, factors)
        self.Q = nn.Embedding(n_items, factors)
        self.h = nn.Linear(factors, 1)
        nn.init.normal_(self.P.weight, 0.0, 0.01)
        nn.init.normal_(self.Q.weight, 0.0, 0.01)
        nn.init.constant_(self.h.bias, mu)

    def features(self, u, i):
        return self.P(u) * self.Q(i)

class MLP(nn.Module):
    def __init__(self, n_users, n_items, mu, layers, dropout):
        super().__init__()
        emb = layers[0] // 2
        self.P = nn.Embedding(n_users, emb)
        self.Q = nn.Embedding(n_items, emb)
        tower = []
        for size_in, size_out in zip(layers[:-1], layers[1:]):
            tower += [nn.Linear(size_in, size_out), nn.ReLU()]
            if dropout > 0:
                tower.append(nn.Dropout(dropout))
        self.tower = nn.Sequential(*tower)
        self.h = nn.Linear(layers[-1], 1)
        nn.init.normal_(self.P.weight, 0.0, 0.01)
        nn.init.normal_(self.Q.weight, 0.0, 0.01)
        nn.init.constant_(self.h.bias, mu)

    def features(self, u, i):
        return self.tower(torch.cat([self.P(u), self.Q(i)], dim=1))

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, mu, cfg):
        super().__init__()
        self.gmf = GMF(n_users, n_items, mu, cfg["factors"])
        self.mlp = MLP(n_users, n_items, mu, cfg["layers"], cfg["dropout"])
        if cfg["share"]:
            self.mlp.P, self.mlp.Q = self.gmf.P, self.gmf.Q
        self.h = nn.Linear(cfg["factors"] + cfg["layers"][-1], 1)
        nn.init.constant_(self.h.bias, mu)
        self.biases = cfg["biases"]
        if self.biases:
            self.b_u = nn.Embedding(n_users, 1)
            self.b_i = nn.Embedding(n_items, 1)
            nn.init.zeros_(self.b_u.weight)
            nn.init.zeros_(self.b_i.weight)

    def forward(self, u, i):
        phi = torch.cat([self.gmf.features(u, i), self.mlp.features(u, i)], dim=1)
        out = self.h(phi).squeeze(1)
        if self.biases:
            out = out + self.b_u(u).squeeze(1) + self.b_i(i).squeeze(1)
        return out

def train_model(table, cfg, device):
    n_users, n_items = table.shape
    users, items = np.nonzero(~np.isnan(table))
    u = torch.tensor(users, device=device)
    i = torch.tensor(items, device=device)
    r = torch.tensor(table[users, items], dtype=torch.float32, device=device)
    mu = float(r.mean())

    model = NeuMF(n_users, n_items, mu, cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])

    bs = cfg["batch_size"]
    for _ in trange(cfg["epochs"], desc="Training"):
        model.train()
        perm = torch.randperm(len(r), device=device)
        for s in range(0, len(r), bs):
            b = perm[s:s + bs]
            loss = ((model(u[b], i[b]) - r[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def predict_all(model, n_users, n_items, device):
    model.eval()
    items = torch.arange(n_items, device=device)
    pred = np.empty((n_users, n_items), dtype=np.float32)
    for user in range(n_users):
        users = torch.full((n_items,), user, device=device)
        pred[user] = model(users, items).clamp(0.5, 5).cpu().numpy()
    return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_train.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...')
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')

    # Any method you want
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(hyperparams["seed"])
    model = train_model(table, hyperparams, device)
    pred = predict_all(model, *table.shape, device)
    table = np.where(np.isnan(table), pred, table)   # keep known ratings, fill the missing ones

    # Save the completed table
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE