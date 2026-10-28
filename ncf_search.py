"""MF, GMF, MLP and NeuMF for explicit ratings (same models as ncf.ipynb), with a grid search.

One run (validation RMSE):
    python ncf_search.py --model NeuMF --factors 8 --layers 16 8 --dropout 0.2 --weight-decay 1
Same, then retrain on all of ratings_train and score once on ratings_test:
    python ncf_search.py --model NeuMF --factors 8 --layers 16 8 --dropout 0.2 --weight-decay 1 --test
Grid search over NeuMF (every valid combination of GRID below, validation only):
    python ncf_search.py --search --out results/search1
Random subset of the grid (faster):
    python ncf_search.py --search --sample 500 --out results/search1
Split the search into 20 parts, one per Slurm job (job array), part 0 to 19:
    python ncf_search.py --search --part 0 --n-parts 20 --out results/search1
Rerun the 10 best configurations with 3 seeds (is the ranking stable?):
    python ncf_search.py --top 10 --seeds 20 21 22 --out results/search1
"""
import argparse
import copy
import csv
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# PLAY: values tried by --search (NeuMF only). Invalid combinations are skipped.
# Removed after the notebook experiments (add them back if you have time):
#   biases=False (biases never hurt), weight_decay=0.1 (same as 0), factors=64 and layers
#   (128, 64, 32, 16) (too big for movies with ~2 ratings), tiny (8, 4), lr=3e-4 and
#   batch_size=128 (slowest to train).
GRID = dict(model=["NeuMF"],
            factors=[4, 8, 16, 32],                        # GMF embedding size
            layers=[(16, 8), (32, 16), (32, 16, 8), (64, 32, 16, 8)],   # MLP
            dropout=[0.0, 0.2, 0.5],                       # after each ReLU of the MLP
            weight_decay=[0.0, 0.3, 1.0, 3.0],             # AdamW, embeddings and weight matrices
            biases=[True],                                 # user and item biases
            share=[False, True],                           # only valid if layers[0] == 2 * factors
            lr=[1e-3, 3e-3],
            batch_size=[256, 1024])


#  data

def to_triples(R_full, mask):
    """Observed cells of an items x users matrix as (user, item, rating) tensors."""
    items, users = np.nonzero(mask)
    return (torch.tensor(users, device=device),
            torch.tensor(items, device=device),
            torch.tensor(R_full[items, users], dtype=torch.float32, device=device))


def load_data(folder, val_frac, split_seed):
    """Same 80/20 split of ratings_train as the notebook, plus the full file and the test file."""
    R = np.load(os.path.join(folder, "ratings_train.npy")).T          # items x users
    observed = ~np.isnan(R)
    R0 = np.nan_to_num(R)

    rng = np.random.default_rng(split_seed)
    rows, cols = np.where(observed)
    perm = rng.permutation(len(rows))
    n_train = int((1 - val_frac) * len(rows))
    tr, vl = perm[:n_train], perm[n_train:]
    train_mask = np.zeros_like(observed)
    train_mask[rows[tr], cols[tr]] = True
    val_mask = np.zeros_like(observed)
    val_mask[rows[vl], cols[vl]] = True

    R_test = np.load(os.path.join(folder, "ratings_test.npy")).T
    return dict(train=to_triples(R0, train_mask),
                val=to_triples(R0, val_mask),
                full=to_triples(R0, observed),
                test=to_triples(np.nan_to_num(R_test), ~np.isnan(R_test)),
                mu_train=float(R0[train_mask].mean()),
                mu_full=float(R0[observed].mean()),
                n_items=R.shape[0], n_users=R.shape[1])


# models

class MF(nn.Module):
    """Standard MF: r̂ = p_u · q_i + b."""

    def __init__(self, n_users, n_items, mu, factors):
        super().__init__()
        self.P = nn.Embedding(n_users, factors)
        self.Q = nn.Embedding(n_items, factors)
        self.b = nn.Parameter(torch.tensor(mu))
        nn.init.normal_(self.P.weight, 0.0, 0.01)
        nn.init.normal_(self.Q.weight, 0.0, 0.01)

    def forward(self, u, i):
        return (self.P(u) * self.Q(i)).sum(dim=1) + self.b


class GMF(nn.Module):
    """r̂ = hᵀ(p_u ⊙ q_i) + b."""

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

    def forward(self, u, i):
        return self.h(self.features(u, i)).squeeze(1)


class MLP(nn.Module):
    """r̂ = hᵀ MLP([p_u ; q_i]) + b, with dropout after each ReLU."""

    def __init__(self, n_users, n_items, mu, layers, dropout=0.0):
        super().__init__()
        emb = layers[0] // 2                        # each embedding is half of the concatenation
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

    def forward(self, u, i):
        return self.h(self.features(u, i)).squeeze(1)


class NeuMF(nn.Module):
    """r̂ = hᵀ[φ_GMF ; φ_MLP] + b."""

    def __init__(self, gmf, mlp, mu):
        super().__init__()
        self.gmf, self.mlp = gmf, mlp
        self.h = nn.Linear(gmf.h.in_features + mlp.h.in_features, 1)
        nn.init.constant_(self.h.bias, mu)

    def forward(self, u, i):
        phi = torch.cat([self.gmf.features(u, i), self.mlp.features(u, i)], dim=1)
        return self.h(phi).squeeze(1)


class WithBiases(nn.Module):
    """r̂ = model(u, i) + b_u + b_i."""

    def __init__(self, model, n_users, n_items):
        super().__init__()
        self.model = model
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)

    def forward(self, u, i):
        return self.model(u, i) + self.b_u(u).squeeze(1) + self.b_i(i).squeeze(1)


def make_model(cfg, n_users, n_items, mu):
    name, f, layers = cfg["model"], cfg["factors"], cfg["layers"]
    if name == "MF":
        model = MF(n_users, n_items, mu, f)
    elif name == "GMF":
        model = GMF(n_users, n_items, mu, f)
    elif name == "MLP":
        model = MLP(n_users, n_items, mu, layers, cfg["dropout"])
    else:
        gmf = GMF(n_users, n_items, mu, f)
        mlp = MLP(n_users, n_items, mu, layers, cfg["dropout"])
        if cfg["share"]:                            # both branches use the same user and item tables
            mlp.P, mlp.Q = gmf.P, gmf.Q
        model = NeuMF(gmf, mlp, mu)
    return WithBiases(model, n_users, n_items) if cfg["biases"] else model


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#  training

@torch.no_grad()
def rmse_t(model, data, batch_size=65536):
    model.eval()
    u, i, r = data
    err = 0.0
    for s in range(0, len(r), batch_size):
        pred = model(u[s:s + batch_size], i[s:s + batch_size]).clamp(0.5, 5)
        err += ((pred - r[s:s + batch_size]) ** 2).sum().item()
    return float(np.sqrt(err / len(r)))


def train(model, train_data, val_data, cfg):
    """AdamW on squared error. With val_data: early stopping and the best weights are restored.
    Without val_data: runs exactly cfg["epochs"] epochs."""
    model.to(device)
    # decay embeddings and weight matrices, not the global offset or the layers' biases
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    u, i, r = train_data
    history = {"train": [], "val": []}
    best, best_state, bad = np.inf, None, 0

    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(r), device=device)
        for s in range(0, len(r), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            loss = ((model(u[b], i[b]) - r[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        history["train"].append(rmse_t(model, train_data))
        if val_data is not None:
            v = rmse_t(model, val_data)
            history["val"].append(v)
            if v < best:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def run(cfg, data, seed):
    """Trains one model on the 80% split. Returns its validation results."""
    torch.manual_seed(seed)
    model = make_model(cfg, data["n_users"], data["n_items"], data["mu_train"])
    t0 = time.time()
    h = train(model, data["train"], data["val"], cfg)
    best = int(np.argmin(h["val"]))
    return dict(val_rmse=h["val"][best], best_epoch=best + 1, epochs_run=len(h["val"]),
                params=n_params(model), seconds=time.time() - t0, history=h)


def run_test(cfg, data, seed, epochs):
    """Retrains on the whole ratings_train file for `epochs` epochs, scores once on ratings_test."""
    torch.manual_seed(seed)
    model = make_model(cfg, data["n_users"], data["n_items"], data["mu_full"])
    train(model, data["full"], None, dict(cfg, epochs=epochs))
    return rmse_t(model, data["test"])


#  search

def clean(cfg):
    """Sets the settings a model doesn't use to a fixed value, so duplicates can be removed.
    Returns None for invalid combinations."""
    cfg = dict(cfg)
    if cfg["model"] in ("MF", "GMF"):
        cfg.update(layers=None, dropout=0.0, share=False)
    if cfg["model"] == "MLP":
        cfg.update(factors=None, share=False)
    if cfg["layers"] is not None and cfg["layers"][0] % 2 == 1:
        return None                                 # layers[0] is split in two halves (user, item)
    if cfg["share"] and cfg["layers"][0] != 2 * cfg["factors"]:
        return None                                 # shared tables need the same embedding size
    return cfg


def grid_configs(base):
    """Every valid combination of GRID, without duplicates, always in the same order."""
    names = list(GRID)
    configs, seen = [], set()
    for values in itertools.product(*GRID.values()):
        cfg = clean(dict(base, **dict(zip(names, values))))
        if cfg is not None and key(cfg) not in seen:
            seen.add(key(cfg))
            configs.append(cfg)
    return configs


COLUMNS = ["key", "model", "factors", "layers", "dropout", "weight_decay", "biases", "share",
           "lr", "batch_size", "val_rmse", "val_std", "best_epoch", "params", "seconds"]


def key(cfg):
    """Short text that identifies a configuration (used to skip runs already done)."""
    layers = "-".join(map(str, cfg["layers"])) if cfg["layers"] else ""
    return (f"{cfg['model']}|f={cfg['factors']}|l={layers}|d={cfg['dropout']}|wd={cfg['weight_decay']}|"
            f"b={cfg['biases']}|s={cfg['share']}|lr={cfg['lr']}|bs={cfg['batch_size']}")


def evaluate(cfg, data, seeds):
    """Trains cfg once per seed. Returns one CSV row and the training curves."""
    results = [run(cfg, data, s) for s in seeds]
    rmses = [r["val_rmse"] for r in results]
    row = dict(key=key(cfg), model=cfg["model"], factors=cfg["factors"],
               layers="-".join(map(str, cfg["layers"])) if cfg["layers"] else "",
               dropout=cfg["dropout"], weight_decay=cfg["weight_decay"],
               biases=cfg["biases"], share=cfg["share"], lr=cfg["lr"], batch_size=cfg["batch_size"],
               val_rmse=round(float(np.mean(rmses)), 5), val_std=round(float(np.std(rmses)), 5),
               best_epoch=int(np.mean([r["best_epoch"] for r in results])),
               params=results[0]["params"],
               seconds=round(sum(r["seconds"] for r in results), 1))
    return row, [r["history"] for r in results]


def read_rows(out, pattern="results"):
    """All rows of the results*.csv files of a search folder (one file per part)."""
    rows = []
    for name in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        if name.startswith(pattern) and name.endswith(".csv"):
            with open(os.path.join(out, name)) as f:
                rows += list(csv.DictReader(f))
    return rows


def fmt(seconds):
    """1h05m / 12m30s"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def save_rows(configs, data, seeds, csv_path, hist_path):
    """Evaluates each config and appends its row right away (nothing is lost if the job stops)."""
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f_csv, open(hist_path, "a") as f_h:
        writer = csv.DictWriter(f_csv, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        t0 = time.time()
        for n, cfg in enumerate(configs, 1):
            row, histories = evaluate(cfg, data, seeds)
            writer.writerow(row)
            f_csv.flush()
            f_h.write(json.dumps(dict(key=row["key"], histories=histories)) + "\n")
            f_h.flush()
            elapsed = time.time() - t0
            left = elapsed / n * (len(configs) - n)          # average time so far x configs left
            print(f"[{n}/{len(configs)} {n / len(configs):4.0%}] val RMSE = {row['val_rmse']:.4f} "
                  f"(epoch {row['best_epoch']:>3}, {row['seconds']:5.1f}s)  {row['key']}  "
                  f"| elapsed {fmt(elapsed)}, left ~{fmt(left)}", flush=True)


def search(base, data, seeds, out, sample=None, part=0, n_parts=1):
    os.makedirs(out, exist_ok=True)
    configs = grid_configs(base)
    if sample is not None and sample < len(configs):      # random search: a fixed random subset
        idx = np.random.default_rng(0).choice(len(configs), sample, replace=False)
        configs = [configs[k] for k in sorted(idx)]
    configs = configs[part::n_parts]                       # this job's share of the configurations

    done = {row["key"] for row in read_rows(out)}           # resume: skip what any part already did
    todo = [c for c in configs if key(c) not in done]
    print(f"part {part + 1}/{n_parts}: {len(configs)} configurations, "
          f"{len(configs) - len(todo)} already done, {len(todo)} to run")

    with open(os.path.join(out, "search_settings.json"), "w") as f:
        json.dump(dict(grid={k: [list(v) if isinstance(v, tuple) else v for v in vals]
                             for k, vals in GRID.items()},
                       base=base, seeds=seeds, sample=sample, n_parts=n_parts), f, indent=2)

    suffix = f"_{part:03d}" if n_parts > 1 else ""
    save_rows(todo, data, seeds, os.path.join(out, f"results{suffix}.csv"),
              os.path.join(out, f"histories{suffix}.jsonl"))
    print(f"\npart {part + 1}/{n_parts} finished.")
    show_best(read_rows(out))


def rerun_top(base, data, seeds, out, k):
    """Retrains the k best configurations of a search with several seeds (written to top.csv)."""
    rows = sorted(read_rows(out), key=lambda r: float(r["val_rmse"]))[:k]
    if not rows:
        raise SystemExit(f"No results in {out}: run --search with the same --out first.")
    configs = []
    for r in rows:
        configs.append(dict(base, model=r["model"],
                            factors=int(r["factors"]) if r["factors"] not in ("", "None") else None,
                            layers=tuple(int(x) for x in r["layers"].split("-")) if r["layers"] else None,
                            dropout=float(r["dropout"]), weight_decay=float(r["weight_decay"]),
                            biases=r["biases"] == "True", share=r["share"] == "True",
                            lr=float(r["lr"]), batch_size=int(r["batch_size"])))
    print(f"rerunning the {len(configs)} best configurations with seeds {seeds}")
    path = os.path.join(out, "top.csv")
    if os.path.exists(path):
        os.remove(path)
    save_rows(configs, data, seeds, path, os.path.join(out, "top_histories.jsonl"))
    show_best(read_rows(out, pattern="top"), top=k)


def show_best(rows, top=10):
    rows = sorted(rows, key=lambda r: float(r["val_rmse"]))
    print(f"\n{len(rows)} configurations. Top {top} (validation RMSE):")
    for r in rows[:top]:
        print(f"  {r['val_rmse']}  ±{r['val_std']}  {r['key']}  (epoch {r['best_epoch']}, {r['params']} params)")


#  main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["MF", "GMF", "MLP", "NeuMF"], default="NeuMF")
    p.add_argument("--factors", type=int, default=8, help="embedding size of MF and GMF")
    p.add_argument("--layers", type=int, nargs="+", default=[16, 8], help="MLP layers, e.g. 64 32 16 8")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--no-biases", action="store_true", help="no user and item biases")
    p.add_argument("--share", action="store_true", help="NeuMF: GMF and MLP share embeddings")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150, help="max epochs (early stopping usually ends sooner)")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[20], help="several seeds: mean and std are reported")
    p.add_argument("--split-seed", type=int, default=42, help="seed of the 80/20 split (notebook: 42)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--data", default="datasets", help="folder with ratings_train.npy and ratings_test.npy")
    p.add_argument("--search", action="store_true", help="try every valid combination of GRID")
    p.add_argument("--sample", type=int, default=None, help="--search: only this many random configurations")
    p.add_argument("--part", type=int, default=0, help="--search: which part this job runs (0, 1, ...)")
    p.add_argument("--n-parts", type=int, default=1, help="--search: number of parts (Slurm job array)")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for torch (default: all)")
    p.add_argument("--top", type=int, default=None, help="rerun the TOP best configurations of --out with --seeds")
    p.add_argument("--out", default="results/search", help="output folder of --search and --top")
    p.add_argument("--test", action="store_true", help="also retrain on all ratings_train and score on ratings_test")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    elif "SLURM_CPUS_PER_TASK" in os.environ:       # on the cluster: use only the CPUs Slurm gave us
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    data = load_data(args.data, args.val_frac, args.split_seed)
    cfg = dict(model=args.model, factors=args.factors, layers=tuple(args.layers),
               dropout=args.dropout, weight_decay=args.weight_decay, biases=not args.no_biases,
               share=args.share, lr=args.lr, batch_size=args.batch_size,
               epochs=args.epochs, patience=args.patience)
    print(f"device: {device}")

    if args.search:
        search(cfg, data, args.seeds, args.out, args.sample, args.part, args.n_parts)
    elif args.top:
        rerun_top(cfg, data, args.seeds, args.out, args.top)
    else:
        cfg = clean(cfg)
        if cfg is None:
            raise SystemExit("Invalid combination: layers[0] must be even, and with --share, layers[0] = 2 * factors.")
        results = [run(cfg, data, s) for s in args.seeds]
        rmses = [r["val_rmse"] for r in results]
        best_ep = int(np.mean([r["best_epoch"] for r in results]))
        print(f"{key(cfg)}")
        print(f"validation RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}  "
              f"(best epoch {best_ep}, {results[0]['params']:,} params, "
              f"{np.mean([r['seconds'] for r in results]):.1f}s per seed)")
        if args.test:
            tests = [run_test(cfg, data, s, best_ep) for s in args.seeds]
            print(f"test RMSE = {np.mean(tests):.4f} ± {np.std(tests):.4f}  "
                  f"(retrained on all of ratings_train for {best_ep} epochs)")
