# comum.py — dados, modelo MLP, Muon próprio (Newton–Schulz), run_id
import math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

META = ["id", "data", "ano", "mandante", "visitante", "y"]

def rid(opt, lr, sched, seed, la=None):
    s = f"{opt}_lr{lr:g}_{sched}_s{seed}"
    return s + (f"_la{la:g}" if opt == "muon" else "")

def carrega(dev="cpu"):
    df = pd.read_csv("data/features.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    feats = [c for c in df.columns if c not in META]
    tr, va, te = df[df.ano <= 2017], df[df.ano.between(2018, 2020)], df[df.ano >= 2021]
    mu, sd = tr[feats].mean(), tr[feats].std() + 1e-8          # normaliza SÓ com estatísticas do treino
    f = lambda d: (torch.tensor(((d[feats] - mu) / sd).values, dtype=torch.float32, device=dev),
                   torch.tensor(d["y"].values, dtype=torch.long, device=dev))
    return dict(tr=f(tr), va=f(va), te=f(te), n_in=len(feats), feats=feats)

class MLP(nn.Module):
    def __init__(self, n_in, h=(256, 256, 128), p=0.2):
        super().__init__()
        self.fc1, self.n1 = nn.Linear(n_in, h[0]), nn.LayerNorm(h[0])
        self.fc2, self.n2 = nn.Linear(h[0], h[1]), nn.LayerNorm(h[1])
        self.fc3, self.n3 = nn.Linear(h[1], h[2]), nn.LayerNorm(h[2])
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(h[2], 3)                          # "head" => AdamW quando o otimizador é Muon
    def forward(self, x):
        x = self.drop(F.gelu(self.n1(self.fc1(x))))
        x = self.drop(F.gelu(self.n2(self.fc2(x))))
        x = self.drop(F.gelu(self.n3(self.fc3(x))))
        return self.head(x)

def ortogonaliza(G, passos=5):
    """Newton–Schulz: aproxima U V^T de G = U S V^T (5 iterações)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    T = X.size(0) > X.size(1)
    if T: X = X.T
    for _ in range(passos):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if T else X

class Muon(torch.optim.Optimizer):
    """Muon próprio: momento -> ortogonalização (Newton–Schulz) -> passo. Só para matrizes 2-D."""
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "mom" not in st: st["mom"] = torch.zeros_like(p.grad)
                st["mom"].mul_(g["momentum"]).add_(p.grad)
                upd = ortogonaliza(st["mom"])
                if g["weight_decay"]: p.mul_(1 - g["lr"] * g["weight_decay"])   # decaimento desacoplado
                escala = math.sqrt(max(1, p.shape[0] / max(1, p.shape[1])))
                p.add_(upd, alpha=-g["lr"] * escala)
