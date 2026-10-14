# painel.py → painel.pdf
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from rodar import BASELINE
CAM = ["fc1.weight", "fc2.weight", "head.weight"]
d = pd.read_csv("experimentos.csv"); d["la"] = d.lr_adamw.fillna(0); ok = d[~d.divergiu]
L = lambda r: pd.read_csv(f"logs/{r}.csv")
def seed0(opt, lr, sched, la=0.0):
    q = d[(d.opt == opt) & np.isclose(d.lr, lr) & (d.sched == sched) & np.isclose(d.la, la) & (d.seed == 0)]
    return q.run_id.iloc[0]
g = ok.groupby(["opt", "lr", "sched", "la"]).agg(n=("seed", "nunique"), m=("val_loss_best", "mean")).reset_index()
fin = g[g.n >= 3].sort_values("m")
melhor = seed0(fin.iloc[0].opt, fin.iloc[0].lr, fin.iloc[0].sched, fin.iloc[0].la)
base = seed0(*BASELINE)
mu = ok[(ok.opt == "muon") & (ok.seed == 0)].sort_values("val_loss_best").run_id.iloc[0]
aw = ok[(ok.opt == "adamw") & (ok.seed == 0)].sort_values("val_loss_best").run_id.iloc[0]
dv = d[d.divergiu].sort_values("lr", ascending=False)
if len(dv): inst = dv.run_id.iloc[0]
else:
    pico = {r: (lambda l: l.grad.max() / l.grad.median())(L(r)) for r in d.run_id}; inst = max(pico, key=pico.get)
def sinais(r, titulo):
    l = L(r); fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax[0].plot(l.passo, l.perda.rolling(5, min_periods=1).mean(), label="treino (lote, média móvel)")
    v = l.dropna(subset=["val_loss"]); ax[0].plot(v.passo, v.val_loss, "o-", ms=3, label="validação"); ax[0].legend()
    ax[0].set_ylabel("perda"); ax[1].semilogy(l.passo, l.grad); ax[1].set_ylabel("‖g‖ (antes do clip)")
    ax[2].plot(l.passo, l.theta); ax[2].set_ylabel("‖θ‖")
    ax[3].plot(l.passo, l.lr, label="η matrizes / único")
    if "lr_adamw" in l: ax[3].plot(l.passo, l.lr_adamw, "--", label="η AdamW (Muon)")
    ax[3].set_ylabel("η_t"); ax[3].set_xlabel("passo"); ax[3].legend()
    for x in ax: x.grid(alpha=.3)
    fig.suptitle(f"{titulo}\n{r}", fontsize=10); fig.tight_layout(); return fig
def pag_rho():
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for r, c in [(aw, "C0"), (mu, "C3")]:
        l = L(r)
        for j, cam in enumerate(CAM):
            ax[0, j].semilogy(l.passo, l[f"rho/{cam}"], c, label=r.split("_lr")[0]); ax[0, j].set_title(cam)
            ax[1, j].semilogy(l.passo, l[f"grad/{cam}"], c)
    ax[0, 0].set_ylabel("razão de atualização ‖Δθ‖/‖θ‖"); ax[1, 0].set_ylabel("‖g‖ da camada (antes do clip)")
    for x in ax[1]: x.set_xlabel("passo")
    for x in ax.ravel(): x.grid(alpha=.3)
    ax[0, 0].legend(); fig.suptitle(f"Razão de atualização e gradiente por camada: AdamW × Muon\n{aw} | {mu}", fontsize=10)
    fig.tight_layout(); return fig
def pag_lr():
    fig, ax = plt.subplots(figsize=(9, 6)); s = d[(d.sched == "warmcos") & (d.seed == 0)]
    top = np.nanmax(s.val_loss_best) * 1.02
    for o in s.opt.unique():
        q = s[s.opt == o].sort_values("lr"); ax.semilogx(q.lr, q.val_loss_best, "o-", label=o)
        x = q[q.divergiu]; ax.scatter(x.lr, [top] * len(x), marker="x", c="r", zorder=5)
    ax.set_xlabel("η (no Muon: η das matrizes)"); ax.set_ylabel("val_loss_best")
    ax.set_title("Métrica × η por otimizador (x vermelho = divergiu)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); return fig
with PdfPages("painel.pdf") as pdf:
    for r, t in [(melhor, "Melhor configuração"), (base, "Baseline"), (mu, "Melhor com Muon"), (inst, "Configuração instável")]:
        f = sinais(r, t); pdf.savefig(f); plt.close(f)
    for f in (pag_rho(), pag_lr()): pdf.savefig(f); plt.close(f)
print("painel.pdf gerado")
