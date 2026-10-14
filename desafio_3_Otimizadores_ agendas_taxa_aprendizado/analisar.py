# analisar.py — tabelas e regra de decisão
import numpy as np, pandas as pd
from rodar import BASELINE
d = pd.read_csv("experimentos.csv"); d["la"] = d.lr_adamw.fillna(0)
print(f"{len(d)} execuções | {int(d.divergiu.sum())} divergiram")
print("\nDivergentes:\n", d[d.divergiu][["run_id"]].to_string(index=False))
print("\n== Etapa 2: val_loss_best × η (warmcos, semente 0; NaN = divergiu) ==")
s = d[(d.sched == "warmcos") & (d.seed == 0)]
print(s.pivot_table(index=["opt", "la"], columns="lr", values="val_loss_best", dropna=False).round(4).to_string())
print("\n== Etapa 3: agendas (semente 0) ==")
p = d[d.seed == 0].pivot_table(index=["opt", "lr", "la"], columns="sched", values="val_loss_best").dropna(thresh=2)
print(p.round(4).to_string())
print("\n== Etapa 4: confirmação (>=3 sementes) ==")
cfg = ["opt", "lr", "sched", "la"]
g = (d.groupby(cfg).agg(n=("seed", "nunique"), media=("val_loss_best", "mean"), dp=("val_loss_best", "std"),
                        acc=("val_acc_best", "mean"), f1=("val_f1_best", "mean")).reset_index())
g = g[g.n >= 3].sort_values("media").reset_index(drop=True)
b = g[(g.opt == BASELINE[0]) & np.isclose(g.lr, BASELINE[1]) & (g.sched == BASELINE[2])].iloc[0]
g["ganho_vs_base"] = b.media - g.media
g["supera_dp"] = g.ganho_vs_base > np.maximum(g.dp, b.dp)
print(g.round(4).to_string(index=False))
print("\nBaseline:", f"{b.media:.4f} ± {b.dp:.4f}")
if len(g) >= 2:
    a_, c_ = g.iloc[0], g.iloc[1]
    print("1º vs 2º: diferença =", round(c_.media - a_.media, 4), "| maior desvio =", round(max(a_.dp, c_.dp), 4),
          "→", "há vencedor" if (c_.media - a_.media) > max(a_.dp, c_.dp) else "NÃO há vencedor claro")
