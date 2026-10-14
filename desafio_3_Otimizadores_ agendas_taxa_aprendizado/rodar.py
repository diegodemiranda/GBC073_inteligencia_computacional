# rodar.py — uso: python rodar.py 1|2|3|4|tudo   (retoma de onde parou)
import os, subprocess, sys
import numpy as np, pandas as pd
from comum import rid

LADDER = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
SCHEDS = ["const", "cos", "warmcos", "wsd"]
OPTS   = ["sgd", "sgd_nesterov", "adam", "adamw", "muon"]
BASELINE = ("adamw", 3e-4, "warmcos")  # = padrão do treino.py original do grupo; troque se a receita atual for outra e declare no README

def res():
    return pd.read_csv("experimentos.csv") if os.path.exists("experimentos.csv") else pd.DataFrame()

def rodar(opt, lr, sched="warmcos", seed=0, la=None):
    d = res()
    if len(d) and rid(opt, lr, sched, seed, la) in set(d.run_id): return
    cmd = [sys.executable, "treino.py", "--opt", opt, "--lr", str(lr), "--sched", sched, "--seed", str(seed)]
    if opt == "muon": cmd += ["--lr_adamw", str(la)]
    print(">>", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)

def linha(opt, lr, sched, seed, la=None):
    d = res()
    if d.empty: return None
    m = (d.opt == opt) & (d.sched == sched) & (d.seed == seed) & np.isclose(d.lr, lr)
    if la is not None: m &= np.isclose(d.lr_adamw.astype(float), la)
    return d[m].iloc[0] if m.any() else None

def tabela(opt, sched="warmcos", seed=0, la=None):
    d = res()
    if d.empty: return d
    d = d[(d.opt == opt) & (d.sched == sched) & (d.seed == seed) & (~d.divergiu)]
    if la is not None: d = d[np.isclose(d.lr_adamw.astype(float), la)]
    return d.sort_values("val_loss_best")

def melhor_lr(opt, la=None):
    return float(tabela(opt, "warmcos", 0, la).iloc[0].lr)

def varre(opt, lo, hi, la=None):
    """>=5 valores; se o melhor cair na borda, estende a grade."""
    i, j = LADDER.index(lo), LADDER.index(hi)
    while True:
        for lr in LADDER[i:j + 1]: rodar(opt, lr, "warmcos", 0, la)
        t = tabela(opt, "warmcos", 0, la)
        if t.empty: break
        b = float(t.iloc[0].lr)
        if np.isclose(b, LADDER[i]) and i > 0: i -= 1
        elif np.isclose(b, LADDER[j]) and j < len(LADDER) - 1: j += 1
        else: break

def etapa1():
    for s in (0, 1, 2): rodar(BASELINE[0], BASELINE[1], BASELINE[2], s)

def etapa2():
    varre("sgd", 1e-3, 3e-1); varre("sgd_nesterov", 1e-3, 3e-1)
    varre("adam", 1e-4, 1e-2); varre("adamw", 1e-4, 1e-2)
    la = melhor_lr("adamw")
    varre("muon", 1e-3, 1e-1, la)

def melhores():
    la = melhor_lr("adamw"); m = {}
    for o in OPTS:
        t = tabela(o, "warmcos", 0, la if o == "muon" else None)
        if len(t): m[o] = (float(t.iloc[0].lr), float(t.iloc[0].val_loss_best))
    return m, la

def etapa3():
    m, la = melhores(); ordem = sorted(m, key=lambda o: m[o][1])
    alvos = ordem[:2] + (["muon"] if "muon" not in ordem[:2] else [])
    print("Otimizadores na etapa 3:", alvos)
    for o in alvos:
        for s in SCHEDS: rodar(o, m[o][0], s, 0, la if o == "muon" else None)

def etapa4():
    m, la = melhores(); ordem = sorted(m, key=lambda o: m[o][1])
    alvos = ordem[:2] + (["muon"] if "muon" not in ordem[:2] else [])
    cands = []
    for o in alvos:
        for s in SCHEDS:
            r = linha(o, m[o][0], s, 0, la if o == "muon" else None)
            if r is not None and not r.divergiu: cands.append((float(r.val_loss_best), o, m[o][0], s))
    top = sorted(cands)[:3]; print("Finalistas:", top)
    for _, o, lr, s in top:
        for seed in (1, 2): rodar(o, lr, s, seed, la if o == "muon" else None)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "tudo":
        for f in (etapa1, etapa2, etapa3, etapa4): f()
    elif arg in ("1", "2", "3", "4"):
        {"1": etapa1, "2": etapa2, "3": etapa3, "4": etapa4}[arg]()
    else:
        sys.exit("uso: python rodar.py 1|2|3|4|tudo")
