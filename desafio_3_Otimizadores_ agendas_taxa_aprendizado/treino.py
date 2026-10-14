# treino.py — configurável: otimizador, η, agenda, semente (entrega principal)
import argparse, csv, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from sklearn.metrics import f1_score
from comum import carrega, MLP, rid, Muon

p = argparse.ArgumentParser()
p.add_argument("--opt", required=True, choices=["sgd", "sgd_nesterov", "adam", "adamw", "muon"])
p.add_argument("--lr", type=float, required=True)                # no Muon: η das matrizes
p.add_argument("--lr_adamw", type=float, default=float("nan"))   # só Muon: η da parte AdamW
p.add_argument("--sched", default="warmcos", choices=["const", "cos", "warmcos", "wsd"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--clip", type=float, default=5.0)
p.add_argument("--k", type=int, default=25)
p.add_argument("--eval_every", type=int, default=50)
p.add_argument("--device", default="cpu")
p.add_argument("--no_save", action="store_true")
a = p.parse_args()
assert a.eval_every % a.k == 0
if a.opt == "muon" and math.isnan(a.lr_adamw):
    p.error("--lr_adamw é obrigatório com --opt muon")

RID = rid(a.opt, a.lr, a.sched, a.seed, a.lr_adamw)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = a.device
D = carrega(dev)
Xtr, ytr = D["tr"]; Xva, yva = D["va"]
model = MLP(D["n_in"]).to(dev)
named = dict(model.named_parameters())
CAM = ["fc1.weight", "fc2.weight", "head.weight"]               # primeira, do meio e última

if a.opt == "muon":
    muon_p = [q for n, q in model.named_parameters() if q.ndim == 2 and "head" not in n and "emb" not in n]
    ids = {id(q) for q in muon_p}
    adamw_p = [q for q in model.parameters() if id(q) not in ids]
    opts = [Muon(muon_p, lr=a.lr, momentum=0.95, weight_decay=a.wd),
            torch.optim.AdamW(adamw_p, lr=a.lr_adamw, weight_decay=a.wd)]
else:
    ps = model.parameters()
    opts = [{"sgd":          lambda: torch.optim.SGD(ps, lr=a.lr, momentum=0.9, weight_decay=a.wd),
             "sgd_nesterov": lambda: torch.optim.SGD(ps, lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd),
             "adam":         lambda: torch.optim.Adam(ps, lr=a.lr, weight_decay=a.wd),
             "adamw":        lambda: torch.optim.AdamW(ps, lr=a.lr, weight_decay=a.wd)}[a.opt]()]

T, W = a.steps, max(1, int(0.05 * a.steps))
def fator(t):
    if a.sched == "const":   return 1.0
    if a.sched == "cos":     return 0.5 * (1 + math.cos(math.pi * t / T))
    if a.sched == "warmcos": return (t + 1) / W if t < W else 0.5 * (1 + math.cos(math.pi * (t - W) / max(1, T - W)))
    if a.sched == "wsd":
        if t < W: return (t + 1) / W
        if t < 0.8 * T: return 1.0
        return max(0.0, (T - t) / (0.2 * T))
scheds = [torch.optim.lr_scheduler.LambdaLR(o, fator) for o in opts]   # a MESMA agenda nos dois otimizadores

n = len(ytr); g = torch.Generator().manual_seed(a.seed)
def lotes():
    while True:
        perm = torch.randperm(n, generator=g)
        for i in range(0, n - a.bs + 1, a.bs):
            yield perm[i:i + a.bs]
it = lotes()

@torch.no_grad()
def avalia(X, y):
    model.eval(); out = model(X)
    l = F.cross_entropy(out, y).item(); acc = (out.argmax(1) == y).float().mean().item()
    model.train(); return l, acc

log, diverg, t0 = [], False, time.time()
best = dict(loss=float("inf"), step=-1, state=None)
model.train()
for t in range(a.steps):
    idx = next(it).to(dev); x, y = Xtr[idx], ytr[idx]
    for o in opts: o.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y); loss.backward()
    logar = (t % a.k == 0)
    if logar:                                                   # tudo ANTES do clipping e do step
        gl  = {nm: named[nm].grad.norm().item() for nm in CAM}
        th  = torch.sqrt(sum((q.detach() ** 2).sum() for q in model.parameters())).item()
        thl = {nm: named[nm].detach().norm().item() for nm in CAM}
        antes = {nm: named[nm].detach().clone() for nm in CAM}
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()   # norma ANTES do corte
    row = None
    if logar:
        row = dict(passo=t, perda=loss.item(), grad=gnorm, theta=th, lr=opts[0].param_groups[0]["lr"],
                   **{f"grad/{nm}": v for nm, v in gl.items()}, **{f"theta/{nm}": v for nm, v in thl.items()})
        if len(opts) > 1: row["lr_adamw"] = opts[1].param_groups[0]["lr"]
    if (not math.isfinite(loss.item())) or (not math.isfinite(gnorm)) or loss.item() > 20:
        diverg = True
        if row: log.append(row)
        break
    for o in opts: o.step()
    if logar:
        row.update({f"rho/{nm}": ((named[nm].detach() - antes[nm]).norm() / antes[nm].norm()).item() for nm in CAM})
        if t % a.eval_every == 0:
            vl, vacc = avalia(Xva, yva); row.update(val_loss=vl, val_acc=vacc)
            if vl < best["loss"]:
                best.update(loss=vl, step=t, state={k_: v_.detach().clone() for k_, v_ in model.state_dict().items()})
        log.append(row)
    for s in scheds: s.step()

if not diverg:
    vl, _ = avalia(Xva, yva); val_final = vl
    if vl < best["loss"]:
        best.update(loss=vl, step=a.steps, state={k_: v_.detach().clone() for k_, v_ in model.state_dict().items()})
else:
    val_final = float("nan")

if (not diverg) and best["state"] is not None:
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad(): pred = model(Xva).argmax(1).cpu().numpy()
    vacc = float((pred == yva.cpu().numpy()).mean())
    vf1 = float(f1_score(yva.cpu().numpy(), pred, average="macro"))
    vbest, sbest = best["loss"], best["step"]
else:
    vbest = vacc = vf1 = float("nan"); sbest = -1

res = dict(run_id=RID, opt=a.opt, lr=a.lr, lr_adamw="" if a.opt != "muon" else a.lr_adamw, sched=a.sched,
           seed=a.seed, steps=a.steps, bs=a.bs, wd=a.wd, clip=a.clip, k=a.k,
           val_loss_best=vbest, step_best=sbest, val_loss_final=val_final, val_acc_best=vacc, val_f1_best=vf1,
           divergiu=diverg, tempo_s=round(time.time() - t0, 1), device=dev, torch=torch.__version__)
print("RESULT " + json.dumps(res))

if not a.no_save:
    os.makedirs("logs", exist_ok=True); os.makedirs("ckpt", exist_ok=True)
    pd.DataFrame(log).to_csv(f"logs/{RID}.csv", index=False)
    if not diverg and best["state"] is not None: torch.save(best["state"], f"ckpt/{RID}.pt")
    novo = not os.path.exists("experimentos.csv")
    with open("experimentos.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        if novo: w.writeheader()
        w.writerow(res)
