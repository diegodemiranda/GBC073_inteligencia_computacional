# avaliar_teste.py — teste, UMA única vez
import os, sys, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from comum import carrega, MLP
if os.path.exists("teste.csv"): sys.exit("teste.csv já existe: o teste só deve ser usado uma vez.")
D = carrega("cpu"); X, y = D["te"]; yt = y.numpy()
d = pd.read_csv("experimentos.csv"); d["la"] = d.lr_adamw.fillna(0)
cont = d.groupby(["opt", "lr", "sched", "la"]).seed.transform("nunique")
runs = d[cont >= 3].run_id.tolist()
maj = int(np.bincount(D["tr"][1].numpy()).argmax())
print("acurácia 'sempre classe", maj, "' no teste:", round(float((yt == maj).mean()), 4))
rows = []
for r in runs:
    m = MLP(D["n_in"]); m.load_state_dict(torch.load(f"ckpt/{r}.pt", map_location="cpu")); m.eval()
    with torch.no_grad(): out = m(X)
    pred = out.argmax(1).numpy()
    rows.append(dict(run_id=r, test_loss=F.cross_entropy(out, y).item(), acc=float((pred == yt).mean()),
                     f1_macro=float(f1_score(yt, pred, average="macro"))))
    if r.endswith("_s0") or "_s0_" in r:
        print("\n", r); print(confusion_matrix(yt, pred)); print(classification_report(yt, pred, digits=3, zero_division=0))
pd.DataFrame(rows).to_csv("teste.csv", index=False); print(pd.DataFrame(rows).round(4).to_string(index=False))
