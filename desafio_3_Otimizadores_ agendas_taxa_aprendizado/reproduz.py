# reproduz.py — uso: python reproduz.py <run_id>
import json, math, subprocess, sys, pandas as pd
r = pd.read_csv("experimentos.csv").set_index("run_id").loc[sys.argv[1]]
cmd = [sys.executable, "treino.py", "--opt", r.opt, "--lr", str(r.lr), "--sched", r.sched, "--seed", str(int(r.seed)),
       "--steps", str(int(r.steps)), "--bs", str(int(r.bs)), "--wd", str(r.wd), "--clip", str(r.clip),
       "--k", str(int(r.k)), "--device", r.device, "--no_save"]
if r.opt == "muon": cmd += ["--lr_adamw", str(r.lr_adamw)]
out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.splitlines()
novo = json.loads([l for l in out if l.startswith("RESULT ")][-1][7:])
print("CSV :", r.val_loss_best, "\nNovo:", novo["val_loss_best"])
print("OK" if math.isclose(novo["val_loss_best"], r.val_loss_best, rel_tol=1e-4, abs_tol=1e-6) else "DIFERE")
