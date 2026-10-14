"""Teste de sanidade por otimizador (~1 min cada, CPU).
Uso:  python teste_otimizador.py adam
      python teste_otimizador.py adamw
      python teste_otimizador.py todos     # sgd, sgd_nesterov, adam, adamw, muon
Roda treino.py duas vezes (mesma semente) em pasta temporária, sem tocar em logs/ nem experimentos.csv."""
import json, math, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")  # console do Windows
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")  # console do Windows
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from dados import localiza
DADOS = str(localiza())   # gera data/features.csv sozinho se só existir o CSV bruto
STEPS = 300
# η só para o teste de sanidade (não são os valores finais — quem decide é a varredura)
LR = {"sgd": 1e-2, "sgd_nesterov": 1e-2, "adam": 1e-3, "adamw": 1e-3, "muon": 1e-2}
LR_ADAMW_MUON = 1e-3

def roda(opt):
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(os.path.join(AQUI, "data"), os.path.join(tmp, "data"))  # cópia: symlink exige admin no Windows
        cmd = [sys.executable, os.path.join(AQUI, "treino.py"), "--opt", opt, "--lr", str(LR[opt]),
               "--sched", "warmcos", "--seed", "0", "--steps", str(STEPS)]
        if opt == "muon": cmd += ["--lr_adamw", str(LR_ADAMW_MUON)]
        p = subprocess.run(cmd, cwd=tmp, env=dict(os.environ, PYTHONPATH=AQUI, PYTHONIOENCODING="utf-8"),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print(p.stderr[-2000:]); sys.exit(f"treino.py falhou com {opt} — veja o erro acima")
        res = json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1][7:])
        return res, pd.read_csv(os.path.join(tmp, "logs", f"{res['run_id']}.csv"))

def testa(opt):
    print(f"\n===== {opt} (η={LR[opt]:g}{', η_adamw=%g' % LR_ADAMW_MUON if opt == 'muon' else ''}) =====")
    falhas = []
    def check(nome, cond, info=""):
        print(("[OK]    " if cond else "[FALHA] ") + nome, info)
        if not cond: falhas.append(nome)
    r1, log = roda(opt); r2, _ = roda(opt)
    lr0 = LR[opt]
    check("não divergiu", not r1["divergiu"])
    check("perda inicial perto de ln3 (≈1,10)", 0.9 < log.perda.iloc[0] < 1.5, f"({log.perda.iloc[0]:.3f})")
    cols = ["passo", "perda", "grad", "theta", "lr", "val_loss", "grad/fc1.weight", "grad/fc2.weight",
            "grad/head.weight", "theta/fc1.weight", "rho/fc1.weight", "rho/fc2.weight", "rho/head.weight"]
    if opt == "muon": cols.append("lr_adamw")
    check("colunas do registro presentes", all(c in log.columns for c in cols))
    check("registro a cada k=25 passos", list(log.passo[:3]) == [0, 25, 50])
    check("val_loss a cada 50 passos", log.val_loss.notna().sum() >= STEPS // 50)
    check("grad e theta finitos", np.isfinite(log[["grad", "theta"]].values).all())
    check("η começa baixa (warmup)", log.lr.iloc[0] < 0.5 * lr0, f"({log.lr.iloc[0]:.2e})")
    check("η atinge ~pico após warmup", abs(log.lr.max() - lr0) < 0.1 * lr0, f"({log.lr.max():.2e})")
    check("η termina perto de 0 (cosseno)", log.lr.iloc[-1] < 0.1 * lr0, f"({log.lr.iloc[-1]:.2e})")
    check("razão de atualização plausível (1e-8 a 1)", log["rho/fc2.weight"].between(1e-8, 1).all(),
          f"(mediana fc2: {log['rho/fc2.weight'].median():.1e}; referência ~1e-3)")
    check("determinístico (2 execuções, semente 0)",
          math.isclose(r1["val_loss_best"], r2["val_loss_best"], rel_tol=1e-4, abs_tol=1e-6),
          f"({r1['val_loss_best']:.5f} vs {r2['val_loss_best']:.5f})")
    print(f"val_loss_best={r1['val_loss_best']:.4f} | acc={r1['val_acc_best']:.3f} | F1={r1['val_f1_best']:.3f} "
          f"| melhor passo={r1['step_best']} | {r1['tempo_s']}s")
    return falhas, r1

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "adam"
    ops = list(LR) if arg == "todos" else [arg]
    if any(o not in LR for o in ops): sys.exit(f"otimizador inválido; use: {', '.join(LR)} ou todos")
    resumo = {o: testa(o) for o in ops}
    print("\nReferência na validação: 'sempre mandante' ≈ 0,489 | só Elo ≈ 0,515. Acurácia > ~0,70 sugere vazamento.")
    ruins = {o: f for o, (f, _) in resumo.items() if f}
    if ruins: sys.exit(f"\nFalhas: {ruins}")
    print("\nTudo certo. Pode seguir com python rodar.py 1")
