"""dados.py — localiza (e, se preciso, GERA) data/features.csv. Só usa pandas/numpy (sem torch).
Funciona mesmo se você só tiver o CSV bruto do Brasileirão em qualquer um destes lugares:
  data/brasileirao_bruto.csv | brasileirao_bruto.csv | features.csv | data/features.csv (se for o bruto)
Regra anti-vazamento: toda feature de um jogo usa SOMENTE jogos anteriores; o Elo é gravado ANTES da atualização.
Classes: 0 = mandante | 1 = empate | 2 = visitante."""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "features.csv"

def _cols(p):
    return [c.strip().lower() for c in pd.read_csv(p, nrows=0).columns]

def processado(p):
    return "h_pts_5" in _cols(p)

def bruto(p):
    c = _cols(p)
    return "mandante_placar" in c and "h_pts_5" not in c

def gerar(src, dst=PROC, verbose=True):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if src == dst:   # nunca sobrescrever o bruto com o processado
        novo = dst.parent / "brasileirao_bruto.csv"
        shutil.move(str(src), str(novo)); src = novo
        if verbose: print(f"[dados] o bruto estava em {dst.name}; movido para {novo.name}")
    j = pd.read_csv(src)
    j.columns = [c.strip() for c in j.columns]
    j["data"] = pd.to_datetime(j["data"], dayfirst=True, errors="coerce")
    j = j.dropna(subset=["data", "mandante_Placar", "visitante_Placar"])
    j = j[j["data"] <= "2023-12-31"].sort_values(["data", "ID"]).reset_index(drop=True)
    assert j["ID"].is_unique
    j["gm"], j["gv"] = j["mandante_Placar"].astype(int), j["visitante_Placar"].astype(int)
    j["y"] = np.select([j.gm > j.gv, j.gm == j.gv], [0, 1], 2)   # alvo vem do PLACAR

    elo, eh_l, ea_l = {}, [], []                                  # Elo ANTES da partida
    for r in j.itertuples():
        eh, ea = elo.get(r.mandante, 1500.0), elo.get(r.visitante, 1500.0)
        eh_l.append(eh); ea_l.append(ea)
        esp = 1 / (1 + 10 ** ((ea - (eh + 60)) / 400))            # 60 = vantagem de mando
        s = 1.0 if r.gm > r.gv else 0.5 if r.gm == r.gv else 0.0
        elo[r.mandante] = eh + 20 * (s - esp); elo[r.visitante] = ea - 20 * (s - esp)
    j["elo_h"], j["elo_a"] = eh_l, ea_l

    h = pd.DataFrame(dict(ID=j.ID, data=j.data, time=j.mandante, casa=1, gf=j.gm, ga=j.gv))
    v = pd.DataFrame(dict(ID=j.ID, data=j.data, time=j.visitante, casa=0, gf=j.gv, ga=j.gm))
    L = pd.concat([h, v]).sort_values(["time", "data", "ID"]).reset_index(drop=True)
    L["pts"] = np.select([L.gf > L.ga, L.gf == L.ga], [3, 1], 0)
    L["vit"] = (L.gf > L.ga).astype(float)
    rolar = lambda g, c, w, m: g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=m).mean())
    gt = L.groupby("time")
    for w in (5, 20):
        for c in ("pts", "gf", "ga", "vit"): L[f"{c}_{w}"] = rolar(gt, c, w, 5)
    gl = L.groupby(["time", "casa"])
    for c in ("pts", "gf", "ga"): L[f"{c}_mando5"] = rolar(gl, c, 5, 3)
    L["descanso"] = gt["data"].diff().dt.days.clip(upper=30)

    feats = [c for c in L.columns if c not in ("ID", "data", "time", "casa", "gf", "ga", "pts", "vit")]
    H = L[L.casa == 1].set_index("ID")[feats].add_prefix("h_")
    A = L[L.casa == 0].set_index("ID")[feats].add_prefix("a_")
    D = j.set_index("ID").join(H).join(A)
    D["h_elo"], D["a_elo"] = j.set_index("ID")["elo_h"], j.set_index("ID")["elo_a"]
    for c in feats + ["elo"]: D[f"d_{c}"] = D[f"h_{c}"] - D[f"a_{c}"]
    D["ano"] = D["data"].dt.year
    cols_x = [c for c in D.columns if c.startswith(("h_", "a_", "d_"))]
    out = D.reset_index()[["ID", "data", "ano", "mandante", "visitante", "y"] + cols_x].dropna()
    out["data"] = out["data"].dt.strftime("%d/%m/%Y")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    if verbose:
        print(f"[dados] gerado {dst} | {len(out)} jogos | {len(cols_x)} features")
        for nome, m in [("treino <=2017", out.ano <= 2017), ("val 2018-2020", out.ano.between(2018, 2020)),
                        ("teste >=2021", out.ano >= 2021)]:
            print(f"        {nome}: {int(m.sum())} jogos | classe majoritária {out[m].y.value_counts(normalize=True).max():.3f}")
    return out

def localiza():
    """Devolve o caminho do CSV processado; gera a partir do bruto se necessário."""
    cwd = Path.cwd()
    for p in (cwd / "data" / "features.csv", PROC):
        if p.exists() and processado(p): return p
    for p in (BASE / "data" / "brasileirao_bruto.csv", BASE / "brasileirao_bruto.csv", BASE / "features.csv",
              PROC, cwd / "features.csv", cwd / "data" / "features.csv"):
        if p.exists() and bruto(p):
            gerar(p, PROC); return PROC
    raise FileNotFoundError("Não achei os dados. Coloque o CSV bruto do Brasileirão (com colunas ID, data, mandante, "
                            f"visitante, mandante_Placar, visitante_Placar) em {BASE} e rode de novo.")
