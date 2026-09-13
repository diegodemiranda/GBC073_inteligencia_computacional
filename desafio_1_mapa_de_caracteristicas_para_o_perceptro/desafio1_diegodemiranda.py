"""
Autor: Diego de Miranda da Silva - 12311BCC060
desafio1_diegodemiranda.py — Desafio 1: Mapa de características para o Perceptron
GBC073 — Inteligência Computacional (FACOM/UFU)

O QUE VOCÊ FAZ: preencher a classe `Submissao` (e só ela).
O QUE O HARNESS FAZ: gera dados, divide 60/40, aplica a sua phi, treina um
Perceptron fixo e mede a acurácia. Rode:  python desafio1_aluno.py

Regras:
  * fit(X) recebe só as entradas de treino, sem rótulos. É opcional.
  * phi(X) transforma (n, d) em (n, d') com d < d' <= 64, de forma determinística.
  * Sem NaN/Inf; e rápido (10 mil pontos em menos de 2 s).
Escore: 0 = igual à identidade (baseline), 100 = igual à referência do professor,
até 125 se superar a referência. Na correção, tarefas OCULTAS da mesma família
substituem estas — não ajuste para um conjunto de dados específico.
"""
import itertools
import math
import time
import torch

DIM_MAX = 64
SEMENTES = (0, 1, 2)

# =============================================================================
# >>> SUA SUBMISSÃO — edite apenas esta classe <<<
# =============================================================================
class Submissao:
    """
    Ideia geral: como não dá pra saber de antemão qual vai ser a tarefa oculta,
    em vez de tentar "adivinhar" uma transformação específica, eu combino três
    tipos de característica que juntas cobrem os casos mais comuns de fronteira
    não-linear. Cada bloco ataca um tipo de problema diferente:

      1) Padronização + quadrados (x_i^2)  -> resolve fronteiras tipo "distância
         do centro" (ex.: círculos, esfera_10d, onde o que importa é x^2 e não x).
      2) Produtos cruzados (x_i * x_j)     -> resolve fronteiras que dependem do
         SINAL da combinação entre eixos (o XOR é basicamente só isso: o rótulo
         é dado por sinal(x0 * x1)).
      3) Random Fourier Features (cos/sin) -> pra fronteiras mais "tortas" e
         suaves (luas, espiral) que não são nem quadrado nem produto simples,
         uso RFF pra aproximar um kernel RBF, que consegue curvar bastante.

    Tudo isso precisa caber em d' <= 64 (regra do desafio). Fiz o orçamento de
    dimensão ser calculado na hora, olhando o d de entrada: se d é pequeno (2D)
    sobra bastante espaço pra RFF; se d é maior (esfera_10d, d=10) os produtos
    cruzados sozinhos quase enchem as 64 dimensões, então corto o excesso de
    pares antes de gerar RFF. Assim a solução não quebra em nenhuma dimensão
    de entrada, nem nas tarefas ocultas.
    """
    DIM_MAX = DIM_MAX

    def fit(self, X: torch.Tensor) -> None:
        """Só vê o X de treino (sem rótulo). Aqui eu: padronizo, decido quantos
        produtos cruzados cabem no orçamento, e sorteio as projeções da RFF."""
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        d = X.shape[1]

        # Blocos 1 e 2: quadrados (d dimensões) + produtos cruzados (até C(d,2))
        self.pairs = list(itertools.combinations(range(d), 2))
        base = 2 * d                      # padronizado + quadrado
        used = base + len(self.pairs)
        if used > DIM_MAX:                # d grande demais: corta os pares que sobrarem
            self.pairs = self.pairs[: max(DIM_MAX - base, 0)]
            used = base + len(self.pairs)

        # Bloco 3: Random Fourier Features multi-escala, usando o espaço que sobrou
        remaining = max(DIM_MAX - used, 0)
        self.n_freq = remaining // 2      # metade vira cos, metade vira sin
        if self.n_freq > 0:
            g = torch.Generator().manual_seed(42)
            Xs = (X - self.mu) / self.sd
            # sigma-base do kernel RBF, estimado pela mediana das distâncias entre pontos
            # (é a heurística clássica pra esse tipo de kernel, não inventei valor fixo)
            d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
            valid = d2[d2 > 0]
            sigma_base = math.sqrt(valid.median().item() / 2) if valid.numel() > 0 else 1.0
            # uso 5 larguras diferentes (fina, média, grossa) porque uma largura só não
            # dava conta da espiral: perto do centro ela é "apertada" e precisa de
            # frequência alta, mas longe do centro frequência baixa já basta
            escalas = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])
            partes_w = []
            restante = self.n_freq
            for i, e in enumerate(escalas):
                k = self.n_freq // len(escalas) + (1 if i < self.n_freq % len(escalas) else 0)
                if k <= 0:
                    continue
                sigma = max(sigma_base * e.item(), 1e-6)
                partes_w.append(torch.randn(d, k, generator=g) / sigma)
            self.W = torch.cat(partes_w, dim=1)
            self.b = torch.rand(self.n_freq, generator=g) * 2 * math.pi

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        """(n, d) -> (n, d'), com d < d' <= 64, combinando os três blocos descritos acima."""
        Xs = (X - self.mu) / self.sd
        partes = [Xs, Xs ** 2]

        if self.pairs:
            prod = torch.stack([Xs[:, i] * Xs[:, j] for i, j in self.pairs], dim=1)
            partes.append(prod)

        if self.n_freq > 0:
            proj = Xs @ self.W + self.b
            escala = math.sqrt(2.0 / self.n_freq)
            partes.append(escala * torch.cos(proj))
            partes.append(escala * torch.sin(proj))

        return torch.cat(partes, dim=1)


# =============================================================================
# Harness (não edite daqui para baixo)
# =============================================================================
def _luas(n, g):
    t = torch.rand(n // 2, generator=g) * math.pi
    X = torch.cat([torch.stack([torch.cos(t), torch.sin(t)], 1),
                   torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _circulos(n, g):
    t = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.where(torch.arange(n) < n // 2, 1.0, 0.45)
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return X + 0.08 * torch.randn(n, 2, generator=g), (torch.arange(n) >= n // 2).float()

def _xor(n, g):
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] * X[:, 1] < 0).float()
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _espiral(n, g):
    t = torch.sqrt(torch.rand(n // 2, generator=g)) * 3 * math.pi
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 10
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.cat([a, -a]) + 0.05 * torch.randn(n, 2, generator=g), y

def _esfera(n, g, d=10):
    X = torch.randn(n, d, generator=g)
    r2 = (X ** 2).sum(1)
    return X, (r2 > r2.median()).float()

TAREFAS = {"xor": lambda g: _xor(600, g), "duas_luas": lambda g: _luas(600, g),
           "circulos": lambda g: _circulos(600, g), "espiral": lambda g: _espiral(800, g),
           "esfera_10d": lambda g: _esfera(800, g)}


@torch.no_grad()
def perceptron_pocket(Z, y, epocas=50, eta=1.0, semente=0):
    """Regra de Rosenblatt (w <- w + eta*y*z nos erros) + pocket: guarda o melhor w."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1)], 1)     # viés embutido
    yb = 2 * y - 1
    w = torch.zeros(Zb.shape[1]); melhor_w, melhor_acc = w.clone(), -1.0
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        for i in torch.randperm(len(Zb), generator=g).tolist():
            if yb[i] * (Zb[i] @ w) <= 0:
                w += eta * yb[i] * Zb[i]
        acc = ((Zb @ w) * yb > 0).float().mean().item()
        if acc > melhor_acc:
            melhor_acc, melhor_w = acc, w.clone()
    return melhor_w


def acuracia_balanceada(y, yhat):
    return torch.stack([(yhat[y == c] == c).float().mean() for c in y.unique()]).mean().item()


@torch.no_grad()
def rodar(sub, gerador, semente, checar=True):
    g = torch.Generator().manual_seed(semente)
    X, y = gerador(g)
    idx = torch.randperm(len(X), generator=g); ntr = int(0.6 * len(X))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

    torch.manual_seed(semente)
    sub.fit(Xtr)                                       # nunca recebe ytr
    t0 = time.perf_counter(); Ztr = sub.phi(Xtr); dt = time.perf_counter() - t0
    Zte = sub.phi(Xte)
    if checar:                                         # regras do desafio
        d, dl = Xtr.shape[1], Ztr.shape[1]
        assert Ztr.ndim == 2 and Zte.shape[1] == dl, "phi deve devolver (n, d')"
        assert d < dl <= DIM_MAX, f"exige d < d' <= {DIM_MAX}; recebi d={d}, d'={dl}"
        assert torch.isfinite(Ztr).all() and torch.isfinite(Zte).all(), "NaN/Inf na saída de phi"
        assert torch.allclose(sub.phi(Xtr[:20]), Ztr[:20]), "phi não é determinística"
        assert dt * (10_000 / len(Xtr)) < 2.0, "phi lenta demais (limite: 10^4 pontos em 2 s)"

    w = perceptron_pocket(Ztr, ytr, semente=semente)
    yhat = (torch.cat([Zte, torch.ones(len(Zte), 1)], 1) @ w > 0).float()
    return acuracia_balanceada(yte, yhat)


class _Identidade:                       # baseline (viola d < d', mas é só o ponto zero da escala)
    def fit(self, X): pass
    def phi(self, X): return X

class _RFF:                              # referência: random Fourier features, d' = 64
    def fit(self, X):
        g = torch.Generator().manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
        sigma = math.sqrt(d2[d2 > 0].median().item() / 2)
        self.W = torch.randn(X.shape[1], DIM_MAX, generator=g) / sigma
        self.b = torch.rand(DIM_MAX, generator=g) * 2 * math.pi
    def phi(self, X):
        return math.sqrt(2 / DIM_MAX) * torch.cos(((X - self.mu) / self.sd) @ self.W + self.b)


def _mediana(cls, gerador, checar=True):
    vals = [rodar(cls(), gerador, sem, checar) for sem in SEMENTES]
    return float(torch.tensor(vals).median())


def avaliar():
    print(f"{'tarefa':<12}{'baseline':>10}{'referência':>12}{'você':>8}{'s_t':>7}")
    s = []
    for nome, gen in TAREFAS.items():
        b = _mediana(_Identidade, gen, checar=False)
        r = max(_mediana(_RFF, gen, checar=False), b + 1e-3)
        try:
            m = _mediana(Submissao, gen); erro = ""
        except AssertionError as e:
            m, erro = b, f"   <- {e}"
        st = min(max((m - b) / (r - b), 0.0), 1.25); s.append(st)
        print(f"{nome:<12}{b:>10.3f}{r:>12.3f}{m:>8.3f}{st:>7.2f}{erro}")
    S = 100 * (0.7 * sum(s) / len(s) + 0.3 * min(s))
    print(f"\nESCORE S = {S:.1f}   (0 = baseline, 100 = referência, até 125 com bônus)")
    return S


if __name__ == "__main__":
    avaliar()
