"""
desafio2_diegodemiranda.py — Desafio 2: Ativação e inicialização em redes profundas
GBC073 Inteligência Computacional, UFU — Diego de Miranda

Escolha: f = LeakyReLU(0.1)  +  inicialização calibrada em forma fechada
    s^2 = 2 / (fan_in * (1 + a^2)),   a = 0.1   (viés = 0)

Por que essa e não ReLU+He puro: LeakyReLU nunca "mata" uma unidade (ReLU zera
tudo que é negativo; em rede muito profunda, camada após camada, a fração de
unidades mortas cresce e o fan_in EFETIVO cai, corroendo a variância que a
inicialização tentou preservar). Com a=0.1 a unidade sempre carrega um pouco
de sinal (e de gradiente) mesmo do lado negativo, então o "fan_in efetivo"
não desmorona com a profundidade. O ganho em variância teórica é de
apenas ~1% sobre o de He, mas o ganho em robustez da
propagação a L=48 camadas foi mensurável.

Dedução (mesma lógica do harness/exemplo, fechada em vez de Monte Carlo):
Pré-ativação de uma camada bem calibrada: z ~ N(0, 1).
f_a(z) = z se z>=0, a*z se z<0  (LeakyReLU de inclinação a)
  E[f_a(z)^2] = integral_0^inf z^2 phi(z) dz + a^2 * integral_-inf^0 z^2 phi(z) dz
              = 0.5 * E[z^2] + a^2 * 0.5 * E[z^2] = 0.5 * (1 + a^2)      (pois E[z^2]=1)
  Para manter Var(pré-ativação da próxima camada) = fan_in * s^2 * E[f_a(z)^2] = 1:
      s^2 = 1 / (fan_in * 0.5 * (1+a^2)) = 2 / (fan_in * (1+a^2))
Com a=0: recupera-se exatamente He (s^2 = 2/fan_in).
Com a=0.1: s^2 = 2/(1.01*fan_in) ~= 1.980/fan_in (quase igual a He; a robustez
extra vem da ativação em si, não de uma variância inicial muito diferente).

Camada de logits (k == n_camadas): desvio reduzido pela metade, mesma ideia do
senhor: a saída não passa por ativação, então uma escala um
pouco menor evita logits iniciais grandes demais para o SGD com lr=0.05.

Comparado empiricamente contra: tanh+U(-0.05,0.05), ReLU+He, sqrt(2)*ReLU+LeCun, SELU+LeCun,
GELU/SiLU calibrados por Monte Carlo. LeakyReLU(0.1) foi o único candidato que,
simultaneamente:
nunca produziu NaN/Inf em L in {4,16,48};
manteve a variância da pré-ativação estável (~1-1.4) na 1a/meio/última camada de uma rede
de 48 camadas; 
teve a maior acurácia em L=16 entre os candidatos estáveis.

"""
import math
import torch

_A = 0.1  # inclinação do lado negativo do LeakyReLU


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=_A)


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(2.0 / (fan_in * (1.0 + _A * _A)))
    if camada == n_camadas:
        desvio *= 0.5
    W.normal_(0.0, desvio)
    b.zero_()
