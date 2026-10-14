# Desafio 3 — FutStats
Predição de resultados do Brasileirão (3 classes) com MLP. Otimizadores: SGD, SGD+Nesterov, Adam, AdamW, Muon (próprio, Newton–Schulz).
## Reproduzir
pip install -U torch pandas scikit-learn matplotlib
python rodar.py 1 && python rodar.py 2 && python rodar.py 3 && python rodar.py 4
python analisar.py && python avaliar_teste.py && python painel.py
Uma linha do CSV: python reproduz.py <run_id>
(features: python prep_dados.py caminho_do_csv_bruto.csv)
## Orçamento fixo
2000 passos, lote 128, wd 0,01, clip 5,0, k=25, avaliação a cada 50 passos, MLP 256-256-128, CPU.
## Dados e split
Brasileirão Série A 2003–2023. Treino ≤2017, validação 2018–2020, teste ≥2021.
## Métrica de seleção
Menor perda de validação ao longo do treino (val_loss_best).
## Baseline
Declarar aqui a receita (ver BASELINE em rodar.py).
## Versões
torch ___ | python ___ | máquina ___
## Limitações
Validação pequena; temporada 2020 esticada até 2021; ruído alto do alvo.
