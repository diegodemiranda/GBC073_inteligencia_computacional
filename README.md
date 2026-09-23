# GBC073 - Inteligência Computacional

Este repositório contém os projetos e trabalhos práticos desenvolvidos durante a disciplina de Inteligência Computacional.

## Sobre a Disciplina

A disciplina aborda conceitos fundamentais e técnicas avançadas de inteligência computacional, introduzindo conceitos básicos sobre os três principais paradigmas da inteligência
computacional: Redes Neurais, Computação Evolutiva e Sistemas Nebulosos (Fuzzy).
Exemplificar a modelagem e aplicação desses paradigmas em problemas reais.

## Estrutura do Repositório

Cada pasta neste repositório corresponde a um projeto ou desafio específico, contendo o código-fonte, conjuntos de dados (quando aplicável) e a documentação necessária para execução.

## Projeto Final

Detecção Automática de Impactos entre Jogadores de Futebol Americano utilizando Redes Neurais Profundas e Análise Temporal de Vídeo.

#### dataset:

O dataset do Kaggle possui vídeos de jogadas, sendo disponibilizadas duas perspectivas para cada jogada. As anotações incluem a posição dos capacetes, indicador de impacto, tipo de impacto, confiança e visibilidade. Também existem dados de tracking dos jogadores contendo posição, velocidade, aceleração, orientação e direção.

#### Objetivo:
Avaliar se a utilização de informações temporais de vídeo influencia o desempenho de modelos de Deep Learning na detecção automática de impactos em jogadores de futebol americano em comparação com modelos baseados exclusivamente em frames individuais.

[NFL 1st and Future - Impact Detection](https://www.kaggle.com/c/nfl-impact-detection/data?utm_source=chatgpt.com)

Os tipos de impacto incluem, por exemplo:
- helmet;
- shoulder;
- body;
- ground.

### Estrutura

                    VÍDEO
                      │
                      ▼
               Extração de frames
                      │
                      ▼
                YOLO Detector
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     Capacete A              Capacete B
          │                       │
          └───────────┬───────────┘
                      ▼
               Rastreamento
                      │
                      ▼
             Janela temporal
             t-4 ... t ... t+4
                      │
                      ▼
                3D CNN / LSTM
                      │
                      ▼
             ┌────────────────┐
             │  Classificação │
             └────────────────┘
                      │
             ┌────────┴────────┐
             ▼                 ▼
          IMPACTO          NÃO IMPACTO
             │
             ▼
       Tipo de impacto
       ├── Helmet
       ├── Shoulder
       ├── Body
       └── Ground
---

## Como Executar

Instruções específicas sobre como configurar o ambiente e executar cada projeto podem ser encontradas em seus respectivos diretórios.

*Repositório mantido por Diego de Miranda.*
