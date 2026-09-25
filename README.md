# Previsão do preço do Bitcoin com LSTM — estudo empírico em blocos

Aplica a metodologia de `METODOLOGIA.md` (a mesma do estudo com CNN no CIFAR-10) a LSTMs para a série diária do Bitcoin
(`data-bitcoin_timedata-2023_v2 - ….csv`, 2017-08-17 → 2023-08-01; teste = último ano). Outras séries (ex.: ações via
yfinance) entram só mudando `config/estudo.json → dados`.
O entregável é o notebook `Kaggle_LSTM.ipynb`: roda de ponta a ponta no Kaggle (GPU, W&B, resultados espelhados no GitHub)
e, sem nenhum desses, no Colab ou num Jupyter local.

Repositório: https://github.com/diegoflyra/if702-miniproject-2 (código na raiz; resultados no branch `resultados`).

## Hiperparâmetros e blocos

| Bloco | Hiperparâmetros (item da lista) | Busca |
|---|---|---|
| 0 — Referência | passeio aleatório, média, último retorno, regressão linear, LSTM padrão (5 seeds, para medir o ruído) | 9 configs |
| 1 — Entrada | janela, features, alvo | grid 48 |
| 1b — Contexto externo | calendário, derivativos (funding), on-chain, sentimento (medo e ganância), todos | grid 6 |
| 2 — Capacidade | camadas ocultas e nós (1), unidades densas (2), resumo da sequência (último estado, média, atenção) | aleatória 60 de 441 |
| 3 — Ativação e inicialização | ativação da célula LSTM e das densas (6), inicialização dos pesos (4) | aleatória 40 de 96 |
| 4 — Otimização | otimizador, taxa de aprendizagem (7), decaimento da lr (5), momentum, batch | aleatória 60 de 2.100 |
| Checagem | 2º e 3º do Bloco 2 com a otimização campeã | 2 configs |
| 5 — Regularização | dropout (3): após a LSTM, entre LSTMs, na entrada; decaimento dos pesos L2 (5) | aleatória 50 de 540 |
| Complementar A | camadas × nós sob regularização | grid 12 |
| Complementar B | épocas: paciência do early stopping com até 500 épocas | grid 4 |
| Bônus | LSTM × GRU | 2 configs |
| Bônus — Augmentation | jittering, scaling, magnitude warping, time warping, permutation, window slicing, jitter+scaling × intensidade (fraca/forte) | grid 15 |
| Ablação (B10) | desfaz cada mudança do campeão principal (volta ao padrão), uma por vez | automática |

O `padrao` (base do Bloco 1) segue os pontos de partida da lista: 1 camada oculta de 50 nós, densa de 10 unidades com ReLU,
tanh na célula, dropout de 20%, Adam com lr 0,001 e decaimento de 0,97 por época. A "taxa de decaimento" é aplicada à taxa
de aprendizagem; o decaimento dos pesos é o `weight_decay` (L2), testado junto com o dropout.

**Métricas** (validação e teste, gerais e por série): `mse`, `rmse`, `mae` (log-retorno), `mape` (% no preço), `theil`
(erro ÷ erro do passeio aleatório), **`pocid`** (% de acerto na direção: D_t = 1 se (P_t − P_t−1)(P̂_t − P̂_t−1) > 0),
`da`, `skill`, `r2`, `ic`; no preço: `arv`, `smape`, `mase` e `rmse_preco` (US$). A métrica de decisão fica em `config/estudo.json → decisao.metrica`.

## O notebook em fases

`Kaggle_LSTM.ipynb` conta o estudo na ordem das descobertas; cada fase é uma tentativa de melhorar o modelo
(roteiro e textos em `grids/_fases.json`):

1. **Fase 1 — Hiperparâmetros da lista** (série 2017–2023, `config/estudo.json`, resultados em `outputs/`): blocos B0–B10.
   Ganhos na validação que não se sustentam no teste.
2. **Fase 2 — Revisão das divergências** (sem treino): quantas configurações divergiram e se isso mudou alguma escolha
   (não mudou).
3. **Fase 3 — Mais dados** (BTC-USD 2014–2026, `config/estudo_btc_longo.json`, `outputs_btc_longo/`): referência com
   5 seeds e transplante do campeão da fase 1. O LSTM padrão passa a bater o passeio aleatório; o campeão da fase 1 não.
4. **Fase 4 — Contexto externo e atenção** (série longa): funding, on-chain, sentimento, calendário e atenção.
5. **Fase 5 — O espaço de busca completo** (série longa): outras famílias de modelos (SVR, Random Forest, XGBoost,
   CNN 1D, receita do paper de Wu et al., 2025) → entrada (todas as features, incluindo mercado: ETH, ouro, S&P 500,
   VIX, juro, dólar, Nvidia, Tesla) → normalização (scaler, alvo pela volatilidade, por janela, início do treino) →
   arquitetura (LSTM/GRU, bidirecional, pilhas decrescentes, atenção, LayerNorm, residual, CNN-LSTM) → ativações e
   inicialização → função de erro (MSE, MAE, Huber, log-cosh, direcional) → otimização (com agendas da taxa) →
   checagem → regularização (dropout, entre camadas, na entrada, recorrente; weight decay; corte de gradiente) →
   épocas → augmentation → ablação → fusão de modelos (média, desempenho, diversidade cognitiva; escore e rank).

6. **Fase 6 — Lacunas da fase 5** (série longa): ativações e regularização refeitas (a célula LSTM própria passou a
   suportar bidirecional, então ativações ≠ tanh e dropout recorrente entram de verdade), horizonte de 1, 5 e 20 dias e
   bônus TimeGAN (treinado dentro de cada fold, com diagnósticos: fatos estilizados, score discriminativo, TSTR).
7. **Fase 7 — Várias criptomoedas** (`config/estudo_multicripto.json`, `outputs_multicripto/`): o melhor LSTM da fase 6
   treinado com as janelas de 1, 2, 5, 10 ou 15 moedas, avaliado só no BTC; bônus TimeGAN sobre o conjunto.
8. **Fase 8 — Dados por hora** (`config/estudo_btc_horario.json`, `outputs_btc_horario/`): BTC/USDT de hora em hora
   (Binance, desde 2017), alvo 24 h à frente, avaliado só às 00:00 UTC (os mesmos pontos da série diária).

`FASES` na célula de configuração escolhe quais fases treinam; as outras só são lidas, então dá para reaproveitar uma
fase já rodada (ex.: a fase 1 do Kaggle) colocando a pasta de resultados dela no lugar ou usando `RESUME_FROM`.

## Dados

- Série curta: `data-bitcoin_timedata-2023_v2 - ….csv` (raiz). Série longa: `data/btc-usd_yahoo_2014-09-17_2026-09-23.csv`.
- Várias criptomoedas: `data/cripto/<MOEDA>.csv` (15 moedas, Yahoo). Por hora: `data/btc-usdt_binance_1h_2017-08-17_2026-09-23.csv`.
- `data/externos/`: funding (BitMEX, desde 2016-05), on-chain (blockchain.com, desde 2009) e medo e ganância
  (alternative.me, desde 2018-02), congelados. Para atualizar: `python src/external.py --baixar`.

**Features** (`src/data.py`, todas calculadas só com informação até o dia t):
retorno, amplitude máxima/mínima, corpo do candle, variação e z-score do volume, volatilidade de 20 dias, distância das
médias de 10 e 50 dias, RSI de 14 dias; seno/cosseno do dia da semana; funding (nível, z-score de 30 dias, disponível);
on-chain (variação de transações e de endereços ativos, z-score do volume transacionado, variação de 7 dias do hash
rate); medo e ganância (nível, variação, disponível). Antes do início de uma fonte, o valor é 0 e `*_disp` vale 0.
Open interest e long/short ratio não entram (APIs gratuitas só têm 30 dias); netflows e baleias só existem pagos.

## Tokens

Copie `.env.example` para `.env` (o git ignora) e preencha `GITHUB_TOKEN` e `WANDB_API_KEY`. No Kaggle, use *Add-ons →
Secrets* com os mesmos nomes. Sem token, tudo roda e fica só em disco.

## Estrutura

```
config/estudo.json        séries, datas, partições walk-forward, métrica de decisão e valores padrão de TODOS os hiperparâmetros
grids/_fases.json         roteiro do notebook: fases, estudo de cada uma, blocos, motivação e descobertas
grids/lstm_b*.json        um arquivo por bloco: eixos, tipo de busca, herança, textos do notebook
src/data.py               preços (cache → Kaggle Input → yfinance), features, folds por data, janelas sem vazamento
src/external.py           séries externas congeladas em data/externos/: funding (BitMEX), on-chain (blockchain.com), medo e ganância
src/models.py             LSTM configurável (ativação da célula, inicialização, densas), GRU, regressão linear e referências ingênuas
src/metrics.py            mse, rmse, mae, mape, theil, POCID, acurácia direcional, skill, r2, IC — gerais e por série
src/timegan.py            TimeGAN (dados sintéticos por fold) e diagnósticos do gerador
src/augment.py            data augmentation de séries (só no treino): jitter, scaling, magwarp, timewarp, permutation, slicing
src/train.py              treina UMA configuração nos folds pedidos; grava tudo em disco a cada época; retomável
src/grid_search.py        executa UM bloco: expande (grid ou aleatória), herda o campeão, paraleliza nas GPUs, triagem → confirmação
src/results.py            leitura dos resultados em disco (rankings, resumos por fold)
src/report_utils.py       tabelas e figuras do notebook (rankings, heatmaps, efeito marginal, pareado, relatório final)
src/wandb_report.py       runs de resumo no W&B: hiperparâmetro × métrica, efeitos, ablação, cadeia de decisões
src/github_sync.py        espelho opcional de outputs/ num branch de resultados no GitHub (push/pull para retomar)
tests/check_*.py          verificações sem treino: vazamento nas partições, modelos (célula própria = nn.LSTM), métricas, augmentation, grids
tools/build_notebook.py   gera Kaggle_LSTM.ipynb a partir de config/ e grids/
```

## Fluxo para mudar o estudo

1. Séries e datas: `config/estudo.json → dados` (`fonte`: `csv`, `yfinance` ou `synthetic` para testar sem internet).
2. Hiperparâmetros padrão (a base do Bloco 1): `config/estudo.json → padrao`.
3. Blocos: edite/crie `grids/*.json` e a ordem das fases em `grids/_fases.json`.
4. `python tools/build_notebook.py` → o notebook sai consistente com os grids (tabelas, contagens, heatmaps, análises).
5. `python tests/check_grids.py` → confere quantas configurações cada bloco vai treinar.

### Especificação de um bloco

| Campo | Significado |
|---|---|
| `herda_de` | blocos cujo campeão é a base (o melhor entre eles); `[]` = `padrao` |
| `fixos` | sobrescreve a base neste bloco (ex.: mais épocas) |
| `eixos` | `{param: [valores]}`; um valor pode ser um dict que fixa vários params juntos (`"_rotulo"` dá o nome) |
| `busca` | `{"tipo": "grid"}` ou `{"tipo": "aleatoria", "n": 40, "seed": 0}` (n combinações distintas do produto) |
| `configs` | configurações explícitas (`"_nome"` + params), ex.: referências do Bloco 0 |
| `incluir_base` | re-treina a base sem mudanças (referência pareada e reprodutibilidade) |
| `checagem` | `{"bloco_origem": b, "posicoes": [2, 3]}`: checagem de interação |
| `triagem` / `n_finalistas` | triagem em parte dos folds e confirmação das finalistas nos K folds |
| `restricoes` | `{"max_parametros": N, "expr": ["hidden_size * num_layers <= 512"]}` |
| `titulo`, `pergunta`, `descricao`, `observar`, `analise`, `heatmaps`, `pareado`, `papel` | textos e células do notebook |

Configurações equivalentes (ex.: `rnn_dropout` com 1 camada, `momentum` com Adam) são detectadas e descartadas antes do treino.

## Execução local rápida

```bash
pip install -r requirements.txt
python src/data.py --preparar                                   # congela a série em data/precos/ (com MD5)
python tests/check_data.py && python tests/check_models.py && python tests/check_metrics.py && python tests/check_augment.py && python tests/check_features.py && python tests/check_grids.py
python src/grid_search.py grids/lstm_b0_referencia.json         # um bloco
```

Os resultados vão para `outputs/` (ou `EXP_OUTPUT_DIR`). O CSV de origem é a fonte da verdade: `data/precos/` é regerado
dele a cada execução, e o MD5 da origem vai para o `parametros.json` de cada experimento. O CSV é procurado em `DATA_PATH`,
na raiz do repositório, em `/kaggle/input` e em `/content`.

## Pipeline no Kaggle

1. Envie o CSV como Dataset do Kaggle; importe `Kaggle_LSTM.ipynb` (*File → Import Notebook*).
2. *Add Input* → o dataset; *Add-ons → Secrets* → `GITHUB_TOKEN` e `WANDB_API_KEY`; *Settings* → GPU T4 ×2 e Internet On.
3. *Save Version → Save & Run All*. O código vem do GitHub (commit mais recente do branch principal); os resultados vão
   para *Output* e para o branch `resultados`.
4. Se a sessão cair ou estourar 12 h: mesmo `RUN_NAME`, `RESUME_FROM = "github"`, e rode de novo; o que terminou é pulado.

## Weights & Biases (opcional)

- **Uma run por configuração** (grupo = bloco): hiperparâmetros e nº de parâmetros no `config`; curvas por época de cada
  fold (`fold{k}/train/*`, `fold{k}/val/*`, `fold{k}/gap/*`, `lr_atual`); médias entre folds no resumo (`val/*_mean`).
  O teste nunca vai para essas runs.
- **Uma run de resumo por bloco** (`{bloco}__resumo`): tabela `ranking`; gráficos **hiperparâmetro × métrica**
  (`<eixo>/<métrica>`: média por valor; `<eixo>/<métrica>_dispersao`: todas as configurações, em eixos numéricos) para
  rmse, mse, mae, mape, theil, pocid e da; tabelas e gráfico de efeitos e importância; no bloco de ablação, a ablação.
- **`estudo__resumo`** (última célula): cadeia de decisões, ruído entre seeds e o relatório final com o teste.
- Nos painéis nativos do projeto (*Add panel → Parameter importance / Parallel coordinates*), escolha `val/rmse_mean`
  ou `val/pocid_mean` como métrica: os dados já estão lá.

## Como o estudo responde "o que realmente melhora"

- **Ruído entre seeds** (Bloco 0): o LSTM padrão com 5 seeds define o menor efeito detectável (2× o desvio da diferença
  pareada entre seeds).
- **Importância dos eixos** (cada bloco): modelo aditivo sobre todas as configurações; o efeito de cada valor em relação à
  base, com os outros eixos controlados, e o veredito frente ao ruído.
- **Comparação pareada por fold**: Δ fold a fold contra a base re-treinada no mesmo bloco, com vitórias em K folds.
- **Cadeia de decisões**: o caminho do LSTM padrão ao campeão, passo a passo, com o Δ de cada decisão.
- **Ablação** (Bloco 10): desfaz cada mudança do campeão, uma de cada vez; o que piora ao desfazer é o que realmente ajuda.
