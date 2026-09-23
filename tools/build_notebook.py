"""Gera Kaggle_LSTM.ipynb a partir de config/estudo.json e grids/*.json.

Os textos de cada bloco (pergunta, descrição, o que observar, perguntas da análise) ficam no próprio grid,
então mudar hiperparâmetros é: editar o grid → rodar este script → o notebook sai consistente.
Uso: python tools/build_notebook.py [--saida Kaggle_LSTM.ipynb]
"""
import argparse
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRIDS = os.path.join(ROOT, "grids")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(text)}


def _lines(text):
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def load(name):
    with open(os.path.join(GRIDS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def fmt(v):
    if isinstance(v, dict):
        inner = ", ".join(f"`{k}={fmt_plain(x)}`" for k, x in v.items() if not k.startswith("_"))
        return f"**{v.get('_rotulo', '?')}** ({inner})"
    return f"`{fmt_plain(v)}`"


def fmt_plain(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:g}"
    return json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)


# --------------------------------------------------------------------------------------------------

def header(study, order, specs):
    rows = []
    for name in order["blocos"]:
        s = specs[name]
        if s.get("eixos"):
            grid = " × ".join(s["eixos"])
        elif s.get("ablacao"):
            grid = "desfaz, uma de cada vez, cada mudança do campeão em relação ao padrão"
        elif s.get("checagem"):
            grid = f"{', '.join(map(str, s['checagem']['posicoes']))}º de `{s['checagem']['bloco_origem']}` com a config. atual"
        else:
            grid = ", ".join(c["_nome"] for c in s.get("configs", []))
        busca = s.get("busca", {}).get("tipo", "grid")
        if s.get("eixos") and busca == "aleatoria":
            grid += f" (aleatória, {s['busca']['n']})"
        rows.append(f"| {s['titulo']} | {s['pergunta']} | {grid} |")
    d, p = study["decisao"], study["particoes"]
    dados = study["dados"]
    series = ", ".join(f"`{t}`" + (f" (`{dados['arquivos'][t]}`)" if dados.get("arquivos", {}).get(t) else "") for t in dados["tickers"])
    return f"""# {study.get('titulo', 'Previsão de séries financeiras com LSTM')} — Relatório de experimentos (busca em blocos)

**Objetivo:** descobrir o que **realmente** melhora o modelo quando cada hiperparâmetro muda, separando efeito de ruído.
Para isso o estudo mede o ruído entre seeds (Bloco 0), estima o efeito de cada eixo controlando os demais, reconstrói a
cadeia de decisões e termina com uma ablação do campeão.

**Metodologia** (ver `METODOLOGIA.md`). Em vez de escolher um valor por vez ou cruzar tudo num grid gigante, o estudo é
dividido em **blocos sequenciais**, cada um respondendo a uma pergunta. O campeão de um bloco é lido automaticamente pelo
bloco seguinte, sem nenhum valor escolhido à mão. Como cada bloco varia só um grupo coeso de hiperparâmetros, o espaço
total explorado pode ser bem maior que o de um grid puro. Blocos com muitos eixos usam **busca aleatória** (n combinações
distintas sorteadas do produto cartesiano, com seed fixa), lida pelo efeito marginal de cada eixo.

| Bloco | Pergunta | Espaço de busca |
|---|---|---|
{chr(10).join(rows)}

**Dados.** Série(s): {series}, de {dados['inicio']} a {dados['fim']} (fonte `{dados['fonte']}`, preço `{dados['coluna_preco']}`).
Com várias séries, um único modelo é treinado com as janelas de todas elas (normalização por série) e as métricas saem no
geral e **por série**. Os preços ficam congelados em `data/precos/`, com o MD5 no `parametros.json` de cada experimento.

**Protocolo de avaliação.**
- **Validação walk-forward em {p['n_folds']} folds** (`{p['tipo']}`, blocos de {p.get('tamanho_validacao_dias')} dias): o fold k treina em tudo antes do bloco de validação k e valida nele. As partições são por data e são as mesmas em todos os experimentos (comparações pareadas). Um embargo de h dias impede que alvos do treino entrem na validação.
- **Métrica de decisão:** `{d['metrica']}` ({'menor' if d['modo'] == 'min' else 'maior'} é melhor). Além dela são registradas, no geral e por ação: `mse`, `rmse` e `mae` (no log-retorno de h dias), `mape` (% no preço), `theil` (U de Theil: erro do modelo ÷ erro do passeio aleatório, < 1 = bate a referência), **`pocid`** (Prediction Of Change In Direction, %: acerto na previsão de subida/queda da cotação, com D_t = 1 se (P_t − P_t−1)(P̂_t − P̂_t−1) > 0), `da` (acurácia direcional em relação ao preço atual, %), `skill`, `r2` e `ic`. As métricas são comparáveis entre alvos (retorno ou preço).
- **Triagem:** cada configuração roda os folds {d['triagem_folds']}; as {d['n_finalistas']} melhores **completam os demais folds** (sem refazer nada) e o campeão é a maior média nos {p['n_folds']} folds, nunca a média parcial da triagem (maldição do vencedor).
- **Escolha sempre pela validação.** O teste (a partir de {p['teste_inicio']}) só é revelado na seção final, para os campeões e as referências.
- Diferenças menores que o desvio entre folds não devem ser tratadas como melhora real: use a comparação pareada.

**Registro dos resultados.** Cada experimento grava em `outputs/{{exp_name}}/`:

| Arquivo | Conteúdo |
|---|---|
| `parametros.json` | hiperparâmetros exatos, arquitetura, partições, MD5 dos dados, versões e commit |
| `historico_treino.csv` | uma linha por **fold × época**: loss e métricas de treino/validação (gerais e por ação), gap |
| `historico_treino_agregado.csv` | média e desvio entre folds, por época |
| `resultados.json` | por fold: validação na melhor época e teste do modelo restaurado; médias |
| `melhor_modelo.pth` | pesos do melhor fold; cada fold em `folds/fold_k/` (com `predicoes_val.csv` e `predicoes_test.csv`) |

Cada bloco grava em `outputs/_grids/{{bloco}}/` (`configs.csv`, `descartadas.csv`, `ranking_triagem.csv`, `ranking_final.csv`,
`campeao.json`, `params/`, `logs/`). Tabelas e figuras ficam em `outputs/_relatorio/`. Tudo é gravado a cada época **em disco
primeiro**; W&B e GitHub são espelhos opcionais.

**Tempo estimado:** cerca de 1.000 treinos de fold. Medido numa GPU de notebook (RTX 3050): um fold do LSTM padrão leva
3–13 s, e o Bloco 0 inteiro (9 configurações × 5 folds) leva 1,5 min. Estimativa para o estudo completo no Kaggle com 2× T4:
**~1–3 h**. O Bloco 3 é o mais lento, porque ativações ≠ tanh usam uma célula LSTM própria, e os blocos com mais épocas (5, 7)
também demoram mais. Se o limite de 12 h estourar, basta retomar (veja abaixo)."""


def environment_md():
    return """## Como executar

O notebook é o mesmo em qualquer ambiente: ele clona o código do GitHub (ou usa a cópia local), acha os dados onde
estiverem e pula o que não estiver disponível. **Nada além de Python + GPU é obrigatório** (W&B e GitHub são opcionais).

### No Kaggle (execução principal)

1. **Dataset:** *Datasets → New Dataset*, envie o CSV do Bitcoin (`data-bitcoin_timedata-2023_v2 - ….csv`).
2. **Notebook:** *Code → New Notebook → File → Import Notebook* e escolha `Kaggle_LSTM.ipynb`.
3. **Input:** *Add Input* → o dataset do passo 1. O notebook acha o CSV sozinho em `/kaggle/input` (pelo nome, ou o
   único CSV com colunas `date`/`close`).
4. **Secrets:** *Add-ons → Secrets* → `GITHUB_TOKEN` e `WANDB_API_KEY` (os mesmos valores do `.env`), marcados como anexados.
5. **Settings:** *Accelerator* **GPU T4 ×2**, *Internet* **On** (clone do código, W&B e envio dos resultados).
6. *Save Version → **Save & Run All (Commit)***. A execução roda em segundo plano (limite de 12 h por sessão); a saída
   fica em *Output* (`outputs/` e `outputs.zip`) e, com o token, no branch `resultados` do GitHub.

### No Colab ou Jupyter local

| Item | Colab | Jupyter local |
|---|---|---|
| GPU | *Runtime → Change runtime type → GPU* | a que o PyTorch enxergar (CPU funciona, mais devagar) |
| Código | clonado de `REPO_URL` | abra o notebook **dentro** do repositório |
| Dados | CSV em `/content`, ou `DATA_PATH` | o CSV na raiz do repositório, ou `DATA_PATH` |
| Tokens | *Colab Secrets* (mesmos nomes) | arquivo `.env` na raiz (modelo em `.env.example`) |
| Saída | `/content/outputs` | `outputs/` no repositório |

**W&B.** Uma run por configuração (grupo = bloco, `config` = hiperparâmetros, resumo = médias entre folds). Os painéis
*Parameter importance* e *Parallel coordinates* do projeto mostram o mesmo que as análises deste notebook.
O teste nunca é enviado ao W&B (só aparece na seção final).

**Resultados no GitHub.** Com `GITHUB_TOKEN`, ao fim de cada bloco os CSV/JSON/PNG de `outputs/` vão para o branch
`RESULTS_BRANCH`, na pasta `RUN_NAME/outputs/` (branch órfão, separado do código; `.pth` só com `SYNC_PESOS = True`).

**Retomada.** Todo experimento pula os folds já concluídos. Para continuar uma execução interrompida (ex.: limite de 12 h):
`RUN_NAME` igual ao da execução anterior e `RESUME_FROM = "github"`, ou `RESUME_FROM = "/kaggle/input/<output anterior>/outputs"`."""


def setup_cells():
    cfg = code('''# ===== Configuração da execução (edite aqui) =====
import os

REPO_URL = "https://github.com/diegoflyra/if702-miniproject-2.git"  # repositório com src/, grids/, config/
RESULTS_REPO_URL = REPO_URL      # onde espelhar os resultados ("" = não enviar ao GitHub)
RESULTS_BRANCH = "resultados"    # branch órfão só de resultados
RUN_NAME = ""                    # vazio = "lstm-AAAAMMDD-HHMM"; fixe um nome para retomar pelo GitHub
RESUME_FROM = ""                 # "" | "github" | pasta outputs de uma execução anterior
WORKERS_PER_GPU = 2              # experimentos simultâneos por GPU (LSTMs pequenos; o Kaggle tem 4 CPUs)
CPU_WORKERS = 1                  # sem GPU
DATA_PATH = ""                   # CSV (ou pasta) dos dados; vazio = raiz do repo → /kaggle/input → /content
SYNC_PESOS = False               # enviar também os .pth ao GitHub
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "if702-miniproject-2-lstm")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
USAR_WANDB = True                # False = não envia nada ao W&B, mesmo com a chave
USAR_GITHUB = True               # False = não espelha os resultados no GitHub, mesmo com o token
MODO_TESTE = False               # True = passada rápida (3 configs por bloco, 2 épocas) em outputs_teste/, para validar o
                                 # notebook inteiro em minutos antes da execução completa''')
    clone = code('''import os
import shutil
import subprocess
import sys
import time


def _load_dotenv(path=".env"):
    """Tokens em um .env local (GITHUB_TOKEN, WANDB_API_KEY); nunca sobrescreve variáveis já definidas."""
    if os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#") and value.strip() and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value.strip().strip("'").strip('"')


_load_dotenv()


def _secret(name):
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        pass
    return ""


def _is_repo(path):
    return all(os.path.exists(os.path.join(path, p)) for p in ("src/grid_search.py", "grids", "config/estudo.json"))


GITHUB_TOKEN = _secret("GITHUB_TOKEN")
here = os.getcwd()
if _is_repo(here):
    REPO_DIR = here
    print(f"Código: cópia local em {REPO_DIR}")
else:
    REPO_DIR = "/tmp/lstm-acoes"
    clone_url = REPO_URL.replace("https://", f"https://x-access-token:{GITHUB_TOKEN}@", 1) if GITHUB_TOKEN else REPO_URL
    print("GitHub: usando GITHUB_TOKEN" if GITHUB_TOKEN else "GitHub: clone sem token")
    if _is_repo(REPO_DIR):
        subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=False)
    else:
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        cloned = subprocess.run(["git", "clone", "-q", clone_url, REPO_DIR])
        if cloned.returncode != 0 or not _is_repo(REPO_DIR):
            raise RuntimeError("Falha ao clonar o repositório. Repositório privado exige GITHUB_TOKEN "
                               "(Kaggle Secrets, Colab Secrets ou variável de ambiente), ou abra o notebook "
                               "a partir de uma cópia local do projeto (pasta com src/, grids/ e config/).")
    print(f"Código: {REPO_DIR}")

os.chdir(REPO_DIR)
%cd {REPO_DIR}
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout.strip())''')
    pip = code('''import importlib.util

skip, pkgs = [], []
with open("requirements.txt", encoding="utf-8") as f:
    for line in f:
        pkg = line.strip()
        if not pkg or pkg.startswith("#"):
            continue
        name = pkg.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("[")[0].strip().lower()
        if name == "torch" and importlib.util.find_spec(name) is not None:
            skip.append(name)
            continue
        pkgs.append(pkg)
if skip:
    print("Já instalado, não reinstalar (preserva CUDA do ambiente):", ", ".join(skip))
if pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])''')
    env = code('''import pandas as pd

if os.path.isdir("/kaggle/working"):
    WORKDIR = "/kaggle/working"
elif os.path.isdir("/content"):
    WORKDIR = "/content"
else:
    WORKDIR = REPO_DIR
OUTPUTS = os.path.join(WORKDIR, "outputs_teste" if MODO_TESTE else "outputs")  # o teste nunca se mistura com a execução real
GRID_EXTRA = "--max_configs 3 --epochs 2" if MODO_TESTE else ""
GRID_EXTRA_B0 = "--epochs 2" if MODO_TESTE else ""  # o Bloco 0 roda completo: as 5 seeds medem o ruído
if MODO_TESTE:
    print("MODO_TESTE: 3 configurações por bloco, 2 épocas; resultados em outputs_teste/ (não servem como resultado).")
os.environ["EXP_OUTPUT_DIR"] = OUTPUTS
os.makedirs(OUTPUTS, exist_ok=True)
print(f"Resultados em {OUTPUTS}")

RUN_NAME = RUN_NAME or time.strftime("lstm-%Y%m%d-%H%M")
if MODO_TESTE and not RUN_NAME.startswith("teste-"):
    RUN_NAME = "teste-" + RUN_NAME
if not USAR_GITHUB:
    RESULTS_REPO_URL = ""
os.environ.update({"RUN_NAME": RUN_NAME, "RESULTS_REPO_URL": RESULTS_REPO_URL, "RESULTS_BRANCH": RESULTS_BRANCH,
                   "GITHUB_TOKEN": GITHUB_TOKEN, "WANDB_PROJECT": WANDB_PROJECT, "WANDB_ENTITY": WANDB_ENTITY})
if DATA_PATH:
    os.environ["DATA_PATH"] = DATA_PATH
print(f"RUN_NAME = {RUN_NAME}")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import data as data_mod
import github_sync
import report_utils as rep

if RESUME_FROM == "github":
    github_sync.pull(OUTPUTS)
elif RESUME_FROM:
    shutil.copytree(RESUME_FROM, OUTPUTS, dirs_exist_ok=True)
    print(f"Resultados anteriores copiados de {RESUME_FROM}; o que já foi concluído será pulado.")

wandb_key = _secret("WANDB_API_KEY") if USAR_WANDB else ""
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
    os.environ.pop("WANDB_MODE", None)
    print(f"W&B: chave carregada (projeto {WANDB_PROJECT}).")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print(f"W&B desativado ({'USAR_WANDB = False' if not USAR_WANDB else 'sem chave'}). Resultados continuam em {OUTPUTS}.")

if not (GITHUB_TOKEN and RESULTS_REPO_URL):
    print("GitHub: sem GITHUB_TOKEN/RESULTS_REPO_URL, os resultados ficam só em disco (e em outputs.zip).")


def backup(bloco=""):
    """Backup parcial ao fim de cada bloco: outputs.zip (sem .pth) + espelho no GitHub, se configurado."""
    import zipfile

    zpath = OUTPUTS + ".zip"
    with zipfile.ZipFile(zpath + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUTPUTS):
            for name in files:
                if not name.endswith((".pth", ".tmp")):
                    full = os.path.join(root, name)
                    z.write(full, os.path.relpath(full, WORKDIR))
    os.replace(zpath + ".tmp", zpath)
    print(f"{os.path.basename(zpath)}: {os.path.getsize(zpath) / 1e6:.1f} MB")
    github_sync.push(OUTPUTS, weights=SYNC_PESOS, message=f"{RUN_NAME}: {bloco or 'backup'}")''')
    data_md = md("""### Dados e partições

Preços congelados por ação (MD5 no manifesto) e as datas de cada fold walk-forward. O teste aparece só como período; nenhum número dele é usado até o fim.""")
    data_cell = code('''manifesto = data_mod.prepare_prices()
display(data_mod.describe_folds())''')
    checks = code('''# Verificações rápidas antes de gastar GPU (sem treino): GPUs, dados sem vazamento, modelos e todos os grids
if shutil.which("nvidia-smi"):
    !nvidia-smi -L
else:
    print("nvidia-smi não encontrado; o treino usa o device que o PyTorch enxergar (CPU se não houver GPU).")
!python tests/check_data.py
!python tests/check_models.py
!python tests/check_metrics.py
!python tests/check_augment.py
!python tests/check_grids.py''')
    return [md("## 0. Preparação do ambiente"), cfg, clone, pip, env, data_md, data_cell, checks]


def block_markdown(s, study):
    d = study["decisao"]
    parts = [f"## {s['titulo']}", "", f"**Pergunta:** {s['pergunta']}", "", s.get("descricao", "")]
    if s.get("herda_de"):
        parts += ["", f"**Base:** o melhor campeão entre {', '.join(f'`{b}`' for b in s['herda_de'])}."]
    else:
        parts += ["", "**Base:** `config/estudo.json → padrao` (o LSTM padrão)."]
    if s.get("eixos"):
        parts += ["", "| Eixo | Valores |", "|---|---|"]
        parts += [f"| `{k}` | {', '.join(fmt(v) for v in vals)} |" for k, vals in s["eixos"].items()]
        total = math.prod(len(v) for v in s["eixos"].values())
        busca = s.get("busca", {"tipo": "grid"})
        if busca.get("tipo") == "aleatoria" and busca["n"] < total:
            count = f"**{total} combinações possíveis; busca aleatória de {busca['n']}** (seed {busca.get('seed', 0)})."
        else:
            count = f"**{total} combinações.**"
    elif s.get("configs"):
        parts += ["", "| Configuração | Parâmetros |", "|---|---|"]
        parts += [f"| `{c['_nome']}` | {', '.join(f'`{k}={fmt_plain(v)}`' for k, v in c.items() if k != '_nome')} |" for c in s["configs"]]
        count = f"**{len(s['configs'])} configurações.**"
    else:
        count = ""
    extra = []
    if s.get("fixos"):
        extra.append("Fixos no bloco: " + ", ".join(f"`{k}={fmt_plain(v)}`" for k, v in s["fixos"].items()) + ".")
    if s.get("incluir_base"):
        extra.append("A base é re-treinada sem mudanças (referência pareada).")
    if s.get("triagem", True):
        n = s.get("n_finalistas", d["n_finalistas"])
        extra.append(f"Triagem nos folds {d['triagem_folds']} de {study['particoes']['n_folds']}; as {n} melhores completam os {study['particoes']['n_folds']} folds.")
    else:
        extra.append(f"Todas as configurações rodam os {study['particoes']['n_folds']} folds.")
    if s.get("restricoes", {}).get("max_parametros"):
        extra.append(f"Limite: {int(s['restricoes']['max_parametros']):,} parâmetros.")
    parts += ["", " ".join(x for x in [count] + extra if x)]
    if s.get("observar"):
        parts += ["", f"**O que observar:** {s['observar']}"]
    return "\n".join(parts)


def block_cells(name, s, study):
    cells = [md(block_markdown(s, study))]
    if s.get("herda_de"):
        cells.append(code("\n".join(f'rep.show_champion("{b}")' for b in s["herda_de"])))
    extra = "$GRID_EXTRA_B0" if s.get("papel") == "referencia" else "$GRID_EXTRA"
    cells.append(code(f"!python src/grid_search.py grids/{name}.json --workers_per_gpu $WORKERS_PER_GPU --cpu_workers $CPU_WORKERS {extra}"))
    if s.get("papel") == "ablacao":
        cells.append(md(f"#### Resultado da ablação — `{name}`\n\nΔ = (campeão sem a mudança) − (campeão), fold a fold. "
                        "Verde: a mudança ajuda além do ruído entre seeds; cinza: dispensável; vermelho: atrapalhava."))
        cells.append(code(f'rep.discarded_configs("{name}")  # mudanças cuja reversão não altera nada'))
        cells.append(code(f'rep.ablation_table("{name}")'))
        short = s["titulo"].split("—")[0].strip()
        cells.append(md(f"### 📝 Análise — {short}\n\n" + "\n".join(f"- **{q}** _…_" for q in s.get("analise", []))))
        cells.append(code(f'backup("{name}")'))
        return cells
    random = s.get("busca", {}).get("tipo") == "aleatoria"
    multi_axis = sum(len(v) > 1 for v in s.get("eixos", {}).values())
    if s.get("triagem", True):
        cells.append(md(f"#### Resultados da triagem — `{name}`\n\nTodas as configurações, ordenadas pela **validação** (média ± desvio nos folds de triagem). O teste não aparece aqui de propósito."))
        cells.append(code(f'rep.grid_ranking("{name}", "triagem")'))
    cells.append(code(f'rep.discarded_configs("{name}")  # combinações descartadas antes do treino e o motivo'))
    if multi_axis:
        cells.append(md(f"#### O que cada hiperparâmetro muda — `{name}`\n\n"
                        "Efeito de cada valor em relação ao valor da base, **controlando os outros eixos** (modelo aditivo "
                        "sobre todas as configurações), e quanto da variação da métrica cada eixo explica. "
                        "Verde = melhora além de 2× o ruído entre seeds; cinza = indistinguível de ruído."))
        cells.append(code(f'importancia, efeitos = rep.axis_importance("{name}")\ndisplay(importancia)\ndisplay(efeitos)'))
    for hm in s.get("heatmaps", []):
        args = ", ".join(f'{k}={json.dumps(v)}' for k, v in hm.items())
        cells.append(code(f'rep.heatmap("{name}", {args})'))
    if random or multi_axis > 2:
        cells.append(code(f'rep.param_effect("{name}")'))
    if s.get("eixos") or len(s.get("configs", [])) > 1:
        cells.append(code(f'rep.plot_grid_bars("{name}")'))
    cells.append(md(f"#### Confirmação das finalistas e campeão — `{name}`\n\nAs finalistas completaram todos os folds; o campeão é a melhor média da métrica de decisão nos K folds."))
    cells.append(code(f'rep.grid_ranking("{name}", "final")'))
    cells.append(code(f'rep.show_champion("{name}")\nrep.plot_finalists("{name}")'))
    if s.get("ruido"):
        cells.append(md("#### Ruído entre seeds\n\nA régua do estudo: o menor efeito que se distingue de sorte na inicialização."))
        cells.append(code('ruido = rep.noise_floor()\nrep.noise_floor("val/pocid")'))
    if s.get("pareado") or s.get("incluir_base"):
        cells.append(md("#### Comparação pareada por fold (validação)\n\nReferência: a base do bloco re-treinada aqui (mesmos folds e seed); `veredito` compara o Δ com o ruído entre seeds."))
        cells.append(code(f'rep.paired_comparison(rep.base_config_name("{name}"), "{name}__*")'))
    if s.get("checagem"):
        prev = s["herda_de"][0]
        cells.append(code(f'pd.concat([rep.grid_ranking("{prev}", "final").head(1),\n           rep.grid_ranking("{name}", "final")], ignore_index=True)'))
    short = s["titulo"].split("—")[0].strip()
    cells.append(md(f"### 📝 Análise — {short}\n\n" + "\n".join(f"- **{q}** _…_" for q in s.get("analise", []))))
    cells.append(code(f'backup("{name}")'))
    return cells


def chain_cells(order):
    main = order["campeao_principal"]
    return [md(f"""## O que realmente melhorou o modelo (validação)

Antes de revelar o teste: a **cadeia de decisões** reconstrói, pela herança entre blocos, o caminho do LSTM padrão até o
campeão principal (`{main}`). Para cada passo: o que mudou, o Δ pareado por fold, em quantos folds a mudança venceu e se o
ganho passa do ruído entre seeds. Junto com a ablação (que desfaz cada mudança do campeão), mostra quais decisões de
hiperparâmetros realmente importam e quais só entraram por ruído."""),
            code(f'cadeia = rep.decision_chain("{main}")\ncadeia'),
            code('backup("cadeia")')]


def final_cells(order, specs):
    blocks = order["blocos"]
    main, bonus = order["campeao_principal"], order.get("bonus", [])
    ref = next((b for b in blocks if specs[b].get("papel") == "referencia"), None)
    cells = [md(f"""## Resultado final — teste revelado

Até aqui todas as escolhas foram feitas pela validação walk-forward. Agora o período de teste é usado **uma única vez**,
para os campeões de cada bloco e para as referências. A tabela mostra média ± desvio do teste entre os K modelos (um por fold).
O campeão da sequência principal é o de `{main}`; {', '.join(f'`{b}`' for b in bonus) or 'nenhum bônus'} mostra(m) eixos ortogonais sobre ele.
A concordância entre validação e teste é a evidência de que a metodologia não vazou informação; uma discrepância grande
(ex.: mudança de regime no período de teste) é um achado a investigar."""),
             code(f"final = rep.final_report({json.dumps([b for b in blocks if specs[b].get('papel') != 'ablacao'])})\nfinal")]
    cells.append(md("Desempenho por série no teste: referências × campeão principal × bônus."))
    cells.append(code(f'''campeao_principal = rep.champion_name("{main}")
comparar = {f'list(rep.results.configs("{ref}")["exp_name"])' if ref else '[]'} + [campeao_principal] + [rep.champion_name(b) for b in {json.dumps(bonus)}]
rep.plot_per_ticker(comparar, metric="theil")
rep.plot_per_ticker(comparar, metric="pocid")'''))
    cells.append(md("#### Métricas por série do campeão principal\n\nMédia ± desvio entre os K modelos (um por fold), no teste. Figura e CSV em `outputs/_relatorio/metricas_por_serie_test_<experimento>.*`."))
    cells.append(code(f'rep.champion_ticker_metrics("{main}")'))
    cells.append(md("#### Previsão × real (ensemble dos K modelos)"))
    cells.append(code('''for t in data_mod.common.load_study()["dados"]["tickers"]:
    rep.plot_predictions(campeao_principal, ticker=t)'''))
    cells.append(md("#### Valor econômico (ilustrativo)\n\nPosição = sinal da previsão, sem custos de transação. **Não** é critério de decisão do estudo; só mostra se o sinal direcional tem valor além das métricas."))
    # referências treináveis/direcionais (passeio aleatório e média não têm sinal útil para a estratégia)
    ref_rw = (f'[e for e in rep.results.configs("{ref}")["exp_name"] if not e.endswith(("passeio_aleatorio", "media_historica"))]'
              if ref else "[]")
    cells.append(code(f"rep.plot_strategy({ref_rw} + [campeao_principal])"))
    cells.append(md("#### Resumo no W&B\n\nRun `estudo__resumo`: cadeia de decisões, ruído entre seeds e o relatório final (o teste só é enviado agora, depois de revelado)."))
    report_blocks = [b for b in blocks if specs[b].get("papel") != "ablacao"]
    cells.append(code(f'import wandb_report\nwandb_report.log_study("{main}", {json.dumps(report_blocks)})'))
    cells.append(code('backup("final")'))
    cells.append(md("""## 📝 Conclusões

- **O LSTM bateu o passeio aleatório no teste (theil < 1, skill > 0)? Por quanto, e de forma consistente entre folds e ações?** _…_
- **POCID: o modelo acerta a direção da cotação acima de 50%? Isso se converte em valor na estratégia ilustrativa?** _…_
- **Entrada (Bloco 1): janela, features e alvo, e por quê:** _…_
- **Camadas ocultas, nós e camada densa (Bloco 2): a regra prática (1–2 camadas, 5–10 unidades densas) se confirmou?** _…_
- **Ativações e inicialização (Bloco 3):** _…_
- **Otimização (Bloco 4): lr, decaimento (0,97?), momentum e batch:** _…_
- **A checagem de interação confirmou a escolha em blocos?** _…_
- **Dropout e decaimento dos pesos (Bloco 5): 20% foi o melhor compromisso?** _…_
- **Complementares: mais nós com regularização compensaram? O resultado estava limitado pelas épocas?** _…_
- **Bônus — LSTM × GRU:** _…_
- **Validação × teste concordaram? O período de teste (2022–2023) teve mudança de regime?** _…_"""))
    return cells


def build(out):
    with open(os.path.join(ROOT, "config", "estudo.json"), encoding="utf-8") as f:
        study = json.load(f)
    order = load("_ordem")
    specs = {b: load(b) for b in order["blocos"]}
    cells = [md(header(study, order, specs)), md(environment_md())] + setup_cells()
    for name in order["blocos"]:
        cells += block_cells(name, specs[name], study)
    cells += chain_cells(order)
    cells += final_cells(order, specs)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{out}: {len(cells)} células")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", default=os.path.join(ROOT, "Kaggle_LSTM.ipynb"))
    build(parser.parse_args().saida)
