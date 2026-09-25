"""Gera Kaggle_LSTM.ipynb a partir de grids/_fases.json, config/estudo*.json e grids/*.json.

O notebook conta o estudo na ordem das descobertas: cada fase é uma tentativa de melhorar o modelo (motivação →
blocos → teste → descoberta), e a fase seguinte parte do que a anterior revelou. Os textos de cada bloco ficam no
próprio grid, e os de cada fase em grids/_fases.json: editar esses arquivos → rodar este script → notebook consistente.
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

def load_study(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def search_space(s):
    if s.get("eixos"):
        grid = " × ".join(s["eixos"])
        if s.get("busca", {}).get("tipo") == "aleatoria":
            grid += f" (aleatória, {s['busca']['n']})"
        return grid
    if s.get("ablacao"):
        return "desfaz cada mudança do campeão, uma por vez"
    if s.get("checagem"):
        return f"{', '.join(map(str, s['checagem']['posicoes']))}º de `{s['checagem']['bloco_origem']}` com a config. atual"
    if s.get("configs_de"):
        return ", ".join(c["_nome"] for c in s["configs_de"])
    return ", ".join(c["_nome"] for c in s.get("configs", []))


def header(fases, specs):
    base = load_study(fases[0]["estudo"])
    d, p = base["decisao"], base["particoes"]
    roteiro = []
    for ph in fases:
        blocos = ", ".join(f"`{b}`" for b in ph["blocos"]) or "sem treino (reanálise)"
        roteiro.append(f"- **{ph['titulo']}** — {blocos}\n  - *Descoberta:* {ph['descoberta'][0]}")
    return f"""# Previsão do preço do Bitcoin com LSTM — o estudo em fases

**Objetivo:** descobrir o que **realmente** melhora a previsão quando mexemos no modelo, separando efeito de ruído.
O notebook segue a ordem das descobertas: **cada fase é uma tentativa de melhorar o modelo**, motivada pelo que a fase
anterior revelou. Dentro de cada fase, a metodologia é a mesma (ver `METODOLOGIA.md`): blocos sequenciais em que cada um
herda automaticamente o campeão do anterior, ruído entre seeds como régua, e o teste revelado só no fim da fase.

## Roteiro

{chr(10).join(roteiro)}

As descobertas acima são das execuções de referência (fase 1 no Kaggle com 2× T4; fases 3 e 4 numa GPU local). Ao rodar,
o notebook recalcula tudo, e as células de análise de cada fase mostram os números desta execução.

## Protocolo (vale para todas as fases)

- **Validação walk-forward em {p['n_folds']} folds:** o fold k treina em tudo antes do bloco de validação k e valida nele. As partições são por data e são as mesmas em todos os experimentos da fase (comparações pareadas); um embargo de h dias impede que alvos do treino entrem na validação.
- **Métrica de decisão:** `{d['metrica']}` ({'menor' if d['modo'] == 'min' else 'maior'} é melhor). Também são registradas `mse`, `mae`, `mape` (% no preço), `theil` (erro ÷ erro do passeio aleatório; < 1 = bate a referência), **`pocid`** (% de acerto na direção: D_t = 1 se (P_t − P_t−1)(P̂_t − P̂_t−1) > 0), `da` (acerto de direção em relação ao preço atual), `skill`, `r2` e `ic`.
- **Triagem e confirmação:** cada configuração roda os folds {d['triagem_folds']}; as {d['n_finalistas']} melhores completam os {p['n_folds']} folds, e o campeão é a melhor média nos {p['n_folds']} folds (nunca a média parcial da triagem).
- **Divergência:** um fold diverge se `val/theil` > {d.get('divergencia_theil', 10):g} ou se a métrica não é finita. Configurações divergentes não chegam a finais, não viram campeãs e ficam fora das médias (listadas à parte).
- **Ruído entre seeds:** o LSTM padrão roda com 5 seeds na referência de cada série; um efeito só é "real" se passar de 2× o desvio da diferença pareada entre seeds.
- **Teste:** usado só no fim de cada fase, para os campeões e as referências.

**Resultados em disco** (W&B e GitHub são só espelhos): cada série tem sua pasta, `outputs/` (série 2017–2023) e
`outputs_btc_longo/` (série 2014–2026). Em cada uma: `{{exp_name}}/` por experimento (parâmetros, histórico por época,
resultados por fold, pesos, previsões), `_grids/{{bloco}}/` por bloco (configurações, rankings, campeão, logs) e
`_relatorio/` com tabelas e figuras.

**Tempo estimado:** a fase 1 tem ~1.000 treinos de fold (~1–3 h no Kaggle com 2× T4); as fases 3 e 4 somam ~100 treinos na
série longa (~20–40 min); a fase 5 tem ~1.200 treinos na série longa (~2–4 h); as fases 6 a 8 somam ~600 treinos, parte
deles com dezenas de milhares de janelas (várias moedas, dados por hora) e com o TimeGAN (~2–4 h). Para rodar só uma parte,
use `FASES` (e `RESUME_FROM` para trazer os resultados das fases que já rodaram). Se a sessão cair ou passar de 12 h, retome (veja abaixo): o que terminou é pulado."""


def environment_md():
    return """## Como executar

O notebook é o mesmo em qualquer ambiente: clona o código do GitHub (ou usa a cópia local), acha os dados onde estiverem
e pula o que não estiver disponível. **Nada além de Python + GPU é obrigatório** (W&B e GitHub são opcionais).

### No Kaggle (execução principal)

1. **Dados:** os CSVs vêm no repositório clonado (série curta na raiz, série longa e séries externas em `data/`). Anexar o
   CSV curto como Dataset é opcional: o notebook o acha em `/kaggle/input` pelo nome ou pelas colunas `date`/`close`.
2. **Notebook:** *Code → New Notebook → File → Import Notebook* → `Kaggle_LSTM.ipynb`.
3. **Secrets:** *Add-ons → Secrets* → `GITHUB_TOKEN` e `WANDB_API_KEY`, marcados como anexados.
4. **Settings:** *Accelerator* **GPU T4 ×2**, *Internet* **On** (clone do código, pip, W&B e envio dos resultados).
5. *Save Version → **Save & Run All (Commit)***: roda em segundo plano (limite de 12 h); a saída fica em *Output*.
6. Para validar antes em minutos: `MODO_TESTE = True` (3 configurações por bloco, 2 épocas, pastas `*_teste`).

### No Colab ou Jupyter local

- **Colab:** *Runtime → Change runtime type → GPU*; o código é clonado; tokens em *Colab Secrets*; saída em `/content`.
- **Jupyter local:** abra o notebook dentro do repositório; tokens no `.env` (modelo em `.env.example`); saída no repositório.

**W&B.** Uma run por configuração, agrupada por bloco, num projeto por série (`if702-miniproject-2-lstm` e
`if702-miniproject-2-lstm-longo`); uma run de resumo por bloco (hiperparâmetro × métrica) e uma por fase. O teste só é
enviado na run de resumo da fase, depois de revelado.

**GitHub.** Com `GITHUB_TOKEN` com permissão de escrita (*Contents: Read and write*), ao fim de cada bloco os resultados vão
para o branch `resultados`, em `RUN_NAME/<pasta>/`.

**Retomada.** Para continuar uma execução interrompida: o mesmo `RUN_NAME` com `RESUME_FROM = "github"`, ou
`RESUME_FROM = "/kaggle/input/<output anterior>"` (a pasta que contém `outputs/` e `outputs_btc_longo/`). Para reaproveitar
uma fase já rodada (ex.: a fase 1 do Kaggle), basta que a pasta dela esteja lá: nada que já terminou é treinado de novo."""


def setup_cells():
    cfg = code('''# ===== Configuração da execução (edite aqui) =====
import os

REPO_URL = "https://github.com/diegoflyra/if702-miniproject-2.git"  # repositório com src/, grids/, config/
RESULTS_REPO_URL = REPO_URL      # onde espelhar os resultados ("" = não enviar ao GitHub)
RESULTS_BRANCH = "resultados"    # branch órfão só de resultados
RUN_NAME = ""                    # vazio = "lstm-AAAAMMDD-HHMM"; fixe um nome para retomar pelo GitHub
RESUME_FROM = ""                 # "" | "github" | pasta com os resultados de uma execução anterior
FASES = ["fase1", "fase2", "fase3", "fase4", "fase5", "fase6", "fase7", "fase8"]  # fases a executar (as demais só são lidas, se já tiverem resultados)
WORKERS_PER_GPU = 2              # experimentos simultâneos por GPU (LSTMs pequenos; o Kaggle tem 4 CPUs)
CPU_WORKERS = 1                  # sem GPU
DATA_PATH = ""                   # CSV (ou pasta) da série curta; vazio = raiz do repo → /kaggle/input → /content
SYNC_PESOS = False               # enviar também os .pth ao GitHub
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
USAR_WANDB = True                # False = não envia nada ao W&B, mesmo com a chave
USAR_GITHUB = True               # False = não espelha os resultados no GitHub, mesmo com o token
MODO_TESTE = False               # True = passada rápida (3 configs por bloco, 2 épocas) em pastas *_teste, para validar o
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
    env = code('''import json

import pandas as pd

if os.path.isdir("/kaggle/working"):
    WORKDIR = "/kaggle/working"
elif os.path.isdir("/content"):
    WORKDIR = "/content"
else:
    WORKDIR = REPO_DIR
os.environ["EXP_OUTPUT_ROOT"] = WORKDIR
os.environ["EXP_OUTPUT_SUFFIX"] = "_teste" if MODO_TESTE else ""  # o teste nunca se mistura com a execução real
os.environ.pop("EXP_OUTPUT_DIR", None)
GRID_EXTRA = "--max_configs 3 --epochs 2" if MODO_TESTE else ""
GRID_EXTRA_B0 = "--epochs 2" if MODO_TESTE else ""  # a referência roda completa: as 5 seeds medem o ruído
if MODO_TESTE:
    print("MODO_TESTE: 3 configurações por bloco, 2 épocas, pastas *_teste (não servem como resultado).")

RUN_NAME = RUN_NAME or time.strftime("lstm-%Y%m%d-%H%M")
if MODO_TESTE and not RUN_NAME.startswith("teste-"):
    RUN_NAME = "teste-" + RUN_NAME
if not USAR_GITHUB:
    RESULTS_REPO_URL = ""
os.environ.update({"RUN_NAME": RUN_NAME, "RESULTS_REPO_URL": RESULTS_REPO_URL, "RESULTS_BRANCH": RESULTS_BRANCH,
                   "GITHUB_TOKEN": GITHUB_TOKEN, "WANDB_ENTITY": WANDB_ENTITY})
if DATA_PATH:
    os.environ["DATA_PATH"] = DATA_PATH
print(f"RUN_NAME = {RUN_NAME} | resultados em {WORKDIR}")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import common
import data as data_mod
import github_sync
import report_utils as rep
import wandb_report

if RESUME_FROM == "github":
    github_sync.pull(WORKDIR)
elif RESUME_FROM:
    subdirs = [d for d in os.listdir(RESUME_FROM) if d.startswith("outputs") and os.path.isdir(os.path.join(RESUME_FROM, d))]
    for d in subdirs or [os.path.basename(os.path.normpath(RESUME_FROM))]:
        src = os.path.join(RESUME_FROM, d) if subdirs else RESUME_FROM
        shutil.copytree(src, os.path.join(WORKDIR, d), dirs_exist_ok=True)
    print(f"Resultados anteriores copiados de {RESUME_FROM}; o que já foi concluído será pulado.")

wandb_key = _secret("WANDB_API_KEY") if USAR_WANDB else ""
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key
    os.environ.pop("WANDB_MODE", None)
    print("W&B: chave carregada.")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print(f"W&B desativado ({'USAR_WANDB = False' if not USAR_WANDB else 'sem chave'}).")
if not (GITHUB_TOKEN and RESULTS_REPO_URL):
    print("GitHub: sem GITHUB_TOKEN/RESULTS_REPO_URL, os resultados ficam só em disco (e nos .zip).")


def usar_estudo(config):
    """Troca a série/estudo ativo: dados, partições, pasta de resultados e projeto do W&B."""
    global OUTPUTS
    os.environ["ESTUDO_CONFIG"] = config
    est = common.load_study()
    os.environ["WANDB_PROJECT"] = est.get("wandb_project", "if702-miniproject-2-lstm")
    OUTPUTS = common.output_dir()
    print(f"Estudo ativo: {est.get('titulo')} ({config}) → {OUTPUTS} | W&B: {os.environ['WANDB_PROJECT']}")


def executar(fase):
    """True se a fase está em FASES; senão as células de treino dela são puladas (e as análises leem o que houver)."""
    if fase not in FASES:
        print(f"{fase} fora de FASES: treino pulado; as análises abaixo usam os resultados já existentes, se houver.")
    return fase in FASES


def backup(bloco=""):
    """Backup parcial ao fim de cada bloco: <pasta>.zip (sem .pth) + espelho no GitHub, se configurado."""
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
    checks = code('''# Verificações rápidas antes de gastar GPU (sem treino): GPUs, modelos, métricas, augmentation e todos os grids
if shutil.which("nvidia-smi"):
    !nvidia-smi -L
else:
    print("nvidia-smi não encontrado; o treino usa o device que o PyTorch enxergar (CPU se não houver GPU).")
!python tests/check_models.py
!python tests/check_metrics.py
!python tests/check_augment.py
!python tests/check_grids.py''')
    return [md("# 0. Preparação do ambiente"), cfg, clone, pip, env, checks]


def phase_intro(ph, specs):
    st = load_study(ph["estudo"])
    dados, part = st["dados"], st["particoes"]
    blocos = "\n".join(f"- `{b}` — {specs[b]['titulo']}: {specs[b]['pergunta']} ({search_space(specs[b])})" for b in ph["blocos"])
    txt = [f"# {ph['titulo']}", "", f"**Motivação.** {ph['motivacao']}", ""]
    if ph["blocos"]:
        txt += ["**Blocos desta fase:**", "", blocos, ""]
    txt += [f"**Série:** `{dados['arquivos'][dados['tickers'][0]]}`, de {dados['inicio']} a {dados['fim']}; "
            f"{part['n_folds']} folds de validação de {part['tamanho_validacao_dias']} dias; teste a partir de {part['teste_inicio']}."]
    if ph.get("ressalva"):
        txt += ["", f"**Ressalva:** {ph['ressalva']}"]
    return md("\n".join(txt))


def phase_data_cells(ph):
    return [code(f'''usar_estudo("{ph['estudo']}")
manifesto = data_mod.prepare_prices()
display(data_mod.describe_folds())
!python tests/check_data.py
!python tests/check_features.py''')]


def descoberta_md(ph):
    itens = "\n".join(f"- {t}" for t in ph["descoberta"])
    return md(f"""### O que descobrimos nesta fase

Na execução de referência:

{itens}

**📝 Nesta execução:** _…_ (confira nas tabelas acima se os números se repetem)""")


def run_cell(cell, ph):
    """Células de treino só rodam se a fase estiver em FASES (as de análise sempre rodam)."""
    src = "".join(cell["source"])
    if src.startswith("!python src/grid_search.py"):
        return code(f'''if executar("{ph['id']}"):
    {src}''')
    return cell


def phase_test_cells(ph, specs):
    blocks = [b for b in ph["blocos"] if specs[b].get("papel") != "ablacao"]
    main = ph.get("campeao_principal")
    cells = [md(f"""## {ph['titulo'].split('—')[0].strip()} · Teste revelado

Até aqui todas as escolhas desta fase usaram só a validação. Agora o período de teste é usado **uma única vez**, para as
referências e o campeão de cada bloco (média ± desvio entre os modelos dos K folds). A concordância entre validação e
teste é a evidência de que não houve vazamento; uma discrepância grande (ex.: mudança de regime) é um achado."""),
             code(f"final = rep.final_report({json.dumps(blocks)})\nfinal")]
    if main:
        cells.append(code(f'''campeao_fase = rep.champion_name("{main}")
rep.plot_predictions(campeao_fase)'''))
    if ph.get("fusao"):
        fz = ph["fusao"]
        cells.append(md("""### Fusão de modelos

Inspirada na Combinatorial Fusion Analysis do paper de Wu et al. (2025): todas as combinações de 2 a N modelos, com
pesos iguais, por desempenho (1/MSE de validação) ou por **diversidade cognitiva** (distância entre as funções
rank-score dos modelos), combinando valores previstos (escore) ou posições (rank). Diferente do paper, os pesos vêm só da
validação walk-forward (cada dia previsto por um modelo que ainda não o viu) e a fusão é **escolhida uma vez, pela
validação**, antes de olhar o teste; o paper escolhia a cada dia a combinação mais próxima do preço real."""))
        cells.append(code(f'''modelos_fusao = list(rep.results.configs("{fz['referencias']}")["exp_name"]) + [rep.champion_name(b) for b in {json.dumps(fz['campeoes'])} if rep.champion_name(b)]
fusao = rep.fusion_report(modelos_fusao, max_size={fz.get('max_modelos', 5)})
fusao.head(15)'''))
        cells.append(code(f'wandb_report.log_study("{main}", {json.dumps(blocks)}, nome="{ph["id"]}__resumo")\nbackup("{ph["id"]}")'))
    return cells


def revisao_cells(ph, fases, specs):
    alvo = next(f for f in fases if f["id"] == ph["revisa"])
    blocos = [b for b in alvo["blocos"] if specs[b].get("papel") != "ablacao"]
    return [code(f'usar_estudo("{ph["estudo"]}")'),
            md("## Quantas configurações divergiram, e o que elas têm em comum"),
            code(f"por_bloco, padroes = rep.divergence_summary({json.dumps(blocos)})\ndisplay(por_bloco)\ndisplay(padroes)"),
            md("## A correção mudou alguma decisão?\n\nPara cada bloco: finalistas divergentes, se o campeão divergiu e se ele é o melhor entre os não divergentes."),
            code(f"rep.selection_check({json.dumps(blocos)})")]


def chain_cells(ph):
    main = ph["campeao_principal"]
    return [md(f"""## {ph['titulo'].split('—')[0].strip()} · O que realmente melhorou o modelo (validação)

A **cadeia de decisões** reconstrói, pela herança entre blocos, o caminho do LSTM padrão até o campeão (`{main}`): o que
mudou em cada passo, o Δ pareado por fold, em quantos folds a mudança venceu e se o ganho passa do ruído entre seeds."""),
            code(f'cadeia = rep.decision_chain("{main}")\ncadeia')]


def closing_cells(doc):
    passos = "\n".join(f"- {t}" for t in doc.get("proximos_passos", []))
    return [md("""# Comparação entre as fases

Para cada fase com treino: o passeio aleatório, o LSTM padrão e o campeão da fase, na validação e no teste. É o resumo de
tudo o que as tentativas de melhorar o modelo conseguiram."""),
            code("comparacao = rep.compare_phases()\ncomparacao"),
            md(f"""# Próximos passos

{passos}

## 📝 Conclusões

- **Alguma tentativa fez o LSTM bater o passeio aleatório no teste de forma consistente? Qual e por quanto?** _…_
- **O que os hiperparâmetros da lista mudaram de fato (fase 1), e por que não se sustentou no teste?** _…_
- **Mais dados (fase 3) ou mais informação (fase 4): o que pesou mais?** _…_
- **Acerto de direção (POCID): em algum momento saiu de ~50%?** _…_""")]


def block_markdown(s, study):
    d = study["decisao"]
    parts = [f"## {s['titulo']}", "", f"**Pergunta:** {s['pergunta']}", "", s.get("descricao", "")]
    if s.get("herda_de"):
        parts += ["", f"**Base:** o melhor campeão entre {', '.join(f'`{b}`' for b in s['herda_de'])}."]
    elif s.get("configs_de"):
        parts += ["", "**Configurações:** " + "; ".join(
            f"`{c['_nome']}` = campeão de `{c['bloco']}` ({c['estudo']})" + (f" com {c['sobrescrever']}" if c.get("sobrescrever") else "")
            for c in s["configs_de"]) + "."]
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
    cells.append(code(f'rep.diverged_configs("{name}")  # treinaram, mas divergiram (val/theil > limite): fora das médias'))
    if multi_axis:
        cells.append(md(f"#### O que cada hiperparâmetro muda — `{name}`\n\n"
                        "Efeito de cada valor em relação ao valor da base, **controlando os outros eixos** (modelo aditivo "
                        "sobre todas as configurações), e quanto da variação da métrica cada eixo explica. "
                        "Verde = melhora além de 2× o ruído entre seeds; cinza = indistinguível de ruído."))
        met = f', metric="{s["metrica_decisao"]}"' if s.get("metrica_decisao") else ""
        cells.append(code(f'importancia, efeitos = rep.axis_importance("{name}"{met})\ndisplay(importancia)\ndisplay(efeitos)'))
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
        met = f', metric="{s["metrica_decisao"]}"' if s.get("metrica_decisao") else ""
        cells.append(code(f'rep.paired_comparison(rep.base_config_name("{name}"), "{name}__*"{met})'))
        cells.append(code(f'rep.paired_comparison(rep.base_config_name("{name}"), "{name}__*", metric="val/pocid")'))
    if s.get("timegan"):
        cells.append(md("#### Diagnósticos do TimeGAN\n\nO gerador reproduz as propriedades do BTC ou só o ruído? Fatos estilizados do retorno (real × sintético), score discriminativo e o TSTR (linha \"só sintético\")."))
        cells.append(code(f'rep.timegan_diagnostics("{name}")'))
    if s.get("comparar_com"):
        cells.append(md(f"#### Comparação pareada com `{s['comparar_com']}` (validação)"))
        cells.append(code(f'rep.paired_comparison("{s["comparar_com"]}", "{name}__*")'))
    if s.get("checagem"):
        prev = s["herda_de"][0]
        cells.append(code(f'pd.concat([rep.grid_ranking("{prev}", "final").head(1),\n           rep.grid_ranking("{name}", "final")], ignore_index=True)'))
    short = s["titulo"].split("—")[0].strip()
    cells.append(md(f"### 📝 Análise — {short}\n\n" + "\n".join(f"- **{q}** _…_" for q in s.get("analise", []))))
    cells.append(code(f'backup("{name}")'))
    return cells


def build(out):
    doc = load("_fases")
    fases = doc["fases"]
    specs = {b: load(b) for ph in fases for b in ph["blocos"]}
    cells = [md(header(fases, specs)), md(environment_md())] + setup_cells()
    for ph in fases:
        cells.append(phase_intro(ph, specs))
        if ph.get("tipo") == "revisao":
            cells += revisao_cells(ph, fases, specs)
        else:
            cells += phase_data_cells(ph)
            st = load_study(ph["estudo"])
            for name in ph["blocos"]:
                cells += [run_cell(c, ph) for c in block_cells(name, specs[name], st)]
            if ph.get("campeao_principal") and any(specs[b].get("herda_de") for b in ph["blocos"]):
                cells += chain_cells(ph)
            cells += phase_test_cells(ph, specs)
        cells.append(descoberta_md(ph))
    cells += closing_cells(doc)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{out}: {len(cells)} células, {len(fases)} fases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", default=os.path.join(ROOT, "Kaggle_LSTM.ipynb"))
    build(parser.parse_args().saida)
