"""Caminhos, configuração do estudo e utilidades compartilhadas por treino, grid e relatório."""
import copy
import json
import os
import random
import subprocess

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path=os.path.join(REPO_DIR, ".env")):
    """Lê KEY=VALOR de um .env local sem sobrescrever variáveis já definidas (Kaggle/Colab Secrets têm prioridade)."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value and not os.environ.get(key.strip()):
                os.environ[key.strip()] = value


load_dotenv()
DATA_DIR = os.environ.get("EXP_DATA_DIR", os.path.join(REPO_DIR, "data"))


def config_path():
    """Configuração do estudo ativo: ESTUDO_CONFIG (relido a cada chamada: o notebook troca de estudo entre fases)."""
    path = os.environ.get("ESTUDO_CONFIG", os.path.join("config", "estudo.json"))
    return path if os.path.isabs(path) else os.path.join(REPO_DIR, path)


def study_output_dir(config=None):
    """Pasta de resultados de um estudo: EXP_OUTPUT_ROOT (padrão: raiz do repo) / `saida` do estudo + EXP_OUTPUT_SUFFIX.

    EXP_OUTPUT_DIR, se definido, vale para o estudo ativo (uso avulso pela linha de comando).
    """
    study = load_json(config if config and os.path.isabs(config) else os.path.join(REPO_DIR, config)) if config else load_study()
    root = os.environ.get("EXP_OUTPUT_ROOT", REPO_DIR)
    return os.path.join(root, study.get("saida", "outputs") + os.environ.get("EXP_OUTPUT_SUFFIX", ""))


def output_dir():
    """Resultados do estudo ativo: estudos diferentes nunca se misturam."""
    path = os.environ.get("EXP_OUTPUT_DIR") or study_output_dir()
    os.makedirs(path, exist_ok=True)
    return path


def grids_dir():
    return os.path.join(output_dir(), "_grids")


def report_dir():
    path = os.path.join(output_dir(), "_relatorio")
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    """Grava de forma atômica: uma queda no meio da escrita nunca deixa um JSON corrompido."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_study():
    study = load_json(config_path())
    if study is None:
        raise FileNotFoundError(f"Configuração do estudo não encontrada: {config_path()}")
    return study


def phases():
    """Fases do estudo (grids/_fases.json), na ordem das descobertas."""
    return load_json(os.path.join(REPO_DIR, "grids", "_fases.json"))["fases"]


def default_params():
    return copy.deepcopy(load_study()["padrao"])


def decision():
    """(métrica, modo, folds de triagem, nº de finalistas) definidos em config/estudo.json."""
    d = load_study()["decisao"]
    return d["metrica"], d["modo"], list(d["triagem_folds"]), int(d["n_finalistas"])


def divergence_limit():
    """Theil de validação acima do qual um fold é considerado divergente (config/estudo.json → decisao)."""
    return float(load_study()["decisao"].get("divergencia_theil", 10.0))


def fold_diverged(val_metrics):
    """True se o fold divergiu: métrica de decisão não finita ou val/theil acima do limite."""
    metric = load_study()["decisao"]["metrica"]
    v, theil = val_metrics.get(metric), val_metrics.get("val/theil")
    if v is None or not np.isfinite(v):
        return True
    return theil is not None and (not np.isfinite(theil) or theil > divergence_limit())


def n_folds():
    return int(load_study()["particoes"]["n_folds"])


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        out = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None
