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
CONFIG_PATH = os.environ.get("ESTUDO_CONFIG", os.path.join(REPO_DIR, "config", "estudo.json"))
DATA_DIR = os.environ.get("EXP_DATA_DIR", os.path.join(REPO_DIR, "data"))


def output_dir():
    path = os.environ.get("EXP_OUTPUT_DIR", os.path.join(REPO_DIR, "outputs"))
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
    study = load_json(CONFIG_PATH)
    if study is None:
        raise FileNotFoundError(f"Configuração do estudo não encontrada: {CONFIG_PATH}")
    return study


def default_params():
    return copy.deepcopy(load_study()["padrao"])


def decision():
    """(métrica, modo, folds de triagem, nº de finalistas) definidos em config/estudo.json."""
    d = load_study()["decisao"]
    return d["metrica"], d["modo"], list(d["triagem_folds"]), int(d["n_finalistas"])


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
