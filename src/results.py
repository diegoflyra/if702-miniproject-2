"""Leitura dos resultados gravados em disco (compartilhado por grid_search e report_utils).

Tudo aqui lê só os arquivos locais de outputs/: nada depende do W&B nem da sessão que treinou.
"""
import os

import numpy as np
import pandas as pd

import common


def exp_dir(exp_name):
    return os.path.join(common.output_dir(), exp_name)


def block_dir(block):
    return os.path.join(common.grids_dir(), block)


def fold_results(exp_name):
    """{fold: resultado.json} só dos folds concluídos (com done.json)."""
    base = os.path.join(exp_dir(exp_name), "folds")
    out = {}
    if not os.path.isdir(base):
        return out
    for d in os.listdir(base):
        path = os.path.join(base, d)
        res = common.load_json(os.path.join(path, "resultado.json"))
        if res and os.path.isfile(os.path.join(path, "done.json")):
            out[int(res["fold"])] = res
    return dict(sorted(out.items()))


def summarize(exp_name, folds=None, splits=("val", "gap")):
    """Média ± desvio das métricas nos `folds` pedidos (None = todos os concluídos).

    Devolve None se algum fold pedido não estiver concluído. O teste só entra se "test" estiver em `splits`.
    """
    res = fold_results(exp_name)
    folds = sorted(res) if folds is None else list(folds)
    if not folds or any(f not in res for f in folds):
        return None
    row = {"exp_name": exp_name, "n_folds": len(folds)}
    for split in splits:
        source = "val" if split == "gap" else split  # o gap é gravado junto da validação (melhor época)
        keys = sorted({k for f in folds for k in res[f][source] if k.startswith(split + "/") and "/ticker/" not in k})
        for k in keys:
            vals = [res[f][source][k] for f in folds if k in res[f][source]]
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    bad = [f for f in folds if common.fold_diverged(res[f]["val"])]
    row["folds_divergentes"] = len(bad)
    row["divergiu"] = bool(bad)
    row["melhor_epoca_media"] = float(np.mean([res[f]["melhor_epoca"] for f in folds]))
    row["epocas_treinadas_media"] = float(np.mean([res[f]["epocas_treinadas"] for f in folds]))
    row["tempo_s"] = float(sum(res[f]["tempo_s"] for f in folds))
    return row


def per_fold(exp_name, metric, split="val"):
    """Série fold → valor, para comparações pareadas."""
    return pd.Series({f: r[split].get(metric, np.nan) for f, r in fold_results(exp_name).items()}, name=exp_name)


def configs(block):
    path = os.path.join(block_dir(block), "configs.csv")
    # rótulos como "None" (ex.: treino_inicio = todo o histórico) são texto, não valor ausente
    return pd.read_csv(path, keep_default_na=False, na_values=[""]) if os.path.isfile(path) else pd.DataFrame()


def ranking(block, stage="final", with_test=False):
    """Ranking do bloco pela métrica de decisão (na validação).

    stage="triagem": todas as configurações, média só nos folds de triagem.
    stage="final": só as que completaram TODOS os folds, média nos K folds.
    Configurações com algum fold divergente (coluna `divergiu`) ficam no fim, qualquer que seja a média.
    """
    metric, mode, triage_folds, _ = common.decision()
    cfg = configs(block)
    if cfg.empty:
        return pd.DataFrame()
    spec = common.load_json(os.path.join(block_dir(block), "spec.json"), {})
    if spec.get("metrica_decisao"):  # o bloco pode decidir por outra métrica (ex.: Theil entre horizontes)
        metric = spec["metrica_decisao"]
        mode = "min" if metric.split("/")[-1] in ("rmse", "mse", "mae", "mape", "theil", "loss", "arv", "smape", "mase") else "max"
    all_folds = list(range(1, common.n_folds() + 1))
    folds = triage_folds if (stage == "triagem" and spec.get("triagem_efetiva", spec.get("triagem", True))) else all_folds
    splits = ("val", "gap", "test") if with_test else ("val", "gap")
    rows = []
    for _, c in cfg.iterrows():
        s = summarize(c["exp_name"], folds, splits)
        if s is not None:
            rows.append({**c.to_dict(), **s})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    col = f"{metric}_mean"
    # configurações com fold divergente vão para o fim: nunca viram finalistas nem campeãs
    df = df.sort_values(["divergiu", col], ascending=[True, mode == "min"], na_position="last").reset_index(drop=True)
    df.insert(0, "posicao", np.arange(1, len(df) + 1))
    return df


def champion(block):
    return common.load_json(os.path.join(block_dir(block), "campeao.json"))
