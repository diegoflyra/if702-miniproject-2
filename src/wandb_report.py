"""Painéis de resultados no W&B (opcional; sem WANDB_API_KEY não faz nada).

Além das runs de cada configuração (gravadas pelo train.py), cria uma run de resumo por bloco (`{bloco}__resumo`,
job_type "resumo") com:
  - `ranking`: tabela interativa com todas as configurações, eixos e métricas de validação;
  - `<eixo>/<métrica>`: barras com a média da métrica por valor do hiperparâmetro (hiperparâmetro × acerto/erro),
    para cada eixo do bloco e cada métrica (rmse, mse, mae, mape, theil, pocid, da);
  - `<eixo>/<métrica>_dispersao`: dispersão de todas as configurações, quando o eixo é numérico;
  - `efeitos` / `importancia`: efeito de cada valor controlando os outros eixos e a imagem correspondente;
  - no bloco de ablação: a tabela e o gráfico da ablação.
E uma run final `estudo__resumo` com a cadeia de decisões, o ruído entre seeds e o relatório final (com o teste,
que só é enviado aqui, depois de revelado no notebook).
Tudo vem dos arquivos locais de outputs/: o W&B é só uma vitrine, nunca a fonte dos números.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import results  # noqa: E402

CHART_METRICS = ["rmse", "mse", "mae", "mape", "theil", "pocid", "da", "arv", "smape", "mase", "rmse_preco"]


def enabled():
    return os.environ.get("WANDB_MODE") != "disabled" and bool(os.environ.get("WANDB_API_KEY"))


def _init(name, group, config=None):
    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    run_id = hashlib.md5(f"{os.environ.get('RUN_NAME', 'local')}/{name}".encode()).hexdigest()[:16]
    return wandb.init(project=os.environ.get("WANDB_PROJECT", "if702-miniproject-2-lstm"),
                      entity=os.environ.get("WANDB_ENTITY") or None, id=run_id, resume="allow",
                      group=group, job_type="resumo", name=name, config=config or {})


def _image(path):
    import wandb

    return wandb.Image(path) if os.path.isfile(path) else None


def _as_number(values):
    try:
        return pd.to_numeric(values)
    except (TypeError, ValueError):
        return None


def log_block(block):
    """Run de resumo do bloco: ranking, hiperparâmetro × métrica, efeitos e (se houver) ablação."""
    if not enabled():
        return
    import matplotlib

    matplotlib.use("Agg")
    import report_utils as rep
    import wandb

    spec = common.load_json(os.path.join(results.block_dir(block), "spec.json"), {})
    stage = "triagem" if spec.get("triagem_efetiva") else "final"
    df = results.ranking(block, stage)
    if df.empty:
        return
    if "divergiu" in df:  # divergentes distorcem as médias dos gráficos; ficam só na tabela `ranking`
        df_all, df = df, df[~df["divergiu"]].reset_index(drop=True)
        if df.empty:
            df = df_all
    axes = [a for a in spec.get("eixos", {}) if a in df and df[a].astype(object).where(df[a].notna(), '—').astype(str).nunique() > 1]
    label_cols = axes or [c for c in ("desfeito", "origem", "config") if c in df][:1]
    for c in ("desfeito", "origem", "config"):
        if c in df:
            df[c] = df[c].where(df[c].notna(), df.get("config", "base"))  # a base re-treinada vira "base"
    metric_cols = [f"val/{m}_mean" for m in CHART_METRICS if f"val/{m}_mean" in df]
    run = _init(f"{block}__resumo", block, {"bloco": block, "etapa": stage, "n_configs": len(df),
                                              "eixos": axes, "metrica_decisao": common.decision()[0]})
    try:
        cols = ["posicao", "exp_name"] + label_cols + metric_cols + [
            c for c in ("gap/rmse_mean", "melhor_epoca_media", "num_parameters") if c in df]
        table = df[cols].copy()
        for c in label_cols:
            table[c] = table[c].astype(str)
        payload = {"ranking": wandb.Table(dataframe=table)}

        for axis in label_cols:
            groups = df.groupby(df[axis].astype(object).where(df[axis].notna(), '—').astype(str))
            numeric = _as_number(df[axis]) if axis in axes else None
            for col in metric_cols:
                m = col.split("/")[1].replace("_mean", "")
                agg = groups[col].agg(["mean", "count"]).reset_index()
                agg.columns = [axis, m, "n_configs"]
                if numeric is not None:
                    agg = agg.assign(_k=pd.to_numeric(agg[axis])).sort_values("_k").drop(columns="_k")
                payload[f"{axis}/{m}"] = wandb.plot.bar(wandb.Table(dataframe=agg), axis, m,
                                                        title=f"{block}: {axis} × {m} (validação, média)")
                if numeric is not None:
                    pts = pd.DataFrame({axis: numeric, m: df[col]})
                    payload[f"{axis}/{m}_dispersao"] = wandb.plot.scatter(
                        wandb.Table(dataframe=pts), axis, m, title=f"{block}: {axis} × {m} (cada configuração)")

        if axes:
            importance, effects = rep.axis_importance(block)
            if effects is not None:
                payload["efeitos"] = wandb.Table(dataframe=effects.astype({"valor": str}))
                payload["importancia"] = wandb.Table(dataframe=importance)
                img = _image(os.path.join(common.report_dir(), f"efeitos_{block}.png"))
                if img:
                    payload["efeitos_grafico"] = img
        if spec.get("ablacao"):
            abl = rep.ablation_table(block)
            if not abl.empty:
                payload["ablacao"] = wandb.Table(dataframe=abl.astype({"campeão": str, "padrão": str}))
                img = _image(os.path.join(common.report_dir(), f"ablacao_{block}.png"))
                if img:
                    payload["ablacao_grafico"] = img
        champ = results.champion(block)
        if champ:
            run.summary.update({"campeao": champ["exp_name"], **{k: v for k, v in champ["metricas"].items()
                                                                   if k.startswith("val/") and "/ticker/" not in k}})
        run.log(payload)
    finally:
        run.finish()


def log_study(final_block, report_blocks, nome="estudo__resumo"):
    """Run de resumo de uma fase: cadeia de decisões, ruído entre seeds e relatório final (validação × teste)."""
    if not enabled():
        return
    import matplotlib

    matplotlib.use("Agg")
    import report_utils as rep
    import wandb

    run = _init(nome, "estudo", {"campeao_principal": final_block})
    try:
        payload = {}
        spec = common.load_json(os.path.join(results.block_dir(final_block), "spec.json"), {})
        chain = rep.decision_chain(final_block) if spec.get("herda_de") else None
        if chain is not None:
            payload["cadeia_decisoes"] = wandb.Table(dataframe=chain.astype(str))
        img = _image(os.path.join(common.report_dir(), f"cadeia_decisoes_{final_block}.png"))
        if img:
            payload["cadeia_decisoes_grafico"] = img
        final = rep.final_report(report_blocks)
        payload["resultado_final"] = wandb.Table(dataframe=final)
        for m in CHART_METRICS:
            for split in ("val", "test"):
                col = f"{split}/{m}"
                if col in final:
                    # rótulo único por linha: bloco + configuração (várias linhas se chamam "base")
                    t = pd.DataFrame({"modelo": final["bloco"].str.replace("lstm_", "") + ": "
                                      + final["exp_name"].str.split("__").str[-1], col: final[col]})
                    payload[f"final/{split}_{m}"] = wandb.plot.bar(wandb.Table(dataframe=t), "modelo", col,
                                                                   title=f"Campeões e referências: {col}")
        sd = rep.noise_floor(verbose=False)
        if sd is not None:
            run.summary["ruido_entre_seeds"] = sd
        champ = results.champion(final_block)
        if champ:
            s = results.summarize(champ["exp_name"], splits=("val", "test"))
            run.summary.update({k: v for k, v in (s or {}).items() if isinstance(v, (int, float)) and np.isfinite(v)})
        run.log(payload)
    finally:
        run.finish()
