"""Tabelas e figuras do relatório. Lê só os arquivos locais de outputs/ (nunca o W&B).

Funções que mostram o TESTE (use só na seção final do notebook): final_report, champion_ticker_metrics,
plot_per_ticker(split="test"), plot_predictions(split="test"), plot_strategy.
Tabelas e figuras são gravadas também em outputs/_relatorio/.
"""
import fnmatch
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import metrics as M  # noqa: E402
import results  # noqa: E402

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)


def _metric():
    return common.decision()[0]


def _save(fig, name):
    path = os.path.join(common.report_dir(), name + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def _save_csv(df, name):
    df.to_csv(os.path.join(common.report_dir(), name + ".csv"), index=False)


def _short(exp, block=None):
    return exp.split("__", 1)[1] if "__" in exp else exp


def _axes(block):
    cfg = results.configs(block)
    return [c for c in cfg.columns if c not in ("exp_name", "e_base", "num_parameters")]


def _stable(df, block):
    """Tira as configurações com fold divergente das médias (elas distorcem tudo) e avisa quantas foram."""
    if df.empty or "divergiu" not in df:
        return df
    n = int(df["divergiu"].sum())
    if n:
        print(f"{block}: {n} de {len(df)} configurações divergiram (val/theil > {common.divergence_limit():g}) "
              "e ficam fora das médias abaixo; veja rep.diverged_configs().")
    return df[~df["divergiu"]].reset_index(drop=True)


def diverged_configs(block, stage="triagem"):
    """Configurações com algum fold divergente, com os hiperparâmetros que as distinguem."""
    df = results.ranking(block, stage)
    if df.empty:
        df = results.ranking(block, "final")
    if df.empty or "divergiu" not in df or not df["divergiu"].any():
        print(f"{block}: nenhuma configuração divergiu.")
        return pd.DataFrame()
    return df[df["divergiu"]][["exp_name"] + _axes(block) + ["folds_divergentes", "val/theil_mean"]]


# --------------------------------------------------------------------------------------------------
# Rankings e visão do espaço de busca (só validação)
# --------------------------------------------------------------------------------------------------

def grid_ranking(block, stage="final"):
    """Ranking pela validação. stage="triagem" (folds de triagem, todas) ou "final" (K folds, finalistas)."""
    df = results.ranking(block, stage)
    if df.empty:
        print(f"{block}: sem resultados para '{stage}'.")
        return df
    _save_csv(df, f"ranking_{stage}_{block}")
    metric = _metric()
    cols = ["posicao", "exp_name"] + _axes(block) + [f"{metric}_mean", f"{metric}_std"]
    for m in M.METRICS + ["loss"]:
        c = f"val/{m}_mean"
        if c in df.columns and c not in cols:
            cols.append(c)
    cols += [c for c in ["gap/rmse_mean", "melhor_epoca_media", "epocas_treinadas_media", "num_parameters", "n_folds", "e_base"]
             if c in df.columns]
    return df[cols]


def discarded_configs(block):
    path = os.path.join(results.block_dir(block), "descartadas.csv")
    df = pd.read_csv(path) if os.path.isfile(path) and os.path.getsize(path) > 1 else pd.DataFrame()
    if df.empty:
        print(f"{block}: nenhuma combinação descartada.")
    return df


def _txt(col):
    """Coluna de rótulos como texto, com ausentes explícitos (o pandas 3 mantém NaN depois de astype(str))."""
    return col.astype(object).where(col.notna(), "—").astype(str)


def _label(df, cols):
    cols = [cols] if isinstance(cols, str) else list(cols)
    return df[cols].fillna("—").astype(str).agg(" | ".join, axis=1), " | ".join(cols)


def heatmap(block, row, col, facet=None, value=None, stage="triagem"):
    """Média de `value` (padrão: métrica de decisão) por célula row × col, com facetas opcionais.

    Na busca aleatória, várias configurações caem na mesma célula: mostra a média e (n) configurações.
    """
    df = results.ranking(block, stage)
    if df.empty and stage == "triagem":
        df = results.ranking(block, "final")
    if df.empty:
        print(f"{block}: sem resultados.")
        return
    value = value or f"{_metric()}_mean"
    df = _stable(df, block).copy()
    df["_row"], row_name = _label(df, row)
    df["_col"], col_name = _label(df, col)
    facets = [None] if facet is None else sorted(_txt(df[facet]).unique())
    fig, axs = plt.subplots(1, len(facets), figsize=(max(5, 1.3 * df["_col"].nunique() + 2) * len(facets), 0.6 * df["_row"].nunique() + 2.5),
                            squeeze=False)
    lower = M.is_lower_better(value.replace("_mean", "")) if value.endswith("_mean") else False
    cmap = "viridis_r" if lower else "viridis"
    vmin, vmax = df[value].min(), df[value].max()
    for ax, f in zip(axs[0], facets):
        sub = df if f is None else df[_txt(df[facet]) == f]
        piv = sub.pivot_table(index="_row", columns="_col", values=value, aggfunc="mean")
        cnt = sub.pivot_table(index="_row", columns="_col", values=value, aggfunc="count")
        im = ax.imshow(piv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(piv.shape[0]), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    n = cnt.values[i, j]
                    ax.text(j, i, f"{v:.4g}" + (f"\n({int(n)})" if n > 1 else ""), ha="center", va="center", fontsize=8, color="w")
        ax.set_xlabel(col_name)
        ax.set_ylabel(row_name)
        ax.set_title(f"{facet} = {f}" if f is not None else value)
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label=value)
    fig.suptitle(f"{block} — {value} ({stage})", y=1.02)
    _save(fig, f"heatmap_{block}_{value.replace('/', '-')}")
    plt.show()


def param_effect(block, params=None, stage="triagem"):
    """Efeito marginal de cada eixo: distribuição da métrica de decisão por valor do eixo.

    É a leitura principal de um bloco com busca aleatória (poucas repetições por célula de heatmap).
    """
    df = results.ranking(block, stage)
    if df.empty:
        print(f"{block}: sem resultados.")
        return
    metric = f"{_metric()}_mean"
    df = _stable(df, block)
    params = params or [a for a in _axes(block) if df[a].nunique() > 1]
    fig, axs = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.5), squeeze=False, sharey=True)
    summary = []
    for ax, p in zip(axs[0], params):
        groups = sorted(_txt(df[p]).unique(), key=lambda s: (len(s), s))
        data = [df.loc[_txt(df[p]) == g, metric].values for g in groups]
        ax.boxplot(data, showfliers=False)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.uniform(-0.12, 0.12, len(d)), d, s=12, alpha=0.7)
            summary.append({"eixo": p, "valor": groups[i - 1], "n": len(d), "media": d.mean(), "melhor": d.min() if M.is_lower_better(_metric()) else d.max()})
        ax.set_xticks(range(1, len(groups) + 1), groups, rotation=45, ha="right")
        ax.set_title(p)
    axs[0][0].set_ylabel(metric)
    fig.suptitle(f"{block} — efeito marginal de cada eixo ({stage})", y=1.02)
    _save(fig, f"efeito_eixos_{block}")
    plt.show()
    return pd.DataFrame(summary)


def plot_grid_bars(block, stage="triagem"):
    df = results.ranking(block, stage)
    if df.empty:
        df = results.ranking(block, "final")
    if df.empty:
        return
    metric = _metric()
    df = _stable(df, block)
    finalists = set(results.ranking(block, "final")["exp_name"]) if stage == "triagem" else set()
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(df) + 1.5))
    y = np.arange(len(df))[::-1]
    colors = ["tab:orange" if e in finalists else "tab:blue" for e in df["exp_name"]]
    ax.barh(y, df[f"{metric}_mean"], xerr=df[f"{metric}_std"], color=colors, alpha=0.85)
    ax.set_yticks(y, [_short(e) for e in df["exp_name"]], fontsize=8)
    rw = _random_walk_reference(metric)
    if rw is not None:
        ax.axvline(rw, color="k", ls="--", lw=1, label="passeio aleatório (Bloco 0)")
        ax.legend(loc="lower right")
    lo, hi = (df[f"{metric}_mean"] - df[f"{metric}_std"]).min(), (df[f"{metric}_mean"] + df[f"{metric}_std"]).max()
    pad = 0.05 * (hi - lo + 1e-12)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(f"{metric} (média ± desvio, {stage}); laranja = finalistas")
    ax.set_title(block)
    _save(fig, f"barras_{block}")
    plt.show()


def _random_walk_reference(metric):
    """Valor médio da métrica do passeio aleatório (naive_zero), se algum bloco de referência o treinou."""
    base = common.grids_dir()
    if not os.path.isdir(base):
        return None
    for block in sorted(os.listdir(base)):
        cfg = results.configs(block)
        if cfg.empty:
            continue
        for exp in cfg["exp_name"]:
            p = common.load_json(os.path.join(results.block_dir(block), "params", exp + ".json"), {})
            if p.get("model") == "naive_zero":
                s = results.summarize(exp)
                if s and f"{metric}_mean" in s:
                    return s[f"{metric}_mean"]
    return None


# --------------------------------------------------------------------------------------------------
# Campeões, curvas e comparações pareadas
# --------------------------------------------------------------------------------------------------

def champion_name(block):
    c = results.champion(block)
    return c["exp_name"] if c else None


def base_config_name(block):
    cfg = results.configs(block)
    hits = cfg.loc[cfg["e_base"].astype(bool), "exp_name"] if "e_base" in cfg else []
    return hits.iloc[0] if len(hits) else None


def show_champion(block):
    c = results.champion(block)
    if not c:
        print(f"{block}: campeão ainda não definido.")
        return
    metric = c["metrica_decisao"]
    base = common.load_json(os.path.join(results.block_dir(block), "base.json"), {})
    diff = {k: v for k, v in c["params"].items() if base.get("params", {}).get(k) != v}
    print(f"Campeão de {block}: {c['exp_name']}")
    print(f"  {metric} = {c['metricas'][metric + '_mean']:.6f} ± {c['metricas'][metric + '_std']:.6f} (validação, K folds)")
    print(f"  base do bloco: {base.get('origem')}")
    print(f"  mudanças em relação à base: {diff if diff else 'nenhuma (a base venceu)'}")
    shown = {k: c["params"][k] for k in sorted(c["params"])}
    print("  hiperparâmetros:", shown)


def plot_curves(exps, metrics=("loss", None), title=None, name=None):
    """Curvas médias entre folds (treino tracejado, validação contínua) por época."""
    metrics = [m or _metric().split("/")[-1] for m in metrics]
    fig, axs = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, m in zip(axs[0], metrics):
        for i, exp in enumerate(exps):
            path = os.path.join(results.exp_dir(exp), "historico_treino_agregado.csv")
            if not os.path.isfile(path):
                continue
            h = pd.read_csv(path)
            if h["epoch"].max() == 0:
                continue  # referência ingênua (sem épocas)
            color = f"C{i}"
            for split, ls in (("val", "-"), ("train", "--")):
                col = f"{split}/{m}_mean"
                if col in h:
                    ax.plot(h["epoch"], h[col], ls, color=color, label=f"{_short(exp)} ({split})" if split == "val" else None)
                    if split == "val" and f"{split}/{m}_std" in h:
                        ax.fill_between(h["epoch"], h[col] - h[f"{split}/{m}_std"], h[col] + h[f"{split}/{m}_std"], color=color, alpha=0.12)
        ax.set_xlabel("época")
        ax.set_ylabel(m)
        ax.set_title(f"{m}: validação (—) e treino (--)")
    axs[0][0].legend(fontsize=7)
    if title:
        fig.suptitle(title, y=1.02)
    _save(fig, name or f"curvas_{_short(exps[0])}")
    plt.show()


def plot_finalists(block):
    final = results.ranking(block, "final")
    if final.empty:
        return
    plot_curves(list(final["exp_name"].head(4)), title=f"{block} — finalistas", name=f"curvas_finalistas_{block}")


def paired_comparison(reference, others, metric=None):
    """Diferença fold a fold (outra − referência) na validação: média, desvio e vitórias em K folds.

    `others`: lista de nomes ou um padrão glob (ex.: "lstm_b4_bonus__*").
    """
    metric = metric or _metric()
    if isinstance(others, str):
        base = common.output_dir()
        others = sorted(d for d in os.listdir(base) if fnmatch.fnmatch(d, others) and d != reference)
    ref = results.per_fold(reference, metric)
    lower = M.is_lower_better(metric)
    sd = noise_floor(metric, verbose=False)
    rows = []
    for exp in others:
        s = results.per_fold(exp, metric)
        common_folds = ref.index.intersection(s.index)
        if len(common_folds) == 0:
            continue
        d = s[common_folds] - ref[common_folds]
        wins = int((d < 0).sum() if lower else (d > 0).sum())
        rows.append({"exp_name": exp, "folds": len(common_folds), f"Δ {metric} médio": d.mean(),
                     "Δ desvio": d.std(ddof=1) if len(d) > 1 else 0.0, "vitórias": f"{wins}/{len(common_folds)}",
                     "veredito": _beyond_noise(d.mean(), sd, metric),
                     **{f"Δ fold {f}": d[f] for f in common_folds}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(f"Δ {metric} médio", ascending=lower).reset_index(drop=True)
        print(f"Referência: {reference}. Δ = outra − referência; {'negativo' if lower else 'positivo'} = melhor que a referência."
              + (f" Veredito: |Δ| > 2 × ruído entre seeds ({2 * sd:.3g})." if sd else ""))
        _save_csv(df, f"pareado_{reference}")
    return df


# --------------------------------------------------------------------------------------------------
# Seção final: teste revelado uma única vez
# --------------------------------------------------------------------------------------------------

def final_report(blocks):
    """Referências (blocos com "papel": "referencia") e o campeão de cada bloco: validação × teste."""
    rows = []
    metric = _metric()
    for block in blocks:
        spec = common.load_json(os.path.join(results.block_dir(block), "spec.json"), {})
        if spec.get("papel") in ("referencia", "referencia_modelos"):
            exps = [(e, "referência") for e in results.configs(block)["exp_name"]]
        else:
            c = champion_name(block)
            exps = [(c, "campeão")] if c else []
        for exp, role in exps:
            s = results.summarize(exp, splits=("val", "gap", "test"))
            if s is None:
                continue
            row = {"bloco": block, "papel": role, "exp_name": exp, "n_folds": s["n_folds"]}
            for m in [metric.split("/")[-1]] + [x for x in M.METRICS if x != metric.split("/")[-1]]:
                for split in ("val", "test"):
                    if f"{split}/{m}_mean" in s:
                        row[f"{split}/{m}"] = s[f"{split}/{m}_mean"]
                        row[f"{split}/{m}_std"] = s[f"{split}/{m}_std"]
            rows.append(row)
    df = pd.DataFrame(rows)
    _save_csv(df, "resultado_final")
    return df


def ensemble_predictions(exp, split="test"):
    """Previsões de cada fold empilhadas + a média entre os modelos dos K folds (ensemble)."""
    frames = []
    for f in results.fold_results(exp):
        path = os.path.join(results.exp_dir(exp), "folds", f"fold_{f}", f"predicoes_{split}.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path, parse_dates=["data"]).assign(fold=f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    return df.groupby(["data", "ticker"], as_index=False).agg(previsto_lr=("previsto_lr", "mean"), real_lr=("real_lr", "first"),
                                                             preco_t=("preco_t", "first"))


def champion_ticker_metrics(block, split="test"):
    """Métricas por série (média ± desvio entre os modelos dos K folds) do campeão do bloco."""
    exp = champion_name(block)
    res = results.fold_results(exp)
    tickers = common.load_study()["dados"]["tickers"]
    rows = []
    for t in tickers:
        row = {"ticker": t}
        for m in M.METRICS:
            vals = [r[split].get(f"{split}/ticker/{t}/{m}") for r in res.values()]
            vals = [v for v in vals if v is not None]
            if vals:
                row[m] = np.mean(vals)
                row[f"{m}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    _save_csv(df, f"metricas_por_serie_{split}_{exp}")
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.5))
    for ax, m in zip(axs, ["rmse", "mape", "theil", "pocid"]):
        ax.bar(df["ticker"], df[m], yerr=df[f"{m}_std"], alpha=0.85)
        if m == "theil":
            ax.axhline(1, color="k", lw=1, ls="--")  # passeio aleatório
        if m == "pocid":
            ax.axhline(50, color="k", lw=1, ls="--")  # acaso
        ax.set_title(f"{m} ({split})")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{exp}: métricas por série ({split}, média ± desvio entre folds)", y=1.03)
    _save(fig, f"metricas_por_serie_{split}_{exp}")
    plt.show()
    return df


def plot_per_ticker(exps, metric="rmse", split="test"):
    tickers = common.load_study()["dados"]["tickers"]
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(tickers)), 4))
    w = 0.8 / max(len(exps), 1)
    for i, exp in enumerate(exps):
        res = results.fold_results(exp)
        vals = [np.mean([r[split].get(f"{split}/ticker/{t}/{metric}", np.nan) for r in res.values()]) for t in tickers]
        ax.bar(np.arange(len(tickers)) + i * w, vals, w, label=_short(exp))
    ax.set_xticks(np.arange(len(tickers)) + w * (len(exps) - 1) / 2, tickers, rotation=45)
    ax.set_ylabel(f"{split}/{metric}")
    ax.legend(fontsize=7)
    _save(fig, f"por_serie_{split}_{metric}")
    plt.show()


def plot_predictions(exp, ticker=None, split="test", last_days=250):
    """Retorno previsto (ensemble dos K folds) × real, em série temporal e dispersão."""
    df = ensemble_predictions(exp, split)
    ticker = ticker or df["ticker"].iloc[0]
    d = df[df["ticker"] == ticker].sort_values("data").tail(last_days)
    fig, axs = plt.subplots(1, 2, figsize=(15, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    axs[0].plot(d["data"], d["real_lr"], lw=0.8, label="real")
    axs[0].plot(d["data"], d["previsto_lr"], lw=1.2, label="previsto (ensemble)")
    axs[0].legend()
    axs[0].set_title(f"{_short(exp)} — {ticker} ({split})")
    axs[1].scatter(d["previsto_lr"], d["real_lr"], s=6, alpha=0.5)
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].axvline(0, color="k", lw=0.5)
    axs[1].set_xlabel("previsto")
    axs[1].set_ylabel("real")
    _save(fig, f"predicoes_{split}_{exp}_{ticker}")
    plt.show()


def plot_strategy(exps, split="test"):
    """Ilustrativo (sem custos, h=1): posição = sinal da previsão do ensemble; carteira igualmente ponderada.

    Não é critério de decisão do estudo: mostra se o sinal direcional tem valor econômico além das métricas.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    bh_done = False
    rows = []
    for i, exp in enumerate(exps):
        df = ensemble_predictions(exp, split)
        if df.empty:
            continue
        h = int(common.load_json(os.path.join(results.exp_dir(exp), "parametros.json"))["params"]["horizon"])
        df["estrategia"] = np.sign(df["previsto_lr"]) * df["real_lr"] / h
        daily = df.groupby("data")[["estrategia", "real_lr"]].mean()
        ax.plot(daily.index, np.exp(daily["estrategia"].cumsum()), label=_short(exp), color=f"C{i}")
        if not bh_done:
            ax.plot(daily.index, np.exp((daily["real_lr"] / h).cumsum()), "k--", lw=1, label="compra e mantém")
            bh_done = True
        r = daily["estrategia"]
        rows.append({"exp_name": exp, "retorno_total": float(np.exp(r.sum()) - 1),
                     "sharpe_anual": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0})
    ax.set_ylabel("valor da carteira (início = 1)")
    ax.set_title(f"Estratégia ilustrativa pelo sinal previsto ({split}, sem custos)")
    ax.legend(fontsize=7)
    _save(fig, f"estrategia_{split}")
    plt.show()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# O que realmente melhora: ruído entre seeds, importância dos eixos, cadeia de decisões e ablação (só validação)
# --------------------------------------------------------------------------------------------------

def _reference_block():
    """Bloco de referência (papel "referencia") do estudo ativo, procurado nos resultados dele."""
    base = common.grids_dir()
    for b in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        spec = common.load_json(os.path.join(base, b, "spec.json"), {})
        if spec.get("papel") == "referencia":
            return b, spec
    return None, {}


def noise_floor(metric=None, folds=None, verbose=True):
    """Ruído entre seeds: a mesma configuração (LSTM padrão) treinada com seeds diferentes (Bloco 0, campo "ruido").

    Devolve o desvio da diferença pareada entre duas execuções que só diferem na seed (média nos folds): uma mudança
    de hiperparâmetro cujo efeito não passa de ~2× esse valor é indistinguível de sorte na inicialização.
    """
    metric = metric or _metric()
    block, spec = _reference_block()
    names = [f"{block}__{n}" for n in spec.get("ruido", [])]
    series = [results.per_fold(e, metric) for e in names]
    series = [s for s in series if len(s)]
    if len(series) < 2:
        if verbose:
            print("Ruído entre seeds indisponível (rode o Bloco 0 com as configurações de 'ruido').")
        return None
    common_folds = sorted(set.intersection(*[set(s.index) for s in series]))
    if folds is not None:
        common_folds = [f for f in common_folds if f in folds]
    means = np.array([s[common_folds].mean() for s in series])
    diffs = [a - b for i, a in enumerate(means) for b in means[i + 1:]]
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float(abs(diffs[0]))
    if verbose:
        print(f"Ruído entre seeds ({len(series)} seeds, folds {common_folds}): {metric} médio = {means.mean():.6g} "
              f"± {means.std(ddof=1):.3g}; desvio da diferença pareada = {sd:.3g} → efeitos menores que ~{2 * sd:.3g} "
              "não se distinguem de ruído.")
    return sd


def _beyond_noise(delta, sd, metric):
    """'melhora' / 'piora' se |Δ| > 2·ruído (Δ = nova − referência), senão 'ruído'."""
    if sd is None or not np.isfinite(delta):
        return "?"
    if abs(delta) <= 2 * sd:
        return "ruído"
    good = delta < 0 if M.is_lower_better(metric) else delta > 0
    return "melhora" if good else "piora"


def axis_importance(block, metric=None, stage="triagem"):
    """Efeito de cada eixo controlando os demais (modelo aditivo por mínimos quadrados sobre as configurações).

    - `importancia`: quanto do R² se perde ao tirar o eixo do modelo (fração da variação da métrica que ele explica);
    - `efeitos`: para cada valor do eixo, a mudança média na métrica em relação ao valor da base do bloco,
      já descontado o efeito dos outros eixos, com o veredito frente ao ruído entre seeds.
    É a leitura certa de uma busca aleatória, em que cada valor aparece combinado com valores diferentes dos outros eixos.
    """
    metric = metric or _metric()
    col = f"{metric}_mean"
    df = results.ranking(block, stage)
    if df.empty or col not in df:
        print(f"{block}: sem resultados.")
        return None, None
    spec = common.load_json(os.path.join(results.block_dir(block), "spec.json"), {})
    div_rate = {}
    if "divergiu" in df:
        for a in spec.get("eixos", {}):
            if a in df:
                div_rate.update({(a, str(k)): float(v) for k, v in df.groupby(_txt(df[a]))["divergiu"].mean().items()})
    df = _stable(df, block)
    df = df[np.isfinite(df[col])]
    axes = [a for a in spec.get("eixos", {}) if a in df and _txt(df[a]).nunique() > 1]
    if not axes:
        print(f"{block}: nenhum eixo com mais de um valor.")
        return None, None
    base_row = df[df["e_base"].astype(bool)] if "e_base" in df else df.iloc[:0]
    ref = {a: (str(base_row[a].iloc[0]) if len(base_row) else _txt(df[a]).mode()[0]) for a in axes}
    y = df[col].to_numpy(float)

    def design(use):
        cols, names = [np.ones(len(df))], ["_intercepto"]
        for a in use:
            for lvl in sorted(_txt(df[a]).unique()):
                if lvl != ref[a]:
                    cols.append((_txt(df[a]) == lvl).to_numpy(float))
                    names.append((a, lvl))
        return np.column_stack(cols), names

    def r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sst = ((y - y.mean()) ** 2).sum()
        return (1 - (resid ** 2).sum() / sst if sst > 0 else 0.0), beta

    X, names = design(axes)
    additive = X.shape[0] > X.shape[1]
    sd = noise_floor(metric, folds=common.decision()[2] if stage == "triagem" else None, verbose=False)
    if additive:
        r2_full, beta = r2(X)
        coef = dict(zip(names, beta))
        imp = [{"eixo": a, "importancia (ΔR²)": r2_full - r2(design([b for b in axes if b != a])[0])[0]} for a in axes]
    else:  # poucas configurações para o modelo aditivo: diferenças de médias marginais
        r2_full, coef, imp = float("nan"), {}, []
        for a in axes:
            m = df.groupby(_txt(df[a]))[col].mean()
            coef.update({(a, lvl): m[lvl] - m[ref[a]] for lvl in m.index if lvl != ref[a]})
            imp.append({"eixo": a, "importancia (ΔR²)": float("nan")})
    rows = []
    for a in axes:
        for lvl in sorted(_txt(df[a]).unique(), key=lambda s: (len(s), s)):
            d = 0.0 if lvl == ref[a] else coef.get((a, lvl), np.nan)
            rows.append({"eixo": a, "valor": lvl, "n": int((_txt(df[a]) == lvl).sum()),
                         f"Δ {metric} vs base": d, "é o valor da base": lvl == ref[a],
                         "veredito": "—" if lvl == ref[a] else _beyond_noise(d, sd, metric),
                         "% divergiu": 100 * div_rate.get((a, lvl), 0.0)})
    effects = pd.DataFrame(rows)
    importance = pd.DataFrame(imp).sort_values("importancia (ΔR²)", ascending=False, na_position="last")
    model_txt = f"aditivo (R² = {r2_full:.2f})" if additive else "marginal (poucas configurações)"
    better_txt = "Δ < 0 é melhor" if M.is_lower_better(metric) else "Δ > 0 é melhor"
    print(f"{block}: {len(df)} configurações, modelo {model_txt}; {better_txt} para {metric}.")
    _save_csv(effects, f"efeitos_{block}")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(effects) + 1.2))
    e = effects[~effects["é o valor da base"]].iloc[::-1]
    colors = ["tab:green" if v == "melhora" else "tab:red" if v == "piora" else "tab:gray" for v in e["veredito"]]
    ax.barh([f"{a} = {v}" for a, v in zip(e["eixo"], e["valor"])], e[f"Δ {metric} vs base"], color=colors)
    if sd:
        ax.axvspan(-2 * sd, 2 * sd, color="k", alpha=0.08, label="±2× ruído entre seeds")
        ax.legend(fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Δ {metric} em relação ao valor da base (outros eixos controlados)")
    ax.set_title(f"{block}: efeito de cada valor (verde = melhora além do ruído)")
    _save(fig, f"efeitos_{block}")
    plt.show()
    return importance, effects


def _paired(a, b, metric):
    """Δ fold a fold (b − a) nos folds em comum."""
    sa, sb = results.per_fold(a, metric), results.per_fold(b, metric)
    folds = sa.index.intersection(sb.index)
    return (sb[folds] - sa[folds]) if len(folds) else pd.Series(dtype=float)


def _param_diff(p_old, p_new):
    keys = sorted(k for k in set(p_old) | set(p_new) if p_old.get(k) != p_new.get(k) and not k.startswith("_"))
    return ", ".join(f"{k}: {p_old.get(k)} → {p_new.get(k)}" for k in keys) or "nenhuma"


def _exp_params(exp):
    return common.load_json(os.path.join(results.exp_dir(exp), "parametros.json"), {}).get("params", {})


def decision_chain(final_block, start=None, metric=None, extra_metric="val/pocid"):
    """Cadeia de decisões que levou ao campeão final, reconstruída de trás para frente pela herança (base.json).

    Para cada bloco: o que mudou em relação ao campeão anterior, o Δ pareado por fold na validação, em quantos
    folds a mudança venceu e se o ganho supera o ruído entre seeds. Responde "o que realmente melhorou o modelo".
    """
    metric = metric or _metric()
    ref_block, _ = _reference_block()
    start = start or (f"{ref_block}__lstm_padrao" if ref_block else None)
    steps, block = [], final_block
    while block:
        champ = champion_name(block)
        base = common.load_json(os.path.join(results.block_dir(block), "base.json"), {})
        prev = base.get("origem_exp") or start
        prev_block = prev.split("__", 1)[0] if prev else None
        steps.append((block, champ, prev))
        block = prev_block if prev and prev != start and prev_block != block else None
    steps = steps[::-1]
    sd = noise_floor(metric, verbose=True)
    lower = M.is_lower_better(metric)
    rows = [{"passo": 0, "bloco": ref_block, "campeão": start, "mudanças": "ponto de partida (padrão do estudo)",
             f"{metric}": results.summarize(start)[f"{metric}_mean"] if results.summarize(start) else np.nan}]
    for i, (block, champ, prev) in enumerate(steps, 1):
        d = _paired(prev, champ, metric)
        s = results.summarize(champ)
        wins = int((d < 0).sum() if lower else (d > 0).sum())
        row = {"passo": i, "bloco": block, "campeão": champ, "mudanças": _param_diff(_exp_params(prev), _exp_params(champ)),
               f"{metric}": s[f"{metric}_mean"] if s else np.nan, "Δ vs anterior": d.mean() if len(d) else np.nan,
               "vitórias": f"{wins}/{len(d)}", "veredito": _beyond_noise(d.mean() if len(d) else np.nan, sd, metric)}
        if extra_metric:
            de = _paired(prev, champ, extra_metric)
            row[f"Δ {extra_metric}"] = de.mean() if len(de) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    _save_csv(df, f"cadeia_decisoes_{final_block}")
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(df)), 4))
    ax.plot(df["passo"], df[metric], "o-", color="tab:blue")
    if sd and np.isfinite(df[metric].iloc[0]):
        ax.axhspan(df[metric].iloc[0] - 2 * sd, df[metric].iloc[0] + 2 * sd, color="k", alpha=0.08,
                   label="ponto de partida ± 2× ruído entre seeds")
        ax.legend(fontsize=7)
    rw = _random_walk_reference(metric)
    if rw is not None:
        ax.axhline(rw, color="k", ls="--", lw=1)
        ax.text(0, rw, " passeio aleatório", va="bottom", fontsize=7)
    ax.set_xticks(df["passo"], [str(b).replace("lstm_", "") for b in df["bloco"]], rotation=30, ha="right")
    ax.set_ylabel(f"{metric} (validação, média nos K folds)")
    ax.set_title("Cadeia de decisões: campeão de cada bloco")
    _save(fig, f"cadeia_decisoes_{final_block}")
    plt.show()
    return df


def ablation_table(block, metric=None, extra_metric="val/pocid"):
    """Ablação do campeão: cada linha desfaz UMA mudança (volta ao padrão do estudo) e mede o Δ pareado.

    Δ = (sem a mudança) − (campeão). Para métricas de erro, Δ > 0 além do ruído = a mudança realmente ajuda;
    Δ dentro do ruído = a mudança é dispensável; Δ < 0 = a mudança atrapalhava (o padrão era melhor).
    """
    metric = metric or _metric()
    base = base_config_name(block)
    cfg = results.configs(block)
    sd = noise_floor(metric, verbose=True)
    lower = M.is_lower_better(metric)
    rows = []
    for _, c in cfg[cfg["exp_name"] != base].iterrows():
        d = _paired(base, c["exp_name"], metric)
        if not len(d):
            continue
        # Δ de "desfazer": piorar ao desfazer significa que a mudança ajuda
        helps = (d.mean() > 0) if lower else (d.mean() < 0)
        s_abl = results.summarize(c["exp_name"])
        if s_abl and s_abl.get("divergiu"):
            verdict = f"sem ela diverge ({s_abl['folds_divergentes']}/{s_abl['n_folds']} folds)"
        else:
            verdict = "ruído (dispensável)" if sd is not None and abs(d.mean()) <= 2 * sd else (
                "ajuda" if helps else "atrapalha")
        wins = int((d > 0).sum() if lower else (d < 0).sum())
        row = {"hiperparâmetro": c.get("desfeito"), "campeão": c.get("campeao"), "padrão": c.get("padrao"),
               f"Δ {metric} ao desfazer": d.mean(), "desvio": d.std(ddof=1) if len(d) > 1 else 0.0,
               "folds em que o campeão vence": f"{wins}/{len(d)}", "a mudança": verdict if sd is not None else "?"}
        if extra_metric:
            de = _paired(base, c["exp_name"], extra_metric)
            row[f"Δ {extra_metric} ao desfazer"] = de.mean() if len(de) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"{block}: sem resultados de ablação.")
        return df
    df = df.sort_values(f"Δ {metric} ao desfazer", ascending=not lower).reset_index(drop=True)
    _save_csv(df, f"ablacao_{block}")
    desc = cfg[cfg["exp_name"] == base]
    print(f"Campeão re-treinado: {base}. Δ = (sem a mudança) − (campeão).")
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(df) + 1.2))
    e = df.iloc[::-1]
    e = e[~e["a mudança"].str.startswith("sem ela diverge")]  # Δ gigante: fica só na tabela
    colors = ["tab:green" if v == "ajuda" else "tab:red" if v == "atrapalha" else "tab:gray" for v in e["a mudança"]]
    ax.barh([f"{h}: {c} → {p}" for h, c, p in zip(e["hiperparâmetro"], e["campeão"], e["padrão"])],
            e[f"Δ {metric} ao desfazer"], xerr=e["desvio"], color=colors)
    if sd:
        ax.axvspan(-2 * sd, 2 * sd, color="k", alpha=0.08, label="±2× ruído entre seeds")
        ax.legend(fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"Δ {metric} ao voltar para o padrão (verde = a mudança ajuda)")
    ax.set_title(f"Ablação do campeão ({len(desc) and desc['exp_name'].iloc[0]})")
    _save(fig, f"ablacao_{block}")
    plt.show()
    return df


# --------------------------------------------------------------------------------------------------
# Revisão (fase 2) e comparação entre fases
# --------------------------------------------------------------------------------------------------

KEY_PARAMS = ["target", "features", "lstm_activation", "optimizer", "lr", "num_layers", "hidden_size", "lookback"]


def divergence_summary(blocks):
    """Quantas configurações divergiram em cada bloco e quais hiperparâmetros elas têm em comum."""
    rows, details = [], []
    for b in blocks:
        df = results.ranking(b, "triagem")
        if df.empty:
            df = results.ranking(b, "final")
        if df.empty or "divergiu" not in df:
            continue
        rows.append({"bloco": b, "configurações": len(df), "divergiram": int(df["divergiu"].sum())})
        for e in df.loc[df["divergiu"], "exp_name"]:
            p = common.load_json(os.path.join(results.exp_dir(e), "parametros.json"), {}).get("params", {})
            details.append({"bloco": b, **{k: str(p.get(k)) for k in KEY_PARAMS}})
    per_block = pd.DataFrame(rows)
    det = pd.DataFrame(details)
    if not det.empty:
        print(f"{len(det)} configurações divergentes. Combinações de hiperparâmetros mais comuns entre elas:")
        pattern = det.groupby(["target", "lstm_activation", "optimizer", "lr"]).size().rename("n").reset_index()
        return per_block, pattern.sort_values("n", ascending=False).reset_index(drop=True)
    return per_block, det


def selection_check(blocks):
    """Com o critério de divergência: alguma configuração divergente chegou a uma final ou virou campeã?"""
    rows = []
    for b in blocks:
        fin = results.ranking(b, "final")
        champ = results.champion(b) or {}
        if fin.empty:
            continue
        s = results.summarize(champ.get("exp_name")) if champ.get("exp_name") else None
        rows.append({"bloco": b, "finalistas": len(fin), "finalistas divergentes": int(fin["divergiu"].sum()),
                     "campeão": str(champ.get("exp_name", "—")).split("__", 1)[-1],
                     "campeão divergiu": bool(s and s["divergiu"]),
                     "campeão = melhor não divergente": champ.get("exp_name") == fin.loc[~fin["divergiu"], "exp_name"].head(1).squeeze()
                     if (~fin["divergiu"]).any() else False})
    return pd.DataFrame(rows)


def compare_phases(metrics=("theil", "pocid", "da")):
    """Referência (passeio aleatório e LSTM padrão) e campeão principal de cada fase com treino: validação × teste."""
    active = os.environ.get("ESTUDO_CONFIG")
    rows = []
    try:
        for ph in common.phases():
            if not ph.get("blocos"):
                continue
            os.environ["ESTUDO_CONFIG"] = ph["estudo"]
            ref, _ = _reference_block()
            candidates = [("passeio aleatório", f"{ref}__passeio_aleatorio"), ("LSTM padrão", f"{ref}__lstm_padrao")]
            champ = results.champion(ph.get("campeao_principal", "")) if ph.get("campeao_principal") else None
            if champ:
                candidates.append(("campeão da fase", champ["exp_name"]))
            for role, exp in candidates:
                s = results.summarize(exp, splits=("val", "test")) if ref or role == "campeão da fase" else None
                if not s:
                    continue
                rows.append({"fase": ph["id"], "papel": role, "exp_name": exp.split("__", 1)[-1],
                             **{f"{sp}/{m}": s.get(f"{sp}/{m}_mean") for m in metrics for sp in ("val", "test")}})
    finally:
        if active is None:
            os.environ.pop("ESTUDO_CONFIG", None)
        else:
            os.environ["ESTUDO_CONFIG"] = active
    df = pd.DataFrame(rows)
    if not df.empty:
        _save_csv(df, "comparacao_fases")
    return df


# --------------------------------------------------------------------------------------------------
# Fusão de modelos (inspirada na Combinatorial Fusion Analysis de Wu et al., IEEE CAI 2025), sem vazamento
# --------------------------------------------------------------------------------------------------

def _stacked_predictions(exp, split):
    """Validação: cada dia previsto pelo modelo do fold que ainda não o viu (walk-forward, fora da amostra).
    Teste: média dos modelos dos K folds. Índice (data, ticker)."""
    frames = []
    for f in results.fold_results(exp):
        path = os.path.join(results.exp_dir(exp), "folds", f"fold_{f}", f"predicoes_{split}.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path, parse_dates=["data"]))
    if not frames:
        return None
    df = pd.concat(frames).groupby(["data", "ticker"], as_index=True).agg(
        previsto_lr=("previsto_lr", "mean"), real_lr=("real_lr", "first"), preco_t=("preco_t", "first"))
    return df.sort_index()


def _rsc(values):
    """Função rank-score (RSC): escores normalizados em [0, 1] ordenados do maior para o menor."""
    v = np.asarray(values, float)
    span = v.max() - v.min()
    return np.sort((v - v.min()) / span if span > 0 else np.zeros_like(v))[::-1]


def _combine(preds, weights, kind):
    """Combinação por escore (média ponderada das previsões) ou por rank (média ponderada das posições, mapeada de
    volta para retorno pela média das curvas de valores ordenados dos modelos)."""
    P = np.column_stack(preds)
    w = np.asarray(weights, float) / np.sum(weights)
    if kind == "escore":
        return P @ w
    ranks = np.column_stack([pd.Series(c).rank(method="average").to_numpy() - 1 for c in P.T])  # 0 = menor
    avg_rank = ranks @ w
    curve = np.mean([np.sort(c) for c in P.T], axis=0)  # valor típico de cada posição
    return np.interp(avg_rank, np.arange(len(curve)), curve)


def fusion_report(exps, max_size=5, metric=None):
    """Combina as previsões de vários modelos em todas as combinações de 2 a `max_size` modelos, com três pesos
    (média; desempenho = 1/MSE de validação; diversidade = força de diversidade cognitiva) e dois tipos (escore, rank).

    Os pesos vêm só da validação walk-forward (fora da amostra de cada fold); a melhor fusão é escolhida pela
    validação e só então o teste é mostrado. Diferente do paper, que escolhia a cada dia a combinação mais próxima do
    preço real (vazamento), aqui a escolha é única e feita antes de olhar o teste.
    """
    import itertools

    metric = metric or _metric()
    val, test = {}, {}
    for e in exps:
        v, t = _stacked_predictions(e, "val"), _stacked_predictions(e, "test")
        if v is not None and t is not None and not (results.summarize(e) or {}).get("divergiu", True):
            val[e], test[e] = v, t
    names = list(val)
    if len(names) < 2:
        print("Fusão: menos de 2 modelos com previsões válidas.")
        return pd.DataFrame()
    common_val = sorted(set.intersection(*[set(val[e].index) for e in names]))
    common_test = sorted(set.intersection(*[set(test[e].index) for e in names]))
    V = {e: val[e].loc[common_val] for e in names}
    T = {e: test[e].loc[common_test] for e in names}
    ref_v, ref_t = V[names[0]], T[names[0]]
    rsc = {e: _rsc(V[e]["previsto_lr"]) for e in names}
    cd = {(a, b): float(np.sqrt(np.mean((rsc[a] - rsc[b]) ** 2))) for a in names for b in names if a != b}
    mse_v = {e: float(np.mean((V[e]["previsto_lr"] - V[e]["real_lr"]) ** 2)) for e in names}

    def score(pred, ref, split):
        tick = np.zeros(len(ref), int)
        rows = np.arange(len(ref))  # dias consecutivos (uma série)
        return M.compute(pred, ref["real_lr"].to_numpy(), tick, ["BTC"], split, ref["preco_t"].to_numpy(), rows)

    rows = []
    for e in names:  # modelos individuais
        sv, st = score(V[e]["previsto_lr"].to_numpy(), ref_v, "val"), score(T[e]["previsto_lr"].to_numpy(), ref_t, "test")
        rows.append({"modelos": _short(e), "n": 1, "combinação": "individual",
                     **{f"val/{m}": sv[f"val/{m}"] for m in ("theil", "rmse", "pocid", "mape")},
                     **{f"test/{m}": st[f"test/{m}"] for m in ("theil", "rmse", "pocid", "mape")}})
    for k in range(2, min(max_size, len(names)) + 1):
        for group in itertools.combinations(names, k):
            ds = [np.mean([cd[(a, b)] for b in group if b != a]) for a in group]
            weights = {"média": [1.0] * k, "desempenho": [1 / mse_v[a] for a in group], "diversidade": ds}
            for wname, w in weights.items():
                if np.sum(w) <= 0:
                    continue
                for kind in ("escore", "rank"):
                    pv = _combine([V[a]["previsto_lr"].to_numpy() for a in group], w, kind)
                    pt = _combine([T[a]["previsto_lr"].to_numpy() for a in group], w, kind)
                    sv, st = score(pv, ref_v, "val"), score(pt, ref_t, "test")
                    rows.append({"modelos": " + ".join(_short(a) for a in group), "n": k, "combinação": f"{kind} · {wname}",
                                 **{f"val/{m}": sv[f"val/{m}"] for m in ("theil", "rmse", "pocid", "mape")},
                                 **{f"test/{m}": st[f"test/{m}"] for m in ("theil", "rmse", "pocid", "mape")}})
    df = pd.DataFrame(rows)
    col = metric if metric in df else "val/theil"
    df = df.sort_values(col, ascending=M.is_lower_better(col)).reset_index(drop=True)
    _save_csv(df, "fusao_modelos")
    best = df.iloc[0]
    print(f"{len(names)} modelos, {len(df) - len(names)} fusões. Escolhida pela validação: {best['modelos']} "
          f"({best['combinação']}): {col} {best[col]:.5f} → teste Theil {best['test/theil']:.4f}.")
    print("Diversidade cognitiva média de cada modelo (validação):",
          {_short(a): round(float(np.mean([cd[(a, b)] for b in names if b != a])), 4) for a in names})
    return df


def timegan_diagnostics(block):
    """Diagnósticos do TimeGAN por configuração (média entre folds): o gerador reproduz as propriedades do BTC?

    Compara, no canal de retorno, curtose (caudas pesadas), autocorrelação do retorno e do |retorno| (aglomerados de
    volatilidade) entre janelas reais e sintéticas, e mostra o score discriminativo (0 = indistinguíveis, 0,5 =
    trivialmente distinguíveis) e o Theil de validação. O TSTR é a linha "só sintético".
    """
    rows = []
    for e in results.configs(block)["exp_name"]:
        res = results.fold_results(e)
        diags = [r["timegan"] for r in res.values() if "timegan" in r]
        s = results.summarize(e)
        row = {"configuração": _short(e), "val/theil": s["val/theil_mean"] if s else np.nan,
               "val/pocid": s["val/pocid_mean"] if s else np.nan}
        if diags:
            row.update({"discriminativo": np.mean([d["discriminativo"] for d in diags]),
                        "curtose real": np.mean([d["real"]["curtose"] for d in diags]),
                        "curtose sint.": np.mean([d["sintetico"]["curtose"] for d in diags]),
                        "acf ret. real": np.mean([d["real"]["acf_retorno"] for d in diags]),
                        "acf ret. sint.": np.mean([d["sintetico"]["acf_retorno"] for d in diags]),
                        "acf |ret| real": np.mean([d["real"]["acf_abs_retorno"] for d in diags]),
                        "acf |ret| sint.": np.mean([d["sintetico"]["acf_abs_retorno"] for d in diags]),
                        "janelas reais": int(np.mean([d["n_real"] for d in diags])),
                        "segundos/fold": np.mean([d["segundos"] for d in diags])})
        rows.append(row)
    df = pd.DataFrame(rows)
    _save_csv(df, f"timegan_{block}")
    if "curtose sint." in df:
        print("Se o sintético reproduz o BTC, as colunas 'real' e 'sint.' ficam próximas e o discriminativo fica perto de 0.")
    return df
