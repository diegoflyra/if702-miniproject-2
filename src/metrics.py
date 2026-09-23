"""Métricas de previsão. Comparáveis entre alvos (`log_return` ou `close`): o modelo sempre é convertido para o
log-retorno previsto de h dias r̂ e para o preço previsto P̂(t+h) = P(t)·exp(r̂).

Erro (menor é melhor):
- mse, rmse, mae: erro no log-retorno de h dias;
- mape: erro percentual absoluto médio no PREÇO (%);
- theil: U de Theil = Σ(P − P̂)² / Σ(P(t+h) − P(t))², a razão entre o erro quadrático do modelo e o do passeio
  aleatório (prever que o preço não muda). theil < 1 = bate o passeio aleatório; theil ≈ 1 = não aprendeu nada.
Acerto (maior é melhor):
- pocid: Prediction Of Change In Direction (%), definição clássica: D_t = 1 se
  (P(t) − P(t−1))·(P̂(t) − P̂(t−1)) > 0, calculado entre dias consecutivos da mesma ação, com P e P̂ nos dias-alvo.
  Mede se a previsão sobe/desce junto com a cotação;
- da: acurácia direcional (%) na variante "previsto × preço atual": sinal(P̂(t+h) − P(t)) = sinal(P(t+h) − P(t)),
  só nos dias em que a previsão aposta numa direção (NaN para o passeio aleatório);
- skill: 1 − MSE/MSE_passeio_aleatório no retorno (> 0 = bate o passeio aleatório; ≈ 1 − theil);
- r2: 1 − SSE/SST no retorno; ic: correlação de Pearson entre retorno previsto e real.
"""
import numpy as np

METRICS = ["mse", "rmse", "mae", "mape", "theil", "pocid", "da", "skill", "r2", "ic"]
LOWER_IS_BETTER = {"loss", "mse", "rmse", "mae", "mape", "theil"}


def _core(pred, true, price, consecutive):
    err = pred - true
    mse = float(np.mean(err ** 2))
    sst = float(np.sum((true - true.mean()) ** 2))
    mse_rw = float(np.mean(true ** 2))
    p_true, p_pred = price * np.exp(true), price * np.exp(pred)
    rw_sse = float(np.sum((p_true - price) ** 2))
    moved = (pred != 0) & (true != 0)
    d = (np.diff(p_true) * np.diff(p_pred))[consecutive]
    ic = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 1e-12 and true.std() > 1e-12 else 0.0
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": float(np.mean(np.abs(err))),
        "mape": float(100 * np.mean(np.abs(p_true - p_pred) / np.abs(p_true))),
        "theil": float(np.sum((p_true - p_pred) ** 2)) / rw_sse if rw_sse > 0 else float("nan"),
        "pocid": float(100 * np.mean(d > 0)) if d.size else float("nan"),
        "da": float(100 * np.mean(np.sign(pred[moved]) == np.sign(true[moved]))) if moved.any() else float("nan"),
        "skill": 1 - mse / mse_rw if mse_rw > 0 else 0.0,
        "r2": 1 - float(np.sum(err ** 2)) / sst if sst > 0 else 0.0,
        "ic": ic,
    }


def compute(pred, true, tick_id, tickers, prefix, price, rows):
    """Métricas gerais e por ação: {'val/rmse': …, 'val/ticker/PETR4.SA/pocid': …}.

    `price` = P(t) de cada amostra; `rows` = linha da amostra na tabela empilhada (dias seguidos = linhas seguidas).
    """
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    price, rows, tick_id = np.asarray(price, np.float64), np.asarray(rows), np.asarray(tick_id)
    # pares (t−1, t) de dias seguidos na mesma ação: a POCID nunca emenda uma ação na seguinte
    consecutive = (np.diff(rows) == 1) & (np.diff(tick_id) == 0)
    out = {f"{prefix}/{k}": v for k, v in _core(pred, true, price, consecutive).items()}
    for t, name in enumerate(tickers):
        mask = tick_id == t
        if mask.sum() > 1:
            r = rows[mask]
            for k, v in _core(pred[mask], true[mask], price[mask], np.diff(r) == 1).items():
                out[f"{prefix}/ticker/{name}/{k}"] = v
    return out


def better(a, b, metric):
    """True se o valor `a` é melhor que `b` para a métrica (considera se menor é melhor)."""
    return a < b if is_lower_better(metric) else a > b


def is_lower_better(metric):
    return metric.split("/")[-1] in LOWER_IS_BETTER
