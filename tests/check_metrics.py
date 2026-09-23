"""Verifica as métricas em casos com resposta conhecida (POCID, Theil, MAPE)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import metrics as M  # noqa: E402

prices = np.array([10.0, 11.0, 10.5, 10.8, 10.2, 10.9, 11.3])
price_t, true = prices[:-1], np.log(prices[1:] / prices[:-1])
rows, tick = np.arange(len(true)), np.zeros(len(true), int)

perfect = M.compute(true, true, tick, ["X"], "t", price_t, rows)
assert perfect["t/pocid"] == 100 and perfect["t/theil"] == 0 and perfect["t/mape"] == 0
rw = M.compute(np.zeros_like(true), true, tick, ["X"], "t", price_t, rows)
assert abs(rw["t/theil"] - 1) < 1e-12 and np.isnan(rw["t/da"])
inverse = M.compute(-true, true, tick, ["X"], "t", price_t, rows)
assert inverse["t/da"] == 0
# POCID conferida contra o cálculo manual da definição clássica
p_true = prices[1:]
p_hat = price_t * np.exp(-true)
manual = 100 * np.mean(np.diff(p_true) * np.diff(p_hat) > 0)
assert abs(inverse["t/pocid"] - manual) < 1e-12
# ações diferentes nunca formam par na POCID
two = M.compute(true.tolist() * 2, true.tolist() * 2, np.repeat([0, 1], len(true)), ["X", "Y"], "t",
                np.tile(price_t, 2), np.arange(2 * len(true)))
assert two["t/pocid"] == 100 and two["t/ticker/Y/pocid"] == 100
print(f"OK: POCID (perfeita 100%, invertida {inverse['t/pocid']:.0f}%), Theil (passeio aleatório = 1), MAPE, DA.")
