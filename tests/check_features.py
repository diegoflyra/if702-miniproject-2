"""Verifica as features (sem treino): sem valores faltando e sem olhar para o futuro.

Teste de antecipação: recalcula cada feature com os preços E as séries externas cortados numa data D. Se algum valor
até D mudar, a feature usou informação posterior a D (vazamento).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common  # noqa: E402
import data  # noqa: E402
import external  # noqa: E402

study = common.load_study()
ticker = study["dados"]["tickers"][0]
prices = data.load_prices(ticker)
names = [f for f in data.FEATURES if f not in ("price", "log_price")]
full = data._feature_columns(prices, study["dados"]["coluna_preco"], names)

cut = prices.index[len(prices) * 2 // 3]
orig_load = external.load
external.load = lambda source: orig_load(source).loc[:cut]  # o "futuro" das séries externas deixa de existir
try:
    trunc = data._feature_columns(prices.loc[:cut], study["dados"]["coluna_preco"], names)
finally:
    external.load = orig_load

bad = []
for n in names:
    a, b = full[n].loc[:cut], trunc[n]
    if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, atol=1e-12):
        bad.append(n)
assert not bad, f"features que usam informação futura: {bad}"

rows = []
for n in names:
    s = full[n]
    first = s.dropna().index.min()
    rows.append({"feature": n, "primeiro valor": str(first.date()) if first is not pd.NaT else "—",
                 "NaN após o início": int(s.loc[first:].isna().sum()) if first is not pd.NaT else len(s),
                 "média": round(float(s.mean()), 4), "desvio": round(float(s.std()), 4)})
print(pd.DataFrame(rows).to_string(index=False))
for name, feats in data.FEATURE_SETS.items():
    if name == "preco":
        continue
    frame = data.build_frame(ticker, data.resolve_features(name), "log_return", 1)
    assert not frame[data.resolve_features(name)].isna().any().any(), f"{name}: NaN depois do corte inicial"
print(f"OK: {len(names)} features sem antecipação (corte em {cut.date()}); todos os conjuntos sem NaN.")
