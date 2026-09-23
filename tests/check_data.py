"""Verifica as partições e a normalização (sem treino): nenhum alvo de treino enxerga a validação/teste."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import common  # noqa: E402
import data  # noqa: E402

p = common.default_params()
folds, test_start = data.fold_boundaries()
rows = []
for k, (t0, v0, v1) in folds.items():
    fd = data.FoldData(p, k)
    h = int(p["horizon"])
    info = {}
    for split in ("train", "val", "test"):
        r = fd.idx[split].numpy()
        assert np.all(fd.tick_id[r + h] == fd.tick_id[r]), "alvo cruzou para outra ação"
        dec, tgt = pd.to_datetime(fd.dates[r]), pd.to_datetime(fd.dates[r + h])
        info[split] = (dec.min(), dec.max(), tgt.max())
    assert info["train"][2] < v0, f"fold {k}: alvo de treino dentro da validação"
    assert info["val"][0] >= v0 and info["val"][2] < v1, f"fold {k}: validação fora do bloco"
    assert info["val"][2] < test_start, f"fold {k}: alvo de validação dentro do teste"
    assert info["test"][0] >= test_start, f"fold {k}: teste começa antes de {test_start.date()}"
    # normalização ajustada só no treino: nas linhas de treino, média ≈ 0 e desvio ≈ 1 (por ação)
    tr = fd.idx["train"].numpy()
    for t in range(len(fd.tickers)):
        rr = tr[fd.tick_id[tr] == t]
        block = fd.feats[rr.min(): rr.max() + 1].numpy()
        if p["scaler"] == "standard":
            assert np.allclose(block.mean(0), 0, atol=1e-3) and np.allclose(block.std(0), 1, atol=1e-2), "scaler não ajustado no treino"
    x = fd.windows(fd.idx["val"][:4])
    assert x.shape == (4, fd.lookback, fd.n_features)
    rows.append({"fold": k, "treino": f"{info['train'][0].date()} → {info['train'][1].date()}",
                 "validação": f"{info['val'][0].date()} → {info['val'][1].date()}",
                 "amostras treino/val/teste": f"{fd.size('train')}/{fd.size('val')}/{fd.size('test')}"})
print(pd.DataFrame(rows).to_string(index=False))
print("OK: partições sem sobreposição de alvos, embargo respeitado, normalização só no treino, janelas "
      f"[{p['lookback']} × {len(data.resolve_features(p['features']))}].")
