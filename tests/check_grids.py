"""Expande todos os blocos de grids/_ordem.json (sem treino): contagens, descartes e nomes únicos.

Blocos cujo campeão anterior ainda não existe usam o padrão do estudo como base provisória.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import common  # noqa: E402
import grid_search  # noqa: E402

order = common.load_json(os.path.join(ROOT, "grids", "_ordem.json"))
rows, names = [], set()
for block in order["blocos"]:
    spec = grid_search.load_spec(os.path.join(ROOT, "grids", block + ".json"))
    assert spec["bloco"] == block, f"{block}.json declara bloco={spec['bloco']}"
    with redirect_stdout(io.StringIO()):
        base, origem = grid_search.resolve_base(spec, dry=True)
    if spec.get("checagem"):
        rows.append({"bloco": block, "busca": "checagem", "possíveis": "—", "a treinar": len(spec["checagem"]["posicoes"]),
                     "descartadas": 0, "base": origem})
        continue
    kept, discarded, total = grid_search.expand(spec, base)
    for c in kept:
        assert c["exp_name"] not in names, f"nome repetido: {c['exp_name']}"
        names.add(c["exp_name"])
    rows.append({"bloco": block, "busca": spec["busca"].get("tipo", "grid"), "possíveis": total or len(kept),
                 "a treinar": len(kept), "descartadas": len(discarded), "base": origem})
print(pd.DataFrame(rows).to_string(index=False))
print("OK: todos os blocos expandem sem erro.")
