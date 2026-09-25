"""Expande todos os blocos de todas as fases (grids/_fases.json), sem treino: contagens, descartes e nomes únicos.

Blocos cujo campeão anterior ainda não existe usam o padrão do estudo como base provisória; blocos que transplantam o
campeão de outra fase (`configs_de`) só são contados depois que aquela fase rodou.
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

rows, names = [], set()
active = os.environ.get("ESTUDO_CONFIG")
try:
    for phase in common.phases():
        os.environ["ESTUDO_CONFIG"] = phase["estudo"]
        for block in phase["blocos"]:
            spec = grid_search.load_spec(os.path.join(ROOT, "grids", block + ".json"))
            assert spec["bloco"] == block, f"{block}.json declara bloco={spec['bloco']}"
            with redirect_stdout(io.StringIO()):
                base, origem = grid_search.resolve_base(spec, dry=True)
            row = {"fase": phase["id"], "bloco": block, "busca": spec["busca"].get("tipo", "grid")}
            if spec.get("checagem"):
                rows.append({**row, "busca": "checagem", "possíveis": "—", "a treinar": len(spec["checagem"]["posicoes"]),
                             "descartadas": 0})
                continue
            try:
                kept, discarded, total = grid_search.expand(spec, base)
            except RuntimeError as e:
                rows.append({**row, "possíveis": "—", "a treinar": len(spec.get("configs_de", [])), "descartadas": 0,
                             "obs": "depende de outra fase"})
                continue
            for c in kept:
                assert c["exp_name"] not in names, f"nome repetido: {c['exp_name']}"
                names.add(c["exp_name"])
            rows.append({**row, "possíveis": total or len(kept), "a treinar": len(kept), "descartadas": len(discarded)})
finally:
    if active is None:
        os.environ.pop("ESTUDO_CONFIG", None)
    else:
        os.environ["ESTUDO_CONFIG"] = active
print(pd.DataFrame(rows).fillna("").to_string(index=False))
print("OK: todos os blocos de todas as fases expandem sem erro.")
