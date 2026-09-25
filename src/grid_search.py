"""Executa UM bloco do estudo: expande o espaço de busca, herda o campeão anterior, treina e escolhe o campeão.

Especificação do bloco (grids/*.json), campos principais:
  bloco          nome (prefixo dos experimentos e pasta em outputs/_grids/)
  herda_de       blocos cujo campeão é a base (o melhor entre eles); [] = config/estudo.json → padrao
  fixos          sobrescreve a base (ex.: mais épocas neste bloco)
  eixos          {param: [valores]}; um valor pode ser um dict que fixa vários params juntos, com "_rotulo"
  busca          {"tipo": "grid"} (produto cartesiano) ou {"tipo": "aleatoria", "n": 40, "seed": 0}
                 (n combinações distintas sorteadas do produto — permite espaços bem maiores)
  configs        lista de configurações explícitas (cada uma com "_nome"), além/no lugar dos eixos
  incluir_base   re-treina a base sem mudanças (referência pareada / reprodutibilidade)
  checagem       {"bloco_origem": b, "posicoes": [2, 3]}: aplica à base os valores dos eixos das
                 configurações nas posições pedidas do ranking final de b (checagem de interação)
  triagem        true (padrão): folds de triagem para todos, K folds para as n finalistas; false: K folds para todos
  n_finalistas   sobrescreve config/estudo.json → decisao.n_finalistas
  restricoes     {"max_parametros": N, "expr": ["hidden_size * num_layers <= 512"]}
  configs_de     [{"_nome", "estudo", "bloco", "sobrescrever"}]: campeão de um bloco de OUTRO estudo (transplante)
  ablacao        {"ignorar": [...]}: para cada hiperparâmetro em que a base difere do padrão do estudo, treina a base
                 com só aquele valor desfeito (quanto cada mudança contribui, medido de forma pareada)

Saída em outputs/_grids/{bloco}/: spec.json, base.json, configs.csv, descartadas.csv, ranking_triagem.csv,
ranking_final.csv, campeao.json, params/ (JSON de cada configuração) e logs/ (um log por job).

Uso: python src/grid_search.py grids/lstm_b1_capacidade.json [--workers_per_gpu 2] [--dry]
"""
import argparse
import copy
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment  # noqa: E402
import common  # noqa: E402
import data as data_mod  # noqa: E402
import models  # noqa: E402
import results  # noqa: E402

SRC = os.path.dirname(os.path.abspath(__file__))
RECURRENT_ONLY = {"cell", "hidden_size", "num_layers", "bidirectional", "pooling", "fc_neurons", "activation",
                  "lstm_activation", "weight_init", "layer_norm", "rnn_dropout", "input_dropout", "hidden_sizes",
                  "residual", "recurrent_dropout", "conv_layers", "conv_filters", "conv_kernel"}
TRAINING_ONLY = {"dropout", "optimizer", "lr", "momentum", "weight_decay", "scheduler", "decay_rate", "batch_size",
                 "grad_clip", "loss_fn", "huber_delta", "loss_lambda", "epochs", "patience", "monitor", "eval_train",
                 "augment", "aug_strength", "aug_prob"}
IGNORED_FOR_NAIVE = RECURRENT_ONLY | TRAINING_ONLY | {"scaler", "seed", "alvo_vol", "norm_janela"}


# --------------------------------------------------------------------------------------------------
# Especificação → lista de configurações
# --------------------------------------------------------------------------------------------------

def load_spec(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    spec.setdefault("herda_de", [])
    spec.setdefault("fixos", {})
    spec.setdefault("eixos", {})
    spec.setdefault("busca", {"tipo": "grid"})
    spec.setdefault("configs", [])
    spec.setdefault("triagem", True)
    spec.setdefault("restricoes", {})
    return spec


def resolve_base(spec, dry=False):
    """Base do bloco: o melhor campeão entre `herda_de` (ou o padrão do estudo)."""
    base = common.default_params()
    origem = "config/estudo.json → padrao"
    spec["_origem_exp"] = None
    if spec["herda_de"]:
        metric, mode, _, _ = common.decision()
        found = [c for c in (results.champion(b) for b in spec["herda_de"]) if c]
        if len(found) < len(spec["herda_de"]):
            missing = [b for b in spec["herda_de"] if not results.champion(b)]
            if not dry:
                raise RuntimeError(f"campeão ainda não definido para {missing}: rode esses blocos antes")
            print(f"[dry] sem campeão de {missing}; usando o padrão do estudo como base provisória")
        if found:
            best = sorted(found, key=lambda c: (bool(c.get("divergiu")),
                                                c["metricas"][f"{metric}_mean"] * (1 if mode == "min" else -1)))[0]
            base.update(best["params"])
            origem = f"campeão de {best['bloco']} ({best['exp_name']})"
            spec["_origem_exp"] = best["exp_name"]
    return base, origem


def _fmt(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v) if v else "0"
    return str(v)


def axis_label(value):
    return value["_rotulo"] if isinstance(value, dict) else _fmt(value)


def axis_updates(axis, value):
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if not k.startswith("_")}
    return {axis: value}


def _combinations(spec):
    """Combinações dos eixos. Grid: todas, em ordem. Aleatória: ordem sorteada (seed fixa); `expand` aceita as
    primeiras n válidas e não equivalentes, então descartes não consomem o orçamento da busca."""
    axes = spec["eixos"]
    names = list(axes)
    if not names:
        return [], 0
    sizes = [len(axes[n]) for n in names]
    total = math.prod(sizes)
    busca = spec["busca"]
    if busca.get("tipo", "grid") == "grid":
        picks = list(itertools.product(*[range(s) for s in sizes]))
    elif busca["tipo"] == "aleatoria":
        rng = np.random.default_rng(int(busca.get("seed", 0)))
        if total <= 1_000_000:
            flat = rng.permutation(total)
        else:
            flat = list(dict.fromkeys(int(i) for i in rng.integers(total, size=50 * int(busca["n"]))))
        picks = [tuple(int(i) for i in np.unravel_index(f, sizes)) for f in flat]
    else:
        raise ValueError(f"tipo de busca desconhecido: {busca['tipo']}")
    return [{n: axes[n][i] for n, i in zip(names, pick)} for pick in picks], total


def effective(p):
    """Forma canônica: parâmetros sem efeito são neutralizados, para detectar configurações equivalentes."""
    q = {k: v for k, v in copy.deepcopy(p).items() if not k.startswith("_")}
    if q["model"] in models.NAIVE_MODELS:
        for k in IGNORED_FOR_NAIVE:
            q.pop(k, None)
        return q
    if q["model"] in models.SKLEARN_MODELS:  # sem épocas nem rede: só entrada, alvo e normalização importam
        for k in RECURRENT_ONLY | TRAINING_ONLY:
            q.pop(k, None)
        return q
    if q["model"] == "cnn1d":
        for k in ("cell", "hidden_size", "num_layers", "bidirectional", "pooling", "lstm_activation", "rnn_dropout",
                  "recurrent_dropout", "hidden_sizes", "residual", "layer_norm", "conv_layers"):
            q.pop(k, None)
    if q["model"] == "linear":
        for k in RECURRENT_ONLY:
            q.pop(k, None)
    if q.get("hidden_sizes"):  # pilha explícita: substitui hidden_size × num_layers
        q.pop("hidden_size", None)
        q["num_layers"] = len(q["hidden_sizes"])
    if int(q.get("num_layers", 1)) == 1:
        q["rnn_dropout"] = 0.0  # dropout entre camadas recorrentes não existe com uma camada
    if q.get("cell", "lstm") != "lstm":
        q.pop("recurrent_dropout", None)
    if int(q.get("conv_layers", 0)) == 0 and q.get("model") != "cnn1d":
        q.pop("conv_filters", None)
        q.pop("conv_kernel", None)
    if q.get("loss_fn") != "huber":
        q.pop("huber_delta", None)
    if q.get("loss_fn") != "direcional":
        q.pop("loss_lambda", None)
    if q.get("target") != "log_return":
        q.pop("alvo_vol", None)
    if not q.get("fc_neurons"):
        q.pop("activation", None)  # sem camada densa, a ativação densa não é usada
    if q.get("cell", "lstm") != "lstm":
        q.pop("lstm_activation", None)
    if q.get("optimizer") in ("adam", "adamw"):
        q.pop("momentum", None)  # Adam não usa momentum explícito
    if q.get("augment", "none") in ("none", "", None):
        q.pop("aug_strength", None)  # sem augmentation, intensidade e probabilidade não têm efeito
        q.pop("aug_prob", None)
    if q.get("scheduler") != "exponential" or float(q.get("decay_rate", 1.0)) >= 1:
        q.pop("decay_rate", None)
        if q.get("scheduler") == "exponential":
            q["scheduler"] = "none"  # decay_rate = 1 ≡ sem agenda
    return q


def expand(spec, base):
    """Lista de configurações {exp_name, rotulos, params} e lista de descartadas {exp_name, motivo}."""
    block = spec["bloco"]
    base = {**base, **spec["fixos"]}
    candidates = []
    combos, total = _combinations(spec)
    for combo in combos:
        p = copy.deepcopy(base)
        labels = {}
        for axis, value in combo.items():
            p.update(axis_updates(axis, value))
            labels[axis] = axis_label(value)
        name = block + "__" + "__".join(f"{a}-{l}" for a, l in labels.items())
        candidates.append({"exp_name": name, "rotulos": labels, "params": p, "do_eixo": True})

    for extra in spec["configs"]:
        extra = copy.deepcopy(extra)
        name = extra.pop("_nome")
        p = {**copy.deepcopy(base), **extra}
        candidates.append({"exp_name": f"{block}__{name}", "rotulos": {"config": name}, "params": p})

    for item in spec.get("configs_de", []):
        # configuração campeã de OUTRO estudo (ex.: transplante do campeão da fase 1 para a série longa)
        other = os.path.join(common.study_output_dir(item["estudo"]), "_grids", item["bloco"], "campeao.json")
        champ = common.load_json(other)
        if champ is None:
            raise RuntimeError(f"{block}: campeão de {item['bloco']} ({item['estudo']}) não encontrado em {other}; "
                               "rode aquela fase antes (ou retome os resultados dela)")
        p = {**common.default_params(), **champ["params"], **item.get("sobrescrever", {})}
        candidates.append({"exp_name": f"{block}__{item['_nome']}", "rotulos": {"config": item["_nome"]}, "params": p})

    if spec.get("checagem"):
        chk = spec["checagem"]
        origin_spec = common.load_json(os.path.join(results.block_dir(chk["bloco_origem"]), "spec.json"))
        origin_rank = results.ranking(chk["bloco_origem"], "final")
        if origin_spec is None or origin_rank.empty:
            raise RuntimeError(f"checagem: bloco de origem {chk['bloco_origem']} ainda não rodou")
        keys = set()
        for axis, values in origin_spec["eixos"].items():
            for v in values:
                keys |= set(axis_updates(axis, v))
        for pos in chk["posicoes"]:
            if pos > len(origin_rank):
                continue
            src = common.load_json(os.path.join(results.block_dir(chk["bloco_origem"]), "params",
                                                origin_rank.loc[pos - 1, "exp_name"] + ".json"))
            p = copy.deepcopy(base)
            p.update({k: src[k] for k in keys if k in src})
            candidates.append({"exp_name": f"{block}__origem{pos}", "rotulos": {"origem": f"{pos}º de {chk['bloco_origem']}"},
                               "params": p})

    if spec.get("ablacao"):
        # desfaz, uma de cada vez, cada mudança do campeão em relação ao padrão do estudo
        ref = common.default_params()
        ignore = set(spec["ablacao"].get("ignorar", []))
        for k in sorted(base):
            if k in ref and k not in ignore and not k.startswith("_") and base[k] != ref[k]:
                p = copy.deepcopy(base)
                p[k] = ref[k]
                candidates.append({"exp_name": f"{block}__sem_{k}",
                                   "rotulos": {"desfeito": k, "campeao": _fmt(base[k]), "padrao": _fmt(ref[k])},
                                   "params": p})

    if spec.get("incluir_base"):
        candidates.insert(0, {"exp_name": f"{block}__base", "rotulos": {"config": "base"}, "params": copy.deepcopy(base)})

    # base, configs explícitas e checagens também ganham rótulo em cada eixo (entram nos heatmaps e no efeito marginal)
    for c in candidates:
        for axis, values in spec["eixos"].items():
            if axis not in c["rotulos"]:
                match = [v for v in values if all(c["params"].get(k) == x for k, x in axis_updates(axis, v).items())]
                c["rotulos"][axis] = axis_label(match[0]) if match else (
                    "outro" if isinstance(values[0], dict) else _fmt(c["params"].get(axis)))

    kept, discarded, seen = [], [], {}
    limits = spec["restricoes"]
    base_key = json.dumps(effective(base), sort_keys=True)
    budget = int(spec["busca"]["n"]) if spec["busca"].get("tipo") == "aleatoria" else None
    sampled = 0
    for c in sorted(candidates, key=lambda c: c.get("do_eixo", False)):  # base/extras primeiro
        if budget is not None and c.get("do_eixo") and sampled >= budget:
            break  # orçamento da busca aleatória completo; o resto nem foi sorteado
        p = c["params"]
        reason = _check(p, limits)
        key = json.dumps(effective(p), sort_keys=True)
        if reason is None and key in seen:
            reason = f"equivalente a {seen[key]}"
        if reason:
            discarded.append({"exp_name": c["exp_name"], **c["rotulos"], "motivo": reason})
            continue
        seen[key] = c["exp_name"]
        c["e_base"] = key == base_key
        if c.pop("do_eixo", False):
            sampled += 1
        kept.append(c)
    return kept, discarded, total


def _check(p, limits):
    try:
        n_feat = len(data_mod.resolve_features(p["features"]))
        n_params = models.describe(p, n_feat)["num_parameters"]
        augment.parse(p.get("augment", "none"))
    except Exception as e:  # noqa: BLE001
        return f"inválida: {e}"
    p["_num_parameters"] = n_params
    if limits.get("max_parametros") and n_params > float(limits["max_parametros"]):
        return f"{n_params:,} parâmetros > limite {int(limits['max_parametros']):,}"
    for expr in limits.get("expr", []):
        try:
            if not eval(expr, {"__builtins__": {}}, dict(p)):  # noqa: S307 (expressões do próprio grid)
                return f"restrição: {expr}"
        except Exception as e:  # noqa: BLE001
            return f"restrição inválida ({expr}): {e}"
    return None


# --------------------------------------------------------------------------------------------------
# Execução paralela: um processo por job, distribuídos entre as GPUs visíveis
# --------------------------------------------------------------------------------------------------

def _gpus():
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001
        return []


def run_jobs(jobs, block, workers_per_gpu, cpu_workers):
    """jobs: [(exp_name, params_path, folds)]. Pula o que já está concluído em disco."""
    logs = os.path.join(results.block_dir(block), "logs")
    os.makedirs(logs, exist_ok=True)
    pending = []
    for exp, ppath, folds in jobs:
        done = results.fold_results(exp)
        todo = [f for f in folds if f not in done]
        if todo:
            pending.append((exp, ppath, todo))
    if not pending:
        print(f"Nada a treinar: {len(jobs)} jobs já concluídos em disco.")
        return []
    gpus = _gpus()
    slots = [str(g) for g in gpus for _ in range(workers_per_gpu)] or [""] * cpu_workers
    free, running, failed = list(slots), {}, []
    threads = str(max(1, (os.cpu_count() or 1) // len(slots)))  # divide a CPU entre os processos paralelos
    total, finished, t0 = len(pending), 0, time.time()
    print(f"{total} jobs, {len(slots)} em paralelo ({'GPUs ' + ','.join(map(str, gpus)) if gpus else 'CPU'})", flush=True)
    queue = list(pending)
    while queue or running:
        while queue and free:
            exp, ppath, folds = queue.pop(0)
            slot = free.pop(0)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "TORCH_THREADS": threads, "OMP_NUM_THREADS": threads}
            if slot:
                env["CUDA_VISIBLE_DEVICES"] = slot
            log = open(os.path.join(logs, f"{exp}.log"), "a", encoding="utf-8")
            cmd = [sys.executable, os.path.join(SRC, "train.py"), "--params", ppath, "--exp_name", exp,
                   "--folds", ",".join(map(str, folds)), "--block", block]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running[proc] = (exp, slot, log, time.time())
        time.sleep(0.5)
        for proc in [p for p in running if p.poll() is not None]:
            exp, slot, log, start = running.pop(proc)
            log.close()
            free.append(slot)
            finished += 1
            status = "ok" if proc.returncode == 0 else f"FALHOU (código {proc.returncode})"
            if proc.returncode != 0:
                failed.append(exp)
            elapsed = (time.time() - t0) / 60
            print(f"[{finished}/{total}] {status} {exp} ({(time.time() - start) / 60:.1f} min; total {elapsed:.1f} min)", flush=True)
    for exp in failed:
        path = os.path.join(logs, f"{exp}.log")
        tail = open(path, encoding="utf-8").read().splitlines()[-15:]
        print(f"\n--- {exp} (últimas linhas de {path}) ---\n" + "\n".join(tail))
    return failed


# --------------------------------------------------------------------------------------------------
# Bloco completo: triagem → confirmação → campeão
# --------------------------------------------------------------------------------------------------

def run_block(spec_path, workers_per_gpu=1, cpu_workers=1, dry=False, max_configs=None, overrides=None):
    spec = load_spec(spec_path)
    block = spec["bloco"]
    bdir = results.block_dir(block)
    os.makedirs(os.path.join(bdir, "params"), exist_ok=True)
    if not dry:
        data_mod.prepare_prices(verbose=False)  # uma vez, antes dos workers (nada de downloads concorrentes)
    base, origem = resolve_base(spec, dry=dry)
    kept, discarded, total = expand(spec, base)
    if max_configs:
        kept = kept[:max_configs]
    if overrides:
        for c in kept:
            c["params"].update(overrides)

    metric, mode, triage_folds, n_final = common.decision()
    n_final = int(spec.get("n_finalistas", n_final))
    all_folds = list(range(1, common.n_folds() + 1))
    use_triage = spec["triagem"] and len(kept) > n_final
    spec["triagem_efetiva"] = use_triage
    common.save_json(os.path.join(bdir, "spec.json"), spec)
    common.save_json(os.path.join(bdir, "base.json"), {"origem": origem, "origem_exp": spec.get("_origem_exp"),
                                                       "params": base})
    rows = []
    for c in kept:
        n_params = c["params"].pop("_num_parameters", None)
        common.save_json(os.path.join(bdir, "params", c["exp_name"] + ".json"), c["params"])
        rows.append({"exp_name": c["exp_name"], **c["rotulos"], "e_base": c["e_base"], "num_parameters": n_params})
    pd.DataFrame(rows).to_csv(os.path.join(bdir, "configs.csv"), index=False)
    pd.DataFrame(discarded, columns=["exp_name", "motivo"] if not discarded else None).to_csv(
        os.path.join(bdir, "descartadas.csv"), index=False)

    busca = spec["busca"].get("tipo", "grid")
    print(f"Bloco {block}: base = {origem}")
    print(f"  busca {busca}: {len(kept)} configurações a treinar"
          + (f" (de {total} combinações possíveis)" if total else "")
          + (f", {len(discarded)} descartadas" if discarded else ""))
    print(f"  {'triagem nos folds ' + str(triage_folds) + f', {n_final} finalistas completam os {len(all_folds)} folds' if use_triage else f'todas rodam os {len(all_folds)} folds'}")
    if dry:
        return pd.DataFrame(rows)

    ppath = lambda c: os.path.join(bdir, "params", c["exp_name"] + ".json")  # noqa: E731
    first = triage_folds if use_triage else all_folds
    failed = run_jobs([(c["exp_name"], ppath(c), first) for c in kept], block, workers_per_gpu, cpu_workers)

    if use_triage:
        tri = results.ranking(block, "triagem")
        tri.to_csv(os.path.join(bdir, "ranking_triagem.csv"), index=False)
        finalists = list(tri.loc[~tri["divergiu"], "exp_name"].head(n_final)) or list(tri["exp_name"].head(n_final))
        n_div = int(tri["divergiu"].sum())
        if n_div:
            print(f"  {n_div} configurações divergiram na triagem (val/theil > {common.divergence_limit():g}); fora da final")
        # a referência pareada (base) sempre completa os K folds, se existir
        finalists += [c["exp_name"] for c in kept if c["e_base"] and c["exp_name"] not in finalists]
        print(f"Finalistas: {finalists}")
        failed += run_jobs([(e, os.path.join(bdir, "params", e + ".json"), all_folds) for e in finalists],
                           block, workers_per_gpu, cpu_workers)

    final = results.ranking(block, "final")
    final.to_csv(os.path.join(bdir, "ranking_final.csv"), index=False)
    if final.empty:
        raise RuntimeError(f"nenhuma configuração de {block} completou os {len(all_folds)} folds")
    best = final.iloc[0]
    if bool(best["divergiu"]):
        print(f"ATENÇÃO: todas as configurações finais de {block} divergiram; o campeão abaixo é só o menos ruim")
    champion = {"bloco": block, "exp_name": best["exp_name"], "divergiu": bool(best["divergiu"]),
                "params": common.load_json(os.path.join(bdir, "params", best["exp_name"] + ".json")),
                "metricas": {k: best[k] for k in final.columns if k.endswith(("_mean", "_std"))},
                "metrica_decisao": metric, "modo": mode, "falhas": failed}
    common.save_json(os.path.join(bdir, "campeao.json"), champion)
    print(f"Campeão de {block}: {best['exp_name']}  {metric} = {best[metric + '_mean']:.6f} ± {best[metric + '_std']:.6f}")
    try:  # painéis do W&B (hiperparâmetro × métrica); nunca derruba o estudo
        import wandb_report

        wandb_report.log_block(block)
    except Exception as e:  # noqa: BLE001
        print(f"[W&B] resumo do bloco não enviado: {e}")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--cpu_workers", type=int, default=1)
    parser.add_argument("--dry", action="store_true", help="só expande e valida o bloco, sem treinar")
    parser.add_argument("--max_configs", type=int, default=None, help="(teste) limita o nº de configurações")
    parser.add_argument("--epochs", type=int, default=None, help="(teste) sobrescreve epochs")
    args = parser.parse_args()
    overrides = {"epochs": args.epochs} if args.epochs else None
    run_block(args.spec, args.workers_per_gpu, args.cpu_workers, args.dry, args.max_configs, overrides)


if __name__ == "__main__":
    main()
