"""Treina UM experimento (uma configuração) nos folds pedidos e grava tudo em outputs/{exp_name}/.

Arquivos (gravados a cada época, antes de qualquer envio ao W&B):
  parametros.json                hiperparâmetros exatos, arquitetura, partições, dados (MD5), versões e commit
  historico_treino.csv           uma linha por fold × época: loss e métricas de treino/validação (gerais e por ação), gap
  historico_treino_agregado.csv  média e desvio entre folds, por época
  resultados.json                por fold: métricas na melhor época (validação) e do modelo restaurado (teste); médias
  melhor_modelo.pth              pesos do melhor fold (pela métrica de decisão); cada fold em folds/fold_k/
  folds/fold_k/predicoes_{val,test}.csv   previsão × real por dia e ação

Retomada: folds com folds/fold_k/done.json são pulados; um fold interrompido no meio recomeça do zero.
Uso: python src/train.py --params cfg.json --exp_name nome --folds 1,2,3 [--block nome_do_bloco]
"""
import argparse
import os
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augment  # noqa: E402
import common  # noqa: E402
import data as data_mod  # noqa: E402
import metrics as M  # noqa: E402
import models  # noqa: E402


# --------------------------------------------------------------------------------------------------
# W&B opcional: qualquer falha dele nunca interrompe o treino (o disco é a fonte da verdade)
# --------------------------------------------------------------------------------------------------

class Tracker:
    """Uma run do W&B por experimento (configuração): grupo = bloco, config = hiperparâmetros.

    Cada fold loga em `fold{k}/...` com seu próprio eixo de épocas; o resumo final traz as médias entre folds
    (`val/<métrica>_mean`), o que alimenta os painéis de importância de parâmetros e coordenadas paralelas do W&B.
    O id da run é fixo por (RUN_NAME, experimento): uma execução retomada continua a mesma run.
    """

    def __init__(self, exp_name, params, block):
        self.run = None
        if os.environ.get("WANDB_MODE") == "disabled" or not os.environ.get("WANDB_API_KEY"):
            return
        try:
            import hashlib

            import wandb

            os.environ.setdefault("WANDB_SILENT", "true")
            run_id = hashlib.md5(f"{os.environ.get('RUN_NAME', 'local')}/{exp_name}".encode()).hexdigest()[:16]
            self.run = wandb.init(project=os.environ.get("WANDB_PROJECT", "if702-miniproject-2-lstm"),
                                  entity=os.environ.get("WANDB_ENTITY") or None, id=run_id, resume="allow",
                                  group=block or "avulso", job_type=block or "avulso", name=exp_name,
                                  config={**params, "bloco": block, "run_name": os.environ.get("RUN_NAME")})
        except Exception as e:  # noqa: BLE001
            print(f"[W&B] desativado para este experimento: {e}")

    def log(self, row):
        if not self.run:
            return
        try:
            f = row["fold"]
            self.run.define_metric(f"fold{f}/*", step_metric=f"fold{f}/epoch")
            self.run.log({f"fold{f}/{k}": v for k, v in row.items()
                          if isinstance(v, (int, float)) and k != "fold" and "/ticker/" not in k})
        except Exception:  # noqa: BLE001
            pass

    def finish(self, summary):
        if not self.run:
            return
        try:
            if os.environ.get("WANDB_LOG_TEST") != "1":  # o teste só é revelado no relatório final
                summary = {k: v for k, v in summary.items() if not k.startswith("test/")}
            self.run.summary.update({k: v for k, v in summary.items() if "/ticker/" not in k})
            self.run.finish()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------------
# Treino e avaliação
# --------------------------------------------------------------------------------------------------

def make_optimizer(model, p):
    kind, lr, wd = p["optimizer"], float(p["lr"]), float(p["weight_decay"])
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(p["momentum"]), weight_decay=wd)
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=float(p["momentum"]), weight_decay=wd)
    raise ValueError(f"otimizador desconhecido: {kind}")


def make_scheduler(opt, p):
    """`exponential`: lr ← lr·decay_rate a cada época (decay_rate = 1 desliga; 0,97 é o ponto de partida)."""
    kind = p.get("scheduler", "none")
    if kind == "none":
        return None
    if kind == "exponential":
        rate = float(p.get("decay_rate", 1.0))
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=rate) if rate < 1 else None
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, int(p["patience"]) // 3))
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(p["epochs"]))
    raise ValueError(f"scheduler desconhecido: {kind}")


def make_loss(name):
    return {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.SmoothL1Loss(beta=1.0)}[name]


@torch.no_grad()
def predict(model, fd, split, batch=4096):
    model.eval()
    rows = fd.idx[split]
    out = [model(fd.windows(rows[i:i + batch])) for i in range(0, rows.numel(), batch)]
    return rows, torch.cat(out) if out else torch.empty(0, device=fd.device)


@torch.no_grad()
def evaluate(model, fd, split, loss_fn):
    rows, y_hat = predict(model, fd, split)
    loss = float(loss_fn(y_hat, fd.y[rows]))
    rows_np = rows.cpu().numpy()
    pred_lr = fd.to_log_return(rows_np, y_hat.cpu().numpy())
    out = {f"{split}/loss": loss}
    out.update(M.compute(pred_lr, fd.lr_h[rows_np], fd.tick_id[rows_np], fd.tickers, split, fd.price[rows_np], rows_np))
    return out, rows_np, pred_lr


def naive_predict(p, fd, split):
    rows = fd.idx[split].cpu().numpy()
    kind, h = p["model"], int(p["horizon"])
    if kind == "naive_zero":
        pred = np.zeros(len(rows))
    elif kind == "naive_mean":
        means = fd.train_mean_log_return()
        pred = np.array([means[t] for t in fd.tick_id[rows]])
    elif kind == "naive_last":
        prev = np.clip(rows - h, 0, None)
        same = fd.tick_id[prev] == fd.tick_id[rows]
        pred = np.where(same, np.log(fd.price[rows] / fd.price[prev]), 0.0)
    else:
        raise ValueError(kind)
    out = M.compute(pred, fd.lr_h[rows], fd.tick_id[rows], fd.tickers, split, fd.price[rows], rows)
    return out, rows, pred


def save_predictions(path, fd, rows, pred):
    pd.DataFrame({"data": pd.to_datetime(fd.dates[rows]), "ticker": np.array(fd.tickers)[fd.tick_id[rows]],
                  "preco_t": fd.price[rows], "previsto_lr": pred, "real_lr": fd.lr_h[rows]}).to_csv(path, index=False)


def add_gap(row):
    for key in ["loss"] + M.METRICS:
        if f"train/{key}" in row and f"val/{key}" in row:
            row[f"gap/{key}"] = row[f"val/{key}"] - row[f"train/{key}"]
    return row


def append_history(path, rows):
    df = pd.DataFrame(rows)
    if os.path.isfile(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def drop_fold_history(path, fold):
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df[df["fold"] != fold].to_csv(path, index=False)


def run_fold(p, fold, out_dir, block, device, tracker):
    fold_dir = os.path.join(out_dir, "folds", f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    hist_path = os.path.join(out_dir, "historico_treino.csv")
    drop_fold_history(hist_path, fold)  # fold interrompido: recomeça do zero
    common.set_seed(int(p["seed"]))
    t_start = time.time()
    fd = data_mod.FoldData(p, fold, device=device)
    decision_metric = common.decision()[0]

    if p["model"] in models.NAIVE_MODELS:
        tr, _, _ = naive_predict(p, fd, "train")
        va, v_rows, v_pred = naive_predict(p, fd, "val")
        te, t_rows, t_pred = naive_predict(p, fd, "test")
        row = add_gap({"fold": fold, "epoch": 0, **tr, **va})
        append_history(hist_path, [row])
        tracker.log(row)
        best_epoch, epochs_run, best_row, train_final = 0, 0, row, tr
    else:
        model = models.build_model(fd.n_features, p).to(device)
        opt = make_optimizer(model, p)
        sched = make_scheduler(opt, p)
        loss_fn = make_loss(p["loss_fn"])
        monitor = p["monitor"]
        best_val, best_epoch, best_row, bad = None, 0, None, 0
        ckpt = os.path.join(fold_dir, "modelo.pth")
        bs, n_train = int(p["batch_size"]), fd.size("train")
        epochs_run = 0
        for epoch in range(1, int(p["epochs"]) + 1):
            model.train()
            t0 = time.time()
            perm = fd.idx["train"][torch.randperm(n_train, device=device)]
            running, seen = 0.0, 0
            for i in range(0, n_train, bs):
                rows = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                xb, yb = augment.apply(fd.windows(rows), fd.y[rows], p)  # só no treino; "none" = sem mudança
                loss = loss_fn(model(xb), yb)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                if p.get("grad_clip"):
                    nn.utils.clip_grad_norm_(model.parameters(), float(p["grad_clip"]))
                opt.step()
                running += loss.item() * rows.numel()
                seen += rows.numel()
            row = {"fold": fold, "epoch": epoch, "lr_atual": opt.param_groups[0]["lr"],
                   "train/loss_batches": running / max(seen, 1)}
            if p.get("eval_train", True):  # métricas de treino em modo avaliação (sem dropout)
                row.update(evaluate(model, fd, "train", loss_fn)[0])
            row.update(evaluate(model, fd, "val", loss_fn)[0])
            row["tempo_epoca_s"] = time.time() - t0
            add_gap(row)
            append_history(hist_path, [row])
            tracker.log(row)
            epochs_run = epoch
            current = row[monitor]
            diverged = not np.isfinite(current)
            if not diverged and (best_val is None or M.better(current, best_val, monitor)):
                best_val, best_epoch, best_row, bad = current, epoch, row, 0
                torch.save(model.state_dict(), ckpt)
            else:
                bad += 1
            if sched is not None:
                sched.step(current) if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau) else sched.step()
            if diverged or bad >= int(p["patience"]):
                break
        if best_row is None:  # divergiu já na 1ª época: registra o que houver, sem pesos
            best_row = row
            torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        train_final = evaluate(model, fd, "train", loss_fn)[0]
        _, v_rows, v_pred = evaluate(model, fd, "val", loss_fn)
        te, t_rows, t_pred = evaluate(model, fd, "test", loss_fn)

    va = {k: v for k, v in best_row.items() if k.startswith(("val/", "gap/"))}
    save_predictions(os.path.join(fold_dir, "predicoes_val.csv"), fd, v_rows, v_pred)
    save_predictions(os.path.join(fold_dir, "predicoes_test.csv"), fd, t_rows, t_pred)
    result = {"fold": fold, "melhor_epoca": best_epoch, "epocas_treinadas": epochs_run,
              "amostras": {s: fd.size(s) for s in ("train", "val", "test")},
              "tempo_s": time.time() - t_start,
              "train": {k: v for k, v in train_final.items() if k.startswith("train/")},
              "val": va, "test": {k: v for k, v in te.items() if k.startswith("test/")}}
    common.save_json(os.path.join(fold_dir, "resultado.json"), result)
    common.save_json(os.path.join(fold_dir, "done.json"), {"fold": fold, "concluido_em": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"  fold {fold}: {decision_metric}={va.get(decision_metric, float('nan')):.6f} "
          f"melhor época {best_epoch}/{epochs_run} ({result['tempo_s']:.0f}s)", flush=True)
    return result


def aggregate(out_dir):
    """Reconstrói resultados.json, o histórico agregado e melhor_modelo.pth a partir dos folds concluídos."""
    decision_metric, mode, _, _ = common.decision()
    fold_results = {}
    for d in sorted(os.listdir(os.path.join(out_dir, "folds"))):
        res = common.load_json(os.path.join(out_dir, "folds", d, "resultado.json"))
        if res and os.path.isfile(os.path.join(out_dir, "folds", d, "done.json")):
            fold_results[int(res["fold"])] = res
    summary = {}
    if fold_results:
        for split in ("train", "val", "test"):
            keys = sorted({k for r in fold_results.values() for k in r[split]})
            for k in keys:
                vals = [r[split][k] for r in fold_results.values() if k in r[split]]
                summary[f"{k}_mean"] = float(np.mean(vals))
                summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary["melhor_epoca_media"] = float(np.mean([r["melhor_epoca"] for r in fold_results.values()]))
        summary["epocas_treinadas_media"] = float(np.mean([r["epocas_treinadas"] for r in fold_results.values()]))
        summary["tempo_total_s"] = float(sum(r["tempo_s"] for r in fold_results.values()))
        sign = 1 if mode == "min" else -1
        best_fold = min(fold_results, key=lambda f: sign * fold_results[f]["val"].get(decision_metric, np.inf * sign))
        summary["melhor_fold"] = best_fold
        src = os.path.join(out_dir, "folds", f"fold_{best_fold}", "modelo.pth")
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, "melhor_modelo.pth"))
    common.save_json(os.path.join(out_dir, "resultados.json"), {
        "exp_name": os.path.basename(out_dir), "folds_concluidos": sorted(fold_results),
        "folds": {str(k): v for k, v in sorted(fold_results.items())}, "media": summary})

    hist_path = os.path.join(out_dir, "historico_treino.csv")
    if os.path.isfile(hist_path):
        hist = pd.read_csv(hist_path)
        hist = hist[hist["fold"].isin(fold_results)]
        if len(hist):
            num = hist.drop(columns=["fold"]).groupby("epoch")
            agg = pd.concat([num.mean().add_suffix("_mean"), num.std().add_suffix("_std"),
                             num.size().rename("n_folds")], axis=1)
            agg.reset_index().to_csv(os.path.join(out_dir, "historico_treino_agregado.csv"), index=False)
    return summary


def write_params(p, out_dir, block):
    features = data_mod.resolve_features(p["features"])
    folds, test_start = data_mod.fold_boundaries()
    common.save_json(os.path.join(out_dir, "parametros.json"), {
        "exp_name": os.path.basename(out_dir), "bloco": block, "params": p,
        "features_resolvidas": features,
        "arquitetura": models.describe(p, len(features)),
        "particoes": {str(k): [str(d.date()) for d in v] for k, v in folds.items()},
        "teste_inicio": str(test_start.date()),
        "estudo": common.load_study(), "dados_manifesto": data_mod.data_manifest(),
        "versoes": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "commit": common.git_commit()})


def run_experiment(p, exp_name, folds, block=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(common.output_dir(), exp_name)
    os.makedirs(os.path.join(out_dir, "folds"), exist_ok=True)
    write_params(p, out_dir, block)
    todo = [f for f in folds if not os.path.isfile(os.path.join(out_dir, "folds", f"fold_{f}", "done.json"))]
    skipped = sorted(set(folds) - set(todo))
    print(f"[{exp_name}] device={device} folds={folds}" + (f" (já concluídos: {skipped})" if skipped else ""), flush=True)
    n_params = models.describe(p, len(data_mod.resolve_features(p["features"])))["num_parameters"]
    wb_config = {**p, "num_parameters": n_params, "fc_neurons_rotulo": "-".join(map(str, p.get("fc_neurons", []))) or "0"}
    tracker = Tracker(exp_name, wb_config, block) if todo else None
    for fold in todo:
        run_fold(p, fold, out_dir, block, device, tracker)
        aggregate(out_dir)
    summary = aggregate(out_dir)
    if tracker:
        tracker.finish(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True, help="JSON com os hiperparâmetros completos")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--folds", required=True, help="ex.: 1,2,3")
    parser.add_argument("--block", default=None)
    args = parser.parse_args()
    if os.environ.get("TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["TORCH_THREADS"]))
    p = {**common.default_params(), **common.load_json(args.params)}
    run_experiment(p, args.exp_name, [int(f) for f in args.folds.split(",")], args.block)


if __name__ == "__main__":
    main()
