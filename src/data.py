"""Dados: obtenção/cache das séries de preços (CSV local, yfinance ou sintética), features, partições walk-forward e janelas para o LSTM.

Garantias contra vazamento (verificadas em tests/check_data.py):
- as partições são por DATA: o fold k treina em tudo antes do bloco de validação k e valida no bloco k;
  o teste é o período final (config/estudo.json → particoes.teste_inicio) e nunca entra em escolha;
- uma amostra decidida no dia t tem alvo em t+h; ela só pertence a uma partição se t+h também cair nela
  (embargo de h dias entre treino e validação), então nenhum alvo de treino enxerga a validação;
- normalização de features e do alvo é ajustada, por ação, só nas linhas de treino do fold;
- as janelas de entrada podem usar dias anteriores ao início da partição (é passado, não vazamento).

Uso direto:  python src/data.py --preparar   (localiza a origem e gera os CSVs normalizados em data/precos/)
"""
import argparse
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

PRICES_SUBDIR = "precos"
OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# --------------------------------------------------------------------------------------------------
# Preços: cópia local primeiro, download só se não houver
# --------------------------------------------------------------------------------------------------

def prices_dir():
    return os.path.join(common.DATA_DIR, PRICES_SUBDIR)


def _safe_name(ticker):
    return ticker.replace("/", "_").replace("^", "IDX_")


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _search_roots():
    roots = [os.environ.get("PRICES_PATH", "").strip(), prices_dir(), "/kaggle/input",
             "/content/precos", "/content/data/precos"]
    return [r for r in roots if r and os.path.exists(r)]


def _find_csv(ticker):
    name = _safe_name(ticker) + ".csv"
    for root in _search_roots():
        direct = os.path.join(root, name)
        if os.path.isfile(direct):
            return direct
        if root.startswith("/kaggle/input"):
            hits = glob.glob(os.path.join(root, "**", name), recursive=True)
            if hits:
                return hits[0]
    return None


def _normalize_frame(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.strip().title() for c in df.columns})
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df.index.name = "Date"
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes no CSV de preços: {missing}")
    return df[OHLCV].sort_index().dropna(subset=["Adj Close"])


def _download_yfinance(ticker, start, end):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"yfinance não retornou dados para {ticker} ({start} → {end})")
    return _normalize_frame(df)


def _synthetic(ticker, start, end):
    """Série sintética com clusters de volatilidade e leve autocorrelação: só para testes sem internet."""
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    vol = np.empty(n)
    ret = np.empty(n)
    vol[0], ret[0] = 0.015, 0.0
    for t in range(1, n):
        vol[t] = np.sqrt(1e-6 + 0.08 * ret[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)
        ret[t] = 0.08 * ret[t - 1] + vol[t] * rng.standard_normal()
    close = 20 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(0.3 * vol * rng.standard_normal(n))
    high = np.maximum(open_, close) * np.exp(np.abs(0.5 * vol * rng.standard_normal(n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.5 * vol * rng.standard_normal(n)))
    volume = np.round(1e6 * np.exp(0.3 * rng.standard_normal(n) + 20 * np.abs(ret)))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close,
                       "Volume": volume}, index=pd.DatetimeIndex(dates, name="Date"))
    return df


def _find_source_csv(name):
    """Localiza o CSV de origem (fonte `csv`) em qualquer ambiente, na ordem:
    DATA_PATH (arquivo ou pasta) → caminho do config (relativo ao repositório) → Kaggle Input → Colab (/content)
    → se nada casar pelo nome, o único CSV do Kaggle Input que tenha colunas de data e fechamento.
    """
    base = os.path.basename(name)
    explicit = os.environ.get("DATA_PATH", "").strip()
    candidates = []
    if explicit:
        candidates += [explicit, os.path.join(explicit, base)]
    candidates += [name if os.path.isabs(name) else os.path.join(common.REPO_DIR, name)]
    for c in candidates:
        if os.path.isfile(c):
            return c
    for root in ("/kaggle/input", "/content"):
        if os.path.isdir(root):
            depth = "**" if root == "/kaggle/input" else "*"  # /content/drive é grande demais para busca recursiva
            hits = glob.glob(os.path.join(root, depth, base), recursive=True) + glob.glob(os.path.join(root, base))
            if hits:
                return sorted(hits)[0]
    if os.path.isdir("/kaggle/input"):
        usable = []
        for path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
            cols = {c.strip().lower() for c in pd.read_csv(path, nrows=1).columns}
            if "date" in cols and "close" in cols:
                usable.append(path)
        if len(usable) == 1:
            return usable[0]
    raise FileNotFoundError(
        f"CSV de dados não encontrado: '{base}'. No Kaggle, anexe o dataset (Add Input); no Colab/local, coloque o "
        f"arquivo na raiz do repositório ou defina DATA_PATH com o caminho do arquivo ou da pasta.")


def prepare_prices(verbose=True):
    """Gera um CSV normalizado por série em data/precos/ e grava um manifesto com o MD5 da origem e da cópia.

    Fonte `csv`: a cópia é refeita a cada execução a partir do arquivo de origem (nunca fica desatualizada).
    Fontes `yfinance`/`synthetic`: a cópia já congelada é reaproveitada (o yfinance pode revisar séries antigas);
    senão PRICES_PATH / Kaggle Input / Colab → download.
    O manifesto entra no parametros.json de cada experimento, amarrando o resultado aos dados exatos.
    """
    dados = common.load_study()["dados"]
    os.makedirs(prices_dir(), exist_ok=True)
    manifest = {}
    for ticker in dados["tickers"]:
        target = os.path.join(prices_dir(), _safe_name(ticker) + ".csv")
        origem, md5_origem = "cópia congelada", None
        if dados["fonte"] == "csv":
            path = _find_source_csv(dados["arquivos"][ticker])
            df = _normalize_frame(pd.read_csv(path).rename(columns=dados.get("colunas", {})))
            origem, md5_origem = path, _md5(path)
            df.loc[dados["inicio"]:dados["fim"]].to_csv(target + ".tmp", float_format="%.6f")
            os.replace(target + ".tmp", target)
        elif not os.path.isfile(target):
            found = _find_csv(ticker)
            if found:
                df = _normalize_frame(pd.read_csv(found))
                origem = f"copiado de {found}"
            elif dados["fonte"] == "synthetic":
                df = _synthetic(ticker, dados["inicio"], dados["fim"])
                origem = "sintético"
            else:
                df = _download_yfinance(ticker, dados["inicio"], dados["fim"])
                origem = "download yfinance"
            df.loc[dados["inicio"]:dados["fim"]].to_csv(target, float_format="%.6f")
        df = pd.read_csv(target, index_col="Date", parse_dates=True)
        manifest[ticker] = {"arquivo": os.path.relpath(target, common.DATA_DIR), "md5": _md5(target),
                            "origem": origem, "md5_origem": md5_origem, "linhas": len(df),
                            "primeira_data": str(df.index.min().date()), "ultima_data": str(df.index.max().date())}
        if verbose:
            m = manifest[ticker]
            print(f"{ticker:>12}: {m['linhas']} dias, {m['primeira_data']} → {m['ultima_data']}"
                  f" | origem: {origem}" + (f" (md5 {md5_origem})" if md5_origem else ""))
    common.save_json(os.path.join(prices_dir(), "manifest.json"), manifest)
    return manifest


def load_prices(ticker):
    path = os.path.join(prices_dir(), _safe_name(ticker) + ".csv")
    if not os.path.isfile(path):
        prepare_prices(verbose=False)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    dados = common.load_study()["dados"]
    return df.loc[dados["inicio"]:dados["fim"]]


def data_manifest():
    return common.load_json(os.path.join(prices_dir(), "manifest.json"), {})


# --------------------------------------------------------------------------------------------------
# Features e alvo
# --------------------------------------------------------------------------------------------------

def _rsi(p, n=14):
    delta = p.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def _feature_columns(df, price_col):
    p = df[price_col]
    lr = np.log(p).diff()
    logv = np.log1p(df["Volume"])
    return {
        "price": p,
        "log_price": np.log(p),
        "log_return": lr,
        "hl_range": np.log(df["High"] / df["Low"]),
        "oc_range": np.log(df["Close"] / df["Open"]),
        "volume_change": logv.diff(),
        "volume_z20": (logv - logv.rolling(20).mean()) / logv.rolling(20).std(),
        "volatility_20": lr.rolling(20).std(),
        "ma_ratio_10": p / p.rolling(10).mean() - 1,
        "ma_ratio_50": p / p.rolling(50).mean() - 1,
        "rsi_14": _rsi(p) / 100 - 0.5,
    }


FEATURES = ["price", "log_price", "log_return", "hl_range", "oc_range", "volume_change", "volume_z20",
            "volatility_20", "ma_ratio_10", "ma_ratio_50", "rsi_14"]

FEATURE_SETS = {
    "preco": ["price"],
    "retornos": ["log_return"],
    "retornos_volume": ["log_return", "volume_change"],
    "ohlcv": ["log_return", "hl_range", "oc_range", "volume_change"],
    "tecnicos": ["log_return", "hl_range", "oc_range", "volume_change", "volatility_20",
                 "ma_ratio_10", "ma_ratio_50", "rsi_14"],
}

TARGETS = ["log_return", "close"]


def resolve_features(spec):
    names = FEATURE_SETS[spec] if isinstance(spec, str) else list(spec)
    unknown = [n for n in names if n not in FEATURES]
    if isinstance(spec, str) and spec not in FEATURE_SETS:
        raise ValueError(f"conjunto de features desconhecido: {spec} (opções: {list(FEATURE_SETS)})")
    if unknown:
        raise ValueError(f"features desconhecidas: {unknown} (opções: {FEATURES})")
    return names


def build_frame(ticker, features, target, horizon):
    """Tabela diária de uma ação: features no dia t (só informação até t) e alvo em t+h.

    Colunas extras: `price` (preço em t), `lr_h` (log-retorno de t a t+h, a grandeza em que TODAS
    as métricas são calculadas, qualquer que seja o alvo do modelo) e `y` (alvo bruto do modelo).
    """
    price_col = common.load_study()["dados"]["coluna_preco"]
    df = load_prices(ticker)
    cols = _feature_columns(df, price_col)
    out = pd.DataFrame({name: cols[name] for name in features}, index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan).ffill()
    p = df[price_col]
    out["_price"] = p
    out["_lr_h"] = np.log(p.shift(-horizon) / p)
    out["_y"] = out["_lr_h"] if target == "log_return" else p.shift(-horizon)
    first_valid = out[features].dropna().index.min()
    return out.loc[first_valid:]


# --------------------------------------------------------------------------------------------------
# Partições walk-forward (por data, comuns a todas as ações e a todos os experimentos)
# --------------------------------------------------------------------------------------------------

def calendar():
    dados = common.load_study()["dados"]
    dates = None
    for ticker in dados["tickers"]:
        idx = load_prices(ticker).index
        if dates is None:
            dates = idx
        elif dados.get("alinhamento", "intersecao") == "intersecao":
            dates = dates.intersection(idx)
        else:
            dates = dates.union(idx)
    return dates.sort_values()


def fold_boundaries():
    """Datas de corte de cada fold: {k: (treino_inicio, val_inicio, val_fim_exclusivo)} e o início do teste.

    Com `tamanho_validacao_dias`, os K blocos de validação são os últimos K×tamanho dias antes do teste;
    sem ele, o desenvolvimento é dividido em K+1 partes iguais (como o TimeSeriesSplit do scikit-learn).
    `tipo=sliding` limita o treino aos últimos `janela_treino_max_dias` dias antes da validação.
    """
    part = common.load_study()["particoes"]
    cal = calendar()
    test_start = pd.Timestamp(part["teste_inicio"])
    dev = cal[cal < test_start]
    k = int(part["n_folds"])
    size = part.get("tamanho_validacao_dias") or len(dev) // (k + 1)
    first_val = len(dev) - k * size
    if first_val < size:
        raise ValueError(f"período de desenvolvimento curto demais para {k} folds de {size} dias")
    folds = {}
    for i in range(k):
        v0 = first_val + i * size
        v1 = v0 + size
        t0 = 0
        if part.get("tipo", "expanding") == "sliding" and part.get("janela_treino_max_dias"):
            t0 = max(0, v0 - int(part["janela_treino_max_dias"]))
        val_end = dev[v1] if v1 < len(dev) else test_start
        folds[i + 1] = (dev[t0], dev[v0], val_end)
    return folds, test_start


def describe_folds():
    folds, test_start = fold_boundaries()
    cal = calendar()
    rows = []
    for k, (t0, v0, v1) in folds.items():
        rows.append({"fold": k, "treino": f"{t0.date()} → {cal[cal < v0][-1].date()}",
                     "validação": f"{v0.date()} → {cal[cal < v1][-1].date()}",
                     "dias treino": int(((cal >= t0) & (cal < v0)).sum()),
                     "dias validação": int(((cal >= v0) & (cal < v1)).sum())})
    rows.append({"fold": "teste", "treino": "—", "validação": f"{test_start.date()} → {cal[-1].date()}",
                 "dias treino": 0, "dias validação": int((cal >= test_start).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Tensores de um fold: features normalizadas empilhadas + índices do fim de cada janela
# --------------------------------------------------------------------------------------------------

class _Scaler:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, x):
        if self.kind == "standard":
            self.a, self.b = np.nanmean(x, 0), np.nanstd(x, 0)
        elif self.kind == "minmax":
            lo, hi = np.nanmin(x, 0), np.nanmax(x, 0)
            self.a, self.b = lo, hi - lo
        elif self.kind == "robust":
            med = np.nanmedian(x, 0)
            q75, q25 = np.nanpercentile(x, [75, 25], 0)
            self.a, self.b = med, q75 - q25
        elif self.kind == "none":
            self.a, self.b = np.zeros(x.shape[1:]), np.ones(x.shape[1:])
        else:
            raise ValueError(f"scaler desconhecido: {self.kind}")
        self.b = np.where(self.b > 1e-12, self.b, 1.0)
        return self

    def transform(self, x):
        return (x - self.a) / self.b

    def inverse(self, x):
        return x * self.b + self.a


class FoldData:
    """Tudo o que um fold precisa, já no device.

    `feats[R, F]` empilha as linhas (dias) de todas as ações; `idx_*` são as linhas onde termina cada janela
    (o dia t da decisão). Uma janela é `feats[t-L+1 : t+1]`, montada por indexação na hora do batch.
    """

    def __init__(self, params, fold, device="cpu", tickers=None):
        import torch

        features = resolve_features(params["features"])
        target, h, L = params["target"], int(params["horizon"]), int(params["lookback"])
        if target not in TARGETS:
            raise ValueError(f"alvo desconhecido: {target} (opções: {TARGETS})")
        tickers = tickers or common.load_study()["dados"]["tickers"]
        folds, test_start = fold_boundaries()
        t0, v0, v1 = folds[fold]

        feats, ys, lr_h, price, tick_id, dates = [], [], [], [], [], []
        idx = {"train": [], "val": [], "test": []}
        y_a, y_b = [], []
        offset = 0
        for tid, ticker in enumerate(tickers):
            df = build_frame(ticker, features, target, h)
            d = df.index
            n = len(df)
            pos = np.arange(n)
            has_window = pos >= L - 1
            tgt_date = pd.Series(d).shift(-h).to_numpy()  # data do alvo (NaT nos últimos h dias)
            ok = has_window & ~pd.isna(tgt_date) & ~np.isnan(df["_y"].to_numpy())
            dd = d.to_numpy()
            train = ok & (dd >= np.datetime64(t0)) & (tgt_date < np.datetime64(v0))
            val = ok & (dd >= np.datetime64(v0)) & (tgt_date < np.datetime64(v1))
            test = ok & (dd >= np.datetime64(test_start))
            if train.sum() == 0:
                raise ValueError(f"{ticker}: fold {fold} sem amostras de treino (lookback/histórico curtos?)")

            x = df[features].to_numpy(np.float64)
            fscaler = _Scaler(params["scaler"]).fit(x[pos[train].min(): pos[train].max() + 1])
            y = df["_y"].to_numpy(np.float64)
            tscaler = _Scaler("standard" if params["scaler"] == "none" else params["scaler"])
            tscaler.fit(y[train][:, None])

            feats.append(fscaler.transform(x))
            ys.append(tscaler.transform(np.nan_to_num(y)[:, None])[:, 0])
            y_a.append(np.full(n, tscaler.a[0]))
            y_b.append(np.full(n, tscaler.b[0]))
            lr_h.append(np.nan_to_num(df["_lr_h"].to_numpy(np.float64)))
            price.append(df["_price"].to_numpy(np.float64))
            tick_id.append(np.full(n, tid))
            dates.append(dd)
            for name, mask in (("train", train), ("val", val), ("test", test)):
                idx[name].append(pos[mask] + offset)
            offset += n

        cat = np.concatenate
        self.tickers, self.features, self.lookback, self.target = tickers, features, L, target
        self.n_features = len(features)
        self.device = device
        self.feats = torch.tensor(cat(feats), dtype=torch.float32, device=device)
        self.y = torch.tensor(cat(ys), dtype=torch.float32, device=device)
        self.idx = {k: torch.tensor(cat(v), dtype=torch.long, device=device) for k, v in idx.items()}
        # Em numpy, para métricas/inversão do alvo (as métricas são sempre no log-retorno de h dias).
        self.y_a, self.y_b = cat(y_a), cat(y_b)
        self.lr_h, self.price, self.tick_id, self.dates = cat(lr_h), cat(price), cat(tick_id), cat(dates)
        self._offsets = torch.arange(-L + 1, 1, device=device)

    def windows(self, rows):
        """rows[B] (linhas de fim de janela) → X[B, L, F]."""
        return self.feats[rows[:, None] + self._offsets]

    def size(self, split):
        return int(self.idx[split].numel())

    def to_log_return(self, rows, y_scaled):
        """Previsão do modelo (alvo normalizado) → log-retorno previsto de h dias, na escala original."""
        rows = np.asarray(rows)
        y = np.asarray(y_scaled, dtype=np.float64) * self.y_b[rows] + self.y_a[rows]
        if self.target == "log_return":
            return y
        return np.log(np.clip(y, 1e-8, None) / self.price[rows])

    def train_mean_log_return(self):
        """Por ação: média do log-retorno de h dias no treino (usada pela referência ingênua `naive_mean`)."""
        rows = self.idx["train"].cpu().numpy()
        return {t: float(self.lr_h[rows][self.tick_id[rows] == t].mean()) for t in range(len(self.tickers))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preparar", action="store_true", help="localiza/baixa e congela os preços")
    args = parser.parse_args()
    if args.preparar:
        prepare_prices()
        print(describe_folds().to_string(index=False))


if __name__ == "__main__":
    main()
