"""Séries externas ao preço (derivativos, on-chain, sentimento), congeladas em data/externos/.

Fontes públicas e gratuitas, com histórico longo:
  funding      BitMEX XBTUSD, taxa de financiamento do perpétuo (3 por dia), desde 2016-05. Agregada por dia UTC:
               média do dia (`funding_rate`)
  onchain      blockchain.com charts: n-transactions (transações/dia), n-unique-addresses (endereços ativos),
               estimated-transaction-volume-usd (volume transacionado em US$), hash-rate; desde 2009-2010
  fear_greed   alternative.me Crypto Fear & Greed Index (0–100), desde 2018-02
  mercado      Yahoo Finance: ETH (desde 2017-11), ouro, S&P 500, VIX, juro de 10 anos, dólar, Nvidia, Tesla (desde 2014)
Open interest e long/short ratio não entram: as APIs gratuitas só guardam os últimos 30 dias. Exchange netflows e
contagem de transações de baleias só existem em serviços pagos (CryptoQuant, Glassnode).

Os CSVs ficam versionados no repositório: o estudo roda sem internet e sempre com os mesmos dados.
Uso: python src/external.py --baixar   (atualiza os CSVs; depois rode os testes e faça commit)
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

EXT_DIR = os.path.join(common.REPO_DIR, "data", "externos")
ONCHAIN = {"n-transactions": "tx_count", "n-unique-addresses": "active_addresses",
           "estimated-transaction-volume-usd": "tx_volume_usd", "hash-rate": "hash_rate"}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def download_funding():
    """BitMEX XBTUSD, paginado (500 por requisição); média diária da taxa de financiamento."""
    rows, start = [], "2016-01-01T00:00:00Z"
    while True:
        batch = _get(f"https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&reverse=false&startTime={start}")
        if not batch:
            break
        rows += [(b["timestamp"], b["fundingRate"]) for b in batch]
        last = pd.Timestamp(batch[-1]["timestamp"]) + pd.Timedelta(seconds=1)
        start = last.strftime("%Y-%m-%dT%H:%M:%SZ")
        if len(batch) < 500:
            break
        time.sleep(1.1)  # limite público da BitMEX
    df = pd.DataFrame(rows, columns=["ts", "funding_rate"])
    df["Date"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
    return df.groupby("Date")[["funding_rate"]].mean()


def download_onchain():
    cols = {}
    for chart, name in ONCHAIN.items():
        data = _get(f"https://api.blockchain.info/charts/{chart}?timespan=all&format=json&sampled=false")["values"]
        s = pd.Series({pd.Timestamp(dt.datetime.fromtimestamp(v["x"], dt.UTC).date()): v["y"] for v in data}, name=name)
        cols[name] = s.groupby(level=0).mean()
    df = pd.DataFrame(cols)
    df.index.name = "Date"
    return df


def download_fear_greed():
    data = _get("https://api.alternative.me/fng/?limit=0&format=json")["data"]
    s = pd.Series({pd.Timestamp(dt.datetime.fromtimestamp(int(d["timestamp"]), dt.UTC).date()): float(d["value"])
                   for d in data}, name="fear_greed")
    df = s.sort_index().to_frame()
    df.index.name = "Date"
    return df


MARKET = {"ETH-USD": "eth", "GC=F": "ouro", "^GSPC": "sp500", "^VIX": "vix", "^TNX": "juro10a", "DX-Y.NYB": "dolar",
          "NVDA": "nvidia", "TSLA": "tesla"}


def download_market():
    """Fechamentos diários (Yahoo Finance) dos ativos usados no paper de Wu et al. (IEEE CAI 2025): ETH, ouro,
    S&P 500, VIX, juro de 10 anos dos EUA, índice do dólar, Nvidia e Tesla. Mercados tradicionais não abrem no fim de
    semana; o alinhamento ao calendário do Bitcoin (7 dias) preenche com o último valor conhecido."""
    import yfinance as yf

    cols = {}
    for ticker, name in MARKET.items():
        df = yf.download(ticker, start="2014-01-01", auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        cols[name] = s
    df = pd.DataFrame(cols)
    df.index.name = "Date"
    return df


SOURCES = {"funding": download_funding, "onchain": download_onchain, "fear_greed": download_fear_greed,
           "mercado": download_market}


def path(source):
    return os.path.join(EXT_DIR, f"{source}.csv")


def load(source):
    """DataFrame diário (índice Date) da fonte; os valores só existem a partir do início do histórico dela."""
    p = path(source)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p} não existe: rode `python src/external.py --baixar` (precisa de internet)")
    return pd.read_csv(p, index_col="Date", parse_dates=True).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baixar", action="store_true")
    parser.add_argument("--fontes", nargs="*", default=None, help="só estas fontes (padrão: todas)")
    args = parser.parse_args()
    if args.baixar:
        os.makedirs(EXT_DIR, exist_ok=True)
        for name, fn in SOURCES.items():
            if args.fontes and name not in args.fontes:
                continue
            df = fn()
            df.to_csv(path(name), float_format="%.10g")
            print(f"{name:10s}: {len(df)} dias, {df.index.min().date()} → {df.index.max().date()} | colunas {list(df.columns)}")


if __name__ == "__main__":
    main()
