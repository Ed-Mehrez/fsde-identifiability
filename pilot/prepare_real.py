"""Build daily real-data series for Gate 1 stage B.

Sources (Hugging Face; all forecast targets fall in 2026, after every model's release):
- rogerdehe/klines-binance, futures/5m/*.parquet -> per asset, per UTC day:
    log realised variance  log(sum of squared 5-minute log returns)   [long memory]
    log close price                                                     [control, H ~ 1/2]
- hsilvosa/entsoe-observatory, data/zone_daily/part-0.parquet -> daily mean
  day-ahead price (EUR/MWh) per bidding zone.
- National-Digital/au-water-storage-levels, data.parquet -> log daily mean
  storage volume per Australian reservoir (Bureau of Meteorology), a
  persistent long-memory series.

Writes a long-format CSV with columns: domain, series, date, value.

    python pilot/prepare_real.py --binance DIR/futures/5m --entsoe FILE --out series_daily.csv
    python pilot/prepare_real.py --water FILE --out series_water.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def crypto_daily(path: str, min_bars: int = 280) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["date", "close"]).sort_values("date")
    df = df[df.close > 0]
    df["r"] = np.log(df.close).diff()
    df["day"] = pd.to_datetime(df.date).dt.tz_localize(None).dt.floor("D") if pd.api.types.is_datetime64_any_dtype(df.date) \
        else pd.to_datetime(df.date, utc=True).dt.tz_localize(None).dt.floor("D")
    g = df.groupby("day").agg(rv=("r", lambda x: np.nansum(x.values ** 2)), n=("r", "size"),
                              close=("close", "last"))
    g = g[(g.n >= min_bars) & (g.rv > 0)]
    name = os.path.basename(path).replace("_USDT_USDT.parquet", "")
    out = [pd.DataFrame(dict(domain="crypto_log_rv", series=name, date=g.index, value=np.log(g.rv.values))),
           pd.DataFrame(dict(domain="crypto_log_price", series=name, date=g.index, value=np.log(g.close.values)))]
    return pd.concat(out)


def entsoe_daily(path: str, min_cov: float = 0.95) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[(d.price_coverage >= min_cov) & d.price_eur_mean.notna()]
    return pd.DataFrame(dict(domain="entsoe_price", series=d.zone_key.values,
                             date=pd.to_datetime(d.local_date).values, value=d.price_eur_mean.values))


def water_daily(path: str) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["station_no", "date", "volume_ml"])
    d = d[d.volume_ml > 0].copy()
    d["date"] = pd.to_datetime(d.date)
    d = d.drop_duplicates(["station_no", "date"])
    return pd.DataFrame(dict(domain="au_reservoir_log_volume", series=d.station_no.values,
                             date=d.date.values, value=np.log(d.volume_ml.values)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binance", help="directory with futures/5m parquet files")
    ap.add_argument("--entsoe", help="zone_daily parquet file")
    ap.add_argument("--water", help="au-water-storage-levels data.parquet")
    ap.add_argument("--out", default="results/pilot_gate1_real/series_daily.csv")
    args = ap.parse_args()
    parts = [crypto_daily(p) for p in sorted(glob.glob(os.path.join(args.binance, "*.parquet")))] \
        if args.binance else []
    if args.entsoe:
        parts.append(entsoe_daily(args.entsoe))
    if args.water:
        parts.append(water_daily(args.water))
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("domain").agg(series=("series", "nunique"), rows=("value", "size"),
                                   start=("date", "min"), end=("date", "max")))


if __name__ == "__main__":
    main()
