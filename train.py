"""Freight rate pipeline: clean -> features -> time-based validation -> fit -> predict.

Run end-to-end with:  python train.py
Deterministic: fixed random_state everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path(__file__).parent / "data"
ROOT = Path(__file__).parent

EARTH_R_MI = 3958.8
RPM_LO, RPM_HI = 1.0, 6.0      # training rows outside this rate/mile band are dropped
EQUIPMENT_CATEGORIES = ["Dry Van", "Flatbed", "Reefer"]

PARAM_GRID = [
    {"learning_rate": lr, "max_iter": mi, "max_leaf_nodes": ln}
    for lr in (0.05, 0.1)
    for mi in (200, 400)
    for ln in (31, 63)
]
RANDOM_STATE = 42


def haversine_mi(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(s.astype(float)) for s in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_MI * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Same cleaning for training and prediction data. Outlier rates dropped for train only.

    Distance is never altered: loads far above the straight-line distance are short
    hauls billed at a 70-mile floor, and their rates match the listed distance.
    """
    df = df.copy()
    neg = (df["weight"] < 0).sum()
    if neg:
        print(f"clean: abs() {neg} negative weights")
    df["weight"] = df["weight"].abs()

    if is_train:
        rpm = df["posted_rate"] / df["distance"]
        drop = (rpm < RPM_LO) | (rpm > RPM_HI)
        print(f"clean: dropped {drop.sum()} outlier rate-per-mile rows (train only)")
        df = df.loc[~drop].reset_index(drop=True)
    return df


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model features. Coordinates (not city names) so unseen cities work.

    No month / day-of-year: Nov-Dec never appear in training; market_index
    already carries the seasonal signal. NaNs left native for the model.
    """
    straight = haversine_mi(df["pickup_lat"], df["pickup_lon"],
                            df["delivery_lat"], df["delivery_lon"])
    X = pd.DataFrame({
        "distance": df["distance"].astype(float),
        "straight_dist": straight,
        "detour_ratio": df["distance"].astype(float) / straight,
        "pickup_lat": df["pickup_lat"].astype(float),
        "pickup_lon": df["pickup_lon"].astype(float),
        "delivery_lat": df["delivery_lat"].astype(float),
        "delivery_lon": df["delivery_lon"].astype(float),
        "dlat": df["delivery_lat"].astype(float) - df["pickup_lat"].astype(float),
        "dlon": df["delivery_lon"].astype(float) - df["pickup_lon"].astype(float),
        "weight": df["weight"].astype(float),
        "market_index": df["market_index"].astype(float),
        "quote_signal": df["quote_signal"].astype(float),
        "day_of_week": pd.to_datetime(df["date"]).dt.dayofweek.astype(float),
        "equipment": pd.Categorical(df["equipment"], categories=EQUIPMENT_CATEGORIES),
    })
    return X


def make_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype",  # uses the equipment categorical dtype
        early_stopping=False,
        random_state=RANDOM_STATE,
        **params,
    )


def fit_target(df: pd.DataFrame) -> pd.Series:
    """log(rate per mile) keeps the target stable across short and long trips."""
    return np.log(df["posted_rate"] / df["distance"])


def predict_rate(model, df: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(features(df))) * df["distance"].to_numpy(dtype=float)


def evaluate(actual, pred) -> dict:
    err = pred - actual
    return {
        "MAE": float(np.abs(err).mean()),
        "MAPE": float((np.abs(err) / actual).mean() * 100),
        "RMSE": float(np.sqrt((err ** 2).mean())),
    }


def baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Median rate-per-mile by equipment (train fold) x distance."""
    rpm = train_df["posted_rate"] / train_df["distance"]
    med = rpm.groupby(train_df["equipment"]).median()
    pred_rpm = test_df["equipment"].map(med).fillna(rpm.median())
    return pred_rpm.to_numpy(dtype=float) * test_df["distance"].to_numpy(dtype=float)


def run_experiment(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   params: dict, rows: list) -> None:
    model = make_model(params).fit(features(train_df), fit_target(train_df))
    actual = test_df["posted_rate"].to_numpy(dtype=float)
    m = evaluate(actual, predict_rate(model, test_df))
    b = evaluate(actual, baseline_predict(train_df, test_df))
    rows.append({
        "experiment": name,
        "MAE": m["MAE"], "MAPE": m["MAPE"], "RMSE": m["RMSE"],
        "base_MAE": b["MAE"], "base_MAPE": b["MAPE"], "base_RMSE": b["RMSE"],
    })


def fill_december(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Return december inputs with coords + per-day market_index/quote_signal filled.

    Coords come from the Lexington / Fort Wayne training rows; daily
    market_index and quote_signal are the per-day means of validation.csv's
    December rows. Original string formatting is preserved for scoring.
    """
    dec_raw = pd.read_csv(DATA_DIR / "december_chart_inputs.csv", dtype=str, keep_default_na=False)
    lex = train_df[train_df["pickup"] == "Lexington"].iloc[0]
    fw = train_df[train_df["delivery"] == "Fort Wayne"].iloc[0]

    dec = dec_raw.copy()
    dec["pickup_lat"] = float(lex["pickup_lat"])
    dec["pickup_lon"] = float(lex["pickup_lon"])
    dec["delivery_lat"] = float(fw["delivery_lat"])
    dec["delivery_lon"] = float(fw["delivery_lon"])
    dec["date"] = pd.to_datetime(dec["date"])
    for col in ("distance", "weight"):
        dec[col] = pd.to_numeric(dec[col])

    val_dec = val_df[val_df["date"].dt.month == 12]
    daily = val_dec.groupby(val_dec["date"].dt.date)[["market_index", "quote_signal"]].mean()
    fallback = daily.mean()
    keyed = dec["date"].dt.date.map(lambda d: daily.loc[d] if d in daily.index else fallback)
    dec["market_index"] = [v["market_index"] for v in keyed]
    dec["quote_signal"] = [v["quote_signal"] for v in keyed]
    return dec_raw, clean(dec, is_train=False)


def main() -> None:
    raw = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    train = clean(raw, is_train=True)
    val = clean(pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"]), is_train=False)

    # --- Validation: main holdout copies the real task (train past, test 2 future months)
    holdout_train = train[train["date"] < "2025-09-01"]
    holdout_test = train[train["date"] >= "2025-09-01"]

    sweep_rows = []
    for params in PARAM_GRID:
        model = make_model(params).fit(features(holdout_train), fit_target(holdout_train))
        m = evaluate(holdout_test["posted_rate"].to_numpy(dtype=float),
                     predict_rate(model, holdout_test))
        sweep_rows.append({**params, "MAE": m["MAE"], "MAPE": m["MAPE"], "RMSE": m["RMSE"]})
    sweep = pd.DataFrame(sweep_rows).sort_values("MAE").reset_index(drop=True)
    print("\nHyperparameter sweep (holdout: train Jan-Aug, test Sep-Oct), sorted by MAE:")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    best = {k: sweep.loc[0, k] for k in ("learning_rate", "max_iter", "max_leaf_nodes")}
    best = {"learning_rate": float(best["learning_rate"]), "max_iter": int(best["max_iter"]),
            "max_leaf_nodes": int(best["max_leaf_nodes"])}
    print(f"\nChosen params: {best}")

    rows = []
    run_experiment("holdout Sep-Oct", holdout_train, holdout_test, best, rows)
    raw_test = clean(raw[raw["date"] >= "2025-09-01"], is_train=False)
    run_experiment("holdout Sep-Oct incl. outliers", holdout_train, raw_test, best, rows)

    # --- Expanding window: train through month M, test month M+1 (M = Jun..Sep)
    for m_end, m_test in ((6, 7), (7, 8), (8, 9), (9, 10)):
        tr = train[train["date"].dt.month <= m_end]
        te = train[train["date"].dt.month == m_test]
        run_experiment(f"expand Jan-{m_end:02d} -> {m_test:02d}", tr, te, best, rows)

    # --- Unseen-city check: hold out 8 cities entirely (mimics validation-only cities)
    rng = np.random.default_rng(RANDOM_STATE)
    cities = np.array(sorted(set(train["pickup"]) | set(train["delivery"])))
    held_out = set(rng.choice(cities, size=8, replace=False).tolist())
    mask = train["pickup"].isin(held_out) | train["delivery"].isin(held_out)
    print(f"Unseen-city holdout: {sorted(held_out)}")
    run_experiment("unseen-city (8)", train[~mask], train[mask], best, rows)

    table = pd.DataFrame(rows)
    print("\nMetrics (rate in $; baseline = median equipment rate/mile x distance):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # --- Final fit on all of Jan-Oct, predict validation.csv
    final = make_model(best).fit(features(train), fit_target(train))
    val_pred = predict_rate(final, val)
    out = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val_pred.round(2)})
    out = out.sort_values("load_id")
    assert (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"\nWrote validation_predictions.csv ({len(out):,} rows, "
          f"range ${out['predicted_rate'].min():.2f}-${out['predicted_rate'].max():.2f})")

    # --- December chart inputs: fill missing fields, predict, keep 7-col order
    dec_raw, dec = fill_december(train, val)
    dec_pred = predict_rate(final, dec)
    dec_raw["predicted_rate"] = [f"{v:.2f}" for v in dec_pred]
    dec_raw.to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)
    print(f"Wrote data/december_chart_inputs.csv (31 rows, predicted_rate "
          f"${dec_pred.min():.2f}-${dec_pred.max():.2f})")


if __name__ == "__main__":
    main()
