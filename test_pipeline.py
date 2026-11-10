"""Assert-based checks for train.py. Run: python test_pipeline.py"""
import numpy as np
import pandas as pd

from train import clean, features, make_model, fit_target, predict_rate


def sample_df() -> pd.DataFrame:
    """One Lexington->Fort Wayne-like row plus variants, matching raw CSV columns."""
    return pd.DataFrame({
        "load_id": ["X-1", "X-2", "X-3"],
        "pickup": ["Lexington"] * 3,
        "delivery": ["Fort Wayne"] * 3,
        "pickup_lat": [36.99152, 36.99152, 36.99152],
        "pickup_lon": [-84.99876, -84.99876, -84.99876],
        "delivery_lat": [41.31561, 41.31561, 41.31561],
        "delivery_lon": [-85.36206, -85.36206, -85.36206],
        "distance": [358.0, 999.0, 360.0],          # X-2: far above straight-line, must stay
        "equipment": ["Dry Van", "Reefer", "Flatbed"],
        "weight": [-32000.0, 28000.0, 45000.0],     # X-1: negative weight
        "date": ["2025-06-01", "2025-06-02", "2025-06-03"],
        "market_index": [1.0, 1.1, 0.9],
        "quote_signal": [2.0, 2.1, 1.9],
        "posted_rate": [900.0, 99999.0, 850.0],     # X-2: rpm far above 6.0
    })


def main() -> None:
    df = sample_df()

    cleaned = clean(df, is_train=False)
    assert (cleaned["weight"] > 0).all(), "negative weights must become positive"

    assert cleaned["distance"].equals(df["distance"]), "distance must never be altered"

    train_cleaned = clean(df, is_train=True)
    assert len(train_cleaned) == 2, "outlier rates dropped for training data"
    assert len(cleaned) == 3, "outlier rates kept for prediction data"

    X = features(cleaned)
    assert not any("month" in c for c in X.columns), "feature frame must contain no month column"

    # predictions: positive and same length as input
    params = {"learning_rate": 0.1, "max_iter": 20, "max_leaf_nodes": 15}
    model = make_model(params).fit(features(train_cleaned), fit_target(train_cleaned))
    pred = predict_rate(model, cleaned)
    assert len(pred) == len(cleaned), "prediction length must match input length"
    assert np.isfinite(pred).all() and (pred > 0).all(), "predictions must be positive"

    print("all 5 pipeline checks passed")


if __name__ == "__main__":
    main()
