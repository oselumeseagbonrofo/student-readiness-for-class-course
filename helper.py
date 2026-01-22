import sys

try:
    import re
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception as e:
    print("Missing Python packages. Install with:")
    print("  pip install pandas scikit-learn")
    print("Full error:", e)
    sys.exit(1)


def load_data(path="/data/survey.csv"):
    df = pd.read_csv(path)
    # Calculate test size as 31.25% of total rows, rounded down
    test_size = int(0.3125 * len(df))
    if test_size == 0 and len(df) > 0:
        test_size = 1  # Ensure at least 1 row for test if possible
    train_df = df[:-test_size] if test_size > 0 else df
    test_df = df[-test_size:] if test_size > 0 else pd.DataFrame()
    return train_df, test_df


def to_midpoint(val):
    # convert ranges like "1-2" or "3-5" to numeric midpoint; leave numeric as-is
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (a + b) / 2.0
    # try to parse single number
    try:
        return float(s)
    except Exception:
        return np.nan


def preprocess(
    df,
    drop_first_column=True,
    scaler=None,
    category_maps=None,
    return_state=False,
):
    X = df.copy()
    # Drop the first column if requested so identifier-like fields stay out of the analysis.
    if drop_first_column and X.shape[1] > 0:
        X = X.iloc[:, 1:]
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the first column.")

    # Map Yes/No to 1/0 for any column that contains those values
    for col in X.columns:
        if X[col].dropna().isin(["Yes", "No"]).all():
            X[col] = X[col].map({"Yes": 1, "No": 0})

    # Convert ranges in columns mentioning years or experience
    year_cols = [c for c in X.columns if "Years" in c or "experience" in c.lower()]
    for c in year_cols:
        X[c] = X[c].apply(to_midpoint)

    # Convert other object columns to deterministic integer codes using stored mappings when provided
    category_maps = (
        {} if category_maps is None else {k: list(v) for k, v in category_maps.items()}
    )
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        if c not in category_maps:
            # Capture observed categories in order of appearance for reproducibility
            category_maps[c] = pd.Categorical(X[c]).categories.tolist()
        cat = pd.Categorical(X[c], categories=category_maps[c])
        codes = cat.codes.astype(float)
        X[c] = codes

    # At this point, all columns should be numeric-ish; fill NaN with column mean
    X = X.astype(float)
    X = X.fillna(X.mean())

    # Keep a copy of processed features for downstream
    feature_names = X.columns.tolist()

    # Scale features using provided scaler when available so train/test share statistics
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)
    else:
        X_scaled = scaler.transform(X.values)

    if return_state:
        return X_scaled, feature_names, scaler, category_maps
    return X_scaled, feature_names


def save_trained_model(model, path):
    joblib.dump(model, path)
