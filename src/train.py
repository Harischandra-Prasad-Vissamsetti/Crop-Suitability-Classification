"""Shared preprocessing: used by BOTH training and the Streamlit app (no train/serve skew)."""
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
# Physically plausible bounds (domain knowledge). Values outside -> treated as sensor errors.
VALID_RANGES = {
    "N": (0, 200), "P": (0, 200), "K": (0, 250),
    "temperature": (0, 50), "humidity": (0, 100), "ph": (3, 10), "rainfall": (0, 500),
}
UNITS = {"N": "kg/ha", "P": "kg/ha", "K": "kg/ha", "temperature": "°C",
         "humidity": "%", "ph": "", "rainfall": "mm"}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, leakage-free cleaning: drop exact duplicates, null out impossible values.
    (Imputation is learned from training data only, inside the Pipeline.)"""
    df = df.drop_duplicates().copy()
    for col, (lo, hi) in VALID_RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df.reset_index(drop=True)

def make_pipeline(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


LAST_SOURCE = None

def load_data(data_dir="data") -> pd.DataFrame:
    """Load the dataset from data/. Accepts crop_data.csv OR Crop_recommendation.csv (any capitalisation),
    and tolerates different column capitalisation (e.g. 'Label', 'Temperature', 'pH')."""
    from pathlib import Path
    folder = Path(data_dir)
    names = ["crop_data.csv", "crop_recommendation.csv"]
    found = next((f for n in names for f in folder.glob("*.csv") if f.name.lower() == n), None)
    if found is None:
        raise FileNotFoundError(f"No dataset found in '{folder}/'. Put Crop_recommendation.csv (or crop_data.csv) there.")
    global LAST_SOURCE
    LAST_SOURCE = found.name
    df = pd.read_csv(found)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for want in FEATURES + [TARGET]:
        if want not in df.columns and want.lower() in lower:
            rename[lower[want.lower()]] = want
    df = df.rename(columns=rename)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{found.name} is missing columns {missing}. Found: {list(df.columns)}")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    print(f"[data] loaded {found.name}: {df.shape}")
    return df[FEATURES + [TARGET]]


# ---------------- Crop-profile suitability score ----------------
def build_profiles(X: pd.DataFrame, y) -> dict:
    """Per-crop mean/std of every feature (std floored at 5% of the feature range)."""
    mu = X.groupby(y).mean()
    sd = X.groupby(y).std()
    sd = sd.clip(lower=0.05 * (X.max() - X.min()), axis=1)
    return {"classes": [str(c) for c in mu.index], "features": list(X.columns),
            "mean": mu.values.tolist(), "std": sd.values.tolist()}

def profile_scores(values, prof: dict) -> np.ndarray:
    """Suitability score 0-100 per crop: how closely inputs match the crop's typical conditions.
    100 = every input equals the crop's average; falls smoothly as inputs move away."""
    x = np.atleast_2d(np.asarray(values, dtype=float))
    mean, std = np.asarray(prof["mean"]), np.asarray(prof["std"])
    z = np.clip((x[:, None, :] - mean[None]) / std[None], -4, 4)
    return 100 * np.exp(-0.5 * (z ** 2).mean(axis=2))