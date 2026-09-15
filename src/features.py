import numpy as np
import pandas as pd
from src.config import DROP_COLS, TARGET

RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}

FEATURE_NAMES = [
    "type", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
    "temp_diff", "power_w", "strain", "torque_per_rpm", "temp_rpm_ratio",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataframe into model-ready features.
    Used by both training and the app, so they always match.
    """
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.rename(columns=RENAME).copy()

    # encode machine type as ordinal (L=low quality, M=medium, H=high)
    df["type"] = df["type"].map({"L": 0, "M": 1, "H": 2})

    # --- Physics-based features ---

    # 1. Temperature difference: small gap → poor heat dissipation → HDF
    df["temp_diff"] = df["process_temp"] - df["air_temp"]

    # 2. Mechanical power (watts): too low or too high → PWF
    df["power_w"] = df["torque"] * df["rpm"] * 2 * np.pi / 60

    # 3. Tool strain: high wear × high torque → OSF
    df["strain"] = df["tool_wear"] * df["torque"]

    # 4. Torque per RPM: abnormal ratio signals mechanical stress
    df["torque_per_rpm"] = df["torque"] / (df["rpm"] + 1e-9)

    # 5. Temperature-speed interaction: overheating at low speed
    df["temp_rpm_ratio"] = df["process_temp"] / (df["rpm"] + 1e-9)

    return df[FEATURE_NAMES]


def get_X_y(df: pd.DataFrame):
    """Return feature matrix and target series."""
    X = engineer(df)
    y = df[TARGET]
    return X, y