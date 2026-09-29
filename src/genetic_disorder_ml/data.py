from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Genetic_disorder.csv"


DROP_COLUMNS = [
    "test_1",
    "test_2",
    "test_3",
    "test_4",
    "test_5",
    "parental_consent",
    "patient_id",
    "patient_first_name",
    "family_name",
    "father's_name",
    "institute_name",
    "location_of_institute",
]


RENAME_COLUMNS = {
    "autopsy_shows_birth_defect_(if_applicable)": "autopsy_shows_birth_defect",
    "h/o_radiation_exposure_(x-ray)": "ho_radiation_exposure",
    "h/o_substance_abuse": "ho_substance_abuse",
    "h/o_serious_maternal_illness": "ho_serious_maternal_illness",
    "heart_rate_(rates/min": "heart_rate",
    "respiratory_rate_(breaths/min)": "respiratory_rate",
    "white_blood_cell_count_(thousand_per_microliter)": "white_blood_cell_count",
    "blood_cell_count_(mcl)": "blood_cell_count",
    "folic_acid_details_(peri-conceptional)": "folic_acid_details",
    "assisted_conception_ivf/art": "assisted_conception_ivf_art",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df.rename(columns=RENAME_COLUMNS)


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)
    cols_to_drop = [column for column in DROP_COLUMNS if column in df.columns]
    return df.drop(columns=cols_to_drop)


def build_features_and_target(
    df: pd.DataFrame,
    target: str = "status",
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Target '{target}' was not found. Available columns: {available}")

    leakage_columns = {
        "status",
        "genetic_disorder",
        "disorder_subclass",
    }
    feature_drop_columns = [column for column in leakage_columns if column != target and column in df.columns]

    X = df.drop(columns=[target, *feature_drop_columns])
    y = df[target]
    return X, y
