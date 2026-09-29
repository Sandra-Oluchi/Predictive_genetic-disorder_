# Predictive Genetic Disorder Model: Project Write-Up

## Overview

This project demonstrates an end-to-end machine learning workflow for a clinical genetics dataset. The goal is to prepare patient-level clinical data, remove direct identifiers, build a repeatable preprocessing and modeling pipeline, and compare multiple supervised classification algorithms.

The project is structured for portfolio review. It includes a cleaned Python package, a command-line training script, a Jupyter notebook, dependency files, and GitHub Actions CI for a lightweight import check.

## Problem Statement

Clinical genetics datasets often contain a mix of numeric values, categorical variables, missing values, class imbalance, and sensitive patient identifiers. This project addresses those challenges by building a reproducible workflow that can:

- Clean and standardize messy column names.
- Remove direct identifiers and low-value columns.
- Separate model features from the selected target.
- Prevent target leakage by dropping related label columns from the feature set.
- Impute missing numeric and categorical values.
- Scale numeric values and encode categorical values.
- Balance the training data with random undersampling.
- Train and compare SVM, Random Forest, and Decision Tree classifiers.
- Save the best model and evaluation metrics locally.

## Methodology

The workflow starts by loading the dataset from `data/raw/Genetic_disorder.csv`. Column names are standardized by trimming whitespace, lowercasing names, and replacing spaces with underscores. Several columns containing direct identifiers, personal names, institution details, and low-value test flags are removed before modeling.

The model pipeline separates numeric and categorical features. Numeric features are imputed with the median and scaled with `StandardScaler`. Categorical features are imputed with the most frequent value and encoded with `OrdinalEncoder`, including support for unknown categories during inference.

Class imbalance is handled with `RandomUnderSampler` from `imbalanced-learn`. Three classifiers are trained with the same preprocessing flow:

- Support Vector Machine
- Random Forest Classifier
- Decision Tree Classifier

The best model is selected by accuracy on the held-out test set and saved with `joblib`. Model metrics are saved as JSON for review.

## Health Impact

A predictive genetics workflow can have meaningful healthcare value when used responsibly. It can help organize complex clinical information, identify patterns that may require closer review, and support earlier follow-up for patients who may need genetic counseling, screening, or additional clinical monitoring.

Potential health and operational impacts include:

- Earlier risk flagging: Models can help surface patient profiles that may need further attention.
- Better care prioritization: Clinical teams can use risk scores to organize follow-up queues.
- Support for genetic counseling workflows: A model can help identify patients who may benefit from counseling or additional family-history review.
- Improved data quality awareness: The cleaning process highlights missing values, inconsistent labels, and fields that should be standardized.
- Educational decision support: The project demonstrates how machine learning can support, but not replace, clinical judgment.

This model is not a diagnostic tool. Real clinical deployment would require rigorous validation, fairness testing across demographic groups, privacy review, explainability analysis, prospective evaluation, and oversight by licensed healthcare professionals.

## Limitations

- The raw dataset is not included in this public repository because it may contain sensitive patient-level information.
- Accuracy alone is not enough for healthcare evaluation; recall, precision, calibration, subgroup fairness, and clinical utility should also be reviewed.
- Very high model accuracy may indicate target leakage, duplicated patterns, or an overly simple label structure, so results should be interpreted carefully.
- The current workflow is educational and portfolio-focused, not production-ready clinical software.

## How To Run

Place an authorized copy of the dataset at:

```text
data/raw/Genetic_disorder.csv
```

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train the default model:

```powershell
python scripts/train_model.py
```

Train a specific target:

```powershell
python scripts/train_model.py --target genetic_disorder
```

## Code Appendix

### `src/genetic_disorder_ml/__init__.py`

```python
"""Utilities for the genetic disorder machine learning project."""

__version__ = "0.1.0"
```

### `src/genetic_disorder_ml/data.py`

```python
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
```

### `src/genetic_disorder_ml/train.py`

```python
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from genetic_disorder_ml.data import build_features_and_target, clean_dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_models(random_state: int = 42) -> dict[str, object]:
    return {
        "support_vector_machine": SVC(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
    }


def train_and_evaluate(
    data_path: str | Path,
    target: str = "status",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    random_state: int = 42,
) -> dict[str, object]:
    df = clean_dataset(load_dataset(data_path))
    X, y = build_features_and_target(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    results: dict[str, object] = {}
    best_name = ""
    best_score = -1.0
    best_pipeline: ImbPipeline | None = None

    for name, estimator in get_models(random_state=random_state).items():
        pipeline = ImbPipeline(
            steps=[
                ("sampler", RandomUnderSampler(random_state=random_state)),
                ("preprocessor", make_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results[name] = {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }

        if accuracy > best_score:
            best_name = name
            best_score = accuracy
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)

    metrics = {
        "target": target,
        "best_model": best_name,
        "best_accuracy": best_score,
        "models": results,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
```

### `scripts/train_model.py`

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genetic_disorder_ml.data import DEFAULT_DATA_PATH
from genetic_disorder_ml.train import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train genetic disorder dataset classifiers.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the input CSV file.")
    parser.add_argument("--target", default="status", help="Target column to predict.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Path for the saved best model.")
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_PATH, help="Path for metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        data_path=Path(args.data),
        target=args.target,
        model_path=Path(args.model_out),
        metrics_path=Path(args.metrics_out),
    )
    print(f"Best model: {metrics['best_model']}")
    print(f"Best accuracy: {metrics['best_accuracy']:.4f}")
    print(f"Saved model to: {args.model_out}")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
```
