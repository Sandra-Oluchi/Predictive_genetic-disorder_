# Genetic Disorder Machine Learning Project

This project organizes the final genetic disorder machine learning notebook into a clean, reproducible GitHub repository format.

The original notebook analyzes a genetic disorder dataset, performs data cleaning and exploratory analysis, balances the target classes with random undersampling, and compares supervised classification models:

- Support Vector Machine
- Random Forest Classifier
- Decision Tree Classifier

By default, the training script mirrors the notebook target and predicts patient `status`. You can also train against another label, such as `genetic_disorder`.

## Project Structure

```text
data/
  raw/
    .gitkeep
notebooks/
  Final_Project_Genetic_Disorder_task_1.ipynb
src/
  genetic_disorder_ml/
    __init__.py
    data.py
    train.py
scripts/
  train_model.py
models/
reports/
  figures/
requirements.txt
README.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Models

Run the default experiment:

```powershell
python scripts/train_model.py
```

Train a specific target:

```powershell
python scripts/train_model.py --target genetic_disorder
```

Outputs are written to:

- `models/best_model.joblib`
- `reports/metrics.json`

## Dataset

The raw dataset is not committed because it can contain sensitive patient-level attributes and identifiers. To run the project locally, place your authorized copy at:

```text
data/raw/Genetic_disorder.csv
```

The training code expects a clinical genetics CSV with patient demographics, genetic history, clinical observations, symptoms, test values, and disorder labels.

## Notes

This project is for educational machine learning practice only. It should not be used for clinical diagnosis or medical decision-making.
