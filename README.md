# Predictive Genetic Disorder Model

This repository contains a reproducible machine learning project for analyzing a clinical genetics dataset and training predictive classification models. The workflow cleans the data, removes direct identifiers, builds preprocessing pipelines, handles class imbalance, trains multiple supervised models, and saves the best-performing model and metrics locally.

The project compares:

- Support Vector Machine
- Random Forest Classifier
- Decision Tree Classifier

By default, the training script predicts patient `status`. The same workflow can also be pointed at another available target column, such as `genetic_disorder`.

## Access The Code

- Original notebook: [notebooks/Final_Project_Genetic_Disorder_task_1.ipynb](notebooks/Final_Project_Genetic_Disorder_task_1.ipynb)
- Data cleaning module: [src/genetic_disorder_ml/data.py](src/genetic_disorder_ml/data.py)
- Model training module: [src/genetic_disorder_ml/train.py](src/genetic_disorder_ml/train.py)
- Training script: [scripts/train_model.py](scripts/train_model.py)
- Full project write-up: [PROJECT_WRITEUP.md](PROJECT_WRITEUP.md)

## Visualizations

The raw patient-level dataset is not published, but the repository includes aggregate visualizations:

![Patient status distribution](reports/figures/patient_status_distribution.png)

![Genetic disorder category distribution](reports/figures/genetic_disorder_distribution.png)

![Patient age distribution](reports/figures/patient_age_distribution.png)

## Health Impact

Genetic disorder prediction models can support earlier identification of high-risk patient patterns, help health teams prioritize follow-up, and improve the way clinical data is organized for review. In a real healthcare setting, a validated model like this could help flag cases for genetic counseling, additional screening, or closer clinical monitoring.

This project is educational and should not be used for clinical diagnosis, treatment, or medical decision-making. Any real-world use would require clinical validation, bias testing, privacy review, model monitoring, and approval by qualified healthcare professionals.

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
    genetic_disorder_distribution.png
    patient_age_distribution.png
    patient_status_distribution.png
requirements.txt
README.md
PROJECT_WRITEUP.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

The raw dataset is not committed because it can contain sensitive patient-level attributes and identifiers. To run the project locally, place your authorized copy at:

```text
data/raw/Genetic_disorder.csv
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

Outputs are written locally to:

- `models/best_model.joblib`
- `reports/metrics.json`

These generated files are ignored by Git.
