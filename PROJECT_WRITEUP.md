# Predictive Genetic Disorder Model: Project Write-Up

## Overview

This project uses a clinical genetics dataset to explore patient-level patterns and build supervised machine learning models for predictive classification. The main analysis is kept in the original notebook, and the supporting Python files make the workflow easier to rerun.

## Where To Access The Code

- Original notebook with the analysis: [notebooks/Final_Project_Genetic_Disorder_task_1.ipynb](notebooks/Final_Project_Genetic_Disorder_task_1.ipynb)
- Reusable data cleaning code: [src/genetic_disorder_ml/data.py](src/genetic_disorder_ml/data.py)
- Reusable model training code: [src/genetic_disorder_ml/train.py](src/genetic_disorder_ml/train.py)
- Command-line training script: [scripts/train_model.py](scripts/train_model.py)

## Where To Access The Visualizations

The repository includes aggregate, non-identifying charts in [reports/figures](reports/figures/):

- [Patient status distribution](reports/figures/patient_status_distribution.png)
- [Genetic disorder category distribution](reports/figures/genetic_disorder_distribution.png)
- [Patient age distribution](reports/figures/patient_age_distribution.png)

These charts summarize the dataset without publishing the raw patient-level CSV.

## Problem Statement

Clinical genetics datasets often contain mixed numeric and categorical variables, missing values, class imbalance, and sensitive patient-level fields. This project demonstrates how to organize that type of data for machine learning while keeping privacy and reproducibility in mind.

The workflow focuses on:

- Cleaning and standardizing column names.
- Removing direct identifiers before modeling.
- Separating model features from the selected target.
- Avoiding target leakage from related label columns.
- Handling missing values.
- Encoding categorical fields.
- Scaling numeric fields.
- Balancing classes with random undersampling.
- Comparing Support Vector Machine, Random Forest, and Decision Tree models.

## Methodology

The project uses a reproducible Python workflow. The data preparation module standardizes the dataset, removes identifying columns, and builds the feature and target tables. The training module builds preprocessing pipelines for numeric and categorical variables, balances the training set, trains multiple classifiers, and saves the best model and metrics locally.

The original notebook remains available in the repository for reviewers who want to see the exploratory analysis, charts, and modeling process in notebook form.

## Model Results

### Patient Status Prediction

The patient-status model was evaluated on `1,397` test patients.

| Model | Accuracy | Correct Predictions | Wrong Predictions | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Support Vector Machine | 91.84% | 1,283 / 1,397 | 114 / 1,397 | 91.55% | 91.88% |
| Random Forest | 100.00% | 1,397 / 1,397 | 0 / 1,397 | 100.00% | 100.00% |
| Decision Tree | 100.00% | 1,397 / 1,397 | 0 / 1,397 | 100.00% | 100.00% |

Confusion matrix label order: `Alive`, `Deceased`

```text
Support Vector Machine
[[771,  75],
 [ 39, 512]]

Random Forest
[[846,   0],
 [  0, 551]]

Decision Tree
[[846,   0],
 [  0, 551]]
```

For the SVM model, `771` Alive patients and `512` Deceased patients were classified correctly. It misclassified `75` Alive patients as Deceased and `39` Deceased patients as Alive.

### Genetic Disorder Type Prediction

The genetic-disorder type model was evaluated on `1,397` test patients.

| Model | Accuracy | Correct Predictions | Wrong Predictions | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Support Vector Machine | 47.03% | 657 / 1,397 | 740 / 1,397 | 46.43% | 47.03% |
| Random Forest | 48.75% | 681 / 1,397 | 716 / 1,397 | 47.26% | 48.74% |
| Decision Tree | 47.03% | 657 / 1,397 | 740 / 1,397 | 45.15% | 47.87% |

Confusion matrix label order: `Mitochondrial genetic inheritance disorders`, `Multifactorial genetic inheritance disorders`, `Single-gene inheritance diseases`

```text
Support Vector Machine
[[351, 103, 263],
 [ 12, 127,   8],
 [204, 150, 179]]

Random Forest
[[384, 107, 226],
 [ 16, 118,  13],
 [221, 133, 179]]

Decision Tree
[[336, 111, 270],
 [ 28,  91,  28],
 [186, 117, 230]]
```

The best model for genetic disorder type was Random Forest, with `48.75%` accuracy and `681` correct predictions out of `1,397` test cases.

## Model Interpretation: False Positives And False Negatives

For patient-status prediction, false positives and false negatives have different practical meanings.

A false positive occurs when the model predicts `Deceased` for a patient who is actually `Alive`. In the SVM model, there were `75` false positives. In a healthcare workflow, this could create unnecessary concern, extra clinical review, or inefficient use of follow-up resources.

A false negative occurs when the model predicts `Alive` for a patient who is actually `Deceased`. In the SVM model, there were `39` false negatives. This is especially important because it represents missed high-risk cases. If a similar error happened in a real decision-support workflow, it could reduce the urgency given to patients who may need closer review.

For this reason, accuracy alone is not enough. Recall for the higher-risk class is important because the cost of missing a serious outcome may be greater than the cost of flagging an extra patient for review. Precision is also important because too many false alerts can create alert fatigue and waste limited clinical resources.

For genetic-disorder type prediction, false positives and false negatives mean the model placed patients into the wrong disorder category. This matters because different disorder groups may require different genetic counseling, screening, monitoring, or specialist referral pathways. Since the best disorder-type model only reached `48.75%` accuracy, it should be interpreted as an exploratory baseline, not as a reliable clinical classifier.

## Health Impact

Predictive modeling in genetics can support earlier identification of high-risk patient patterns when used responsibly. A validated workflow could help care teams prioritize follow-up, identify patients who may benefit from genetic counseling, and organize complex clinical information for review.

Potential health and operational value includes:

- Earlier risk flagging for patients who may need closer review.
- Better prioritization of follow-up and screening workflows.
- Support for genetic counseling and family-history review.
- Improved awareness of missing or inconsistent clinical data.
- A foundation for decision-support tools that assist, but do not replace, clinicians.

Quantitatively, the patient-status model using SVM correctly classified about `92 out of every 100` patients. Random Forest and Decision Tree reached `100%` on the test split, which is strong but should be interpreted cautiously because perfect healthcare-model performance can indicate possible target leakage or highly predictive variables. For genetic disorder type, the best model correctly classified about `49 out of every 100` patients, making it a useful baseline for exploration but not strong enough for clinical use.

This project is educational and should not be used for clinical diagnosis, treatment, or medical decision-making. Real clinical use would require validation, privacy review, fairness testing, model monitoring, and oversight from qualified healthcare professionals.

## Limitations

- The raw dataset is not included because it may contain sensitive patient-level information.
- Accuracy alone is not enough for healthcare evaluation; recall, precision, calibration, subgroup fairness, and clinical usefulness should also be reviewed.
- Very high model performance should be investigated for possible target leakage or duplicated patterns.
- This is a portfolio project, not production clinical software.
