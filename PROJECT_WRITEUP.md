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

## Health Impact

Predictive modeling in genetics can support earlier identification of high-risk patient patterns when used responsibly. A validated workflow could help care teams prioritize follow-up, identify patients who may benefit from genetic counseling, and organize complex clinical information for review.

Potential health and operational value includes:

- Earlier risk flagging for patients who may need closer review.
- Better prioritization of follow-up and screening workflows.
- Support for genetic counseling and family-history review.
- Improved awareness of missing or inconsistent clinical data.
- A foundation for decision-support tools that assist, but do not replace, clinicians.

This project is educational and should not be used for clinical diagnosis, treatment, or medical decision-making. Real clinical use would require validation, privacy review, fairness testing, model monitoring, and oversight from qualified healthcare professionals.

## Limitations

- The raw dataset is not included because it may contain sensitive patient-level information.
- Accuracy alone is not enough for healthcare evaluation; recall, precision, calibration, subgroup fairness, and clinical usefulness should also be reviewed.
- Very high model performance should be investigated for possible target leakage or duplicated patterns.
- This is a portfolio project, not production clinical software.
