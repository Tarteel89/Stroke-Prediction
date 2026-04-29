# Stroke Prediction — Logistic Regression vs. Random Forest vs. KNN

A core assignment comparing three classification algorithms on an imbalanced medical dataset, with hyperparameter tuning via `GridSearchCV` and a final production recommendation.

## Overview

This project builds and tunes three classifiers — **Logistic Regression**, **K-Nearest Neighbors**, and **Random Forest** — to diagnose stroke from patient health records. Because the dataset is imbalanced (~12% positive class) and the cost of missing a stroke is high, models are optimized for **`recall_macro`** rather than accuracy.

## Dataset

- **File:** `stroke.csv`
- **Size:** 1,136 patients
- **Target:** `stroke` (binary)
- **Class balance:** ~88% no stroke / ~12% stroke

## Workflow

1. **Data cleaning** — drop `id`, fix malformed `age` values, fill missing `bmi` with median
2. **Encoding** — one-hot encode categorical features
3. **Stratified train/test split** — 80% / 20%, preserving class ratio
4. **Scaling** — `StandardScaler` (used for LR and KNN; not needed for RF)
5. **Modeling** — train default and grid-tuned versions of each algorithm
6. **Evaluation** — classification report, confusion matrix, recall comparison

## Hyperparameter Tuning

All grid searches use `scoring='recall_macro'` and `cv=3`.

- **Logistic Regression** — `C`, `penalty`, `solver`, `class_weight` (separate grids per compatible solver/penalty pairing)
- **KNN** — `n_neighbors`, `weights`, `metric`
- **Random Forest** — `n_estimators`, `max_depth`, `min_samples_split`, `max_features`, `class_weight`

## Results (Test Set)

| Model | Recall Macro | Recall (Stroke) | Recall (No Stroke) |
|---|---|---|---|
| **RF Tuned** | **0.835** | **0.889** | 0.781 |
| LR Tuned | 0.814 | 0.852 | 0.776 |
| KNN Default | 0.578 | 0.185 | 0.970 |
| RF Default | 0.557 | 0.148 | 0.965 |
| LR Default | 0.512 | 0.074 | 0.950 |
| KNN Tuned | 0.500 | 0.074 | 0.925 |

## Key Takeaways

- **`class_weight='balanced'`** was the single most impactful hyperparameter for both LR and RF, lifting stroke recall from single digits to over 85%.
- **KNN underperformed** because it has no native class-weighting mechanism, leaving it unable to compensate for the imbalance.
- **Default models are dangerous** in imbalanced medical problems — they look accurate but miss almost all positive cases.

## Production Recommendation

**Tuned Random Forest** is recommended for production:

- Highest stroke recall (0.889) — catches the most real strokes
- Best recall macro (0.835) — fair across both classes
- Captures non-linear interactions between risk factors
- No scaling dependency, simpler to deploy

