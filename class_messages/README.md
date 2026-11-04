# Student Placement Prediction

A machine learning pipeline that predicts whether a registered student will receive a campus placement offer (`placed = 1`) or not (`placed = 0`). The model is optimized for the **F1 score on the positive class**, and ships with a fully self-contained batch inference script.

## Results at a Glance

| Metric (5-fold stratified CV, out-of-fold) | Value |
| :--- | :---: |
| F1 (placed = 1) | **0.9251** |
| ROC-AUC | **0.9755** |
| Precision / Recall | 0.9270 / 0.9231 |
| Accuracy | 0.9580 |
| Decision threshold | 0.485 |

The `status_code`-only baseline heuristic scores F1 = 0.9201. The final model improves on it by resolving the cases where `status_code` disagrees with the outcome.

## Repository Structure

```text
.
├── analysis.ipynb        # EDA, feature engineering, model comparison, threshold tuning
├── predict.py            # Standalone batch inference script (model embedded)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── data/
    ├── train.csv             # Training data (2,500 students)
    └── sample_submission.csv # Expected output layout
```

## Dataset

Each row is one student who registered with the placement cell.

| Column | Description |
| --- | --- |
| `student_id` | Student identifier |
| `department` | CSE, IT, ECE, EEE, MECH, or CIVIL |
| `city_tier` | 1, 2, or 3 |
| `cgpa` | Cumulative GPA |
| `backlogs` | Active backlogs |
| `internship_months` | Internship length in months |
| `coding_score` | Internal coding test score (0-100) |
| `projects` | Number of projects listed |
| `workshops` | Number of workshops attended |
| `status_code` | Field stored on the placement portal |
| `placed` | Target: 1 if placed, 0 otherwise (train only) |

**Key data facts**
- Class imbalance: 702 placed (28.08%) vs. 1,798 unplaced (71.92%).
- Missing values: `coding_score` (21.0%), `backlogs` (6.7%), `cgpa` (4.7%), `status_code` (4.5%).
- `status_code` is highly predictive but not perfect: `0` is 98.0% unplaced (33 exceptions), `1` is 92.1% placed (55 exceptions), and missing is 21.2% placed.
- Placement rates vary by department, from 38.4% (CSE) down to 13.6% (CIVIL).

## Approach

**Feature engineering**
- Missingness indicators: `cgpa_missing`, `coding_score_missing`, `backlogs_missing`, `status_code_missing`
- `cgpa_backlog_ratio` = cgpa / (backlogs + 1)
- `coding_per_cgpa` = coding_score / (cgpa + 0.1)
- `practical_score` = 2.0 × internship_months + 1.5 × projects + 0.5 × workshops
- `tech_composite` = coding_score / 10 + 1.5 × cgpa + 1.5 × internship_months
- `dept_tier`: ordinal department ranking (CSE=5 ... CIVIL=0), plus one-hot department
- `status_code` filled with a `-1` sentinel when missing

**Preprocessing:** median imputation, standard scaling, and one-hot encoding with `handle_unknown='ignore'`, all fitted inside each CV fold to avoid leakage.

**Models compared** (5-fold stratified CV, out-of-fold predictions):

| Model | ROC-AUC | Optimal F1 |
| :--- | :---: | :---: |
| Weighted blend (LR + XGB + LGB) | 0.9751 | 0.9256 |
| **Logistic Regression (C=0.1)** | **0.9755** | 0.9251 |
|
