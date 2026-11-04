## GoML Assessment Submissions

This repository holds my submissions for two take-home cases from the GoML evaluation. Each case is a self-contained project in its own folder, with its own README, code, and notes.

## Why this repo exists

Each case asked for working code plus a written explanation of the choices behind it. Both were graded on a hidden dataset, so the evaluators ran the submitted code on data I never saw. I put them in one repository so the work can be reviewed in one place.

The two cases cover different kinds of ML work:

| | Case 1 | Case 2 |
| :--- | :--- | :--- |
| **Problem** | Tabular classification | Information extraction with a local LLM |
| **Task** | Predict whether a student gets placed | Turn class messages into a JSON task list |
| **Approach** | Classical ML: feature engineering, cross-validated model comparison, threshold tuning | Local LLM (`qwen2.5:1.5b` via Ollama) with Python doing the heavy lifting |
| **Metric** | F1 on the positive class | Correct tasks, no invented names or deadlines, identical output across runs |
| **Result** | CV F1 0.9251, ROC-AUC 0.9755 | Byte-identical output across runs; injection and hallucination guards in place |
| **Folder** | [`case-1-placement-prediction/`](./case-1-placement-prediction) | [`case-2-task-extractor/`](./case-2-task-extractor) |

## Case 1: Student Placement Prediction

Predicts whether a registered student receives a placement offer, using academic, technical, and portal data. The submission is a trained pipeline embedded in a standalone `predict.py`, plus an analysis notebook.

- Compared nine approaches (logistic regression, tree ensembles, boosting, stacking, a blend, and a heuristic baseline) with 5-fold stratified CV.
- Picked a regularized logistic regression with a tuned decision threshold. Its AUC is the highest of the models compared, and its F1 is within noise of the best, but it is simpler and has no native-library dependencies.
- Investigated a suspicious `status_code` field, quantified how much the model leans on it, and documented where it is weak (rows where it is missing).

Details: [`case-1-placement-prediction/README.md`](./case-1-placement-prediction/README.md)

## Case 2: Local-LLM Class Task Extractor

Reads a folder of short class messages and writes one JSON task list, using only a local model. No hosted APIs.

- Built for a CPU-only, low-RAM machine. A small model handles narrow classification and extraction, and Python handles sanitizing, candidate extraction, and validation.
- Two-pass design: classify first, then extract only for actionable messages.
- Layered defenses against prompt injection and hallucination. Owners must appear in the message, and deadlines must come from a regex-extracted candidate list or be `null`.
- Deterministic by construction: fixed seed, zero temperature, schema-constrained output. Two runs were verified byte-identical.

Details: [`case-2-task-extractor/README.md`](./case-2-task-extractor/README.md)

## What the two cases share

Different problems, same engineering priorities:

- **Reproducibility.** Fixed seeds and deterministic settings, with checks to confirm it.
- **Grounding over guessing.** Predictions and extractions are checked against the actual input, and missing information becomes `null` or a documented fallback.
- **Defensive handling of unseen data.** Both programs run on a hidden dataset, so they tolerate missing values, malformed files, and unexpected categories without crashing.
- **Documented trade-offs.** Each folder explains why a choice was made, and where the approach is weak.

## Repository layout

```text
.
├── README.md                      # This file
├── case-1-placement-prediction/
│   ├── analysis.ipynb
│   ├── predict.py
│   ├── requirements.txt
│   └── README.md
└── case-2-task-extractor/
    ├── extract.py
    ├── NOTES.md
    ├── requirements.txt
    └── README.md
```

## Running each case

Each case has its own dependencies, so install them per folder.

```bash
# Case 1
cd case-1-placement-prediction
pip install -r requirements.txt
python predict.py --input data/train.csv --output predictions.csv

# Case 2 (requires Ollama running locally)
cd case-2-task-extractor
ollama pull qwen2.5:1.5b
pip install -r requirements.txt
python extract.py --inbox messages --output tasks.json
```

## Notes

- The training data and practice messages come from the assessment materials. Check before making this repository public whether you are allowed to publish them.
- The metrics reported here are from my own validation runs. The scores on the evaluators' hidden data may differ.
