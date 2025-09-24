# Project Overview — Email SPAM Analysis & Classification

## Goal

Build an end-to-end pipeline to **detect SPAM emails** and produce **content insights**:

* Train a robust SPAM classifier.
* Discover **topics** within predicted SPAM to understand campaigns/trends.
* Measure **semantic distances** between topics (heterogeneity).
* From **HAM** emails, extract **Organizations** (NER) for downstream BI.

---

## Data & Setup

* Dataset: 5,171 emails with `label`/`label_num` and raw `text` containing `Subject:` and body.
* Splits: **Stratified 80/20** hold-out test; **5-fold CV** on the training set.
* Reproducibility: fixed `random_state=42`; artifacts saved to disk (model + JSON).

---

## What we did in this notebook

### 1) Data Intake & Basic EDA

* Loaded the dataset, validated schema, checked nulls/lengths.
* Verified **label ↔ label_num** consistency (SPAM=1).
* Confirmed presence of `Subject:` in all rows.

### 2) In-depth EDA & Data Readiness

* **Duplicates**: identified exact duplicate texts; decided to **drop** duplicates in training.
* **Subject/Body parsing**: extracted `subject`, `body`; computed per-class stats (lengths, token counts, subject/body ratio).
* **Simple signals (meta-features)**: counts of digits, phones, exclamation marks, spammy words; subject flags (`Re:`, `Fwd:`, currency).
* **Formatting & leakage**: removed HTML (none detected), trimmed signatures/threads, and **sanitized PII** (URLs, emails, phones, long numbers) to avoid leakage and improve generalization.
* **Distinctive lexicon**: inspected top n-grams for SPAM vs HAM to guide feature choices.
* **Preprocessing decisions** “frozen” for modeling.

### 3) Feature Representation & Baseline Modeling

* **Text**: TF-IDF on **word 1–2 grams** (optionally + char 3–5 grams).
* **Meta-features**: standardized numerics (kept sparse end-to-end).
* **Pipelines**:
  * **A**: word n-grams + meta-features → LogisticRegression (`class_weight="balanced"`).
  * **B**: word + char n-grams + meta-features → LogisticRegression.
* Chose **Pipeline A** as the default baseline for simplicity/efficiency.

### 4) Topic Modeling on predicted SPAM

* Predicted SPAM with the baseline model; built a TF-IDF matrix (after custom stopwords).
* Evaluated **K** (e.g., {5,8,11,14,17,20}) via topic **stability** and reconstruction error.
* Selected **K ≈ 5** as a good trade-off.
* Computed **semantic distance** between topics (cosine on normalized H) and visualized a **topic map** (clusters/heatmap).

### 5) NER on HAM (Organizations)

* NER pipeline: spaCy model if available; otherwise a **rule-based** `EntityRuler` fallback (robust to missing models).
* Extracted **Organizations** from HAM; produced per-org counts and sample mentions (CSV).

### 6) Calibration, Thresholding, Final Fit & Artifacts

* **Calibration** with `CalibratedClassifierCV(method="sigmoid")` using 5-fold OOF predictions.
* **Robust OOF building**: always selected the positive class column by **class label** (not position), avoiding class order mix-ups.
* Threshold selection on calibrated OOF:
  * **F1-optimal (`f1_global`)**
  * **High-recall** (target recall≈0.95) → `high_recall_global`
  * **High-precision** (target precision≈0.98)
  * **Min-cost** (custom FP/FN costs)
* **Final training** on full training set; **evaluation** on hold-out test.
* **Results (calibrated)**: Hold-out **PR-AUC ≈ 0.987**, **ROC-AUC ≈ 0.995** (representative run).
* **Artifacts saved**:
  * `spam_model_v1.joblib` — full sklearn pipeline (features + classifier).
  * `spam_model_v1_artifacts.json` — thresholds (e.g., `high_recall_global ≈ 0.65`), meta-features, CV settings, test metrics.
  * Error slices: `test_false_positives.csv`, `test_false_negatives.csv`.
