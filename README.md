# IntentCart: Online Purchase Intention

**ES 335 Machine Learning: course project**
Vishwak Mandaram (25110354) · Paila Rahul (25110222)

**Live website:** https://pailarahul.github.io/intentcart/  ← summarises the reasoning; this repository lets you check it.

> **Question.** Given what a visitor did during one session on an online store, can we predict whether the session ended in a purchase?
> Which models beat simple baselines, and where do they fail?

## Main findings

| Setting | Selected model (by validation PR-AUC) | Test PR-AUC [95% CI] | Majority baseline | One-rule baseline |
|---|---|---|---|---|
| **B: without `PageValues`** (main claim) | Gradient boosting | **0.372** [0.323, 0.434] | 0.155 | 0.213 |
| A: all features (optimistic upper bound) | Gradient boosting | 0.743 [0.692, 0.790] | 0.155 | 0.474 |

* Browsing behaviour predicts purchases better than chance and better than a one-rule baseline, but only modestly.
* `PageValues` (a Google Analytics score derived from revenue) roughly doubles PR-AUC. We treat that gain as probable leakage, not behavioural signal.
* Logistic regression, random forest and gradient boosting are statistically indistinguishable in setting B (paired bootstrap CIs include 0). Under a temporal hold-out, logistic regression is the most stable.

Full results, error analysis and limitations are on the website. Exact numbers are in `results/`.

## Repository layout

```
data/online_shoppers_intention.csv   UCI dataset (CC BY 4.0), committed for convenience
src/intentcart.py                    loading, checks, split, preprocessing, model grids, metrics
src/run_all.py                       the whole experiment: tune → select → test → extras → site data
src/download_data.py                 re-downloads the CSV from UCI and verifies its SHA-256
notebooks/IntentCart_analysis.ipynb  EDA + reads results + independent re-fit check (executed)
notebooks/make_notebook.py           regenerates the notebook
results/                             every output (see table below)
docs/                                the GitHub Pages website (static HTML/CSS/JS)
```

## Setup and run

```bash
git clone https://github.com/pailarahul/intentcart.git
cd intentcart
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py   # optional: the CSV is already in data/; this re-fetches and verifies it
python src/run_all.py         # ~3 minutes; rewrites results/ and docs/data/
jupyter nbconvert --to notebook --execute --inplace notebooks/IntentCart_analysis.ipynb
```

To view the website locally, open `docs/index.html` in a browser. It also works from `file://` because the data ships as `docs/data/site.js`.

## Data access

* Source: UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset*, id 468
  https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset · DOI 10.24432/C5F88Q
* Licence: CC BY 4.0 (redistribution with attribution permitted). It contains no personal data.
* File used: `online_shoppers_intention.csv`, 12,330 rows × 18 columns,
  SHA-256 `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9` (checked on every load).

## Exact protocol

| Item | Detail |
|---|---|
| Unit / target | One browsing session / `Revenue` (1 = purchase), 15.5% positive |
| Prediction time | End of session (all features are whole-session aggregates) |
| Feature settings | **A** all 17 features; **B** drops `PageValues` (possible outcome leakage) |
| Split | Stratified 70/15/15 train/val/test, `random_state=42` (`intentcart.make_split`). Row-level assignment saved in `results/splits.csv` |
| Data checks | Missing values (none), duplicates (125 extra copies, all one-page bounces, **kept**), skew, coverage. These learn nothing |
| Preprocessing | Inside a scikit-learn `Pipeline`, fitted on train only. LogReg: `log1p` on counts/durations, then `StandardScaler`. All models: one-hot categoricals with `min_frequency=20`, `handle_unknown="infrequent_if_exist"`. Trees: raw numerics |
| Models | Majority (`DummyClassifier`), one-rule (depth-1 tree), logistic regression, decision tree, random forest (300 trees), `HistGradientBoostingClassifier` |
| Tuning | Small grids (`intentcart.model_grid`): 58 configs per setting, each fitted on train and scored on validation by PR-AUC |
| Threshold | F1-maximising threshold on validation, per model |
| Selection | Highest validation PR-AUC per setting (rule fixed in advance) |
| Test | Every tuned model scored once on test at its validation threshold. 1,000× bootstrap 95% CIs, paired bootstrap for model differences |
| Extra experiments | 5-fold CV on train+val; temporal hold-out (train Feb–Sep, threshold Oct, test Nov–Dec); dedup re-run; ablation without Bounce/Exit rates; permutation importance (validation); calibration |

## Outputs (`results/`)

| File | Content |
|---|---|
| `splits.csv` | `row_id` (0-based row in the CSV) → `train` / `val` / `test` |
| `eda.json` | Data-quality checks and summaries |
| `tuning_log.csv` | Every configuration tried and its validation PR-AUC |
| `model_selection.csv` | Best config per model, validation metrics, threshold, selection flag |
| `test_results.csv` | Test metrics with bootstrap CIs for every tuned model |
| `test_predictions.csv` | Test-set probability from every model in both settings |
| `extra_experiments.json` | CV, temporal hold-out, dedup, ablation, permutation importance, LR coefficients, calibration |
| `error_analysis.json` | Segment error rates, outcome medians, most confident FP/FN sessions |
| `figures/*.png` | Static figures (month rates, PageValues, PR curves, importance) |
| `run_log.txt` | Console log of the run that produced these files |

## Website

The site in `docs/` is plain HTML/CSS/JS with no build step. To publish it: repository **Settings → Pages → Deploy from a branch → `main` / `/docs`**.
Then set `REPO_URL` at the top of `docs/assets/app.js` to your repository URL.
All numbers on the page are read from `docs/data/site.js`, so re-running `src/run_all.py` updates the site automatically.

Interactions: (1) a **threshold explorer** that recomputes the confusion matrix, precision/recall/F1 and PR-curve position from saved test predictions, with a validation-based cost-ratio threshold; (2) a linked **error inspector** listing the most confident false positives and negatives with their features.

## Contributions (equal, 50% each)

| Member | Contribution |
|---|---|
| Vishwak Mandaram (25110354) | Evaluation design (split, metrics, validation-only tuning and thresholds); baselines and model training/tuning; bootstrap CIs and paired comparisons; error analysis and the error inspector; cross-validation and ablation checks; website sections 6–13, README and reproducibility |
| Paila Rahul (25110222) | Problem framing and prediction-time definition; data-quality checks and EDA; PageValues leakage analysis and the two feature settings; preprocessing pipeline; temporal hold-out and duplicate-sensitivity experiments; website sections 1–5 and the threshold explorer |

Both members discussed every major choice together, reviewed each other's work, and can explain the full project.

## AI-use disclosure

We used **Claude (Anthropic)** for brainstorming the evaluation design, drafting and debugging the Python code, drafting website text, and building the website's HTML/CSS/JS. We checked every reported number by re-running `src/run_all.py` and the independent re-fit check in the notebook (it asserts that re-fitted test PR-AUC matches the reported value). We reviewed all reasoning and take responsibility for it. No external code was copied.

## Citations

* Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31, 6893–6908.
* Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
* Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
* Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10(3): e0118432.
* Libraries: NumPy, pandas, Matplotlib, Jupyter.
