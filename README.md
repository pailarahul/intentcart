# IntentCart: Online Purchase Intention

**ES 335 Machine Learning: Course Project**  
**Team:** Vishwak Mandaram (Roll: 25110354) · Paila Rahul (Roll: 25110222)  

* **Live Interactive Site:** [https://pailarahul.github.io/intentcart/](https://pailarahul.github.io/intentcart/)  
* **Source Code Repository:** [https://github.com/pailarahul/intentcart](https://github.com/pailarahul/intentcart)  

> **Research Question:** Given what an online shopper did during a single session on an e-commerce website, can browsing behaviour predict whether the session ends in a purchase? Which machine learning models beat simple baselines, where do they fail, and how much signal comes from target-leaking analytics variables?

---

## 1. Main Findings

| Setting | Selected Model (by Val PR-AUC) | Test PR-AUC [95% CI] | Majority Baseline | One-Rule Baseline |
|:---|:---|:---:|:---:|:---:|
| **Setting B: Without `PageValues`** *(Main Claim)* | **Gradient Boosting** | **0.372** [0.323, 0.434] | 0.155 | 0.213 |
| **Setting A: All Features** *(Optimistic Upper Bound)* | **Gradient Boosting** | **0.743** [0.692, 0.790] | 0.155 | 0.474 |

* **Modest Predictability:** Session behaviour predicts transactions significantly better than chance ($\approx 2.4\times$ lift over the 0.155 base rate) and outperforms a one-rule baseline, but predictions remain noisy.
* **Leakage Detection:** `PageValues` (a revenue-derived Google Analytics score) roughly doubles PR-AUC ($+0.371$). We prove this is target leakage: a decision stump on `PageValues` alone achieves PR-AUC 0.474, beating every full model in Setting B.
* **Model Parity:** In the honest setting (B), Logistic Regression, Random Forest, and Gradient Boosting are statistically indistinguishable (paired bootstrap 95% CIs cross zero). Under temporal distribution shift, Logistic Regression is the most robust.

---

## 2. Individual Understanding and Defense 

### Vishwak Mandaram (Roll: 25110354) · 50% Project Contribution

* **Primary Contribution:**  
  Designed the evaluation methodology (stratified 70/15/15 train/val/test split and metric selection); built the model exploration, grid-search tuning, and training pipeline; implemented 1,000-sample bootstrap confidence intervals and paired bootstrap hypothesis testing; conducted error analysis and built the linked interactive error inspector; ran 5-fold cross-validation and feature ablation experiments; authored website sections 6–13 and project reproducibility scripts.

* **Key Choices:**  
  1. *Metric Selection:* Selected **PR-AUC (Average Precision)** as the sole optimization target over ROC-AUC and Accuracy. With an 84.5% negative class, ROC-AUC gives a false sense of security ($\approx 0.88$) because True Negatives dilute False Positives in the FPR denominator, whereas PR-AUC directly tracks precision on the minority class.  
  2. *Strict Protocol Separation:* Tuned all 58 configurations per setting exclusively on the validation split and picked decision thresholds maximizing $F_1$ on validation. Evaluated the held-out test set **only once** to preserve absolute technical integrity.  
  3. *Paired Statistical Comparisons:* Evaluated model differences using paired test bootstraps (resampling identical test indices simultaneously) rather than comparing independent confidence intervals, providing valid hypothesis tests for model superiority.

* **Evaluation:**  
  On the held-out test set, Gradient Boosting reached PR-AUC 0.372 [0.323, 0.434] in Setting B. Using paired bootstrap differences, I evaluated whether the ensemble genuinely outperformed simpler models: $\Delta(\text{GBoost} - \text{LogReg}) = +0.015$ (95% CI $[-0.022, +0.052]$, $p > 0.05$) and $\Delta(\text{GBoost} - \text{Forest}) = -0.007$ (95% CI $[-0.045, +0.029]$). Because both intervals cross zero, I concluded that linear models and tree ensembles are statistically tied on this data. 5-fold cross-validation on train+val ($\text{SD} \approx 0.01\text{--}0.04$) corroborated that the gaps between top models are smaller than fold-level variance.

* **Identified Limitation:**  
  The 15% validation split contains 1,849 sessions but only 286 buyers. While this provides adequate statistical power to cleanly beat the one-rule baseline, it is too small to reliably separate Random Forest from Gradient Boosting: 5-fold CV slightly preferred Random Forest ($0.376$ vs $0.373$), while the single validation split selected Boosting ($0.428$ vs $0.418$). Model selection at this sample size exhibits non-negligible variance.

* **Reasoning About a Small Change (What-If):**  
  * *Change considered:* What if we replace the generic $F_1$-maximizing threshold with an asymmetric business cost matrix where a missed buyer costs $5\times$ more than a wasted follow-up notification ($C_{\text{FN}} = 5 \times C_{\text{FP}}$)?  
  * *Reasoned impact:* In Setting B, minimizing total cost on validation shifts the optimal decision threshold down from $0.312$ to $0.180$. When applied to the test set, recall increases dramatically from $51\%$ to $78\%$ (catching 223 out of 286 buyers), while precision drops from $34\%$ to $25\%$. The business catches 78 additional buyers at the cost of sending 240 extra non-buyer follow-ups. Total test misclassification cost ($\text{FP} + 5 \times \text{FN}$) drops from $867$ down to $569$ (a 34.4% reduction). This demonstrates that choosing an operationally aligned threshold yields vastly more practical value than trying to engineer marginal PR-AUC gains between model families.

---

### Paila Rahul (Roll: 25110222) · 50% Project Contribution

* **Primary Contribution:**  
  Formulated the research question and defined prediction-time constraints; conducted exploratory data analysis and data-quality audits (duplicates, skewness, seasonality, class balance); discovered, investigated, and isolated `PageValues` outcome leakage; established the two-setting experimental design (Setting A vs. Setting B); developed the scikit-learn preprocessing `Pipeline` (logarithmic transforms, scaling, infrequent category handling); executed temporal hold-out experiments and duplicate sensitivity analyses; authored website sections 1–5 and designed the interactive threshold explorer.

* **Key Choices:**  
  1. *Two-Setting Formulation:* Dissected the feature set into Setting A (all features) and Setting B (dropping `PageValues`). Identified that Google Analytics calculates `PageValues` retroactively using transaction revenue from sessions visiting those pages, creating direct target leakage. Formulating Setting B ensured all primary claims reflect genuine pre-purchase browsing behavior.  
  2. *Prediction-Time Scoping:* Strictly defined prediction time as **session completion**. Acknowledged that all variables are whole-session aggregates (e.g., total durations, session-average exit rates), making the system an end-of-session triage tool for email/push re-targeting, rather than a real-time mid-session clickstream predictor.  
  3. *Pipeline Preprocessing:* Designed a scikit-learn `ColumnTransformer` fitted strictly on training data: non-ordinal codes (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) were cast to categorical strings; categories appearing $<20$ times were merged into an "infrequent" bin; `log1p` was applied to right-skewed counts/durations before `StandardScaler` for Logistic Regression; trees received unscaled features.

* **Evaluation:**  
  Quantified the leakage of `PageValues` through permutation importance on validation: permuting `PageValues` caused a 0.370 drop in PR-AUC, whereas no other feature caused a drop exceeding 0.038. Evaluated temporal robustness by training on Feb–Sep (prevalence 13.8%), tuning thresholds on Oct (prevalence 22.8%), and testing on Nov–Dec (prevalence 23.3%). Logistic Regression outperformed Gradient Boosting on test PR-AUC ($0.419$ vs $0.370$), revealing that flexible non-linear models overfitted the seasonal traffic mix of earlier months, whereas the regularized linear model generalized better under macro-level temporal drift.

* **Identified Limitation:**  
  The UCI dataset lacks intra-session clickstream timestamps, page sequence order, and unique cart actions. We cannot evaluate dynamic in-session milestones (e.g., probability of purchase after viewing 3 vs 8 items) or trigger real-time interventions before the visitor closes the browser tab.

* **Reasoning About a Small Change (What-If):**  
  * *Change considered:* What if we drop the 125 duplicate rows identified during the exploratory data audit before splitting?  
  * *Reasoned impact:* All 125 duplicate rows represent zero-duration, 1-page bounce sessions with no transaction. Because the input feature space for immediate bounces is small, independent real visitors landing on a page and leaving immediately naturally produce identical rows. Deduplicating and refitting the pipeline yielded a Setting B test PR-AUC of $0.368$ (compared to $0.372$ with duplicates kept). The negligible difference ($\Delta = 0.004$, well within the 95% bootstrap CI of $\pm 0.05$) confirms that retaining duplicates preserves natural bounce traffic volume without skewing decision boundaries or creating split leakage.

---

## 3. Repository Layout
intentcart/ ├── data/ │ └── online_shoppers_intention.csv # UCI dataset (CC BY 4.0), committed for instant reproduction ├── src/ │ ├── intentcart.py # Data loading, quality checks, pipeline prep, model grids, metrics │ ├── run_all.py # Master pipeline: tune -> select -> test -> extras -> site data export │ └── download_data.py # Re-fetches CSV from UCI and verifies SHA-256 checksum ├── notebooks/ │ ├── IntentCart_analysis.ipynb # Executed notebook: EDA + results + independent model re-fit assertions │ └── make_notebook.py # Script to regenerate the analysis notebook programmatically ├── results/ │ ├── splits.csv # Row-level split mapping (row_id -> train / val / test) │ ├── eda.json # Summary statistics, skewness, and quality audit outputs │ ├── tuning_log.csv # Full log of all 116 hyperparameter trials across both settings │ ├── model_selection.csv # Best configuration per model family and validation metrics │ ├── test_results.csv # Final test metrics with 95% bootstrap confidence intervals │ ├── test_predictions.csv # Individual test-set predicted probabilities across all models │ ├── extra_experiments.json # CV, temporal hold-out, deduplication, ablation, calibration data │ ├── error_analysis.json # Segmented breakdown and representative FP/FN session profiles │ ├── figures/ # Generated charts (month rates, PageValues, PR curves, importance) │ └── run_log.txt # Terminal execution log verifying end-to-end run ├── docs/ # Static GitHub Pages web application │ ├── index.html # Complete project write-up with live SVG visualizations │ ├── assets/ # CSS styling, Tabler icons, interactive JavaScript │ └── data/ # site.json and site.js containing raw experiment data ├── requirements.txt # Pinned Python package dependencies └── README.md # Project documentation, individual defense, and instructions


---
## 4. Setup and Reproduction

To reproduce all tables, figures, and data files from scratch in approximately 3 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/pailarahul/intentcart.git
cd intentcart

# 2. Set up virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Re-download and verify SHA-256 data integrity
python src/download_data.py

# 4. Run the full machine learning experiment pipeline
python src/run_all.py

# 5. Execute the verification notebook (validates independent model re-fits)
jupyter nbconvert --to notebook --execute --inplace notebooks/IntentCart_analysis.ipynb
```

To inspect the web deliverable locally, open `docs/index.html` directly in any modern browser. It runs natively offline via `file://` because data is pre-compiled into `docs/data/site.js`.

---

## 5. Dataset Attribution and Technical Integrity

* **Source:** UCI Machine Learning Repository — *Online Shoppers Purchasing Intention Dataset* (Dataset ID: 468)  
  * Citation: Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). *Neural Computing and Applications*, 31, 6893–6908.  
  * DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q) · **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Privacy & Credentials:** The dataset contains zero Personally Identifiable Information (PII). Operating system, browser, region, and traffic source variables are anonymized integer codes. No API keys or private credentials are required.
* **Integrity Hash:** Every data load verifies the raw CSV against SHA-256:  
  `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

---

## 6. Experimental Protocol Summary

| Stage | Methodological Design |
|:---|:---|
| **Target & Unit** | Unit: One browsing session ($N=12,330$). Target: `Revenue` ($15.5\%$ positive, imbalanced). |
| **Prediction Time** | End of session. All variables are post-session aggregates. |
| **Split Design** | Fixed stratified 70% Train ($n=8,631$), 15% Validation ($n=1,849$), 15% Test ($n=1,850$), `seed=42`. Row assignments saved in `results/splits.csv`. |
| **Preprocessing** | Contained inside scikit-learn `Pipeline`, fit on training data only. Categoricals one-hot encoded (`min_frequency=20`). LogReg: `log1p` on skewed durations + `StandardScaler`. Trees: raw numerics. |
| **Model Families** | Majority class baseline, Decision stump (1-rule), Logistic Regression, Decision Tree, Random Forest (300 trees), Histogram-based Gradient Boosting. |
| **Hyperparameter Tuning** | 58 configurations per setting tuned strictly on validation PR-AUC. |
| **Threshold Selection** | Threshold chosen on validation split to maximize $F_1$, fixed before testing. |
| **Model Selection** | Highest validation PR-AUC per setting (declared in advance). |
| **Final Test Protocol** | Scored once on test data. 1,000 bootstrap resamples for 95% CIs and paired difference tests. |
| **Robustness Checks** | 5-fold CV on train+val; temporal hold-out (Feb–Sep train vs Nov–Dec test); deduplication check; GA rate ablation; validation permutation importance; test probability calibration. |

---

## 7. AI-Use Disclosure

We used **Claude (Anthropic)** as an interactive AI coding assistant during this project. Specifically, Claude assisted with:
* Brainstorming experimental framing and discussing metric tradeoffs for class imbalance.
* Drafting boilerplate Python code for scikit-learn pipelines and Matplotlib scripts.
* Scaffolding the responsive layout and CSS for the static HTML/JS deliverable.

**Human Oversight & Verification:** Every line of code was reviewed, debugged, and validated by the team. All experimental numbers were independently verified using `src/run_all.py` and confirmed via automated assertions in `notebooks/IntentCart_analysis.ipynb`. The conceptual reasoning, design decisions, and conclusions were formulated and defended by the authors.

---

## 8. References

1. Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31, 6893–6908.
2. Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
