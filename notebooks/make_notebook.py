"""Builds notebooks/IntentCart_analysis.ipynb (run, then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell
cells = [
M("""# IntentCart: Online Purchase Intention
ES 335 course project. This notebook walks through the data checks and EDA, then
loads the outputs of `src/run_all.py` and **independently re-fits the selected
models** from their saved hyper-parameters to confirm the reported test numbers.

Run `python src/run_all.py` first (it writes `results/`)."""),
C("""import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import intentcart as si
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
df = si.load_data()
print(df.shape); df.head()"""),
M("## 1. Data quality\nOne row = one browsing session. Checks below learn nothing, so they are safe to run on the full table."),
C("""q = si.quality_report(df); q"""),
C("""dups = df[df.duplicated(keep=False)]
print('extra duplicate copies:', df.duplicated().sum())
dups[['ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','Revenue']].describe().T"""),
M("""All duplicates are one-page bounce sessions with zero duration and no purchase. With so few
distinct values, different real visitors produce identical rows, so we **keep** them (they are
real traffic) and check in `extra_experiments.json` that removing them changes nothing material."""),
M("## 2. Target and key variables"),
C("""print(df.Revenue.value_counts(), df.Revenue.mean().round(4))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
r = df.groupby('Month', sort=False).Revenue.mean().reindex(si.MONTH_ORDER)
ax[0].bar(r.index, r.values*100); ax[0].set_title('Purchase rate by month (%)')
v = df.groupby('VisitorType').Revenue.mean()
ax[1].bar(v.index, v.values*100); ax[1].set_title('Purchase rate by visitor type (%)')
pv = df.groupby(df.PageValues > 0).Revenue.mean()
ax[2].bar(['PageValues=0','PageValues>0'], pv.values*100); ax[2].set_title('PageValues vs purchase (%)')
plt.tight_layout()"""),
C("""num = si.COUNT_DURATION + si.RATES + si.PAGE_VALUES
pd.DataFrame({'skew': df[num].skew().round(2), 'median_no_buy': df[df.Revenue==0][num].median(),
              'median_buy': df[df.Revenue==1][num].median(), 'max': df[num].max()})"""),
M("""Strong right skew in counts/durations motivates `log1p` before scaling for logistic regression.
`PageValues` is 0 for ~78% of sessions and nearly separates the classes; because Google
Analytics derives it from revenue on the same pages, we treat it as a leakage risk (setting A vs B)."""),
M("## 3. Split (fixed, stratified 70/15/15, seed 42)"),
C("""split = pd.read_csv(ROOT/'results/splits.csv')
assert (split.split.values == si.make_split(df).values).all()   # split is reproducible
split.groupby('split').Revenue.agg(['size','sum','mean'])"""),
M("## 4. Validation results (model selection)"),
C("""sel = pd.read_csv(ROOT/'results/model_selection.csv')
sel[['setting','model','n_configs','params','val_pr_auc','val_f1','val_threshold','selected']].round(3)"""),
M("## 5. Test results (each tuned model scored once, at its validation threshold)"),
C("""tr = pd.read_csv(ROOT/'results/test_results.csv')
tr[['setting','model','selected','test_pr_auc','ci_pr_auc','test_roc_auc','test_precision','test_recall','test_f1','test_accuracy']].round(3)"""),
M("## 6. Independent re-fit check\nRe-build the selected models from saved parameters, train on the train split only, and score on test."),
C("""from sklearn.metrics import average_precision_score
tr_df, te_df = df[split.split.values=='train'], df[split.split.values=='test']
for s in ['A','B']:
    row = sel[(sel.setting==s) & sel.selected].iloc[0]
    feats = si.features_for(s)
    pipe = si.build_model(row.model, json.loads(row.params), feats).fit(tr_df[feats], tr_df.Revenue)
    ap = average_precision_score(te_df.Revenue, pipe.predict_proba(te_df[feats])[:,1])
    rep = tr[(tr.setting==s)&(tr.model==row.model)].test_pr_auc.iloc[0]
    print(s, row.model, 'refit PR-AUC', round(ap,4), 'reported', round(rep,4))
    assert abs(ap-rep) < 1e-6"""),
M("## 7. Extra experiments"),
C("""ex = json.load(open(ROOT/'results/extra_experiments.json'))
print('5-fold CV on train+val (PR-AUC mean ± sd):')
for s in ['A','B']:
    print(s, {m: f"{v['mean']:.3f}±{v['sd']:.3f}" for m, v in ex['cv_trainval'][s].items()})
print('\\nTemporal hold-out (train Feb-Sep, threshold on Oct, test Nov-Dec):')
print(json.dumps(ex['temporal'], indent=1)[:1500])
print('\\nDeduplicated re-run (test PR-AUC):', ex['dedup'])
print('Ablation without BounceRates/ExitRates:', ex['ablation_no_rates'])"""),
C("""from IPython.display import Image, display
for f in ['pr_curves_test.png','permutation_importance.png']:
    display(Image(filename=str(ROOT/'results/figures'/f)))"""),
M("## 8. Error analysis (selected model, setting B)"),
C("""err = json.load(open(ROOT/'results/error_analysis.json'))
print(err['B']['counts'])
display(pd.DataFrame(err['B']['median_by_outcome']).T)
display(pd.DataFrame(err['B']['segments']['VisitorType']))
display(pd.DataFrame(err['B']['confident_fp'])); display(pd.DataFrame(err['B']['confident_fn']))"""),
]
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("IntentCart_analysis.ipynb")
nbf.write(nb, out)
print("wrote", out)
