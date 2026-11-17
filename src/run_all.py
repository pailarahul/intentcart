"""Run the full IntentCart analysis end to end.

    python src/run_all.py            # ~2-4 minutes on a laptop

Outputs
  results/splits.csv                 row id -> train / val / test (exact split)
  results/eda.json                   data-quality checks and summaries
  results/tuning_log.csv             every hyper-parameter config, validation scores
  results/model_selection.csv        best config per model + validation metrics
  results/test_results.csv           test metrics (+ bootstrap 95% CIs) for tuned models
  results/extra_experiments.json     CV, temporal hold-out, duplicates, ablation, ...
  results/error_analysis.json        segments + representative errors
  results/test_predictions.csv       test-set probabilities for every model
  results/figures/*.png              static figures
  docs/data/site.json                everything the website needs
Protocol (fixed before looking at test data)
  1. Fit every config on TRAIN, score on VAL by PR-AUC (average precision).
  2. Keep the best config per model family; choose the decision threshold that
     maximises F1 on VAL.
  3. The model with the highest VAL PR-AUC in each feature setting is "selected".
  4. Score all tuned models ONCE on TEST with the VAL threshold.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intentcart as si  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

RES = si.ROOT / "results"
FIG = RES / "figures"
SITE = si.ROOT / "docs" / "data"
for d in (RES, FIG, SITE):
    d.mkdir(parents=True, exist_ok=True)

MODELS = ["majority", "stump", "logreg", "tree", "forest", "gboost"]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def r(x, k=4):
    return None if x is None else round(float(x), k)


def tune(setting, Xtr, ytr, Xva, yva, log):
    """Grid search on the validation split. Returns {model: info}."""
    feats = si.features_for(setting)
    out = {}
    for m in MODELS:
        best = None
        for params in si.model_grid(m):
            pipe = si.build_model(m, params, feats).fit(Xtr[feats], ytr)
            p = pipe.predict_proba(Xva[feats])[:, 1]
            ap = average_precision_score(yva, p)
            log.append({"setting": setting, "model": m, "params": json.dumps(params),
                        "val_pr_auc": ap})
            if best is None or ap > best["val_pr_auc"] + 1e-12:
                best = {"params": params, "val_pr_auc": ap, "pipe": pipe, "p_val": p}
        t = si.f1_optimal_threshold(yva, best["p_val"]) if m != "majority" else 0.5
        best["threshold"] = t
        best["val_metrics"] = {**si.ranking_metrics(yva, best["p_val"]),
                               **si.threshold_metrics(yva, best["p_val"], t)}
        out[m] = best
        print(f"  [{setting}] {m:9s} val PR-AUC={best['val_pr_auc']:.3f} "
              f"t={t:.3f} params={best['params']}")
    return out


def main():
    t0 = time.time()
    df = si.load_data()
    q = si.quality_report(df)          # before adding any helper columns
    df["row_id"] = np.arange(len(df))

    # ---------------------------------------------------------------- EDA
    month = (df.groupby("Month")[si.TARGET].agg(["size", "mean"])
             .reindex(si.MONTH_ORDER).reset_index())
    visitor = df.groupby("VisitorType")[si.TARGET].agg(["size", "mean"]).reset_index()
    pv_pos = df.PageValues > 0
    num_cols = si.COUNT_DURATION + si.RATES + si.PAGE_VALUES + si.OTHER_NUMERIC
    by_class = df.groupby(si.TARGET)[num_cols].median().T
    eda = {
        "quality": q,
        "month": [{"month": a, "n": int(b), "rate": r(c)} for a, b, c in month.values],
        "visitor": [{"type": a, "n": int(b), "rate": r(c)} for a, b, c in visitor.values],
        "weekend": [{"weekend": int(k), "n": int(v["size"]), "rate": r(v["mean"])}
                    for k, v in df.groupby("Weekend")[si.TARGET].agg(["size", "mean"]).iterrows()],
        "pagevalues": {
            "share_zero": r((~pv_pos).mean()),
            "rate_if_zero": r(df.loc[~pv_pos, si.TARGET].mean()),
            "rate_if_positive": r(df.loc[pv_pos, si.TARGET].mean()),
            "n_zero": int((~pv_pos).sum()), "n_positive": int(pv_pos.sum()),
            "purchases_with_zero_pv": int(df.loc[~pv_pos, si.TARGET].sum()),
        },
        "median_by_class": [{"feature": f, "no_purchase": r(row[0], 3),
                             "purchase": r(row[1], 3)} for f, row in by_class.iterrows()],
        "skew": {c: r(df[c].skew(), 2) for c in num_cols},
        "max": {c: r(df[c].max(), 2) for c in num_cols},
    }
    (RES / "eda.json").write_text(json.dumps(eda, indent=2))
    print("EDA done", q)

    # -------------------------------------------------------------- split
    split = si.make_split(df)
    df["split"] = split.values
    df[["row_id", "split", si.TARGET]].to_csv(RES / "splits.csv", index=False)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ytr, yva, yte = tr[si.TARGET].values, va[si.TARGET].values, te[si.TARGET].values
    split_info = {s: {"n": int((df.split == s).sum()),
                      "positives": int(df.loc[df.split == s, si.TARGET].sum()),
                      "rate": r(df.loc[df.split == s, si.TARGET].mean())}
                  for s in ("train", "val", "test")}
    print("Split", split_info)

    # ---------------------------------------------------- tune + validate
    log, tuned = [], {}
    for s in si.SETTINGS:
        tuned[s] = tune(s, tr, ytr, va, yva, log)
    pd.DataFrame(log).to_csv(RES / "tuning_log.csv", index=False)

    selected = {s: max([m for m in MODELS if m not in ("majority", "stump")],
                       key=lambda m: tuned[s][m]["val_pr_auc"]) for s in si.SETTINGS}
    print("Selected on validation:", selected)

    sel_rows = []
    for s in si.SETTINGS:
        for m in MODELS:
            b = tuned[s][m]
            sel_rows.append({"setting": s, "model": m, "params": json.dumps(b["params"]),
                             "selected": m == selected[s], "n_configs": len(si.model_grid(m)),
                             **{f"val_{k}": v for k, v in b["val_metrics"].items()}})
    pd.DataFrame(sel_rows).to_csv(RES / "model_selection.csv", index=False)

    # --------------------------------------------------------- test (once)
    test_rows, preds = [], {"row_id": te.row_id.values, "y": yte}
    site_models = {}
    for s in si.SETTINGS:
        feats = si.features_for(s)
        site_models[s] = {}
        for m in MODELS:
            b = tuned[s][m]
            p = b["pipe"].predict_proba(te[feats])[:, 1]
            b["p_test"] = p
            preds[f"{s}_{m}"] = p
            tm = {**si.ranking_metrics(yte, p), **si.threshold_metrics(yte, p, b["threshold"])}
            ci = si.bootstrap_ci(yte, p, b["threshold"]) if m != "majority" else None
            b["test_metrics"], b["test_ci"] = tm, ci
            test_rows.append({"setting": s, "model": m, "selected": m == selected[s],
                              **{f"test_{k}": v for k, v in tm.items()},
                              **({f"ci_{k}": v for k, v in ci.items()} if ci else {})})
            site_models[s][m] = {
                "label": si.MODEL_INFO[m], "params": b["params"],
                "n_configs": len(si.model_grid(m)), "threshold": r(b["threshold"]),
                "val": {k: r(v) for k, v in b["val_metrics"].items()},
                "test": {k: r(v) for k, v in tm.items()},
                "ci": {k: [r(a), r(c)] for k, (a, c) in ci.items()} if ci else None,
                "p_test": [r(x) for x in p],
                "p_val": [r(x) for x in b["p_val"]],
            }
    pd.DataFrame(test_rows).to_csv(RES / "test_results.csv", index=False)
    pd.DataFrame(preds).to_csv(RES / "test_predictions.csv", index=False)

    comparisons = {}
    for s in si.SETTINGS:
        ps = tuned[s][selected[s]]["p_test"]
        comparisons[s] = {
            f"{selected[s]}_minus_{o}": si.paired_bootstrap_diff(yte, ps, tuned[s][o]["p_test"])
            for o in ("stump", "logreg", "tree", "forest", "gboost") if o != selected[s]}
    comparisons["A_minus_B_selected"] = si.paired_bootstrap_diff(
        yte, tuned["A"][selected["A"]]["p_test"], tuned["B"][selected["B"]]["p_test"])

    # ------------------------------------------------------ extra experiments
    extra = {}
    # (a) 5-fold CV on train+val with the tuned configs: is the ranking stable?
    trva = pd.concat([tr, va])
    ytv = trva[si.TARGET].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=si.SEED)
    cv = {}
    for s in si.SETTINGS:
        feats = si.features_for(s)
        cv[s] = {}
        for m in MODELS:
            scores = []
            for a, b_ in skf.split(trva, ytv):
                pipe = si.build_model(m, tuned[s][m]["params"], feats)
                pipe.fit(trva.iloc[a][feats], ytv[a])
                scores.append(average_precision_score(ytv[b_], pipe.predict_proba(trva.iloc[b_][feats])[:, 1]))
            cv[s][m] = {"mean": r(np.mean(scores)), "sd": r(np.std(scores, ddof=1)),
                        "folds": [r(x) for x in scores]}
    extra["cv_trainval"] = cv
    print("CV done")

    # (b) Temporal hold-out: train Feb-Sep, threshold on Oct, test Nov-Dec.
    mi = df.Month.map({m: i for i, m in enumerate(si.MONTH_ORDER)})
    ttr = df[mi <= si.MONTH_ORDER.index("Sep")]
    tva = df[df.Month == "Oct"]
    tte = df[df.Month.isin(["Nov", "Dec"])]
    temporal = {"train": {"months": "Feb-Sep", "n": len(ttr), "rate": r(ttr[si.TARGET].mean())},
                "val": {"months": "Oct", "n": len(tva), "rate": r(tva[si.TARGET].mean())},
                "test": {"months": "Nov-Dec", "n": len(tte), "rate": r(tte[si.TARGET].mean())},
                "results": {}}
    for s in si.SETTINGS:
        # Month is constant-per-split here, so it cannot help; drop it.
        feats = [f for f in si.features_for(s) if f != "Month"]
        temporal["results"][s] = {}
        for m in ("stump", "logreg", "forest", "gboost"):
            pipe = si.build_model(m, tuned[s][m]["params"], feats).fit(ttr[feats], ttr[si.TARGET])
            pv = pipe.predict_proba(tva[feats])[:, 1]
            t = si.f1_optimal_threshold(tva[si.TARGET].values, pv)
            pt = pipe.predict_proba(tte[feats])[:, 1]
            yt = tte[si.TARGET].values
            temporal["results"][s][m] = {
                "pr_auc": r(average_precision_score(yt, pt)),
                **{k: r(v) for k, v in si.threshold_metrics(yt, pt, t).items()
                   if k in ("threshold", "precision", "recall", "f1")}}
    extra["temporal"] = temporal
    print("Temporal done")

    # (c) Duplicates: drop the 125 extra copies, re-split, refit tuned configs.
    dd = df.drop_duplicates(subset=[c for c in df.columns if c not in ("row_id", "split")]).reset_index(drop=True)
    dsplit = si.make_split(dd)
    dtr, dte = dd[dsplit.values == "train"], dd[dsplit.values == "test"]
    dup = {"n_rows": len(dd), "results": {}}
    for s in si.SETTINGS:
        feats = si.features_for(s)
        dup["results"][s] = {}
        for m in ("logreg", selected[s]):
            pipe = si.build_model(m, tuned[s][m]["params"], feats).fit(dtr[feats], dtr[si.TARGET])
            dup["results"][s][m] = r(average_precision_score(dte[si.TARGET], pipe.predict_proba(dte[feats])[:, 1]))
    extra["dedup"] = dup

    # (d) Ablation C: also drop the page-level GA rates (BounceRates, ExitRates).
    feats_c = [f for f in si.features_for("B") if f not in si.RATES]
    abl = {}
    for m in ("logreg", "forest", "gboost"):
        best = None
        for params in si.model_grid(m):
            pipe = si.build_model(m, params, feats_c).fit(tr[feats_c], ytr)
            ap = average_precision_score(yva, pipe.predict_proba(va[feats_c])[:, 1])
            if best is None or ap > best[0]:
                best = (ap, pipe)
        abl[m] = {"val_pr_auc": r(best[0]),
                  "test_pr_auc": r(average_precision_score(yte, best[1].predict_proba(te[feats_c])[:, 1]))}
    extra["ablation_no_rates"] = abl

    # (e) Permutation importance on VALIDATION for the selected models.
    imp = {}
    for s in si.SETTINGS:
        feats = si.features_for(s)
        pi = permutation_importance(tuned[s][selected[s]]["pipe"], va[feats], yva,
                                    scoring="average_precision", n_repeats=10,
                                    random_state=si.SEED, n_jobs=-1)
        order = np.argsort(-pi.importances_mean)
        imp[s] = [{"feature": feats[i], "mean": r(pi.importances_mean[i]),
                   "sd": r(pi.importances_std[i])} for i in order]
    extra["permutation_importance_val"] = imp

    # (f) Logistic regression coefficients (setting B), on standardised inputs.
    lr = tuned["B"]["logreg"]["pipe"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:15]
    extra["logreg_coefficients_B"] = [{"feature": str(names[i]), "coef": r(coefs[i])} for i in order]

    # (g) Calibration of selected models on test.
    cal = {}
    for s in si.SETTINGS:
        p = tuned[s][selected[s]]["p_test"]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        cal[s] = [{"bin": f"{bins[k]:.1f}-{bins[k+1]:.1f}", "n": int((idx == k).sum()),
                   "mean_pred": r(p[idx == k].mean()), "observed": r(yte[idx == k].mean())}
                  for k in range(10) if (idx == k).sum() > 0]
    extra["calibration_test"] = cal
    (RES / "extra_experiments.json").write_text(json.dumps(extra, indent=2))
    print("Extras done")

    # --------------------------------------------------------- error analysis
    err = {}
    show_cols = ["Month", "VisitorType", "Weekend", "Administrative", "Informational",
                 "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
                 "PageValues", "SpecialDay"]
    for s in si.SETTINGS:
        b = tuned[s][selected[s]]
        p, t = b["p_test"], b["threshold"]
        yhat = (p >= t).astype(int)
        kind = np.where((yhat == 1) & (yte == 1), "TP", np.where((yhat == 1) & (yte == 0), "FP",
                        np.where((yhat == 0) & (yte == 1), "FN", "TN")))
        e = te.copy()
        e["p"], e["kind"] = p, kind
        segs = {}
        for col, fn in [("VisitorType", None), ("Month", None), ("PageValues>0", lambda d: (d.PageValues > 0).map({True: "PageValues > 0", False: "PageValues = 0"})),
                        ("Weekend", lambda d: d.Weekend.map({0: "Weekday", 1: "Weekend"}))]:
            key = fn(e) if fn else e[col]
            rows = []
            for k, g in e.groupby(key):
                pos, neg = g[si.TARGET] == 1, g[si.TARGET] == 0
                rows.append({"segment": str(k), "n": int(len(g)), "positives": int(pos.sum()),
                             "recall": r((g.kind[pos] == "TP").mean()) if pos.any() else None,
                             "fp_rate": r((g.kind[neg] == "FP").mean()) if neg.any() else None,
                             "precision": r((g.kind == "TP").sum() / max(1, (g.kind.isin(["TP", "FP"])).sum()))})
            if col == "Month":
                rows.sort(key=lambda d: si.MONTH_ORDER.index(d["segment"]))
            segs[col] = rows
        med = e.groupby("kind")[num_cols].median().reindex(["TP", "FN", "FP", "TN"])
        def pick(k, asc):
            g = e[e.kind == k].sort_values("p", ascending=asc).head(4)
            return [{"row_id": int(x.row_id), "p": r(x.p), **{c: (x[c] if isinstance(x[c], str) else r(x[c], 3)) for c in show_cols}} for _, x in g.iterrows()]
        err[s] = {"model": selected[s], "threshold": r(t),
                  "counts": {k: int((kind == k).sum()) for k in ("TP", "FP", "FN", "TN")},
                  "segments": segs,
                  "median_by_outcome": {k: {c: r(med.loc[k, c], 3) for c in num_cols} for k in med.index},
                  "confident_fp": pick("FP", False), "confident_fn": pick("FN", True)}
    (RES / "error_analysis.json").write_text(json.dumps(err, indent=2, default=str))

    # ---------------------------------------------------------------- figures
    make_figures(df, eda, tuned, selected, yte, imp)

    # ------------------------------------------------------- site export
    test_rows_site = te[["row_id", "Month", "VisitorType", "Weekend", "Administrative",
                         "Administrative_Duration", "Informational", "ProductRelated",
                         "ProductRelated_Duration", "BounceRates", "ExitRates",
                         "PageValues", "SpecialDay", si.TARGET]].copy()
    for c in test_rows_site.select_dtypes("float").columns:
        test_rows_site[c] = test_rows_site[c].round(3)
    site = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "settings": {k: {kk: vv for kk, vv in v.items()} | {"features": si.features_for(k)}
                     for k, v in si.SETTINGS.items()},
        "split": split_info, "eda": eda, "selected": selected,
        "models": site_models, "comparisons": comparisons, "extra": extra,
        "errors": err, "y_test": [int(v) for v in yte], "y_val": [int(v) for v in yva],
        "test_rows": test_rows_site.to_dict(orient="list"),
        "tuning_counts": {m: len(si.model_grid(m)) for m in MODELS},
    }
    payload = json.dumps(site, separators=(",", ":"), default=str)
    (SITE / "site.json").write_text(payload)
    # Same data as a script so the page also works when opened from disk (file://).
    (SITE / "site.js").write_text("window.SITE=" + payload + ";\n")
    print(f"Done in {time.time() - t0:.0f}s")


def make_figures(df, eda, tuned, selected, yte, imp):
    # 1. Purchase rate by month with session counts
    m = pd.DataFrame(eda["month"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(m.month, m.rate * 100, color=BLUE, width=0.6)
    for i, row in m.iterrows():
        ax.text(i, row.rate * 100 + 0.6, f"n={row.n}", ha="center", fontsize=8, color="#52514e")
    ax.set_ylabel("Purchase rate (%)"); ax.set_title("Purchase rate by month (no Jan/Apr in data)")
    fig.tight_layout(); fig.savefig(FIG / "month_rate.png", dpi=150); plt.close(fig)

    # 2. PageValues > 0 vs purchase
    pv = eda["pagevalues"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["PageValues = 0", "PageValues > 0"], [pv["rate_if_zero"] * 100, pv["rate_if_positive"] * 100],
           color=[GRAY, ORANGE], width=0.5)
    ax.set_ylabel("Purchase rate (%)"); ax.set_title("PageValues almost separates the classes")
    fig.tight_layout(); fig.savefig(FIG / "pagevalues.png", dpi=150); plt.close(fig)

    # 3. PR curves on test, per setting
    colors = {"stump": GRAY, "logreg": BLUE, "tree": "#6250d6", "forest": AQUA, "gboost": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axes, si.SETTINGS):
        for mname, c in colors.items():
            p = tuned[s][mname]["p_test"]
            pr, rc, _ = precision_recall_curve(yte, p)
            ax.plot(rc, pr, color=c, lw=2 if mname == selected[s] else 1.2,
                    label=f"{si.MODEL_INFO[mname]} ({average_precision_score(yte, p):.3f})")
        ax.axhline(yte.mean(), color="#b5b4ae", ls="--", lw=1, label=f"Majority / prevalence ({yte.mean():.3f})")
        ax.set_title(si.SETTINGS[s]["name"]); ax.set_xlabel("Recall")
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    axes[0].set_ylabel("Precision")
    fig.suptitle("Precision-recall on the held-out test set (PR-AUC in brackets)")
    fig.tight_layout(); fig.savefig(FIG / "pr_curves_test.png", dpi=150); plt.close(fig)

    # 4. Permutation importance
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, s in zip(axes, si.SETTINGS):
        d = pd.DataFrame(imp[s]).head(8)[::-1]
        ax.barh(d.feature, d["mean"], xerr=d["sd"], color=BLUE, height=0.6)
        ax.set_title(f"{s}: {si.MODEL_INFO[selected[s]]}"); ax.set_xlabel("Drop in validation PR-AUC")
    fig.tight_layout(); fig.savefig(FIG / "permutation_importance.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
