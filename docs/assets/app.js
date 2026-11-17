/* IntentCart website: every number is read from data/site.js, which src/run_all.py writes. */
(function () {
  "use strict";
  // ---- Set this to your GitHub repository URL after pushing -----------------
  const REPO_URL = "https://github.com/pailarahul/intentcart";
  // ---------------------------------------------------------------------------

  const S = window.SITE;
  if (!S) { document.body.insertAdjacentHTML("afterbegin", "<p style='padding:20px'>Could not load data/site.js. Run <code>python src/run_all.py</code>.</p>"); return; }

  const $ = (q, el = document) => el.querySelector(q);
  const $$ = (q, el = document) => Array.from(el.querySelectorAll(q));
  const f3 = x => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(3);
  const f2 = x => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(2);
  const pct = (x, d = 1) => (x == null || isNaN(x)) ? "–" : (x * 100).toFixed(d) + "%";
  const int = x => Math.round(x).toLocaleString("en-US");
  const sgn = x => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const ORDER = ["majority", "stump", "logreg", "tree", "forest", "gboost"];
  const COLORS = { majority: "#b5b4ae", stump: "#8a8984", logreg: "#2a78d6", gboost: "#eb6834", forest: "#1baf7a", tree: "#6250d6" };
  const M = S.models, sel = S.selected;
  const yTest = S.y_test, yVal = S.y_val;
  const prevTest = S.split.test.positives / S.split.test.n;

  // ------------------------------------------------------------ repo links
  $$(".repo-link").forEach(a => a.href = REPO_URL);
  $$(".repo-text").forEach(s => s.textContent = REPO_URL + ".git");

  // ------------------------------------------------------------ narrative values
  const q = S.eda.quality, pv = S.eda.pagevalues;
  const monthRate = m => (S.eda.month.find(d => d.month === m) || {}).rate;
  const visRate = t => (S.eda.visitor.find(d => d.type === t) || {}).rate;
  const SB = M.B[sel.B], SA = M.A[sel.A];
  const cmpTxt = (s, other) => {
    const c = S.comparisons[s][`${sel[s]}_minus_${other}`];
    return c ? `${sgn(c.mean)} [${sgn(c.ci95[0])}, ${sgn(c.ci95[1])}]` : "n/a (same model)";
  };
  const impA = S.extra.permutation_importance_val.A;
  const segPV = S.errors.B.segments["PageValues>0"];
  const ab = S.extra.ablation_no_rates;
  const tmp = S.extra.temporal;
  const nConfigs = 2 * Object.values(S.tuning_counts).reduce((a, b) => a + b, 0);
  const V = {
    n_rows: int(q.n_rows), n_pos: int(q.n_positive), pos_rate: pct(q.positive_rate),
    dup_extra: int(q.duplicate_rows_extra_copies), dup_rows: int(q.rows_in_duplicate_groups),
    skew_prd: f2(S.eda.skew.ProductRelated_Duration), max_prd: int(S.eda.max.ProductRelated_Duration),
    pv_zero: pct(pv.share_zero), pv_rate0: pct(pv.rate_if_zero), pv_rate1: pct(pv.rate_if_positive),
    feb_rate: pct(monthRate("Feb")), nov_rate: pct(monthRate("Nov")),
    new_rate: pct(visRate("New_Visitor")), ret_rate: pct(visRate("Returning_Visitor")),
    n_train: int(S.split.train.n), n_val: int(S.split.val.n), n_test: int(S.split.test.n),
    prev: f3(prevTest),
    selA_label: SA.label.toLowerCase() + " model", selB_label: SB.label.toLowerCase() + " model",
    A_pr: f3(SA.test.pr_auc), A_ci: `[${f3(SA.ci.pr_auc[0])}, ${f3(SA.ci.pr_auc[1])}]`,
    B_pr: f3(SB.test.pr_auc), B_ci: `[${f3(SB.ci.pr_auc[0])}, ${f3(SB.ci.pr_auc[1])}]`,
    A_stump_pr: f3(M.A.stump.test.pr_auc),
    B_lift: (SB.test.pr_auc / prevTest).toFixed(1) + "×",
    B_rec_p: pct(SB.test.recall, 0), B_prec_p: pct(SB.test.precision, 0),
    B_thr: f3(SB.threshold), B_tp: int(SB.test.tp), B_fp: int(SB.test.fp), B_fn: int(SB.test.fn), B_tn: int(SB.test.tn),
    maj_acc: pct(M.B.majority.test.accuracy),
    cmp_B_stump: cmpTxt("B", "stump"), cmp_B_lr: cmpTxt("B", "logreg"), cmp_B_forest: cmpTxt("B", "forest"),
    pi_A_pv: f3(impA.find(d => d.feature === "PageValues").mean),
    pi_A_2nd: f3(Math.max(...impA.filter(d => d.feature !== "PageValues").map(d => d.mean))),
    A_minus_B: (() => { const c = S.comparisons.A_minus_B_selected; return `${f3(c.mean)} [${f3(c.ci95[0])}, ${f3(c.ci95[1])}]`; })(),
    tp_pv: f2(S.errors.B.median_by_outcome.TP.PageValues),
    seg_pv0_prec: pct(segPV.find(d => d.segment === "PageValues = 0").precision, 0),
    seg_pv1_prec: pct(segPV.find(d => d.segment === "PageValues > 0").precision, 0),
    tmp_train_rate: pct(tmp.train.rate), tmp_test_rate: pct(tmp.test.rate),
    tmp_B_lr: f3(tmp.results.B.logreg.pr_auc), tmp_B_gb: f3(tmp.results.B.gboost.pr_auc),
    dedup_B: f3(S.extra.dedup.results.B[sel.B]),
    abl_best: f3(Math.max(...Object.values(ab).map(d => d.test_pr_auc))),
    n_configs: int(nConfigs), generated: S.generated,
  };
  $$("[data-v]").forEach(el => { const k = el.dataset.v; el.textContent = (k in V) ? V[k] : "??" + k; });

  // ------------------------------------------------------------ tooltip
  const tt = $("#tt");
  function showTip(html, ev) {
    tt.innerHTML = html; tt.style.opacity = 1;
    const w = tt.offsetWidth, x = ev.pageX + 14, maxX = window.scrollX + document.documentElement.clientWidth - w - 8;
    tt.style.left = Math.min(x, maxX) + "px"; tt.style.top = (ev.pageY + 14) + "px";
  }
  function hideTip() { tt.style.opacity = 0; }

  // ------------------------------------------------------------ SVG helpers
  const NS = "http://www.w3.org/2000/svg";
  function svgEl(w, h) { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`); s.setAttribute("role", "img"); return s; }
  function add(parent, tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    parent.appendChild(e); return e;
  }
  function roundTopBar(x, y, w, h, r) {
    r = Math.min(r, w / 2, h);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  /* Vertical bar chart. items: {label, value, color, tip, lo, hi} */
  function barChart(el, items, o = {}) {
    const W = o.w || 520, H = o.h || 240, m = { t: 14, r: 10, b: 34, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const yMax = o.yMax || Math.max(...items.map(d => d.hi != null ? d.hi : d.value)) * 1.12;
    const svg = svgEl(W, H), g = add(svg, "g", { transform: `translate(${m.l},${m.t})` });
    const y = v => ih - v / yMax * ih;
    const ticks = o.ticks || 4;
    for (let i = 0; i <= ticks; i++) {
      const v = yMax * i / ticks;
      add(g, "line", { x1: 0, x2: iw, y1: y(v), y2: y(v), class: "gridline" });
      add(g, "text", { x: -6, y: y(v) + 4, "text-anchor": "end" }, o.yFmt ? o.yFmt(v) : v.toFixed(2));
    }
    const step = iw / items.length, bw = Math.min(56, step * 0.62);
    items.forEach((d, i) => {
      const x = i * step + (step - bw) / 2, h = Math.max(0, ih - y(d.value));
      const bar = add(g, "path", { d: roundTopBar(x, y(d.value), bw, h, 4), fill: d.color || "var(--series-1)" });
      if (d.lo != null) {
        const cx = x + bw / 2;
        add(g, "line", { x1: cx, x2: cx, y1: y(d.lo), y2: y(d.hi), stroke: "#141413", "stroke-width": 1.5 });
        [d.lo, d.hi].forEach(v => add(g, "line", { x1: cx - 5, x2: cx + 5, y1: y(v), y2: y(v), stroke: "#141413", "stroke-width": 1.5 }));
      }
      if (o.valueLabels) add(g, "text", { x: x + bw / 2, y: y(d.hi != null ? d.hi : d.value) - 5, "text-anchor": "middle", style: "fill:var(--text-secondary);font-weight:600" }, o.valueLabels(d));
      add(g, "text", { x: x + bw / 2, y: ih + 16, "text-anchor": "middle" }, d.label);
      if (d.sub) add(g, "text", { x: x + bw / 2, y: ih + 29, "text-anchor": "middle" }, d.sub);
      const hit = add(g, "rect", { x: i * step, y: 0, width: step, height: ih, fill: "transparent" });
      hit.addEventListener("mousemove", ev => { bar.style.opacity = .8; showTip(d.tip, ev); });
      hit.addEventListener("mouseleave", () => { bar.style.opacity = 1; hideTip(); });
    });
    if (o.refLine != null) {
      add(g, "line", { x1: 0, x2: iw, y1: y(o.refLine), y2: y(o.refLine), stroke: "#77766f", "stroke-dasharray": "4 4", "stroke-width": 1.2 });
      if (o.refLabel) add(g, "text", { x: o.refAnchor === "start" ? 2 : iw, y: y(o.refLine) - 5, "text-anchor": o.refAnchor || "end" }, o.refLabel);
    }
    add(g, "line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: "#c9c8c0" });
    el.innerHTML = ""; el.appendChild(svg);
  }

  /* Horizontal bars (importance). items: {label, value, sd} */
  function hBarChart(el, items, o = {}) {
    const W = 420, rowH = 24, m = { t: 6, r: 50, b: 22, l: 170 };
    const H = m.t + m.b + rowH * items.length, iw = W - m.l - m.r;
    const xMax = Math.max(...items.map(d => d.value + (d.sd || 0))) * 1.05 || 1;
    const svg = svgEl(W, H), g = add(svg, "g", { transform: `translate(${m.l},${m.t})` });
    const x = v => Math.max(0, v) / xMax * iw;
    items.forEach((d, i) => {
      const yy = i * rowH + 4, bh = rowH - 9;
      add(g, "text", { x: -8, y: yy + bh / 2 + 4, "text-anchor": "end", style: "fill:var(--text-secondary)" }, d.label);
      const b = add(g, "rect", { x: 0, y: yy, width: Math.max(1, x(d.value)), height: bh, rx: 3, fill: "var(--series-1)" });
      add(g, "text", { x: x(d.value) + 6, y: yy + bh / 2 + 4 }, f3(d.value));
      const hit = add(g, "rect", { x: -m.l, y: yy - 3, width: W, height: rowH, fill: "transparent" });
      hit.addEventListener("mousemove", ev => { b.style.opacity = .8; showTip(`<b>${esc(d.label)}</b><br>PR-AUC drop ${f3(d.value)} ± ${f3(d.sd)}`, ev); });
      hit.addEventListener("mouseleave", () => { b.style.opacity = 1; hideTip(); });
    });
    add(g, "line", { x1: 0, x2: 0, y1: 0, y2: rowH * items.length, stroke: "#c9c8c0" });
    el.innerHTML = ""; el.appendChild(svg);
  }

  /* Line chart with crosshair tooltip. series: {name, color, pts:[[x,y]], width, dash} */
  function lineChart(el, series, o = {}) {
    const W = o.w || 520, H = o.h || 300, m = { t: 12, r: 14, b: 38, l: 44 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xMax = o.xMax || 1, yMax = o.yMax || 1;
    const X = v => v / xMax * iw, Y = v => ih - v / yMax * ih;
    const svg = svgEl(W, H), g = add(svg, "g", { transform: `translate(${m.l},${m.t})` });
    for (let i = 0; i <= 5; i++) {
      const v = yMax * i / 5; add(g, "line", { x1: 0, x2: iw, y1: Y(v), y2: Y(v), class: "gridline" });
      add(g, "text", { x: -6, y: Y(v) + 4, "text-anchor": "end" }, v.toFixed(1));
      const xv = xMax * i / 5; add(g, "text", { x: X(xv), y: ih + 15, "text-anchor": "middle" }, xv.toFixed(1));
    }
    add(g, "line", { x1: 0, x2: iw, y1: ih, y2: ih, stroke: "#c9c8c0" });
    add(g, "line", { x1: 0, x2: 0, y1: 0, y2: ih, stroke: "#c9c8c0" });
    if (o.xLabel) add(g, "text", { x: iw / 2, y: ih + 32, "text-anchor": "middle", style: "fill:var(--text-secondary)" }, o.xLabel);
    if (o.yLabel) add(g, "text", { x: -ih / 2, y: -33, transform: "rotate(-90)", "text-anchor": "middle", style: "fill:var(--text-secondary)" }, o.yLabel);
    if (o.refY != null) add(g, "line", { x1: 0, x2: iw, y1: Y(o.refY), y2: Y(o.refY), stroke: "#77766f", "stroke-dasharray": "4 4" });
    if (o.vline != null) add(g, "line", { x1: X(o.vline), x2: X(o.vline), y1: 0, y2: ih, stroke: "#141413", "stroke-width": 1.2 });
    if (o.vline2 != null) add(g, "line", { x1: X(o.vline2), x2: X(o.vline2), y1: 0, y2: ih, stroke: "#77766f", "stroke-dasharray": "3 3" });
    series.forEach(s => {
      const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
      add(g, "path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round" });
    });
    if (o.marker) {
      add(g, "circle", { cx: X(o.marker[0]), cy: Y(o.marker[1]), r: 6, fill: o.markerColor || "#141413", stroke: "#fcfcfb", "stroke-width": 2 });
    }
    const cross = add(g, "line", { x1: 0, x2: 0, y1: 0, y2: ih, stroke: "#77766f", "stroke-width": 1, opacity: 0 });
    const hit = add(g, "rect", { x: 0, y: 0, width: iw, height: ih, fill: "transparent" });
    hit.addEventListener("mousemove", ev => {
      const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width - m.l;
      const xv = Math.max(0, Math.min(xMax, sx / iw * xMax));
      cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("opacity", 1);
      const rows = series.filter(s => !s.noTip).map(s => {
        let best = s.pts[0]; for (const p of s.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        return `<span style="color:${s.color}">●</span> ${esc(s.name)}: ${f3(best[1])}`;
      });
      showTip(`<b>${o.xLabel || "x"} ≈ ${xv.toFixed(2)}</b><br>${rows.join("<br>")}`, ev);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
    el.innerHTML = ""; el.appendChild(svg);
  }

  // ------------------------------------------------------------ metric helpers
  function confusion(y, p, t) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < y.length; i++) {
      const h = p[i] >= t;
      if (h && y[i]) tp++; else if (h) fp++; else if (y[i]) fn++; else tn++;
    }
    const prec = tp + fp ? tp / (tp + fp) : null, rec = tp + fn ? tp / (tp + fn) : 0;
    return { tp, fp, fn, tn, prec, rec, f1: (prec != null && prec + rec) ? 2 * prec * rec / (prec + rec) : 0, acc: (tp + tn) / y.length };
  }
  function prCurve(y, p) {
    const idx = p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
    const P = y.reduce((a, b) => a + b, 0);
    let tp = 0, fp = 0; const pts = [];
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k]; if (y[i]) tp++; else fp++;
      if (k === idx.length - 1 || p[idx[k + 1]] !== p[i]) pts.push([tp / P, tp / (tp + fp)]);
    }
    return [[0, pts.length ? pts[0][1] : 1], ...pts];
  }
  const THR = Array.from({ length: 95 }, (_, i) => +(0.01 + i * 0.01).toFixed(2));

  // ------------------------------------------------------------ §2 charts & tables
  barChart($("#chart-month"), S.eda.month.map(d => ({
    label: d.month, value: d.rate, color: "var(--series-1)",
    tip: `<b>${d.month}</b><br>${int(d.n)} sessions<br>${pct(d.rate)} bought`
  })), { yFmt: v => pct(v, 0), yMax: 0.3, ticks: 3, refLine: q.positive_rate, refLabel: "overall " + pct(q.positive_rate), refAnchor: "start" });
  barChart($("#chart-pv"), [
    { label: "PageValues = 0", sub: `${int(pv.n_zero)} sessions`, value: pv.rate_if_zero, color: "var(--neutral)", tip: `<b>PageValues = 0</b><br>${int(pv.n_zero)} sessions<br>${pct(pv.rate_if_zero)} bought (${int(pv.purchases_with_zero_pv)} purchases)` },
    { label: "PageValues > 0", sub: `${int(pv.n_positive)} sessions`, value: pv.rate_if_positive, color: "var(--series-2)", tip: `<b>PageValues > 0</b><br>${int(pv.n_positive)} sessions<br>${pct(pv.rate_if_positive)} bought` },
  ], { yFmt: v => pct(v, 0), yMax: 0.8, valueLabels: d => pct(d.value) });

  $("#tbl-median").innerHTML = "<table><tr><th>Feature</th><th class='n'>No purchase</th><th class='n'>Purchase</th><th class='n'>Skew</th></tr>" +
    S.eda.median_by_class.map(d => `<tr><td><code>${d.feature}</code></td><td class="n">${f2(d.no_purchase)}</td><td class="n">${f2(d.purchase)}</td><td class="n">${f2(S.eda.skew[d.feature])}</td></tr>`).join("") + "</table>";

  // ------------------------------------------------------------ §6 results
  const paramTxt = p => Object.keys(p).length ? Object.entries(p).map(([k, v]) => `${k}=${v === null ? "None" : v}`).join(", ") : "–";
  function renderResults(s) {
    const ms = M[s];
    $("#tbl-results").innerHTML = `<table><tr><th>Model</th><th>Chosen settings</th><th class="n">Val PR-AUC</th><th class="n">Test PR-AUC [95% CI]</th><th class="n">ROC-AUC</th><th class="n">Threshold</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">Accuracy</th></tr>` +
      ORDER.map(k => {
        const d = ms[k], isSel = k === sel[s];
        return `<tr class="${isSel ? "sel" : ""}"><td>${d.label}${isSel ? ' <span class="tag blue">selected</span>' : ""}</td>
        <td class="small mono">${esc(paramTxt(d.params))}</td><td class="n">${f3(d.val.pr_auc)}</td>
        <td class="n"><b>${f3(d.test.pr_auc)}</b> ${d.ci ? `<span class="dim">[${f3(d.ci.pr_auc[0])}, ${f3(d.ci.pr_auc[1])}]</span>` : ""}</td>
        <td class="n">${f3(d.test.roc_auc)}</td><td class="n">${k === "majority" ? "–" : f3(d.threshold)}</td>
        <td class="n">${f3(d.test.precision)}</td><td class="n">${f3(d.test.recall)}</td><td class="n">${f3(d.test.f1)}</td><td class="n">${pct(d.test.accuracy)}</td></tr>`;
      }).join("") + `</table><p class="small">Setting ${s}: ${esc(S.settings[s].why)}</p>`;

    barChart($("#chart-pr-bars"), ORDER.filter(k => k !== "majority").map(k => ({
      label: { stump: "1-rule", logreg: "LogReg", tree: "Tree", forest: "Forest", gboost: "Boosting" }[k],
      value: ms[k].test.pr_auc, lo: ms[k].ci.pr_auc[0], hi: ms[k].ci.pr_auc[1], color: COLORS[k],
      tip: `<b>${ms[k].label}</b>${k === sel[s] ? " (selected)" : ""}<br>Test PR-AUC ${f3(ms[k].test.pr_auc)}<br>95% CI [${f3(ms[k].ci.pr_auc[0])}, ${f3(ms[k].ci.pr_auc[1])}]<br>Validation ${f3(ms[k].val.pr_auc)}`
    })), { yMax: 1, yFmt: v => v.toFixed(2), refLine: prevTest, valueLabels: d => f3(d.value) });

    const curveModels = ["stump", "logreg", "tree", "forest", "gboost"];
    lineChart($("#chart-pr-curves"), curveModels.map(k => ({
      name: ms[k].label, color: COLORS[k], width: k === sel[s] ? 2.6 : 1.6, pts: prCurve(yTest, ms[k].p_test)
    })), { xLabel: "Recall", yLabel: "Precision", refY: prevTest });
    $("#legend-pr").innerHTML = curveModels.map(k => `<span style="--c:${COLORS[k]}">${ms[k].label} (${f3(ms[k].test.pr_auc)})</span>`).join("") + `<span style="--c:#77766f">Majority (dashed)</span>`;
  }
  $$("#res-setting button").forEach(b => b.addEventListener("click", () => {
    $$("#res-setting button").forEach(x => x.classList.toggle("on", x === b)); renderResults(b.dataset.s);
  }));
  renderResults("B");

  ["A", "B"].forEach(s => hBarChart($("#chart-imp-" + s), S.extra.permutation_importance_val[s].slice(0, 8).map(d => ({ label: d.feature, value: d.mean, sd: d.sd }))));
  $("#tbl-cal").innerHTML = "<table><tr><th>Predicted bin</th><th class='n'>Sessions</th><th class='n'>Mean predicted</th><th class='n'>Observed buy rate</th></tr>" +
    S.extra.calibration_test.B.map(d => `<tr><td>${d.bin}</td><td class="n">${d.n}</td><td class="n">${f3(d.mean_pred)}</td><td class="n">${f3(d.observed)}</td></tr>`).join("") + "</table>";

  // ------------------------------------------------------------ §7 explorer
  const st = { s: "B", m: sel.B, t: M.B[sel.B].threshold, kind: "FP", shown: 10 };
  function fillModelSelect() {
    $("#ex-model").innerHTML = ORDER.filter(k => k !== "majority").map(k => `<option value="${k}" ${k === st.m ? "selected" : ""}>${M[st.s][k].label}${k === sel[st.s] ? " (selected)" : ""}</option>`).join("");
  }
  function cell(n, total, label, color, text) {
    return `<div class="c" style="background:${color};color:${text}"><b>${int(n)}</b><span>${label}</span><br><span>${pct(n / total)} of class</span></div>`;
  }
  function renderExplorer() {
    const md = M[st.s][st.m], p = md.p_test, t = st.t;
    $("#ex-thr").value = t; $("#ex-thr-val").textContent = t.toFixed(3);
    const c = confusion(yTest, p, t), pos = c.tp + c.fn, neg = c.fp + c.tn;
    const shade = (v, tot) => { const a = tot ? v / tot : 0; return `rgba(42,120,214,${0.10 + 0.55 * a})`; };
    const ink = (v, tot) => (tot && v / tot > 0.6) ? "#fff" : "#141413";
    $("#ex-cm").innerHTML = `<div></div><div class="h">Predicted buy</div><div class="h">Predicted no buy</div>
      <div class="h">Actual buyer<br>(${int(pos)})</div>${cell(c.tp, pos, "true positives", shade(c.tp, pos), ink(c.tp, pos))}${cell(c.fn, pos, "missed buyers (FN)", shade(c.fn, pos), ink(c.fn, pos))}
      <div class="h">Actual non-buyer<br>(${int(neg)})</div>${cell(c.fp, neg, "false alarms (FP)", shade(c.fp, neg), ink(c.fp, neg))}${cell(c.tn, neg, "true negatives", shade(c.tn, neg), ink(c.tn, neg))}`;
    $("#ex-metrics").innerHTML = [["Precision", f3(c.prec), "flagged sessions that buy"], ["Recall", f3(c.rec), "buyers caught"],
      ["F1", f3(c.f1), "balance of the two"], ["Accuracy", pct(c.acc), "majority gets " + pct(1 - prevTest)]]
      .map(([a, b, d]) => `<div class="metric"><b>${b}</b><div>${a} · ${d}</div></div>`).join("");
    const curve = prCurve(yTest, p);
    lineChart($("#ex-pr"), [{ name: md.label, color: COLORS[st.m], pts: curve }], { xLabel: "Recall", yLabel: "Precision", refY: prevTest, marker: [c.rec, c.prec == null ? 1 : c.prec], w: 360, h: 270 });
    const cs = THR.map(x => [x, confusion(yTest, p, x)]);
    lineChart($("#ex-curves"), [
      { name: "Precision", color: "var(--series-1)", pts: cs.filter(([, d]) => d.prec != null).map(([x, d]) => [x, d.prec]) },
      { name: "Recall", color: "var(--series-2)", pts: cs.map(([x, d]) => [x, d.rec]) },
      { name: "F1", color: "var(--series-3)", pts: cs.map(([x, d]) => [x, d.f1]) },
    ], { xLabel: "Threshold", vline: t, vline2: md.threshold, w: 360, h: 270 });
    const flagged = c.tp + c.fp;
    $("#ex-readout").innerHTML = `At threshold <b>${t.toFixed(3)}</b>, the ${esc(md.label.toLowerCase())} (setting ${st.s}) flags <b>${int(flagged)}</b> of ${int(yTest.length)} test sessions.
      It catches <b>${int(c.tp)} of ${int(pos)}</b> buyers (${pct(c.rec, 0)}) and wastes <b>${int(c.fp)}</b> follow-ups on non-buyers, so ${flagged ? pct(c.prec, 0) : "–"} of flags are right.
      <span class="dim">Dashed line on the right chart = threshold chosen on validation (${f3(md.threshold)}).</span>`;
    renderCost();
    renderErrors();
  }
  function renderCost() {
    const md = M[st.s][st.m], k = Math.max(1, +$("#ex-cost").value || 1);
    let best = null;
    for (const x of THR) { const c = confusion(yVal, md.p_val, x), cost = c.fp + k * c.fn; if (!best || cost < best.cost) best = { x, cost }; }
    st.costT = best.x;
    $("#ex-cost-thr").textContent = best.x.toFixed(2);
    const ct = confusion(yTest, md.p_test, best.x);
    $("#ex-cost-note").textContent = `On test this would catch ${ct.tp} of ${ct.tp + ct.fn} buyers (recall ${f3(ct.rec)}) with precision ${ct.prec == null ? "–" : f3(ct.prec)}. Higher cost ratios push the threshold down: more buyers caught, more wasted follow-ups.`;
  }
  $$("#ex-setting button").forEach(b => b.addEventListener("click", () => {
    $$("#ex-setting button").forEach(x => x.classList.toggle("on", x === b));
    st.s = b.dataset.s; st.m = sel[st.s]; st.t = M[st.s][st.m].threshold; fillModelSelect(); renderExplorer();
  }));
  $("#ex-model").addEventListener("change", e => { st.m = e.target.value; st.t = M[st.s][st.m].threshold; renderExplorer(); });
  $("#ex-thr").addEventListener("input", e => { st.t = +e.target.value; renderExplorer(); });
  $("#ex-reset").addEventListener("click", () => { st.t = M[st.s][st.m].threshold; renderExplorer(); });
  $("#ex-half").addEventListener("click", () => { st.t = 0.5; renderExplorer(); });
  $("#ex-cost").addEventListener("input", renderCost);
  $("#ex-cost-apply").addEventListener("click", () => { st.t = st.costT; renderExplorer(); });

  // ------------------------------------------------------------ §8 errors
  const E = S.errors.B;
  function renderSeg() {
    const by = $("#seg-by").value, rows = E.segments[by];
    $("#tbl-seg").innerHTML = `<table><tr><th>Segment</th><th class="n">Sessions</th><th class="n">Buyers</th><th class="n">Recall</th><th class="n">FP rate</th><th class="n">Precision</th></tr>` +
      rows.map(d => `<tr><td>${esc(d.segment)}</td><td class="n">${d.n}</td><td class="n">${d.positives}</td><td class="n">${d.recall == null ? "–" : pct(d.recall, 0)}</td><td class="n">${d.fp_rate == null ? "–" : pct(d.fp_rate, 0)}</td><td class="n">${pct(d.precision, 0)}</td></tr>`).join("") +
      `</table><p class="small">Selected setting-B model at its validation threshold. Small segments (n &lt; 50) are noisy.</p>`;
  }
  $("#seg-by").addEventListener("change", renderSeg); renderSeg();
  const oc = E.median_by_outcome, ocF = [["ProductRelated", "Product pages"], ["ProductRelated_Duration", "Product time (s)"], ["Administrative", "Admin pages"], ["ExitRates", "Exit rate"], ["BounceRates", "Bounce rate"], ["PageValues", "PageValues*"]];
  $("#tbl-outcome").innerHTML = `<table><tr><th>Median</th>${["TP", "FN", "FP", "TN"].map(k => `<th class="n"><span class="kind ${k}">${k}</span></th>`).join("")}</tr>` +
    ocF.map(([f, l]) => `<tr><td>${l}</td>${["TP", "FN", "FP", "TN"].map(k => `<td class="n">${f === "ProductRelated_Duration" ? int(oc[k][f]) : f2(oc[k][f])}</td>`).join("")}</tr>`).join("") +
    `</table><p class="small">*Not an input to model B. Shown to see what it is missing. Counts: ${["TP", "FN", "FP", "TN"].map(k => `${k} ${E.counts[k]}`).join(" · ")}.</p>`;

  const R = S.test_rows;
  function renderErrors() {
    const md = M[st.s][st.m], p = md.p_test, t = st.t;
    const kindOf = i => { const h = p[i] >= t, y = yTest[i]; return h && y ? "TP" : h ? "FP" : y ? "FN" : "TN"; };
    let idx = p.map((_, i) => i).filter(i => kindOf(i) === st.kind);
    const desc = st.kind === "FP" || st.kind === "TP";
    idx.sort((a, b) => desc ? p[b] - p[a] : p[a] - p[b]);
    $("#err-ctx").innerHTML = `${idx.length} sessions · setting ${st.s}, ${esc(md.label.toLowerCase())}, threshold ${t.toFixed(3)} · sorted ${desc ? "highest" : "lowest"} probability first`;
    const cols = [["Month", "Month"], ["VisitorType", "Visitor"], ["Weekend", "Wknd"], ["Administrative", "Admin pg"], ["Informational", "Info pg"], ["ProductRelated", "Product pg"], ["ProductRelated_Duration", "Product time (min)"], ["BounceRates", "Bounce"], ["ExitRates", "Exit"], ["PageValues", "PageValues"], ["SpecialDay", "Special day"]];
    const fmt = (k, v) => k === "ProductRelated_Duration" ? (v / 60).toFixed(1) : k === "VisitorType" ? v.replace("_Visitor", "") : k === "Weekend" ? (v ? "yes" : "no") : (typeof v === "number" && !Number.isInteger(v)) ? f3(v) : v;
    const dimPV = st.s === "B";
    $("#tbl-err").innerHTML = `<table><tr><th>Session</th><th class="n">P(buy)</th><th>Outcome</th>${cols.map(([k, l]) => `<th class="${k === "PageValues" && dimPV ? "dim" : ""}">${l}${k === "PageValues" && dimPV ? "*" : ""}</th>`).join("")}</tr>` +
      idx.slice(0, st.shown).map(i => `<tr><td class="mono">#${R.row_id[i]}</td><td class="n"><b>${f3(p[i])}</b></td><td><span class="kind ${st.kind}">${st.kind}</span></td>${cols.map(([k]) => `<td class="${k === "PageValues" && dimPV ? "dim" : ""}">${fmt(k, R[k][i])}</td>`).join("")}</tr>`).join("") +
      `</table>${dimPV ? '<p class="small">*PageValues is not used by setting-B models; it is shown to reveal what they cannot see.</p>' : ""}`;
    $("#err-more").style.display = idx.length > st.shown ? "" : "none";
  }
  $$("#err-kind button").forEach(b => b.addEventListener("click", () => {
    $$("#err-kind button").forEach(x => x.classList.toggle("on", x === b)); st.kind = b.dataset.k; st.shown = 10; renderErrors();
  }));
  $("#err-more").addEventListener("click", () => { st.shown += 10; renderErrors(); });

  fillModelSelect(); renderExplorer();

  // ------------------------------------------------------------ §9 tables
  const cv = S.extra.cv_trainval;
  $("#tbl-cv").innerHTML = `<table><tr><th>Model</th><th class="n">Setting A: mean ± SD</th><th class="n">Setting B: mean ± SD</th></tr>` +
    ORDER.map(k => `<tr class="${k === sel.B ? "sel" : ""}"><td>${M.B[k].label}</td><td class="n">${f3(cv.A[k].mean)} ± ${f3(cv.A[k].sd)}</td><td class="n">${f3(cv.B[k].mean)} ± ${f3(cv.B[k].sd)}</td></tr>`).join("") + "</table>";
  const tr = tmp.results;
  $("#tbl-temporal").innerHTML = `<table><tr><th>Model</th><th class="n">A: PR-AUC</th><th class="n">A: F1</th><th class="n">B: PR-AUC</th><th class="n">B: precision / recall</th><th class="n">B: F1</th></tr>` +
    ["stump", "logreg", "forest", "gboost"].map(k => `<tr><td>${M.B[k].label}</td><td class="n">${f3(tr.A[k].pr_auc)}</td><td class="n">${f3(tr.A[k].f1)}</td><td class="n">${f3(tr.B[k].pr_auc)}</td><td class="n">${f3(tr.B[k].precision)} / ${f3(tr.B[k].recall)}</td><td class="n">${f3(tr.B[k].f1)}</td></tr>`).join("") +
    `</table><p class="small">Train ${int(tmp.train.n)} sessions (${tmp.train.months}), threshold set on ${int(tmp.val.n)} (${tmp.val.months}), test ${int(tmp.test.n)} (${tmp.test.months}). Chance PR-AUC on this test = ${f3(tmp.test.rate)}. Month is dropped as a feature here because the test months never appear in training, so it has nothing to learn from.</p>`;

  // ------------------------------------------------------------ nav highlight
  const links = $$("nav.toc a[href^='#']");
  const obs = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
  }), { rootMargin: "-30% 0px -60% 0px" });
  $$("main section, header.hero").forEach(s => obs.observe(s));
})();
