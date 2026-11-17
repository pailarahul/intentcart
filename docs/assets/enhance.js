/* Visual enhancements only: progress bar, stage labels, reveal-on-scroll, count-up, back-to-top. */
(function () {
  "use strict";
  const groups = {
    question: "Framing", data: "Framing",
    preprocessing: "Preparation", methods: "Preparation", evaluation: "Preparation",
    results: "Evidence", explorer: "Evidence · interactive", errors: "Evidence", robustness: "Evidence",
    limitations: "Wrap-up", conclusion: "Wrap-up", reproduce: "Wrap-up", about: "Wrap-up",
  };
  const dot = { Framing: "var(--g-frame)", Preparation: "var(--g-prep)", Evidence: "var(--g-evid)", "Evidence · interactive": "var(--g-evid)", "Wrap-up": "var(--g-wrap)" };

  const icons = { question: "help-circle", data: "database", preprocessing: "adjustments", methods: "hierarchy-3",
    evaluation: "scale", results: "chart-bar", explorer: "hand-click", errors: "bug", robustness: "shield-check",
    limitations: "alert-triangle", conclusion: "flag", reproduce: "terminal-2", about: "users" };
  // Header band: big icon + stage label + title
  document.querySelectorAll("main section[id]").forEach(sec => {
    const g = groups[sec.id]; if (!g) return;
    const h2 = sec.querySelector("h2"); if (!h2) return;
    const head = document.createElement("div"); head.className = "sec-head";
    head.innerHTML = `<div class="icon"><span class="ti ti-${icons[sec.id]}"></span></div><div><div class="eyebrow">${g}</div></div>`;
    sec.insertBefore(head, h2); head.lastChild.appendChild(h2);
    const link = document.querySelector(`nav.toc a[href="#${sec.id}"]`);
    if (link) link.style.setProperty("--dot", dot[g]);
  });

  // Reading progress + back-to-top + "seen" dots in the sidebar
  const bar = document.getElementById("progress"), top = document.getElementById("totop");
  const links = Array.from(document.querySelectorAll("nav.toc a[href^='#']"));
  function onScroll() {
    const h = document.documentElement, max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
    top.classList.toggle("show", h.scrollTop > 700);
    links.forEach(a => {
      const t = document.querySelector(a.getAttribute("href"));
      if (t) a.classList.toggle("seen", t.getBoundingClientRect().top < window.innerHeight * 0.5);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  top.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));

  // Fade sections in as they scroll into view
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce && "IntersectionObserver" in window) {
    const secs = Array.from(document.querySelectorAll("main section"));
    secs.forEach(s => { if (s.getBoundingClientRect().top > window.innerHeight) s.classList.add("reveal"); });
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { rootMargin: "0px 0px -8% 0px" });
    secs.forEach(s => io.observe(s));
  }

  // Hero infographic: 100 sessions, buyers lit up
  const art = document.getElementById("hero-art");
  if (art && window.SITE) {
    const rate = SITE.eda.quality.positive_rate, nBuy = Math.round(rate * 100);
    const pick = new Set(); let x = 7;
    while (pick.size < nBuy) { x = (x * 37 + 11) % 100; pick.add(x); }
    let cells = "";
    for (let i = 0; i < 100; i++) cells += `<span class="ti ti-shopping-cart${pick.has(i) ? " buy" : ""}"></span>`;
    art.innerHTML = `<div class="waffle">${cells}</div><div class="cap">Out of every 100 sessions, only <b>${nBuy}</b> end in a purchase. Finding them is the task.</div>`;
    if (!reduce) {
      const buys = Array.from(art.querySelectorAll(".buy"));
      buys.forEach((c, i) => setTimeout(() => c.classList.add("on"), 300 + i * 70));
    } else art.querySelectorAll(".buy").forEach(c => c.classList.add("on"));
  }

  // Count up the three headline numbers
  if (!reduce) {
    document.querySelectorAll("header.hero .tile .big").forEach(el => {
      const txt = el.textContent.trim(), m = txt.match(/^([\d.]+)(%?)$/);
      if (!m) return;
      const end = parseFloat(m[1]), dec = (m[1].split(".")[1] || "").length, suf = m[2];
      const t0 = performance.now(), dur = 1100;
      (function step(t) {
        const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
        el.textContent = (end * e).toFixed(dec) + suf;
        if (k < 1) requestAnimationFrame(step); else el.textContent = txt;
      })(t0);
    });
  }
})();
