// DarkPatternGuard content script.
// 1) Segments the page the same way the ec-darkpattern dataset was built
//    (port of yamanalab/ec-darkpattern scraping/src/scraping-tools/segment/PageSegment.ts),
// 2) sends new segments to the local API, 3) outlines flagged elements and shows a tooltip on hover,
// 4) re-scans when the page changes (pop-ups, timers, infinite scroll) while auto-scan is on.
(() => {
  // A copy injected before the extension was reloaded stays on the page but can no longer talk to
  // the extension. Only skip loading if a *live* copy is already running; otherwise clean up and start.
  if (window.__dpgAlive && window.__dpgAlive()) return;
  window.__dpgAlive = () => { try { return !!chrome.runtime.id; } catch (e) { return false; } };
  document.querySelectorAll("[data-dpg-ui]").forEach(n => n.remove());
  document.querySelectorAll("[data-dpg]").forEach(n => { n.removeAttribute("data-dpg"); n.removeAttribute("data-dpg-text"); });

  const BLOCK = new Set(["ADDRESS","ARTICLE","ASIDE","BLOCKQUOTE","DETAILS","DIALOG","DD","DIV","DL","DT","FIELDSET",
    "FIGCAPTION","FIGURE","FOOTER","FORM","H1","H2","H3","H4","H5","H6","HEADER","HGROUP","HR","LI","MAIN","NAV","OL",
    "P","PRE","SECTION","TABLE","UL",
    // not in the original list, but structural in modern pages - treated as blocks so we recurse into them
    "TBODY","THEAD","TFOOT","TR","TD","TH","BODY","CENTER","MENU","SUMMARY"]);
  const INLINE = new Set(["A","ABBR","ACRONYM","AUDIO","B","BDI","BDO","BIG","BR","BUTTON","CANVAS","CITE","CODE","DATA",
    "DATALIST","DEL","DFN","EM","EMBED","I","IFRAME","IMG","INPUT","INS","KBD","LABEL","MAP","MARK","METER","NOSCRIPT",
    "OBJECT","OUTPUT","PICTURE","PROGRESS","Q","RUBY","S","SAMP","SCRIPT","SELECT","SLOT","SMALL","SPAN","STRONG","SUB",
    "SUP","SVG","TEMPLATE","TEXTAREA","TIME","U","TT","VAR","VIDEO","WBR"]);
  const IGNORE = new Set(["SCRIPT","STYLE","NOSCRIPT","BR","HR"]);
  const SKIP_SUBTREE = new Set(["SCRIPT","STYLE","NOSCRIPT","TEMPLATE","SVG","CANVAS","IFRAME","VIDEO","AUDIO","HEAD"]);
  const PRIVACY = "Privacy choice / consent";
  const HOST = location.hostname;

  const state = { auto: true, siteOn: true, showChip: true, minP: 0.5, scannedOnce: false,
                  results: new Map(), findings: [], scanning: false, offline: false };
  const active = () => state.auto && state.siteOn;   // automatic scanning allowed on this page?

  const norm = t => (t || "").replace(/\s+/g, " ").trim();
  const isOurs = el => el.closest && el.closest("[data-dpg-ui]");
  // like textContent, but with a space at element boundaries so "<button>Accept All</button><button>Ask
  // me later</button>" reads "Accept All Ask me later" instead of "Accept AllAsk me later"
  function textOf(node) {
    if (node.nodeType === 3) return node.textContent;
    if (node.nodeType !== 1 || SKIP_SUBTREE.has(node.tagName)) return "";
    let s = "";
    for (const c of node.childNodes) s += c.nodeType === 1 ? " " + textOf(c) + " " : textOf(c);
    return s;
  }
  function hasIgnoreChild(el) {
    for (const c of el.childNodes) if (c.nodeType === 1 && IGNORE.has(c.tagName)) return true;
    return false;
  }
  function allTextOrInline(el) {
    for (const c of el.childNodes) {
      if (c.nodeType === 3) continue;
      if (c.nodeType === 1 && INLINE.has(c.tagName)) continue;
      if (c.nodeType === 8) continue; // comments
      return false;
    }
    return true;
  }
  function segment(el, out) {
    if (!el || SKIP_SUBTREE.has(el.tagName) || isOurs(el)) return;
    if (!hasIgnoreChild(el) && allTextOrInline(el)) {
      const t = norm(textOf(el));
      if (t) { out.push({ text: t, el }); return; }
    }
    for (const c of el.childNodes) {
      if (c.nodeType === 3) {
        const t = norm(c.textContent);
        if (t) out.push({ text: t, el }); // bare text node -> highlight its parent block
        continue;
      }
      if (c.nodeType !== 1 || IGNORE.has(c.tagName)) continue;
      if (INLINE.has(c.tagName)) {
        if (SKIP_SUBTREE.has(c.tagName)) continue;
        const t = norm(textOf(c));
        if (t) out.push({ text: t, el: c });
      } else {
        segment(c, out); // BLOCK, or unknown/custom element (treated as block)
      }
    }
  }
  function visible(el) {
    if (!el.getClientRects().length) return false;
    const s = getComputedStyle(el);
    return s.visibility !== "hidden" && s.display !== "none" && parseFloat(s.opacity || "1") > 0.05;
  }
  function usable(t) {
    return t.length >= 3 && t.length <= 400 && (/[a-z]/i.test(t) || /\d{1,2}\s*:\s*\d{2}/.test(t));
  }

  // ---------- page styles + tooltip ----------
  const COLOURS = { high: "#D6293E", medium: "#E8710A", low: "#C99A06", privacy: "#7A4FD0" };
  const FONT = `"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif`;
  const style = document.createElement("style");
  style.setAttribute("data-dpg-ui", "");
  style.textContent = Object.entries(COLOURS).map(([k, c]) =>
    `[data-dpg="${k}"]{outline:2px solid ${c}!important;outline-offset:2px!important;border-radius:2px}`
  ).join("\n") + `\n[data-dpg-flash]{animation:dpgflash .9s ease-in-out 3}@keyframes dpgflash{50%{outline-width:5px}}`;
  (document.head || document.documentElement).appendChild(style);

  const hostEl = document.createElement("div");
  hostEl.setAttribute("data-dpg-ui", "");
  hostEl.style.cssText = "all:initial;position:fixed;z-index:2147483647;top:0;left:0;pointer-events:none";
  const shadow = hostEl.attachShadow({ mode: "open" });
  shadow.innerHTML = `<style>
    *{box-sizing:border-box;font-family:${FONT}}
    .tip{position:fixed;width:320px;background:#fff;color:#1C1F24;border:1px solid #E3E5E8;border-radius:8px;
      box-shadow:0 8px 28px rgba(28,31,36,.16);display:none;overflow:hidden;font-size:13px;line-height:1.45}
    .tip .top{display:flex;align-items:center;gap:8px;padding:10px 12px 8px}
    .tip .pill{color:#fff;font-weight:600;font-size:12px;padding:2px 8px;border-radius:10px}
    .tip .p{margin-left:auto;font-weight:600;font-variant-numeric:tabular-nums}
    .tip .body{padding:0 12px 10px}
    .tip .rule{margin-top:6px}.tip .rule b{font-weight:600}
    .tip .ev{color:#5A616B}.tip .q{color:#5A616B;margin-top:2px}
    .tip .foot{background:#F4F5F7;color:#5A616B;font-size:11.5px;padding:7px 12px;border-top:1px solid #E3E5E8}
    .chip{position:fixed;right:16px;bottom:16px;pointer-events:auto;display:none;align-items:center;gap:10px;width:312px;
      background:#fff;color:#1C1F24;border:1px solid #E3E5E8;border-radius:10px;padding:10px 8px 10px 12px;font-size:13px;
      box-shadow:0 6px 20px rgba(28,31,36,.12)}
    .chip .logo{flex:none;width:22px;height:22px;border-radius:6px;background:#D6293E;display:grid;place-items:center}
    .chip .txt{flex:1;min-width:0;line-height:1.3}
    .chip .t1{font-weight:600}
    .chip .t2{color:#5A616B;font-size:12px}
    .chip .t1,.chip .t2{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .chip .num{color:#D6293E}
    .chip.ok .num{color:#1E8E5A}.chip.off .logo{background:#8A9099}
    .chip button{font:inherit;border:0;background:none;cursor:pointer;color:#5A616B;border-radius:6px}
    .chip .go{padding:5px 8px;font-weight:600;color:#D6293E}
    .chip .go:hover,.chip .x:hover{background:#F4F5F7}
    .chip .x{width:24px;height:24px;font-size:16px;line-height:1}
    .chip.ok .go,.chip.off .go{display:none}
  </style><div class="tip"></div>
  <div class="chip" role="status">
    <span class="logo"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round"><circle cx="10" cy="10" r="6"/><path d="M15 15l5 5"/></svg></span>
    <span class="txt"><div class="t1"></div><div class="t2"></div></span>
    <button class="go" title="Jump to the next finding">Next</button>
    <button class="x" title="Hide on this page" aria-label="Hide on this page">×</button>
  </div>`;
  const tip = shadow.querySelector(".tip"), chip = shadow.querySelector(".chip");
  document.documentElement.appendChild(hostEl);
  let chipIdx = -1, chipHidden = false;
  chip.querySelector(".go").addEventListener("click", () => {
    if (!state.findings.length) return;
    chipIdx = (chipIdx + 1) % state.findings.length;
    reveal(chipIdx);
    chip.querySelector(".t2").textContent = `Showing ${chipIdx + 1} of ${state.findings.length}`;
  });
  chip.querySelector(".x").addEventListener("click", () => { chipHidden = true; chip.style.display = "none"; });

  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function tipHtml(r, lv) {
    const rules = r.rules.map(h => `<div class="rule"><b>${esc(h.name)}</b>
      <div class="ev">Matched “${esc(h.evidence.join("”, “"))}”</div><div class="q">${esc(h.question)}</div></div>`).join("");
    const model = r.model === "both"
      ? `TF-IDF ${Math.round(100 * r.p_tfidf)}% and RoBERTa ${Math.round(100 * r.p_roberta)}% agree`
      : `TF-IDF model ${Math.round(100 * r.p_dark)}%`;
    const pct = lv === "privacy" && r.p_dark < 0.5 ? "Rule match" : `${Math.round(100 * r.p_dark)}% likely`;
    return `<div class="top"><span class="pill" style="background:${COLOURS[lv]}">${esc(r.category || "Dark pattern")}</span>
      <span class="p">${pct}</span></div>
      <div class="body">${rules || '<div class="q">No rule matched. Flagged by the model only.</div>'}</div>
      <div class="foot">${model}. A screening aid, not legal advice.</div>`;
  }
  document.addEventListener("mouseover", e => {
    const el = e.target.closest && e.target.closest("[data-dpg]");
    if (!el) { tip.style.display = "none"; return; }
    const r = state.results.get(el.getAttribute("data-dpg-text"));
    if (!r) return;
    tip.innerHTML = tipHtml(r, el.getAttribute("data-dpg"));
    tip.style.display = "block";
    const rect = el.getBoundingClientRect();
    const top = rect.bottom + 8 + tip.offsetHeight > innerHeight ? rect.top - tip.offsetHeight - 8 : rect.bottom + 8;
    tip.style.top = Math.max(4, top) + "px";
    tip.style.left = Math.min(Math.max(4, rect.left), innerWidth - tip.offsetWidth - 8) + "px";
  }, true);

  // ---------- scanning ----------
  function level(r) {
    if (r.category === PRIVACY || !r.p_dark || r.p_dark < 0.5) return "privacy";
    const k = r.risk.split(" ")[0].toLowerCase();
    return COLOURS[k] ? k : "low";
  }
  // privacy/consent rule hits are always shown; model hits must clear the chosen sensitivity
  function isFlag(r) { return r.category === PRIVACY || (r.flagged && r.p_dark >= state.minP); }
  function clearMarks() {
    document.querySelectorAll("[data-dpg]").forEach(el => { el.removeAttribute("data-dpg"); el.removeAttribute("data-dpg-text"); });
  }
  function send(msg) {
    return new Promise(res => {
      try { chrome.runtime.sendMessage(msg, r => { void chrome.runtime.lastError; res(r); }); }
      catch (e) { res(null); }
    });
  }
  function updateUi() {
    const order = { high: 0, medium: 1, low: 2, privacy: 3 };
    const worst = state.findings.reduce((w, f) => (order[f.level] < order[w] ? f.level : w), "privacy");
    send({ type: "badge", count: state.findings.length, worst: state.findings.length ? worst : "none",
           offline: state.offline, paused: !state.siteOn });
    const show = state.showChip && !chipHidden && state.scannedOnce && (state.siteOn || state.findings.length || state.offline);
    if (!show) { chip.style.display = "none"; return; }
    const n = state.findings.length;
    chip.style.display = "flex";
    chip.className = "chip" + (state.offline ? " off" : n ? "" : " ok");
    chip.querySelector(".t1").innerHTML = state.offline ? "Can't check this page"
      : n ? `<span class="num">${n}</span> possible dark pattern${n > 1 ? "s" : ""}`
      : `<span class="num">✓</span> Nothing flagged`;
    chip.querySelector(".t2").textContent = state.offline ? "Start the model server, then scan again."
      : n ? (chipIdx >= 0 ? `Showing ${Math.min(chipIdx + 1, n)} of ${n}` : "Hover a highlight to see why")
      : "DarkPatternGuard checked this page";
  }

  async function scan(force = false) {
    if ((!force && !active()) || state.scanning || !document.body) return;
    state.scanning = true;
    try {
      const segs = [];
      segment(document.body, segs);
      const live = segs.filter(s => usable(s.text) && visible(s.el));
      const fresh = [...new Set(live.map(s => s.text))].filter(t => !state.results.has(t)).slice(0, 600);
      if (fresh.length) {
        const r = await send({ type: "classify", texts: fresh });
        if (!r || !r.ok) { state.offline = true; state.scannedOnce = true; updateUi(); return; }
        state.offline = false;
        fresh.forEach((t, i) => state.results.set(t, r.results[i]));
      }
      state.scannedOnce = true;
      clearMarks();
      const seen = new Set();
      state.findings = [];
      for (const s of live) {
        const r = state.results.get(s.text);
        if (!r || !isFlag(r)) continue;
        const lv = level(r);
        s.el.setAttribute("data-dpg", lv);
        s.el.setAttribute("data-dpg-text", s.text);
        if (!seen.has(s.text)) {
          seen.add(s.text);
          state.findings.push({ text: s.text, level: lv, category: r.category, p: r.p_dark, risk: r.risk, model: r.model, el: s.el });
        }
      }
      updateUi();
    } finally {
      state.scanning = false;
    }
  }

  function reveal(i) {
    const f = state.findings[i];
    if (!f || !f.el.isConnected) return false;
    f.el.scrollIntoView({ behavior: "smooth", block: "center" });
    f.el.setAttribute("data-dpg-flash", "");
    setTimeout(() => f.el.removeAttribute("data-dpg-flash"), 2800);
    return true;
  }
  function pause() { clearMarks(); state.findings = []; tip.style.display = "none"; updateUi(); }

  let timer = null;
  const observer = new MutationObserver(muts => {
    if (!active() || muts.every(m => isOurs(m.target))) return;
    // throttle, not debounce: a ticking countdown changes the DOM every second and would
    // otherwise postpone the re-scan forever
    if (!timer) timer = setTimeout(() => { timer = null; scan(); }, 1200);
  });

  chrome.runtime.onMessage.addListener((msg, _s, sendResponse) => {
    if (msg.type === "getFindings") {
      sendResponse({ host: HOST, siteOn: state.siteOn, auto: state.auto, offline: state.offline, scannedOnce: state.scannedOnce,
        scanned: state.results.size, findings: state.findings.map(({ el, ...f }) => f) });
    } else if (msg.type === "reveal") {
      sendResponse({ ok: reveal(msg.idx) });
    } else if (msg.type === "scanNow") {          // the popup's "Scan this page" button
      state.results.clear();
      scan(true).then(() => sendResponse({ ok: !state.offline }));
      return true;
    }
  });

  const DEFAULTS = { autoScan: true, showChip: true, minP: 0.5, pausedSites: [] };
  chrome.storage.local.get(DEFAULTS, cfg => {
    state.auto = cfg.autoScan; state.showChip = cfg.showChip; state.minP = cfg.minP;
    state.siteOn = !cfg.pausedSites.includes(HOST);
    if (active()) scan(); else updateUi();
    observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  });
  chrome.storage.onChanged.addListener(ch => {
    if (ch.showChip) { state.showChip = ch.showChip.newValue; updateUi(); }
    if (ch.minP) { state.minP = ch.minP.newValue; if (state.scannedOnce) scan(true); }
    if (ch.autoScan) { state.auto = ch.autoScan.newValue; if (active()) scan(); }
    if (ch.pausedSites) {
      const was = state.siteOn;
      state.siteOn = !ch.pausedSites.newValue.includes(HOST);
      if (was && !state.siteOn) pause();
      if (!was && state.siteOn && state.auto) scan(true);
    }
  });
})();
