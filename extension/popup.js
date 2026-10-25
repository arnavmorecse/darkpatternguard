const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const COLOURS = { high: "#D6293E", medium: "#E8710A", low: "#C99A06", privacy: "#7A4FD0" };
// The online web app. Use Settings → Web app address to point at http://localhost:8501 instead.
const DEFAULT_APP_URL = "https://darkpatternguard.streamlit.app";
const DEFAULTS = { autoScan: true, showChip: true, minP: 0.5, pausedSites: [], appUrl: DEFAULT_APP_URL };
const appBase = () => (cfg.appUrl || DEFAULT_APP_URL).replace(/\/+$/, "");
const openApp = params => { params.set("model", "roberta"); chrome.tabs.create({ url: appBase() + "/?" + params.toString() }); };
let lastFindings = [];
const SENS_HELP = { "0.5": "Flags text the models rate at least 50% likely. Catches more, with some false alarms.",
                    "0.8": "Flags only text rated at least 80% likely. Fewer false alarms, may miss borderline cases." };
let tabId = null, tabUrl = "", host = null, cfg = { ...DEFAULTS };

const ask = msg => new Promise(res => chrome.tabs.sendMessage(tabId, msg, r => { void chrome.runtime.lastError; res(r); }));
const bg = msg => new Promise(res => chrome.runtime.sendMessage(msg, r => { void chrome.runtime.lastError; res(r); }));
const save = obj => new Promise(res => chrome.storage.local.set(obj, res));
// Firefox and Zen treat site access as optional: the user can withhold or revoke it.
const ORIGINS = ["http://*/*", "https://*/*", "http://127.0.0.1:8765/*", "http://localhost:8765/*"];
const hasAccess = () => new Promise(res => {
  try { chrome.permissions.contains({ origins: ORIGINS }, ok => { void chrome.runtime.lastError; res(ok !== false); }); }
  catch (e) { res(true); }
});
let access = true;
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

async function serverStatus() {
  const h = await bg({ type: "health" });
  const ok = !!(h && h.ok);
  $("dot").className = "dot " + (ok ? "on" : "off");
  $("server").textContent = ok ? (h.roberta ? "Model server running: TF-IDF and RoBERTa" : "Model server running: TF-IDF")
                               : "Model server is not running";
  $("server-detail").innerHTML = ok
    ? (h.roberta ? "Running on this computer. Text is flagged only when both TF-IDF and RoBERTa agree." : "Running on this computer with the TF-IDF model.")
    : "Not running. Double-click <b>RUN_EXTENSION_SERVER.bat</b> in the project folder and wait until it says the API is ready.";
  return ok;
}

function render(r, serverOk) {
  const count = $("count"), label = $("count-label"), note = $("note");
  $("cats").innerHTML = ""; $("list").innerHTML = ""; note.innerHTML = "";
  $("reload").hidden = true; $("grant").hidden = true; $("report").hidden = true;
  if (!access && scannable()) {   // Firefox / Zen: site access not granted yet
    $("host").textContent = new URL(tabUrl).hostname;
    $("site-label").textContent = "No access to websites yet";
    $("site-on").disabled = true; $("scan").hidden = true;
    count.textContent = ""; count.className = "count idle";
    label.textContent = "Allow DarkPatternGuard to read pages";
    note.textContent = "It needs to read page text to check it. The text goes only to the model server on this computer.";
    $("grant").hidden = false;
    return;
  }
  $("scan").hidden = false;
  if (!r) {
    $("site-on").disabled = true; $("scan").disabled = true;
    count.textContent = ""; count.className = "count idle";
    if (scannable()) {   // a normal website where Chrome refused or the page is still loading
      $("host").textContent = new URL(tabUrl).hostname;
      $("site-label").textContent = "Not connected to this tab yet";
      label.textContent = "Reload the page to start checking it";
      $("reload").hidden = false;
    } else {             // chrome:// pages, the new tab page, the Web Store...
      $("host").textContent = "Browser page";
      $("site-label").textContent = "Not a website";
      label.textContent = "Browsers don't let extensions check their own pages";
      note.textContent = "Open a normal website, such as a shopping site.";
    }
    return;
  }
  $("host").textContent = r.host || "This page";
  $("site-on").disabled = false; $("scan").disabled = !serverOk;
  $("site-on").checked = r.siteOn;
  $("site-label").textContent = r.siteOn ? (r.auto ? "Scanning this site automatically" : "Automatic scanning is off") : "Paused on this site";

  if (!serverOk) {
    count.textContent = "!"; count.className = "count idle";
    label.textContent = "Can't scan right now";
    note.innerHTML = "Start the model server: double-click <code>RUN_EXTENSION_SERVER.bat</code>, then press Scan this page.";
    return;
  }
  if (!r.scannedOnce) {
    count.textContent = "–"; count.className = "count idle";
    label.textContent = r.siteOn ? "Not scanned yet" : "Scanning is paused here";
    note.textContent = "Press Scan this page to check it once.";
    return;
  }
  const n = r.findings.length;
  count.textContent = n; count.className = "count" + (n ? "" : " ok");
  label.textContent = n ? `possible dark pattern${n === 1 ? "" : "s"} on this page` : "Nothing flagged on this page";
  if (!n) note.textContent = `Checked ${plural(r.scanned, "piece")} of text.`;

  const cats = {};
  r.findings.forEach(f => { const k = f.category || "Dark pattern"; (cats[k] = cats[k] || { n: 0, lv: f.level }).n++; });
  $("cats").innerHTML = Object.entries(cats).sort((a, b) => b[1].n - a[1].n)
    .map(([k, v]) => `<span class="cat" style="--c:${COLOURS[v.lv]}"><i></i>${esc(k)} <b>${v.n}</b></span>`).join("");

  $("list").innerHTML = r.findings.map((f, i) => {
    const pct = f.level === "privacy" && f.p < 0.5 ? "rule" : `${Math.round(100 * f.p)}%`;
    return `<li data-i="${i}" style="--c:${COLOURS[f.level]}" title="Show on page">
      <span class="d"></span><span class="t">${esc(f.text)}</span><span class="pc">${pct}</span>
      <span class="k">${esc(f.category || "Dark pattern")}</span>
      <button class="open" data-i="${i}" title="Open an in-depth analysis of this text in the web app">Analyse ↗</button></li>`;
  }).join("");
  lastFindings = r.findings;
  $("report").hidden = !n;
  document.querySelectorAll("#list li").forEach(li => li.onclick = () => ask({ type: "reveal", idx: +li.dataset.i }));
  document.querySelectorAll("#list .open").forEach(b => b.onclick = e => {
    e.stopPropagation();
    openApp(new URLSearchParams({ text: lastFindings[+b.dataset.i].text.slice(0, 2000) }));
  });
}

const scannable = () => /^https?:/i.test(tabUrl);

// Tabs opened before the extension was installed or reloaded don't have the page script yet.
// The popup injects it on demand (allowed by activeTab while the popup is open).
async function getFindings() {
  let r = await ask({ type: "getFindings" });
  if (r || !scannable() || !chrome.scripting) return r;
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    await new Promise(res => setTimeout(res, 400));
    r = await ask({ type: "getFindings" });
  } catch (e) { /* e.g. the Web Store or a PDF viewer: Chrome blocks scripts there */ }
  return r;
}

async function refresh() {
  access = await hasAccess();
  const [ok, r] = await Promise.all([serverStatus(), access ? getFindings() : null]);
  if (r) host = r.host;
  render(r, ok);
}

function renderSettings() {
  $("auto").checked = cfg.autoScan;
  $("chip").checked = cfg.showChip;
  document.querySelectorAll(".segmented button").forEach(b => b.setAttribute("aria-checked", String(parseFloat(b.dataset.v) === cfg.minP)));
  $("sens-help").textContent = SENS_HELP[String(cfg.minP)];
  $("app-url").value = cfg.appUrl || DEFAULT_APP_URL;
  $("paused").innerHTML = cfg.pausedSites.length ? cfg.pausedSites.map(esc).join("<br>") : "None";
}

chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
  tabId = tabs[0] && tabs[0].id;
  tabUrl = (tabs[0] && (tabs[0].url || tabs[0].pendingUrl)) || "";
  chrome.storage.local.get(DEFAULTS, c => { cfg = c; renderSettings(); refresh(); });
});

$("site-on").onchange = async e => {
  if (!host) return;
  const set = new Set(cfg.pausedSites);
  e.target.checked ? set.delete(host) : set.add(host);
  cfg.pausedSites = [...set];
  await save({ pausedSites: cfg.pausedSites });
  renderSettings();
  setTimeout(refresh, 900);
};
$("scan").onclick = async () => {
  const b = $("scan");
  b.disabled = true; b.textContent = "Scanning…";
  await ask({ type: "scanNow" });
  b.textContent = "Scan this page";
  refresh();
};
$("grant").onclick = () => {
  // Must be called straight from the click. Firefox may close the popup while its prompt is open.
  chrome.permissions.request({ origins: ORIGINS }, ok => {
    void chrome.runtime.lastError;
    if (ok) { chrome.tabs.reload(tabId); window.close(); } else refresh();
  });
};
// Whole page: send the flagged sections (capped so the link stays a sensible length).
$("report").onclick = () => {
  const q = new URLSearchParams({ page: host || "" });
  let size = 0;
  for (const f of lastFindings) {
    const t = f.text.slice(0, 600);
    if (size + t.length > 6000 || q.getAll("t").length >= 40) break;
    q.append("t", t); size += t.length;
  }
  openApp(q);
};
$("app-url-save").onclick = async () => {
  const v = $("app-url").value.trim() || DEFAULT_APP_URL;
  let ok = false;
  try { ok = /^https?:$/.test(new URL(v).protocol); } catch (e) {}
  if (!ok) { $("app-url-msg").textContent = "Enter a full address starting with http:// or https://"; return; }
  cfg.appUrl = v.replace(/\/+$/, ""); await save({ appUrl: cfg.appUrl });
  $("app-url").value = cfg.appUrl; $("app-url-msg").textContent = "Saved.";
};
$("reload").onclick = () => { chrome.tabs.reload(tabId); window.close(); };
$("open-settings").onclick = () => { $("main").hidden = true; $("settings").hidden = false; };
$("close-settings").onclick = () => { $("settings").hidden = true; $("main").hidden = false; refresh(); };
$("auto").onchange = e => { cfg.autoScan = e.target.checked; save({ autoScan: cfg.autoScan }); };
$("chip").onchange = e => { cfg.showChip = e.target.checked; save({ showChip: cfg.showChip }); };
document.querySelectorAll(".segmented button").forEach(b => b.onclick = () => {
  cfg.minP = parseFloat(b.dataset.v); save({ minP: cfg.minP }); renderSettings();
});
