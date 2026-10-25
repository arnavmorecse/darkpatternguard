"""DarkPatternGuard - Streamlit demo.   Run:  streamlit run app.py"""
import os, sys, html
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import DarkPatternGuard, roberta_available
from fetch_models import ensure_roberta
import audit_engine

RED, NEUTRAL = "#D6293E", "#9AA1AB"
# Theme-neutral styling: borders and muted text are derived from the current text colour, so the custom
# blocks below read correctly in both the light and the dark theme (switchable in the top-right menu).
LINE = "color-mix(in srgb, currentColor 18%, transparent)"
MUTED = "color-mix(in srgb, currentColor 68%, transparent)"
RISK = {"High": "#D6293E", "Medium": "#E8710A", "Low": "#C99A06", "No": "#1E8E5A"}

icon = ROOT / "extension" / "icons" / "icon128.png"
st.set_page_config(page_title="DarkPatternGuard", page_icon=str(icon) if icon.exists() else None, layout="wide")
st.markdown(f"""<style>
  /* Streamlit's toolbar floats over the top of the page: clear it, then sit the content lower on tall screens */
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: clamp(5.5rem, 14vh, 10rem); padding-bottom: 4rem; max-width: 1180px; }}
  section[data-testid="stSidebar"] > div:first-child {{ padding-top: clamp(4rem, 12vh, 9rem); }}
  h1, h2, h3 {{ letter-spacing: -.01em; }}
  .brand {{ display:flex; align-items:center; gap:12px; margin-bottom:2px; }}
  .brand img {{ width:34px; height:34px; border-radius:8px; }}
  .brand h1 {{ font-size:1.75rem; margin:0; padding:0; }}
  .tagline {{ color:{MUTED}; margin:0 0 1.2rem; }}
  .verdict {{ display:grid; grid-template-columns:repeat(3,1fr); border:1px solid {LINE}; border-radius:8px; margin:.5rem 0 1.5rem; }}
  .verdict > div {{ padding:14px 18px; }}
  .verdict > div + div {{ border-left:1px solid {LINE}; }}
  .verdict .k {{ color:{MUTED}; font-size:.85rem; }}
  .verdict .v {{ font-size:1.25rem; font-weight:600; margin-top:2px; }}
  .verdict .s {{ color:{MUTED}; font-size:.85rem; }}
  .pill {{ display:inline-block; padding:2px 12px; border-radius:12px; color:#fff; font-weight:600; font-size:.95rem; }}
  .rule {{ padding:10px 14px; border:1px solid {LINE}; border-left:3px solid var(--rc); border-radius:6px; margin-bottom:10px; }}
  .rule b {{ font-weight:600; }} .rule .c {{ color:{MUTED}; }}
  .rule mark {{ background:rgba(214,41,62,.18); color:inherit; padding:0 4px; border-radius:3px; }}
  .rule .q {{ color:{MUTED}; font-size:.9rem; margin-top:2px; }}
  .ref {{ padding:8px 0; border-top:1px solid {LINE}; }} .ref:first-of-type {{ border-top:0; }}
  .ref .law {{ color:{MUTED}; font-size:.9rem; }}
  .fine {{ color:#8A9099; font-size:.82rem; margin-top:1.5rem; }}
  @media (max-width: 700px) {{ .verdict {{ grid-template-columns:1fr; }} .verdict > div + div {{ border-left:0; border-top:1px solid {LINE}; }} }}
</style>""", unsafe_allow_html=True)

esc = lambda s: html.escape(str(s))


@st.cache_resource(show_spinner="Loading the models. The first time takes up to a minute.")
def load(backend):
    return DarkPatternGuard(backend=backend)


def logo_tag():
    if not icon.exists():
        return ""
    import base64
    return f'<img src="data:image/png;base64,{base64.b64encode(icon.read_bytes()).decode()}" alt="">'


@st.cache_data(show_spinner=False, max_entries=300)
def analyse_cached(backend, text, n_samples, with_lime=True):
    return load(backend).analyse(text, num_samples=n_samples, with_lime=with_lime)


def show_result(r, k):
    """Verdict, LIME chart, rule findings, regulations and downloads for one analysed text."""
    s1, s2 = r["stage1"], r["stage2"]
    word = r["risk_level"].split(" ")[0]
    sub = r["risk_level"].split(" - ")[1] if " - " in r["risk_level"] else "No action needed"
    st.markdown(f"""<div class="verdict">
      <div><div class="k">Detection</div><div class="v">{"Likely a dark pattern" if s1["label"] else "Not a dark pattern"}</div>
           <div class="s">{100*s1["p_dark"]:.0f}% likely</div></div>
      <div><div class="k">Type</div><div class="v">{esc(s2["category"]) if s2 else "None"}</div>
           <div class="s">{f'{100*s2["confidence"]:.0f}% confidence' if s2 else "No type assigned"}</div></div>
      <div><div class="k">Review priority</div><div class="v"><span class="pill" style="background:{RISK[word]}">{esc(r["risk_level"].split(" - ")[0])}</span></div>
           <div class="s">{esc(sub.capitalize())}</div></div></div>""", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Which words drove the decision")
        if r["explanation"]:
            w = pd.DataFrame(r["explanation"]["weights"], columns=["word", "weight"])
            w["direction"] = w["weight"].map(lambda v: "towards" if v > 0 else "away")
            chart = alt.Chart(w).mark_bar(cornerRadius=2).encode(
                x=alt.X("weight:Q", title=f"Push towards “{r['explanation']['class_name']}”"),
                y=alt.Y("word:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=180, labelFontSize=13)),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["towards", "away"], range=[RED, NEUTRAL]), legend=None),
                tooltip=["word", alt.Tooltip("weight:Q", format="+.3f")]).properties(height=max(180, 34 * len(w)))
            st.altair_chart(chart, width="stretch", key=f"{k}_lime")
            st.caption("Red words push the prediction towards a dark pattern, grey words push away. "
                       "LIME estimates this by removing words at random, so treat it as a guide, not proof.")
        else:
            st.caption("The text is a single word, so there's nothing to compare.")
        if s2:
            st.subheader("How sure it is about the type")
            p = pd.DataFrame({"type": list(s2["probs"]), "p": list(s2["probs"].values())})
            p["top"] = p["type"] == s2["category"]
            st.altair_chart(alt.Chart(p).mark_bar(cornerRadius=2).encode(
                x=alt.X("p:Q", title="Probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("type:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=180, labelFontSize=13)),
                color=alt.Color("top:N", scale=alt.Scale(domain=[True, False], range=[RED, NEUTRAL]), legend=None),
                tooltip=["type", alt.Tooltip("p:Q", format=".1%")]).properties(height=180), width="stretch", key=f"{k}_type")
    with right:
        st.subheader("Rule findings")
        if r["rule_hits"]:
            for h in r["rule_hits"]:
                rc = "#7A4FD0" if h["ccpa"] else RISK[word]
                marks = " ".join(f"<mark>{esc(e)}</mark>" for e in h["evidence"])
                st.markdown(f'<div class="rule" style="--rc:{rc}"><b>{esc(h["name"])}</b> <span class="c">({esc(h["category"])})</span>'
                            f'<div>Matched {marks}</div><div class="q">To check: {esc(h["question"])}</div></div>',
                            unsafe_allow_html=True)
        else:
            st.write("No rule matched this text.")
        if r["model_rule_agreement"] is not None:
            st.caption("The model and the rules point to the same type." if r["model_rule_agreement"]
                       else "The model and the rules disagree on the type. Worth a closer look.")
        st.subheader("Regulations to review")
        if r["regulatory_references"]:
            for ref in r["regulatory_references"]:
                st.markdown(f'<div class="ref"><b>{esc(ref["provision"])}</b><div class="law">{esc(ref["law"])}, '
                            f'relevant to {esc(ref["for_category"])}. <a href="{esc(ref["source"])}" target="_blank">Source</a></div></div>',
                            unsafe_allow_html=True)
        else:
            st.write("None for this text.")

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download report (HTML)", audit_engine.render(r, "html"), file_name="audit_report.html", mime="text/html", key=f"{k}_dl_html")
    d2.download_button("Download report (Markdown)", audit_engine.render(r, "md"), file_name="audit_report.md", key=f"{k}_dl_md")
    st.markdown(f'<p class="fine">{esc(r["disclaimer"])}</p>', unsafe_allow_html=True)


# Links from the browser extension:  ?text=<one piece of text>   or   ?page=<site>&t=<text>&t=<text>...
qp = st.query_params
link_text = qp.get("text", "").strip()[:2000]
link_model = qp.get("model", "roberta" if (link_text or qp.get_all("t")) else "")   # links use RoBERTa by default
page_name = qp.get("page", "").strip()[:200]
page_texts = [t.strip()[:600] for t in qp.get_all("t") if t.strip()][:40]

st.markdown(f'<div class="brand">{logo_tag()}<h1>DarkPatternGuard</h1></div>'
            '<p class="tagline">Paste any text from a website, such as a banner, button or pop-up, and see whether it uses a '
            'manipulative design trick, which kind, and why.</p>', unsafe_allow_html=True)

def _models_url():
    try:
        return st.secrets.get("MODELS_URL", "") or os.environ.get("MODELS_URL", "")
    except Exception:   # no secrets file
        return os.environ.get("MODELS_URL", "")


@st.cache_resource(show_spinner="Downloading the RoBERTa models (about 480 MB). This happens once, after the app starts.")
def fetch_roberta(url):
    return ensure_roberta(url)


fetched, fetch_msg = fetch_roberta(_models_url())

with st.sidebar:
    st.subheader("Model")
    options = (["roberta"] if roberta_available() else []) + ["tfidf"]
    backend = st.radio("Model", options, label_visibility="collapsed",
                       index=options.index(link_model) if link_model in options else 0,
                       format_func=lambda b: {"roberta": "RoBERTa (fine-tuned, more accurate)",
                                              "tfidf": "TF-IDF + logistic regression (fast baseline)"}[b])
    if not roberta_available():
        if fetch_msg == "MODELS_URL is not set":   # running locally without the fine-tuned weights
            st.info("RoBERTa isn't installed yet, so the fast baseline is used. Run the Colab notebook and unzip "
                    "roberta_outputs.zip into the project folder to add it.")
        else:
            st.info(f"RoBERTa couldn't be loaded ({fetch_msg}), so the fast baseline is used.")
    st.subheader("Try an example")
    examples = {
        "Countdown and low stock": "Hurry! Only 3 left in stock - offer ends in 02:14:55",
        "Guilt-trip decline button": "No thanks, I don't want to save money",
        "Someone just bought this": "Michael in Pontiac, United States purchased a Classic Tee 4 minutes ago",
        "Cookie banner": "We use cookies to personalise ads. Accept All | Ask me later",
        "Ordinary text": "Free returns within 30 days of delivery",
    }
    pick = st.selectbox("Example", ["Write my own"] + list(examples), label_visibility="collapsed")
    with st.expander("Advanced"):
        n_samples = st.slider("Explanation samples", 100, 1500, 400, 100,
                              help="How many variations LIME tries. More gives a steadier explanation but is slower.")

g = load(backend)
tabs = st.tabs((["This page"] if page_texts else []) + ["Check a piece of text", "Check a whole file"])
if page_texts:
    tab_page, tab1, tab2 = tabs
    with tab_page:
        st.markdown(f"The extension flagged **{len(page_texts)} section{'s' if len(page_texts) != 1 else ''}** "
                    f"on **{esc(page_name) or 'this page'}**. Here is what the models and rules make of each one.")
        rows = []
        with st.spinner("Checking the sections…"):
            for t in page_texts:
                r = analyse_cached(backend, t, n_samples, with_lime=False)
                kind = r["stage2"]["category"] if r["stage2"] else ""
                if any(h["ccpa"] for h in r["rule_hits"]):   # same labelling as the extension
                    kind = "Privacy choice / consent"
                rows.append({"text": t, "likely dark (%)": round(100 * r["stage1"]["p_dark"]), "type": kind,
                             "review priority": r["risk_level"].split(" - ")[0],
                             "rules matched": ", ".join(h["name"] for h in r["rule_hits"])})
        out = pd.DataFrame(rows)
        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button("Download this table (CSV)", out.to_csv(index=False), "page_audit.csv", key="page_csv")
        st.subheader("Look at one section in depth")
        i = st.selectbox("Section", range(len(page_texts)), label_visibility="collapsed",
                         format_func=lambda i: page_texts[i] if len(page_texts[i]) < 110 else page_texts[i][:107] + "…")
        with st.spinner("Working out which words mattered…"):
            r = analyse_cached(backend, page_texts[i], n_samples)
        st.markdown(f"> {esc(page_texts[i])}")
        show_result(r, "page")
else:
    tab1, tab2 = tabs

with tab1:
    text = st.text_area("Text from the website", examples.get(pick) or link_text, height=96,
                        placeholder="For example: Only 2 rooms left at this price!")
    go = st.button("Check text", type="primary")
    if link_text and not st.session_state.get("_link_done"):   # opened from the extension: analyse straight away
        go, st.session_state["_link_done"] = True, True
        st.caption("Sent from the DarkPatternGuard extension and checked automatically with "
                   + ("RoBERTa." if backend == "roberta" else "the TF-IDF model (RoBERTa isn't installed)."))
    if go and text.strip():
        with st.spinner("Checking the text and working out which words mattered…"):
            r = g.analyse(text.strip(), num_samples=n_samples)
        show_result(r, "single")
    elif go:
        st.warning("Paste some text from a website first.")

with tab2:
    st.write("Upload a CSV file with a column called **text**, one piece of website text per row. "
             "Explanations are skipped here to keep it fast.")
    f = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    if f:
        df = pd.read_csv(f)
        if "text" not in df:
            st.error("This file has no column called “text”. Rename the column with the website text to “text” and upload it again.")
        else:
            rows, texts = [], df["text"].astype(str).head(500).tolist()
            bar = st.progress(0.0, text="Checking rows…")
            for i, t in enumerate(texts):
                r = g.analyse(t, with_lime=False)
                rows.append({"text": t, "likely dark (%)": round(100 * r["stage1"]["p_dark"]),
                             "type": r["stage2"]["category"] if r["stage2"] else "",
                             "review priority": r["risk_level"].split(" - ")[0],
                             "rules matched": ", ".join(h["name"] for h in r["rule_hits"])})
                bar.progress((i + 1) / len(texts), text=f"Checked {i + 1} of {len(texts)} rows")
            bar.empty()
            out = pd.DataFrame(rows)
            flagged = int((out["likely dark (%)"] >= 50).sum())
            st.markdown(f"**{flagged} of {len(out)}** rows look like possible dark patterns.")
            st.dataframe(out, width="stretch", hide_index=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False), "batch_audit.csv")
