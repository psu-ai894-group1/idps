"""
AI-894
This file provides a Streamlit-based web dashboard for interacting with the intrusion detection system. It allows users to run batch tests or perform live traffic detection.
Authors: Karla Gonzalez Caballero (kxg5613@psu.edu), Christopher Umbel (czu5008@psu.edu)

IDPS — Intrusion Detection Dashboard
=====================================
Screens:
  0  HOME      — two big buttons: Import Test Data / Live Detection
  1  BATCH     — file upload + run
  2  SNIFF     — interface picker, duration, start/stop
  3  CAPTURING — countdown progress bar (live sniff only)
  4  ANALYZING — spinner while model runs
  5  RESULTS   — full eval + DB history
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, time, subprocess, threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDPS · Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — blue theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #070b14;
    color: #cbd5e1;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Streamlit 1.55 — the actual container that holds page top padding */
.stMainBlockContainer {
    padding-top: 0.75rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 0 !important;
}

/* ── hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"]     { display: none !important; }
[data-testid="stToolbar"]    { display: none !important; }

/* ── page shell ── */
.page {
    padding: 1.25rem 3rem 4rem;
    max-width: 1360px;
    margin: 0 auto;
}

/* ── top nav bar ── */
.navbar {
    display: flex; align-items: center; gap: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #0f1c2e;
    margin-bottom: 1.5rem;
}
.nav-logo {
    display: flex; align-items: center; gap: .6rem;
    font-size: 1.05rem; font-weight: 700; color: #e2e8f0;
    letter-spacing: -.01em;
}
.nav-logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.nav-status {
    margin-left: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .68rem; color: #3b82f6;
    letter-spacing: .06em;
    background: #0a1628; border: 1px solid #1e3a5f;
    padding: 4px 12px; border-radius: 999px;
}
.nav-back {
    font-size: .8rem; color: #3b82f6; cursor: pointer;
    border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 5px 14px; background: #0a1628;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: .03em;
    text-decoration: none;
    transition: background .15s;
}
.nav-back:hover { background: #0f2040; }

/* ── section heading ── */
.section-h { font-size: 1.45rem; font-weight: 700; color: #e2e8f0; margin-bottom: .2rem; margin-top: 0; }
.section-sub { font-size: .82rem; color: #64748b; margin-bottom: 1.25rem; margin-top: 0; }

/* ── home cards ── */
.home-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem; }
.home-card {
    background: #0a1628;
    border: 1.5px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.2rem 2rem;
    cursor: pointer;
    transition: border-color .18s, transform .18s;
    position: relative; overflow: hidden;
}
.home-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1d4ed8, #3b82f6);
    border-radius: 16px 16px 0 0;
}
.home-card:hover { border-color: #3b82f6; transform: translateY(-2px); }
.home-card-icon { font-size: 2.2rem; margin-bottom: 1rem; }
.home-card-title { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-bottom: .4rem; }
.home-card-desc  { font-size: .82rem; color: #64748b; line-height: 1.55; }

/* ── form card ── */
.form-card {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.25rem;
    overflow: hidden;
}
.form-card-title {
    font-size: .75rem; font-weight: 600; color: #3b82f6;
    text-transform: uppercase; letter-spacing: .1em;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: .5rem;
}
.form-card-title::before { content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #3b82f6; }

/* ── label above inputs ── */
.field-label {
    font-size: .72rem; font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase; letter-spacing: .08em;
    color: #475569; margin-bottom: .35rem;
}

/* ── primary button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #fff; border: none; border-radius: 9px;
    padding: .65rem 1.5rem; font-size: .88rem; font-weight: 600;
    font-family: 'IBM Plex Sans', sans-serif;
    letter-spacing: .01em; width: 100%;
    box-shadow: 0 4px 14px rgba(37,99,235,.35);
    transition: opacity .15s;
}
div[data-testid="stButton"] > button:hover { opacity: .88; }

/* ── metric grid ── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.m-tile {
    background: #0a1628; border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 1.2rem 1.4rem; position: relative; overflow: hidden;
}
.m-tile::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 12px 12px 0 0;
}
.m-tile.g::after  { background: #22c55e; }
.m-tile.b::after  { background: #3b82f6; }
.m-tile.y::after  { background: #f59e0b; }
.m-tile.r::after  { background: #ef4444; }
.m-label { font-size: .65rem; color: #475569; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.m-value { font-size: 1.9rem; font-weight: 700; color: #e2e8f0; line-height: 1; font-family: 'IBM Plex Mono', monospace; }
.m-sub   { font-size: .68rem; color: #334155; margin-top: 5px; font-family: 'IBM Plex Mono', monospace; }

/* ── alert banners ── */
.banner-ok  { background: #052e16; border: 1.5px solid #15803d; border-radius: 10px; padding: .9rem 1.3rem; display: flex; align-items: center; gap: .8rem; margin-bottom: 1.5rem; }
.banner-bad { background: #2d0a0a; border: 1.5px solid #dc2626; border-radius: 10px; padding: .9rem 1.3rem; display: flex; align-items: center; gap: .8rem; margin-bottom: 1.5rem; }
.banner-icon  { font-size: 1.4rem; }
.banner-title { font-size: .95rem; font-weight: 700; color: #f1f5f9; }
.banner-body  { font-size: .78rem; color: #94a3b8; margin-top: 2px; font-family: 'IBM Plex Mono', monospace; }

/* ── chart card ── */
.chart-card {
    background: #0a1628; border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1.25rem;
}
.chart-title {
    font-size: .72rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: .09em; margin-bottom: .9rem;
    display: flex; align-items: center; gap: .45rem;
}
.chart-title::before { content: ''; display: inline-block; width: 5px; height: 5px; border-radius: 50%; background: #3b82f6; }

/* ── countdown ── */
.countdown-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 60vh; gap: 1.5rem;
    text-align: center;
}
.countdown-ring {
    width: 160px; height: 160px; border-radius: 50%;
    border: 6px solid #1e3a5f; position: relative;
    display: flex; align-items: center; justify-content: center;
}
.countdown-number {
    font-size: 3rem; font-weight: 700; color: #3b82f6;
    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}
.countdown-label { font-size: .75rem; color: #475569; font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .1em; }
.analyzing-wrap  { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; gap: 1.2rem; text-align: center; }
.analyzing-title { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; }
.analyzing-sub   { font-size: .8rem; color: #475569; font-family: 'IBM Plex Mono', monospace; }

/* ── DB table ── */
.db-section { margin-top: 2.5rem; border-top: 1px solid #0f1c2e; padding-top: 2rem; }

/* ── streamlit overrides ── */
div[data-testid="stFileUploader"] {
    background: #070b14; border: 1.5px dashed #1e3a5f;
    border-radius: 10px; padding: .5rem;
}
div[data-testid="stFileUploader"]:hover { border-color: #3b82f6; }
div[data-testid="stSlider"] > div { color: #64748b; }
div[data-testid="stInfo"]    { background: #0a1628; border: 1px solid #1e3a5f; border-radius: 8px; color: #64748b; }
div[data-testid="stSuccess"] { background: #052e16; border: 1px solid #15803d; border-radius: 8px; }
div[data-testid="stWarning"] { background: #1c1400; border: 1px solid #b45309; border-radius: 8px; }
div[data-testid="stError"]   { background: #1a0505; border: 1px solid #dc2626; border-radius: 8px; }
div[data-testid="stDataFrame"] { border: 1px solid #1e3a5f; border-radius: 8px; overflow: hidden; }
div[data-testid="stSpinner"] > div { color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
SCREENS = ["home", "batch_setup", "sniff_setup", "capturing", "analyzing", "results"]
for k, v in {
    "screen":          "home",
    "mode":            None,
    "results":         None,
    "sniff_csv":       None,
    "sniff_proc":      None,
    "capture_start":   None,
    "capture_dur":     30,
    "sniff_iface":     "eth0",
    "uploaded_file":   None,
    "capture_done":    False,   # True once timer finished / stopped early
    "capture_has_data": None,   # True/False/None (None = not checked yet)
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def goto(screen):
    st.session_state.screen = screen
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Heavy model — cached
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    import joblib
    from model import load_model_weights
    MD  = os.path.join(PROJECT_ROOT, "models")
    scaler = joblib.load(os.path.join(MD, "idps.weights.h5.scaler.joblib"))
    model  = load_model_weights(os.path.join(MD, "idps.weights.h5"),
                                num_features=len(scaler.mean_))
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_on_df(flows_df):
    import tensorflow as tf
    from flows_to_tensors import flows_to_tensors
    from model import predict, _normalize_adjacency, _scipy_to_tf_sparse
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, log_loss,
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score,
    )
    from config import class_names

    model, scaler = load_model_and_scaler()
    node_features, adjacency, labels, _ = flows_to_tensors(flows_df, scaler=scaler, log_transform=True)
    preds, confidences = predict(model, node_features, adjacency)

    adj_tf  = _scipy_to_tf_sparse(_normalize_adjacency(adjacency))
    probs   = tf.nn.softmax(model(node_features, adj_tf), axis=1).numpy()

    res = {
        "preds": preds, "confidences": confidences, "probs": probs,
        "flows_df": flows_df, "class_names": class_names,
        "n_flows": len(flows_df),
    }

    if labels is not None:
        true = labels.numpy()
        n    = len(class_names)
        res.update({
            "true": true,
            "accuracy":          accuracy_score(true, preds),
            "balanced_accuracy": balanced_accuracy_score(true, preds),
            "f1":                f1_score(true, preds, average="macro", zero_division=0),
            "log_loss":          log_loss(true, probs),
            "confusion_matrix":  confusion_matrix(true, preds, labels=list(range(n))),
        })
        roc_data, pr_data = {}, {}
        for i, name in enumerate(class_names):
            yb = (true == i).astype(int)
            fpr, tpr, _ = roc_curve(yb, probs[:, i])
            roc_data[name] = (fpr, tpr, auc(fpr, tpr))
            prec, rec, _   = precision_recall_curve(yb, probs[:, i])
            pr_data[name]  = (prec, rec, average_precision_score(yb, probs[:, i]))
        res["roc_data"] = roc_data
        res["pr_data"]  = pr_data

    return res


def save_to_db(res):
    from db import save_flows_and_inferences
    save_flows_and_inferences(
        res["flows_df"], res["preds"], res["confidences"], res["class_names"])


def load_db_history(limit=100):
    try:
        from db import get_connection
        conn = get_connection()
        df = pd.read_sql_query("""
            SELECT f.id, f.captured_at, f.src_ip, f.src_port,
                   f.dst_ip, f.dst_port,
                   i.predicted_label, ROUND(i.confidence,4) AS confidence
            FROM flows f
            INNER JOIN inferences i ON i.flow_id = f.id
            ORDER BY f.captured_at DESC LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers  (blue dark theme)
# ─────────────────────────────────────────────────────────────────────────────
PAL = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a78bfa", "#34d399"]

def _fig(w=5.2, h=4.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#070b14")
    ax.tick_params(colors="#475569", labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#0f1c2e")
    ax.xaxis.label.set_color("#475569")
    ax.yaxis.label.set_color("#475569")
    ax.title.set_color("#64748b")
    ax.grid(True, color="#0f1c2e", linewidth=0.8)
    return fig, ax

def plot_cm(cm, class_names):
    fig, ax = _fig(5.5, 4.4)
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_cm", ["#070d1a","#1d4ed8","#3b82f6"])
    im   = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8, color="#94a3b8")
    ax.set_yticklabels(class_names, fontsize=8, color="#94a3b8")
    ax.set_xlabel("Predicted", fontsize=8.5, labelpad=8, color="#475569")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    vmax = max(cm.max(), 1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm[i, j]
            c = "#070b14" if v > vmax * 0.5 else "#3b82f6"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color=c, fontweight="700")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors="#475569")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#475569")
    fig.tight_layout(); return fig

def plot_roc(roc_data):
    fig, ax = _fig()
    ax.plot([0,1],[0,1], color="#1e3a5f", lw=1.2, linestyle="--")
    for i, (name, (fpr, tpr, ra)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, color=PAL[i % len(PAL)], lw=2,
                label=f"{name}  AUC={ra:.3f}")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False Positive Rate", fontsize=8.5)
    ax.set_ylabel("True Positive Rate", fontsize=8.5)
    ax.set_title("ROC Curves", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right",
              facecolor="#0a1628", edgecolor="#1e3a5f", labelcolor="#94a3b8")
    fig.tight_layout(); return fig

def plot_pr(pr_data):
    fig, ax = _fig()
    for i, (name, (prec, rec, ap)) in enumerate(pr_data.items()):
        ax.plot(rec, prec, color=PAL[i % len(PAL)], lw=2,
                label=f"{name}  AP={ap:.3f}")
    ax.set_xlim([0,1.02]); ax.set_ylim([0,1.02])
    ax.set_xlabel("Recall", fontsize=8.5); ax.set_ylabel("Precision", fontsize=8.5)
    ax.set_title("Precision-Recall Curves", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left",
              facecolor="#0a1628", edgecolor="#1e3a5f", labelcolor="#94a3b8")
    fig.tight_layout(); return fig

def plot_bar(preds, class_names):
    counts = np.bincount(preds, minlength=len(class_names))
    fig, ax = _fig(5.2, 3.0)
    bars = ax.bar(class_names, counts,
                  color=[PAL[i % len(PAL)] for i in range(len(class_names))],
                  width=0.55, edgecolor="#070b14", linewidth=1.2)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(counts, default=[1])*0.01,
                f"{c:,}", ha="center", va="bottom", fontsize=8, color="#64748b")
    ax.set_title("Prediction Distribution", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Count", fontsize=8.5); ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); return fig


# ─────────────────────────────────────────────────────────────────────────────
# Shared nav bar
# ─────────────────────────────────────────────────────────────────────────────
def render_navbar(show_back=True, back_label="← Start Over"):
    col_logo, col_back, col_status = st.columns([3, 2, 2])
    with col_logo:
        st.markdown("""
        <div class="nav-logo">
          <div class="nav-logo-icon">🛡️</div>
          IDPS · Intrusion Detection
        </div>""", unsafe_allow_html=True)
    with col_back:
        if show_back:
            if st.button(back_label, key="nav_back", use_container_width=True):
                # kill any running sniff process
                if st.session_state.sniff_proc and st.session_state.sniff_proc.poll() is None:
                    st.session_state.sniff_proc.terminate()
                st.session_state.sniff_proc       = None
                st.session_state.sniff_csv        = None
                st.session_state.capture_start    = None
                st.session_state.capture_done     = False
                st.session_state.capture_has_data = None
                st.session_state.results          = None
                goto("home")
    with col_status:
        st.markdown('<div style="text-align:right"><span class="nav-status">● SYSTEM ONLINE</span></div>',
                    unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #0f1c2e;margin:.4rem 0 1.2rem;">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 0 — HOME
# ═══════════════════════════════════════════════════════════════════════════
def screen_home():
    st.markdown('<div class="page">', unsafe_allow_html=True)
    render_navbar(show_back=False)

    st.markdown("""
    <div class="section-h">Network Intrusion Detection System</div>
    <div class="section-sub">GCN-based real-time and batch network flow classifier. Choose how you want to run the model.</div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="home-card">
          <div class="home-card-icon">📂</div>
          <div class="home-card-title">Import Test Data</div>
          <div class="home-card-desc">
            Upload a CIC-IDS-2017 labeled CSV and run the GCN classifier on it.
            See accuracy, F1, confusion matrix, ROC and PR curves.
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Import Test Data →", key="home_batch", use_container_width=True):
            st.session_state.mode = "batch"
            goto("batch_setup")

    with col2:
        st.markdown("""
        <div class="home-card">
          <div class="home-card-icon">📡</div>
          <div class="home-card-title">Live Detection</div>
          <div class="home-card-desc">
            Capture live network traffic from an interface, classify flows
            in real time, and get alerted instantly if an attack is detected.
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Live Detection →", key="home_sniff", use_container_width=True):
            st.session_state.mode = "sniff"
            goto("sniff_setup")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 1 — BATCH SETUP
# ═══════════════════════════════════════════════════════════════════════════
def screen_batch_setup():
    st.markdown('<div class="page">', unsafe_allow_html=True)
    render_navbar()

    st.markdown("""
    <div class="section-h">Import Test Data</div>
    <div class="section-sub">Upload a CIC-IDS-2017 formatted CSV. The model will classify every flow and compute full evaluation metrics.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-card"><div class="form-card-title">Dataset File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your CSV here", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        run = st.button("▶  Run Model", key="batch_run", use_container_width=True,
                        disabled=(uploaded is None))

    if uploaded is None:
        st.info("📎  Upload a CSV file above to enable the Run button.")

    if run and uploaded is not None:
        # go straight to analyzing
        st.session_state.uploaded_file = uploaded
        st.session_state.screen = "analyzing"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 2 — SNIFF SETUP
# ═══════════════════════════════════════════════════════════════════════════
def screen_sniff_setup():
    st.markdown('<div class="page">', unsafe_allow_html=True)
    render_navbar()

    st.markdown("""
    <div class="section-h">Live Detection Setup</div>
    <div class="section-sub">Configure the network interface and capture duration, then start sniffing.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-card"><div class="form-card-title">Capture Settings</div>', unsafe_allow_html=True)

    col_i, col_d = st.columns(2)
    with col_i:
        st.markdown('<div class="field-label">Network Interface</div>', unsafe_allow_html=True)
        iface = st.text_input("", value=st.session_state.sniff_iface,
                              placeholder="e.g. eth0, Wi-Fi, en0",
                              label_visibility="collapsed", key="iface_input")
    with col_d:
        st.markdown('<div class="field-label">Capture Duration (seconds)</div>', unsafe_allow_html=True)
        duration = st.slider("", min_value=10, max_value=300, value=st.session_state.capture_dur,
                             label_visibility="collapsed", key="dur_slider")

    st.markdown(f"""
    <div style="margin-top:.75rem;padding:.75rem 1rem;background:#070b14;border:1px solid #0f1c2e;
                border-radius:8px;font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#475569;">
        Interface: <span style="color:#3b82f6">{iface}</span> &nbsp;·&nbsp;
        Duration: <span style="color:#3b82f6">{duration}s</span> &nbsp;·&nbsp;
        Output: <span style="color:#3b82f6">output/sniff_*.csv</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("📡  Start Capture", key="sniff_go", use_container_width=True):
            st.session_state.sniff_iface = iface
            st.session_state.capture_dur = duration

            out_dir  = os.path.join(PROJECT_ROOT, "output")
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"sniff_{int(time.time())}.csv")
            st.session_state.sniff_csv = csv_path

            try:
                proc = subprocess.Popen(
                    [sys.executable, os.path.join(PROJECT_ROOT, "sniff.py"),
                     "-i", iface, "--to-csv-path", csv_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if sys.platform == "win32" else 0,
                )
                st.session_state.sniff_proc       = proc
                st.session_state.capture_start    = time.time()
                st.session_state.capture_done     = False
                st.session_state.capture_has_data = None
                goto("capturing")
            except Exception as e:
                st.error(f"❌  Failed to start sniffer: {e}\n\n"
                         "Make sure pyflowmeter is installed and you have admin/root privileges.")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 3 — CAPTURING  (live countdown → done state)
# ═══════════════════════════════════════════════════════════════════════════
def screen_capturing():
    st.markdown('<div class="page">', unsafe_allow_html=True)
    render_navbar(back_label="✕  Cancel Capture")

    duration   = st.session_state.capture_dur
    start_time = st.session_state.capture_start or time.time()
    elapsed    = time.time() - start_time
    remaining  = max(0.0, duration - elapsed)
    pct        = min(elapsed / duration, 1.0)
    proc       = st.session_state.sniff_proc

    # ── Already done (stopped early or timer finished on a previous rerun) ─
    if st.session_state.capture_done:
        _show_capture_done()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Process died before timer ended — show error ──────────────────────
    if proc is not None and proc.poll() is not None and remaining > 0:
        returncode = proc.returncode
        try:
            stderr_out = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        except Exception:
            stderr_out = ""
        st.session_state.sniff_proc = None

        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    gap:1rem;padding:3rem 0;text-align:center;">
          <div style="font-size:2.8rem;">❌</div>
          <div style="font-size:1.25rem;font-weight:700;color:#ef4444;">
            Sniffer Process Crashed
          </div>
          <div style="font-size:.8rem;color:#64748b;font-family:'IBM Plex Mono',monospace;
                      max-width:500px;line-height:1.6;">
            The capture process exited early (code {returncode}).<br>
            Common causes: wrong interface name, missing admin privileges,
            or pyflowmeter not installed.
          </div>
        </div>""", unsafe_allow_html=True)

        if stderr_out.strip():
            with st.expander("Show error details"):
                st.code(stderr_out[-2000:], language="text")

        _, col_retry, _ = st.columns([2, 1, 2])
        with col_retry:
            if st.button("↩  Back to Setup", key="crash_retry", use_container_width=True):
                st.session_state.capture_done     = False
                st.session_state.capture_has_data = None
                goto("sniff_setup")

        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Timer elapsed — stop the process now ──────────────────────────────
    if remaining <= 0:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=8)
            except Exception: pass
        st.session_state.sniff_proc   = None
        st.session_state.capture_done = True
        # Don't st.rerun() here — fall through to _show_capture_done immediately
        _show_capture_done()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Timer still running — show countdown ──────────────────────────────
    mins     = int(remaining) // 60
    secs     = int(remaining) % 60
    time_str = f"{mins:02d}:{secs:02d}" if mins > 0 else f"{int(remaining)}s"

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                gap:1.2rem;padding:2.5rem 0;text-align:center;">
      <div style="font-size:.72rem;color:#3b82f6;font-family:'IBM Plex Mono',monospace;
                  text-transform:uppercase;letter-spacing:.12em;">
        📡 &nbsp;Capturing on &nbsp;<strong style="color:#e2e8f0">{st.session_state.sniff_iface}</strong>
      </div>
      <div style="width:150px;height:150px;border-radius:50%;
                  border:5px solid #1e3a5f;
                  background:conic-gradient(#3b82f6 {pct*360:.0f}deg, #0f1c2e 0deg);
                  display:flex;align-items:center;justify-content:center;">
        <div style="width:118px;height:118px;border-radius:50%;background:#070b14;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;">
          <div style="font-size:2.6rem;font-weight:700;color:#3b82f6;
                      font-family:'IBM Plex Mono',monospace;line-height:1;">{time_str}</div>
          <div style="font-size:.62rem;color:#475569;font-family:'IBM Plex Mono',monospace;
                      text-transform:uppercase;letter-spacing:.1em;">remaining</div>
        </div>
      </div>
      <div style="font-size:.75rem;color:#334155;font-family:'IBM Plex Mono',monospace;">
        {pct*100:.0f}% &nbsp;·&nbsp; {os.path.basename(st.session_state.sniff_csv or 'sniff.csv')}
      </div>
    </div>""", unsafe_allow_html=True)

    st.progress(pct)

    _, col_stop, _ = st.columns([2, 1, 2])
    with col_stop:
        if st.button("⏹  Stop Early & Analyze", key="stop_early", use_container_width=True):
            if proc and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=8)
                except Exception: pass
            st.session_state.sniff_proc   = None
            st.session_state.capture_done = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    time.sleep(1)
    st.rerun()


def _show_capture_done():
    """Render the post-capture UI: check for data, show Run Model or No Traffic."""
    if st.session_state.capture_has_data is None:
        csv_path = st.session_state.sniff_csv or ""
        has_data = False
        if csv_path and os.path.exists(csv_path):
            try:
                df_peek  = pd.read_csv(csv_path, nrows=2)
                has_data = len(df_peek) > 0
            except Exception:
                has_data = False
        st.session_state.capture_has_data = has_data

    has_data = st.session_state.capture_has_data

    if has_data:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    gap:1rem;padding:3rem 0;text-align:center;">
          <div style="font-size:2.8rem;">✅</div>
          <div style="font-size:1.25rem;font-weight:700;color:#e2e8f0;">Capture Complete</div>
          <div style="font-size:.8rem;color:#475569;font-family:'IBM Plex Mono',monospace;">
            Network flows recorded successfully. Ready to classify.
          </div>
        </div>""", unsafe_allow_html=True)

        _, col_run, _ = st.columns([2, 1, 2])
        with col_run:
            if st.button("▶  Run Model", key="run_after_capture", use_container_width=True):
                st.session_state.capture_done     = False
                st.session_state.capture_has_data = None
                goto("analyzing")
    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    gap:1rem;padding:3rem 0;text-align:center;">
          <div style="font-size:2.8rem;">📭</div>
          <div style="font-size:1.25rem;font-weight:700;color:#e2e8f0;">
            No Network Traffic Captured
          </div>
          <div style="font-size:.8rem;color:#64748b;font-family:'IBM Plex Mono',monospace;
                      max-width:420px;line-height:1.6;">
            No flows were finalized during the capture window.<br>
            Try a longer duration, verify the interface name,<br>
            or generate some traffic during capture.
          </div>
        </div>""", unsafe_allow_html=True)

        _, col_retry, _ = st.columns([2, 1, 2])
        with col_retry:
            if st.button("↩  Try Again", key="retry_capture", use_container_width=True):
                st.session_state.capture_done     = False
                st.session_state.capture_has_data = None
                goto("sniff_setup")


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 4 — ANALYZING  (runs model, then jumps to results)
# ═══════════════════════════════════════════════════════════════════════════
def screen_analyzing():
    st.markdown('<div class="page">', unsafe_allow_html=True)
    render_navbar(show_back=False)

    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .5rem;">
      <div style="font-size:2rem;margin-bottom:.5rem;">🧠</div>
      <div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;">Running the GCN Model…</div>
      <div style="font-size:.78rem;color:#475569;font-family:'IBM Plex Mono',monospace;margin-top:.3rem;">
        Loading weights · Building graph · Running inference
      </div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Classifying flows — this may take a moment…"):
        try:
            from cic_to_flowmeter import cic_to_pyflowmeter_columns

            if st.session_state.mode == "batch":
                uf = st.session_state.uploaded_file
                if uf is None:
                    st.error("No file found. Please go back and upload a CSV.")
                    goto("batch_setup")
                    return
                flows_df = pd.read_csv(uf)

            else:  # sniff
                csv_path = st.session_state.sniff_csv
                if not csv_path or not os.path.exists(csv_path):
                    st.error("No capture file found.")
                    goto("sniff_setup")
                    return
                flows_df = pd.read_csv(csv_path)
                if flows_df.empty:
                    st.session_state.capture_has_data = False
                    goto("capturing")
                    return

            flows_df = cic_to_pyflowmeter_columns(flows_df)
            result   = run_inference_on_df(flows_df)
            save_to_db(result)
            st.session_state.results = result
            goto("results")

        except Exception as e:
            st.error(f"❌  Inference failed: {e}")
            st.exception(e)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SCREEN 5 — RESULTS
# ═══════════════════════════════════════════════════════════════════════════
def screen_results():
    st.markdown('<div class="page">', unsafe_allow_html=True)
    render_navbar(back_label="← Start Over")

    res         = st.session_state.results
    if res is None:
        st.warning("No results to display. Please run the model first.")
        goto("home"); return

    class_names = res["class_names"]
    preds       = res["preds"]
    n_attacks   = int((preds > 0).sum())
    n_safe      = int((preds == 0).sum())
    mode_label  = "Live Detection" if st.session_state.mode == "sniff" else "Batch Test"

    st.markdown(f"""
    <div class="section-h">Results — {mode_label}</div>
    <div class="section-sub">{res['n_flows']:,} flows processed · Results saved to database</div>
    """, unsafe_allow_html=True)

    # ── Alert banner ──────────────────────────────────────────────────────
    if n_attacks > 0:
        from collections import Counter
        attack_types = [class_names[p] for p in preds if p > 0]
        breakdown    = "  ·  ".join(f"{l}: {c}" for l, c in Counter(attack_types).most_common(3))
        pct          = n_attacks / len(preds) * 100
        st.markdown(f"""
        <div class="banner-bad">
          <div class="banner-icon">🚨</div>
          <div>
            <div class="banner-title">Attack Traffic Detected</div>
            <div class="banner-body">{n_attacks:,} malicious flows ({pct:.1f}%) — {breakdown}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="banner-ok">
          <div class="banner-icon">✅</div>
          <div>
            <div class="banner-title">No Attacks Detected</div>
            <div class="banner-body">All {len(preds):,} flows classified as BENIGN</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Metric tiles ──────────────────────────────────────────────────────
    if "accuracy" in res:
        st.markdown(f"""
        <div class="metric-grid">
          <div class="m-tile g">
            <div class="m-label">Accuracy</div>
            <div class="m-value">{res['accuracy']:.3f}</div>
            <div class="m-sub">{res['n_flows']:,} flows</div>
          </div>
          <div class="m-tile b">
            <div class="m-label">Balanced Accuracy</div>
            <div class="m-value">{res['balanced_accuracy']:.3f}</div>
            <div class="m-sub">Class-balanced</div>
          </div>
          <div class="m-tile y">
            <div class="m-label">F1 Macro</div>
            <div class="m-value">{res['f1']:.3f}</div>
            <div class="m-sub">Macro average</div>
          </div>
          <div class="m-tile r">
            <div class="m-label">Log Loss</div>
            <div class="m-value">{res['log_loss']:.3f}</div>
            <div class="m-sub">Cross-entropy</div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Live mode — no ground truth, show flow counts
        st.markdown(f"""
        <div class="metric-grid">
          <div class="m-tile b">
            <div class="m-label">Total Flows</div>
            <div class="m-value">{len(preds):,}</div>
            <div class="m-sub">Captured</div>
          </div>
          <div class="m-tile {'r' if n_attacks > 0 else 'g'}">
            <div class="m-label">Malicious</div>
            <div class="m-value">{n_attacks:,}</div>
            <div class="m-sub">{'⚠ Attacks' if n_attacks > 0 else 'None found'}</div>
          </div>
          <div class="m-tile g">
            <div class="m-label">Benign</div>
            <div class="m-value">{n_safe:,}</div>
            <div class="m-sub">Normal traffic</div>
          </div>
          <div class="m-tile y">
            <div class="m-label">Attack Rate</div>
            <div class="m-value">{n_attacks/len(preds)*100:.1f}%</div>
            <div class="m-sub">Of traffic</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────
    has_curves = "roc_data" in res

    if has_curves:
        c1, c2, c3 = st.columns(3)
        cols_cm = c1; cols_roc = c2; cols_pr = c3
    else:
        c1, c2 = st.columns(2)
        cols_cm = c1; cols_bar2 = c2

    # Confusion matrix
    if "confusion_matrix" in res:
        with cols_cm:
            st.markdown('<div class="chart-card"><div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
            fig = plot_cm(res["confusion_matrix"], class_names)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    if has_curves:
        with cols_roc:
            st.markdown('<div class="chart-card"><div class="chart-title">ROC Curves</div>', unsafe_allow_html=True)
            fig = plot_roc(res["roc_data"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols_pr:
            st.markdown('<div class="chart-card"><div class="chart-title">Precision-Recall Curves</div>', unsafe_allow_html=True)
            fig = plot_pr(res["pr_data"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Distribution + per-class table ────────────────────────────────────
    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown('<div class="chart-card"><div class="chart-title">Prediction Distribution</div>', unsafe_allow_html=True)
        fig = plot_bar(preds, class_names)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with cb2:
        st.markdown('<div class="chart-card"><div class="chart-title">Per-Class Summary</div>', unsafe_allow_html=True)
        counts = np.bincount(preds, minlength=len(class_names))
        tdf = pd.DataFrame({
            "Class":      class_names,
            "Flows":      counts,
            "% of Total": (counts / counts.sum() * 100).round(2),
            "Status":     ["⚠ Attack" if i > 0 else "✓ Benign" for i in range(len(class_names))],
        })
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DB History ────────────────────────────────────────────────────────
    st.markdown('<div class="db-section">', unsafe_allow_html=True)
    col_dbtitle, col_dbrefresh = st.columns([5, 1])
    with col_dbtitle:
        st.markdown('<div class="chart-title" style="margin-bottom:.75rem;">Database — Recent Inference History</div>', unsafe_allow_html=True)
    with col_dbrefresh:
        if st.button("↻  Refresh", key="db_refresh", use_container_width=True):
            st.rerun()

    hist = load_db_history(100)
    if hist.empty:
        st.info("No records in the database yet.")
    else:
        def _lc(v):
            return "color: #22c55e" if v == "BENIGN" else "color: #ef4444"
        styled = hist.style.applymap(_lc, subset=["predicted_label"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=300)
        st.caption(f"{len(hist)} most recent records · data/idps.db")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_MAP = {
    "home":        screen_home,
    "batch_setup": screen_batch_setup,
    "sniff_setup": screen_sniff_setup,
    "capturing":   screen_capturing,
    "analyzing":   screen_analyzing,
    "results":     screen_results,
}

SCREEN_MAP.get(st.session_state.screen, screen_home)()