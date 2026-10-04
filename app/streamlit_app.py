"""Crop Suitability Classification System - Streamlit prototype.
Author: Harischandra Prasad Vissamsetti, M.C.A
Run from the project root:  streamlit run app/streamlit_app.py"""
import sys, json
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import FEATURES, VALID_RANGES, UNITS   # type: ignore

st.set_page_config(page_title="Crop Suitability Classification", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg,#14532d,#15803d); padding: 26px 32px; border-radius: 14px; color: #fff; margin-bottom: 22px;}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: 6px 0 0; opacity: .9; font-size: 1rem;}
.card {border: 1px solid #e5e7eb; border-radius: 14px; padding: 22px 26px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.05);}
.lbl {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #6b7280; margin-bottom: 4px;}
.crop {font-size: 2.3rem; font-weight: 700; color: #14532d; line-height: 1.15;}
.conf {font-size: 1.6rem; font-weight: 600; color: #111827;}
.badge {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-left: 8px;}
.b-high {background:#dcfce7; color:#166534;} .b-mod {background:#fef3c7; color:#92400e;} .b-low {background:#fee2e2; color:#991b1b;}
.row {display:flex; align-items:center; margin: 9px 0; font-size: .95rem;}
.row .n {width: 130px; color:#374151;} .row .bar {flex:1; background:#f3f4f6; border-radius:6px; height:12px; margin:0 12px;}
.row .fill {background:#16a34a; height:12px; border-radius:6px;} .row .p {width:52px; text-align:right; font-weight:600; color:#111827;}
.tbl {width:100%; border-collapse:collapse; font-size:.95rem;} .tbl th {text-align:left; background:#f3f4f6; padding:8px 12px; color:#374151;} .tbl td {padding:8px 12px; border-bottom:1px solid #e5e7eb;}
.foot {color:#6b7280; font-size:.8rem; margin-top: 26px; border-top: 1px solid #e5e7eb; padding-top: 12px;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load():
    return joblib.load(ROOT / "models" / "crop_model.joblib"), json.load(open(ROOT / "models" / "metadata.json"))
try:
    model, meta = load()
except FileNotFoundError:
    st.error("Trained model not found. Run `python src/train.py` from the project root, then restart this app.")
    st.stop()
med = meta["feature_medians"]

NAMES = {"kidneybeans": "Kidney Beans", "pigeonpeas": "Pigeon Peas", "mothbeans": "Moth Beans",
         "mungbean": "Mung Bean", "blackgram": "Black Gram"}
disp = lambda c: NAMES.get(c, c.title())

st.markdown(f"""<div class="hero"><h1>🌾 Crop Suitability Assessment System</h1>
<p>Data-driven crop recommendation based on soil nutrient and climatic conditions</p><p style="font-size:.85rem;opacity:.8;margin-top:10px; margin-left:18px" > Developed by Harischandra Prasad Vissamsetti, M.C.A</p></div>""",
            unsafe_allow_html=True)

# ---------------- Sidebar inputs ----------------
def field(f, label, step):
    lo, hi = VALID_RANGES[f]
    unit = f" ({UNITS[f]})" if UNITS[f] else ""
    rng = meta.get("feature_ranges", {}).get(f)
    hint = f"Range observed in training data: {rng[0]:g} to {rng[1]:g}" if rng else None
    return st.sidebar.number_input(f"{label}{unit}", float(lo), float(hi), float(med[f]), step, key=f, help=hint)

vals = {}
st.sidebar.subheader("Soil Characteristics")
vals["N"] = field("N", "Nitrogen", 1.0)
vals["P"] = field("P", "Phosphorus", 1.0)
vals["K"] = field("K", "Potassium", 1.0)
vals["ph"] = field("ph", "Soil pH", 0.1)
st.sidebar.subheader("Climatic Conditions")
vals["temperature"] = field("temperature", "Temperature", 0.1)
vals["humidity"] = field("humidity", "Relative Humidity", 1.0)
vals["rainfall"] = field("rainfall", "Rainfall", 1.0)
run = st.sidebar.button("Assess Crop Suitability", type="primary", use_container_width=True)

# ---------------- Results ----------------
if run:
    X = pd.DataFrame([vals], columns=FEATURES)
    st.session_state["result"] = (X, model.predict_proba(X)[0])

if "result" not in st.session_state:
    st.info("Provide the soil and climatic parameters in the sidebar, then select **Assess Crop Suitability** to generate a recommendation.")
else:
    X, proba = st.session_state["result"]
    classes = model.classes_
    top = np.argsort(proba)[::-1][:5]
    conf = float(proba[top[0]])
    # Ranking uses the SAME model probabilities as the headline, so it is always sorted and consistent
    rank = [(classes[k], float(proba[k]) * 100) for k in top]
    rank_label = "Crop Suitability Ranking · Top 5"
    if conf >= 0.85:
        level, cls, note = "High Confidence", "b-high", "Soil and climatic conditions are strongly aligned with the requirements of this crop."
    elif conf >= 0.60:
        level, cls, note = "Moderate Confidence", "b-mod", "Conditions are generally favourable for this crop; the alternatives listed may also be viable."
    else:
        level, cls, note = "Low Confidence", "b-low", "Conditions overlap with several crops. Review the alternatives and seek local agronomic guidance."

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.markdown(f"""<div class="card"><div class="lbl">Most Suitable Crop</div>
        <div class="crop">{disp(classes[top[0]])}</div><br>
        <div class="lbl">Prediction Confidence</div>
        <div class="conf">{conf:.1%}<span class="badge {cls}">{level}</span></div>
        <p style="color:#4b5563;margin-top:12px">{note}</p></div>""", unsafe_allow_html=True)
    with c2:
        rows = "".join(f"""<div class="row"><span class="n">{disp(c)}</span>
            <span class="bar"><div class="fill" style="width:{v:.1f}%"></div></span>
            <span class="p">{v:.1f}%</span></div>""" for c, v in rank)
        st.markdown(f"""<div class="card"><div class="lbl">{rank_label}</div>{rows}
        <p style="color:#6b7280;font-size:.8rem;margin:10px 0 0">Probability assigned to each crop by the classification model; values across all 22 crops sum to 100%.</p></div>""",
                    unsafe_allow_html=True)

    st.markdown("#### Input Parameters Summary")
    summ = pd.DataFrame({"Parameter": ["Nitrogen", "Phosphorus", "Potassium", "Soil pH", "Temperature", "Relative Humidity", "Rainfall"],
                         "Value": [X.at[0, f] for f in ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]],
                         "Unit": [UNITS[f] or "-" for f in ["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]})
    trs = "".join(f"<tr><td>{r.Parameter}</td><td>{r.Value:g}</td><td>{r.Unit}</td></tr>" for r in summ.itertuples())
    st.markdown(f"""<table class="tbl"><thead><tr><th>Parameter</th><th>Value</th><th>Unit</th></tr></thead><tbody>{trs}</tbody></table>""",
                unsafe_allow_html=True)

# ---------------- Model information (read from the trained artefacts) ----------------
mfile = ROOT / "reports" / "metrics.json"
if mfile.exists():
    try:
        m = json.load(open(mfile)); r = m["results"][m["selected"]]
        st.markdown("#### Model Performance (Hold-out Test Set)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{r['test_accuracy']:.1%}")
        k2.metric("Macro F1-Score", f"{r['test_f1']:.3f}")
        k3.metric("Macro ROC-AUC", f"{r['test_roc_auc']:.3f}")
        k4.metric("Crop Classes", meta.get("n_classes", len(meta["classes"])))
    except (KeyError, ValueError):
        pass

n_rec = f" · {meta['n_records']:,} training records" if meta.get("n_records") else ""
st.markdown(f"""<div class="foot">Model: {meta['selected_model']} · {len(meta['classes'])} crop classes{n_rec}<br>
This tool provides data-driven estimates intended to support, not replace, professional agronomic advice.<br> <center>© 2026 Harischandra Prasad Vissamsetti, M.C.A</center> </div>""",
            unsafe_allow_html=True)
