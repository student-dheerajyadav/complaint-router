import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Complaint Router", layout="wide")

INK, TEAL, SLATE, LINE = "#0B1F3A", "#0E7C86", "#5B6B82", "#D9E1EC"
DISPLAY = {"credit_reporting": "Credit reporting", "debt_collection": "Debt collection",
           "mortgages_and_loans": "Mortgages and loans", "credit_card": "Credit card",
           "retail_banking": "Retail banking"}
DEPT = {"Credit reporting": "Credit Bureau Disputes", "Debt collection": "Collections Compliance",
        "Mortgages and loans": "Loans and Mortgages", "Credit card": "Cards Operations",
        "Retail banking": "Branch and Retail Banking"}
EXAMPLES = {"Debt collection": "A collector keeps calling me about a debt I already paid and refuses to stop.",
            "Credit report error": "There is a late payment on my credit report that is wrong. Please remove it.",
            "Account closure": "My bank closed my checking account without any notice and I cannot access my money."}


def pretty(c):
    return DISPLAY.get(c, str(c).replace("_", " ").capitalize())


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 4.5rem; max-width: 1200px; }
.topbar { border-bottom: 3px solid #0E7C86; padding-bottom: 12px; margin-bottom: 18px; }
.topbar h1 { margin: 0; font-size: 1.7rem; font-weight: 600; color: #0B1F3A; }
.topbar p { margin: 4px 0 0; color: #5B6B82; }
.kpi { border: 1px solid #D9E1EC; border-radius: 6px; padding: 12px 16px; background: #fff; }
.kpi { color: #0B1F3A; }
.kpi .v { font-size: 1.5rem; font-weight: 600; color: #0B1F3A; }
.kpi .l { font-size: .85rem; color: #5B6B82; }
.result, .result b { color: #0B1F3A; }
.result { border: 1px solid #D9E1EC; border-left: 6px solid #0E7C86; border-radius: 6px; padding: 16px 20px; background: #fff; }
.result .l { font-size: .85rem; color: #5B6B82; }
.result .cat { font-size: 1.6rem; font-weight: 600; color: #0B1F3A; }
.meter { height: 8px; background: #E6ECF3; border-radius: 4px; margin: 10px 0 4px; }
.meter span { display: block; height: 8px; border-radius: 4px; background: #0E7C86; }
.chip { display: inline-block; border: 1px solid #0E7C86; color: #0E7C86; font-weight: 500;
        padding: 2px 10px; border-radius: 14px; margin: 3px 6px 3px 0; font-size: .9rem; }
.stTabs [data-baseweb="tab"] { font-weight: 500; }
</style>
<div class="topbar"><h1>Complaint Router</h1>
<p>Classifies banking complaints and routes each one to the responsible department.</p></div>
""", unsafe_allow_html=True)


def style(fig, h=320):
    fig.update_layout(height=h, font_family="IBM Plex Sans", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=10, t=30, b=0))
    return fig


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


# ---------------- Data ----------------
with st.sidebar:
    st.subheader("Dataset")
    up = st.file_uploader("Upload a labelled CSV (optional)", type="csv")
    if up:
        raw = pd.read_csv(up)
        tcol = st.selectbox("Text column", raw.columns)
        ccol = st.selectbox("Category column", raw.columns, index=min(1, len(raw.columns) - 1))
        df = raw[[tcol, ccol]].dropna().rename(columns={tcol: "text", ccol: "category"})
    elif os.path.exists("clean_complaints.csv"):
        df = pd.read_csv("clean_complaints.csv").dropna().rename(columns={"narrative": "text", "product": "category"})
    else:
        st.error("No dataset found. Upload a CSV with a text column and a category column.")
        st.stop()
    df["text"], df["category"] = df["text"].astype(str), df["category"].astype(str)
    st.divider()
    st.subheader("Settings")
    thr = st.slider("Manual review threshold", 0.3, 0.9, 0.6, 0.05,
                    help="Predictions with confidence below this value are flagged for human review.")


@st.cache_resource
def train(data):
    Xtr, Xte, ytr, yte = train_test_split(data.text, data.category, test_size=0.2, random_state=42,
                                          stratify=data.category)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
              "Naive Bayes": MultinomialNB()}
    labels = sorted(data.category.unique())
    rows, cms, reps = [], {}, {}
    for name, m in models.items():
        m.fit(A, ytr)
        p = m.predict(B)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, p), "Macro F1": f1_score(yte, p, average="macro")})
        cms[name] = confusion_matrix(yte, p, labels=labels)
        reps[name] = pd.DataFrame(classification_report(yte, p, labels=labels, output_dict=True, zero_division=0)).T.loc[labels]
    return vec, models, pd.DataFrame(rows), cms, reps, labels, len(yte)


with st.spinner("Training models on the dataset..."):
    vec, models, results, cms, reps, labels, n_test = train(df)
best = results.sort_values("Macro F1", ascending=False).iloc[0]

with st.sidebar:
    mname = st.radio("Model used for predictions", list(models), index=list(models).index(best["Model"]))
model = models[mname]

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Complaints in dataset", f"{len(df):,}")
kpi(c2, "Categories", df.category.nunique())
kpi(c3, "Best model (macro F1)", best["Model"])
kpi(c4, "Test accuracy", f"{results.set_index('Model').loc[mname, 'Accuracy']:.1%}")
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Classify", "Batch", "Model performance", "Keywords", "Dataset"])

# ---------------- Classify ----------------
with tab1:
    st.session_state.setdefault("text", "")
    st.caption("Load an example or write your own complaint.")
    ex = st.columns(len(EXAMPLES))
    for col, (k, v) in zip(ex, EXAMPLES.items()):
        col.button(k, on_click=lambda t=v: st.session_state.update(text=t), use_container_width=True)
    left, right = st.columns(2, gap="large")
    text = left.text_area("Complaint text", key="text", height=200, placeholder="Type or paste a customer complaint")
    with right:
        if text.strip():
            v = vec.transform([text])
            probs = model.predict_proba(v)[0]
            order = np.argsort(probs)[::-1]
            top, conf = model.classes_[order[0]], probs[order[0]]
            st.markdown(f"""<div class="result"><div class="l">Predicted category</div>
            <div class="cat">{pretty(top)}</div>
            <div>Route to <b>{DEPT.get(pretty(top), pretty(top))}</b></div>
            <div class="meter"><span style="width:{conf * 100:.0f}%"></span></div>
            <div class="l">Confidence {conf:.0%}</div></div>""", unsafe_allow_html=True)
            if conf < thr:
                st.warning(f"Confidence is below the {thr:.0%} review threshold. Send this complaint for manual review.")
        else:
            st.info("Enter a complaint to see the predicted category and department.")
    if text.strip():
        a, b = st.columns(2, gap="large")
        fig = go.Figure(go.Bar(x=probs[order][::-1], y=[pretty(c) for c in model.classes_[order][::-1]],
                               orientation="h", text=[f"{p:.0%}" for p in probs[order][::-1]], textposition="outside",
                               marker_color=[TEAL if i == len(order) - 1 else LINE for i in range(len(order))]))
        fig.update_xaxes(range=[0, 1.15], visible=False)
        a.markdown("**Probability by category**")
        a.plotly_chart(style(fig, 260), use_container_width=True)
        lr = models["Logistic Regression"]
        ci = list(lr.classes_).index(top)
        coef = lr.coef_[ci] if lr.coef_.shape[0] > 1 else lr.coef_[0] * (1 if ci == 1 else -1)
        contrib = np.asarray(v.multiply(coef).todense()).ravel()
        idx = [i for i in np.argsort(contrib)[::-1][:6] if contrib[i] > 0]
        b.markdown("**Key terms behind this prediction**")
        if idx:
            terms = vec.get_feature_names_out()
            b.markdown("".join(f'<span class="chip">{terms[i]}</span>' for i in idx), unsafe_allow_html=True)
        else:
            b.caption("No known terms from the training vocabulary were found in this text.")

# ---------------- Batch ----------------
with tab2:
    st.markdown("Classify many complaints at once. Upload a CSV, choose the text column, and download the results.")
    f = st.file_uploader("Complaints CSV", type="csv", key="batch")
    if f:
        bdf = pd.read_csv(f)
        col = st.selectbox("Column containing complaint text", bdf.columns, key="bcol")
        P = model.predict_proba(vec.transform(bdf[col].astype(str)))
        out = bdf.copy()
        out["predicted_category"] = [pretty(c) for c in model.classes_[P.argmax(1)]]
        out["route_to"] = out["predicted_category"].map(DEPT).fillna(out["predicted_category"])
        out["confidence"] = P.max(1).round(3)
        out["needs_review"] = out["confidence"] < thr
        m1, m2 = st.columns(2)
        kpi(m1, "Complaints classified", f"{len(out):,}")
        kpi(m2, "Flagged for manual review", f"{int(out.needs_review.sum()):,}")
        st.write("")
        st.dataframe(out, use_container_width=True, height=320)
        st.download_button("Download results (CSV)", out.to_csv(index=False), "classified_complaints.csv", "text/csv")

# ---------------- Performance ----------------
with tab3:
    st.caption(f"All scores are measured on {n_test:,} complaints held out from training (20% of the dataset).")
    a, b = st.columns(2, gap="large")
    long = results.melt("Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group", text_auto=".1%",
                 color_discrete_sequence=[TEAL, INK])
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    a.markdown("**Model comparison**")
    a.plotly_chart(style(fig), use_container_width=True)
    pick = b.radio("Model", list(models), horizontal=True, key="pick")
    norm = b.toggle("Show as percentage of each true category", value=True)
    cm = cms[pick]
    z = cm / cm.sum(1, keepdims=True) if norm else cm
    fig = px.imshow(z, x=[pretty(c) for c in labels], y=[pretty(c) for c in labels], text_auto=".0%" if norm else True,
                    color_continuous_scale=["#FFFFFF", TEAL], labels=dict(x="Predicted", y="Actual"))
    fig.update_coloraxes(showscale=False)
    b.plotly_chart(style(fig, 300), use_container_width=True)
    st.markdown(f"**Per-category results: {pick}**")
    rep = reps[pick][["precision", "recall", "f1-score", "support"]].copy()
    rep.index = [pretty(c) for c in rep.index]
    st.dataframe(rep.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "f1-score": "{:.1%}", "support": "{:.0f}"}),
                 use_container_width=True)

# ---------------- Keywords ----------------
with tab4:
    a, b = st.columns([1, 3], gap="large")
    cat = a.selectbox("Category", labels, format_func=pretty)
    n = a.slider("Number of keywords", 5, 25, 12)
    lr = models["Logistic Regression"]
    ci = list(lr.classes_).index(cat)
    coef = lr.coef_[ci] if lr.coef_.shape[0] > 1 else lr.coef_[0] * (1 if ci == 1 else -1)
    idx = np.argsort(coef)[::-1][:n][::-1]
    fig = px.bar(x=coef[idx], y=vec.get_feature_names_out()[idx], orientation="h",
                 labels=dict(x="Importance", y=""), color_discrete_sequence=[TEAL])
    b.markdown(f"**Most indicative terms: {pretty(cat)}**")
    b.plotly_chart(style(fig, 120 + 26 * n), use_container_width=True)

# ---------------- Dataset ----------------
with tab5:
    a, b = st.columns([1, 2], gap="large")
    counts = df.category.map(pretty).value_counts().reset_index()
    counts.columns = ["Category", "Complaints"]
    fig = px.bar(counts, x="Complaints", y="Category", orientation="h", color_discrete_sequence=[TEAL])
    fig.update_yaxes(autorange="reversed")
    a.markdown("**Complaints per category**")
    a.plotly_chart(style(fig, 280), use_container_width=True)
    with b:
        sel = st.multiselect("Filter by category", labels, default=labels, format_func=pretty)
        q = st.text_input("Search complaint text")
        view = df[df.category.isin(sel)]
        if q:
            view = view[view.text.str.contains(q, case=False, regex=False)]
        st.caption(f"{len(view):,} complaints match")
        st.dataframe(view.head(200).assign(category=view.head(200).category.map(pretty)), use_container_width=True, height=260)
