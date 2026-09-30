import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Complaint Router", page_icon="📨", layout="wide")

INK, TEAL, AMBER, MIST = "#14213D", "#0FA3B1", "#F5A623", "#EEF3F8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
.hero {{ background: linear-gradient(120deg, {INK} 0%, #1f3a6b 100%); color: white;
        padding: 28px 32px; border-radius: 18px; margin-bottom: 18px; }}
.hero h1 {{ margin: 0; font-weight: 800; font-size: 2rem; color: white; }}
.hero p {{ margin: 6px 0 0; opacity: .8; }}
.result {{ background: {MIST}; border-left: 8px solid {TEAL}; padding: 18px 22px; border-radius: 12px; }}
.result .cat {{ font-size: 1.8rem; font-weight: 800; color: {INK}; }}
.result .dept {{ color: #4a5a75; }}
.chip {{ display:inline-block; background:{AMBER}; color:{INK}; font-weight:600;
        padding:3px 12px; border-radius:20px; margin:3px 4px 3px 0; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
<div class="hero"><h1>📨 Complaint Router</h1>
<p>Complaint likho, aur model turant batayega kaunsi category aur department ko jani chahiye.</p></div>
""", unsafe_allow_html=True)

# ---------------- Data ----------------
SEED = {
    "Billing": ["I was charged twice for my subscription this month", "The invoice amount is higher than what was agreed",
                "My refund has not reached my account yet", "There is an unknown fee on my bill",
                "I was billed even after cancelling my plan", "The discount code was not applied at payment"],
    "Delivery": ["My order has not arrived and it is two weeks late", "The courier marked it delivered but I never got it",
                 "The package was delivered to the wrong address", "Tracking has not updated for many days",
                 "The delivery person was rude and left the parcel outside", "I received only part of my order"],
    "Product Quality": ["The product stopped working after two days", "The item arrived damaged and scratched",
                        "The quality is much worse than the photos", "The fabric tore the first time I wore it",
                        "The battery drains very fast and the device overheats", "I received a wrong size and a faulty piece"],
    "Technical Support": ["The app keeps crashing when I open it", "The website shows an error while checking out",
                          "I cannot connect the device to wifi even after reset", "The software update broke my settings",
                          "Support has not replied to my ticket for a week", "The chat bot keeps giving me wrong answers"],
    "Account Access": ["I cannot log in to my account", "The password reset email never arrives",
                       "My account got locked without any reason", "I did not receive the verification code",
                       "Someone else may have accessed my account", "I want to change my registered email but it fails"],
}
DEPT = {"Billing": "Finance Team", "Delivery": "Logistics Team", "Product Quality": "Quality Assurance",
        "Technical Support": "Tech Support", "Account Access": "Security & Accounts",
        "credit_reporting": "Credit Bureau Disputes Team", "debt_collection": "Collections Compliance Team",
        "mortgages_and_loans": "Loans & Mortgage Team", "credit_card": "Cards Team",
        "retail_banking": "Branch & Retail Banking Team"}
FILL = ["Please fix this soon.", "This is very frustrating.", "I have contacted you before.", "I need help urgently.", ""]


@st.cache_data
def sample_data():
    rnd = random.Random(7)
    rows = []
    for cat, sents in SEED.items():
        for _ in range(150):
            text = " and ".join(rnd.sample(sents, rnd.choice([1, 1, 2]))) + ". " + rnd.choice(FILL)
            rows.append((text.strip(), cat))
    return pd.DataFrame(rows, columns=["text", "category"]).sample(frac=1, random_state=1).reset_index(drop=True)


@st.cache_resource
def train(df):
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.category, test_size=0.2, random_state=42, stratify=df.category)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    A, B = vec.fit_transform(Xtr), vec.transform(Xte)
    models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"), "Naive Bayes": MultinomialNB()}
    rows, cms = [], {}
    labels = sorted(df.category.unique())
    for name, m in models.items():
        m.fit(A, ytr)
        p = m.predict(B)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, p), "F1 (macro)": f1_score(yte, p, average="macro")})
        cms[name] = confusion_matrix(yte, p, labels=labels)
    return vec, models, pd.DataFrame(rows), cms, labels


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Data")
    up = st.file_uploader("Apna complaints CSV upload karo", type="csv")
    if up:
        raw = pd.read_csv(up)
        tcol = st.selectbox("Text column", raw.columns)
        ccol = st.selectbox("Category column", raw.columns, index=min(1, len(raw.columns) - 1))
        df = raw[[tcol, ccol]].dropna().rename(columns={tcol: "text", ccol: "category"})
        df["text"], df["category"] = df["text"].astype(str), df["category"].astype(str)
    else:
        df = sample_data()
        st.info("Abhi sample data use ho raha hai. Real dataset milte hi CSV upload kar do.")
    st.metric("Total complaints", len(df))
    st.metric("Categories", df.category.nunique())

vec, models, results, cms, labels = train(df)
best = results.sort_values("Accuracy", ascending=False).iloc[0]["Model"]

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "⚖️ Model Compare", "🔤 Top Words", "📊 Data Explorer"])

# ---------------- Predict ----------------
with tab1:
    EXAMPLES = {"💳 Double charge": "I was charged twice for my subscription and need a refund urgently.",
                "📦 Late order": "My parcel is two weeks late and tracking has not updated.",
                "🔐 Login issue": "I cannot log in and the password reset email never arrives."}
    if "text" not in st.session_state:
        st.session_state.text = ""

    def setter(t):
        st.session_state.text = t

    st.caption("Quick examples try karo:")
    cols = st.columns(len(EXAMPLES))
    for c, (k, v) in zip(cols, EXAMPLES.items()):
        c.button(k, on_click=setter, args=(v,), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        mname = st.radio("Model chuno", list(models), index=list(models).index(best), horizontal=True)
        text = st.text_area("Complaint likho", key="text", height=170, placeholder="Yahan complaint paste karo...")
        go_btn = st.button("Category batao", type="primary", use_container_width=True)
    with right:
        if text.strip():
            m = models[mname]
            v = vec.transform([text])
            probs = m.predict_proba(v)[0]
            order = np.argsort(probs)[::-1]
            top = m.classes_[order[0]]
            st.markdown(f"""<div class="result"><div class="cat">{top}</div>
            <div class="dept">Bhejo: {DEPT.get(top, top)} &nbsp;|&nbsp; Confidence: <b>{probs[order[0]]:.0%}</b></div></div>""",
                        unsafe_allow_html=True)
            if probs[order[0]] < 0.5:
                st.warning("Confidence kam hai. Is complaint ko manual review ke liye bhejna better hoga.")
            fig = go.Figure(go.Bar(x=probs[order][::-1], y=m.classes_[order][::-1], orientation="h",
                                   marker_color=[TEAL if i == len(order) - 1 else "#b8c4d6" for i in range(len(order))],
                                   text=[f"{p:.0%}" for p in probs[order][::-1]], textposition="outside"))
            fig.update_layout(height=260, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 1.1], visible=False),
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            if mname == "Logistic Regression":
                idx = order[0]
                coef = m.coef_[idx] if m.coef_.shape[0] > 1 else m.coef_[0] * (1 if idx == 1 else -1)
                contrib = v.multiply(coef).toarray()[0]
                words = vec.get_feature_names_out()[np.argsort(contrib)[::-1][:6]]
                words = [w for w, c in zip(words, np.sort(contrib)[::-1][:6]) if c > 0]
                if words:
                    st.markdown("**Prediction ke peeche ke words:**")
                    st.markdown("".join(f'<span class="chip">{w}</span>' for w in words), unsafe_allow_html=True)
        else:
            st.info("👈 Complaint likho ya upar se example chuno.")

# ---------------- Compare ----------------
with tab2:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Scores")
        long = results.melt("Model", var_name="Metric", value_name="Score")
        fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=[TEAL, AMBER], text_auto=".2%")
        fig.update_layout(yaxis=dict(range=[0, 1.05]), height=340, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Best model: **{best}**")
    with c2:
        st.subheader("Confusion matrix")
        pick = st.radio("Model", list(models), horizontal=True, key="cm")
        fig = px.imshow(cms[pick], x=labels, y=labels, text_auto=True, color_continuous_scale=["#ffffff", TEAL],
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Sample data synthetic hai, isliye scores bahut high aayenge. Real dataset par realistic scores milenge.")

# ---------------- Top words ----------------
with tab3:
    a, b = st.columns([1, 2])
    cat = a.selectbox("Category", labels)
    n = a.slider("Kitne words", 5, 25, 12)
    lr = models["Logistic Regression"]
    ci = list(lr.classes_).index(cat)
    coef = lr.coef_[ci] if lr.coef_.shape[0] > 1 else lr.coef_[0] * (1 if ci == 1 else -1)
    feats = vec.get_feature_names_out()
    idx = np.argsort(coef)[::-1][:n]
    fig = px.bar(x=coef[idx][::-1], y=feats[idx][::-1], orientation="h", color=coef[idx][::-1],
                 color_continuous_scale=["#cfeff2", TEAL], labels=dict(x="Importance", y=""))
    fig.update_layout(height=120 + 26 * n, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)")
    b.plotly_chart(fig, use_container_width=True)

# ---------------- Explorer ----------------
with tab4:
    x, y = st.columns([1, 2])
    counts = df.category.value_counts().reset_index()
    counts.columns = ["category", "count"]
    x.plotly_chart(px.pie(counts, names="category", values="count", hole=0.55,
                          color_discrete_sequence=[TEAL, AMBER, INK, "#7aa6d6", "#b8c4d6", "#e07a5f"]).update_layout(height=340),
                   use_container_width=True)
    with y:
        f = st.multiselect("Category filter", labels, default=labels[:1])
        q = st.text_input("Text search")
        view = df[df.category.isin(f)] if f else df
        if q:
            view = view[view.text.str.contains(q, case=False)]
        st.dataframe(view.head(200), use_container_width=True, height=280)
