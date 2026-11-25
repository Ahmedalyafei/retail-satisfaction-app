"""
Retail Customer Satisfaction Prediction - advanced Streamlit app.

Run locally:   streamlit run app.py
Needs (created by train_model.py):
    satisfaction_model.pkl, model_metrics.json, test_predictions.csv, sample_data.csv
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Satisfaction Predictor", page_icon="🛍️", layout="wide")

NAVY, ORANGE, SOFT, GREEN, GREY = "#14213D", "#C8431F", "#F4A261", "#2A9D8F", "#8D99AE"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
NUM_COLS = ["Age", "Total_Purchases", "Amount", "Total_Amount"]
CAT_COLS = ["Gender", "Income", "Customer_Segment", "Country", "Month",
            "Product_Category", "Shipping_Method", "Payment_Method", "Order_Status"]
FEATURES = NUM_COLS + CAT_COLS
ACTIONABLE = ["Shipping_Method", "Payment_Method", "Order_Status"]


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_model():
    return joblib.load("satisfaction_model.pkl")


@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_sample():
    return pd.read_csv("sample_data.csv")


@st.cache_data
def load_test_predictions():
    return pd.read_csv("test_predictions.csv")


model = load_model()
metrics = load_metrics()
OPTIONS = metrics["options"]
OPTIONS["Month"] = [m for m in MONTHS if m in OPTIONS["Month"]]
LOW_CUT, HIGH_CUT = metrics["proba_quantiles"]["0.33"], metrics["proba_quantiles"]["0.66"]

if "history" not in st.session_state:
    st.session_state.history = []


# ------------------------------------------------------------------ helpers
def risk_level(p):
    if p < LOW_CUT:
        return "Low"
    if p < HIGH_CUT:
        return "Medium"
    return "High"


RISK_COLORS = {"Low": GREEN, "Medium": SOFT, "High": ORANGE}


def predict_proba(df):
    return model.predict_proba(df[FEATURES])[:, 1]


def gauge(p, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probability of dissatisfaction"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": NAVY},
            "steps": [
                {"range": [0, LOW_CUT * 100], "color": "#D8EFEC"},
                {"range": [LOW_CUT * 100, HIGH_CUT * 100], "color": "#FCE9D6"},
                {"range": [HIGH_CUT * 100, 100], "color": "#F6D3C8"},
            ],
            "threshold": {"line": {"color": ORANGE, "width": 4}, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def what_if(row):
    """Try every option of each actionable feature and report how the risk changes."""
    base = predict_proba(row)[0]
    trials = []
    for col in ACTIONABLE:
        for opt in OPTIONS[col]:
            r = row.copy()
            r[col] = opt
            trials.append((col, opt, r))
    probs = predict_proba(pd.concat([t[2] for t in trials], ignore_index=True))
    out = pd.DataFrame({"Feature": [t[0] for t in trials], "Option": [t[1] for t in trials],
                        "Probability": probs})
    out["Change"] = out["Probability"] - base
    out["Current"] = [row.iloc[0][c] == o for c, o in zip(out["Feature"], out["Option"])]
    return base, out


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛍️ Satisfaction Predictor")
threshold = st.sidebar.slider(
    "Decision threshold", 0.30, 0.70, 0.50, 0.01,
    help="A customer is flagged as dissatisfied when the probability is at or above this value. "
         "Lower it to catch more unhappy customers (higher recall), raise it for fewer false alarms.")
st.sidebar.caption(f"Risk bands: Low < {LOW_CUT:.0%} ≤ Medium < {HIGH_CUT:.0%} ≤ High")


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("Retail Customer Satisfaction Prediction")
    st.write("Predict which customers are likely to be **dissatisfied** with a purchase, "
             "so the business can step in before they leave.")

    s = metrics["test_scores"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions used", f"{metrics['n_train'] + metrics['n_test']:,}")
    c2.metric("Dissatisfied rate", f"{metrics['target_rate']:.0%}")
    c3.metric("Recall (dissatisfied)", f"{s['recall']:.0%}")
    c4.metric("ROC-AUC", f"{s['roc_auc']:.2f}")

    st.subheader("What you can do in this app")
    a, b, c = st.columns(3)
    with a:
        st.markdown("##### 🔮 Predict\nScore one customer, see the risk gauge and get "
                    "suggestions that would lower the risk.")
    with b:
        st.markdown("##### 📊 Data explorer\nFilter the data and see where dissatisfaction is highest.")
        st.markdown("##### 📈 Model performance\nConfusion matrix, ROC curve, feature importance and "
                    "a live threshold tool.")
    with c:
        st.markdown("##### 🕘 History\nEvery prediction you make in this session, ready to download.")
        st.markdown("##### ⚙️ Threshold\nUse the sidebar slider to trade recall against false alarms.")

    with st.expander("How the project followed CRISP-DM"):
        st.markdown(
            "1. **Business understanding** – catch unhappy customers early; main metric is recall.\n"
            "2. **Data understanding** – 302K transactions, 30 columns; `Ratings` leaks the target.\n"
            "3. **Data preparation** – 13 features, median / most-frequent imputation, one-hot encoding, scaling.\n"
            "4. **Modeling** – Logistic Regression, KNN, Decision Tree, Random Forest with balanced class weights.\n"
            "5. **Evaluation** – Random Forest wins (macro F1 0.63, recall 0.70).\n"
            "6. **Deployment** – this Streamlit app.")


def page_explorer():
    st.title("📊 Data Explorer")
    df = load_sample()
    st.caption(f"Based on a random sample of {len(df):,} transactions.")

    f1, f2, f3 = st.columns(3)
    countries = f1.multiselect("Country", OPTIONS["Country"], default=OPTIONS["Country"])
    segments = f2.multiselect("Customer segment", OPTIONS["Customer_Segment"], default=OPTIONS["Customer_Segment"])
    cats = f3.multiselect("Product category", OPTIONS["Product_Category"], default=OPTIONS["Product_Category"])
    d = df[df["Country"].isin(countries) & df["Customer_Segment"].isin(segments) & df["Product_Category"].isin(cats)]
    if d.empty:
        st.warning("No rows match these filters.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{len(d):,}")
    k2.metric("Dissatisfied rate", f"{d['Dissatisfied'].mean():.1%}",
              f"{(d['Dissatisfied'].mean() - df['Dissatisfied'].mean()) * 100:+.1f} pts vs all", delta_color="inverse")
    k3.metric("Average order value", f"{d['Total_Amount'].mean():,.0f}")
    k4.metric("Average age", f"{d['Age'].mean():.0f}")

    left, right = st.columns(2)
    with left:
        dim = st.selectbox("Dissatisfaction rate by", CAT_COLS, index=CAT_COLS.index("Customer_Segment"))
        g = d.groupby(dim)["Dissatisfied"].mean().reset_index()
        if dim == "Month":
            g[dim] = pd.Categorical(g[dim], MONTHS, ordered=True)
            g = g.sort_values(dim)
        else:
            g = g.sort_values("Dissatisfied", ascending=False)
        fig = px.bar(g, x=dim, y="Dissatisfied", text_auto=".0%", color_discrete_sequence=[ORANGE])
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Dissatisfied rate", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Feedback breakdown**")
        fb = d["Feedback"].value_counts().reindex(["Excellent", "Good", "Average", "Bad"]).reset_index()
        fb.columns = ["Feedback", "Count"]
        fig = px.pie(fb, names="Feedback", values="Count", hole=0.5,
                     color="Feedback", color_discrete_map={"Excellent": NAVY, "Good": GREY,
                                                           "Average": SOFT, "Bad": ORANGE})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        lab = d.assign(Status=d["Dissatisfied"].map({0: "Satisfied", 1: "Dissatisfied"}))
        fig = px.histogram(lab, x="Age", color="Status", barmode="overlay", nbins=30,
                           color_discrete_map={"Satisfied": NAVY, "Dissatisfied": ORANGE},
                           title="Age distribution")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(lab, x="Status", y="Total_Amount", color="Status",
                     color_discrete_map={"Satisfied": NAVY, "Dissatisfied": ORANGE},
                     title="Order value by satisfaction")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show the filtered data"):
        st.dataframe(d, use_container_width=True, height=300)


def page_predict():
    st.title("🔮 Predict a customer")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Customer**")
            age = st.slider("Age", 18, 70, 30)
            gender = st.selectbox("Gender", OPTIONS["Gender"])
            income = st.selectbox("Income", OPTIONS["Income"], index=OPTIONS["Income"].index("Medium"))
            segment = st.selectbox("Customer segment", OPTIONS["Customer_Segment"])
        with c2:
            st.markdown("**Order**")
            country = st.selectbox("Country", OPTIONS["Country"])
            month = st.selectbox("Month", OPTIONS["Month"])
            category = st.selectbox("Product category", OPTIONS["Product_Category"])
            qty = st.number_input("Number of items", 1, 10, 3)
            price = st.number_input("Price per item", 10.0, 500.0, 250.0, step=10.0)
        with c3:
            st.markdown("**Service**")
            shipping = st.selectbox("Shipping method", OPTIONS["Shipping_Method"])
            payment = st.selectbox("Payment method", OPTIONS["Payment_Method"])
            status = st.selectbox("Order status", OPTIONS["Order_Status"])
            st.info(f"Total amount: **{qty * price:,.2f}**")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if not submitted and "last_row" not in st.session_state:
        st.caption("Fill in the details and click **Predict**.")
        return

    if submitted:
        row = pd.DataFrame([{
            "Age": age, "Gender": gender, "Income": income, "Customer_Segment": segment,
            "Country": country, "Month": month, "Total_Purchases": qty, "Amount": price,
            "Total_Amount": qty * price, "Product_Category": category,
            "Shipping_Method": shipping, "Payment_Method": payment, "Order_Status": status,
        }])
        st.session_state.last_row = row
        p = float(predict_proba(row)[0])
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"), **row.iloc[0].to_dict(),
            "Probability": round(p, 3), "Risk": risk_level(p),
            "Prediction": "Dissatisfied" if p >= threshold else "Satisfied",
        })

    row = st.session_state.last_row
    p, trials = what_if(row)
    risk = risk_level(p)

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(p, threshold), use_container_width=True)
    with right:
        st.markdown("### Result")
        if p >= threshold:
            st.error(f"**Likely DISSATISFIED** – probability {p:.0%} (threshold {threshold:.0%})")
        else:
            st.success(f"**Likely SATISFIED** – probability {p:.0%} (threshold {threshold:.0%})")
        st.markdown(f"Risk level: <span style='color:{RISK_COLORS[risk]};font-weight:700;font-size:1.3em'>"
                    f"{risk}</span>", unsafe_allow_html=True)
        if risk == "High":
            st.markdown("**Suggested action:** contact the customer after delivery and offer a discount "
                        "or priority support.")
        elif risk == "Medium":
            st.markdown("**Suggested action:** send a follow-up email and monitor the order closely.")
        else:
            st.markdown("**Suggested action:** no action needed; a good candidate for a review request.")

    st.subheader("What would lower the risk?")
    st.caption("The model re-scores this customer with each option the business controls.")
    best = (trials[~trials["Current"]].sort_values("Change").groupby("Feature").head(1))
    best = best[best["Change"] < -0.005]
    if best.empty:
        st.write("The current shipping, payment and order status are already the lowest-risk options.")
    else:
        for _, r in best.iterrows():
            st.markdown(f"- Change **{r['Feature'].replace('_', ' ').lower()}** to **{r['Option']}** → "
                        f"risk {p:.0%} → **{r['Probability']:.0%}** ({r['Change'] * 100:+.1f} pts)")

    feat = st.selectbox("Compare options for", ACTIONABLE, format_func=lambda c: c.replace("_", " "))
    t = trials[trials["Feature"] == feat].copy()
    t["Label"] = np.where(t["Current"], t["Option"] + " (current)", t["Option"])
    fig = px.bar(t.sort_values("Probability"), x="Probability", y="Label", orientation="h",
                 text_auto=".0%", color="Current", color_discrete_map={True: NAVY, False: GREY})
    fig.add_vline(x=threshold, line_dash="dash", line_color=ORANGE, annotation_text="threshold")
    fig.update_layout(height=280, showlegend=False, xaxis_tickformat=".0%", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def page_performance():
    st.title("📈 Model performance")
    st.caption(f"{metrics['final_model']} · evaluated on {metrics['n_test']:,} held-out transactions")

    tp = load_test_predictions()
    y, pr = tp["y_true"].values, tp["proba"].values
    pred = (pr >= threshold).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    recall = TP / (TP + FN); precision = TP / (TP + FP) if TP + FP else 0
    acc = (TP + TN) / len(y)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    st.markdown(f"#### At threshold **{threshold:.2f}** (change it in the sidebar)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall", f"{recall:.1%}", help="Share of dissatisfied customers the model catches")
    m2.metric("Precision", f"{precision:.1%}", help="Share of flagged customers who really are dissatisfied")
    m3.metric("Accuracy", f"{acc:.1%}")
    m4.metric("F1 (dissatisfied)", f"{f1:.2f}")

    left, right = st.columns(2)
    with left:
        cm = np.array([[TN, FP], [FN, TP]])
        fig = px.imshow(cm, text_auto=",", color_continuous_scale=[[0, "#EFECE4"], [1, NAVY]],
                        x=["Predicted satisfied", "Predicted dissatisfied"],
                        y=["Actually satisfied", "Actually dissatisfied"], title="Confusion matrix")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=ORANGE, width=3),
                                 name=f"Random Forest (AUC {metrics['test_scores']['roc_auc']:.2f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"),
                                 name="Random guess"))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate",
                          height=400, legend=dict(x=0.45, y=0.08))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fi = pd.Series(metrics["feature_importance"]).sort_values()
        fig = px.bar(x=fi.values, y=fi.index.str.replace("_", " "), orientation="h", title="Feature importance",
                     color_discrete_sequence=[NAVY])
        fig.update_layout(height=450, xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        ts = np.round(np.arange(0.30, 0.71, 0.02), 2)
        rows = []
        for t in ts:
            pt = pr >= t
            tp_ = (pt & (y == 1)).sum()
            rows.append({"Threshold": t, "Recall": tp_ / (y == 1).sum(),
                         "Precision": tp_ / pt.sum() if pt.sum() else np.nan})
        tr = pd.DataFrame(rows).melt("Threshold", var_name="Metric", value_name="Value")
        fig = px.line(tr, x="Threshold", y="Value", color="Metric", title="Recall vs precision by threshold",
                      color_discrete_map={"Recall": ORANGE, "Precision": NAVY})
        fig.add_vline(x=threshold, line_dash="dash", line_color=GREY)
        fig.update_layout(height=450, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model comparison (threshold 0.50)")
    comp = pd.DataFrame(metrics["comparison"]).T
    comp.columns = ["Accuracy", "Precision", "Recall", "Macro F1", "ROC-AUC"]
    st.dataframe(comp.style.format("{:.3f}").highlight_max(axis=0, color="#F9E3DA"), use_container_width=True)


def page_history():
    st.title("🕘 Prediction history")
    h = st.session_state.history
    if not h:
        st.info("No predictions yet. Go to **Predict** and score a customer.")
        return
    df = pd.DataFrame(h)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predictions", len(df))
    c2.metric("Flagged dissatisfied", int((df["Prediction"] == "Dissatisfied").sum()))
    c3.metric("Average probability", f"{df['Probability'].mean():.0%}")
    st.dataframe(df.iloc[::-1], use_container_width=True)
    a, b = st.columns(2)
    a.download_button("Download history CSV", df.to_csv(index=False), "prediction_history.csv", "text/csv")
    if b.button("Clear history"):
        st.session_state.history = []
        st.rerun()


pages = [
    st.Page(page_overview, title="Overview", icon="🏠", default=True, url_path="overview"),
    st.Page(page_explorer, title="Data explorer", icon="📊", url_path="explorer"),
    st.Page(page_predict, title="Predict", icon="🔮", url_path="predict"),
    st.Page(page_performance, title="Model performance", icon="📈", url_path="performance"),
    st.Page(page_history, title="History", icon="🕘", url_path="history"),
]
st.navigation(pages).run()
