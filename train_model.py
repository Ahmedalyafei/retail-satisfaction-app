"""
Train the customer satisfaction model and save everything the app needs.

Run once from the project folder:
    python train_model.py

It needs new_retail_data.csv in the same folder, and creates:
    satisfaction_model.pkl   - the full pipeline (preprocessing + Random Forest)
    model_metrics.json       - scores, feature importance, model comparison
    test_predictions.csv     - true labels + predicted probabilities (for the threshold tool)
    sample_data.csv          - a 20,000-row sample for the Data Explorer page
    requirements.txt         - the exact library versions you trained with
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = "new_retail_data.csv"
NUM_COLS = ["Age", "Total_Purchases", "Amount", "Total_Amount"]
CAT_COLS = ["Gender", "Income", "Customer_Segment", "Country", "Month",
            "Product_Category", "Shipping_Method", "Payment_Method", "Order_Status"]
FEATURES = NUM_COLS + CAT_COLS


def load_data():
    print(f"Loading {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["Feedback"])
    df["Dissatisfied"] = df["Feedback"].isin(["Bad", "Average"]).astype(int)
    return df


def build_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, NUM_COLS), ("cat", cat_pipe, CAT_COLS)])


def scores(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def grouped_importance(pipeline):
    """Add up one-hot importances so each original column gets one score."""
    pre = pipeline.named_steps["preprocessor"]
    names = pre.get_feature_names_out()
    imps = pipeline.named_steps["rf"].feature_importances_
    totals = {c: 0.0 for c in FEATURES}
    for name, imp in zip(names, imps):
        raw = name.split("__", 1)[1]
        for col in FEATURES:
            if raw == col or raw.startswith(col + "_"):
                totals[col] += float(imp)
                break
    return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))


def main():
    df = load_data()
    X, y = df[FEATURES], df["Dissatisfied"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # quick comparison of simpler models (on a 60k-row sample to keep it fast)
    comparison = {}
    small = X_train.sample(60000, random_state=42)
    small_y = y_train.loc[small.index]
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    }
    for name, clf in candidates.items():
        print(f"Training {name} ...")
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("clf", clf)]).fit(small, small_y)
        proba = pipe.predict_proba(X_test)[:, 1]
        comparison[name] = scores(y_test, (proba >= 0.5).astype(int), proba)

    # final model
    print("Training Random Forest (final model) ...")
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                      random_state=42, n_jobs=-1)),
    ]).fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    comparison["Random Forest"] = scores(y_test, pred, proba)
    print(classification_report(y_test, pred))

    fpr, tpr, _ = roc_curve(y_test, proba)
    step = max(1, len(fpr) // 300)
    metrics = {
        "final_model": "Random Forest (100 trees, max depth 12)",
        "test_scores": comparison["Random Forest"],
        "comparison": comparison,
        "feature_importance": grouped_importance(model),
        "roc_curve": {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_rate": round(float(y.mean()), 4),
        "options": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS},
        "ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in NUM_COLS},
        "proba_quantiles": {q: round(float(np.quantile(proba, q)), 4) for q in [0.33, 0.66]},
    }

    joblib.dump(model, "satisfaction_model.pkl", compress=9)
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"y_true": y_test.values, "proba": proba.round(4)}).to_csv("test_predictions.csv", index=False)
    df[FEATURES + ["Feedback", "Dissatisfied"]].sample(20000, random_state=42).to_csv("sample_data.csv", index=False)

    import streamlit, plotly
    with open("requirements.txt", "w") as f:
        f.write(f"streamlit=={streamlit.__version__}\n")
        f.write(f"scikit-learn=={sklearn.__version__}\n")
        f.write(f"pandas=={pd.__version__}\n")
        f.write(f"numpy=={np.__version__}\n")
        f.write(f"plotly=={plotly.__version__}\n")
        f.write("joblib\n")

    print("\nDone. Files saved: satisfaction_model.pkl, model_metrics.json, "
          "test_predictions.csv, sample_data.csv, requirements.txt")
    print(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__}")


if __name__ == "__main__":
    main()
