"""
Customer Retention Intelligence Assistant

Loads the churn dataset + trained model, explains predictions with SHAP,
suggests retention actions, and answers English questions.

Flow:
  question -> Ollama intent JSON (keyword fallback) -> Python calc -> answer

The LLM only classifies intent/entities. All numbers come from pandas/model/SHAP.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import shap

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Customer_Churn_Dataset.xlsx"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_retention_model.joblib"
DATA_SHEET = "01 Churn-Dataset"

HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT_SECONDS = 90

VALID_INTENTS = {
    "customer_prediction",
    "customer_explanation",
    "customer_retention",
    "customer_profile",
    "overall_churn",
    "global_drivers",
    "segment_analysis",
    "feature_analysis",
    "unsupported",
}

# Cache parsed intents within a process (same question asked again)
_PARSE_CACHE: dict[str, tuple[dict, str]] = {}

# Natural language -> dataset column
FEATURE_ALIASES = {
    "monthly charges": "MonthlyCharges",
    "monthly charge": "MonthlyCharges",
    "monthlycharges": "MonthlyCharges",
    "charges": "MonthlyCharges",
    "expensive plans": "MonthlyCharges",
    "price": "MonthlyCharges",
    "pricing": "MonthlyCharges",
    "total charges": "TotalCharges",
    "total charge": "TotalCharges",
    "totalcharges": "TotalCharges",
    "contract": "Contract",
    "contract type": "Contract",
    "tenure": "tenure",
    "tech support": "TechSupport",
    "technical support": "TechSupport",
    "techsupport": "TechSupport",
    "payment": "PaymentMethod",
    "payment method": "PaymentMethod",
    "paymentmethod": "PaymentMethod",
    "internet": "InternetService",
    "internet service": "InternetService",
    "internetservice": "InternetService",
    "technical tickets": "numTechTickets",
    "technical-support tickets": "numTechTickets",
    "technical support tickets": "numTechTickets",
    "tech tickets": "numTechTickets",
    "tech support tickets": "numTechTickets",
    "support tickets": "numTechTickets",
    "numtechtickets": "numTechTickets",
    "admin tickets": "numAdminTickets",
    "admin ticket": "numAdminTickets",
    "administrative tickets": "numAdminTickets",
    "administrative ticket": "numAdminTickets",
    "numadmintickets": "numAdminTickets",
    "online security": "OnlineSecurity",
    "paperless billing": "PaperlessBilling",
    "gender": "gender",
    "partner": "Partner",
    "dependents": "Dependents",
    "senior citizen": "SeniorCitizen",
    "phone service": "PhoneService",
    "multiple lines": "MultipleLines",
    "online backup": "OnlineBackup",
    "device protection": "DeviceProtection",
    "streaming tv": "StreamingTV",
    "streaming movies": "StreamingMovies",
}

CATEGORICAL_FEATURES = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
}

NUMERIC_FEATURES = {
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "numAdminTickets",
    "numTechTickets",
}

DEFAULT_SEGMENT_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
]


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
def load_resources(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Load dataset, trained pipeline, and precompute SHAP values."""
    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    df = pd.read_excel(data_path, sheet_name=DATA_SHEET)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    customer_ids = df["customerID"].astype(str).copy()
    df_model = df.drop(columns=["customerID"]).copy()
    df_model["Churn"] = df_model["Churn"].replace({"No": 0, "Yes": 1})

    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    X_processed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    masker = shap.maskers.Independent(X_processed, max_samples=X_processed.shape[0])
    explainer = shap.LinearExplainer(classifier, masker)
    shap_values = explainer(X_processed)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    global_importance = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
        .assign(readable_feature=lambda d: d["feature"].map(clean_feature_name))
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model": model,
        "df": df,
        "X": X,
        "y": y,
        "customer_ids": customer_ids,
        "feature_names": feature_names,
        "shap_values": shap_values,
        "global_importance": global_importance,
        "median_monthly_charges": float(df["MonthlyCharges"].median()),
    }


# ---------------------------------------------------------------------------
# Feature name helpers
# ---------------------------------------------------------------------------
def clean_feature_name(name: str) -> str:
    """Convert scikit-learn / one-hot feature names into readable labels."""
    name = str(name)
    if "__" in name:
        name = name.split("__", 1)[1]

    replacements = {
        "MonthlyCharges": "monthly charges",
        "TotalCharges": "total charges",
        "tenure": "tenure",
        "SeniorCitizen": "senior-citizen status",
        "numTechTickets": "number of technical-support tickets",
        "numAdminTickets": "number of administrative tickets",
        "PaperlessBilling": "paperless billing",
        "PaymentMethod": "payment method",
        "InternetService": "internet service",
        "OnlineSecurity": "online security",
        "OnlineBackup": "online backup",
        "DeviceProtection": "device protection",
        "TechSupport": "technical support",
        "StreamingTV": "streaming TV",
        "StreamingMovies": "streaming movies",
        "PhoneService": "phone service",
        "MultipleLines": "multiple lines",
        "Dependents": "dependents",
        "Partner": "partner status",
        "Contract": "contract type",
        "gender": "gender",
    }

    for raw, readable in replacements.items():
        prefix = raw + "_"
        if name.startswith(prefix):
            value = name[len(prefix) :].replace("_", " ")
            return f"{readable} = {value}"

    return replacements.get(name, name.replace("_", " "))


def resolve_feature_name(raw: str | None) -> str | None:
    """Map a natural-language or column name to an actual dataset column."""
    if raw is None:
        return None

    text = str(raw).strip()
    if not text:
        return None

    # Exact column match (case-insensitive)
    for col in CATEGORICAL_FEATURES | NUMERIC_FEATURES:
        if text.lower() == col.lower():
            return col

    key = re.sub(r"\s+", " ", text.lower().replace("-", " ").strip())
    if key in FEATURE_ALIASES:
        return FEATURE_ALIASES[key]

    # Partial alias match (longer keys first)
    for alias in sorted(FEATURE_ALIASES, key=len, reverse=True):
        alias_norm = alias.replace("-", " ")
        if alias_norm in key:
            return FEATURE_ALIASES[alias]

    return None


def extract_customer_id(question: str, resources: dict | None = None) -> str | None:
    """Pull a customer ID from free text."""
    match = re.search(r"\b[A-Za-z0-9]{4}-[A-Za-z0-9]{5}\b", question)
    if match:
        return match.group(0)

    if resources is not None:
        q_upper = question.upper()
        for cid in resources["customer_ids"]:
            if str(cid).upper() in q_upper:
                return str(cid)

    return None


def extract_feature_from_text(question: str) -> str | None:
    """Find a known feature mentioned in the question text."""
    q = re.sub(r"\s+", " ", question.lower().replace("-", " "))
    for alias in sorted(FEATURE_ALIASES, key=len, reverse=True):
        if alias.replace("-", " ") in q:
            return FEATURE_ALIASES[alias]
    return None


# ---------------------------------------------------------------------------
# Prediction / explanation / recommendations
# ---------------------------------------------------------------------------
def _risk_level(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def find_customer(resources: dict, customer_id: str) -> int | None:
    """Return row index for a customer ID (case-insensitive), or None."""
    matches = resources["customer_ids"][
        resources["customer_ids"].str.upper() == str(customer_id).upper()
    ].index
    if len(matches) == 0:
        return None
    return int(matches[0])


def predict_customer(resources: dict, customer_id: str) -> dict | None:
    """Return churn probability, predicted label, and risk level."""
    idx = find_customer(resources, customer_id)
    if idx is None:
        return None

    row = resources["X"].iloc[[idx]]
    model = resources["model"]
    probability = float(model.predict_proba(row)[0][1])
    prediction = int(model.predict(row)[0])

    return {
        "customer_id": resources["customer_ids"].iloc[idx],
        "index": idx,
        "churn_probability": probability,
        "predicted_churn": prediction,
        "risk_level": _risk_level(probability),
    }


def explain_customer(resources: dict, customer_id: str, top_n: int = 5) -> dict | None:
    """Explain one customer's prediction using SHAP (associations, not causation)."""
    prediction = predict_customer(resources, customer_id)
    if prediction is None:
        return None

    idx = prediction["index"]
    shap_df = pd.DataFrame(
        {
            "feature": resources["feature_names"],
            "shap_value": resources["shap_values"].values[idx],
        }
    )
    shap_df["readable_feature"] = shap_df["feature"].map(clean_feature_name)
    shap_df["absolute_impact"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("absolute_impact", ascending=False)

    risk_factors = shap_df[shap_df["shap_value"] > 0].head(top_n)
    protective_factors = shap_df[shap_df["shap_value"] < 0].head(top_n)

    return {
        **prediction,
        "risk_factors": risk_factors[["readable_feature", "shap_value"]].to_dict(
            "records"
        ),
        "protective_factors": protective_factors[
            ["readable_feature", "shap_value"]
        ].to_dict("records"),
    }


def explain_overall_churn(resources: dict, top_n: int = 8) -> dict:
    """Return overall churn rate and strongest global SHAP features."""
    top = resources["global_importance"].head(top_n)
    return {
        "churn_rate": float(resources["y"].mean()),
        "top_features": top[["readable_feature", "importance"]].to_dict("records"),
    }


def recommend_retention_actions(resources: dict, customer_id: str) -> list[str]:
    """Concise rule-based retention suggestions from actionable attributes."""
    idx = find_customer(resources, customer_id)
    if idx is None:
        return []

    c = resources["df"].iloc[idx]
    recommendations: list[str] = []

    if str(c.get("Contract", "")).lower() == "month-to-month":
        recommendations.append("Offer an incentive for a longer-term contract.")

    if float(c.get("tenure", 999)) <= 12:
        recommendations.append("Strengthen onboarding and early engagement.")

    if float(c.get("MonthlyCharges", 0)) > resources["median_monthly_charges"]:
        recommendations.append("Review the customer's current plan and pricing.")

    if str(c.get("TechSupport", "")).lower() == "no":
        recommendations.append("Provide proactive technical-support outreach.")

    if str(c.get("OnlineSecurity", "")).lower() == "no":
        recommendations.append("Highlight relevant security features.")

    if float(c.get("numTechTickets", 0)) >= 2:
        recommendations.append("Prioritize service-recovery outreach for open technical issues.")

    if float(c.get("numAdminTickets", 0)) >= 2:
        recommendations.append("Review billing or administrative friction.")

    if "electronic check" in str(c.get("PaymentMethod", "")).lower():
        recommendations.append("Encourage an automatic-payment option if appropriate.")

    if not recommendations:
        recommendations.append(
            "No clear rule-based action stood out; review the customer's risk drivers first."
        )

    return recommendations[:5]


def get_customer_profile(resources: dict, customer_id: str) -> dict | None:
    """Compact business profile for one customer."""
    prediction = predict_customer(resources, customer_id)
    if prediction is None:
        return None

    c = resources["df"].iloc[prediction["index"]]
    return {
        **prediction,
        "tenure": int(c.get("tenure", 0)),
        "contract": str(c.get("Contract", "")),
        "monthly_charges": float(c.get("MonthlyCharges", 0)),
        "payment_method": str(c.get("PaymentMethod", "")),
        "tech_support": str(c.get("TechSupport", "")),
        "internet_service": str(c.get("InternetService", "")),
    }


# ---------------------------------------------------------------------------
# Feature / segment analysis (pandas only)
# ---------------------------------------------------------------------------
def analyze_feature(resources: dict, feature: str) -> dict | None:
    """
    Calculate churn rate by category (categorical) or by bins (numeric).
    All statistics come from the dataset.
    """
    feature = resolve_feature_name(feature) or feature
    if feature not in resources["df"].columns:
        return None

    df = resources["df"]
    churn = (df["Churn"].astype(str).str.lower() == "yes").astype(int)

    if feature in CATEGORICAL_FEATURES or df[feature].dtype == object:
        grouped = (
            pd.DataFrame({"group": df[feature].astype(str), "churn": churn})
            .groupby("group", dropna=False)["churn"]
            .agg(churn_rate="mean", count="count")
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        if grouped.empty:
            return None

        return {
            "feature": feature,
            "kind": "categorical",
            "rows": grouped.to_dict("records"),
            "highest": {
                "group": grouped.iloc[0]["group"],
                "churn_rate": float(grouped.iloc[0]["churn_rate"]),
                "count": int(grouped.iloc[0]["count"]),
            },
            "lowest": {
                "group": grouped.iloc[-1]["group"],
                "churn_rate": float(grouped.iloc[-1]["churn_rate"]),
                "count": int(grouped.iloc[-1]["count"]),
            },
        }

    # Numeric: choose bins that stay readable for skewed ticket counts
    series = pd.to_numeric(df[feature], errors="coerce")

    if feature in {"numTechTickets", "numAdminTickets"}:
        # Discrete counts — group exact values so bins do not collapse under qcut
        grouped = (
            pd.DataFrame(
                {
                    "group": series.fillna(0).astype(int).astype(str) + " tickets",
                    "churn": churn,
                    "sort_key": series.fillna(0).astype(int),
                }
            )
            .groupby(["group", "sort_key"], dropna=False)["churn"]
            .agg(churn_rate="mean", count="count")
            .reset_index()
            .sort_values("sort_key")
            .drop(columns=["sort_key"])
        )
        # Keep the answer scannable if there are many rare high counts
        if len(grouped) > 8:
            top = grouped.head(7).copy()
            rest = grouped.iloc[7:]
            top = pd.concat(
                [
                    top,
                    pd.DataFrame(
                        [
                            {
                                "group": f"{int(rest['group'].str.extract(r'(\d+)')[0].astype(float).min())}+ tickets",
                                "churn_rate": (
                                    (rest["churn_rate"] * rest["count"]).sum()
                                    / rest["count"].sum()
                                ),
                                "count": int(rest["count"].sum()),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            grouped = top
    else:
        try:
            bins = pd.qcut(series, q=4, duplicates="drop")
        except ValueError:
            bins = pd.cut(series, bins=4, duplicates="drop")

        grouped = (
            pd.DataFrame({"group": bins.astype(str), "churn": churn, "value": series})
            .dropna(subset=["group"])
            .groupby("group", observed=False)["churn"]
            .agg(churn_rate="mean", count="count")
            .reset_index()
        )

        def _bin_sort_key(label: str) -> float:
            nums = re.findall(r"[-+]?\d*\.?\d+", label)
            return float(nums[0]) if nums else 0.0

        grouped = grouped.sort_values(
            "group", key=lambda s: s.map(_bin_sort_key)
        ).reset_index(drop=True)

    if grouped.empty:
        return None

    return {
        "feature": feature,
        "kind": "numeric",
        "rows": grouped.to_dict("records"),
        "highest": {
            "group": grouped.loc[grouped["churn_rate"].idxmax(), "group"],
            "churn_rate": float(grouped["churn_rate"].max()),
            "count": int(grouped.loc[grouped["churn_rate"].idxmax(), "count"]),
        },
        "lowest": {
            "group": grouped.loc[grouped["churn_rate"].idxmin(), "group"],
            "churn_rate": float(grouped["churn_rate"].min()),
            "count": int(grouped.loc[grouped["churn_rate"].idxmin(), "count"]),
        },
    }


# ---------------------------------------------------------------------------
# Answer formatters
# ---------------------------------------------------------------------------
def _unknown_customer(customer_id: str) -> str:
    return (
        f"Customer ID '{customer_id}' was not found in the dataset. "
        "Please check the ID and try again."
    )


def _parse_interval_bounds(label) -> tuple[float, float, bool, bool] | None:
    """
    Parse a pandas Interval or interval-like string.
    Returns (low, high, left_closed, right_closed) or None.
    """
    if hasattr(label, "left") and hasattr(label, "right"):
        left_closed = str(getattr(label, "closed", "right")) in {"left", "both"}
        right_closed = str(getattr(label, "closed", "right")) in {"right", "both"}
        return float(label.left), float(label.right), left_closed, right_closed

    text = str(label).strip()
    match = re.match(
        r"^[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]$",
        text,
    )
    if not match:
        return None

    low, high = float(match.group(1)), float(match.group(2))
    left_closed = text.startswith("[")
    right_closed = text.endswith("]")
    return low, high, left_closed, right_closed


def _format_number(value: float, decimals: int = 0) -> str:
    """Format a number without ugly trailing .0 when whole."""
    if decimals <= 0:
        return f"{int(round(value)):,}"
    text = f"{value:,.{decimals}f}"
    if decimals > 0 and "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _integer_bin_bounds(low: float, high: float, left_closed: bool, right_closed: bool) -> tuple[int, int]:
    """Convert continuous interval bounds into inclusive integer display bounds."""
    if left_closed:
        low_i = int(math.ceil(low - 1e-9))
    else:
        # (9.0, 29.0] -> 10 ... 29
        low_i = int(math.floor(low + 1e-9)) + 1

    if right_closed:
        high_i = int(math.floor(high + 1e-9))
    else:
        high_i = int(math.ceil(high - 1e-9)) - 1

    low_i = max(0, low_i)
    if high_i < low_i:
        high_i = low_i
    return low_i, high_i


def _format_bin_label(label, feature: str) -> str:
    """Turn pandas interval labels into readable business ranges."""
    text = str(label).strip()

    # Already-friendly discrete labels from ticket analysis
    if re.fullmatch(r"\d+\+?\s+tickets?", text, flags=re.IGNORECASE):
        nums = re.findall(r"\d+", text)
        if not nums:
            return text
        n = int(nums[0])
        if text.endswith("+ tickets") or "+" in text:
            return f"{n}+ tickets"
        if n == 1:
            return "1 ticket"
        return f"{n} tickets"

    parsed = _parse_interval_bounds(label)
    if parsed is None:
        return text

    low, high, left_closed, right_closed = parsed

    if feature == "tenure":
        low_i, high_i = _integer_bin_bounds(low, high, left_closed, right_closed)
        return f"{low_i}–{high_i} months"

    if feature in {"MonthlyCharges", "TotalCharges"}:
        # Keep small polish on left edge so qcut's -epsilon does not show as negative
        low_v = max(0.0, low)
        return f"${low_v:,.2f}–${high:,.2f}"

    if feature in {"numTechTickets", "numAdminTickets"}:
        low_i, high_i = _integer_bin_bounds(low, high, left_closed, right_closed)
        if low_i == high_i:
            return "1 ticket" if low_i == 1 else f"{low_i} tickets"
        return f"{low_i}–{high_i} tickets"

    low_v = max(0.0, low)
    return f"{_format_number(low_v, 2)}–{_format_number(high, 2)}"


def _numeric_trend_insight(analysis: dict) -> str:
    """
    Build one concise insight from the actual ordered bin churn rates.
    Detects rising, falling, or mixed relationships.
    """
    feature = analysis["feature"]
    rows = analysis["rows"]
    if len(rows) < 2:
        return "**Key insight:** Not enough groups to summarize a clear relationship."

    rates = [float(row["churn_rate"]) for row in rows]
    labels = [_format_bin_label(row["group"], feature) for row in rows]
    diffs = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
    n_up = sum(1 for d in diffs if d > 0.005)
    n_down = sum(1 for d in diffs if d < -0.005)
    overall = rates[-1] - rates[0]

    # Prefer the overall direction when most step changes agree
    rising = overall > 0.02 and n_up >= n_down
    falling = overall < -0.02 and n_down >= n_up

    first_label, last_label = labels[0], labels[-1]
    first_rate, last_rate = rates[0], rates[-1]
    high = analysis["highest"]
    high_label = _format_bin_label(high["group"], feature)

    if feature == "tenure":
        if falling:
            return (
                "**Key insight:** Churn is highest among newer customers and steadily "
                f"decreases with tenure. Customers with {first_label} have a "
                f"{first_rate:.1%} observed churn rate, compared with {last_rate:.1%} "
                f"among customers with {last_label}."
            )
        if rising:
            return (
                "**Key insight:** Observed churn increases with tenure in this dataset, "
                f"from {first_rate:.1%} in {first_label} to {last_rate:.1%} in {last_label}."
            )

    if feature == "MonthlyCharges":
        if rising:
            return (
                "**Key insight:** Higher monthly-charge groups show higher observed "
                f"churn than lower-charge groups ({first_rate:.1%} in {first_label} vs "
                f"{last_rate:.1%} in {last_label})."
            )
        if falling:
            return (
                "**Key insight:** Higher monthly-charge groups show lower observed "
                f"churn than lower-charge groups in this dataset."
            )

    if feature == "TotalCharges":
        if falling:
            return (
                "**Key insight:** Customers with lower total charges show higher "
                f"observed churn ({first_rate:.1%} in {first_label} vs {last_rate:.1%} "
                f"in {last_label})."
            )
        if rising:
            return (
                "**Key insight:** Observed churn rises with total charges in this dataset."
            )

    if feature in {"numTechTickets", "numAdminTickets"}:
        kind = (
            "technical-support"
            if feature == "numTechTickets"
            else "administrative"
        )
        if rising:
            return (
                f"**Key insight:** Customers with more {kind} tickets show higher "
                f"observed churn, peaking at {high['churn_rate']:.1%} for {high_label}."
            )
        if falling:
            return (
                f"**Key insight:** Groups with fewer {kind} tickets show higher "
                "observed churn in this dataset."
            )

    if rising:
        return (
            "**Key insight:** Observed churn generally increases across the higher "
            f"value groups, from {first_rate:.1%} to {last_rate:.1%}."
        )
    if falling:
        return (
            "**Key insight:** Observed churn generally decreases across the higher "
            f"value groups, from {first_rate:.1%} to {last_rate:.1%}."
        )

    return (
        "**Key insight:** The relationship is mixed across the observed groups."
    )


def _unsupported_message() -> str:
    return (
        "I couldn't determine the analysis you were asking for. "
        "Try asking about overall churn, a customer ID, a customer segment, "
        "or a feature such as tenure, contract type, monthly charges, or payment method."
    )


def _format_prediction(info: dict) -> str:
    return (
        f"### Customer {info['customer_id']}\n\n"
        f"**Churn probability:** {info['churn_probability']:.1%}\n\n"
        f"**Risk level:** {info['risk_level']}"
    )


def _format_explanation(info: dict) -> str:
    lines = [
        f"### Customer {info['customer_id']}",
        "",
        f"**Churn probability:** {info['churn_probability']:.1%}",
        f"**Risk level:** {info['risk_level']}",
    ]
    if info["risk_factors"]:
        lines.extend(["", "#### Main churn drivers"])
        for item in info["risk_factors"]:
            lines.append(f"- {item['readable_feature']}")
    if info["protective_factors"]:
        lines.extend(["", "#### Factors supporting retention"])
        for item in info["protective_factors"]:
            lines.append(f"- {item['readable_feature']}")
    return "\n".join(lines)


def _format_retention(resources: dict, info: dict) -> str:
    actions = recommend_retention_actions(resources, info["customer_id"])
    lines = [
        f"### Customer {info['customer_id']}",
        "",
        f"**Churn probability:** {info['churn_probability']:.1%}",
        f"**Risk level:** {info['risk_level']}",
        "",
        "#### Recommended retention actions",
    ]
    for i, action in enumerate(actions, start=1):
        lines.append(f"{i}. {action}")
    return "\n".join(lines)


def _format_profile(resources: dict, customer_id: str) -> str:
    profile = get_customer_profile(resources, customer_id)
    if profile is None:
        return _unknown_customer(customer_id)

    return "\n".join(
        [
            f"### Customer {profile['customer_id']}",
            "",
            f"**Tenure:** {profile['tenure']} months",
            f"**Contract:** {profile['contract']}",
            f"**Monthly charges:** ${profile['monthly_charges']:.2f}",
            f"**Payment method:** {profile['payment_method']}",
            f"**Technical support:** {profile['tech_support']}",
            f"**Internet service:** {profile['internet_service']}",
            "",
            f"**Predicted churn probability:** {profile['churn_probability']:.1%}",
            f"**Risk level:** {profile['risk_level']}",
        ]
    )


def _format_overall_rate(resources: dict) -> str:
    rate = float(resources["y"].mean())
    return (
        f"### Overall Churn Rate\n\n"
        f"**{rate:.1%}** of customers in this dataset churned "
        f"({int(resources['y'].sum())} of {len(resources['y'])})."
    )


def _format_global_drivers(resources: dict) -> str:
    overall = explain_overall_churn(resources)
    lines = [
        "### Why Customers Are Churning",
        "",
        f"**Overall churn rate:** {overall['churn_rate']:.1%}",
        "",
        "#### Strongest churn-related factors",
    ]
    for item in overall["top_features"]:
        lines.append(f"- {item['readable_feature']}")
    return "\n".join(lines)


def _format_feature_analysis(analysis: dict) -> str:
    feature = analysis["feature"]

    title_map = {
        "tenure": "How tenure relates to churn",
        "MonthlyCharges": "How monthly charges relate to churn",
        "TotalCharges": "How total charges relate to churn",
        "numTechTickets": "How technical-support tickets relate to churn",
        "numAdminTickets": "How administrative tickets relate to churn",
        "Contract": "How contract type relates to churn",
        "PaymentMethod": "How payment method relates to churn",
        "InternetService": "How internet service relates to churn",
        "TechSupport": "How technical support relates to churn",
    }
    title = title_map.get(feature, f"How {feature} relates to churn")
    lines = [f"### {title}", ""]

    for row in analysis["rows"][:8]:
        label = (
            _format_bin_label(row["group"], feature)
            if analysis["kind"] == "numeric"
            else str(row["group"])
        )
        count = int(row["count"])
        lines.append(
            f"- **{label}:** {row['churn_rate']:.1%} churn ({count:,} customers)"
        )

    if analysis["kind"] == "numeric":
        insight = _numeric_trend_insight(analysis)
    else:
        high = analysis["highest"]
        insight = (
            f"**Key insight:** **{high['group']}** has the highest observed churn "
            f"({high['churn_rate']:.1%})."
        )

    lines.extend(["", insight])
    return "\n".join(lines)


def _format_multi_segment(resources: dict) -> str:
    lines = ["### Segments with the Highest Churn", ""]
    for col in DEFAULT_SEGMENT_FEATURES:
        analysis = analyze_feature(resources, col)
        if analysis is None:
            continue
        high = analysis["highest"]
        lines.append(
            f"- **{col}:** {high['group']} ({high['churn_rate']:.1%})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama intent parser + keyword fallback
# ---------------------------------------------------------------------------
def _empty_parsed(intent: str = "unsupported") -> dict:
    return {
        "intent": intent,
        "customer_id": None,
        "feature": None,
        "segment": None,
        "metric": None,
    }


def validate_parsed(parsed: dict | None, question: str, resources: dict) -> dict:
    """Validate/normalize LLM or fallback JSON before routing."""
    if not isinstance(parsed, dict):
        return _empty_parsed("unsupported")

    intent = str(parsed.get("intent") or "unsupported").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "unsupported"

    customer_id = parsed.get("customer_id")
    if customer_id in ("", "null", "None"):
        customer_id = None
    if customer_id is not None:
        customer_id = str(customer_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9]{4}-[A-Za-z0-9]{5}", customer_id):
            customer_id = extract_customer_id(question, resources)

    # Prefer an ID that actually appears in the question
    question_id = extract_customer_id(question, resources)
    if question_id:
        customer_id = question_id
    elif customer_id is not None:
        # Never keep an invented ID that is not in the question text
        customer_id = None

    feature = resolve_feature_name(parsed.get("feature"))
    if feature is None:
        feature = extract_feature_from_text(question)

    result = {
        "intent": intent,
        "customer_id": customer_id,
        "feature": feature,
        "segment": parsed.get("segment")
        if parsed.get("segment") not in ("", "null", None)
        else None,
        "metric": parsed.get("metric")
        if parsed.get("metric") not in ("", "null", None)
        else None,
    }
    return _refine_intent(result, question)


def _refine_intent(parsed: dict, question: str) -> dict:
    """
    Correct common small-model intent mistakes with light deterministic rules.
    Does not invent numbers; only adjusts routing.
    """
    q = question.strip().lower()
    intent = parsed["intent"]
    customer_id = parsed["customer_id"]
    feature = parsed["feature"]

    # Customer profile / "what do we know"
    if customer_id and any(
        t in q
        for t in [
            "what do we know",
            "tell me about",
            "customer profile",
            "about customer",
            "who is customer",
        ]
    ):
        parsed["intent"] = "customer_profile"
        return parsed

    # Customer + factors / why / leave => explanation (not retention/prediction)
    if customer_id and any(t in q for t in ["factor", "why", "leave", "leaving", "explain"]):
        parsed["intent"] = "customer_explanation"
        return parsed

    # Retention wording without "factor/why"
    if customer_id and any(
        t in q for t in ["retain", "retention", "what can we do", "what can", "what should"]
    ):
        parsed["intent"] = "customer_retention"
        return parsed

    if customer_id and any(t in q for t in ["probability", "chance", "risk"]):
        parsed["intent"] = "customer_prediction"
        return parsed

    # Generic lookup with a customer ID and no clearer intent
    if customer_id and intent in {"unsupported", "customer_prediction", "customer_explanation"}:
        if q.strip() in {customer_id.lower(), f"customer {customer_id.lower()}"}:
            parsed["intent"] = "customer_profile"
            return parsed

    # No customer ID: general dataset questions
    if customer_id is None:
        if any(t in q for t in ["segment", "segments"]) and "churn" in q:
            parsed["intent"] = "segment_analysis"
            return parsed

        if feature and any(
            t in q
            for t in [
                "highest",
                "lowest",
                "which payment",
                "which contract",
                "which internet",
            ]
        ):
            parsed["intent"] = "segment_analysis"
            return parsed

        if feature and any(
            t in q
            for t in [
                "affect",
                "relate",
                "relationship",
                "churn more",
                "leave more",
                "how does",
                "does ",
                "impact",
            ]
        ):
            parsed["intent"] = "feature_analysis"
            return parsed

        if any(
            t in q
            for t in [
                "most associated",
                "which factors",
                "drivers",
                "generally churning",
                "why are customers",
                "why customers",
            ]
        ):
            parsed["intent"] = "global_drivers"
            return parsed

        if "churn rate" in q or ("overall" in q and "churn" in q):
            parsed["intent"] = "overall_churn"
            return parsed

    # If model claimed a customer intent without an ID, treat as unsupported/general
    if intent.startswith("customer_") and not customer_id:
        if feature:
            parsed["intent"] = "feature_analysis"
        else:
            parsed["intent"] = "unsupported"

    return parsed


def parse_question_with_llm(question: str) -> dict | None:
    """
    Ask local Ollama to return intent JSON only.
    Returns None if Ollama is unavailable or response is invalid.
    """
    known_features = ", ".join(sorted(CATEGORICAL_FEATURES | NUMERIC_FEATURES))
    prompt = f"""
Classify the question. JSON only. Do not answer it.

Intents:
- customer_prediction
- customer_explanation
- customer_retention
- customer_profile
- overall_churn
- global_drivers
- segment_analysis
- feature_analysis
- unsupported

Features or null: {known_features}
Return: {{"intent":"...","customer_id":null,"feature":null,"segment":null,"metric":null}}

Question: {question}
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": 200},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("response", "")
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            return None
        return parsed
    except Exception:
        return None


def parse_question_with_keywords(question: str, resources: dict) -> dict:
    """Keyword/regex router used when Ollama is unavailable."""
    q = question.strip().lower()
    parsed = _empty_parsed()
    parsed["customer_id"] = extract_customer_id(question, resources)
    parsed["feature"] = extract_feature_from_text(question)

    if not q:
        return parsed

    # Customer-specific
    if parsed["customer_id"]:
        if any(
            t in q
            for t in [
                "what do we know",
                "tell me about",
                "customer profile",
                "about customer",
                "who is customer",
            ]
        ):
            parsed["intent"] = "customer_profile"
            return parsed

        if any(t in q for t in ["retain", "retention", "what can we do", "what can", "what should", "keep"]):
            if "factor" in q or "why" in q:
                parsed["intent"] = "customer_explanation"
            else:
                parsed["intent"] = "customer_retention"
            return parsed

        if "stay" in q and "factor" not in q and "why" not in q:
            parsed["intent"] = "customer_retention"
            return parsed

        if any(t in q for t in ["why", "factor", "leave", "leaving", "explain"]):
            parsed["intent"] = "customer_explanation"
            return parsed

        if any(t in q for t in ["probability", "risk", "chance", "likely"]):
            parsed["intent"] = "customer_prediction"
            return parsed

        parsed["intent"] = "customer_profile"
        return parsed

    # General / data questions
    if any(t in q for t in ["segment", "segments"]) and "churn" in q:
        parsed["intent"] = "segment_analysis"
        return parsed

    if any(
        t in q
        for t in [
            "highest churn",
            "which payment",
            "which contract",
            "which internet",
        ]
    ):
        parsed["intent"] = "segment_analysis"
        if parsed["feature"] is None and "payment" in q:
            parsed["feature"] = "PaymentMethod"
        return parsed

    if parsed["feature"] and any(
        t in q
        for t in [
            "affect",
            "relate",
            "relationship",
            "associated",
            "impact",
            "churn more",
            "higher churn",
            "leave more",
            "how does",
            "does ",
        ]
    ):
        # Categorical "which X has highest" already handled; treat relation Qs as feature_analysis
        if parsed["feature"] in CATEGORICAL_FEATURES and any(
            t in q for t in ["which", "highest", "lowest"]
        ):
            parsed["intent"] = "segment_analysis"
        else:
            parsed["intent"] = "feature_analysis"
        return parsed

    if any(t in q for t in ["most associated", "drivers", "which factors", "global"]):
        parsed["intent"] = "global_drivers"
        return parsed

    if any(
        t in q
        for t in [
            "generally",
            "overall churn",
            "churn rate",
            "why are customers",
            "why customers",
            "in general",
        ]
    ):
        if "factor" in q or "driver" in q or "associated" in q or "why" in q:
            parsed["intent"] = "global_drivers"
        else:
            parsed["intent"] = "overall_churn"
        return parsed

    if "overall" in q and "churn" in q:
        parsed["intent"] = "overall_churn"
        return parsed

    return parsed


def parse_question(question: str, resources: dict) -> tuple[dict, str]:
    """
    Parse with Ollama when possible; otherwise keyword fallback.
    Returns (validated_parsed, source) where source is 'ollama' or 'fallback'.
    """
    cache_key = question.strip().lower()
    if cache_key in _PARSE_CACHE:
        return _PARSE_CACHE[cache_key]

    llm_raw = parse_question_with_llm(question)
    if llm_raw is not None:
        result = validate_parsed(llm_raw, question, resources), "ollama"
    else:
        result = (
            validate_parsed(
                parse_question_with_keywords(question, resources), question, resources
            ),
            "fallback",
        )

    _PARSE_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
def route_intent(resources: dict, parsed: dict) -> str:
    """Execute a predefined Python function for the validated intent."""
    intent = parsed["intent"]
    customer_id = parsed["customer_id"]
    feature = parsed["feature"]

    customer_intents = {
        "customer_prediction",
        "customer_explanation",
        "customer_retention",
        "customer_profile",
    }

    if intent in customer_intents:
        if not customer_id:
            return (
                "Please include a customer ID (for example 4183-MYFRB) "
                "so I can look up that customer."
            )
        if find_customer(resources, customer_id) is None:
            return _unknown_customer(customer_id)

        if intent == "customer_prediction":
            return _format_prediction(predict_customer(resources, customer_id))
        if intent == "customer_profile":
            return _format_profile(resources, customer_id)
        if intent == "customer_retention":
            info = explain_customer(resources, customer_id)
            return _format_retention(resources, info)
        return _format_explanation(explain_customer(resources, customer_id))

    if intent == "overall_churn":
        return _format_overall_rate(resources)

    if intent == "global_drivers":
        return _format_global_drivers(resources)

    if intent == "segment_analysis":
        if feature:
            analysis = analyze_feature(resources, feature)
            if analysis is None:
                return (
                    "I couldn't analyze that segment. Try Contract, PaymentMethod, "
                    "InternetService, or TechSupport."
                )
            return _format_feature_analysis(analysis)
        return _format_multi_segment(resources)

    if intent == "feature_analysis":
        if not feature:
            return (
                "Please name a feature to analyze, such as contract type, "
                "monthly charges, tenure, or payment method."
            )
        analysis = analyze_feature(resources, feature)
        if analysis is None:
            return (
                "That feature isn't available. Try contract type, monthly charges, "
                "tenure, payment method, internet service, or technical support."
            )
        return _format_feature_analysis(analysis)

    return _unsupported_message()


def answer_question(resources: dict, question: str) -> dict:
    """
    Route an English question to dataset/model analytics.

    Returns:
        {
            "answer": "<clean Markdown string for the UI>",
            "metadata": {"parser": "...", "intent": "...", "parsed": {...}}
        }
    """
    if not str(question).strip():
        return {
            "answer": (
                "Please enter a question about customer churn or a specific customer ID."
            ),
            "metadata": {"parser": "none", "intent": "unsupported", "parsed": None},
        }

    parsed, source = parse_question(question, resources)
    answer = route_intent(resources, parsed)
    return {
        "answer": answer,
        "metadata": {
            "parser": source,
            "intent": parsed.get("intent"),
            "parsed": parsed,
        },
    }
