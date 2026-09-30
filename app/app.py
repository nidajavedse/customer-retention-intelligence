"""
Customer Retention Intelligence — Streamlit frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "Customer_Churn_Dataset.xlsx"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retention_assistant import answer_question, load_resources  # noqa: E402

EXAMPLE_OVERALL = [
    "Why are customers churning?",
    "Which customer segments have the highest churn?",
    "How does tenure relate to churn?",
]
EXAMPLE_CUSTOMER = [
    "Why is customer 4183-MYFRB likely to churn?",
    "What is customer 4183-MYFRB's churn probability?",
    "What retention actions should we consider for customer 4183-MYFRB?",
]


@st.cache_resource(show_spinner="Loading model and explanations...")
def get_resources():
    return load_resources()


def extract_answer_text(result) -> str:
    """
    Pull only the user-facing Markdown string from answer_question().

    Supports:
    - {"answer": "...", "metadata": {...}}
    - ("...", metadata)
    - plain string
    """
    if isinstance(result, dict):
        answer = result.get("answer", "")
    elif isinstance(result, (tuple, list)) and result:
        answer = result[0]
    else:
        answer = result

    if not isinstance(answer, str):
        answer = str(answer)

    # Repair any accidentally escaped newlines from older sessions/objects
    if "\\n" in answer and "\n" not in answer:
        answer = answer.replace("\\n", "\n")

    return answer.strip()


def is_clean_message_content(content) -> bool:
    """Reject old chat entries that stored tuples/dicts/metadata."""
    if not isinstance(content, str):
        return False
    lowered = content.lower()
    bad_markers = (
        "'parser'",
        '"parser"',
        "'parsed'",
        '"parsed"',
        "'intent'",
        "metadata",
        "{'parser'",
        '{"parser"',
    )
    if any(marker in lowered for marker in bad_markers):
        return False
    if content.lstrip().startswith(("(", "{", "[")):
        return False
    return True


def render_sidebar() -> str:
    st.sidebar.markdown("### Customer Retention Intelligence")
    page = st.sidebar.radio(
        "Navigate",
        ["Assistant", "How It Works"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    if st.sidebar.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("#### Dataset")
    st.sidebar.caption("Download the customer churn dataset used by this app.")

    if DATA_PATH.exists():
        st.sidebar.download_button(
            label="Download Dataset",
            data=DATA_PATH.read_bytes(),
            file_name="Customer_Churn_Dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.sidebar.warning("Dataset file not found.")

    return page


def render_how_it_works() -> None:
    st.title("How Customer Retention Intelligence Works")
    st.markdown(
        "This application helps teams explore **why customers churn**, "
        "assess **individual customer risk**, and identify **potential retention actions**."
    )

    st.markdown("## 1. Customer Data")
    st.markdown(
        """
Historical customer information includes attributes such as:

- tenure
- contract type
- monthly charges
- payment method
- services
- support interactions
- historical churn outcome
        """
    )

    st.markdown("## 2. Churn Prediction")
    st.markdown(
        """
A machine-learning classification model learns patterns associated with customers
who stayed or churned.

For an individual customer, the model produces:

- churn probability
- predicted churn outcome
- risk level
        """
    )

    st.markdown("## 3. Explainable AI")
    st.markdown(
        """
SHAP is used to explain model predictions.

For an individual customer, the system identifies:

- factors increasing predicted churn risk
- factors reducing predicted churn risk

Across the full customer base, SHAP also identifies the strongest overall
churn-related features.

Feature relationships and SHAP explanations describe observed or model-learned
associations and should not be interpreted as proof of causation.
        """
    )

    st.markdown("## 4. Retention Intelligence")
    st.markdown(
        """
The system combines:

**Customer Data → Machine Learning → SHAP Explainability → Retention Recommendations**

This allows business users to move beyond simply predicting churn and understand
what may be driving customer risk.
        """
    )

    st.markdown("## 5. Natural-Language Analytics")
    st.markdown(
        """
Users do not need to write Python or SQL.

A local language model interprets English questions and routes them to predefined
analytics functions.

**Example**

User: *Do customers with high monthly charges churn more?*

The language model understands the question. Python then calculates the answer
directly from the dataset.

> The language model interprets the question, while all statistics, predictions,
> and analytical results are calculated from the actual dataset and trained model.
        """
    )

    st.markdown("## Architecture")
    st.markdown(
        """
<style>
.arch-wrap {
  max-width: 520px;
  margin: 0.5rem auto 0.25rem auto;
  font-family: "Source Sans Pro", "Segoe UI", sans-serif;
}
.arch-card {
  background: #f7f8fa;
  border: 1px solid #d9dee7;
  border-radius: 12px;
  padding: 0.9rem 1rem;
  text-align: center;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.arch-card .arch-title {
  font-size: 1.02rem;
  font-weight: 700;
  color: #1f2937;
  margin: 0;
  line-height: 1.3;
}
.arch-card .arch-sub {
  font-size: 0.88rem;
  color: #5b6472;
  margin: 0.25rem 0 0 0;
  line-height: 1.35;
}
.arch-arrow {
  text-align: center;
  color: #6b7280;
  font-size: 1.35rem;
  line-height: 1;
  margin: 0.35rem 0;
  letter-spacing: 0;
}
.arch-card.final {
  background: #eef2f7;
  border-color: #c8d2e0;
}
</style>
<div class="arch-wrap">
  <div class="arch-card">
    <p class="arch-title">User Question</p>
  </div>
  <div class="arch-arrow">↓</div>
  <div class="arch-card">
    <p class="arch-title">Local LLM</p>
    <p class="arch-sub">Question Understanding</p>
  </div>
  <div class="arch-arrow">↓</div>
  <div class="arch-card">
    <p class="arch-title">Analytics Engine</p>
    <p class="arch-sub">Pandas / ML / SHAP</p>
  </div>
  <div class="arch-arrow">↓</div>
  <div class="arch-card">
    <p class="arch-title">Customer Data + Trained Model</p>
  </div>
  <div class="arch-arrow">↓</div>
  <div class="arch-card final">
    <p class="arch-title">Data-Grounded Answer</p>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Technology")
    st.markdown(
        """
- Python
- Pandas
- Scikit-learn
- Logistic Regression
- SHAP
- Ollama
- Streamlit
        """
    )


def render_assistant(resources: dict) -> None:
    st.title("Customer Retention Intelligence")
    st.markdown(
        "Explore overall churn patterns or investigate individual customer "
        "retention risk using machine learning and explainable AI."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Overall analysis**")
        for q in EXAMPLE_OVERALL:
            st.markdown(f"- {q}")
    with right:
        st.markdown("**Customer analysis**")
        for q in EXAMPLE_CUSTOMER:
            st.markdown(f"- {q}")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Drop corrupted older messages that stored tuples/dicts instead of text
    st.session_state.messages = [
        m
        for m in st.session_state.messages
        if isinstance(m, dict)
        and m.get("role") in {"user", "assistant"}
        and is_clean_message_content(m.get("content"))
    ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about overall churn or a specific customer...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                result = answer_question(resources, question)
                answer = extract_answer_text(result)
            except Exception:
                answer = (
                    "Something went wrong while analyzing that question. "
                    "Please try again or rephrase it."
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(
        page_title="Customer Retention Intelligence",
        page_icon="📊",
        layout="wide",
    )

    page = render_sidebar()

    if page == "How It Works":
        render_how_it_works()
        return

    try:
        resources = get_resources()
    except Exception as exc:
        st.error(
            "Could not load the dataset or trained model. "
            "Please confirm the data and models folders are present."
        )
        st.caption(str(exc))
        return

    render_assistant(resources)


if __name__ == "__main__":
    main()
