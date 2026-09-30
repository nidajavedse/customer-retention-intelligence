# Customer Retention Intelligence Assistant

An explainable machine-learning application that allows users to explore customer churn patterns, investigate individual customer risk, and receive retention recommendations through natural-language questions.

## What it does

- **Overall retention intelligence** — churn rate, key drivers, segment and feature analysis
- **Individual customer intelligence** — churn probability, risk level, SHAP explanations, retention actions
- **Natural-language analytics** — ask questions in English; answers are calculated from the dataset and trained model
- **Fully local** — free open-source stack, no paid APIs or cloud services

## How it works

1. A local language model interprets the question (intent only).
2. Python routes to a predefined analytics function.
3. Pandas, the trained Logistic Regression model, and SHAP produce the result.
4. Streamlit displays a clear business-friendly answer.

All statistics, predictions, and explanations come from the actual dataset and model — not from the language model.

## Setup

```bash
cd customer-retention-intelligence
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (better natural-language understanding):

```bash
ollama serve                       # skip if already running
ollama pull llama3.2:3b
```

If Ollama is unavailable, a keyword-based router is used automatically.

## Run

```bash
streamlit run app/app.py
```

## Example questions

**Overall**
- Why are customers churning?
- What is the overall churn rate?
- Which customer segments have the highest churn?
- How does tenure relate to churn?
- Do customers with high monthly charges churn more?

**Customer**
- What is the churn probability for customer 4183-MYFRB?
- Why is customer 4183-MYFRB likely to churn?
- What factors are helping customer 4183-MYFRB stay?
- What retention actions should we consider for customer 4183-MYFRB?
- What do we know about customer 4183-MYFRB?

## Project layout

```
customer-retention-intelligence/
├── app/app.py
├── data/Customer_Churn_Dataset.xlsx
├── models/baseline_retention_model.joblib
├── src/retention_assistant.py
├── requirements.txt
└── README.md
```

`notebooks/` is optional development history and is not required to run the app.

## Technology

Python · Pandas · Scikit-learn · Logistic Regression · SHAP · Ollama · Streamlit

## Limitations

- SHAP factors are associations with the model prediction, not proven causal causes.
- Retention actions are suggested opportunities, not guarantees.
- Intent understanding depends on the local model when available; keyword fallback covers common questions.
