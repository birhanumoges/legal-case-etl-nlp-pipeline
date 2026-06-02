# legal-case-etl-nlp-pipeline
ETL pipeline that converts historical court case HTML/JSON into structured NLP-ready datasets with verdict extraction, legal citation parsing, and case classification for building legal AI assistants.Production-grade, research-quality NLP pipeline for historical legal case analysis.

## Features

| Component | Description |
|-----------|-------------|
| **ETL** | JSON → structured DataFrame with verdict, case type, sub-type, citations |
| **Classification** | 3 targets × 3 models (LR, SVM, XGBoost) with SMOTE + class weighting |
| **Time Series** | Trend analysis + ARIMA/Prophet forecasting |
| **RAG** | FAISS vectorstore + LLM query engine |
| **Explainability** | SHAP + TF-IDF feature importance |
| **API** | FastAPI REST endpoints for prediction + RAG |

## Folder Structure

```
legal_nlp/
├── main.py                  # Orchestrator — run this
├── config.py                # All paths & constants
├── requirements.txt
├── extractors/              # JSON → text, metadata, verdict, citations
├── preprocessing/           # Clean, encode, split, embed
├── modeling/                # 3 models × 3 targets, trainer, SHAP
├── rag/                     # Retriever, generator, RAG pipeline
├── vectorstore/             # FAISS index + embeddings
├── database/                # PostgreSQL ORM + schema
├── explainability/          # SHAP + feature importance
├── pipeline/                # ETL orchestration + checkpoints
├── api/                     # FastAPI app
├── evaluation/              # Metrics, class balance, RAG eval
├── visualization/           # EDA, confusion matrix, SHAP plots
├── time_series/             # Trend analysis + forecasting
├── utils/                   # Logger, text helpers, config loader
├── notebooks/               # Jupyter notebooks
├── tests/                   # pytest test suite
└── output/                  # All saved outputs
    ├── clean_data.csv
    ├── unknown_case_data.csv
    ├── full_data.csv
    ├── models/
    ├── reports/
    ├── plots/
    └── vectorstore/
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Configure data path
# Edit config.py → ROOT_PATH to point to your JSON data folder

# 4. (Optional) Set up PostgreSQL
psql -U postgres -c "CREATE DATABASE legal_nlp;"
psql -U postgres -d legal_nlp -f database/schema.sql
```

## Running the Pipeline

```bash
# Full pipeline (ETL → EDA → Models → SHAP → Time Series → RAG)
python main.py

# Skip ETL (use existing clean_data.csv)
python main.py --skip-etl

# ETL only
python main.py --etl-only

# Skip RAG build (faster)
python main.py --no-rag

# Force re-run ETL ignoring checkpoint
python main.py --force-etl

# Skip SHAP (much faster)
python main.py --no-shap
```

## Model Comparison

| Target | Model 1 | Model 2 | Model 3 |
|--------|---------|---------|---------|
| Case_Type | Logistic Regression | Linear SVM | XGBoost |
| Sub_Type | Linear SVM | XGBoost | Hierarchical SVM |
| Verdict | Logistic Regression | XGBoost | Linear SVM |

**Class imbalance strategy:** SMOTE inside pipeline + `class_weight='balanced'`

## Class Mappings

### Case Type (5 classes)
`CIVIL | CRIMINAL | CONTRACT | PROPERTY | TORTS`

### Verdict (5 classes)
`AFFIRMED | REVERSED | DENIED | GRANTED | OTHER`

## API

```bash
# Start API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"case_text": "The plaintiff filed for breach of contract...", "court": "Superior Court", "num_citations": 3}'

# RAG query
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the outcome of contract cases?", "top_k": 5}'
```

## Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_extractors.py -v
pytest tests/test_verdict.py -v
pytest tests/test_classifier.py -v
pytest tests/test_rag.py -v
pytest tests/test_vectorstore.py -v
pytest tests/test_time_series.py -v
pytest tests/test_pipeline.py -v
```

## Outputs

All outputs are saved to `output/`:

| File/Folder | Contents |
|-------------|----------|
| `clean_data.csv` | Cleaned + mapped data used for modeling |
| `unknown_case_data.csv` | Unclassified / verdict-unknown rows |
| `full_data.csv` | All extracted rows (raw) |
| `reports/*.json` | Per-target model reports (accuracy, F1, confusion matrix) |
| `reports/*_model_comparison.csv` | All 3 models compared per target |
| `reports/*_actual_vs_predicted_*.csv` | Full prediction output with all columns |
| `reports/ts_*.csv` | Time-series trend tables |
| `plots/eda_*.png` | EDA distribution charts |
| `plots/cm_*.png` | Confusion matrices (raw + normalised) |
| `plots/shap_*.png` | SHAP feature importance charts |
| `plots/ts_*.png` | Time-series + forecast plots |
| `models/*.joblib` | Saved best models + feature encoder |
| `vectorstore/` | FAISS index + embeddings |
