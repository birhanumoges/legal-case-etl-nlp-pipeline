
<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3c72,100:2a5298&height=200&section=header&text=Legal%20NLP%20Platform&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Court%20Case%20Intelligence%20%E2%80%A2%20Classification%20%E2%80%A2%20RAG%20Search&descAlignY=58&descSize=18" width="100%"/>
<a href="https://github.com">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=2800&pause=900&color=2A5298&center=true&vCenter=true&width=780&lines=ETL+%E2%86%92+NLP+%E2%86%92+Classification+%E2%86%92+RAG+%E2%86%92+API;17%2C987+court+cases+processed;96.4%25+verdict+prediction+accuracy;FAISS-powered+semantic+case+search" alt="Typing SVG" />
</a>
<br/>
<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Accuracy-96.4%25-2ea44f?style=flat-square"/>
  <img src="https://img.shields.io/badge/Records-17%2C987-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/RAG%20Chunks-3%2C487-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square"/>
</p>
</div>
<br/>

## 📚 Table of Contents
 
<div align="center">
| [🚀 Overview](#-overview) | [🏗️ Architecture](#️-architecture) | [✨ Features](#-features) | [📁 Structure](#-project-structure) |
|:---:|:---:|:---:|:---:|
| [⚡ Quick Start](#-quick-start) | [🔌 API Reference](#-api-reference) | [📊 Results](#-key-results) | [🐳 Docker](#-docker-full-stack) |
| [🧪 Testing](#-testing) | [🗄️ Migrations](#️-database-migrations) | [🤝 Contributing](#-contributing) | [📄 License](#-license) |
 
</div>
---
 
## 🚀 Overview
 
**Legal NLP Platform** is a production-grade, end-to-end system that turns raw historical court case data (HTML/JSON) into a fully queryable legal intelligence platform — combining rule-based ETL, multi-model classification, time-series forecasting, and retrieval-augmented generation (RAG), all served through a FastAPI backend and a React dashboard.
 
> Built for teams building **legal AI assistants**, case-outcome research tools, or judicial analytics products.


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
Legal NLP Platform
End-to-end legal case classification, verdict prediction, analytics, and semantic search.
---
Project Structure
```
legal_nlp/
├── main.py                    # Data science pipeline orchestrator
├── config.py                  # All paths, constants, hyperparameters
├── requirements.txt
├── alembic.ini                # Database migration config
├── docker-compose.yml         # Full-stack Docker orchestration
├── Dockerfile.api             # Backend Docker image
├── .env.example               # Environment variable template
│
├── extractors/                # Rule-based ETL extractors
├── preprocessing/             # Feature encoding, cleaning, splitting
├── modeling/                  # LR / SVM / XGBoost classifiers
├── pipeline/                  # ETL orchestration + checkpoints
├── evaluation/                # Metrics, class balance, RAG eval
├── visualization/             # EDA, confusion matrix, SHAP plots
├── time_series/               # ARIMA forecasting + trend analysis
├── explainability/            # Feature importance + SHAP
├── rag/                       # Retrieval-Augmented Generation
├── vectorstore/               # FAISS index + embeddings
├── database/                  # SQLAlchemy ORM + migrations
├── utils/                     # Logger, text helpers
├── tests/                     # pytest test suite
├── notebooks/                 # Jupyter analysis notebooks
│
├── api/                       # FastAPI backend
│   ├── app.py                 # Entry point — registers all routers
│   ├── routes.py              # GET /health, POST /predict, POST /rag/query
│   ├── routes_auth.py         # POST /auth/login
│   ├── routes_cases.py        # GET/POST /cases  GET /cases/{id}
│   ├── routes_analytics.py    # GET /analytics/stats|yearly|forecast|models
│   ├── routes_predict.py      # POST /predict/batch
│   ├── schemas.py             # All Pydantic models
│   └── dependencies.py        # JWT auth, pagination
│
├── scripts/
│   ├── run_api.py             # Start API server
│   ├── seed_db.py             # Load clean_data.csv → PostgreSQL
│   ├── build_vectorstore.py   # Build FAISS index
│   ├── export_reports.py      # Export all outputs to ZIP
│   └── check_health.py        # Verify pipeline + API health
│
├── frontend/                  # React + Tailwind frontend
│   ├── src/
│   │   ├── App.jsx            # Router
│   │   ├── pages/
│   │   │   ├── Login.jsx      # JWT login
│   │   │   ├── Dashboard.jsx  # Overview stats + charts
│   │   │   ├── Cases.jsx      # Browseable + filterable case list
│   │   │   ├── CaseDetail.jsx # Full case detail + similar cases
│   │   │   ├── Predict.jsx    # Single + batch prediction UI
│   │   │   ├── RAG.jsx        # Chat-style RAG query interface
│   │   │   ├── Analytics.jsx  # Distributions, trends, forecast
│   │   │   └── ModelReports.jsx # Model comparison + radar charts
│   │   ├── components/        # Layout, StatCard, Badge, Pagination, …
│   │   ├── services/api.js    # Axios API client
│   │   ├── context/           # Auth context
│   │   ├── hooks/             # React Query hooks
│   │   └── utils/             # Helpers, color maps
│   ├── Dockerfile.frontend
│   └── nginx.conf
│
└── output/                    # All pipeline outputs (auto-created)
    ├── clean_data.csv
    ├── unknown_case_data.csv
    ├── models/
    ├── reports/
    ├── plots/
    └── vectorstore/
```
---
Quick Start
1. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Configure environment
```bash
cp .env.example .env
# Edit .env — set LEGAL_DATA_ROOT to your JSON data folder
```
3. Run the data science pipeline
```bash
python main.py                    # Full pipeline (~4-5 hours on 12k cases)
python main.py --skip-etl         # Skip ETL, load existing clean_data.csv
python main.py --etl-only         # ETL only, then exit
python main.py --no-rag           # Skip RAG vectorstore build
python main.py --no-shap          # Skip SHAP (faster)
```
4. Start the API server
```bash
python scripts/run_api.py
# or directly:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
5. Start the frontend
```bash
cd frontend
npm install
npm run dev                       # Development: http://localhost:3000
npm run build                     # Production build → frontend/dist/
```
6. Open the platform
Frontend: http://localhost:3000
API docs: http://localhost:8000/docs
Login: admin / admin123
---
Docker (Full Stack)
```bash
docker-compose up --build
```
Frontend: http://localhost:3000
API:      http://localhost:8000
Database: localhost:5432
---
API Endpoints
Method	Endpoint	Description
GET	`/api/v1/health`	Health check
POST	`/api/v1/auth/login`	Get JWT token
POST	`/api/v1/predict`	Single case prediction
POST	`/api/v1/predict/batch`	Batch prediction (up to 50)
POST	`/api/v1/rag/query`	RAG semantic query
GET	`/api/v1/cases`	List cases (paginated, filtered)
POST	`/api/v1/cases/search`	Full-text + filtered search
GET	`/api/v1/cases/{id}`	Case detail
GET	`/api/v1/cases/{id}/similar`	Semantically similar cases
GET	`/api/v1/analytics/stats`	Corpus statistics
GET	`/api/v1/analytics/yearly`	Yearly statistics
GET	`/api/v1/analytics/forecast`	ARIMA forecast
GET	`/api/v1/analytics/models`	All model reports
GET	`/api/v1/analytics/models/{target}`	Single target report
---
Database Migrations
```bash
# Initialise Alembic (first time)
alembic init database/migrations

# Run all migrations
alembic upgrade head

# Seed from CSV
python scripts/seed_db.py

# Create new migration after model changes
alembic revision --autogenerate -m "describe change"
```
---
Tests
```bash
pytest                                  # All tests
pytest tests/test_extractors.py -v     # Extractor tests
pytest tests/test_verdict.py -v        # Verdict extractor tests
pytest tests/test_classifier.py -v     # ML model tests
pytest tests/test_rag.py -v            # RAG tests
pytest tests/test_pipeline.py -v       # Pipeline tests
```
---
Key Results
Target	Best Model	Accuracy	Macro-F1
Case Type (5 classes)	XGBoost	84.5%	83.8%
Sub-Type (21 classes)	XGBoost	76.8%	70.3%
Verdict (4 classes)	XGBoost	96.4%	96.2%
Total validated records: 17,987
Clean labelled records: 11,712
ARIMA(2,2,1) forecast: AIC = 1,560.38
RAG vectorstore: 3,487 chunks indexed in FAISS
