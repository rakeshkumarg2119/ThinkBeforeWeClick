# 🛡️ Think Before You Click

An AI-powered URL safety checker that detects phishing, malware, scams, piracy, financial fraud, and gambling sites using machine learning models built with Scikit-learn. Features a FastAPI backend, SQLite caching layer, and a Streamlit frontend.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Frontend Guide](#frontend-guide)
- [Core Engine & ML Models](#core-engine--ml-models)
- [Database Schema](#database-schema)
- [Scoring System](#scoring-system)
- [Threat Detection Logic](#threat-detection-logic)
- [Caching Strategy](#caching-strategy)
- [Configuration & Customization](#configuration--customization)
- [Development Notes](#development-notes)

---

## Overview

URL Risk Analyzer is a full-stack application that takes any URL as input and produces a detailed risk report in seconds. It combines rule-based heuristics (domain reputation, keyword matching, TLD scoring, redirect analysis) with machine learning (Random Forest classifiers, Isolation Forest anomaly detection) to assess URLs across six threat categories.

Results are cached in SQLite so repeated lookups are near-instant. As the database grows, the ML models auto-retrain to improve accuracy over time.

---

## Features

- **6 Threat Categories** — Phishing, Malware, Scam, Financial Fraud, Piracy, Gambling/Betting
- **5-Component Scoring** — Domain, URL structure, Keywords, Security (HTTPS), Redirects
- **ML-Powered Classification** — Random Forest for risk level and threat type prediction
- **Anomaly Detection** — Isolation Forest flags unusual URL patterns
- **Smart Caching** — SQLite cache returns cached results instantly with a visual badge
- **Gambling Warnings** — Tiered financial risk warnings for gambling/betting sites
- **Trusted Domain Whitelist** — Major platforms (Google, GitHub, etc.) always score zero risk
- **Auto-Retraining** — Models retrain automatically every 50 new URL submissions
- **REST API** — Clean FastAPI endpoints with full Pydantic validation and OpenAPI docs
- **Streamlit UI** — Dark-themed, responsive frontend with live score bars and KPI cards

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / Browser                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP (port 8501)
┌──────────────────────────────▼──────────────────────────────────┐
│                    Streamlit Frontend                            │
│  frontend/app.py                                                 │
│  • URL input form            • Score breakdown bars             │
│  • Risk badge display        • Gambling warnings                │
│  • KPI cards (score,         • Raw JSON expander                │
│    confidence, severity)     • Backend health check             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP POST /api/v1/analyze
                               │ (port 8000)
┌──────────────────────────────▼──────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                  │
│  main.py                  routers/analysis.py                   │
│  • App factory            • POST /analyze                       │
│  • CORS middleware        • GET  /health                        │
│  • Lifespan (DB init)     • HTTP error handling                 │
│                                                                  │
│  services/analysis_service.py    models/schemas.py              │
│  • run_analysis()                • URLRequest (Pydantic)        │
│  • _patch_gambling_warning()     • AnalysisResult (Pydantic)    │
│  • ensure_db_ready()             • HealthResponse               │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Python function call
┌──────────────────────────────▼──────────────────────────────────┐
│                      Core Engine                                 │
│  core_engine.py                                                  │
│                                                                  │
│  Feature Extraction Layer                                        │
│  ├── calculate_domain_score()    TLD reputation, hyphens        │
│  ├── calculate_url_score()       Length, @ symbol, subdomains   │
│  ├── calculate_keyword_score()   6 keyword category matchers    │
│  ├── calculate_security_score()  HTTPS check                    │
│  └── calculate_redirect_score()  Live redirect chain analysis   │
│                                                                  │
│  ML Inference Layer                                              │
│  ├── RandomForestClassifier      → risk level (Low/Med/High)    │
│  ├── RandomForestClassifier      → threat type (Phishing/etc.)  │
│  └── IsolationForest             → anomaly flag                 │
│                                                                  │
│  Lookup Tables                                                   │
│  ├── TRUSTED_DOMAINS             ~50 major platforms            │
│  ├── GAMBLING_PLATFORMS          ~30 known gambling sites       │
│  └── TLD_REPUTATION              ~50 TLD risk scores            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ SQLite read/write
┌──────────────────────────────▼──────────────────────────────────┐
│                      Data Layer                                  │
│                                                                  │
│  database.py                      db/url_risk.db                │
│  • initialize_database()          • url_analysis table          │
│  • get_cached_result()            • Indexes on url, domain,     │
│  • store_analysis()                 analyzed_at                 │
│  • get_training_data()                                          │
│  • get_record_count()             models/                       │
│  • get_class_distribution()       • risk_model.pkl              │
│  • update_labels()                • risk_type_model.pkl         │
│                                   • anomaly_model.pkl           │
└─────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
Browser/Streamlit
      │
      │  POST {"url": "https://example.com"}
      ▼
FastAPI Router  →  Pydantic validates URLRequest
      │
      ▼
analysis_service.run_analysis(url)
      │
      ├─► database.get_cached_result(url)
      │         │
      │    Hit ─┘─► _patch_gambling_warning() ─► return cached dict
      │
      │    Miss ─► core_engine.analyze_url(url)
      │                  │
      │                  ├─ extract_features()  (pure Python heuristics)
      │                  ├─ load_models()       (joblib .pkl files)
      │                  ├─ model.predict()     (Scikit-learn inference)
      │                  ├─ get_gambling_warning()
      │                  └─ store_analysis()    (write to SQLite)
      │
      ▼
AnalysisResult (Pydantic) → JSON response
```

---

## Project Structure

```
url-risk-analyzer/
│
├── backend/
│   ├── main.py                   # FastAPI app factory, CORS, lifespan
│   ├── routers/
│   │   └── analysis.py           # Route handlers: /analyze, /health
│   ├── services/
│   │   └── analysis_service.py   # Business logic between router & engine
│   └── models/
│       └── schemas.py            # Pydantic request/response schemas
│
├── frontend/
│   └── app.py                    # Streamlit UI (single file)
│
├── core_engine.py                # ML feature extraction + inference engine
├── database.py                   # SQLite CRUD + thread-safe access
│
├── db/
│   └── url_risk.db               # SQLite database (auto-created)
│
├── models/
│   ├── risk_model.pkl            # Random Forest — risk level
│   ├── risk_type_model.pkl       # Random Forest — threat type
│   └── anomaly_model.pkl         # Isolation Forest — anomaly detection
│
├── .devcontainer/
│   └── devcontainer.json         # GitHub Codespaces config
│
├── .vscode/
│   └── settings.json             # Python path extras
│
├── .gitignore
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit 1.x | Web UI — forms, charts, styled components |
| Backend | FastAPI 0.100+ | REST API — routing, validation, error handling |
| Validation | Pydantic v2 | Request/response schema enforcement |
| ML Models | Scikit-learn | Risk classification + anomaly detection |
| Model Persistence | Joblib | Serialize/deserialize `.pkl` model files |
| Numerics | NumPy | Feature array construction for model inference |
| HTTP Client | Requests | Live redirect chain analysis |
| Database | SQLite 3 | Result caching + training data storage |
| DB Access | Python stdlib `sqlite3` | Thread-safe connection management |
| Server | Uvicorn | ASGI server for FastAPI |
| Dev Container | Docker (Codespaces) | Reproducible dev environment |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/url-risk-analyzer.git
cd url-risk-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install fastapi uvicorn[standard] streamlit pydantic requests \
            scikit-learn joblib numpy
```

### 4. Verify Installation

```bash
python -c "import fastapi, streamlit, sklearn, joblib, numpy; print('All OK')"
```

---

## Running the App

The application has two independently running processes: the FastAPI backend and the Streamlit frontend. Both must be running simultaneously.

### Terminal 1 — Start the Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Expected output:
```
✓ Database initialized: db/url_risk.db
✓ Database ready
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Terminal 2 — Start the Frontend

```bash
streamlit run frontend/app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### Open in Browser

- **Streamlit UI** → http://localhost:8501
- **FastAPI Docs (Swagger)** → http://localhost:8000/docs
- **FastAPI Docs (ReDoc)** → http://localhost:8000/redoc

### GitHub Codespaces

The `.devcontainer/devcontainer.json` is pre-configured. When you open this repo in Codespaces, Streamlit will start automatically on port 8501. You'll still need to start the FastAPI backend manually in a second terminal.

---

## API Reference

### Base URL

```
http://localhost:8000/api/v1
```

---

### `POST /analyze`

Analyzes a URL and returns a full risk report.

**Request Body**

```json
{
  "url": "https://example.com"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `url` | `string` | Yes | Must start with `http://` or `https://` |

**Response — 200 OK**

```json
{
  "url": "https://example.com",
  "domain": "example.com",
  "domain_score": 0,
  "url_score": 0,
  "keyword_score": 0,
  "security_score": 0,
  "redirect_score": 0,
  "total_score": 0,
  "risk_level": "Low",
  "risk_level_numeric": 0,
  "risk_type": "Safe",
  "confidence_percent": 95.0,
  "risk_severity_index": 28,
  "anomaly_detected": false,
  "why_risk": "Verified trusted domain",
  "gambling_warning": null,
  "cached": false
}
```

**Response Fields**

| Field | Type | Description |
|---|---|---|
| `url` | `string` | The submitted URL |
| `domain` | `string` | Extracted domain name |
| `domain_score` | `int` | 0–25. Domain reputation risk |
| `url_score` | `int` | 0–25. URL structure risk |
| `keyword_score` | `int` | 0–25. Keyword match risk |
| `security_score` | `int` | 0–15. HTTPS check (0 = has HTTPS, 15 = no HTTPS) |
| `redirect_score` | `int` | 0–10. Redirect chain risk |
| `total_score` | `int` | 0–100. Sum of all component scores |
| `risk_level` | `string` | `"Low"`, `"Medium"`, or `"High"` |
| `risk_level_numeric` | `int` | `0`, `1`, or `2` |
| `risk_type` | `string` | `"Phishing"`, `"Malware"`, `"Scam"`, `"Financial Fraud"`, `"Piracy"`, `"Gambling/Betting"`, `"Safe"`, or `"Unknown"` |
| `confidence_percent` | `float` | 0.0–100.0. Model confidence |
| `risk_severity_index` | `int` | 0–100. Composite severity score |
| `anomaly_detected` | `bool` | `true` if Isolation Forest flags unusual pattern |
| `why_risk` | `string` | Human-readable explanation of risk factors |
| `gambling_warning` | `string\|null` | Tiered financial warning for gambling sites, else `null` |
| `cached` | `bool` | `true` if served from SQLite cache |

**Error Responses**

| Status | Condition |
|---|---|
| `400 Bad Request` | URL doesn't start with http/https, too short, or unresolvable |
| `422 Unprocessable Entity` | Request body missing or malformed |
| `500 Internal Server Error` | Core engine failure |

**Error Response Body**

```json
{
  "detail": "URL must start with http:// or https://"
}
```

---

### `GET /health`

Liveness probe to verify the backend is running.

**Response — 200 OK**

```json
{
  "status": "ok",
  "message": "URL Risk Analyzer API is running",
  "version": "3.0.0"
}
```

---

### `GET /`

Root redirect — returns API info.

```json
{
  "message": "URL Risk Analyzer API v3.0",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

---

### Example cURL Calls

```bash
# Analyze a URL
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://suspicious-login-verify.tk/account"}'

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Frontend Guide

The Streamlit frontend (`frontend/app.py`) is a single-file application.

### UI Components

**Hero Header** — Title and subtitle rendered via raw HTML/CSS injected with `st.markdown(..., unsafe_allow_html=True)`.

**Backend Health Banner** — On load, calls `GET /health`. If the backend is unreachable, an error banner is shown and the Analyze button is disabled.

**URL Input + Analyze Button** — A two-column layout (`[5, 1]` ratio). Input uses `st.text_input`, button uses `st.button`.

**Result Header** — Shows the risk badge (`LOW RISK` / `MEDIUM RISK` / `HIGH RISK`) with color coding, the domain name, response time, and an `⚡ CACHED` tag if served from cache.

**KPI Cards** — Four cards showing Total Score, Confidence %, Severity Index, and Threat Type.

**Score Breakdown Bars** — Horizontal progress bars for each of the 5 scoring components, color-coded green/amber/red based on ratio to maximum.

**Why-Risk Block** — A teal-bordered info block with the plain-English risk explanation.

**Anomaly Block** — A purple-bordered warning block, visible only when `anomaly_detected: true`.

**Gambling Warning Block** — An amber-bordered warning block with tiered financial risk language, visible only for gambling/betting sites.

**Raw JSON Expander** — Collapsible `st.expander` showing the full API response formatted with `st.code`.

### Color Logic

| Score Ratio | Color |
|---|---|
| < 35% of max | `#00e676` (green) |
| 35%–65% of max | `#ffab40` (amber) |
| > 65% of max | `#ff5252` (red) |

### Fonts

- **Syne** (Google Fonts) — headings and body text
- **Space Mono** (Google Fonts) — monospace labels, domain display, timestamps

---

## Core Engine & ML Models

### `core_engine.py` — Module Overview

The core engine is a standalone Python module with no FastAPI dependency. It can also be run directly as a CLI:

```bash
python core_engine.py
# Then enter URLs interactively, or type 'stats' or 'train'
```

### Feature Extraction

Five features are extracted per URL, producing the feature vector `[domain_score, url_score, keyword_score, security_score, redirect_score]` used by all three ML models.

#### 1. Domain Score (0–25)

Evaluates the domain name itself:

| Check | Points |
|---|---|
| Trusted domain (whitelist) | 0 (immediate return) |
| Known gambling platform | 8 (moderate, immediate return) |
| IP address as domain | 25 (immediate return) |
| High-risk TLD (`.tk`, `.ml`, `.win`, etc.) | 18–25 |
| Medium-risk TLD (`.site`, `.online`, etc.) | 10–12 |
| Domain name > 25 chars | +8 |
| Domain name > 15 chars | +5 |
| More than 3 hyphens | +10 |
| More than 5 digits | +8 |
| 4-digit sequence in name | +5 |

#### 2. URL Score (0–25)

Evaluates the full URL string structure:

| Check | Points |
|---|---|
| IP address in netloc | +20 |
| URL length > 120 chars | +10 |
| URL length > 80 chars | +5 |
| `@` symbol in URL | +15 |
| More than 3 subdomain levels | +8 |
| More than 5 special characters | +8 |
| `//` in path | +10 |
| Query string > 100 chars | +8 |

#### 3. Keyword Score (0–25) + Risk Type

Scans the URL and domain for keywords across 6 categories. The category with the highest match count wins and determines `risk_type`.

| Category | Example Keywords |
|---|---|
| Phishing | `login`, `verify`, `account`, `credential`, `suspend` |
| Financial Fraud | `bank`, `crypto`, `bitcoin`, `forex`, `transfer` |
| Scam | `prize`, `winner`, `lottery`, `free`, `guaranteed` |
| Gambling/Betting | `bet`, `casino`, `rummy`, `fantasy`, `jackpot`, `odds` |
| Malware | `download`, `exe`, `install`, `codec`, `setup` |
| Piracy | `crack`, `keygen`, `nulled`, `torrent`, `warez`, `mod-apk` |

Scoring multipliers: Gambling maxes at 18/25 (moderate); Piracy uses `count × 6`; others use `count × 5`.

#### 4. Security Score (0–15)

Simple binary check: `15` if URL uses `http://`, `0` if it uses `https://`.

#### 5. Redirect Score (0–10)

Makes a live HTTP GET request with a 2-second timeout, following redirects:

| Condition | Points |
|---|---|
| Request fails / timeout | 5 |
| > 5 redirects | 10 |
| 3–5 redirects | 7 |
| 1–2 redirects | 4 |
| Final domain differs from original | 6 |
| No redirects, same domain | 0 |

### Machine Learning Models

All three models are stored as `.pkl` files in the `models/` directory using `joblib`.

#### Model 1: Risk Classifier (`risk_model.pkl`)

```
Algorithm:   RandomForestClassifier
Estimators:  100 trees
Max depth:   10
Class weight: balanced
Input:       [domain_score, url_score, keyword_score, security_score, redirect_score]
Output:      0 (Low), 1 (Medium), 2 (High)
```

When no model exists (fewer than 30 training samples), falls back to thresholds:
- `total_score > 60` → High
- `total_score > 35` → Medium
- else → Low

#### Model 2: Type Classifier (`risk_type_model.pkl`)

```
Algorithm:   RandomForestClassifier
Estimators:  100 trees
Max depth:   8
Input:       [domain_score, url_score, keyword_score, security_score, redirect_score]
Output:      threat type string ("Phishing", "Malware", etc.)
```

Only trains when there are at least 10 labeled samples with at least 2 distinct non-Unknown types. Falls back to keyword-inferred type when unavailable.

#### Model 3: Anomaly Detector (`anomaly_model.pkl`)

```
Algorithm:   IsolationForest
Estimators:  100 trees
Contamination: 0.1 (assumes 10% of training data is anomalous)
Input:       [domain_score, url_score, keyword_score, security_score, redirect_score]
Output:      1 (normal) or -1 (anomaly)
```

Skipped entirely for trusted domains and known gambling platforms.

### Auto-Retraining

```python
MIN_SAMPLES_FOR_TRAINING = 30   # Minimum to train at all
RETRAIN_INTERVAL = 50           # Retrain every N new records
```

`check_and_retrain()` is called after every new analysis. If `record_count % 50 == 0` and count ≥ 30, all three models are retrained from scratch using all stored data.

### Severity Index Formula

```python
# For gambling sites (capped moderate severity)
severity = int(35 + (total_score * 0.4) + (confidence * 0.2))

# For all other sites
severity = int((total_score * 0.7) + (confidence * 0.3))

severity = min(severity, 100)
```

### Gambling Warnings

Three tiers based on `total_score`:

| Score | Warning Level |
|---|---|
| > 50 | `⚠️ FINANCIAL RISK WARNING` — Money loss highly probable |
| > 30 | `⚠️ FINANCIAL CAUTION` — Real money, risk of loss |
| ≤ 30 | `ℹ️ ADVISORY` — Platform involves real money gaming |

---

## Database Schema

### Table: `url_analysis`

```sql
CREATE TABLE url_analysis (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    url                  TEXT NOT NULL UNIQUE,
    domain               TEXT NOT NULL,
    domain_score         INTEGER,
    url_score            INTEGER,
    keyword_score        INTEGER,
    security_score       INTEGER,
    redirect_score       INTEGER,
    total_score          INTEGER,
    predicted_risk_level INTEGER,       -- 0=Low, 1=Medium, 2=High
    predicted_risk_type  TEXT,
    confidence_percent   REAL,
    anomaly_detected     INTEGER,       -- 0 or 1
    risk_severity_index  INTEGER,
    why_risk             TEXT,
    actual_risk_level    INTEGER,       -- NULL until manually labeled
    actual_risk_type     TEXT,          -- NULL until manually labeled
    analyzed_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_url         ON url_analysis(url);
CREATE INDEX idx_domain      ON url_analysis(domain);
CREATE INDEX idx_analyzed_at ON url_analysis(analyzed_at);
```

> **Note:** `gambling_warning` is intentionally **not** a database column. It is derived at runtime from stored scores using `get_gambling_warning()`. The `analysis_service._patch_gambling_warning()` function re-generates this field for cached results, ensuring the Pydantic response schema is always complete.

### Thread Safety

All database operations acquire a module-level `threading.Lock` before opening a connection, preventing race conditions when multiple requests arrive simultaneously.

---

## Scoring System

| Component | Max | Weight |
|---|---|---|
| Domain Score | 25 | 25% |
| URL Score | 25 | 25% |
| Keyword Score | 25 | 25% |
| Security Score | 15 | 15% |
| Redirect Score | 10 | 10% |
| **Total** | **100** | **100%** |

### Risk Level Thresholds (fallback, no model)

| Total Score | Risk Level |
|---|---|
| 0–35 | Low |
| 36–60 | Medium |
| 61–100 | High |

---

## Threat Detection Logic

### Trusted Domains

~50 major platforms are hardcoded in `TRUSTED_DOMAINS`. Any URL whose domain exactly matches or is a subdomain of a trusted entry receives:
- `domain_score = 0`
- `risk_level = Low`
- `confidence = 95%`
- `risk_type = Safe`
- Anomaly detection skipped

### Known Gambling Platforms

~30 platforms in `GAMBLING_PLATFORMS`. Matching domains receive:
- `domain_score = 8` (moderate)
- `gambling_count += 3` (keyword boost)
- `risk_label ≥ 1` (always at least Medium)
- Tiered gambling warning generated
- Anomaly detection skipped
- Severity uses the gambling-specific formula

### TLD Reputation

50+ TLDs scored from 0 (safe) to 25 (very high risk). The longest matching TLD wins. Free/abused TLDs like `.tk`, `.ml`, `.cf`, `.gq`, `.ga` carry the highest scores (25).

---

## Caching Strategy

1. On every `analyze_url(url)` call, `get_cached_result(url)` queries SQLite first.
2. Cache hit → return the stored dict immediately with `cached: True`.
3. Cache miss → run full analysis, store result, return with `cached: False`.
4. The cache never expires — it's a permanent record store used for both caching and ML training data.
5. Re-submitting the same URL will always update the record (`INSERT OR REPLACE` logic via check + conditional UPDATE/INSERT).

**The `gambling_warning` gap:** Since `gambling_warning` is not a DB column, cached results don't include it. `analysis_service._patch_gambling_warning()` reconstructs it from the stored scores using the same `get_gambling_warning()` function before the Pydantic model validates the response.

---

## Configuration & Customization

### Adding Trusted Domains

Edit `TRUSTED_DOMAINS` in `core_engine.py`:

```python
TRUSTED_DOMAINS = {
    'your-company.com',
    'internal-tool.net',
    # ...existing entries
}
```

### Adding Gambling Platforms

Edit `GAMBLING_PLATFORMS` in `core_engine.py`:

```python
GAMBLING_PLATFORMS = {
    'newplatform.com',
    # ...existing entries
}
```

### Adjusting TLD Risk Scores

Edit `TLD_REPUTATION` in `core_engine.py`. Score range 0–25:

```python
TLD_REPUTATION = {
    '.newbadtld': 22,
    # ...existing entries
}
```

### Changing Risk Thresholds

Modify the fallback threshold block in `analyze_url()` inside `core_engine.py`:

```python
if features['total_score'] > 60:    # Change this
    risk_label = 2
elif features['total_score'] > 35:  # And this
    risk_label = 1
```

### Changing Retraining Frequency

```python
MIN_SAMPLES_FOR_TRAINING = 30   # Minimum records before first train
RETRAIN_INTERVAL = 50           # Retrain every N records
```

### CORS Settings

Edit `allow_origins` in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://your-production-frontend.com",
        # Remove "*" in production
    ],
)
```

---

## Development Notes

### Running the Core Engine Standalone

```bash
python core_engine.py
>>> https://suspicious-site.tk/login
>>> stats
>>> train
>>> exit
```

### Manually Triggering Model Training

```python
from core_engine import train_models
train_models()
```

### Manually Correcting Labels (Active Learning)

```python
from database import update_labels
update_labels("https://example.com", risk_level=2, risk_type="Phishing")
```

Updated labels are used as `actual_risk_level` / `actual_risk_type` in cache lookups, overriding the model's prediction.

### API Interactive Docs

FastAPI auto-generates interactive documentation at:
- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

You can test the `/analyze` endpoint directly from the browser without any additional tooling.

### Common Issues

**`ModuleNotFoundError: No module named 'backend'`**
Run uvicorn from the project root, not from inside the `backend/` folder:
```bash
# Correct
uvicorn backend.main:app --reload --port 8000
```

**`Cannot reach the FastAPI backend`** banner in Streamlit
The FastAPI server isn't running. Start it in a separate terminal first.

**Models don't exist yet**
This is normal on a fresh install. The system uses score-based fallback thresholds until 30 URLs have been analyzed and `train_models()` runs automatically.

**Redirect check is slow**
The redirect analyzer makes a live HTTP request with a 2-second timeout. Unreachable URLs will always incur a ~2 second penalty. This is expected behavior.

---

## License

MIT License. See `LICENSE` for details.

---

*Built with FastAPI · Streamlit · Scikit-learn · SQLite*
