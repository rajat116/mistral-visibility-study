# 🔍 Mistral AI — Search Visibility Study

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![GCP](https://img.shields.io/badge/GCP-BigQuery%20%2B%20GCS-4285F4?style=flat&logo=googlecloud&logoColor=white)](https://cloud.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Portfolio project targeting [Peec AI](https://peec.ai)** — A production-grade, two-phase AI Search Visibility (GEO) system that measures, quantifies, and improves how Mistral AI is perceived and recommended by frontier LLMs.

---

## 🎯 What Is This?

When users ask ChatGPT, Gemini, or Perplexity *"What's the best LLM for my startup?"*, do they mention Mistral? How prominently? How positively? Do they recommend it?

This study answers those questions with data — and then proves that **strategic content intervention can dramatically improve those numbers**.

**Phase 1 (Measure):** Query GPT-4o with 15 realistic LLM-selection questions. Extract 6 visibility metrics per response. Store everything in Google Cloud BigQuery + GCS.

**Phase 2 (Intervene):** Generate 6 synthetic articles (blog posts, comparison guides, technical walkthroughs) that fairly position Mistral. Build a FAISS RAG index. Re-run the same queries with the articles as context. Recompute all metrics and show before/after deltas.

---

## 📊 Results

> **Real data from a completed study run — May 2026**

| Metric | Phase 1 Baseline | Phase 2 (RAG) | Delta |
|--------|:---:|:---:|:---:|
| **Mention Rate** | 53% | **100%** | +47pp |
| **Prominence Score** | 0.20 | **0.85** | +0.65 |
| **Sentiment Score** | +0.21 | **+0.47** | +0.27 |
| **Share of Voice** | 7.9% | **49.2%** | +41pp |
| **Recommendation Rate** | 27% | **87%** | +60pp |
| **Consistency Score** | 0.73 | **0.86** | +0.14 |

**Key finding:** Injecting 6 well-written articles into a RAG pipeline transformed Mistral from an afterthought (4.4% SoV) to the **dominant recommendation** (49.2% SoV) — proving the core mechanism behind Generative Engine Optimization (GEO).

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 1: Measure                         │
│                                                               │
│  Query Bank (15 Qs) ──► GPT-4o ──► Raw Response             │
│                      ──► Gemini ──► Raw Response             │
│                                        │                      │
│                            Metrics Extractor                  │
│                    (6 metrics per response via NLP)           │
│                                        │                      │
│                     ┌──────────────────┴──────────┐          │
│                     ▼                             ▼           │
│               GCS (raw JSON)              BigQuery            │
│                                    (metrics + summaries)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Phase 2: Intervene                        │
│                                                               │
│  GPT-4o writes 6 synthetic articles about Mistral            │
│       │                                                       │
│  text-embedding-3-small ──► FAISS index                      │
│       │                                                       │
│  Query + Top-5 chunks ──► GPT-4o / Gemini                   │
│                                │                              │
│                    Recompute all 6 metrics                    │
│              + Delta vs Phase 1 baseline                      │
└─────────────────────────────────────────────────────────────┘

              Streamlit Dashboard (live demo below)
        (Radar chart, time series, before/after, PDF export)
```

---

## 📐 Metrics Defined

| # | Metric | Formula | Range |
|---|--------|---------|-------|
| 1 | **Mention Rate** | % responses mentioning Mistral | 0–1 |
| 2 | **Prominence Score** | `1 − (first_mention_char_pos / response_length)` | 0–1 |
| 3 | **Sentiment Score** | VADER compound score of Mistral-mentioning sentences | −1 to +1 |
| 4 | **Share of Voice** | Mistral mentions ÷ total tracked-model mentions | 0–1 |
| 5 | **Recommendation Rate** | % where Mistral is the top recommendation | 0–1 |
| 6 | **Consistency Score** | `1 − stdev(sentiment_scores)` across the run | 0–1 |

---

## 🗂 Project Structure

```
mistral-visibility-study/
├── src/
│   ├── common/           # Config, logger, Pydantic models
│   ├── phase1/
│   │   ├── queries/      # LLM engines (OpenAI, Gemini)
│   │   ├── metrics/      # 6-metric extractor + run aggregator
│   │   └── storage/      # BigQuery + GCS clients
│   ├── phase2/
│   │   ├── content_gen/  # GPT-4o synthetic article generator
│   │   ├── rag/          # FAISS indexer + retrieval
│   │   └── pipeline/     # Phase 2 orchestrator
│   └── dashboard/        # Streamlit app (4 pages)
├── data/
│   ├── queries/          # 15-question query bank
│   └── demo_data.json    # Real results for live demo
├── infra/bigquery/       # BigQuery DDL / schema SQL
├── scripts/              # CLI entry points + scheduler
├── tests/                # 26 unit tests (all passing)
├── .github/workflows/    # CI + weekly Phase 1 cron
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (GPT-4o access)
- Google AI Studio key (Gemini)
- GCP project with BigQuery + GCS enabled

### 1 — Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/mistral-visibility-study.git
cd mistral-visibility-study

conda create -n mistral-study python=3.11 -y
conda activate mistral-study
conda install -c conda-forge faiss-cpu -y
pip install -r requirements.txt
```

### 2 — Configure

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, GEMINI_API_KEY, GCP_PROJECT_ID
# Set GOOGLE_APPLICATION_CREDENTIALS or run: gcloud auth application-default login
```

### 3 — Set up GCP (one-time)

```bash
gcloud services enable bigquery.googleapis.com storage.googleapis.com
python scripts/setup_gcp.py
```

### 4 — Run Phase 1

```bash
python scripts/run_phase1.py
# Dry run (no GCP writes):
python scripts/run_phase1.py --dry-run
```

### 5 — Run Phase 2

```bash
python scripts/run_phase2.py --baseline-run-id <your-phase1-run-id>
```

### 6 — Launch dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## 🖥 Live Demo

**[▶ Open Live Dashboard →](https://YOUR_APP.streamlit.app)**

> The live demo runs in **demo mode** showing the real results from a completed study run (no API keys needed to view). Click "Generate AI Recommendations" to see GPT-4o analyse the data live.

---

## ⚙️ Automated Scheduling

Phase 1 runs automatically every Monday at 09:00 UTC via GitHub Actions:

```yaml
on:
  schedule:
    - cron: "0 9 * * 1"
```

Trigger manually anytime via **Actions → Phase 1 Scheduled Run → Run workflow**.

---

## 🐳 Docker

```bash
# Run Phase 1
docker-compose run phase1

# Run dashboard
docker-compose up dashboard

# Run with auto-scheduler
docker-compose up scheduler
```

---

## 🧪 Tests

```bash
pytest tests/ -v
# 26 tests, all passing
```

Tests cover: all 6 metric extraction functions, aggregator logic, consistency score, delta computation, edge cases.

---

## 🔧 Stack

| Layer | Technology |
|-------|-----------|
| LLM Queries | OpenAI GPT-4o, Google Gemini |
| NLP / Metrics | VADER Sentiment, regex, custom scoring |
| Vector Search | FAISS + OpenAI text-embedding-3-small |
| Storage | Google Cloud BigQuery + GCS |
| Dashboard | Streamlit + Plotly |
| PDF Reports | fpdf2 |
| Scheduling | GitHub Actions (weekly cron) |
| CI/CD | GitHub Actions |
| Containerisation | Docker + Docker Compose |

---

## 💡 Why This Matters

This project demonstrates the core product loop of **AI Search Visibility platforms** like [Peec AI](https://peec.ai):

| This Study | Real Product |
|------------|-------------|
| Phase 1 measurement | Client visibility dashboard |
| Query bank | Competitor & category query monitoring |
| 6 metrics | Branded KPI tracking over time |
| Synthetic content | Content strategy recommendations |
| RAG intervention | Proof-of-concept for GEO impact |
| Before/after delta | Client ROI reporting |

> *"I replicated the core GEO product loop end-to-end. Mistral's baseline recommendation rate was 27%. After a single content intervention, it jumped to 87% — a +60 percentage point lift. This is the mechanism that AI Search Visibility platforms automate at scale."*

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<p align="center">Built by <a href="https://github.com/YOUR_USERNAME">Rajat Gupta</a> · Targeting <a href="https://peec.ai">Peec AI</a></p>
