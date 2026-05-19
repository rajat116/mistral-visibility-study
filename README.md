# 🔍 AI Search Visibility Study — Plug & Play GEO Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)](https://openai.com)
[![GCP](https://img.shields.io/badge/GCP-BigQuery%20%2B%20GCS-4285F4?style=flat&logo=googlecloud&logoColor=white)](https://cloud.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://mistral-visibility-study-zhy6pz25gapa2gphryfqox.streamlit.app/)

> **Measure your brand's visibility in AI-generated answers. Understand where you're missing. Fix it. Track the results.**

When users ask ChatGPT, Perplexity, or Gemini *"What's the best tool for X?"* — does your product get mentioned? How prominently? How positively? Does the LLM recommend it first?

This platform answers those questions with data, and proves that targeted content intervention can dramatically improve your AI search ranking.

**Works for any brand in any category.** Change 6 lines in `.env` and the entire platform — queries, metrics, content, dashboard, PDF reports — adapts to your product.

---

## 🌐 Live Demo

**[▶ Open Interactive Dashboard](https://mistral-visibility-study-zhy6pz25gapa2gphryfqox.streamlit.app/)**

The demo runs a complete study on **Mistral AI** as an example brand — real metrics, real before/after deltas, AI-generated recommendations, and a downloadable PDF report.

When you clone this repo and set `BRAND_NAME=YourProduct` in `.env`, the dashboard shows your brand's data instead.

---

## 🎯 Example Results (Mistral AI, May 2026)

| Metric | Before (Baseline) | After (RAG Intervention) | Delta |
|--------|:-----------------:|:------------------------:|:-----:|
| **Mention Rate** | 53% | **100%** | +47pp |
| **Prominence Score** | 0.20 | **0.85** | +0.65 |
| **Sentiment Score** | +0.21 | **+0.47** | +0.27 |
| **Share of Voice** | 7.9% | **49.2%** | +41pp |
| **Recommendation Rate** | 27% | **87%** | +60pp |
| **Consistency Score** | 0.73 | **0.86** | +0.14 |

**[▶ View Live Demo Dashboard →](https://mistral-visibility-study-zhy6pz25gapa2gphryfqox.streamlit.app/)**
> The demo shows a real study run on **Mistral AI** as an example brand.
> When you deploy with your own brand config, the dashboard shows your data, your metrics, and AI recommendations tailored to your product.

---

## ⚡ Quick Start — 3 Steps

### Step 1 — Configure your brand

```bash
cp .env.example .env
```

Edit these 6 lines for your product:

```env
BRAND_NAME=Acme Database          # Your product name
BRAND_SLUG=acme                   # Lowercase, no spaces
BRAND_ALIASES=acme,acme db        # All names it goes by
BRAND_DESCRIPTION=a fast, open-source time-series database
BRAND_CATEGORY=time-series database
COMPETITOR_NAMES=influxdb,timescaledb,prometheus,clickhouse,questdb
```

Then add your API key:

```env
OPENAI_API_KEY=sk-proj-...
```

### Step 2 — Install & run

```bash
conda create -n visibility python=3.11 -y && conda activate visibility
conda install -c conda-forge faiss-cpu -y
pip install -r requirements.txt

python scripts/run_phase1.py    # measure baseline visibility
```

### Step 3 — View results

```bash
streamlit run src/dashboard/app.py
```

That's it. You now have:
- 6 visibility metrics measured across 15 queries
- A live dashboard with charts and insights
- A downloadable PDF report with AI-generated recommendations

---

## 🏗 How It Works

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1 — MEASURE                                            │
│                                                               │
│  15 query templates     ──► rendered with your brand/category │
│  (e.g. "best {category} for startups?")                       │
│                                │                              │
│            GPT-4o + Gemini answer each query                  │
│                                │                              │
│         6-metric NLP extractor runs on each response          │
│    (mention, prominence, sentiment, SoV, rec rate, consistency)│
│                                │                              │
│         BigQuery + GCS  ◄──────┘                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  PHASE 2 — INTERVENE                                          │
│                                                               │
│  GPT-4o generates 6 brand-favorable articles                  │
│  (blog post, comparison, technical guide, startup guide, ...)  │
│                │                                              │
│  FAISS vector index  (text-embedding-3-small)                 │
│                │                                              │
│  Same 15 queries + top-5 retrieved chunks → LLM              │
│                │                                              │
│  Recompute all 6 metrics  →  before/after delta               │
└──────────────────────────────────────────────────────────────┘

                    Streamlit Dashboard
      Executive Summary · Metric Trends · Before/After Radar
      Response Explorer · AI Recommendations · PDF Export
```

---

## 📐 The 6 Visibility Metrics

| # | Metric | What It Measures | Range |
|---|--------|-----------------|-------|
| 1 | **Mention Rate** | % of LLM responses that mention your brand | 0–1 |
| 2 | **Prominence Score** | How early your brand appears (`1 − char_position / length`) | 0–1 |
| 3 | **Sentiment Score** | VADER sentiment of sentences mentioning your brand | −1 to +1 |
| 4 | **Share of Voice** | Your mentions ÷ total competitor mentions | 0–1 |
| 5 | **Recommendation Rate** | % where your brand is the top recommendation | 0–1 |
| 6 | **Consistency Score** | `1 − stdev(sentiments)` — narrative stability across LLMs | 0–1 |

---

## 📂 Project Structure

```
├── src/
│   ├── common/           # Config (brand settings), logger, Pydantic models
│   ├── phase1/
│   │   ├── queries/      # Query templates + GPT-4o / Gemini engines
│   │   ├── metrics/      # 6-metric extractor + run aggregator
│   │   └── storage/      # BigQuery + GCS clients
│   ├── phase2/
│   │   ├── content_gen/  # Brand-aware article generator
│   │   ├── rag/          # FAISS indexer + chunk retrieval
│   │   └── pipeline/     # Phase 2 orchestrator
│   └── dashboard/        # Streamlit app (4 pages + PDF)
├── data/
│   ├── queries/          # 15 query templates (use {brand}, {category})
│   └── demo_data.json    # Real results for dashboard demo mode
├── scripts/              # CLI entry points + weekly scheduler
├── tests/                # 26 unit tests
└── .github/workflows/    # CI + weekly Phase 1 cron
```

---

## 🔧 Brand Configuration Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `BRAND_NAME` | Full product/brand name | `Supabase` |
| `BRAND_SLUG` | Lowercase slug for regex | `supabase` |
| `BRAND_ALIASES` | All names brand is known by | `supabase,supa` |
| `BRAND_DESCRIPTION` | One-liner for content generation | `an open-source Firebase alternative` |
| `BRAND_CATEGORY` | Market category for query templates | `backend-as-a-service platform` |
| `COMPETITOR_NAMES` | Comma-separated competitor list | `firebase,planetscale,neon,railway` |

### More brand examples

<details>
<summary>SaaS API Tool (e.g. Postman)</summary>

```env
BRAND_NAME=Postman
BRAND_SLUG=postman
BRAND_ALIASES=postman,postman api
BRAND_DESCRIPTION=an API platform for building and testing APIs
BRAND_CATEGORY=API testing and development tool
COMPETITOR_NAMES=insomnia,bruno,hoppscotch,paw,thunderclient,swagger
```
</details>

<details>
<summary>Vector Database (e.g. Qdrant)</summary>

```env
BRAND_NAME=Qdrant
BRAND_SLUG=qdrant
BRAND_ALIASES=qdrant
BRAND_DESCRIPTION=a high-performance open-source vector database
BRAND_CATEGORY=vector database
COMPETITOR_NAMES=pinecone,weaviate,chroma,milvus,faiss,pgvector
```
</details>

<details>
<summary>Cloud Provider (e.g. Render)</summary>

```env
BRAND_NAME=Render
BRAND_SLUG=render
BRAND_ALIASES=render,render.com
BRAND_DESCRIPTION=a cloud platform for hosting web apps and APIs
BRAND_CATEGORY=cloud hosting platform
COMPETITOR_NAMES=heroku,railway,fly.io,vercel,netlify,digitalocean
```
</details>

---

## 🗓 Automated Weekly Tracking

Phase 1 runs automatically every Monday at 09:00 UTC via GitHub Actions. Set these secrets in your repo:

| Secret | Value |
|--------|-------|
| `OPENAI_API_KEY` | Your OpenAI key |
| `GCP_PROJECT_ID` | Your GCP project |
| `GCS_BUCKET_NAME` | Your GCS bucket |
| `BQ_DATASET` | Your BigQuery dataset |
| `GCP_SA_KEY` | Service account JSON |

Trigger manually anytime: **Actions → Phase 1 Scheduled Run → Run workflow**

---

## 💾 GCP Setup (Optional — for persistent storage)

Without GCP, the dashboard runs in **demo mode** using `data/demo_data.json`. To enable live BigQuery storage:

```bash
# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable bigquery.googleapis.com storage.googleapis.com

# Create dataset and tables
python scripts/setup_gcp.py
```

---

## 🐳 Docker

```bash
# Run Phase 1
docker-compose run phase1

# Launch dashboard
docker-compose up dashboard

# Run with auto-scheduler (weekly cron)
docker-compose up scheduler
```

---

## 🧪 Tests

```bash
pytest tests/ -v        # 26 tests, all passing
pytest tests/ --cov=src # with coverage report
```

---

## 📄 License

MIT — use freely, modify, deploy for your company or clients.

---

## 💡 What Is GEO?

**Generative Engine Optimization (GEO)** is the practice of improving a brand's visibility in AI-generated answers — analogous to SEO but for LLMs.

When ChatGPT, Perplexity, or Gemini answer product recommendation questions, they pull from:
- Their training data
- Real-time web search results
- Retrieved documents (RAG)

Brands that publish high-quality, LLM-readable content about their products gain disproportionate share in AI-generated recommendations. This platform **measures** that share, **simulates** the impact of content intervention, and helps you **track improvement over time**.

---

<p align="center">
  Built by <a href="https://github.com/rajat116">Rajat Gupta</a>
</p>
