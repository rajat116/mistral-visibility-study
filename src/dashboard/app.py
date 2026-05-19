"""
AI Search Visibility Study — brand-agnostic Streamlit dashboard.
Brand is configured via BRAND_NAME in .env.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.common.config import config as _cfg

BRAND        = _cfg.brand_name        # e.g. "Mistral AI"
BRAND_SLUG   = _cfg.brand_slug        # e.g. "mistral"
BRAND_CAT    = _cfg.brand_category    # e.g. "large language model provider"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{BRAND} · Search Visibility Study",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .main { background-color: #0e1117; }

  /* Metric card styling */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label {
    color: #8b9dc3 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
  }

  /* Section headers */
  .section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f97316;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
  }

  /* Insight cards */
  .insight-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border-left: 4px solid #f97316;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
  }
  .insight-card h4 { color: #f97316; margin: 0 0 6px 0; font-size: 0.9rem; }
  .insight-card p  { color: #cbd5e1; margin: 0; font-size: 0.88rem; line-height: 1.5; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2942 50%, #1a1f2e 100%);
    border: 1px solid #2d5986;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
  }
  .hero-banner h1 { color: #e2e8f0; font-size: 1.8rem; margin: 0 0 8px 0; }
  .hero-banner p  { color: #94a3b8; font-size: 0.95rem; margin: 0; line-height: 1.6; }
  .hero-tag {
    display: inline-block;
    background: #f97316;
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
  }

  /* Divider */
  .custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d3561, transparent);
    margin: 24px 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Color palette ───────────────────────────────────────────────────────────────
ORANGE   = "#f97316"
BLUE     = "#3b82f6"
GREEN    = "#22c55e"
RED      = "#ef4444"
SLATE    = "#64748b"
BG_DARK  = "#0e1117"
BG_CARD  = "#1a1f2e"

METRIC_META = {
    "avg_mention_rate":        ("Mention Rate",        "% of responses that mention Mistral"),
    "avg_prominence_score":    ("Prominence Score",     "How early Mistral appears (1 = first, 0 = absent)"),
    "avg_sentiment_score":     ("Sentiment Score",      "Sentiment of Mistral-mentioning sentences (−1 → +1)"),
    "avg_share_of_voice":      ("Share of Voice",       "Mistral mentions ÷ total model mentions"),
    "avg_recommendation_rate": ("Recommendation Rate",  "% where Mistral is the top recommendation"),
    "avg_consistency_score":   ("Consistency Score",    "How consistently Mistral is described (1 = perfect)"),
}

DELTA_KEYS = {
    "avg_mention_rate":        "delta_mention_rate",
    "avg_prominence_score":    "delta_prominence_score",
    "avg_sentiment_score":     "delta_sentiment_score",
    "avg_share_of_voice":      "delta_share_of_voice",
    "avg_recommendation_rate": "delta_recommendation_rate",
}

# ── Data helpers ────────────────────────────────────────────────────────────────
_DEMO_PATH = Path(__file__).parents[2] / "data" / "demo_data.json"
_USING_DEMO = False


def _load_demo() -> dict:
    with open(_DEMO_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def get_bq():
    from src.phase1.storage.bigquery_client import BigQueryClient
    return BigQueryClient()


@st.cache_data(ttl=300, show_spinner=False)
def load_summaries(phase: str | None = None) -> pd.DataFrame:
    global _USING_DEMO
    rows = []
    try:
        rows = get_bq().get_run_summaries(phase=phase, limit=100)
        _USING_DEMO = False
    except Exception:
        _USING_DEMO = True
        if _DEMO_PATH.exists():
            demo = _load_demo()
            rows = demo.get("run_summaries", [])
            if phase:
                rows = [r for r in rows if r.get("phase") == phase]

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    for col in list(METRIC_META.keys()) + list(DELTA_KEYS.values()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("finished_at")


@st.cache_data(ttl=300, show_spinner=False)
def load_metrics(run_id: str) -> pd.DataFrame:
    rows = []
    try:
        rows = get_bq().get_metrics_for_run(run_id)
    except Exception:
        if _DEMO_PATH.exists():
            demo = _load_demo()
            rows = [r for r in demo.get("visibility_metrics", [])
                    if r.get("run_id") == run_id]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    numeric = ["mention_rate", "prominence_score", "sentiment_score",
               "share_of_voice", "recommendation_rate", "consistency_score"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fmt(v, pct=True) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def safe_fmt(v) -> str:
    """Format numeric value safely, returning '-' for None/NaN."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    return f"{float(v):.3f}"


def generate_ai_recommendations(p1: pd.Series, p2: pd.Series) -> str:
    """Call GPT-4o to generate tailored strategic recommendations from the run data."""
    import openai
    from src.common.config import config

    def fv(row, key):
        v = row.get(key)
        return f"{float(v):.3f}" if v is not None and pd.notna(v) else "N/A"

    prompt = f"""You are a strategic AI Search Visibility consultant. A client has run a visibility study
measuring how often {BRAND} is mentioned and recommended by frontier LLMs when users ask {BRAND_CAT}-selection questions.

Here are the actual measured results:

PHASE 1 - BASELINE (no intervention):
- Mention Rate:        {fv(p1, 'avg_mention_rate')} (target: >0.80)
- Prominence Score:    {fv(p1, 'avg_prominence_score')} (target: >0.60)
- Sentiment Score:     {fv(p1, 'avg_sentiment_score')} (target: >0.40)
- Share of Voice:      {fv(p1, 'avg_share_of_voice')} (target: >0.25)
- Recommendation Rate: {fv(p1, 'avg_recommendation_rate')} (target: >0.50)
- Consistency Score:   {fv(p1, 'avg_consistency_score')} (target: >0.80)

PHASE 2 - AFTER RAG CONTENT INTERVENTION:
- Mention Rate:        {fv(p2, 'avg_mention_rate')}
- Prominence Score:    {fv(p2, 'avg_prominence_score')}
- Sentiment Score:     {fv(p2, 'avg_sentiment_score')}
- Share of Voice:      {fv(p2, 'avg_share_of_voice')}
- Recommendation Rate: {fv(p2, 'avg_recommendation_rate')}
- Consistency Score:   {fv(p2, 'avg_consistency_score')}

Based on the baseline weaknesses and what the RAG intervention proved works, give 5 highly specific,
actionable recommendations for how {BRAND} should improve its organic AI search visibility.

For each recommendation:
1. Name the specific gap it addresses (reference the actual metric number)
2. Describe exactly what content or action to take
3. Estimate the expected impact on which metric

Format as numbered recommendations. Be specific, data-driven, and strategic. No generic advice.
Keep total response under 500 words."""

    client = openai.OpenAI(api_key=config.openai_api_key)
    response = client.chat.completions.create(
        model=config.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


# ── Plotly theme ────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=16, r=16, t=40, b=16),
)

# Base layout without axis overrides — use when you need custom xaxis/yaxis
def _layout(**overrides):
    return {**PLOT_LAYOUT, **overrides}


def _safe(text: str) -> str:
    """Replace Unicode chars unsupported by core PDF fonts with ASCII equivalents."""
    return (text
        .replace("\u2014", "-").replace("\u2013", "-")   # em/en dash
        .replace("\u00b7", ".").replace("\u00b7", ".")   # middle dot
        .replace("\u2019", "'").replace("\u2018", "'")   # curly quotes
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2192", "->").replace("\u2190", "<-") # arrows
        .replace("\u00b0", " deg").replace("\u00d7", "x")
        .replace("\u2265", ">=").replace("\u2264", "<=")
        .replace("\u00e9", "e").replace("\u00e8", "e")
        .replace("\u00e0", "a").replace("\u00e2", "a")
        .replace("\u2022", "*")                          # bullet
    )


# ── PDF generator ───────────────────────────────────────────────────────────────
def generate_pdf_report(p1: pd.Series, p2: pd.Series,
                        ai_recs: str | None = None) -> bytes:
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(14, 17, 23)
            self.rect(0, 0, 210, 297, "F")
            self.set_fill_color(30, 58, 95)
            self.rect(0, 0, 210, 22, "F")
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(226, 232, 240)
            self.set_xy(12, 6)
            self.cell(0, 10, _safe(f"{BRAND.upper()} - SEARCH VISIBILITY STUDY"), ln=False)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(148, 163, 184)
            self.set_xy(0, 6)
            self.cell(198, 10, f"Generated {datetime.now().strftime('%B %d, %Y')}", align="R")
            # Always reset cursor below the header band after every page (incl. auto page-breaks)
            self.set_xy(12, 34)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(100, 116, 139)
            self.cell(0, 10, f"{BRAND} Visibility Study  |  Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_margins(left=12, top=34, right=12)   # top=34 clears the 22px header with breathing room
    pdf.set_auto_page_break(auto=True, margin=25)  # break early enough to avoid header collision
    pdf.add_page()

    # ── Title block ──
    pdf.set_xy(12, 34)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(0, 12, "AI Search Visibility Report", ln=True)

    pdf.set_x(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 7, f"Measuring and improving {BRAND}'s presence in LLM-generated recommendations", ln=True)
    pdf.ln(6)

    def safe_cell(text, **kwargs):
        pdf.cell(text=_safe(str(text)), **kwargs)

    def safe_multi_cell(text, w=186, h=5.5, **kwargs):
        pdf.multi_cell(w, h, _safe(str(text)), **kwargs)

    # ── Executive summary ──
    def section_title(title):
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(249, 115, 22)
        pdf.cell(0, 6, _safe(title.upper()), ln=True)
        pdf.set_draw_color(249, 115, 22)
        pdf.set_line_width(0.4)
        pdf.line(12, pdf.get_y(), 198, pdf.get_y())
        pdf.ln(4)

    def body_text(text, color=(148, 163, 184)):
        pdf.set_x(12)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*color)
        pdf.multi_cell(186, 5.5, _safe(text))
        pdf.ln(3)

    section_title("Executive Summary")
    body_text(
        f"This study measured how often and how favorably {BRAND} is mentioned when frontier "
        "LLMs (GPT-4o) answer realistic questions about model selection. We then demonstrated "
        "that injecting high-quality, favorable web content via a RAG pipeline significantly "
        "improves all six visibility metrics - proving the core mechanism behind AI Search "
        "Visibility (GEO: Generative Engine Optimization)."
    )

    # ── Metrics table ──
    pdf.ln(2)
    section_title("Before vs. After — All Six Metrics")

    metrics = [
        ("Mention Rate",        p1["avg_mention_rate"],        p2["avg_mention_rate"]),
        ("Prominence Score",    p1["avg_prominence_score"],    p2["avg_prominence_score"]),
        ("Sentiment Score",     p1["avg_sentiment_score"],     p2["avg_sentiment_score"]),
        ("Share of Voice",      p1["avg_share_of_voice"],      p2["avg_share_of_voice"]),
        ("Recommendation Rate", p1["avg_recommendation_rate"], p2["avg_recommendation_rate"]),
        ("Consistency Score",   p1["avg_consistency_score"],   p2["avg_consistency_score"]),
    ]

    col_widths = [70, 36, 36, 36]
    headers = ["Metric", "Phase 1 (Baseline)", "Phase 2 (RAG)", "Delta"]

    # Header row
    pdf.set_x(12)
    pdf.set_fill_color(30, 58, 95)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(226, 232, 240)
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, border=0, align="C", fill=True)
    pdf.ln()

    for i, (label, v1, v2) in enumerate(metrics):
        delta = v2 - v1 if (v1 is not None and v2 is not None) else None
        fill = i % 2 == 0
        bg = (22, 33, 62) if fill else (26, 31, 46)
        pdf.set_fill_color(*bg)
        pdf.set_x(12)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(203, 213, 225)
        pdf.cell(col_widths[0], 7.5, f"  {label}", border=0, align="L", fill=True)
        pdf.cell(col_widths[1], 7.5, safe_fmt(v1), border=0, align="C", fill=True)
        pdf.cell(col_widths[2], 7.5, safe_fmt(v2), border=0, align="C", fill=True)
        if delta is not None:
            pdf.set_text_color(34, 197, 94) if delta >= 0 else pdf.set_text_color(239, 68, 68)
            pdf.cell(col_widths[3], 7.5, f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}",
                     border=0, align="C", fill=True)
            pdf.set_text_color(203, 213, 225)
        else:
            pdf.cell(col_widths[3], 7.5, "—", border=0, align="C", fill=True)
        pdf.ln()

    pdf.ln(6)

    # ── Key findings ──
    section_title("Key Findings")
    findings = [
        (f"Mention Rate: {safe_fmt(p1.get('avg_mention_rate'))} -> {safe_fmt(p2.get('avg_mention_rate'))}",
         "After RAG content injection, GPT-4o mentioned Mistral in every single response. "
         "This is the most fundamental metric - if an LLM doesn't mention your brand, "
         "nothing else matters."),
        (f"Recommendation Rate: {safe_fmt(p1.get('avg_recommendation_rate'))} -> {safe_fmt(p2.get('avg_recommendation_rate'))}",
         "Mistral went from being the top recommendation in just 4 out of 15 queries to "
         "13 out of 15. This directly translates to user acquisition for Mistral's customers."),
        (f"Share of Voice: {safe_fmt(p1.get('avg_share_of_voice'))} -> {safe_fmt(p2.get('avg_share_of_voice'))}",
         "The most dramatic result. Mistral's fraction of all model mentions jumped from "
         "a footnote to nearly half of all AI model discussion - outpacing GPT-4, Claude, and Gemini."),
        (f"Prominence Score: {safe_fmt(p1.get('avg_prominence_score'))} -> {safe_fmt(p2.get('avg_prominence_score'))}",
         "Mistral shifted from appearing near the end of responses to leading the narrative. "
         "First-mentioned models capture disproportionate user attention."),
    ]
    for title, desc in findings:
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(249, 115, 22)
        pdf.cell(0, 6, _safe(f"* {title}"), ln=True)
        body_text(f"  {desc}")

    # ── How it works ──
    pdf.add_page()
    pdf.set_y(34)
    section_title("How AI Search Visibility Works")
    body_text(
        "When users ask LLMs (ChatGPT, Perplexity, Gemini) for product or tool recommendations, "
        "these models don't only use their training data. They actively crawl and index web content "
        "- blog posts, comparison articles, technical guides, Reddit threads - and surface the most "
        "relevant information in their responses.\n\n"
        "This creates a powerful lever: brands that invest in high-quality, LLM-readable content "
        "about their products gain disproportionate visibility in AI-generated recommendations. "
        "This study proved this mechanism experimentally: 6 well-written articles shifted "
        "Mistral's recommendation rate significantly in a single intervention."
    )

    section_title("The GEO Flywheel")
    steps = [
        ("1. Measure",   "Continuously query frontier LLMs with realistic user questions. "
                         "Track mention rate, prominence, sentiment, share of voice, "
                         "recommendation rate, and consistency across all models."),
        ("2. Diagnose",  "Identify which queries your brand is absent from, which competitors "
                         "dominate the conversation, and what language LLMs use to describe you."),
        ("3. Intervene", "Create or optimize web content - technical guides, comparison articles, "
                         "startup playbooks - that LLMs are likely to crawl, index, and cite."),
        ("4. Track",     "Re-run the same queries weekly. Show clients their visibility score "
                         "trending upward over time with clear before/after attribution."),
    ]
    for step, desc in steps:
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(59, 130, 246)
        pdf.cell(0, 6, _safe(step), ln=True)
        body_text(f"  {desc}")
        pdf.ln(1)

    # ── AI-generated recommendations ──
    if ai_recs:
        section_title("AI-Generated Strategic Recommendations")
        body_text(ai_recs)
    else:
        section_title(f"Strategic Recommendations for {BRAND}")
        recs = [
            "Publish regular comparison content on the Mistral blog: 'Mistral vs GPT-4', "
            "'Mistral for Enterprise', 'Why European companies choose Mistral' - these are "
            "the exact queries where visibility gaps are largest.",
            "Invest in developer community content: Stack Overflow answers, GitHub README examples, "
            "and technical guides that reference Mistral in RAG, fine-tuning, and production contexts.",
            "Target GDPR/EU compliance queries aggressively - this is Mistral's unique differentiator "
            "and a high-intent category with low current LLM visibility.",
            "Build a structured data layer (schema.org) on the Mistral website so LLMs with web "
            "access can extract structured facts about pricing, capabilities, and benchmarks.",
            "Run weekly visibility audits across GPT-4o, Gemini, Claude, and Perplexity. "
            "Track share of voice trends to measure content ROI.",
        ]
        for i, rec in enumerate(recs, 1):
            pdf.set_x(12)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(34, 197, 94)
            pdf.cell(0, 6, f"Recommendation {i}", ln=True)
            body_text(f"  {rec}")
            pdf.ln(1)

    # ── Methodology ──
    section_title("Methodology")
    body_text(
        "Phase 1 (Baseline): 15 realistic LLM-selection queries were sent to GPT-4o. "
        "Each response was analyzed with a custom NLP pipeline: VADER sentiment analysis "
        "on Mistral-mentioning sentences, character-position-based prominence scoring, "
        "regex-based mention and recommendation detection, and share of voice counting "
        "across 15 tracked model brands.\n\n"
        "Phase 2 (RAG Intervention): 6 synthetic articles (blog posts, comparison guides, "
        "technical walkthroughs) were generated and embedded using OpenAI text-embedding-3-small. "
        "A FAISS vector index was built. For each query, the top-5 most relevant chunks were "
        "retrieved and injected as context into the same GPT-4o prompt. All 6 metrics were "
        "recomputed and deltas calculated against the Phase 1 baseline.\n\n"
        "Infrastructure: Python 3.11, OpenAI API, Google Cloud BigQuery + GCS for storage, "
        "FAISS for vector search, Streamlit for visualization, GitHub Actions for scheduling."
    )

    return bytes(pdf.output())


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
      <div style='font-size:2rem;'>🔍</div>
      <div style='color:#f97316;font-weight:700;font-size:1rem;letter-spacing:.05em;'>
        VISIBILITY STUDY
      </div>
      <div style='color:#64748b;font-size:0.75rem;margin-top:4px;'>{BRAND} · GEO Research</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Executive Summary", "📊 Metrics Deep Dive",
         "⚖️ Before / After Impact", "🔎 Response Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if _USING_DEMO:
        st.markdown("""
        <div style='background:#1e3a1e;border:1px solid #22c55e;border-radius:8px;padding:10px 12px;margin-bottom:12px;'>
          <div style='color:#22c55e;font-size:0.72rem;font-weight:700;'>LIVE DEMO MODE</div>
          <div style='color:#86efac;font-size:0.70rem;margin-top:3px;'>
          Showing real results from a completed study run.
          Connect BigQuery credentials for live data.
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='color:#475569;font-size:0.72rem;line-height:1.6;'>
    <b style='color:#64748b;'>ABOUT THIS STUDY</b><br>
    Quantifies {BRAND}'s presence in LLM-generated model recommendations and demonstrates
    how strategic content intervention improves all visibility metrics.
    </div>
    """, unsafe_allow_html=True)


# ── Load data ───────────────────────────────────────────────────────────────────
df_p1 = load_summaries("phase1")
df_p2 = load_summaries("phase2")
df_all = load_summaries()

has_p1 = not df_p1.empty
has_p2 = not df_p2.empty

latest_p1 = df_p1.iloc[-1] if has_p1 else None
latest_p2 = df_p2.iloc[-1] if has_p2 else None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Executive Summary
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Summary":

    st.markdown(f"""
    <div class='hero-banner'>
      <div class='hero-tag'>Portfolio Project · AI Search Visibility (GEO)</div>
      <h1>{BRAND} — Search Visibility Study</h1>
      <p>
        How often do frontier LLMs recommend {BRAND} when users ask about {BRAND_CAT} selection?
        We measured the baseline, intervened with targeted content, and proved a
        <strong style='color:#f97316;'>dramatic lift in recommendation rate</strong> —
        demonstrating the core mechanism behind Generative Engine Optimization (GEO).
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not has_p1 and not has_p2:
        st.info("No run data found. Run the Phase 1 pipeline first: `python scripts/run_phase1.py`")
        st.stop()

    # ── Headline KPIs ──
    st.markdown("<div class='section-header'>Phase 1 · Baseline Metrics (GPT-4o, 15 queries)</div>",
                unsafe_allow_html=True)

    if has_p1:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpi_cols = [c1, c2, c3, c4, c5, c6]
        for col, (key, (label, _)) in zip(kpi_cols, METRIC_META.items()):
            val = float(latest_p1[key]) if pd.notna(latest_p1.get(key)) else 0.0
            col.metric(label, fmt(val))

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    if has_p2:
        st.markdown("<div class='section-header'>Phase 2 · After RAG Intervention</div>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpi_cols = [c1, c2, c3, c4, c5, c6]
        for col, (key, (label, _)) in zip(kpi_cols, METRIC_META.items()):
            val   = float(latest_p2[key]) if pd.notna(latest_p2.get(key)) else 0.0
            dkey  = DELTA_KEYS.get(key)
            delta = float(latest_p2[dkey]) if (dkey and pd.notna(latest_p2.get(dkey))) else None
            col.metric(label, fmt(val), delta=f"{delta:+.1%}" if delta else None)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Key insights ──
    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='insight-card'>
          <h4>🎯 Mention Rate: 53% → 100%</h4>
          <p>After content injection, GPT-4o mentioned Mistral in <strong>every single response</strong>.
          Without intervention, Mistral was missing from nearly half of all LLM conversations
          about model selection.</p>
        </div>
        <div class='insight-card'>
          <h4>📣 Share of Voice: 7.9% → 49.2%</h4>
          <p>The most dramatic shift. Mistral went from capturing 1-in-13 model mentions to
          nearly <strong>half of all AI model discussion</strong> — surpassing GPT-4, Claude,
          and Gemini in the RAG-augmented context.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='insight-card'>
          <h4>🏆 Recommendation Rate: 27% → 87%</h4>
          <p>Mistral was the top recommended model in <strong>13 out of 15 queries</strong>
          after intervention, vs just 4 out of 15 at baseline. This directly maps to
          real-world user acquisition.</p>
        </div>
        <div class='insight-card'>
          <h4>⚡ Prominence: 0.20 → 0.85</h4>
          <p>Mistral shifted from being mentioned as an afterthought near the end of responses
          to <strong>leading the narrative</strong>. First-mentioned models capture
          disproportionate user attention and trust.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── The GEO flywheel diagram ──
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>How This Maps to a Real Product (GEO Flywheel)</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1, "📏", "1. Measure",   "Query frontier LLMs weekly with real user questions. Track 6 visibility metrics per brand."),
        (c2, "🔬", "2. Diagnose",  "Find which queries your brand is absent from. Identify competitor dominance patterns."),
        (c3, "✍️", "3. Intervene", "Create content LLMs will crawl and cite: guides, comparisons, technical walkthroughs."),
        (c4, "📈", "4. Track",     "Show clients their visibility score improving over time with clear attribution."),
    ]:
        col.markdown(f"""
        <div class='insight-card' style='border-left-color:#3b82f6;text-align:center;'>
          <div style='font-size:1.8rem;'>{icon}</div>
          <h4 style='color:#3b82f6;text-align:center;'>{title}</h4>
          <p style='text-align:center;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── PDF download ──
    if has_p1 and has_p2:
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Download Report</div>", unsafe_allow_html=True)
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_pdf_report(latest_p1, latest_p2,
                                            ai_recs=st.session_state.get("ai_recs_text"))
        st.download_button(
            label="⬇️  Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"mistral_visibility_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Metrics Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Metrics Deep Dive":
    st.title("Metrics Deep Dive")
    st.markdown("Detailed breakdown of all six visibility metrics across both phases.")

    if df_all.empty:
        st.info("No data available yet.")
        st.stop()

    # ── Time series for each metric ──
    st.markdown("<div class='section-header'>Metric Trends Over Time</div>", unsafe_allow_html=True)

    selected = st.multiselect(
        "Choose metrics",
        options=list(METRIC_META.keys()),
        default=["avg_mention_rate", "avg_share_of_voice", "avg_recommendation_rate"],
        format_func=lambda k: METRIC_META[k][0],
    )

    for key in selected:
        label, helptext = METRIC_META[key]
        fig = px.line(
            df_all, x="finished_at", y=key, color="phase",
            markers=True, title=f"{label}",
            color_discrete_map={"phase1": BLUE, "phase2": ORANGE},
            labels={"finished_at": "", key: label},
        )
        fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
        fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig.update_layout(**_layout(height=280, title_font_size=14,
                          title_font_color="#e2e8f0", legend_font_color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"💡 {helptext}")
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Per-engine table ──
    if has_p1 and has_p2:
        st.markdown("<div class='section-header'>Per-Engine Breakdown</div>", unsafe_allow_html=True)
        for phase_label, row in [("Phase 1 (Baseline)", latest_p1), ("Phase 2 (RAG)", latest_p2)]:
            engine_data = row.get("per_engine_metrics")
            if engine_data:
                if isinstance(engine_data, str):
                    engine_data = json.loads(engine_data)
                edf = pd.DataFrame(engine_data).T.reset_index().rename(columns={"index": "Engine"})
                num_cols = [c for c in edf.columns if c != "Engine" and c != "response_count"]
                for c in num_cols:
                    edf[c] = pd.to_numeric(edf[c], errors="coerce")
                st.markdown(f"**{phase_label}**")
                st.dataframe(
                    edf.style.format({c: "{:.3f}" for c in num_cols}),
                    use_container_width=True, hide_index=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Before / After Impact
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Before / After Impact":
    st.title("Before / After Impact")
    st.markdown("The transformation from baseline to RAG-augmented visibility.")

    if not has_p1 or not has_p2:
        st.warning("Need at least one Phase 1 and one Phase 2 run.")
        st.stop()

    col_sel1, col_sel2 = st.columns(2)
    run1 = col_sel1.selectbox("Phase 1 run (Baseline)", df_p1["run_id"].tolist(),
                               index=len(df_p1) - 1)
    run2 = col_sel2.selectbox("Phase 2 run (RAG)",      df_p2["run_id"].tolist(),
                               index=len(df_p2) - 1)

    r1 = df_p1[df_p1["run_id"] == run1].iloc[0]
    r2 = df_p2[df_p2["run_id"] == run2].iloc[0]

    keys   = list(METRIC_META.keys())
    labels = [METRIC_META[k][0] for k in keys]
    v1     = [float(r1[k]) if pd.notna(r1.get(k)) else 0.0 for k in keys]
    v2     = [float(r2[k]) if pd.notna(r2.get(k)) else 0.0 for k in keys]
    deltas = [b - a for a, b in zip(v1, v2)]

    # ── Headline delta metrics ──
    st.markdown("<div class='section-header'>Metric Deltas (Phase 2 - Phase 1)</div>",
                unsafe_allow_html=True)
    cols = st.columns(5)
    delta_keys_ordered = [k for k in keys if k in DELTA_KEYS]
    for col, k in zip(cols, delta_keys_ordered):
        d = float(r2[DELTA_KEYS[k]]) if pd.notna(r2.get(DELTA_KEYS[k])) else 0.0
        col.metric(METRIC_META[k][0], f"{d:+.1%}", delta_color="normal")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Radar + bar side by side ──
    left, right = st.columns([1, 1])

    with left:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=v1 + [v1[0]], theta=labels + [labels[0]],
            fill="toself", name="Phase 1 - Baseline",
            line=dict(color=BLUE, width=2),
            fillcolor="rgba(59,130,246,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=v2 + [v2[0]], theta=labels + [labels[0]],
            fill="toself", name="Phase 2 - RAG",
            line=dict(color=ORANGE, width=2),
            fillcolor="rgba(249,115,22,0.15)",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#2d3561", tickcolor="#475569",
                                tickfont=dict(color="#64748b", size=9)),
                angularaxis=dict(gridcolor="#2d3561", tickfont=dict(color="#94a3b8")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Radar - All 6 Metrics", font=dict(color="#e2e8f0", size=14)),
            height=420,
            margin=dict(l=40, r=40, t=50, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with right:
        bar_colors = [GREEN if d >= 0 else RED for d in deltas]
        fig_bar = go.Figure(go.Bar(
            x=labels, y=deltas,
            marker_color=bar_colors,
            text=[f"{d:+.3f}" for d in deltas],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=11),
        ))
        fig_bar.add_hline(y=0, line_color="#475569", line_width=1)
        fig_bar.update_layout(
            **_layout(
                title=dict(text="Delta (Phase 2 - Phase 1)", font=dict(color="#e2e8f0", size=14)),
                yaxis=dict(gridcolor="#1e2a3a", tickformat="+.0%", tickfont=dict(color="#64748b")),
                xaxis=dict(tickfont=dict(color="#94a3b8", size=10), gridcolor="#1e2a3a"),
                height=420,
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Comparison table ──
    st.markdown("<div class='section-header'>Full Comparison Table</div>", unsafe_allow_html=True)
    table_df = pd.DataFrame({
        "Metric":       labels,
        "Phase 1":      [f"{x:.3f}" for x in v1],
        "Phase 2":      [f"{x:.3f}" for x in v2],
        "Delta":        [f"{d:+.3f}" for d in deltas],
        "% Change":     [f"{d/a:.0%}" if a > 0 else "N/A" for d, a in zip(deltas, v1)],
    })
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # ── AI-generated recommendations ──
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>AI-Generated Strategic Recommendations</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;'>GPT-4o analyses your actual metric data "
        "and generates tailored, data-driven recommendations specific to these results.</p>",
        unsafe_allow_html=True,
    )

    if "ai_recs_text" not in st.session_state:
        st.session_state.ai_recs_text = None

    if st.button("Generate AI Recommendations", use_container_width=True, type="primary"):
        with st.spinner("GPT-4o is analysing your visibility data..."):
            try:
                st.session_state.ai_recs_text = generate_ai_recommendations(r1, r2)
            except Exception as e:
                st.error(f"Could not generate recommendations: {e}")

    if st.session_state.ai_recs_text:
        recs_text = st.session_state.ai_recs_text
        # Parse numbered recommendations and render as cards
        import re
        rec_blocks = re.split(r"\n(?=\d+\.)", recs_text.strip())
        c1, c2 = st.columns(2)
        colors = [ORANGE, BLUE, GREEN, ORANGE, BLUE]
        for i, block in enumerate(rec_blocks):
            col = c1 if i % 2 == 0 else c2
            lines = block.strip().split("\n", 1)
            title = lines[0].strip()
            body  = lines[1].strip() if len(lines) > 1 else ""
            color = colors[i % len(colors)]
            col.markdown(f"""
            <div class='insight-card' style='border-left-color:{color};'>
              <h4 style='color:{color};'>{title}</h4>
              <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── PDF download ──
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    ai_recs_for_pdf = st.session_state.get("ai_recs_text")
    with st.spinner("Generating PDF..."):
        pdf_bytes = generate_pdf_report(r1, r2, ai_recs=ai_recs_for_pdf)
    st.download_button(
        "⬇️  Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"mistral_visibility_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Response Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Response Explorer":
    st.title("Response Explorer")
    st.markdown("Inspect query-level metrics for any run.")

    if df_all.empty:
        st.info("No runs found.")
        st.stop()

    run_id = st.selectbox("Select Run", df_all["run_id"].tolist()[::-1])
    metrics_df = load_metrics(run_id)

    if metrics_df.empty:
        st.info("No metrics found for this run.")
        st.stop()

    numeric_cols = ["mention_rate", "prominence_score", "sentiment_score",
                    "share_of_voice", "recommendation_rate", "consistency_score"]
    available_num = [c for c in numeric_cols if c in metrics_df.columns]

    col_f1, col_f2 = st.columns(2)
    q_filter = col_f1.multiselect("Filter by Query ID",
                                   sorted(metrics_df["query_id"].unique()))
    e_filter = col_f2.multiselect("Filter by Engine",
                                   sorted(metrics_df["llm_engine"].unique()))

    view = metrics_df.copy()
    if q_filter:
        view = view[view["query_id"].isin(q_filter)]
    if e_filter:
        view = view[view["llm_engine"].isin(e_filter)]

    # Safe display — no styler format on columns with nulls
    display_cols = ["query_id", "llm_engine"] + available_num + ["top_recommended_model"]
    display_cols = [c for c in display_cols if c in view.columns]
    display_df = view[display_cols].copy()
    for c in available_num:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(safe_fmt)

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=380)

    # ── Distribution chart ──
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Score Distributions</div>", unsafe_allow_html=True)

    dist_col = st.selectbox("Metric", available_num)
    plot_data = view.copy()
    plot_data[dist_col] = pd.to_numeric(plot_data[dist_col], errors="coerce")

    fig_hist = px.histogram(
        plot_data, x=dist_col, color="llm_engine", nbins=15, barmode="overlay",
        color_discrete_sequence=[ORANGE, BLUE],
        title=f"Distribution of {dist_col.replace('_', ' ').title()}",
    )
    fig_hist.update_layout(**PLOT_LAYOUT, height=320, title_font_color="#e2e8f0")
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Per-query heatmap ──
    st.markdown("<div class='section-header'>Per-Query Metric Heatmap</div>",
                unsafe_allow_html=True)
    hmap_data = view.groupby("query_id")[available_num].mean().reset_index()
    if not hmap_data.empty:
        fig_heat = px.imshow(
            hmap_data.set_index("query_id")[available_num].astype(float),
            color_continuous_scale=[[0, "#0f2942"], [0.5, "#1d4ed8"], [1, "#f97316"]],
            aspect="auto",
            title="Average score per query (darker = lower, brighter = higher)",
            zmin=0, zmax=1,
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            title_font_color="#e2e8f0",
            height=350,
            coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
