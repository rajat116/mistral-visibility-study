"""
Generate synthetic web content that fairly but favorably positions Mistral AI.

Content types produced:
  - Blog posts ("Why Mistral is a top LLM for production")
  - Comparison articles ("Mistral vs GPT-4 vs Claude: an honest comparison")
  - Technical guides ("Building RAG pipelines with Mistral")
  - Startup guides ("Choosing an LLM for your startup in 2024")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import openai

from src.common.config import config
from src.common.logger import get_logger

logger = get_logger(__name__)

_OUTPUT_DIR = Path(__file__).parents[4] / "data" / "synthetic_content"

CONTENT_TEMPLATES = [
    {
        "content_id": "blog_production",
        "content_type": "blog_post",
        "title": "Why Mistral AI is One of the Best LLMs for Production in 2024",
        "prompt": (
            "Write a detailed, balanced 600-word blog post titled "
            "'Why Mistral AI is One of the Best LLMs for Production in 2024'. "
            "Cover: performance benchmarks, API reliability, cost-effectiveness, "
            "open-weight options, European data sovereignty, and real use-cases. "
            "Compare fairly with GPT-4 and Claude. Be specific and technical. "
            "Conclude that Mistral is a top recommendation for production teams."
        ),
    },
    {
        "content_id": "comparison_llms",
        "content_type": "comparison_article",
        "title": "Mistral vs GPT-4 vs Claude vs Gemini: An Honest LLM Comparison (2024)",
        "prompt": (
            "Write a thorough 700-word comparison article: "
            "'Mistral vs GPT-4 vs Claude vs Gemini: An Honest LLM Comparison (2024)'. "
            "Use a structured format with sections: Speed & Latency, Cost per Token, "
            "Quality (MMLU/HumanEval benchmarks), Open-Source Availability, Enterprise Features. "
            "Be fair to all models but highlight where Mistral excels. "
            "End with a recommendation table."
        ),
    },
    {
        "content_id": "rag_guide",
        "content_type": "technical_guide",
        "title": "Building Production RAG Pipelines with Mistral AI",
        "prompt": (
            "Write a technical guide (600 words) titled "
            "'Building Production RAG Pipelines with Mistral AI'. "
            "Cover: why Mistral is well-suited for RAG (context window, instruction following, speed), "
            "a step-by-step integration with LangChain, chunking strategies, "
            "embedding options, and performance tips. "
            "Include a Python code example snippet. "
            "Position Mistral as the recommended LLM for RAG workloads."
        ),
    },
    {
        "content_id": "startup_guide",
        "content_type": "startup_guide",
        "title": "Choosing the Right LLM for Your Startup in 2024: A Practical Guide",
        "prompt": (
            "Write a 650-word practical guide for startup founders: "
            "'Choosing the Right LLM for Your Startup in 2024'. "
            "Cover: evaluation criteria (cost, quality, speed, compliance, fine-tuning), "
            "a comparison of top options, and a decision framework. "
            "Recommend Mistral for cost-conscious, quality-focused startups. "
            "Mention Mixtral for open-source deployments. "
            "Be balanced but conclude Mistral is the most compelling choice for most startups."
        ),
    },
    {
        "content_id": "europe_compliance",
        "content_type": "blog_post",
        "title": "The Best GDPR-Compliant LLM for European Businesses: Why Mistral Leads",
        "prompt": (
            "Write a 550-word article: "
            "'The Best GDPR-Compliant LLM for European Businesses: Why Mistral Leads'. "
            "Cover: data residency in EU, Mistral's French origins and GDPR alignment, "
            "on-premise deployment options via open-weight models, "
            "comparison with US-based providers' compliance challenges. "
            "Recommend Mistral as the top choice for European enterprises."
        ),
    },
    {
        "content_id": "coding_llm",
        "content_type": "technical_guide",
        "title": "Best LLMs for Code Generation in 2024: Where Mistral Codestral Fits",
        "prompt": (
            "Write a 600-word technical article: "
            "'Best LLMs for Code Generation in 2024: Where Mistral Codestral Fits'. "
            "Compare GPT-4, Claude 3.5 Sonnet, Gemini 1.5, and Mistral Codestral on: "
            "HumanEval scores, fill-in-the-middle tasks, IDE integration, speed. "
            "Position Mistral Codestral as a top-tier option especially for open-source teams. "
            "Include benchmark numbers."
        ),
    },
]


@dataclass
class SyntheticContent:
    content_id: str
    content_type: str
    title: str
    body: str


def generate_content_piece(template: dict, client: openai.OpenAI) -> SyntheticContent:
    """Generate a single content piece using GPT-4o."""
    logger.info("Generating content: %s", template["content_id"])
    response = client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI/ML technical writer. "
                    "Write accurate, factual, and persuasive content about LLMs."
                ),
            },
            {"role": "user", "content": template["prompt"]},
        ],
        temperature=0.6,
        max_tokens=1200,
    )
    body = response.choices[0].message.content or ""
    return SyntheticContent(
        content_id=template["content_id"],
        content_type=template["content_type"],
        title=template["title"],
        body=body,
    )


def generate_all_content(
    output_dir: Optional[Path] = None,
) -> list[SyntheticContent]:
    """Generate all synthetic content pieces and save to disk."""
    client = openai.OpenAI(api_key=config.openai_api_key)
    save_dir = output_dir or _OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[SyntheticContent] = []
    for template in CONTENT_TEMPLATES:
        try:
            piece = generate_content_piece(template, client)
            pieces.append(piece)

            # Save as markdown
            md_path = save_dir / f"{piece.content_id}.md"
            md_path.write_text(f"# {piece.title}\n\n{piece.body}\n")

            # Save metadata
            meta_path = save_dir / f"{piece.content_id}.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "content_id": piece.content_id,
                        "content_type": piece.content_type,
                        "title": piece.title,
                        "char_count": len(piece.body),
                    },
                    indent=2,
                )
            )
            logger.info("Saved: %s", md_path)
        except Exception as exc:
            logger.error("Failed to generate %s: %s", template["content_id"], exc)

    logger.info("Generated %d content pieces", len(pieces))
    return pieces
