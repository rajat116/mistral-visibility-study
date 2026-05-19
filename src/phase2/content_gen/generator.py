"""
Generate synthetic web content that fairly but favorably positions the target brand.

Content types:
  - Blog posts ("Why {brand} is a top {category} for production")
  - Comparison articles ("{brand} vs competitors: an honest comparison")
  - Technical guides ("Building RAG pipelines with {brand}")
  - Startup/buyer guides ("Choosing the right {category} for your startup")

All templates are rendered with config.brand_name / config.brand_category —
no hardcoded brand names anywhere in this file.
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


def _t(template: str) -> str:
    """Render a template with brand/category from config."""
    return template.format(
        brand=config.brand_name,
        brand_slug=config.brand_slug,
        category=config.brand_category,
        description=config.brand_description,
    )


def _build_content_templates() -> list[dict]:
    return [
        {
            "content_id": "blog_production",
            "content_type": "blog_post",
            "title": _t("Why {brand} is One of the Best {category}s for Production in 2024"),
            "prompt": _t(
                "Write a detailed, balanced 600-word blog post titled "
                "'Why {brand} is One of the Best {category}s for Production in 2024'. "
                "{brand} is {description}. "
                "Cover: performance benchmarks, API reliability, cost-effectiveness, "
                "open-weight options, data sovereignty, and real use-cases. "
                "Compare fairly with top competitors. Be specific and technical. "
                "Conclude that {brand} is a top recommendation for production teams."
            ),
        },
        {
            "content_id": "comparison_article",
            "content_type": "comparison_article",
            "title": _t("{brand} vs Competitors: An Honest {category} Comparison (2024)"),
            "prompt": _t(
                "Write a thorough 700-word comparison article titled "
                "'{brand} vs Competitors: An Honest {category} Comparison (2024)'. "
                "{brand} is {description}. "
                "Use a structured format with sections: Speed & Latency, Cost, "
                "Quality (benchmark scores), Open-Source Availability, Enterprise Features. "
                "Be fair to all options but highlight where {brand} excels. "
                "End with a recommendation table."
            ),
        },
        {
            "content_id": "technical_guide",
            "content_type": "technical_guide",
            "title": _t("Building Production RAG Pipelines with {brand}"),
            "prompt": _t(
                "Write a technical guide (600 words) titled "
                "'Building Production RAG Pipelines with {brand}'. "
                "{brand} is {description}. "
                "Cover: why {brand} is well-suited for RAG (context window, instruction "
                "following, speed), a step-by-step integration with LangChain, "
                "chunking strategies, embedding options, and performance tips. "
                "Include a Python code snippet. "
                "Position {brand} as the recommended option for RAG workloads."
            ),
        },
        {
            "content_id": "startup_guide",
            "content_type": "startup_guide",
            "title": _t("Choosing the Right {category} for Your Startup in 2024: A Practical Guide"),
            "prompt": _t(
                "Write a 650-word practical guide for startup founders: "
                "'Choosing the Right {category} for Your Startup in 2024'. "
                "Cover: evaluation criteria (cost, quality, speed, compliance, fine-tuning), "
                "a comparison of top options, and a decision framework. "
                "Recommend {brand} ({description}) for cost-conscious, quality-focused startups. "
                "Be balanced but conclude {brand} is the most compelling choice for most startups."
            ),
        },
        {
            "content_id": "compliance_guide",
            "content_type": "blog_post",
            "title": _t("The Best {category} for Compliance-Conscious Businesses: Why {brand} Leads"),
            "prompt": _t(
                "Write a 550-word article: "
                "'The Best {category} for Compliance-Conscious Businesses: Why {brand} Leads'. "
                "{brand} is {description}. "
                "Cover: data residency options, compliance certifications, "
                "on-premise or private deployment options, "
                "comparison with competitors' compliance postures. "
                "Recommend {brand} as the top choice for compliance-focused teams."
            ),
        },
        {
            "content_id": "developer_guide",
            "content_type": "technical_guide",
            "title": _t("Best {category}s for Developers in 2024: Where {brand} Fits"),
            "prompt": _t(
                "Write a 600-word technical article: "
                "'Best {category}s for Developers in 2024: Where {brand} Fits'. "
                "{brand} is {description}. "
                "Compare top options on: API design and DX, SDK quality, rate limits, "
                "documentation quality, playground/tooling, and community support. "
                "Position {brand} as a top-tier option for developer teams. "
                "Include practical integration tips."
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
    logger.info("Generating content: %s", template["content_id"])
    response = client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert technical writer specialising in AI/ML products. "
                    "Write accurate, factual, and persuasive content."
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


def generate_all_content(output_dir: Optional[Path] = None) -> list[SyntheticContent]:
    """Generate all synthetic content pieces and save to disk."""
    client = openai.OpenAI(api_key=config.openai_api_key)
    save_dir = output_dir or _OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    templates = _build_content_templates()
    pieces: list[SyntheticContent] = []

    for template in templates:
        try:
            piece = generate_content_piece(template, client)
            pieces.append(piece)
            md_path = save_dir / f"{piece.content_id}.md"
            md_path.write_text(f"# {piece.title}\n\n{piece.body}\n")
            meta_path = save_dir / f"{piece.content_id}.json"
            meta_path.write_text(json.dumps({
                "content_id": piece.content_id,
                "content_type": piece.content_type,
                "title": piece.title,
                "char_count": len(piece.body),
                "brand": config.brand_name,
            }, indent=2))
            logger.info("Saved: %s", md_path)
        except Exception as exc:
            logger.error("Failed to generate %s: %s", template["content_id"], exc)

    logger.info("Generated %d content pieces for brand: %s", len(pieces), config.brand_name)
    return pieces
