"""CLI entry-point to run the Phase 2 pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.phase2.pipeline.run_phase2 import run_phase2
import argparse

parser = argparse.ArgumentParser(description="Phase 2: RAG-augmented Mistral visibility")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--run-id", type=str, default=None)
parser.add_argument("--baseline-run-id", type=str, default=None, help="Phase 1 run ID to compute deltas against")
parser.add_argument("--regenerate-content", action="store_true")
parser.add_argument("--rebuild-index", action="store_true")
args = parser.parse_args()

result = run_phase2(
    run_id=args.run_id,
    baseline_run_id=args.baseline_run_id,
    dry_run=args.dry_run,
    regenerate_content=args.regenerate_content,
    rebuild_index=args.rebuild_index,
)
print("\n=== RUN SUMMARY ===")
print(result.model_dump_json(indent=2))
