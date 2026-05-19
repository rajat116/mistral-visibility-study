"""CLI entry-point to run the Phase 1 pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.phase1.pipeline import run_phase1
import argparse

parser = argparse.ArgumentParser(description="Phase 1: Measure Mistral visibility")
parser.add_argument("--dry-run", action="store_true", help="Skip GCP writes")
parser.add_argument("--run-id", type=str, default=None)
args = parser.parse_args()

result = run_phase1(run_id=args.run_id, dry_run=args.dry_run)
print("\n=== RUN SUMMARY ===")
print(result.model_dump_json(indent=2))
