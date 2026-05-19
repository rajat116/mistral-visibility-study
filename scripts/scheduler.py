"""
Scheduled runner — executes Phase 1 on a cron schedule.

Default: every Monday at 09:00 UTC (configurable via SCHEDULE_CRON env var).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import schedule
import time
from src.common.config import config
from src.common.logger import get_logger
from src.phase1.pipeline import run_phase1

logger = get_logger("scheduler")


def scheduled_job() -> None:
    logger.info("Scheduled Phase 1 run starting...")
    try:
        summary = run_phase1()
        logger.info(
            "Scheduled run complete | run_id=%s | mention_rate=%.3f",
            summary.run_id,
            summary.avg_mention_rate,
        )
    except Exception as exc:
        logger.error("Scheduled run failed: %s", exc, exc_info=True)


def parse_cron_to_schedule(cron: str) -> None:
    """
    Parse a simple cron expression and register with the `schedule` library.
    Supports: 'minute hour * * day_of_week' format.
    """
    parts = cron.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Unsupported cron format: {cron}")
    minute, hour, _, _, dow = parts

    day_names = {
        "0": "sunday", "1": "monday", "2": "tuesday", "3": "wednesday",
        "4": "thursday", "5": "friday", "6": "saturday",
        "7": "sunday",
    }
    time_str = f"{int(hour):02d}:{int(minute):02d}"

    if dow == "*":
        schedule.every().day.at(time_str).do(scheduled_job)
        logger.info("Scheduled daily at %s UTC", time_str)
    else:
        day = day_names.get(dow, "monday")
        getattr(schedule.every(), day).at(time_str).do(scheduled_job)
        logger.info("Scheduled every %s at %s UTC", day, time_str)


if __name__ == "__main__":
    logger.info("Scheduler starting | cron=%s", config.schedule_cron)
    parse_cron_to_schedule(config.schedule_cron)

    # Run once immediately on startup
    logger.info("Running initial Phase 1 pipeline on startup...")
    scheduled_job()

    while True:
        schedule.run_pending()
        time.sleep(60)
