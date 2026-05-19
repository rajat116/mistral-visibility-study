-- BigQuery schema definitions
-- Project: project-553727f2-fa51-4dd8-8e0
-- Dataset: mistral_visibility

-- Create dataset
CREATE SCHEMA IF NOT EXISTS `project-553727f2-fa51-4dd8-8e0.mistral_visibility`
OPTIONS(location = 'US');

-- Raw LLM responses
CREATE TABLE IF NOT EXISTS `project-553727f2-fa51-4dd8-8e0.mistral_visibility.raw_responses` (
  record_id         STRING NOT NULL,
  run_id            STRING NOT NULL,
  phase             STRING NOT NULL,
  llm_engine        STRING NOT NULL,
  query_id          STRING NOT NULL,
  query_text        STRING,
  response_text     STRING,
  prompt_tokens     INT64,
  completion_tokens INT64,
  latency_ms        FLOAT64,
  rag_context_used  BOOL,
  created_at        TIMESTAMP
);

-- Per-response visibility metrics
CREATE TABLE IF NOT EXISTS `project-553727f2-fa51-4dd8-8e0.mistral_visibility.visibility_metrics` (
  metric_id               STRING NOT NULL,
  record_id               STRING NOT NULL,
  run_id                  STRING NOT NULL,
  phase                   STRING NOT NULL,
  llm_engine              STRING NOT NULL,
  query_id                STRING NOT NULL,
  mention_rate            FLOAT64,
  prominence_score        FLOAT64,
  sentiment_score         FLOAT64,
  share_of_voice          FLOAT64,
  recommendation_rate     FLOAT64,
  consistency_score       FLOAT64,
  mistral_mention_count   INT64,
  total_model_mentions    INT64,
  first_mention_position  FLOAT64,
  top_recommended_model   STRING,
  models_mentioned        STRING,   -- JSON array
  created_at              TIMESTAMP
);

-- Run-level aggregated summaries
CREATE TABLE IF NOT EXISTS `project-553727f2-fa51-4dd8-8e0.mistral_visibility.run_summaries` (
  run_id                   STRING NOT NULL,
  phase                    STRING NOT NULL,
  started_at               TIMESTAMP,
  finished_at              TIMESTAMP,
  total_queries            INT64,
  total_responses          INT64,
  avg_mention_rate         FLOAT64,
  avg_prominence_score     FLOAT64,
  avg_sentiment_score      FLOAT64,
  avg_share_of_voice       FLOAT64,
  avg_recommendation_rate  FLOAT64,
  avg_consistency_score    FLOAT64,
  per_engine_metrics       STRING,   -- JSON object
  delta_mention_rate       FLOAT64,
  delta_prominence_score   FLOAT64,
  delta_sentiment_score    FLOAT64,
  delta_share_of_voice     FLOAT64,
  delta_recommendation_rate FLOAT64
);

-- Useful analytical view: latest run per phase
CREATE OR REPLACE VIEW `project-553727f2-fa51-4dd8-8e0.mistral_visibility.v_latest_run_per_phase` AS
SELECT *
FROM `project-553727f2-fa51-4dd8-8e0.mistral_visibility.run_summaries`
QUALIFY ROW_NUMBER() OVER (PARTITION BY phase ORDER BY finished_at DESC) = 1;

-- Useful view: metric trend over time
CREATE OR REPLACE VIEW `project-553727f2-fa51-4dd8-8e0.mistral_visibility.v_metric_trend` AS
SELECT
  run_id,
  phase,
  finished_at,
  avg_mention_rate,
  avg_prominence_score,
  avg_sentiment_score,
  avg_share_of_voice,
  avg_recommendation_rate,
  avg_consistency_score
FROM `project-553727f2-fa51-4dd8-8e0.mistral_visibility.run_summaries`
ORDER BY finished_at;
