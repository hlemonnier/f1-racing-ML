"""F1ML package exposes CLI utilities for data ingestion, feature engineering, and modeling."""

from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from project-level .env if present (e.g., API keys).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

__all__ = ["config", "io", "fastf1_ingest"]
