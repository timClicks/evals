"""Aider AI leaderboard data extraction.

https://aider.chat/docs/leaderboards/
"""
# TODO: Currently incomplete.

import httpx
import polars as pl
import yaml
from loguru import logger

from .settings import get_settings

BASE_URL = "https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/"
EDIT_URL = BASE_URL + "edit_leaderboard.yml"


def download():
    """Download the leaderboard data."""
    downloads_dir = get_settings().get_downloads_dir("aider")
    logger.info(f"Downloading leaderboard data to {downloads_dir}")
    response = httpx.get(EDIT_URL, follow_redirects=True)
    response.raise_for_status()  # Ensure we got a successful response

    # Load the YAML content and convert to parquet.
    # We rely on Polars to do something smart.
    data = yaml.safe_load(response.text)
    df = pl.DataFrame(data)
    df.write_parquet(downloads_dir / "leaderboard.parquet")


def extract():
    pass
