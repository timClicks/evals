"""Lmsys Chatbot Scraping.

Module for downloading and processing model quality metrics based on human
preference using the lmsys chatbot arena leaderboard
https://chat.lmsys.org/?leaderboard.

@misc{chiang2024chatbot,
    title={Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference},
    author={Wei-Lin Chiang and Lianmin Zheng and Ying Sheng and Anastasios
    Nikolas Angelopoulos and Tianle Li and Dacheng Li and Hao Zhang and Banghua
    Zhu and Michael Jordan and Joseph E. Gonzalez and Ion Stoica},
    year={2024},
    eprint={2403.04132},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
"""

import asyncio
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import polars as pl
from loguru import logger

from .settings import get_settings

BASE_URL = (
    "https://huggingface.co/"
    "spaces/lmsys/chatbot-arena-leaderboard/resolve/main/{file_name}"
)


async def download_date(date: str):
    """Download files for a date (if not already present)."""
    downloads_dir = get_settings().get_downloads_dir("lmsys")
    async with httpx.AsyncClient() as client:
        for file_name in [f"elo_results_{date}.pkl", f"leaderboard_table_{date}.csv"]:
            # Account for differences in filename for this date
            if file_name == "leaderboard_table_20240403.csv":
                file_name = "leaderboard_table_20240404.csv"

            file_path = downloads_dir / file_name
            if file_path.exists():
                logger.info(f"File `{file_name}`: skippping as already exists")
                continue

            url = BASE_URL.format(file_name=file_name)
            # Currently, requests are being redirected...
            response = await client.get(url, follow_redirects=True, timeout=10)
            if response.status_code == 404:
                continue

            response.raise_for_status()

            logger.info(f"Downloading {file_name}")
            with file_path.open("wb") as file:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    file.write(chunk)


async def async_download():
    """Download data for all dates."""
    # Dates at https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard/tree/main
    dates = [
        "20230619",
        "20230717",
        "20230802",
        "20230905",
        "20231002",
        "20231108",
        "20231116",
        "20231206",
        "20231215",
        "20231220",
        "20240109",
        "20240118",
        "20240125",
        "20240202",
        "20240215",
        "20240305",
        "20240307",
        "20240313",
        "20240326",
        "20240403",
        "20240410",
        "20240411",
        "20240413",
        "20240418",
        "20240419",
        "20240422",
        "20240426",
        "20240501",
        "20240508",
        "20240515",
        "20240516",
        "20240519",
        "20240520",
        "20240527",
        "20240602",
        "20240606",
        "20240611",
        "20240617",
        "20240621",
        "20240623",
    ]

    # Start TaskGroup for downloading files concurrently
    async with asyncio.TaskGroup() as tg:
        # Add specific date downloads to TaskGroup
        for date in dates:
            tg.create_task(download_date(date))

        # Add dynamically generated dates to TaskGroup
        date = datetime(2024, 6, 24)
        while date <= datetime.now():
            tg.create_task(download_date(date.strftime("%Y%m%d")))
            date += timedelta(days=1)


def update_frame(df: pd.DataFrame, date: pd.Timestamp, kind: str) -> pd.DataFrame:
    df["date"] = date
    df["kind"] = kind

    # TODO: Move this out to a separate scoring function.
    df["score"] = (df["rating"] - df["rating"].min()) / (
        df["rating"].max() - df["rating"].min()
    )
    # This is missing in the early files and we don't need it.
    if "final_ranking" in df.columns:
        df = df.drop(columns=["final_ranking"])
    return df


def build_frame(
    date: pd.Timestamp,
    full: pd.DataFrame,
    coding: pd.DataFrame | None,
    vision: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = [update_frame(full, date, "full")]
    if coding is not None:
        frames.append(update_frame(coding, date, "coding"))
    if vision is not None:
        frames.append(update_frame(vision, date, "vision"))
    return pd.concat(frames)


def build_extract(file_name: Path) -> pd.DataFrame | None:
    """Load a pickle file and extract the leaderboard data frame.

    Both pandas and plotly are required to load the pickle files.
    """
    with file_name.open("rb") as file:
        data = pickle.load(file)

    logger.info(f"Extracting data from {file_name.name}")
    date = pd.to_datetime(file_name.name[-12:-4], format="%Y%m%d")

    def get_leaderboard_table_df(data_dict, key_paths):
        for keys in key_paths:
            d = data_dict
            try:
                for key in keys:
                    d = d[key]
                return d["leaderboard_table_df"]
            except (KeyError, TypeError):
                continue
        return None

    # Define possible key paths for 'full', 'coding', and 'vision' data
    # The different paths account for changes in the structure of the pickle files
    full_key_paths = [
        ("full",),
        ("text", "full"),
        (),
    ]

    coding_key_paths = [
        ("coding",),
        ("text", "coding"),
    ]

    vision_key_paths = [
        ("vision", "full"),
    ]

    # Attempt to retrieve data using the defined key paths
    full = get_leaderboard_table_df(data, full_key_paths)
    coding = get_leaderboard_table_df(data, coding_key_paths)
    vision = get_leaderboard_table_df(data, vision_key_paths)

    if full is None:
        if "leaderboard_table" in data:
            # In some early pkl files, no data frame was available
            return None
        else:
            # Otherwise, we have an error condition.
            msg = f"Keys of data in {file_name}: {list(data.keys())}"
            raise KeyError(msg)

    return build_frame(date, full, coding, vision)


def download():
    asyncio.run(async_download())


def assemble_frame() -> pl.DataFrame:
    """Iterate over all pickle files and get extracts."""
    settings = get_settings()
    downloads_dir = settings.get_downloads_dir("lmsys")
    paths = sorted(downloads_dir.glob("elo_results_*.pkl"))

    # Do this in parallel. It makes a big difference.
    with ProcessPoolExecutor() as executor:
        frames: list[pd.DataFrame] = [
            r for r in executor.map(build_extract, paths) if r is not None
        ]

    df = pl.concat(
        [pl.DataFrame(extract.reset_index(names=["model"])) for extract in frames]
    )
    return df


def assemble():
    df = assemble_frame()
    df.write_parquet(get_settings().get_frames_dir() / "lmsys.parquet")


def clean():
    download_dir = get_settings().get_downloads_dir("lmsys")
    shutil.rmtree(download_dir)


def all():
    download()
    assemble()
