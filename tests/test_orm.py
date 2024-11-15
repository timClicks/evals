from pathlib import Path

import polars as pl

from evals.orm import Database, Model


async def test_creation(tmp_path: Path, models_csv: Path):
    db_path = tmp_path / "test.sqlite"
    db_path = Path("./test.sqlite")
    db = Database(db_path)
    models_df = pl.read_csv(models_csv)

    async with db:
        for row_dict in models_df.iter_rows(named=True):
            model = await Model.create(**row_dict)
            await model.save()
    assert db_path.exists()
