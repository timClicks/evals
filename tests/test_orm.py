from pathlib import Path

from evals.orm import LLM, Database, PromptRecord, load_tables


async def test_creation(tmp_path: Path):
    db_path = tmp_path / "test.sqlite"
    await load_tables(db_path)

    # Check we have rows.
    async with Database(db_path):
        ps = await PromptRecord.all()
        assert ps
        ms = await LLM.all()
        assert ms
