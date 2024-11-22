from enum import StrEnum

from pydantic import BaseModel, Field


class BenchmarkType(StrEnum):
    """Kind of score."""

    COST = "cost"
    SPEED = "speed"
    QUALITY = "quality"


class ModelScore(BaseModel):
    """Model score."""

    # Model name. Must match the name in the model table.
    model: str

    # This could be:
    # - a category if it is quality score.
    # - a host if it is cost or speed score.
    # - "*" is a catchall that indicates everything (all hosts or any category).
    context: str = "*"

    # Normalized to 0-1
    score: float = Field(ge=0.0, le=1.0)


class BenchmarkResult(BaseModel):
    """Benchmark result."""

    bm_type: BenchmarkType
    scores: list[ModelScore]


# async def assign_scores(path: Path):
#     """Assign scores to models."""
#
#     # TODO: Actually assign scores. We're just filling the table for now.
#     await load_tables(path)
#
#     async with Database(path):
#         prompts = await Prompt.all()
#         models = await LLM.all()
#         for model in models:
#             for prompt in prompts:
#                 prompt_score = ModelPromptScore(
#                     model=model, prompt=prompt, quality_score=150
#                 )
#                 await prompt_score.save()
#             score = ModelScore(model=model, cost_score=50, speed_score=50)
#             await score.save()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("database", help="sqlite file to output")
#     args = parser.parse_args()
#     pth = Path(args.database)
#     if pth.exists():
#         pth.unlink()
#     asyncio.run(assign_scores(pth))
#
#
# if __name__ == "__main__":
#     main()
