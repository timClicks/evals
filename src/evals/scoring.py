import argparse
import asyncio
from pathlib import Path

from .orm import LLM, Database, ModelPromptScore, ModelScore, Prompt, load_tables


async def assign_scores(path: Path):
    """Assign scores to models."""

    # TODO: Actually assign scores. We're just filling the table for now.
    await load_tables(path)

    async with Database(path):
        prompts = await Prompt.all()
        models = await LLM.all()
        for model in models:
            for prompt in prompts:
                prompt_score = ModelPromptScore(
                    model=model, prompt=prompt, quality_score=150
                )
                await prompt_score.save()
            score = ModelScore(model=model, cost_score=50, speed_score=50)
            await score.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", help="sqlite file to output")
    args = parser.parse_args()
    pth = Path(args.database)
    if pth.exists():
        pth.unlink()
    asyncio.run(assign_scores(pth))


if __name__ == "__main__":
    main()
