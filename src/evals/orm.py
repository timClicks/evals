# Ignore Meta overrides for Tortoise models
# pyright: reportIncompatibleVariableOverride=false

from pathlib import Path

from tortoise import Model, Tortoise, fields

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_CSV = BASE_DIR / "tables" / "model.csv"
PROMPT_CSV = BASE_DIR / "tables" / "prompt.csv"


async def init_connection(db_url: str):
    await Tortoise.init(db_url=db_url, modules={"models": ["evals.orm"]})
    await Tortoise.generate_schemas()


async def close_connection():
    await Tortoise.close_connections()


class Database:
    """Helper class to manage connection to the database."""

    def __init__(self, path: Path):
        self.url = f"sqlite://{path}"

    async def __aenter__(self):
        await init_connection(self.url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb  # prevent unused warnings.
        await close_connection()


class PromptRecord(Model):
    id = fields.IntField(pk=True, generated=True)
    name = fields.TextField()
    version = fields.TextField()
    category = fields.TextField()

    # Reverse relations
    model_prompt_scores: fields.ReverseRelation["ModelPromptScore"]

    class Meta:
        table = "prompt"


class ModelRecord(Model):
    """LLM model (We can't use Model!)."""

    id = fields.IntField(pk=True)
    name = fields.TextField()

    # Reverse relations
    model_scores: fields.ReverseRelation["ModelScoreRecord"]
    model_prompt_scores: fields.ReverseRelation["ModelPromptScore"]

    class Meta:
        table = "model"


class ModelScoreRecord(Model):
    id = fields.IntField(pk=True)
    model = fields.ForeignKeyField("models.ModelRecord", related_name="model_scores")
    cost_score = fields.FloatField()
    speed_score = fields.FloatField()

    class Meta:
        table = "model_score"


class ModelPromptScore(Model):
    id = fields.IntField(pk=True)
    model = fields.ForeignKeyField(
        "models.ModelRecord", related_name="model_prompt_scores"
    )
    prompt = fields.ForeignKeyField(
        "models.PromptRecord", related_name="model_prompt_scores"
    )
    quality_score = fields.SmallIntField()

    class Meta:
        table = "model_prompt_scores"
