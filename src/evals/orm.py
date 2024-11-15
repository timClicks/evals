# Ignore Meta overrides for Tortoise models
# pyright: reportIncompatibleVariableOverride=false

from pathlib import Path

from tortoise import Tortoise, fields
from tortoise.models import Model


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


class Prompt(Model):
    id = fields.IntField(pk=True)
    created_by = fields.CharField(max_length=255)
    created_at = fields.DatetimeField(auto_now_add=True)
    modified_at = fields.DatetimeField(auto_now=True)
    type = fields.CharField(max_length=255)
    name = fields.CharField(max_length=255)

    # Reverse relations
    model_prompt_scores: fields.ReverseRelation["ModelPromptScore"]

    class Meta:
        table = "prompts"


class Model(Model):
    id = fields.IntField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    modified_at = fields.DatetimeField(auto_now=True)
    provider = fields.CharField(max_length=255)
    identifier = fields.CharField(max_length=255)
    name = fields.CharField(max_length=255)
    version = fields.CharField(max_length=255)

    # Reverse relations
    model_scores: fields.ReverseRelation["ModelScore"]
    model_prompt_scores: fields.ReverseRelation["ModelPromptScore"]

    class Meta:
        table = "models"


class ModelScore(Model):
    id = fields.IntField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    model = fields.ForeignKeyField("models.Model", related_name="model_scores")
    overall_score = fields.FloatField()
    cost_score = fields.FloatField()
    speed_score = fields.FloatField()

    class Meta:
        table = "model_scores"


class ModelPromptScore(Model):
    id = fields.IntField(pk=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    model = fields.ForeignKeyField("models.Model", related_name="model_prompt_scores")
    prompt = fields.ForeignKeyField("models.Prompt", related_name="model_prompt_scores")
    quality_score = fields.FloatField()

    class Meta:
        table = "model_prompt_scores"


# Generate schemas
async def init_db():
    """Initialize database and create schemas"""
    from tortoise import Tortoise

    await Tortoise.init(
        db_url="sqlite://db.sqlite3",  # Update this with your database URL
        modules={"models": ["__main__"]},
    )
    await Tortoise.generate_schemas()
