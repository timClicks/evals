# Ignore Meta overrides for Tortoise models
# pyright: reportIncompatibleVariableOverride=false

from enum import StrEnum, auto
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


class Category(StrEnum):
    Code = auto()
    Text = auto()
    All = auto()


class Provider(StrEnum):
    Aider = auto()
    LmSys = auto()


class LLM(Model):
    pass


class Quality(Model):
    """Records of LLM quality."""

    id = fields.IntField(pk=True)

    # Provider of the metrics
    provider = fields.CharEnumField(Provider)

    # Name of the model
    model = fields.TextField()
    # model_version = fields.TextField()
    # model_author = fields.TextField()

    # Date metric was calculated
    # when = fields.DateField()

    # Category of quality
    category = fields.CharEnumField(Category)

    # additional info, model specific.
    # subcategory = fields.TextField(null=True)

    # Normalized values between 0 and 100
    score = fields.FloatField()

    # snapshot: fields.ForeignKeyRelation[LLMSnapshotRecord] = fields.ForeignKeyField(
    #     "models.LLMSnapshotRecord", related_name="stats"
    # )

    class Meta:
        table = "quality"
