from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar

import platformdirs


@dataclass
class Settings:
    """Group settings together here."""

    app_name: ClassVar[str] = "stencila-evals"

    def get_log_dir(self) -> Path:
        return Path(
            platformdirs.user_log_dir(appname=self.app_name, ensure_exists=True)
        )

    def get_downloads_dir(self, where: str | None = None) -> Path:
        pth = Path(platformdirs.user_data_dir(appname=self.app_name))
        if where is not None:
            pth = pth / where
        if not pth.exists():
            pth.mkdir(parents=True)
        return pth


@cache
def get_settings() -> Settings:
    settings = Settings()
    return settings
