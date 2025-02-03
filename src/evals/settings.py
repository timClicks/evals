import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import ClassVar


@dataclass
class Settings:
    """Group settings together here."""

    app_name: ClassVar[str] = "stencila-evals"

    def get_base_dir(self) -> Path:
        """Allow for setting a custom base path."""
        base_path = os.environ.get("EVALS_BASE_PATH", "./data")
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def _get_or_create_dir(self, dirname: str, with_subdir: str | None = None) -> Path:
        """Creates a subdirectory of the project's base directory."""
        base = self.get_base_dir()
        dir = base / dirname
        if with_subdir:
            dir = dir / with_subdir
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def get_log_dir(self) -> Path:
        return self._get_or_create_dir("logs")
        # return Path(
        #     platformdirs.user_log_dir(appname=self.app_name, ensure_exists=True)
        # )

    def get_downloads_dir(self, project_name: str | None = None) -> Path:
        dir = self._get_or_create_dir("downloads", with_subdir=project_name)
        return dir

    def get_frames_dir(self) -> Path:
        dir = self._get_or_create_dir("frames")
        return dir

    def get_routing_dir(self) -> Path:
        dir = self._get_or_create_dir("routing")
        return dir

    def get_working_dir(self, project_name: str | None = None) -> Path:
        dir = self._get_or_create_dir("working", with_subdir=project_name)
        return dir

    def get_database_path(self) -> Path:
        pth = self.get_base_dir() / "database.sqlite"
        return pth


@cache
def get_settings() -> Settings:
    settings = Settings()
    return settings
