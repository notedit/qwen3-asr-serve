"""Env-driven runtime config (pydantic-settings)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ServeMode = Literal["asr", "aligner", "both"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Mode ──────────────────────────────────────────────────────────
    mode: ServeMode = "both"

    # ── Models ────────────────────────────────────────────────────────
    model_dir: Path = Path("./models")
    asr_model_path: Path = Path("./models/Qwen3-ASR-1.7B")
    aligner_model_path: Path = Path("./models/Qwen3-ForcedAligner-0.6B")

    # ── Performance ──────────────────────────────────────────────────
    gpu_mem_util: float = 0.6
    asr_batch: int = 256
    aligner_batch: int = 16

    # ── HTTP ──────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── File-path input ───────────────────────────────────────────────
    allow_local_paths: bool = False
    # comma-separated; access the parsed list via `allowed_path_prefixes_list`
    allowed_path_prefixes: str = ""
    max_file_bytes: int = 200 * 1024 * 1024

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "info"

    @property
    def allowed_path_prefixes_list(self) -> List[str]:
        return [p.strip() for p in self.allowed_path_prefixes.split(",") if p.strip()]

    @field_validator("gpu_mem_util")
    @classmethod
    def _check_mem(cls, v):
        if not 0.05 <= v <= 0.95:
            raise ValueError("GPU_MEM_UTIL must be in [0.05, 0.95]")
        return v


settings = Settings()
