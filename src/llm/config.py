from __future__ import annotations

import os
import re
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    model: str = "gpt-5.6-luna"
    timeout_seconds: float = Field(default=20, ge=1, le=60)
    max_output_tokens: int = Field(default=1200, ge=128, le=4000)
    budget_usd: float = Field(default=1.0, gt=0, le=20, allow_inf_nan=False)
    max_calls: int = Field(default=100, ge=1, le=10000)
    ledger_path: Path = ROOT / ".local" / "mentor-usage.sqlite3"

    @field_validator("model")
    @classmethod
    def model_name(cls, value: str) -> str:
        if not re.fullmatch(r"gpt-[a-z0-9.-]{1,60}", value):
            raise ValueError("invalid model name")
        return value

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> "Settings":
        from dotenv import dotenv_values
        values = dict(dotenv_values(env_file, interpolate=False)) if env_file else {}
        values.update(os.environ)
        names = {
            "OPENAI_API_KEY": "api_key", "OPENAI_MODEL": "model",
            "MENTOR_TIMEOUT_SECONDS": "timeout_seconds", "MENTOR_MAX_OUTPUT_TOKENS": "max_output_tokens",
            "MENTOR_BUDGET_USD": "budget_usd", "MENTOR_MAX_CALLS": "max_calls",
            "MENTOR_LEDGER_PATH": "ledger_path",
        }
        return cls(**{target: values[source] for source, target in names.items() if values.get(source)})
