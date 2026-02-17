from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    db_path: Path = Path("data/endurance.db")
    hr_max: int = 194

    strava_access_token: str | None = os.getenv("STRAVA_ACCESS_TOKEN")


CFG = Config()
