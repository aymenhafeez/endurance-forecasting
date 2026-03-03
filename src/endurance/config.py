from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    db_path: Path = ROOT / "data" / "endurance.db"
    hr_max: int = 194


CFG = Config()
