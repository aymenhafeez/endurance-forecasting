import sqlite3
from contextlib import contextmanager
from .config import CFG


@contextmanager
def connect():
    CFG.db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CFG.db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")

    try:
        yield con
        con.commit()
    finally:
        con.close()
