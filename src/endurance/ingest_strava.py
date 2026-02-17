import json
from datetime import datetime, timezone
import requests

from .db import connect
from .config import CFG

STRAVA_API = "https://www.strava.com/api/v3"


def fetch_activities(page: int = 1, per_page: int = 200):
    if not CFG.strava_access_token:
        raise RuntimeError("Set STRAVA_ACCESS_TOKEN environment variable")

    headers = {"Authorisation": f"Bearer {CFG.strava_access_token}"}
    params = {"page": page, "per_page": per_page}
    r = requests.get(
        f"{STRAVA_API}/athlete/activities", headers=headers, params=params, timeout=30
    )

    # raise HTTP error if it occurs
    r.raise_for_status()

    # return decoded json object
    return r.json()


def upsert_raw(activities: list[dict]):
    now = datetime.now(timezone.utc).isoformat()

    with connect() as con:
        for act in activities:
            con.execute(
                """
                INSERT OR REPLACE INTO activities_raw(activity_id, start_date_utc, type, raw_json, ingested_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(act["id"]),
                    act.get("start_date", ""),
                    act.get("type", ""),
                    json.dumps(act),
                    now,
                ),
            )
