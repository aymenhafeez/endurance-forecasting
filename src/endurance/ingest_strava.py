import json
import os
from datetime import datetime, timezone

import requests

from .config import CFG
from .db import connect

STRAVA_API = "https://www.strava.com/api/v3"
STRAVA_OAUTH = "https://www.strava.com/oauth/token"


def refresh_access_token() -> str:
    client_id = os.getenv("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")

    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "Missing Strava OAuth env vars. Set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN."
        )

    r = requests.post(
        STRAVA_OAUTH,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"Could not refresh token: {r.text}")
    return token


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


def ingest_all(max_pages: int = 20):
    total = 0

    for page in range(1, max_pages + 1):
        acts = fetch_activities(page=page)

        if not acts:
            break
        upsert_raw(acts)
        total += len(acts)
        print(f"Fetched page {page}: {len(acts)} (total {total})")

    print("Done")


if __name__ == "__main__":
    ingest_all()
