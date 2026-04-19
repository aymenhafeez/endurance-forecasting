from .build_features import build_daily_and_weekly
from .build_labels import build_weekly_labels
from .build_runs import build_runs
from .ingest_strava import ingest_all
from .schema import init_db
from .train_risk import train_eval


def main():
    init_db()
    ingest_all()
    build_runs()
    build_daily_and_weekly()
    build_weekly_labels()
    train_eval()


if __name__ == "__main__":
    main()
