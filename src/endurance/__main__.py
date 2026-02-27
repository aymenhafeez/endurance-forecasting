from endurance.schema import init_db
from endurance.ingest_strava import ingest_all
from endurance.build_runs import build_runs
from endurance.build_features import build_daily_and_weekly
from endurance.build_labels import build_weekly_labels
from endurance.train_volume import train_eval


def main():
    init_db()
    ingest_all()
    build_runs()
    build_daily_and_weekly()
    build_weekly_labels()
    train_eval()


if __name__ == "__main__":
    main()
