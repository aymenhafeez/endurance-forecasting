# Endurance Forecasting

Time-aware modelling of training load and next-week running volume using Strava data.

To run go to the [autorisation
link](https://www.strava.com/oauth/authorize?client_id=CLIENT_ID>&response_type=code&redirect_uri=http://localhost/exchange_token&approval_prompt=force&scope=activity:read_all)
but with `CLIENT_ID` filled in. Copy and save the code=<CODE> from the
authorization link. Set the following environment variables, for example in fish
shell:

```
set -x STRAVA_CLIENT_ID <CLIENT_ID>
set -x STRAVA_CLIENT_SECRET <CLIENT_SECRENT>
set -x STRAVA_CODE <CODE_FROM_URL>

curl -s -X POST https://www.strava.com/oauth/token \
  -d client_id=$STRAVA_CLIENT_ID \
  -d client_secret=$STRAVA_CLIENT_SECRET \
  -d code=$STRAVA_CODE \
  -d grant_type=authorization_code | cat
```

Then get the refresh token from the response and set it:

```
set -x STRAVA_REFRESH_TOKEN <REFRESH_TOKEN_FROM_RESPONSE>
```

Then run:

```
python -m venv .venv
source .venv/bin/activate(.fish)
pip install -r requirements.txt

python -m src.endurance.schema
python -m src.endurance.ingest_strava
python -m src.endurance.build_runs
python -m src.endurance.build_features
python -m src.endurance.build_labels
python -m src.endurance.train_volume
```
