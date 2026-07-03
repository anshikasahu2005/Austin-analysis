# Austin Behavioral Task Dashboard

Interactive Streamlit dashboard analyzing 20 sessions (May 4-23, 2023) of a mouse
Go/No-Go visual discrimination task, built from the `.mat` files in the uploaded
`Austin` dataset.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## What's included

- `app.py` — the dashboard (Overview, Trial Outcomes, Reaction Times, Licking,
  Session Timing, Parameters, Raw Data Explorer tabs)
- `processed/` — tidy CSV/Parquet tables parsed from the raw `.mat` files:
  - `trials.parquet` — every trial across all sessions (type, outcome, RT, timing)
  - `licks.parquet` — every recorded lick timestamp
  - `cameras.parquet` — widefield camera sync frame timestamps
  - `parameters.csv` — task parameters per session
  - `rt_struct.parquet` — the separate `RT.mat` Target/Distractor RT+ITI data (only logged for 3 sessions)
  - `session_summary.csv` — one row per session with aggregated metrics
- `parse_data.py` — the script that produced `processed/` from the raw `.mat` files (kept for reference/reproducibility; you don't need to rerun it)

## Scope note

The original upload also contained raw widefield imaging (`.tif`, several hundred
MB each) and behavior camera video (`.avi`) files — about 4.5 GB total. Those are
**not included here**: they're raw video/image data, not suited to an interactive
web dashboard, and weren't needed for the behavioral/trial-level analysis. Everything
else in the dataset (trial events, licks, camera sync pulses, reaction times, task
parameters) is fully parsed and analyzed.

## Data notes / interpretation

- Trial types: `Left` / `Right` (stimulus side) and `Catch` (no stimulus).
- Outcomes: `Hit` (correct lick), `Miss` (no response, RT pins at the ~1.2s window
  ceiling), `Incorrect` (rare, very fast RT — looks like a premature/impulsive
  response), and `Catch Trial` (logged for every catch trial regardless of
  response — the raw data doesn't let us cleanly split catch trials into
  false-alarm vs. correct-rejection, so we report catch trial frequency only
  rather than inventing a d-prime).
- 5 of 20 sessions have no `lick_history` file in the raw data (230507, 230508,
  230511, 230520, 230521) — the dashboard notes this where relevant.
- Only 3 sessions (230507-230509) include a separate `RT.mat` file; shown as a
  supplementary chart.
