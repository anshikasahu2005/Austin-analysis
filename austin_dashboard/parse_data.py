"""
Parse Austin behavioral dataset (.mat files) into tidy CSVs for the Streamlit dashboard.
"""
import scipy.io as sio
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
AUSTIN = os.path.join(ROOT, "Austin")
OUT = os.path.join(ROOT, "processed")
os.makedirs(OUT, exist_ok=True)

TYPE_MAP = {1: "Left", 3: "Right", 8: "Catch"}
# Note: outcome code 44 is logged for every Catch trial regardless of whether the
# animal responded (RT never pins at the 1.2s window ceiling the way Miss trials do),
# so it can't be reliably split into "false alarm" vs "correct rejection" from
# event_history alone. We label it neutrally as "Catch Trial" rather than overclaim.
OUTCOME_MAP = {0: "Other", 11: "Hit", 22: "Miss", 33: "Incorrect", 44: "Catch Trial"}

sessions = sorted([os.path.basename(p) for p in glob.glob(os.path.join(AUSTIN, "2305*"))])

def to_date(sess):
    return datetime.strptime("20" + sess, "%Y%m%d").date()

trial_rows = []
lick_rows = []
camera_rows = []
param_rows = []
rt_rows = []
session_summary_rows = []

for sess in sessions:
    d = os.path.join(AUSTIN, sess)
    date = to_date(sess)

    # ---- event_history -> trials ----
    eh_path = os.path.join(d, f"{sess}_event_history.mat")
    eh = sio.loadmat(eh_path)["event_history"]
    n_trials = eh.shape[1]
    type_code = eh[0]
    outcome_code = eh[1]
    rt = eh[2]
    no_lick_interval = eh[3]
    event_time = eh[4]

    for i in range(n_trials):
        trial_rows.append({
            "session": sess,
            "date": date,
            "trial_index": i + 1,
            "type_code": type_code[i],
            "type": TYPE_MAP.get(type_code[i], f"Unknown({type_code[i]})"),
            "outcome_code": outcome_code[i],
            "outcome": OUTCOME_MAP.get(outcome_code[i], f"Unknown({outcome_code[i]})"),
            "rt": rt[i],
            "no_lick_interval": no_lick_interval[i],
            "event_time_s": event_time[i],
        })

    # ---- lick_history ----
    lh_path = os.path.join(d, f"{sess}_lick_history.mat")
    n_licks = 0
    if os.path.exists(lh_path):
        lh = sio.loadmat(lh_path)["lick_history"][0]
        n_licks = len(lh)
        for t in lh:
            lick_rows.append({"session": sess, "date": date, "lick_time_s": t})

    # ---- camera_history ----
    ch_path = os.path.join(d, f"{sess}_camera_history.mat")
    ch = sio.loadmat(ch_path)["camera_history"][0]
    n_frames = len(ch)
    for t in ch:
        camera_rows.append({"session": sess, "date": date, "frame_time_s": t})

    # ---- parameters ----
    p_path = os.path.join(d, f"{sess}_parameters.mat")
    p = sio.loadmat(p_path)["parameters"][0, 0]
    prow = {"session": sess, "date": date}
    for name in p.dtype.names:
        v = p[name]
        if v.dtype.kind in "US":
            prow[name] = str(v[0]) if v.size else None
        else:
            arr = np.array(v).flatten()
            if arr.size == 1:
                prow[name] = arr[0]
            elif arr.size > 1:
                prow[name + "_min"] = arr.min()
                prow[name + "_max"] = arr.max()
                prow[name + "_n"] = arr.size
            else:
                prow[name] = None
    param_rows.append(prow)

    # ---- RT.mat (only some sessions) ----
    rt_path = os.path.join(d, f"{sess}_RT.mat")
    if os.path.exists(rt_path):
        rtm = sio.loadmat(rt_path)["RT"][0, 0]
        names = rtm.dtype.names
        for cat in ["Target", "Distractor"]:
            rt_arr = rtm[f"{cat}_RT"][0]
            iti_arr = rtm[f"{cat}_ITI"][0]
            for j in range(len(rt_arr)):
                rt_rows.append({
                    "session": sess, "date": date, "category": cat,
                    "rt": rt_arr[j], "iti": iti_arr[j] if j < len(iti_arr) else np.nan,
                })

    # ---- session summary ----
    df_t = pd.DataFrame([r for r in trial_rows if r["session"] == sess])
    go = df_t[df_t.type.isin(["Left", "Right"])]
    catch = df_t[df_t.type == "Catch"]
    n_hit = (go.outcome == "Hit").sum()
    n_miss = (go.outcome == "Miss").sum()
    n_incorrect = (go.outcome == "Incorrect").sum()
    n_go = len(go)
    n_catch_trials = len(catch)
    accuracy = n_hit / n_go if n_go else np.nan
    miss_rate = n_miss / n_go if n_go else np.nan
    incorrect_rate = n_incorrect / n_go if n_go else np.nan
    catch_rate = n_catch_trials / n_trials if n_trials else np.nan

    session_duration_min = (event_time.max() - event_time.min()) / 60 if n_trials else np.nan
    lick_rate_hz = n_licks / (session_duration_min * 60) if session_duration_min and n_licks else np.nan
    camera_fps = 1 / np.median(np.diff(ch)) if len(ch) > 1 else np.nan

    session_summary_rows.append({
        "session": sess, "date": date,
        "n_trials": n_trials, "n_left": (df_t.type == "Left").sum(),
        "n_right": (df_t.type == "Right").sum(), "n_catch": n_catch_trials,
        "n_hit": n_hit, "n_miss": n_miss, "n_incorrect": n_incorrect,
        "accuracy": accuracy, "miss_rate": miss_rate, "incorrect_rate": incorrect_rate,
        "catch_rate": catch_rate,
        "mean_rt_hit": go[go.outcome == "Hit"].rt.mean() if n_hit else np.nan,
        "mean_rt_all_go": go.rt.mean() if n_go else np.nan,
        "session_duration_min": session_duration_min,
        "n_licks": n_licks, "lick_rate_hz": lick_rate_hz,
        "n_camera_frames": n_frames, "camera_fps_est": camera_fps,
    })

# ---- write out ----
trials_df = pd.DataFrame(trial_rows)
licks_df = pd.DataFrame(lick_rows)
cameras_df = pd.DataFrame(camera_rows)
params_df = pd.DataFrame(param_rows)
rt_df = pd.DataFrame(rt_rows)
summary_df = pd.DataFrame(session_summary_rows).sort_values("date")

trials_df.to_parquet(os.path.join(OUT, "trials.parquet"), index=False)
licks_df.to_parquet(os.path.join(OUT, "licks.parquet"), index=False)
cameras_df.to_parquet(os.path.join(OUT, "cameras.parquet"), index=False)
params_df.to_csv(os.path.join(OUT, "parameters.csv"), index=False)
rt_df.to_parquet(os.path.join(OUT, "rt_struct.parquet"), index=False)
summary_df.to_csv(os.path.join(OUT, "session_summary.csv"), index=False)

print("trials:", trials_df.shape)
print("licks:", licks_df.shape)
print("cameras:", cameras_df.shape)
print("parameters:", params_df.shape)
print("rt_struct:", rt_df.shape)
print("session_summary:")
print(summary_df[["session","date","n_trials","accuracy","catch_rate"]])
