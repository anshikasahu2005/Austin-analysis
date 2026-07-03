"""
Austin Behavioral Task Dashboard
---------------------------------
Interactive analysis of a mouse Go/No-Go visual discrimination task,
20 sessions recorded May 2023 (session IDs 230504-230523).

Data source: event_history / lick_history / camera_history / parameters / RT
.mat files, pre-parsed into tidy tables under ./processed/. Raw widefield
imaging (.tif) and behavior video (.avi) files (~4.5 GB) are not included --
they are too large for interactive analysis and are not needed for the
behavioral/trial-level analysis below.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Austin Behavioral Dashboard", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")

TYPE_COLORS = {"Left": "#4C72B0", "Right": "#DD8452", "Catch": "#8C8C8C"}
OUTCOME_COLORS = {"Hit": "#55A868", "Miss": "#C44E52", "Incorrect": "#CCB974", "Catch Trial": "#8172B2", "Other": "#999999"}


@st.cache_data
def load_data():
    trials = pd.read_parquet(os.path.join(DATA_DIR, "trials.parquet"))
    licks = pd.read_parquet(os.path.join(DATA_DIR, "licks.parquet"))
    cameras = pd.read_parquet(os.path.join(DATA_DIR, "cameras.parquet"))
    params = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))
    rt_struct = pd.read_parquet(os.path.join(DATA_DIR, "rt_struct.parquet"))
    summary = pd.read_csv(os.path.join(DATA_DIR, "session_summary.csv"))

    for df in (trials, licks, cameras, params, rt_struct, summary):
        df["date"] = pd.to_datetime(df["date"])
        df["session"] = df["session"].astype(str)

    summary = summary.sort_values("date").reset_index(drop=True)
    summary["day_number"] = range(1, len(summary) + 1)
    trials = trials.merge(summary[["session", "day_number"]], on="session", how="left")
    return trials, licks, cameras, params, rt_struct, summary


trials, licks, cameras, params, rt_struct, summary = load_data()
all_sessions = summary["session"].tolist()

# ---------------------------------------------------------------- SIDEBAR --
st.sidebar.title("🐭 Austin Dashboard")
st.sidebar.caption("Go/No-Go visual discrimination task · widefield imaging cohort")

sess_labels = [f"{r.session}  ({r.date.strftime('%b %d')})" for r in summary.itertuples()]
label_to_sess = dict(zip(sess_labels, all_sessions))

selected_labels = st.sidebar.multiselect(
    "Sessions", sess_labels, default=sess_labels,
    help="Filter every tab below to these sessions."
)
selected_sessions = [label_to_sess[l] for l in selected_labels] or all_sessions

type_filter = st.sidebar.multiselect("Trial type", ["Left", "Right", "Catch"], default=["Left", "Right", "Catch"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Not included:** raw widefield imaging (`.tif`) and behavior video (`.avi`) "
    "files (~4.5 GB total) — excluded as too large/raw for interactive analysis. "
    "This dashboard covers all structured behavioral data: trial events, licks, "
    "camera sync pulses, reaction times, and task parameters."
)

t_f = trials[trials.session.isin(selected_sessions) & trials.type.isin(type_filter)]
s_f = summary[summary.session.isin(selected_sessions)]
l_f = licks[licks.session.isin(selected_sessions)]
c_f = cameras[cameras.session.isin(selected_sessions)]

# ------------------------------------------------------------------ TABS --
tab_overview, tab_outcomes, tab_rt, tab_lick, tab_timing, tab_params, tab_raw = st.tabs(
    ["📊 Overview", "🎯 Trial Outcomes", "⏱️ Reaction Times", "👅 Licking", "📷 Session Timing", "⚙️ Parameters", "🔎 Raw Data"]
)

# =========================================================== OVERVIEW ====
with tab_overview:
    st.header("Overview")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sessions", f"{s_f.shape[0]}")
    k2.metric("Total trials", f"{int(s_f.n_trials.sum()):,}")
    k3.metric("Overall accuracy", f"{(s_f.n_hit.sum() / max(s_f.n_hit.sum()+s_f.n_miss.sum()+s_f.n_incorrect.sum(),1)):.1%}")
    k4.metric("Total licks", f"{int(s_f.n_licks.sum()):,}")
    k5.metric("Date range", f"{s_f.date.min().strftime('%b %d')} – {s_f.date.max().strftime('%b %d')}")

    st.subheader("Learning curve — accuracy across sessions")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_f.day_number, y=s_f.accuracy, mode="lines+markers",
                              name="Accuracy (Hit / Go trials)", line=dict(color="#4C72B0", width=3),
                              customdata=s_f.session, hovertemplate="Session %{customdata}<br>Accuracy: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=s_f.day_number, y=s_f.catch_rate, mode="lines+markers",
                              name="Catch trial rate", line=dict(color="#8C8C8C", dash="dot"),
                              customdata=s_f.session, hovertemplate="Session %{customdata}<br>Catch rate: %{y:.1%}<extra></extra>"))
    fig.update_layout(xaxis_title="Training day #", yaxis_title="Rate", yaxis_tickformat=".0%",
                       hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Accuracy = Hit / (Hit + Miss + Incorrect) on Go trials (Left + Right stimulus trials). "
        "Catch trials (no stimulus) are logged with a single outcome code regardless of response, "
        "so a hit/false-alarm split isn't recoverable from event_history alone — shown here only as trial frequency."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trials per session")
        fig2 = px.bar(s_f, x="day_number", y=["n_left", "n_right", "n_catch"],
                       labels={"value": "Trial count", "day_number": "Training day #", "variable": "Type"},
                       color_discrete_sequence=[TYPE_COLORS["Left"], TYPE_COLORS["Right"], TYPE_COLORS["Catch"]],
                       hover_data={"day_number": False})
        fig2.update_layout(legend_title_text="")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        st.subheader("Session duration & lick rate")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=s_f.day_number, y=s_f.session_duration_min, name="Duration (min)",
                               marker_color="#4C72B0", yaxis="y"))
        fig3.add_trace(go.Scatter(x=s_f.day_number, y=s_f.lick_rate_hz, name="Lick rate (Hz)",
                                   mode="lines+markers", marker_color="#C44E52", yaxis="y2"))
        fig3.update_layout(
            xaxis_title="Training day #",
            yaxis=dict(title="Duration (min)"),
            yaxis2=dict(title="Lick rate (Hz)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ======================================================= TRIAL OUTCOMES ==
with tab_outcomes:
    st.header("Trial Outcomes")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Outcome mix per session")
        outc = t_f.groupby(["day_number", "session", "outcome"]).size().reset_index(name="count")
        fig = px.bar(outc, x="day_number", y="count", color="outcome",
                     color_discrete_map=OUTCOME_COLORS, labels={"day_number": "Training day #"},
                     hover_data={"session": True})
        fig.update_layout(legend_title_text="", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Overall breakdown")
        pie = t_f.outcome.value_counts().reset_index()
        pie.columns = ["outcome", "count"]
        fig = px.pie(pie, names="outcome", values="count", color="outcome", color_discrete_map=OUTCOME_COLORS, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Left vs. Right stimulus performance")
    go_trials = t_f[t_f.type.isin(["Left", "Right"])]
    perf = go_trials.groupby(["session", "day_number", "type"]).outcome.apply(
        lambda s: (s == "Hit").sum() / len(s) if len(s) else np.nan
    ).reset_index(name="accuracy")
    fig = px.line(perf, x="day_number", y="accuracy", color="type", markers=True,
                   color_discrete_map=TYPE_COLORS, labels={"day_number": "Training day #", "accuracy": "Hit rate"})
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Session summary table")
    show_cols = ["session", "date", "n_trials", "n_left", "n_right", "n_catch",
                 "n_hit", "n_miss", "n_incorrect", "accuracy", "miss_rate", "incorrect_rate", "catch_rate"]
    pct_cols = ["accuracy", "miss_rate", "incorrect_rate", "catch_rate"]
    table_df = s_f[show_cols].copy()
    table_df[pct_cols] = (table_df[pct_cols] * 100).round(1)
    st.dataframe(
        table_df,
        use_container_width=True, hide_index=True,
        column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in pct_cols},
    )

# ========================================================= REACTION TIMES=
with tab_rt:
    st.header("Reaction Times")
    st.caption("`rt` = time-to-lick within the response window (Go trials), from event_history row 3. "
               "Miss trials pin at the window ceiling (~1.2 s = no response).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RT distribution by outcome")
        rt_plot_df = t_f[t_f.outcome.isin(["Hit", "Miss", "Incorrect"])]
        fig = px.histogram(rt_plot_df, x="rt", color="outcome", nbins=60, opacity=0.7,
                            color_discrete_map=OUTCOME_COLORS, barmode="overlay",
                            labels={"rt": "Reaction time (s)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Hit RT by trial type")
        hit_rt = t_f[t_f.outcome == "Hit"]
        fig = px.box(hit_rt, x="type", y="rt", color="type", color_discrete_map=TYPE_COLORS,
                     labels={"rt": "Reaction time (s)", "type": "Trial type"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Median Hit RT across sessions")
    rt_trend = t_f[t_f.outcome == "Hit"].groupby(["session", "day_number"]).rt.median().reset_index()
    fig = px.line(rt_trend, x="day_number", y="rt", markers=True,
                   labels={"day_number": "Training day #", "rt": "Median Hit RT (s)"})
    st.plotly_chart(fig, use_container_width=True)

    if not rt_struct.empty:
        st.subheader("Supplementary: RT.mat struct (Target/Distractor RT vs. inter-trial interval)")
        st.caption(f"Only logged for sessions {', '.join(sorted(rt_struct.session.unique()))} in the raw data.")
        fig = px.scatter(rt_struct, x="iti", y="rt", color="category", opacity=0.6,
                          facet_col="session", labels={"iti": "ITI (s)", "rt": "RT (s)"})
        st.plotly_chart(fig, use_container_width=True)

# ================================================================ LICKING=
with tab_lick:
    st.header("Licking Behavior")

    st.subheader("Lick rate across sessions")
    fig = px.bar(s_f, x="day_number", y="lick_rate_hz", labels={"day_number": "Training day #", "lick_rate_hz": "Lick rate (Hz)"},
                 hover_data=["session"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lick raster (select a session)")
    sess_for_raster = st.selectbox("Session", selected_sessions,
                                    format_func=lambda s: f"{s} ({summary.loc[summary.session==s,'date'].dt.strftime('%b %d').values[0]})")
    licks_sess = licks[licks.session == sess_for_raster]
    if licks_sess.empty:
        st.info("No lick_history recorded for this session in the raw data.")
    else:
        fig = px.strip(licks_sess, x="lick_time_s", labels={"lick_time_s": "Session time (s)"})
        fig.update_traces(marker=dict(size=3, opacity=0.5))
        fig.update_layout(yaxis_visible=False, height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Lick rate over time — session {sess_for_raster}")
        bins = np.arange(0, licks_sess.lick_time_s.max() + 10, 10)
        counts, edges = np.histogram(licks_sess.lick_time_s, bins=bins)
        rate_df = pd.DataFrame({"t": edges[:-1], "licks_per_10s": counts})
        fig2 = px.area(rate_df, x="t", y="licks_per_10s", labels={"t": "Session time (s)"})
        st.plotly_chart(fig2, use_container_width=True)

    missing = [s for s in selected_sessions if s not in licks.session.unique()]
    if missing:
        st.caption(f"Sessions with no lick_history file in the raw data: {', '.join(sorted(missing))}")

# ========================================================= SESSION TIMING=
with tab_timing:
    st.header("Session Timing & Camera Sync")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Session duration (min)")
        fig = px.bar(s_f, x="day_number", y="session_duration_min", hover_data=["session"],
                     labels={"day_number": "Training day #", "session_duration_min": "Duration (min)"})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Estimated camera frame rate")
        fig = px.bar(s_f, x="day_number", y="camera_fps_est", hover_data=["session", "n_camera_frames"],
                     labels={"day_number": "Training day #", "camera_fps_est": "FPS (median inter-frame interval)"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Camera frame timing (select a session)")
    sess_for_cam = st.selectbox("Session ", selected_sessions, key="cam_sess",
                                 format_func=lambda s: f"{s} ({summary.loc[summary.session==s,'date'].dt.strftime('%b %d').values[0]})")
    cam_sess = cameras[cameras.session == sess_for_cam].sort_values("frame_time_s")
    if len(cam_sess) > 1:
        intervals = np.diff(cam_sess.frame_time_s.values)
        fig = px.histogram(x=intervals, nbins=50, labels={"x": "Inter-frame interval (s)"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(cam_sess):,} frames · median interval {np.median(intervals)*1000:.1f} ms "
                   f"({1/np.median(intervals):.2f} fps) · max gap {intervals.max():.2f} s")

# ============================================================= PARAMETERS=
with tab_params:
    st.header("Task Parameters Across Sessions")
    p_f = params[params.session.isin(selected_sessions)].merge(summary[["session", "day_number"]], on="session").sort_values("day_number")
    display_cols = [c for c in p_f.columns if c not in ("day_number",)]
    st.dataframe(p_f[display_cols], use_container_width=True, hide_index=True)

    numeric_candidates = [c for c in ["No_Lick_Interval_min", "No_Lick_Interval_max", "Total_Rewards",
                                       "Lick_Window", "Solenoid_Open", "Left_Stimulus_Duration", "Right_Stimulus_Duration"]
                           if c in p_f.columns]
    if numeric_candidates:
        st.subheader("Parameter trends across days")
        chosen = st.multiselect("Parameters to plot", numeric_candidates, default=numeric_candidates[:2])
        if chosen:
            melt = p_f.melt(id_vars=["day_number", "session"], value_vars=chosen)
            fig = px.line(melt, x="day_number", y="value", color="variable", markers=True,
                           labels={"day_number": "Training day #", "value": "Value"})
            st.plotly_chart(fig, use_container_width=True)

# ==================================================================== RAW=
with tab_raw:
    st.header("Raw Trial-Level Data Explorer")
    st.caption("Filtered by the sidebar session/type selection. Download as CSV below.")
    st.dataframe(t_f.sort_values(["session", "trial_index"]), use_container_width=True, hide_index=True, height=500)
    st.download_button("Download filtered trials as CSV", t_f.to_csv(index=False).encode(),
                        file_name="austin_trials_filtered.csv", mime="text/csv")
