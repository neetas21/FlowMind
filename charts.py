import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = {
    "primary": "#7C3AED",
    "secondary": "#A78BFA",
    "accent": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "bg": "#0F0F1A",
    "card": "#1A1A2E",
    "text": "#E2E8F0",
    "muted": "#64748B",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"])),
)


def chart_daily_focus(daily_df: pd.DataFrame):
    if daily_df.empty:
        return _empty_chart("No session data yet — start your first session!")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=daily_df["date"], y=daily_df["total_focus_minutes"],
        name="Focus Minutes", marker_color=COLORS["primary"],
        opacity=0.85
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["completed_sessions"],
        name="Sessions Done", mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=7)
    ), secondary_y=True)

    fig.update_layout(
        title="📅 Daily Focus Overview",
        **LAYOUT_DEFAULTS,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Minutes", gridcolor="#2D2D44"),
        yaxis2=dict(title="Sessions", showgrid=False),
    )
    return fig


def chart_mood_vs_productivity(sessions_df: pd.DataFrame):
    if sessions_df.empty:
        return _empty_chart("Log sessions with mood data to see this chart.")

    df = sessions_df.dropna(subset=["mood", "actual_minutes"])
    if df.empty:
        return _empty_chart("No mood data recorded yet.")

    fig = px.scatter(
        df, x="mood", y="actual_minutes",
        color="completed",
        color_discrete_map={0: COLORS["danger"], 1: COLORS["accent"]},
        size="energy",
        hover_data=["task_title", "energy"],
        labels={"mood": "Mood (1–5)", "actual_minutes": "Focus Minutes", "completed": "Completed"},
        title="😊 Mood vs Focus Duration",
        trendline="ols",
        trendline_color_override=COLORS["secondary"],
    )
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_traces(marker=dict(opacity=0.8))
    return fig


def chart_procrastination_scores(proc_df: pd.DataFrame):
    if proc_df.empty or "procrastination_score" not in proc_df.columns:
        return _empty_chart("Add tasks and sessions to see procrastination scores.")

    df = proc_df.sort_values("procrastination_score", ascending=True).tail(12)

    color_map = {"Low 🟢": "#10B981", "Medium 🟡": "#F59E0B",
                 "High 🟠": "#F97316", "Critical 🔴": "#EF4444"}
    colors = df["risk_level"].map(color_map).fillna(COLORS["secondary"])

    fig = go.Figure(go.Bar(
        x=df["procrastination_score"], y=df["title"],
        orientation="h",
        marker_color=colors,
        text=df["risk_level"].astype(str),
        textposition="outside",
    ))
    fig.update_layout(
        title="🚨 Procrastination Risk by Task",
        xaxis_title="Score (0–100)",
        **LAYOUT_DEFAULTS,
        xaxis=dict(range=[0, 110], gridcolor="#2D2D44"),
        yaxis=dict(showgrid=False),
        height=max(300, len(df) * 45),
    )
    return fig


def chart_hourly_heatmap(hourly_df: pd.DataFrame, predictions=None):
    if hourly_df.empty:
        return _empty_chart("Need more sessions across different hours to build this.")

    all_hours = pd.DataFrame({"hour": range(24)})
    merged = all_hours.merge(hourly_df, on="hour", how="left").fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged["hour"],
        y=merged["completion_rate"],
        name="Historical",
        marker_color=COLORS["primary"],
        opacity=0.7,
    ))

    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=predictions["hour"],
            y=predictions["predicted_focus"],
            name="Predicted",
            mode="lines",
            line=dict(color=COLORS["accent"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title="⏰ Your Focus by Hour of Day",
        xaxis=dict(
            tickvals=list(range(24)),
            ticktext=[f"{h}:00" for h in range(24)],
            showgrid=False
        ),
        yaxis=dict(title="Completion Rate", tickformat=".0%", gridcolor="#2D2D44"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_category_breakdown(proc_df: pd.DataFrame):
    if proc_df.empty:
        return _empty_chart("No category data yet.")

    cat_df = proc_df.groupby("category").agg(
        total_sessions=("total_sessions", "sum"),
        avg_score=("procrastination_score", "mean"),
        completed=("completed_sessions", "sum"),
    ).reset_index()

    fig = px.bar(
        cat_df, x="category", y="avg_score",
        color="avg_score",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        text=cat_df["avg_score"].round(1),
        title="📂 Procrastination by Category",
        labels={"avg_score": "Avg Score", "category": "Category"},
    )
    fig.update_layout(**LAYOUT_DEFAULTS, coloraxis_showscale=False)
    fig.update_traces(textposition="outside")
    return fig


def chart_streak_calendar(daily_df: pd.DataFrame):
    if daily_df.empty:
        return _empty_chart("Start completing sessions to build your streak calendar.")

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["weekday"] = df["date"].dt.weekday
    df["label"] = df["date"].dt.strftime("%b %d")

    fig = go.Figure(go.Heatmap(
        x=df["week"],
        y=df["weekday"],
        z=df["completed_sessions"],
        text=df["label"],
        colorscale=[[0, "#1A1A2E"], [0.5, COLORS["secondary"]], [1, COLORS["accent"]]],
        hovertemplate="%{text}<br>Completed: %{z}<extra></extra>",
        showscale=False,
    ))

    fig.update_layout(
        title="🟩 Focus Streak Calendar",
        yaxis=dict(
            tickvals=list(range(7)),
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            showgrid=False,
        ),
        xaxis=dict(showgrid=False, title="Week"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_energy_mood_radar(sessions_df: pd.DataFrame):
    if sessions_df.empty:
        return _empty_chart("No session data for radar chart.")

    completed = sessions_df[sessions_df["completed"] == 1]
    abandoned = sessions_df[sessions_df["completed"] == 0]

    categories = ["Mood", "Energy", "Session Length", "Completion", "Consistency"]

    def normalize_stats(df):
        if df.empty:
            return [0] * 5
        return [
            df["mood"].mean() / 5 if "mood" in df.columns else 0,
            df["energy"].mean() / 5 if "energy" in df.columns else 0,
            min(df["actual_minutes"].mean() / 60, 1) if "actual_minutes" in df.columns else 0,
            len(df) / max(len(sessions_df), 1),
            min(len(df) / 10, 1),
        ]

    fig = go.Figure()
    for label, df, color in [
        ("Completed", completed, COLORS["accent"]),
        ("Abandoned", abandoned, COLORS["danger"]),
    ]:
        vals = normalize_stats(df)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=label,
            line_color=color,
            fillcolor=color.replace("#", "rgba(") + ",0.15)" if color.startswith("#") else color,
            opacity=0.8,
        ))

    fig.update_layout(
        title="🎯 Completed vs Abandoned Sessions",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#2D2D44"),
            angularaxis=dict(gridcolor="#2D2D44"),
        ),
        **LAYOUT_DEFAULTS,
    )
    return fig


def _empty_chart(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=f"📊 {message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=COLORS["muted"]),
        align="center",
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=250,
    )
    return fig
