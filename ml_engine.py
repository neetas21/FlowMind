import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ── PROCRASTINATION SCORING ───────────────────────────────────────────────────

def compute_procrastination_score(proc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a procrastination score (0-100) per task using:
    - delay_count: how many times task was pushed back
    - completion_rate: inverse of how often sessions complete
    - time_efficiency: actual vs estimated time ratio
    - mood_drag: lower mood → higher procrastination weight
    """
    if proc_df.empty:
        return proc_df

    df = proc_df.copy()

    # Fill NaNs safely
    df['delay_count'] = df['delay_count'].fillna(0)
    df['completed_sessions'] = df['completed_sessions'].fillna(0)
    df['total_sessions'] = df['total_sessions'].fillna(0).clip(lower=1)
    df['avg_actual_minutes'] = df['avg_actual_minutes'].fillna(0)
    df['estimated_minutes'] = df['estimated_minutes'].fillna(30).clip(lower=1)
    df['avg_mood'] = df['avg_mood'].fillna(3)

    # Feature engineering
    df['completion_rate'] = df['completed_sessions'] / df['total_sessions']
    df['abandonment_rate'] = 1 - df['completion_rate']
    df['time_efficiency'] = (df['avg_actual_minutes'] / df['estimated_minutes']).clip(0, 2)
    df['mood_drag'] = (5 - df['avg_mood']) / 4   # 0=high mood, 1=low mood

    # Weighted procrastination score (out of 100)
    df['procrastination_score'] = (
        df['delay_count'].clip(0, 5) / 5 * 30 +        # 30% weight: delay frequency
        df['abandonment_rate'] * 40 +                    # 40% weight: not finishing
        (1 - df['time_efficiency'].clip(0,1)) * 15 +    # 15% weight: time under-spend
        df['mood_drag'] * 15                             # 15% weight: mood context
    ).clip(0, 100).round(1)

    df['risk_level'] = pd.cut(
        df['procrastination_score'],
        bins=[0, 25, 50, 75, 100],
        labels=['Low 🟢', 'Medium 🟡', 'High 🟠', 'Critical 🔴'],
        include_lowest=True
    )

    return df


# ── PATTERN CLUSTERING ────────────────────────────────────────────────────────

CLUSTER_LABELS = {
    0: ("The Avoider", "You delay starting tasks, especially low-interest ones."),
    1: ("The Sprinter", "You start strong but abandon sessions before finishing."),
    2: ("The Perfectionist", "You over-plan but under-execute due to fear of failure."),
    3: ("The Drifter", "Your focus shifts rapidly between tasks without completion."),
}

def cluster_procrastination_patterns(proc_df: pd.DataFrame, n_clusters=4):
    """
    Cluster tasks into procrastination archetypes using KMeans.
    Returns df with cluster label + description.
    """
    if len(proc_df) < n_clusters:
        return proc_df, None, "Not enough data for pattern analysis yet."

    features = ['delay_count', 'abandonment_rate', 'time_efficiency', 'mood_drag', 'procrastination_score']
    available = [f for f in features if f in proc_df.columns]

    feature_df = proc_df[available].fillna(0)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=min(n_clusters, len(proc_df)), random_state=42, n_init=10)
    proc_df = proc_df.copy()
    proc_df['cluster'] = kmeans.fit_predict(X)

    # Find dominant cluster
    dominant = proc_df['cluster'].mode()[0]
    label, desc = CLUSTER_LABELS.get(dominant, ("Unknown", "Keep logging more sessions!"))

    proc_df['pattern_type'] = proc_df['cluster'].map(
        lambda c: CLUSTER_LABELS.get(c, ("Unknown", ""))[0]
    )

    return proc_df, kmeans, f"**Your dominant pattern: {label}** — {desc}"


# ── HOURLY FOCUS PREDICTION ───────────────────────────────────────────────────

def predict_best_focus_hours(hourly_df: pd.DataFrame):
    """
    Uses LinearRegression on historical hourly data to predict
    which hours of the day are best for deep focus.
    Returns a dict with best_hour, worst_hour, and a prediction series.
    """
    if hourly_df.empty or len(hourly_df) < 3:
        return None

    df = hourly_df.copy().dropna()
    if len(df) < 3:
        return None

    X = df[['hour']].values
    y = df['completion_rate'].values

    model = LinearRegression()
    model.fit(X, y)

    all_hours = pd.DataFrame({'hour': range(24)})
    all_hours['predicted_focus'] = model.predict(all_hours[['hour']].values).clip(0, 1)

    best_hour = int(all_hours.loc[all_hours['predicted_focus'].idxmax(), 'hour'])
    worst_hour = int(all_hours.loc[all_hours['predicted_focus'].idxmin(), 'hour'])

    return {
        'predictions': all_hours,
        'best_hour': best_hour,
        'worst_hour': worst_hour,
        'model': model
    }


# ── NUDGE TRIGGER DETECTION ───────────────────────────────────────────────────

def detect_nudge_triggers(sessions_df: pd.DataFrame, tasks_df: pd.DataFrame, active_task_id=None):
    """
    Scans recent session + task data to determine if an AI nudge should fire.
    Returns list of (trigger_reason, context_dict) tuples.
    """
    triggers = []

    if sessions_df.empty:
        return triggers

    recent = sessions_df.head(10)

    # Trigger 1: Short abandoned session (< 5 min)
    last_session = recent.iloc[0] if not recent.empty else None
    if last_session is not None:
        if last_session.get('completed') == 0 and last_session.get('actual_minutes', 10) < 5:
            triggers.append(("short_abandon", {
                "task": last_session.get('task_title', 'your task'),
                "minutes": last_session.get('actual_minutes', 0),
                "mood": last_session.get('mood', 3)
            }))

    # Trigger 2: 3+ consecutive incomplete sessions
    recent_complete = recent['completed'].tolist() if 'completed' in recent.columns else []
    if len(recent_complete) >= 3 and sum(recent_complete[:3]) == 0:
        triggers.append(("consecutive_fails", {
            "count": 3,
            "avg_mood": recent.head(3)['mood'].mean() if 'mood' in recent.columns else 3
        }))

    # Trigger 3: High-priority task delayed 2+ times
    if not tasks_df.empty:
        high_prio_delayed = tasks_df[
            (tasks_df['priority'] == 1) &
            (tasks_df['delay_count'] >= 2) &
            (tasks_df['status'] == 'pending')
        ]
        if not high_prio_delayed.empty:
            task = high_prio_delayed.iloc[0]
            triggers.append(("high_prio_delayed", {
                "task": task['title'],
                "delay_count": int(task['delay_count'])
            }))

    # Trigger 4: Low mood + any pending task
    if last_session is not None and last_session.get('mood', 3) <= 2:
        triggers.append(("low_mood", {
            "mood": last_session.get('mood', 2),
            "energy": last_session.get('energy', 2)
        }))

    return triggers


# ── DAILY STREAK CALCULATOR ───────────────────────────────────────────────────

def calculate_streak(daily_df: pd.DataFrame) -> int:
    """Returns current consecutive day streak with at least 1 completed session."""
    if daily_df.empty:
        return 0

    daily_df = daily_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date', ascending=False)

    streak = 0
    today = pd.Timestamp.now().normalize()

    for _, row in daily_df.iterrows():
        expected_date = today - pd.Timedelta(days=streak)
        if row['date'].normalize() == expected_date and row['completed_sessions'] > 0:
            streak += 1
        else:
            break

    return streak
