import os
from groq import Groq

def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are an ADHD Focus Coach — warm, direct, and neurodivergent-affirming.
You understand that ADHD brains work differently: dopamine-seeking, time-blind, prone to rejection sensitivity.
Your nudges are:
- SHORT (max 3 sentences)
- Specific and actionable (not generic "you can do it!")
- Non-judgmental (never say "you should have", "why didn't you")
- Dopamine-aware (use task-shrinking, body doubling, interest-boosting techniques)
- Formatted for quick reading (use ONE emoji max, plain language)

Never lecture. Never use bullet points. Just one powerful, human micro-coaching message."""


def build_nudge_prompt(trigger_reason: str, context: dict) -> str:
    prompts = {
        "short_abandon": f"""The user with ADHD just abandoned their task '{context.get('task', 'a task')}' after only {context.get('minutes', 0):.1f} minutes. 
Their current mood is {context.get('mood', 3)}/5.
Give them a compassionate micro-nudge to re-engage. Suggest a tiny first action (under 2 minutes).""",

        "consecutive_fails": f"""The user with ADHD has had {context.get('count', 3)} sessions in a row where they couldn't complete anything.
Their average mood recently is {context.get('avg_mood', 3):.1f}/5.
They might be in an ADHD paralysis spiral. Give a grounding nudge — acknowledge it's hard, offer one tiny escape hatch.""",

        "high_prio_delayed": f"""The user has delayed their high-priority task '{context.get('task', 'important task')}' {context.get('delay_count', 2)} times already.
This is classic ADHD task avoidance, likely due to the task feeling too big or boring.
Give a nudge that reframes or shrinks the task to make it feel manageable right now.""",

        "low_mood": f"""The user's mood is {context.get('mood', 2)}/5 and energy is {context.get('energy', 2)}/5 right now.
They still have tasks pending. With ADHD, forcing productivity when depleted often backfires.
Give a gentle nudge — maybe suggest a body reset (2 min walk, water, snack) before returning, or doing their easiest task.""",

        "session_start": f"""The user is about to start a focus session on '{context.get('task', 'their task')}'.
Their mood is {context.get('mood', 3)}/5, energy is {context.get('energy', 3)}/5.
Give a quick motivating opener — help them prime their ADHD brain for focus. Keep it punchy.""",

        "end_of_day": f"""The user is ending their day. They completed {context.get('completed', 0)} out of {context.get('total', 0)} planned sessions.
Give a brief, compassionate end-of-day reflection. Celebrate what happened, don't dwell on what didn't.""",

        "pattern_insight": f"""The user has been identified as '{context.get('pattern', 'The Avoider')}' based on their ADHD focus patterns.
Give a brief, insight-driven nudge that acknowledges this pattern and offers one strategy to work WITH it, not against it.""",
    }

    return prompts.get(trigger_reason, f"Give a general ADHD focus nudge for someone who is struggling today. Context: {context}")


def generate_nudge(api_key: str, trigger_reason: str, context: dict) -> str:
    """
    Calls Groq LLaMA 3.3 to generate a personalized ADHD nudge.
    Returns the nudge text string.
    """
    try:
        client = get_groq_client(api_key)
        user_prompt = build_nudge_prompt(trigger_reason, context)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.85,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        fallback_nudges = {
            "short_abandon": "Your brain didn't fail — it got overwhelmed. Try just opening the file and reading one sentence. That's the whole task. 🧠",
            "consecutive_fails": "Three stops in a row? That's your brain telling you something. Take 5 minutes away from screens, then come back to your tiniest task.",
            "high_prio_delayed": "The task isn't the enemy — the size of it is. What's the smallest possible first step? Do only that. Not the whole thing.",
            "low_mood": "Low energy + ADHD is a tough combo. Grab water, move your body for 2 minutes, then return. Your brain needs fuel before focus.",
            "session_start": "You showed up — that's already a win. Set a timer for 10 minutes and just begin. You don't have to finish. Just start. 🎯",
            "end_of_day": "You tried today. Whatever happened, the act of trying matters. Rest well — tomorrow your brain gets a fresh start.",
        }
        return fallback_nudges.get(trigger_reason, "One small step is enough. You've got this. 💙")


def generate_weekly_insight(api_key: str, stats: dict) -> str:
    """Generates a weekly pattern insight using LLaMA."""
    try:
        client = get_groq_client(api_key)
        prompt = f"""Analyze this ADHD focus data from the past week and give one key insight + one actionable tip:
- Total focus sessions: {stats.get('total_sessions', 0)}
- Completion rate: {stats.get('completion_rate', 0):.0%}
- Average session length: {stats.get('avg_session_minutes', 0):.0f} minutes
- Average mood: {stats.get('avg_mood', 3):.1f}/5
- Average energy: {stats.get('avg_energy', 3):.1f}/5
- Best focus hour: {stats.get('best_hour', 'unknown')}
- Most delayed category: {stats.get('most_delayed_category', 'unknown')}

Keep it under 4 sentences. Be specific to ADHD patterns."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Keep tracking your sessions — patterns will emerge soon! (API error: {str(e)[:50]})"
