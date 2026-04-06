"""
Inference Script — Email Triage OpenEnv
=======================================
Runs a language model agent against all 3 tasks and emits structured stdout logs.

Required environment variables:
    API_BASE_URL   LLM endpoint (default: https://api.openai.com/v1)
    MODEL_NAME     Model identifier (default: gpt-4o-mini)
    HF_TOKEN       API key (used as OpenAI API key)

Stdout format (strictly enforced):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import traceback
from typing import Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# The environment server URL (local or deployed HF Space)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK_NAME = "email-triage-env"
SUCCESS_THRESHOLD = 0.5  # grade >= this is considered success

ALL_TASK_IDS = [
    "classify_emails",
    "prioritize_and_label",
    "full_inbox_management",
]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=HF_TOKEN or "sk-placeholder",
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Environment client helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str, seed: int = 42) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> dict:
    resp = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Agent: LLM-driven action selection
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert email triage agent. You manage inboxes by classifying, prioritizing, labeling, flagging, replying to, and archiving emails.

You will receive the current inbox state and task description as JSON. You must respond with a single JSON action object.

Action format:
{"action_type": "<type>", "email_id": "<id or null>", "value": "<value or null>"}

Valid action_types: classify, prioritize, label, reply, archive, flag, skip, done

Valid categories: spam, urgent, normal, newsletter, finance, hr, tech_support, social
Valid priorities: high, medium, low
Valid labels: action_required, fyi, waiting, resolved, duplicate, archived

Rules:
- Always look at the task description and objectives first
- For 'classify': set value to the category name
- For 'prioritize': set value to high/medium/low
- For 'label': set value to the label name
- For 'reply': set value to a professional reply text
- For 'archive', 'flag': value can be null
- When all tasks are done, use action_type='done'
- Respond ONLY with a valid JSON object, no other text
"""


def build_user_prompt(obs: dict) -> str:
    """Build the user message from current observation."""
    task_info = obs.get("task_info", {})
    inbox = obs.get("inbox", [])

    emails_summary = []
    for email in inbox:
        emails_summary.append({
            "id": email["id"],
            "subject": email["subject"],
            "sender": email["sender_name"],
            "body_preview": email["body"][:200],
            "has_attachment": email.get("has_attachment", False),
            "is_reply": email.get("is_reply", False),
            "thread_id": email.get("thread_id"),
            "current_category": email.get("category"),
            "current_priority": email.get("priority"),
            "current_label": email.get("label"),
            "archived": email.get("archived", False),
            "flagged": email.get("flagged", False),
        })

    prompt_data = {
        "task": task_info.get("task_name"),
        "difficulty": task_info.get("difficulty"),
        "description": task_info.get("description"),
        "objectives": task_info.get("objectives", []),
        "hints": task_info.get("hints", []),
        "actions_taken_so_far": task_info.get("actions_taken", []),
        "step": obs.get("step_count"),
        "max_steps": obs.get("max_steps"),
        "last_result": obs.get("last_action_result"),
        "last_error": obs.get("last_action_error"),
        "inbox": emails_summary,
    }
    return json.dumps(prompt_data, indent=2)


def get_agent_action(obs: dict) -> dict:
    """Ask the LLM for the next action."""
    user_message = build_user_prompt(obs)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    action = json.loads(raw)
    return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str, max_steps: int = 40, seed: int = 42) -> dict:
    """Run one full episode and return results."""
    rewards = []
    steps = 0
    last_error = None
    success = False

    # [START]
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env_reset(task_id, seed=seed)
        done = obs.get("done", False)

        while not done and steps < max_steps:
            # Get action from LLM
            try:
                action = get_agent_action(obs)
            except Exception as e:
                action = {"action_type": "skip", "email_id": None, "value": None}
                last_error = f"LLM error: {e}"

            action_str = json.dumps(action, separators=(",", ":"))

            # Apply action
            try:
                result = env_step(action)
                reward_val = result["reward"]["value"]
                done = result["done"]
                last_error = result["observation"].get("last_action_error")
                obs = result["observation"]
            except Exception as e:
                reward_val = 0.0
                last_error = f"step error: {e}"
                done = True

            steps += 1
            rewards.append(reward_val)

            # [STEP]
            error_str = last_error if last_error else "null"
            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward_val:.2f} done={'true' if done else 'false'} "
                f"error={error_str}",
                flush=True,
            )

        # Determine success from final grade
        state = env_state()
        grade = state.get("grade", 0.0)
        success = grade >= SUCCESS_THRESHOLD

    except Exception as e:
        last_error = str(e)
        traceback.print_exc(file=sys.stderr)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # [END]
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps,
        "rewards": rewards,
        "total_reward": sum(rewards),
    }


# ---------------------------------------------------------------------------
# Main: run all 3 tasks
# ---------------------------------------------------------------------------

def main():
    print(f"# Email Triage OpenEnv — Baseline Inference", flush=True)
    print(f"# Model: {MODEL_NAME}", flush=True)
    print(f"# API Base: {API_BASE_URL}", flush=True)
    print(f"# Env URL: {ENV_BASE_URL}", flush=True)
    print("", flush=True)

    results = []
    for task_id in ALL_TASK_IDS:
        result = run_episode(task_id)
        results.append(result)
        print("", flush=True)

    # Summary
    print("# ---- SUMMARY ----", flush=True)
    total_success = sum(1 for r in results if r["success"])
    for r in results:
        print(
            f"# {r['task_id']}: success={r['success']} "
            f"steps={r['steps']} total_reward={r['total_reward']:.2f}",
            flush=True,
        )
    print(f"# Tasks passed: {total_success}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
