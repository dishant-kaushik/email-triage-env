"""
Inference Script — Email Triage OpenEnv
"""

import json
import os
import sys
import traceback

import requests

# Read exactly what the validator injects
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "sk-placeholder")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Normalize the base URL — strip trailing slash, ensure it ends with /v1
_base = API_BASE_URL.rstrip("/")
if not _base.endswith("/v1"):
    _base = _base + "/v1"
API_BASE_URL_NORMALIZED = _base

BENCHMARK_NAME = "email-triage-env"
SUCCESS_THRESHOLD = 0.5

ALL_TASK_IDS = [
    "classify_emails",
    "prioritize_and_label",
    "full_inbox_management",
]

# --- LLM calls via raw HTTP (no openai SDK) to avoid client init crashes ---

def call_llm_raw(messages):
    """Call the LLM proxy directly via requests — bypasses OpenAI SDK init issues."""
    url = f"{API_BASE_URL_NORMALIZED}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# --- Also try OpenAI SDK as secondary method ---
try:
    from openai import OpenAI
    _sdk_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL_NORMALIZED)
except Exception as e:
    sys.stderr.write(f"OpenAI SDK init warning: {e}\n")
    _sdk_client = None


def env_reset(task_id, seed=42):
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action):
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state():
    resp = requests.get(f"{ENV_BASE_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


SYSTEM_PROMPT = """You are an expert email triage agent. Respond ONLY with a valid JSON action object and nothing else.
Action format: {"action_type": "<type>", "email_id": "<id or null>", "value": "<value or null>"}
Valid action_types: classify, prioritize, label, reply, archive, flag, skip, done
Valid categories: spam, urgent, normal, newsletter, finance, hr, tech_support, social
Valid priorities: high, medium, low
Valid labels: action_required, fyi, waiting, resolved, duplicate, archived
Rules:
- classify each email first before other actions
- urgent emails get high priority and action_required label
- spam/newsletter emails get low priority and archived label
- call done when all emails are handled"""


def get_rule_based_action(obs):
    inbox = obs.get("inbox", [])
    for email in inbox:
        eid = email["id"]
        body = (email.get("body", "") + " " + email.get("subject", "")).lower()
        if email.get("category") is None:
            if any(w in body for w in ["prize", "winner", "lottery", "rich", "spam", "earn now"]):
                return {"action_type": "classify", "email_id": eid, "value": "spam"}
            elif any(w in body for w in ["critical", "urgent", "alert", "down", "prod", "deploy", "ceo", "board", "incident", "overdue", "invoice"]):
                return {"action_type": "classify", "email_id": eid, "value": "urgent"}
            elif any(w in body for w in ["newsletter", "unsubscribe", "digest", "weekly", "issue #"]):
                return {"action_type": "classify", "email_id": eid, "value": "newsletter"}
            else:
                return {"action_type": "classify", "email_id": eid, "value": "normal"}
        if email.get("priority") is None:
            cat = email.get("category", "normal")
            if cat == "urgent":
                return {"action_type": "prioritize", "email_id": eid, "value": "high"}
            elif cat in ("spam", "newsletter"):
                return {"action_type": "prioritize", "email_id": eid, "value": "low"}
            else:
                return {"action_type": "prioritize", "email_id": eid, "value": "medium"}
        if email.get("label") is None:
            cat = email.get("category", "normal")
            if cat == "urgent":
                return {"action_type": "label", "email_id": eid, "value": "action_required"}
            elif cat in ("spam", "newsletter"):
                return {"action_type": "label", "email_id": eid, "value": "archived"}
            else:
                return {"action_type": "label", "email_id": eid, "value": "fyi"}
        if not email.get("flagged") and email.get("category") == "urgent":
            return {"action_type": "flag", "email_id": eid, "value": None}
        if email.get("reply_draft") is None and email.get("category") == "urgent":
            return {"action_type": "reply", "email_id": eid, "value": "Thank you for reaching out. I will action this request promptly."}
        if not email.get("archived") and email.get("category") in ("spam", "newsletter"):
            return {"action_type": "archive", "email_id": eid, "value": None}
    return {"action_type": "done", "email_id": None, "value": None}


def build_messages(obs):
    inbox = obs.get("inbox", [])
    task_info = obs.get("task_info", {})
    prompt_data = {
        "task": task_info.get("task_name"),
        "description": task_info.get("description"),
        "objectives": task_info.get("objectives", []),
        "hints": task_info.get("hints", []),
        "actions_taken": task_info.get("actions_taken", []),
        "step": obs.get("step_count"),
        "max_steps": obs.get("max_steps"),
        "inbox": [
            {
                "id": e["id"],
                "subject": e["subject"],
                "sender": e["sender_name"],
                "body": e["body"][:200],
                "category": e.get("category"),
                "priority": e.get("priority"),
                "label": e.get("label"),
            }
            for e in inbox
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt_data)},
    ]


def parse_llm_response(raw):
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def get_agent_action(obs):
    messages = build_messages(obs)

    # Try raw HTTP first (most reliable — no SDK init issues)
    try:
        raw = call_llm_raw(messages)
        return parse_llm_response(raw)
    except Exception as e:
        sys.stderr.write(f"Raw HTTP LLM failed: {e}\n")

    # Try OpenAI SDK as backup
    if _sdk_client is not None:
        try:
            response = _sdk_client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=256,
                messages=messages,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            return parse_llm_response(raw)
        except Exception as e:
            sys.stderr.write(f"SDK LLM failed: {e}\n")

    # Final fallback
    return get_rule_based_action(obs)


def run_episode(task_id, max_steps=40, seed=42):
    rewards = []
    steps = 0
    last_error = None
    success = False

    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env_reset(task_id, seed=seed)
        done = obs.get("done", False)

        while not done and steps < max_steps:
            try:
                action = get_agent_action(obs)
            except Exception as e:
                action = {"action_type": "skip", "email_id": None, "value": None}
                last_error = f"agent error: {e}"

            action_str = json.dumps(action, separators=(",", ":"))

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
            error_str = last_error if last_error else "null"
            print(
                f"[STEP] step={steps} action={action_str} reward={reward_val:.2f} "
                f"done={'true' if done else 'false'} error={error_str}",
                flush=True,
            )

        try:
            state = env_state()
            grade = state.get("grade", 0.0)
            success = grade >= SUCCESS_THRESHOLD
        except Exception:
            success = sum(rewards) > 0

    except Exception as e:
        sys.stderr.write(traceback.format_exc())

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} rewards={rewards_str}",
        flush=True,
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps,
        "rewards": rewards,
        "total_reward": sum(rewards),
    }


def main():
    print(f"# Email Triage OpenEnv — Baseline Inference", flush=True)
    print(f"# Model: {MODEL_NAME}", flush=True)
    print(f"# API Base: {API_BASE_URL}", flush=True)
    print(f"# API Base Normalized: {API_BASE_URL_NORMALIZED}", flush=True)
    print(f"# Env URL: {ENV_BASE_URL}", flush=True)
    print("", flush=True)

    results = []
    for task_id in ALL_TASK_IDS:
        try:
            result = run_episode(task_id)
            results.append(result)
        except Exception as e:
            sys.stderr.write(f"Episode failed: {e}\n")
            print(f"[END] success=false steps=0 rewards=0.00", flush=True)
            results.append({
                "task_id": task_id,
                "success": False,
                "steps": 0,
                "rewards": [],
                "total_reward": 0.0,
            })
        print("", flush=True)

    print("# ---- SUMMARY ----", flush=True)
    for r in results:
        print(
            f"# {r['task_id']}: success={r['success']} steps={r['steps']} "
            f"total_reward={r['total_reward']:.2f}",
            flush=True,
        )
    print(f"# Tasks passed: {sum(1 for r in results if r['success'])}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
