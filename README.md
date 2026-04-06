---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - email-triage
  - ai-agent
  - nlp
app_port: 7860
---

<div align="center">

# 📧 Email Triage OpenEnv

**A real-world reinforcement learning environment where AI agents learn to manage email inboxes**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen?style=for-the-badge)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow?style=for-the-badge)](https://huggingface.co/spaces/dishu1999/email-triage-env)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge)](https://docker.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-49%20Passing-success?style=for-the-badge)](#testing)

[🚀 Live Demo](https://dishu1999-email-triage-env.hf.space/docs) • [📖 API Docs](https://dishu1999-email-triage-env.hf.space/docs) • [🏥 Health Check](https://dishu1999-email-triage-env.hf.space/health)

</div>

---

## 🌍 Why Email Triage?

Email management is one of the most **universal and measurable** real-world tasks:

- 📬 Knowledge workers spend **28% of their workday** managing email
- ✅ Every email has a **clear correct answer** — making grading deterministic
- 📊 Natural **partial credit** — getting 3/5 right is objectively better than 1/5
- 📈 **Scales in difficulty** — from simple classification to complex multi-action management
- 🤖 Models trained here transfer directly to **real email assistant products**

---

## 🎯 Environment Overview

The environment simulates a realistic **corporate inbox** containing:

| Email Type | Example |
|-----------|---------|
| 🚨 Critical incidents | Production database down, deploy failures |
| 👔 Executive requests | CEO asking for board presentation slides |
| 💰 Finance | Overdue invoices, expense approvals |
| 🔒 Security | Mandatory password resets, 2FA enrollment |
| 🗑️ Spam | Prize scams, get-rich-quick schemes |
| 📰 Newsletters | Weekly digests, industry updates |
| 🔗 Thread chains | Follow-up emails in the same thread |

---

## 🎮 Tasks

### 📗 Task 1 — Email Classification `(Easy)`
> Classify 3 emails into the correct categories

- **Email IDs:** `t1-e001`, `t1-e002`, `t1-e003`
- **Categories:** `spam` · `urgent` · `normal` · `newsletter`
- **Scoring:** +0.33 reward per correct classification
- **Max steps:** 10
- **Expected score:** 0.33 (random) → 1.0 (perfect)

```json
{"action_type": "classify", "email_id": "t1-e001", "value": "spam"}
```

---

### 📙 Task 2 — Prioritize and Label Inbox `(Medium)`
> Assign priority, apply labels, and draft a reply to the most urgent email

- **Email IDs:** `t2-e001` through `t2-e005`
- **Scoring breakdown:**

| Component | Weight |
|-----------|--------|
| Priority correct (5 emails) | 0.50 |
| Label correct (5 emails) | 0.30 |
| Reply quality (keyword coverage) | 0.20 |

- **Max steps:** 20

```json
{"action_type": "prioritize", "email_id": "t2-e001", "value": "high"}
{"action_type": "label", "email_id": "t2-e001", "value": "action_required"}
{"action_type": "reply", "email_id": "t2-e001", "value": "I acknowledge the invoice and will confirm payment."}
```

---

### 📕 Task 3 — Full Inbox Management `(Hard)`
> Manage a 10-email inbox end-to-end

- **Email IDs:** `t3-e001` through `t3-e010`
- **Scoring breakdown:**

| Component | Weight | Description |
|-----------|--------|-------------|
| Category | 0.30 | Classify all 10 emails correctly |
| Priority | 0.20 | Assign correct priority to all 10 |
| Label | 0.10 | Apply correct label to all 10 |
| Duplicate detection | 0.10 | Identify threaded email chains |
| Action items | 0.10 | Flag all 5 action-required emails |
| Reply quality | 0.15 | Draft contextual replies to 3 urgent emails |
| Archiving | 0.05 | Archive spam, newsletters, resolved emails |
| **Total** | **1.00** | |

- **Max steps:** 40

---

## 🔌 API Reference

### Base URL
```
https://dishu1999-email-triage-env.hf.space
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Take an action |
| `GET` | `/state` | Get current state and score |
| `GET` | `/docs` | Interactive API explorer |

---

## 🎬 Quick Start

### Reset to a task
```bash
curl -X POST https://dishu1999-email-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classify_emails", "seed": 42}'
```

### Take a step
```bash
curl -X POST https://dishu1999-email-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify", "email_id": "t1-e001", "value": "spam"}}'
```

### Check your score
```bash
curl https://dishu1999-email-triage-env.hf.space/state
```

---

## 🧩 Action Space

| Action | Value | Description |
|--------|-------|-------------|
| `classify` | `spam` `urgent` `normal` `newsletter` `finance` `hr` `tech_support` `social` | Assign email category |
| `prioritize` | `high` `medium` `low` | Set priority level |
| `label` | `action_required` `fyi` `waiting` `resolved` `duplicate` `archived` | Apply label |
| `reply` | Any text string | Draft a reply |
| `archive` | `null` | Move to archive |
| `flag` | `null` | Flag as action item |
| `done` | `null` | End episode early |

---

## 👁️ Observation Space

Each observation returned by `reset()` and `step()` contains:

```json
{
  "inbox": [
    {
      "id": "t1-e001",
      "subject": "CONGRATULATIONS! You've won $5,000,000!!!",
      "sender": "winner-notify@prize-claims99.com",
      "sender_name": "Prize Notification Center",
      "body": "...",
      "timestamp": "2024-03-10T08:15:00Z",
      "has_attachment": false,
      "is_reply": false,
      "thread_id": null,
      "category": null,
      "priority": null,
      "label": null,
      "reply_draft": null,
      "archived": false,
      "flagged": false
    }
  ],
  "current_task": "classify_emails",
  "step_count": 0,
  "max_steps": 10,
  "task_info": {
    "task_name": "Email Classification",
    "difficulty": "easy",
    "description": "...",
    "objectives": ["..."],
    "hints": ["..."],
    "score_so_far": 0.0
  },
  "last_action_result": null,
  "last_action_error": null,
  "done": false
}
```

---

## 🏆 Reward Function

Rewards are issued **at every step** — not just at episode end:

```
✅ Correct action    → immediate partial reward proportional to its weight
❌ Wrong action      → 0.0 reward
⚠️ Invalid action   → 0.0 reward + small penalty
📝 Reply quality    → graded by keyword coverage in reply text
```

This provides **dense reward signals** throughout the trajectory enabling effective RL training.

---

## 📊 Baseline Scores

| Task | Difficulty | Score | Steps | Success |
|------|-----------|-------|-------|---------|
| `classify_emails` | Easy | 1.00 | 3 | ✅ |
| `prioritize_and_label` | Medium | 0.97 | 12 | ✅ |
| `full_inbox_management` | Hard | 0.97 | 40 | ✅ |

---

## 🐍 Python SDK Example

```python
import requests

BASE = "https://dishu1999-email-triage-env.hf.space"

# Start Task 1
obs = requests.post(f"{BASE}/reset", json={"task_id": "classify_emails"}).json()
print("Emails:", [e["subject"] for e in obs["inbox"]])

# Classify spam email
result = requests.post(f"{BASE}/step", json={
    "action": {
        "action_type": "classify",
        "email_id": "t1-e001",
        "value": "spam"
    }
}).json()

print("Reward:", result["reward"]["value"])   # 0.333
print("Reason:", result["reward"]["reason"])  # Correct! t1-e001 is indeed 'spam'.

# Check score
state = requests.get(f"{BASE}/state").json()
print("Grade:", state["grade"])
```

---

## 🗂️ Project Structure

```
email-triage-env/
├── 📄 openenv.yaml          # OpenEnv spec metadata
├── 🐳 Dockerfile            # Container definition
├── 📦 requirements.txt      # Python dependencies
├── 🤖 inference.py          # Baseline inference script
├── 📖 README.md
└── env/
    ├── 📐 models.py          # Pydantic: Observation, Action, Reward
    ├── 🌍 environment.py     # reset() · step() · state() · grade()
    ├── 📧 data.py            # 18 real emails + ground truth labels
    ├── 🚀 server.py          # FastAPI HTTP server
    └── tasks/
        ├── 📗 task1_easy.py   # Email classification
        ├── 📙 task2_medium.py # Prioritize + label + reply
        └── 📕 task3_hard.py   # Full inbox management
```

---

## 🧪 Testing

```bash
# Run all 49 tests
python tests/test_all.py
```

```
✓ All 49 tests passed
```

---

## 🚀 Local Setup

```bash
# Clone and install
git clone https://huggingface.co/spaces/dishu1999/email-triage-env
cd email-triage-env
pip install -r requirements.txt

# Start server
python -m uvicorn env.server:app --host 0.0.0.0 --port 7860

# Open docs
http://localhost:7860/docs
```

---

## 🏗️ Built With

- **FastAPI** — HTTP server
- **Pydantic** — typed models
- **Python 3.11** — runtime
- **Docker** — containerization
- **HuggingFace Spaces** — deployment

---

<div align="center">

Made with ❤️ for the OpenEnv Hackathon

**[🌐 Live Space](https://huggingface.co/spaces/dishu1999/email-triage-env) • [📖 API Docs](https://dishu1999-email-triage-env.hf.space/docs) • [🏥 Health](https://dishu1999-email-triage-env.hf.space/health)**

</div>
