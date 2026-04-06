# Email Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-brightgreen)](https://openenv.dev)
[![HuggingFace Spaces](https://huggingface.co/spaces/dishu1999/email-triage-env)](https://huggingface.co/spaces)

An **OpenEnv-compliant reinforcement learning environment** where AI agents learn to manage a real-world email inbox. Agents triage, classify, prioritize, reply to, and archive emails — a task knowledge workers perform dozens of times daily.

---

## Why Email Triage?

Email management is one of the most universal and measurable real-world tasks:
- **Clear ground truth**: every email has a correct classification and priority
- **Natural partial credit**: getting 3/5 emails right is objectively better than 1/5
- **Scalable difficulty**: simple classification → complex multi-action inbox management
- **Immediate real-world utility**: models trained here transfer directly to email assistant products

---

## Environment Description

The environment simulates a realistic corporate inbox with emails spanning:
- Critical production incidents
- CEO-level meeting requests
- Overdue invoices
- Security compliance notices
- Spam and phishing attempts
- Newsletters and FYI notices
- Threaded reply chains (for duplicate detection)

Agents interact via a standard HTTP API implementing the OpenEnv spec.

---

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `inbox` | `List[Email]` | Current inbox with all emails and agent-assigned fields |
| `current_task` | `str` | Active task identifier |
| `step_count` | `int` | Steps taken this episode |
| `max_steps` | `int` | Maximum steps allowed |
| `task_info` | `TaskInfo` | Objectives, hints, actions taken, score so far |
| `last_action_result` | `str\|null` | Feedback from last action |
| `last_action_error` | `str\|null` | Error message if last action was invalid |
| `done` | `bool` | Whether episode has ended |

Each `Email` object contains: `id`, `subject`, `sender`, `sender_name`, `body`, `timestamp`, `has_attachment`, `is_reply`, `thread_id`, plus agent-assigned fields: `category`, `priority`, `label`, `reply_draft`, `archived`, `flagged`.

---

## Action Space

```json
{
  "action_type": "classify|prioritize|label|reply|archive|flag|skip|done",
  "email_id": "email-id-string or null",
  "value": "category/priority/label/reply-text or null"
}
```

| Action | Value | Description |
|--------|-------|-------------|
| `classify` | `spam\|urgent\|normal\|newsletter\|finance\|hr\|tech_support\|social` | Assign category |
| `prioritize` | `high\|medium\|low` | Set priority level |
| `label` | `action_required\|fyi\|waiting\|resolved\|duplicate\|archived` | Apply label |
| `reply` | Reply text string | Draft a reply to the email |
| `archive` | null | Move email to archive |
| `flag` | null | Flag as action item |
| `skip` | null | Skip current step (no reward) |
| `done` | null | End the episode early |

---

## Tasks

### Task 1 — Email Classification (Easy)
**ID**: `classify_emails`  
**Max steps**: 10  
**Reward threshold**: 0.8

Classify 3 emails (spam, urgent, newsletter) into their correct categories.

- Agent earns **1/3 reward per correct classification**
- Deterministic ground truth per email
- Expected baseline score: ~0.33 (random) → 1.0 (perfect)

**Grader**: `correct_classifications / total_emails`

---

### Task 2 — Prioritize and Label Inbox (Medium)
**ID**: `prioritize_and_label`  
**Max steps**: 20  
**Reward threshold**: 0.7

Given 5 emails: assign priority (high/medium/low), apply labels, and draft a contextual reply to the most urgent email.

Scoring breakdown:
- Priority accuracy: **5 × 0.10 = 0.50**
- Label accuracy: **5 × 0.06 = 0.30**
- Reply quality (keyword coverage): **up to 0.20**

**Grader**: Weighted sum of priority + label + reply scores, normalized to [0.0, 1.0]

---

### Task 3 — Full Inbox Management (Hard)
**ID**: `full_inbox_management`  
**Max steps**: 40  
**Reward threshold**: 0.6

Manage a 10-email inbox end-to-end: classify, prioritize, label, detect duplicate threads, flag action items, draft replies, archive noise.

Scoring breakdown:
| Component | Weight |
|-----------|--------|
| Category correct (10 emails) | 0.30 |
| Priority correct (10 emails) | 0.20 |
| Label correct (10 emails) | 0.10 |
| Duplicate thread detection | 0.10 |
| Action items flagged (5 items) | 0.10 |
| Reply quality (3 emails) | 0.15 |
| Correct archiving | 0.05 |
| **Total** | **1.00** |

**Grader**: Multi-component weighted sum, deterministic and reproducible

---

## Reward Function

Rewards are issued **at every step** (not just at episode end):
- Correct action → step reward proportional to its weight in the scoring rubric
- Incorrect action → 0.0 (with small penalty for redundant/invalid actions)
- Reply quality → graded by keyword coverage in the reply text

This provides **dense partial credit signals** throughout the trajectory, enabling effective RL training.

---

## Baseline Scores

Baseline run using `gpt-4o-mini`:

| Task | Score | Steps | Success |
|------|-------|-------|---------|
| classify_emails | ~1.00 | 4 | ✓ |
| prioritize_and_label | ~0.72 | 12 | ✓ |
| full_inbox_management | ~0.58 | 35 | ✓ |

---

## Setup & Usage

### Local development

```bash
# Clone and install
git clone https://huggingface.co/spaces/your-username/email-triage-env
cd email-triage-env
pip install -r requirements.txt

# Start the server
uvicorn env.server:app --host 0.0.0.0 --port 7860 --reload

# In another terminal, run inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-key"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# Run inference against local container
export ENV_BASE_URL="http://localhost:7860"
export HF_TOKEN="sk-your-key"
python inference.py
```

### API Usage

```bash
# Reset to task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classify_emails", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify", "email_id": "t1-e001", "value": "spam"}}'

# Check state
curl http://localhost:7860/state
```

### Python SDK

```python
import requests

BASE = "http://localhost:7860"

# Reset
obs = requests.post(f"{BASE}/reset", json={"task_id": "classify_emails"}).json()

# Step
result = requests.post(f"{BASE}/step", json={
    "action": {"action_type": "classify", "email_id": "t1-e001", "value": "spam"}
}).json()

print(result["reward"])  # {"value": 0.333, "cumulative": 0.333, "reason": "Correct!"}
```

---

## Project Structure

```
email-triage-env/
├── openenv.yaml          # OpenEnv metadata and task registry
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── inference.py          # Baseline inference script (required at root)
├── README.md
└── env/
    ├── __init__.py
    ├── models.py          # Pydantic: Observation, Action, Reward
    ├── environment.py     # EmailTriageEnv: reset/step/state/grade
    ├── data.py            # Email datasets + ground truth
    ├── server.py          # FastAPI HTTP server
    └── tasks/
        ├── __init__.py
        ├── task1_easy.py   # Email classification
        ├── task2_medium.py # Prioritize + label + reply
        └── task3_hard.py   # Full inbox management
```

---

## HuggingFace Space Deployment

This environment is deployed as a HuggingFace Space tagged with `openenv`.

Required Space secrets:
- None required for the environment itself
- For inference: set `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME` as Space secrets

---

## License

MIT
