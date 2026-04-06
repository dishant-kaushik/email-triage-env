"""
Email dataset used across all tasks.
Ground truth labels enable deterministic grading.
"""

from typing import Dict, List
from env.models import Email

# ---------------------------------------------------------------------------
# Task 1 — Easy: 3-email classification set
# ---------------------------------------------------------------------------

TASK1_EMAILS: List[Dict] = [
    {
        "id": "t1-e001",
        "subject": "CONGRATULATIONS! You've won $5,000,000!!!",
        "sender": "winner-notify@prize-claims99.com",
        "sender_name": "Prize Notification Center",
        "body": (
            "Dear Lucky Winner,\n\n"
            "Your email has been selected in our international lottery draw. "
            "To claim your prize of FIVE MILLION DOLLARS, please send your bank "
            "details and a processing fee of $199 to the address below immediately.\n\n"
            "Act NOW before your prize expires!\n\nPrize Dept."
        ),
        "timestamp": "2024-03-10T08:15:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t1-e002",
        "subject": "Server down — prod DB unresponsive",
        "sender": "alerts@monitoring.company.com",
        "sender_name": "PagerDuty Alert",
        "body": (
            "CRITICAL ALERT — Production database (db-prod-01) has been unresponsive "
            "for 4 minutes. Error rate: 100%. All user-facing services affected.\n\n"
            "Assigned to on-call: devops-team\n"
            "Incident ID: INC-8821\n\n"
            "Please acknowledge and investigate immediately."
        ),
        "timestamp": "2024-03-10T09:02:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t1-e003",
        "subject": "This week's top articles — TechDigest Newsletter",
        "sender": "newsletter@techdigest.io",
        "sender_name": "TechDigest Weekly",
        "body": (
            "Hi there,\n\nHere are this week's most-read stories:\n\n"
            "• How AI is changing software development\n"
            "• 10 productivity hacks for remote workers\n"
            "• The future of quantum computing\n\n"
            "Read more at techdigest.io\n\nUnsubscribe | Manage preferences"
        ),
        "timestamp": "2024-03-10T07:30:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
]

TASK1_GROUND_TRUTH: Dict[str, str] = {
    "t1-e001": "spam",
    "t1-e002": "urgent",
    "t1-e003": "newsletter",
}

# ---------------------------------------------------------------------------
# Task 2 — Medium: 5-email prioritization + label + reply
# ---------------------------------------------------------------------------

TASK2_EMAILS: List[Dict] = [
    {
        "id": "t2-e001",
        "subject": "Invoice #4521 overdue — payment required today",
        "sender": "billing@acme-supplies.com",
        "sender_name": "ACME Billing Dept",
        "body": (
            "Dear Accounts Team,\n\n"
            "This is a final notice regarding invoice #4521 for $12,450 due on March 1st. "
            "Your account is now 10 days overdue. Failure to pay within 24 hours will result "
            "in service suspension and a 5% late fee.\n\n"
            "Please confirm payment or contact us urgently.\n\nACME Billing"
        ),
        "timestamp": "2024-03-10T08:00:00Z",
        "has_attachment": True,
        "is_reply": False,
    },
    {
        "id": "t2-e002",
        "subject": "Team lunch Friday — are you coming?",
        "sender": "mike.jones@company.com",
        "sender_name": "Mike Jones",
        "body": (
            "Hey,\n\nWe're organizing a team lunch this Friday at 1pm at The Green Fork "
            "(downtown). RSVP by Thursday EOD so we can book the right table size.\n\n"
            "Let me know!\nMike"
        ),
        "timestamp": "2024-03-10T09:15:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t2-e003",
        "subject": "RE: Security audit — action required by you",
        "sender": "security@company.com",
        "sender_name": "IT Security Team",
        "body": (
            "Hi,\n\nFollowing our Q1 security audit, your account has been flagged for "
            "mandatory password reset and 2FA enrollment. You must complete this by "
            "March 15 or your access will be suspended.\n\n"
            "Steps:\n1. Go to account.company.com\n2. Reset password\n3. Enroll in 2FA\n\n"
            "IT Security"
        ),
        "timestamp": "2024-03-10T10:00:00Z",
        "has_attachment": False,
        "is_reply": True,
        "thread_id": "thread-sec-001",
    },
    {
        "id": "t2-e004",
        "subject": "FYI: Updated holiday schedule 2024",
        "sender": "hr@company.com",
        "sender_name": "HR Department",
        "body": (
            "Hello everyone,\n\nPlease find attached the updated company holiday schedule "
            "for 2024. No action required — this is for your reference only.\n\n"
            "Key dates:\n• April 19: Good Friday\n• May 27: Memorial Day\n• July 4: Independence Day\n\n"
            "HR Team"
        ),
        "timestamp": "2024-03-10T11:00:00Z",
        "has_attachment": True,
        "is_reply": False,
    },
    {
        "id": "t2-e005",
        "subject": "Your subscription renewal — 3 days left",
        "sender": "noreply@saas-tool.com",
        "sender_name": "SaaS Tool Renewals",
        "body": (
            "Hi,\n\nYour SaaS Tool Pro subscription expires in 3 days on March 13. "
            "To avoid interruption, please renew at saas-tool.com/renew.\n\n"
            "Current plan: Pro ($99/month)\n"
            "Renewal date: March 13, 2024\n\nSaaS Tool Team"
        ),
        "timestamp": "2024-03-10T12:00:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
]

TASK2_GROUND_TRUTH: Dict[str, Dict] = {
    "t2-e001": {"priority": "high",   "label": "action_required"},
    "t2-e002": {"priority": "low",    "label": "fyi"},
    "t2-e003": {"priority": "high",   "label": "action_required"},
    "t2-e004": {"priority": "low",    "label": "fyi"},
    "t2-e005": {"priority": "medium", "label": "action_required"},
}
# Most urgent email requiring a reply
TASK2_REPLY_TARGET = "t2-e001"
TASK2_REPLY_KEYWORDS = ["invoice", "payment", "acknowledge", "confirm", "contact", "billing"]

# ---------------------------------------------------------------------------
# Task 3 — Hard: 10-email full inbox management
# ---------------------------------------------------------------------------

TASK3_EMAILS: List[Dict] = [
    {
        "id": "t3-e001",
        "subject": "Q1 board presentation — slides needed ASAP",
        "sender": "ceo@company.com",
        "sender_name": "Jane CEO",
        "body": (
            "Hi,\n\nI need the Q1 financial slides for the board meeting tomorrow at 9am. "
            "Please send me the final deck by 6pm today. This is the most important "
            "presentation of the quarter.\n\nJane"
        ),
        "timestamp": "2024-03-10T08:00:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e002",
        "subject": "Spam: Make money from home — 100% guaranteed!",
        "sender": "rich-quick@earn-now99.net",
        "sender_name": "Earn Now",
        "body": (
            "Make $5000 a week from home! No experience needed. "
            "Click here to start your journey to financial freedom! "
            "Limited spots — ACT NOW!"
        ),
        "timestamp": "2024-03-10T08:05:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e003",
        "subject": "RE: Q1 board presentation — slides needed ASAP",
        "sender": "finance@company.com",
        "sender_name": "Finance Team",
        "body": (
            "Hi,\n\nJust following up on Jane's earlier request. We have the slides ready "
            "but need your sign-off before sending. Can you review and approve by 5pm?\n\n"
            "Finance Team"
        ),
        "timestamp": "2024-03-10T09:00:00Z",
        "has_attachment": True,
        "is_reply": True,
        "thread_id": "thread-board-q1",
    },
    {
        "id": "t3-e004",
        "subject": "RE: Q1 board presentation — slides needed ASAP",
        "sender": "pa@company.com",
        "sender_name": "Personal Assistant",
        "body": (
            "Adding to the chain — Jane also wants a one-page exec summary included. "
            "See her earlier email.\n\nPA"
        ),
        "timestamp": "2024-03-10T09:30:00Z",
        "has_attachment": False,
        "is_reply": True,
        "thread_id": "thread-board-q1",
    },
    {
        "id": "t3-e005",
        "subject": "Production deploy failed — rollback required",
        "sender": "ci-cd@deploys.company.com",
        "sender_name": "CI/CD Pipeline",
        "body": (
            "DEPLOY FAILED\n\nService: payment-service v2.4.1\n"
            "Stage: production\nError: Health check failed (exit code 1)\n"
            "Auto-rollback: DISABLED\n\n"
            "Manual intervention required. Approx. 200 users impacted.\n"
            "Incident: INC-9901"
        ),
        "timestamp": "2024-03-10T10:00:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e006",
        "subject": "Your March expense report — approved",
        "sender": "expenses@company.com",
        "sender_name": "Expenses System",
        "body": (
            "Your expense report EXP-2024-031 for $342.50 has been approved "
            "and will be reimbursed in your next payroll cycle.\n\nExpenses Team"
        ),
        "timestamp": "2024-03-10T10:30:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e007",
        "subject": "Team standup notes — March 10",
        "sender": "standup-bot@company.com",
        "sender_name": "Standup Bot",
        "body": (
            "Daily standup summary — March 10:\n\n"
            "• Alice: Working on API refactor, blocked on design review\n"
            "• Bob: Completed unit tests for auth module\n"
            "• Carol: Investigating customer bug report #4421\n\n"
            "Full notes: confluence.company.com/standup/2024-03-10"
        ),
        "timestamp": "2024-03-10T10:45:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e008",
        "subject": "Contract renewal — DocuSign signature needed",
        "sender": "esign@docusign.net",
        "sender_name": "DocuSign",
        "body": (
            "You have a document waiting for your signature:\n\n"
            "Document: Vendor Master Agreement — CloudProvider Inc\n"
            "Sent by: procurement@company.com\n"
            "Deadline: March 17, 2024\n\n"
            "Sign at: docusign.net/sign/xxxx"
        ),
        "timestamp": "2024-03-10T11:00:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e009",
        "subject": "Newsletter: Industry Weekly — Issue #88",
        "sender": "newsletter@industryweekly.com",
        "sender_name": "Industry Weekly",
        "body": (
            "This week in tech:\n\n"
            "• AI startups raised $2.1B in Q1\n"
            "• New regulations on data privacy\n"
            "• Top 10 tools for distributed teams\n\n"
            "Read online | Unsubscribe"
        ),
        "timestamp": "2024-03-10T07:00:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
    {
        "id": "t3-e010",
        "subject": "Alice needs design review unblocked — can you help?",
        "sender": "manager@company.com",
        "sender_name": "Bob Manager",
        "body": (
            "Hi,\n\nAlice is blocked on her API refactor because the design review "
            "hasn't been completed. She mentioned it in standup today. "
            "Could you take a look and unblock her today?\n\nBob"
        ),
        "timestamp": "2024-03-10T11:15:00Z",
        "has_attachment": False,
        "is_reply": False,
    },
]

TASK3_GROUND_TRUTH: Dict[str, Dict] = {
    "t3-e001": {"category": "urgent",   "priority": "high",   "label": "action_required", "needs_reply": True,  "archive": False},
    "t3-e002": {"category": "spam",     "priority": "low",    "label": "archived",        "needs_reply": False, "archive": True},
    "t3-e003": {"category": "urgent",   "priority": "high",   "label": "action_required", "needs_reply": True,  "archive": False, "thread_id": "thread-board-q1"},
    "t3-e004": {"category": "normal",   "priority": "medium", "label": "fyi",             "needs_reply": False, "archive": False, "thread_id": "thread-board-q1"},
    "t3-e005": {"category": "urgent",   "priority": "high",   "label": "action_required", "needs_reply": True,  "archive": False},
    "t3-e006": {"category": "normal",   "priority": "low",    "label": "resolved",        "needs_reply": False, "archive": True},
    "t3-e007": {"category": "normal",   "priority": "low",    "label": "fyi",             "needs_reply": False, "archive": True},
    "t3-e008": {"category": "normal",   "priority": "medium", "label": "action_required", "needs_reply": False, "archive": False},
    "t3-e009": {"category": "newsletter","priority": "low",   "label": "archived",        "needs_reply": False, "archive": True},
    "t3-e010": {"category": "urgent",   "priority": "high",   "label": "action_required", "needs_reply": True,  "archive": False},
}

TASK3_DUPLICATE_THREAD = "thread-board-q1"  # e003 and e004 duplicate thread with e001
TASK3_ACTION_ITEMS = ["t3-e001", "t3-e003", "t3-e005", "t3-e008", "t3-e010"]
TASK3_REPLY_KEYWORDS = {
    "t3-e001": ["board", "slides", "presentation", "6pm", "deck", "q1"],
    "t3-e005": ["rollback", "deploy", "production", "incident", "payment-service"],
    "t3-e010": ["design", "review", "alice", "unblock", "today"],
}
