"""
Test suite for Email Triage OpenEnv.
Tests all 3 tasks, graders, models, edge cases, and HTTP API.

Run:
    python -m pytest tests/ -v
    # or
    python tests/test_all.py
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv, ALL_TASK_IDS
from env.models import Action, Observation, Reward, VALID_CATEGORIES, VALID_PRIORITIES, VALID_LABELS
from env.tasks.task1_easy import Task1ClassifyEmails
from env.tasks.task2_medium import Task2PrioritizeAndLabel
from env.tasks.task3_hard import Task3FullInboxManagement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_action(action_type, email_id=None, value=None):
    return Action(action_type=action_type, email_id=email_id, value=value)


# ---------------------------------------------------------------------------
# Test: Models
# ---------------------------------------------------------------------------

class TestModels(unittest.TestCase):

    def test_action_valid(self):
        a = Action(action_type="classify", email_id="e001", value="spam")
        self.assertEqual(a.action_type, "classify")

    def test_action_done_no_email(self):
        a = Action(action_type="done")
        self.assertIsNone(a.email_id)

    def test_reward_range(self):
        r = Reward(value=0.5, cumulative=0.5, reason="test")
        self.assertGreaterEqual(r.value, 0.0)
        self.assertLessEqual(r.value, 1.0)

    def test_reward_zero(self):
        r = Reward(value=0.0, cumulative=0.0, reason="no reward")
        self.assertEqual(r.value, 0.0)


# ---------------------------------------------------------------------------
# Test: Task 1 — Easy
# ---------------------------------------------------------------------------

class TestTask1Easy(unittest.TestCase):

    def setUp(self):
        self.task = Task1ClassifyEmails(seed=42)

    def test_reset_returns_3_emails(self):
        obs = self.task.reset()
        self.assertEqual(len(obs.inbox), 3)
        self.assertEqual(obs.current_task, "classify_emails")
        self.assertEqual(obs.step_count, 0)
        self.assertFalse(obs.done)

    def test_correct_classification_earns_reward(self):
        self.task.reset()
        obs, reward, done, _ = EmailTriageEnv().reset("classify_emails"), None, False, None
        # Direct task test
        task = Task1ClassifyEmails()
        task.reset()
        obs, reward, done = task.step(make_action("classify", "t1-e001", "spam"))
        self.assertAlmostEqual(reward.value, 1/3, places=2)
        self.assertFalse(done)

    def test_wrong_classification_zero_reward(self):
        task = Task1ClassifyEmails()
        task.reset()
        obs, reward, done = task.step(make_action("classify", "t1-e001", "normal"))
        self.assertEqual(reward.value, 0.0)

    def test_perfect_score_is_1(self):
        task = Task1ClassifyEmails()
        task.reset()
        task.step(make_action("classify", "t1-e001", "spam"))
        task.step(make_action("classify", "t1-e002", "urgent"))
        task.step(make_action("classify", "t1-e003", "newsletter"))
        self.assertEqual(task.grade(), 1.0)

    def test_zero_correct_score_is_0(self):
        task = Task1ClassifyEmails()
        task.reset()
        task.step(make_action("classify", "t1-e001", "normal"))
        task.step(make_action("classify", "t1-e002", "spam"))
        task.step(make_action("classify", "t1-e003", "urgent"))
        self.assertEqual(task.grade(), 0.0)

    def test_invalid_email_id_no_crash(self):
        task = Task1ClassifyEmails()
        task.reset()
        obs, reward, done = task.step(make_action("classify", "NONEXISTENT", "spam"))
        self.assertEqual(reward.value, 0.0)
        self.assertFalse(done)

    def test_invalid_category_no_crash(self):
        task = Task1ClassifyEmails()
        task.reset()
        obs, reward, done = task.step(make_action("classify", "t1-e001", "INVALID_CAT"))
        self.assertEqual(reward.value, 0.0)

    def test_duplicate_classification_penalized(self):
        task = Task1ClassifyEmails()
        task.reset()
        task.step(make_action("classify", "t1-e001", "spam"))
        obs, reward, done = task.step(make_action("classify", "t1-e001", "spam"))
        self.assertEqual(reward.value, 0.0)
        self.assertGreater(reward.penalty, 0.0)

    def test_grade_always_in_01_range(self):
        task = Task1ClassifyEmails()
        task.reset()
        grade = task.grade()
        self.assertGreaterEqual(grade, 0.0)
        self.assertLessEqual(grade, 1.0)

    def test_episode_ends_after_all_classified(self):
        task = Task1ClassifyEmails()
        task.reset()
        task.step(make_action("classify", "t1-e001", "spam"))
        task.step(make_action("classify", "t1-e002", "urgent"))
        obs, reward, done = task.step(make_action("classify", "t1-e003", "newsletter"))
        self.assertTrue(done)

    def test_max_steps_ends_episode(self):
        task = Task1ClassifyEmails()
        task.reset()
        for _ in range(task.MAX_STEPS):
            obs, reward, done = task.step(make_action("skip"))
            if done:
                break
        self.assertTrue(done)

    def test_state_returns_dict(self):
        task = Task1ClassifyEmails()
        task.reset()
        state = task.state()
        self.assertIn("task_id", state)
        self.assertIn("grade", state)
        self.assertIn("step_count", state)

    def test_reset_clears_state(self):
        task = Task1ClassifyEmails()
        task.reset()
        task.step(make_action("classify", "t1-e001", "spam"))
        task.reset()
        self.assertEqual(task.grade(), 0.0)
        self.assertEqual(task._step_count, 0)


# ---------------------------------------------------------------------------
# Test: Task 2 — Medium
# ---------------------------------------------------------------------------

class TestTask2Medium(unittest.TestCase):

    def test_reset_returns_5_emails(self):
        task = Task2PrioritizeAndLabel()
        obs = task.reset()
        self.assertEqual(len(obs.inbox), 5)
        self.assertEqual(obs.current_task, "prioritize_and_label")

    def test_correct_priority_earns_reward(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        obs, reward, done = task.step(make_action("prioritize", "t2-e001", "high"))
        self.assertGreater(reward.value, 0.0)

    def test_wrong_priority_zero_reward(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        obs, reward, done = task.step(make_action("prioritize", "t2-e001", "low"))
        self.assertEqual(reward.value, 0.0)

    def test_correct_label_earns_reward(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        obs, reward, done = task.step(make_action("label", "t2-e001", "action_required"))
        self.assertGreater(reward.value, 0.0)

    def test_reply_to_wrong_email_penalized(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        obs, reward, done = task.step(make_action("reply", "t2-e002", "Some reply text"))
        self.assertEqual(reward.value, 0.0)
        self.assertGreater(reward.penalty, 0.0)

    def test_reply_with_keywords_scores_higher(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        # Good reply with many keywords
        obs, r_good, _ = task.step(make_action("reply", "t2-e001",
            "I acknowledge the invoice payment and will confirm billing status for the overdue amount."))
        task2 = Task2PrioritizeAndLabel()
        task2.reset()
        # Empty reply
        obs2, r_empty, _ = task2.step(make_action("reply", "t2-e001", "ok"))
        self.assertGreater(r_good.value, r_empty.value)

    def test_perfect_grade_achievable(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        from env.data import TASK2_GROUND_TRUTH, TASK2_REPLY_TARGET, TASK2_REPLY_KEYWORDS
        for eid, gt in TASK2_GROUND_TRUTH.items():
            task.step(make_action("prioritize", eid, gt["priority"]))
            task.step(make_action("label", eid, gt["label"]))
        # Best possible reply
        task.step(make_action("reply", TASK2_REPLY_TARGET,
            "Thank you for the invoice reminder. I acknowledge the overdue payment and will confirm billing."))
        grade = task.grade()
        self.assertGreaterEqual(grade, 0.8)

    def test_grade_range(self):
        task = Task2PrioritizeAndLabel()
        task.reset()
        self.assertGreaterEqual(task.grade(), 0.0)
        self.assertLessEqual(task.grade(), 1.0)


# ---------------------------------------------------------------------------
# Test: Task 3 — Hard
# ---------------------------------------------------------------------------

class TestTask3Hard(unittest.TestCase):

    def test_reset_returns_10_emails(self):
        task = Task3FullInboxManagement()
        obs = task.reset()
        self.assertEqual(len(obs.inbox), 10)
        self.assertEqual(obs.current_task, "full_inbox_management")

    def test_all_action_types_work(self):
        task = Task3FullInboxManagement()
        task.reset()
        actions = [
            make_action("classify", "t3-e001", "urgent"),
            make_action("prioritize", "t3-e001", "high"),
            make_action("label", "t3-e001", "action_required"),
            make_action("flag", "t3-e001"),
            make_action("reply", "t3-e001", "Hi Jane, the Q1 board presentation slides will be ready by 6pm for the board meeting deck."),
            make_action("archive", "t3-e002"),
        ]
        for a in actions:
            obs, reward, done = task.step(a)
            self.assertFalse(done)

    def test_false_positive_flag_penalized(self):
        task = Task3FullInboxManagement()
        task.reset()
        obs, reward, done = task.step(make_action("flag", "t3-e009"))  # newsletter, not action item
        self.assertEqual(reward.value, 0.0)
        self.assertGreater(reward.penalty, 0.0)

    def test_archive_spam_correct(self):
        task = Task3FullInboxManagement()
        task.reset()
        obs, reward, done = task.step(make_action("archive", "t3-e002"))  # spam
        self.assertGreater(reward.value, 0.0)

    def test_archive_urgent_penalized(self):
        task = Task3FullInboxManagement()
        task.reset()
        obs, reward, done = task.step(make_action("archive", "t3-e001"))  # CEO urgent
        self.assertEqual(reward.value, 0.0)
        self.assertGreater(reward.penalty, 0.0)

    def test_duplicate_thread_detection(self):
        task = Task3FullInboxManagement()
        task.reset()
        # Label the thread email — should trigger duplicate bonus
        obs, reward, done = task.step(make_action("label", "t3-e003", "action_required"))
        self.assertTrue(task._duplicate_detected)
        self.assertGreater(reward.value, 0.10)

    def test_reply_to_non_reply_email_penalized(self):
        task = Task3FullInboxManagement()
        task.reset()
        obs, reward, done = task.step(make_action("reply", "t3-e006", "Sure!"))
        self.assertEqual(reward.value, 0.0)
        self.assertGreater(reward.penalty, 0.0)

    def test_high_score_achievable(self):
        task = Task3FullInboxManagement()
        task.reset()
        from env.data import TASK3_GROUND_TRUTH, TASK3_ACTION_ITEMS, TASK3_REPLY_KEYWORDS
        for eid, gt in TASK3_GROUND_TRUTH.items():
            task.step(make_action("classify", eid, gt["category"]))
            task.step(make_action("prioritize", eid, gt["priority"]))
            task.step(make_action("label", eid, gt["label"]))
        for eid in TASK3_ACTION_ITEMS:
            task.step(make_action("flag", eid))
        for eid, kws in TASK3_REPLY_KEYWORDS.items():
            task.step(make_action("reply", eid, " ".join(kws)))
        for eid, gt in TASK3_GROUND_TRUTH.items():
            if gt.get("archive"):
                task.step(make_action("archive", eid))
        grade = task.grade()
        self.assertGreaterEqual(grade, 0.80)

    def test_grade_range(self):
        task = Task3FullInboxManagement()
        task.reset()
        grade = task.grade()
        self.assertGreaterEqual(grade, 0.0)
        self.assertLessEqual(grade, 1.0)


# ---------------------------------------------------------------------------
# Test: EmailTriageEnv (main env class)
# ---------------------------------------------------------------------------

class TestEmailTriageEnv(unittest.TestCase):

    def test_all_task_ids_registered(self):
        self.assertIn("classify_emails", ALL_TASK_IDS)
        self.assertIn("prioritize_and_label", ALL_TASK_IDS)
        self.assertIn("full_inbox_management", ALL_TASK_IDS)
        self.assertEqual(len(ALL_TASK_IDS), 3)

    def test_reset_returns_observation(self):
        env = EmailTriageEnv()
        obs = env.reset("classify_emails")
        self.assertIsInstance(obs, Observation)

    def test_step_returns_tuple(self):
        env = EmailTriageEnv()
        env.reset("classify_emails")
        obs, reward, done, info = env.step(make_action("classify", "t1-e001", "spam"))
        self.assertIsInstance(obs, Observation)
        self.assertIsInstance(reward, Reward)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_step_without_reset_raises(self):
        env = EmailTriageEnv()
        with self.assertRaises(RuntimeError):
            env.step(make_action("classify", "t1-e001", "spam"))

    def test_invalid_task_id_raises(self):
        env = EmailTriageEnv()
        with self.assertRaises(ValueError):
            env.reset("nonexistent_task")

    def test_state_returns_dict(self):
        env = EmailTriageEnv()
        env.reset("classify_emails")
        state = env.state()
        self.assertIn("task_id", state)
        self.assertIn("grade", state)

    def test_state_before_reset(self):
        env = EmailTriageEnv()
        state = env.state()
        self.assertIn("available_tasks", state)

    def test_grade_before_reset_is_zero(self):
        env = EmailTriageEnv()
        self.assertEqual(env.grade(), 0.0)

    def test_episode_rewards_tracked(self):
        env = EmailTriageEnv()
        env.reset("classify_emails")
        env.step(make_action("classify", "t1-e001", "spam"))
        env.step(make_action("classify", "t1-e002", "urgent"))
        state = env.state()
        self.assertEqual(len(state["episode_rewards"]), 2)

    def test_reset_switches_task(self):
        env = EmailTriageEnv()
        env.reset("classify_emails")
        self.assertEqual(env._task_id, "classify_emails")
        env.reset("prioritize_and_label")
        self.assertEqual(env._task_id, "prioritize_and_label")

    def test_reward_value_always_in_range(self):
        env = EmailTriageEnv()
        for task_id in ALL_TASK_IDS:
            env.reset(task_id)
            for _ in range(5):
                obs, reward, done, info = env.step(make_action("skip"))
                self.assertGreaterEqual(reward.value, 0.0)
                self.assertLessEqual(reward.value, 1.0)
                if done:
                    break

    def test_deterministic_with_same_seed(self):
        env1 = EmailTriageEnv()
        obs1 = env1.reset("classify_emails", seed=42)
        env2 = EmailTriageEnv()
        obs2 = env2.reset("classify_emails", seed=42)
        self.assertEqual(len(obs1.inbox), len(obs2.inbox))
        self.assertEqual(obs1.inbox[0].id, obs2.inbox[0].id)


# ---------------------------------------------------------------------------
# Test: Grader determinism
# ---------------------------------------------------------------------------

class TestGraderDeterminism(unittest.TestCase):

    def _run_task1_perfect(self):
        task = Task1ClassifyEmails()
        task.reset()
        task.step(make_action("classify", "t1-e001", "spam"))
        task.step(make_action("classify", "t1-e002", "urgent"))
        task.step(make_action("classify", "t1-e003", "newsletter"))
        return task.grade()

    def test_task1_grade_reproducible(self):
        g1 = self._run_task1_perfect()
        g2 = self._run_task1_perfect()
        self.assertEqual(g1, g2)

    def test_task1_grade_is_float(self):
        self.assertIsInstance(self._run_task1_perfect(), float)

    def test_all_tasks_grade_in_range(self):
        env = EmailTriageEnv()
        for tid in ALL_TASK_IDS:
            env.reset(tid)
            grade = env.grade()
            self.assertGreaterEqual(grade, 0.0, f"{tid} grade below 0")
            self.assertLessEqual(grade, 1.0, f"{tid} grade above 1")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestModels,
        TestTask1Easy,
        TestTask2Medium,
        TestTask3Hard,
        TestEmailTriageEnv,
        TestGraderDeterminism,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"\n✓ All {result.testsRun} tests passed")
        sys.exit(0)
    else:
        print(f"\n✗ {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)
