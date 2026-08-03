# -*- coding: utf-8 -*-
"""女仆人格数据集与桥接模块测试。"""

import unittest

from ai_brain_entity import AIBrainEntity
from persona_maid import (FEEDBACK_KINDS, MOODS, VERBS, load_persona,
                          maid_express, maid_feedback, maid_greeting,
                          utterance)


class TestMaidDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.persona = load_persona()

    def test_mood_verb_full_coverage(self):
        for mood in MOODS:
            for verb in VERBS:
                lines = self.persona["utterances"][mood][verb]
                self.assertGreaterEqual(len(lines), 1, f"{mood}/{verb}")

    def test_no_empty_or_duplicate_utterances(self):
        seen = set()
        for mood in MOODS:
            for verb in VERBS:
                for s in self.persona["utterances"][mood][verb]:
                    self.assertTrue(s.strip())
                    self.assertNotIn(s, seen, f"重复话术: {s}")
                    seen.add(s)

    def test_feedback_and_greeting_non_empty(self):
        for kind in FEEDBACK_KINDS:
            self.assertTrue(self.persona["feedback"][kind])
        self.assertTrue(self.persona["greetings"])
        self.assertTrue(self.persona["scenes"])

    def test_invalid_structure_rejected(self):
        with self.assertRaises(ValueError):
            from persona_maid import _validate
            _validate({"persona": {}, "utterances": {}, "feedback": {},
                       "greetings": [], "scenes": []})


class TestMaidSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.persona = load_persona()

    def test_utterance_deterministic(self):
        a = utterance(self.persona, "curiosity", "respond", seed=7)
        b = utterance(self.persona, "curiosity", "respond", seed=7)
        self.assertEqual(a, b)
        self.assertIn(a, self.persona["utterances"]["curiosity"]["respond"])

    def test_utterance_fallback(self):
        self.assertEqual(utterance(self.persona, "???", "???", seed=0),
                         self.persona["utterances"]["calm"]["observe"][0])

    def test_feedback_kinds(self):
        self.assertIn(maid_feedback(self.persona, "praised", 1),
                      self.persona["feedback"]["praised"])
        self.assertIn(maid_feedback(self.persona, "scolded", 1),
                      self.persona["feedback"]["scolded"])
        with self.assertRaises(ValueError):
            maid_feedback(self.persona, "angry")

    def test_greeting_in_list(self):
        self.assertIn(maid_greeting(self.persona, 3),
                      self.persona["greetings"])


class TestMaidBrainIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.persona = load_persona()

    def test_maid_express_structure(self):
        brain = AIBrainEntity("小铃", seed=42)
        out = maid_express(brain, "主人回来啦", self.persona)
        self.assertIn("action", out)
        self.assertIn("utterance", out)
        act = out["action"]
        self.assertIn(act["verb"], VERBS)
        self.assertIn(act["mood"], MOODS)

    def test_maid_style_marker(self):
        brain = AIBrainEntity("小铃", seed=42)
        out = maid_express(brain, "你好", self.persona)
        self.assertIn("主人", out["utterance"])

    def test_reward_shifts_mood_and_feedback(self):
        brain = AIBrainEntity("小铃", seed=42)
        brain.reward(0.9)
        self.assertGreaterEqual(brain.emotion["pleasure"], 0.0)
        line = maid_feedback(self.persona, "praised", seed=brain.tick)
        self.assertTrue(line.strip())

    def test_express_does_not_crash_after_many_ticks(self):
        brain = AIBrainEntity("小铃", seed=1)
        for stim in ["茶", "鸟", "夸奖", "责骂", "地板", ""]:
            out = maid_express(brain, stim, self.persona)
            self.assertTrue(out["utterance"].strip())


if __name__ == "__main__":
    unittest.main()
