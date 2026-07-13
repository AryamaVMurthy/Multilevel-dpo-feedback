import unittest

from text_feedback_dpo.training import build_method_config


class MethodConfigTest(unittest.TestCase):
    def test_all_methods_are_full_finetuning_and_share_length_limit(self):
        for method in ("sft", "dpo", "grpo", "dapo"):
            config = build_method_config(method, max_length=4096, max_steps=-1)
            self.assertTrue(config["full_finetuning"])
            self.assertEqual(config["max_length"], 4096)
            self.assertIsNone(config["adapter"])

    def test_unknown_method_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "method"):
            build_method_config("unknown", max_length=4096)


if __name__ == "__main__":
    unittest.main()
