import unittest

from text_feedback_dpo.parsing import TrajectoryParseError, parse_trajectory


VALID = """<plan>Plan.</plan>
<think branch="A">Reason.</think>
<reflect>Verification: checked.</reflect>
<final>4</final>"""


class TrajectoryParsingTest(unittest.TestCase):
    def test_accepts_pdf_think_tag_with_branch_attribute(self):
        trajectory = parse_trajectory(VALID)
        self.assertEqual(trajectory.thinks, ["Reason."])

    def test_rejects_missing_pdf_think_tag(self):
        invalid = VALID.replace('<think branch="A">Reason.</think>\n', "")
        with self.assertRaisesRegex(TrajectoryParseError, "missing <think"):
            parse_trajectory(invalid)


if __name__ == "__main__":
    unittest.main()
