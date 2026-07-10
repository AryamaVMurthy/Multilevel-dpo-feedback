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

    def test_rejects_legacy_thinking_tag_instead_of_silently_normalizing(self):
        invalid = VALID.replace('<think branch="A">Reason.</think>', "<thinking>Reason.</thinking>")
        with self.assertRaisesRegex(TrajectoryParseError, "legacy <thinking>"):
            parse_trajectory(invalid)

    def test_rejects_final_before_think(self):
        invalid = VALID.replace(
            '<think branch="A">Reason.</think>\n<reflect>Verification: checked.</reflect>\n<final>4</final>',
            '<final>4</final>\n<think branch="A">Reason.</think>\n<reflect>Verification: checked.</reflect>',
        )
        with self.assertRaisesRegex(TrajectoryParseError, "blocks are out of order"):
            parse_trajectory(invalid)


if __name__ == "__main__":
    unittest.main()
