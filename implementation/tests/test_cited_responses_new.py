import unittest

from text_feedback_dpo.prompts import (
    build_cited_response_prompt,
    build_search_query_prompt,
    build_short_answer_prompt,
    build_student_prompt,
)
from text_feedback_dpo.responses import (
    CitedResponse,
    CitedResponseFormatError,
    parse_cited_response,
    render_cited_response,
)
from text_feedback_dpo.scoring import score_cited_response


def source(source_id: str, title: str, snippet: str, url: str = "https://example.test/source") -> dict:
    return {"source_id": source_id, "title": title, "snippet": snippet, "url": url}


SOURCES = [
    source("S001", "Ada biography", "Ada Lovelace wrote the first algorithm."),
    source("S003", "Computing history", "Charles Babbage designed the analytical engine."),
]


class CitedResponseParserTest(unittest.TestCase):
    def test_parses_strict_three_line_contract_and_keeps_first_citation_order(self):
        response = parse_cited_response(
            "Answer: Ada Lovelace\n"
            "Reasoning: The source identifies Ada Lovelace as the author [S001].\n"
            "Sources: S001",
            SOURCES,
        )
        self.assertEqual(response.answer, "Ada Lovelace")
        self.assertEqual(response.reasoning, "The source identifies Ada Lovelace as the author [S001].")
        self.assertEqual(response.source_ids, ("S001",))

    def test_rejects_wrong_labels_order_blank_lines_markup_and_urls(self):
        bad_outputs = [
            "Reasoning: claim [S001]\nAnswer: Ada\nSources: S001",
            "Answer: Ada\n\nReasoning: claim [S001]\nSources: S001",
            "Answer: Ada\nReason: claim [S001]\nSources: S001",
            "Answer: Ada\nReasoning: claim [S001] https://evil.example\nSources: S001",
            "Answer: Ada\nReasoning: <xml>claim [S001]</xml>\nSources: S001",
            "Answer: Ada\nReasoning: ```claim [S001]```\nSources: S001",
        ]
        for output in bad_outputs:
            with self.subTest(output=output), self.assertRaises(CitedResponseFormatError):
                parse_cited_response(output, SOURCES)

    def test_rejects_residual_brackets_after_valid_citations_are_removed(self):
        outputs = [
            "Answer: Ada [unexpected]\nReasoning: claim [S001]\nSources: S001",
            "Answer: Ada\nReasoning: claim [S001] and [unexpected]\nSources: S001",
            "Answer: Ada\nReasoning: claim [S001]\nSources: S001 [unexpected]",
        ]
        for output in outputs:
            with self.subTest(output=output), self.assertRaisesRegex(CitedResponseFormatError, "bracket|markup"):
                parse_cited_response(output, SOURCES)

    def test_rejects_scheme_mailto_www_and_bare_domain_urls(self):
        urls = ("foo://bar", "mailto:person@example.com", "www.example.com", "example.com", "docs.example.co.uk/path")
        for url in urls:
            output = f"Answer: Ada\nReasoning: claim [S001] see {url}\nSources: S001"
            with self.subTest(url=url), self.assertRaisesRegex(CitedResponseFormatError, "URL"):
                parse_cited_response(output, SOURCES)

    def test_rejects_length_limits_and_missing_citations(self):
        too_long_answer = "Answer: " + "word " * 17 + "\nReasoning: claim [S001]\nSources: S001"
        too_long_reasoning = "Answer: Ada\nReasoning: " + "word " * 97 + "[S001]\nSources: S001"
        for output in (too_long_answer, too_long_reasoning):
            with self.subTest(), self.assertRaises(CitedResponseFormatError):
                parse_cited_response(output, SOURCES)
        with self.assertRaisesRegex(CitedResponseFormatError, "citation"):
            parse_cited_response("Answer: Ada\nReasoning: unsupported claim\nSources: S001", SOURCES)

    def test_rejects_unknown_duplicate_and_mismatched_citations(self):
        outputs = [
            "Answer: Ada\nReasoning: claim [S001].\nSources: S001, S001",
            "Answer: Ada\nReasoning: claim [S001] and [S003].\nSources: S001",
            "Answer: Ada\nReasoning: claim [S001].\nSources: S003",
            "Answer: Ada\nReasoning: claim [S999].\nSources: S999",
            "Answer: Ada\nReasoning: claim [S001] and [S001].\nSources: S001",
        ]
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(CitedResponseFormatError):
                parse_cited_response(output, SOURCES)


class CitedResponseRenderingTest(unittest.TestCase):
    def test_renderer_uses_canonical_metadata_and_never_model_source_text(self):
        parsed = parse_cited_response(
            "Answer: Ada Lovelace\n"
            "Reasoning: The biography names Ada Lovelace [S001].\n"
            "Sources: S001",
            SOURCES,
        )
        rendered = render_cited_response(parsed, SOURCES)
        self.assertEqual(
            rendered,
            "Answer: Ada Lovelace\n"
            "Reasoning: The biography names Ada Lovelace [S001].\n"
            "Sources:\n"
            "[S001] Ada biography — https://example.test/source",
        )

    def test_renderer_fails_when_cited_metadata_is_unavailable(self):
        parsed = parse_cited_response(
            "Answer: Ada\nReasoning: Source says Ada [S001].\nSources: S001",
            SOURCES,
        )
        missing_url = [source("S001", "Ada biography", "Ada evidence", url="")]
        with self.assertRaises(ValueError):
            render_cited_response(parsed, missing_url)

    def test_renderer_revalidates_manually_constructed_cited_response(self):
        invalid_responses = (
            CitedResponse("", "Source says Ada [S001].", ("S001",)),
            CitedResponse("Ada", "Source says Ada.", ("S001",)),
            CitedResponse("Ada", "Source says Ada [S001].", ("S003",)),
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(CitedResponseFormatError):
                render_cited_response(response, SOURCES)


class CitedResponseScoringTest(unittest.TestCase):
    def test_scores_answer_and_citation_support_metrics(self):
        result = score_cited_response(
            "Answer: Ada Lovelace\n"
            "Reasoning: The source names Ada Lovelace [S001], while history provides context [S003].\n"
            "Sources: S001, S003",
            "Ada Lovelace",
            SOURCES,
        )
        self.assertTrue(result["parse_valid"])
        self.assertIsNone(result["error_code"])
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertTrue(result["answer_correct"])
        self.assertEqual(result["citation_count"], 2)
        self.assertEqual(result["citation_precision"], 0.5)
        self.assertEqual(result["citation_recall"], 1.0)
        self.assertEqual(result["cited_answer_support"], 1.0)
        self.assertEqual(result["unsupported_source_rate"], 0.5)
        self.assertTrue(result["correct"])
        self.assertEqual(result["answer_words"], 2)
        self.assertEqual(result["reasoning_words"], 11)

    def test_malformed_explanation_does_not_corrupt_answer_em(self):
        result = score_cited_response(
            "Answer: Ada Lovelace\nReasoning: no citation here\nSources: S001",
            "Ada Lovelace",
            SOURCES,
        )
        self.assertFalse(result["parse_valid"])
        self.assertEqual(result["error_code"], "missing_citation")
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertTrue(result["answer_correct"])
        self.assertFalse(result["correct"])
        self.assertEqual(result["citation_count"], 0)

    def test_correct_requires_parse_valid_exact_answer_and_cited_support(self):
        unsupported = score_cited_response(
            "Answer: Ada Lovelace\nReasoning: History supports the context [S003].\nSources: S003",
            "Ada Lovelace",
            SOURCES,
        )
        self.assertTrue(unsupported["parse_valid"])
        self.assertTrue(unsupported["answer_correct"])
        self.assertEqual(unsupported["exact_match"], 1.0)
        self.assertFalse(unsupported["cited_answer_support"])
        self.assertFalse(unsupported["correct"])

    def test_malformed_label_order_does_not_extract_an_answer_for_em(self):
        result = score_cited_response(
            "Reasoning: claim [S001]\nAnswer: Ada Lovelace\nSources: S001",
            "Ada Lovelace",
            SOURCES,
        )
        self.assertFalse(result["parse_valid"])
        self.assertEqual(result["exact_match"], 0.0)

    def test_invalid_retrieved_data_raises_instead_of_becoming_model_error(self):
        with self.assertRaises(ValueError):
            score_cited_response(
                "not a response",
                "Ada",
                [source("S001", "", "Ada evidence")],
            )


class CitedPromptTest(unittest.TestCase):
    def test_archival_short_prompt_has_explicit_builder_and_compatibility_wrapper(self):
        example = {"question": "Who?", "packed_evidence": "Ada evidence", "gold_answer": "Ada"}
        self.assertEqual(build_student_prompt(example, []), build_short_answer_prompt(example, []))

    def test_search_query_prompt_has_no_evidence_or_gold_and_ends_with_query_label(self):
        prompt = build_search_query_prompt(
            {"question": "Who wrote the first algorithm?", "gold_answer": "Ada Lovelace", "packed_evidence": "Ada evidence"},
            ["Identify the relevant entity."],
        )
        self.assertTrue(prompt.endswith("Search query:"))
        self.assertIn("one-line", prompt)
        self.assertNotIn("Ada Lovelace", prompt)
        self.assertNotIn("Ada evidence", prompt)

    def test_cited_response_prompt_lists_sources_and_forbids_gold_urls_markup(self):
        prompt = build_cited_response_prompt(
            {"question": "Who wrote the first algorithm?", "gold_answer": "Ada Lovelace"},
            SOURCES,
            ["Use the strongest source."],
        )
        self.assertIn("[S001] Ada biography", prompt)
        self.assertIn("Ada Lovelace wrote the first algorithm.", prompt)
        self.assertIn("Never reproduce URLs", prompt)
        self.assertIn("exactly three nonblank lines", prompt)
        self.assertIn("at most 16 normalized words", prompt)
        self.assertIn("at most 96 words", prompt)
        self.assertIn("Answer:", prompt)
        self.assertIn("Reasoning:", prompt)
        self.assertIn("Sources:", prompt)
        self.assertNotIn("Gold answer", prompt)
        self.assertNotIn("gold_answer", prompt)
        self.assertNotIn("<response>", prompt)
        self.assertNotIn("{\"answer\"", prompt)

    def test_cited_response_prompt_ends_with_response_and_accepts_full_label_completion(self):
        prompt = build_cited_response_prompt(
            {"question": "Who wrote the first algorithm?"},
            SOURCES,
            [],
        )
        self.assertTrue(prompt.endswith("Response:"))
        completion = "Answer: Ada Lovelace\nReasoning: The source names Ada Lovelace [S001].\nSources: S001"
        parsed = parse_cited_response(completion, SOURCES)
        self.assertEqual(parsed.answer, "Ada Lovelace")


if __name__ == "__main__":
    unittest.main()
