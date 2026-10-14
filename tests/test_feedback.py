import unittest

from src.agents import CoachEvaluation, STARBreakdown, verify_evidence_quotes


def make_evaluation(evidence_quotes):
    return CoachEvaluation(
        score=7.0,
        star_assessment=STARBreakdown(
            situation="Clear",
            task="Clear",
            action="Clear",
            result="Clear",
        ),
        candidate_rewrite="Describe your contribution and the result.",
        evidence_quotes=evidence_quotes,
        probe_target="What was the hardest part?",
    )


class FeedbackTests(unittest.TestCase):
    def test_only_exact_answer_quotes_are_kept(self):
        evaluation = make_evaluation([
            "I built the API",
            "I reduced latency by caching",
            "I led a team of ten",
        ])

        verified = verify_evidence_quotes(
            evaluation,
            "I built the API. I reduced latency by caching.",
        )

        self.assertEqual(
            verified.evidence_quotes,
            ["I built the API", "I reduced latency by caching"],
        )


if __name__ == "__main__":
    unittest.main()