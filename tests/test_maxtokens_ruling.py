"""D1 ruling (2026-11-18): maxTokens is NOT sent by default and IS honored
when a node explicitly sets it — default-off, opt-in per use, identical
across the Python executor, the JS SDK, the editor, and MCP.

The runner behavior has long been pinned in test_chat_nodes.py; this file
pins the whole ruling in one place — wire body both directions, the
estimator, and the settings surface — so any side drifting either way
fails loudly.
"""

import unittest

from tests._util import MockedTest
from tests.harness import chat_response

from nanoodle import Workflow
from nanoodle.estimate import estimate_graph_cost


def _llm_graph(max_tokens_marker):
    fields = {"model": "gpt-5o", "prompt": "hi"}
    if max_tokens_marker is not ...:
        fields["maxTokens"] = max_tokens_marker
    return {"nodes": [{"id": "n", "type": "llm", "fields": fields}], "links": []}


class MaxTokensWireTest(MockedTest):
    def test_absent_by_default_and_sent_exactly_when_set(self):
        self.mock.script("POST", "/api/v1/chat/completions", chat_response("ok"))

        self.wf_dict(_llm_graph(...)).run({})
        self.assertNotIn(
            "max_tokens", self.mock.requests_to("/api/v1/chat/completions")[0].json)

        self.wf_dict(_llm_graph("1536")).run({})
        self.assertEqual(
            self.mock.requests_to("/api/v1/chat/completions")[1].json["max_tokens"], 1536)


class MaxTokensEstimateAndSettingsTest(unittest.TestCase):
    def test_estimator_prices_an_explicit_cap_at_the_cap(self):
        catalog = {"chat": [{"id": "gpt-5o", "pricing": {"prompt": 2, "completion": 10}}]}
        capped = estimate_graph_cost(_llm_graph("1536"), catalog)
        uncapped = estimate_graph_cost(_llm_graph(...), catalog)
        self.assertGreater(capped.usd, uncapped.usd)
        self.assertAlmostEqual(capped.usd - uncapped.usd, (1536 - 500) * 10 / 1e6, places=9)

    def test_settings_expose_the_opt_in(self):
        wf = Workflow.from_dict(_llm_graph("1536"), api_key="sk")
        self.assertTrue(any(s.field == "maxTokens" for s in wf.settings))


if __name__ == "__main__":
    unittest.main()
