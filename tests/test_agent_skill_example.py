"""Guards examples/agent-skill/poster-generator against engine renames: its SKILL.md
documents --input "Idea=..." and output key Poster (saved as Poster.<ext> by MIME).
Offline — no server, no run()."""

import os
import unittest

from tests import FIXTURES

from nanoodle import Workflow

WORKFLOW = os.path.join(
    os.path.dirname(os.path.dirname(FIXTURES)),  # repo root (tests/fixtures -> repo)
    "examples", "agent-skill", "poster-generator", "workflows", "poster.noodle-graph.json")


class AgentSkillExampleTest(unittest.TestCase):
    def test_poster_generator_derives_idea_input_and_poster_output(self):
        wf = Workflow.load(WORKFLOW, api_key="unused")
        self.assertEqual(
            [s.key for s in wf.inputs if not s.optional], ["Idea"],
            "required input keys must match SKILL.md's --input flag")
        self.assertEqual(
            [o.key for o in wf.outputs], ["Poster"],
            "output keys must match SKILL.md's documented Poster.<ext>")
        # Optional LLM system field is a useful advanced --input; keep it exposed.
        opt = [s for s in wf.inputs if s.key == "System prompt"]
        self.assertEqual(len(opt), 1)
        self.assertTrue(opt[0].optional)
        self.assertEqual(wf.warnings, [])


if __name__ == "__main__":
    unittest.main()
