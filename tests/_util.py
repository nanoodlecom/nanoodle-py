"""Shared test scaffolding: a MockNanoGPT-backed TestCase base + fast poll knobs."""

import unittest

from tests import fixture
from tests.harness import MockNanoGPT

from nanoodle import Workflow

# Short injectable poll intervals / timeouts so polling tests run in milliseconds.
FAST = {"poll_intervals": {"video": 0.01, "audio": 0.01},
        "timeouts": {"video": 1.0, "audio": 1.0}}


def tripwire_http(*a, **kw):
    """An http transport that fails the test on ANY network call."""
    raise AssertionError("unexpected network call: %r" % (a[:2],))


class MockedTest(unittest.TestCase):
    """TestCase with a fresh mock NanoGPT server per test."""

    def setUp(self):
        self.mock = MockNanoGPT().start()
        self.addCleanup(self.mock.stop)

    def wf(self, name, **opts):
        opts.setdefault("api_key", "test-key")
        opts.setdefault("base_url", self.mock.base_url)
        return Workflow.load(fixture(name), **opts)

    def wf_dict(self, data, **opts):
        opts.setdefault("api_key", "test-key")
        opts.setdefault("base_url", self.mock.base_url)
        return Workflow.from_dict(data, **opts)
