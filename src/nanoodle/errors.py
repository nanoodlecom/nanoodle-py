"""Exception types for the nanoodle workflow executor."""


class NanoodleError(Exception):
    """Base class for every error raised by this library."""


class UnsupportedNodeError(NanoodleError):
    """Raised at run start when the graph contains a node this library cannot execute.

    Raised BEFORE any network call is made (fail fast, before spending).
    """

    def __init__(self, node_id, node_type, message):
        super().__init__(message)
        self.node_id = node_id
        self.node_type = node_type


class RunError(NanoodleError):
    """Raised by Workflow.run() when a sink (output) node failed.

    Carries the partial results in ``.result`` (a RunResult): every node that
    did complete keeps its output, cost and timing there.
    """

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result
