"""Server process with a stand-in grader, for tests that run before WP4's grader exists.

Usage: python tests/leak/fake_grader_server.py [server options]
The grader's output depends on the truth and carries truth data (issue ids, error text), so the
tests can check that none of it crosses the process boundary.
"""

from __future__ import annotations

import sys

from auditgym.contracts.grading_api import GradeResult
from auditgym.server.app import main

SECRET_REASON = "SECRET-GRADER-REASON"


def grader(payload, truth, *, episode_id, inspection_steps, viewed_ids=None, known_ids=None):
    n = len(truth.issues)
    return GradeResult(
        valid=True,
        error=SECRET_REASON,
        reward=round(0.1 * n - 0.002 * inspection_steps, 6),
        experimental_reward=None,
        tp=n,
        fp=0,
        fn=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        inspection_steps=inspection_steps,
        matches=tuple((i, issue.issue_id) for i, issue in enumerate(truth.issues)),
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], grader=grader))
