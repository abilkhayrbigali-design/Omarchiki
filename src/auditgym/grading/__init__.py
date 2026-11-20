"""AuditGym grader (WP4): findings parsing, one-to-one matcher, metrics, terminal reward.

This package is the published normalization, tolerance and tie-breaking code required by S06.
It depends only on `auditgym.contracts` and `auditgym.util`, keeps no state and does no I/O:
the same input always gives the same GradeResult.

1. Structure (parse.py, S01-S03)
   The `submit` argument must match schemas/submission.schema.json exactly and carry the current
   `episode_id`. At most 30 findings, 1-10 `affected_ids`, 0-10 `evidence_ids`, ids of 1-64
   characters, `claimed_amount_minor` a JSON integer >= 0 (not bool, not 5.0), `currency`
   exactly three letters A-Z, `explanation` at most 1000 characters, no missing or extra keys.
   Any deviation: valid=False, reward -1, the reason code goes to the grader log only.

2. Normalization
   - `type`: exact string from ISSUE_TYPES. No case folding, no aliases.
   - `affected_ids`, `evidence_ids`: sets. Order and repeats inside one list do not matter.
     Ids are opaque and compared by exact string equality: no trimming, no case folding, no
     Unicode normalization ("d-i3" and "D-I3 " are not "D-I3").
   - `claimed_amount_minor`: integer minor units of `currency`, a magnitude (>= 0), like the
     reference `expected_amount_minor` (docs/DOMAIN.md, section 6).
   - `currency`: compared exactly; a lowercase code is a structural error, not normalized.
   - `explanation`: kept, never read. Text cannot change any count or reward (S09).

3. Compatibility of a finding F and a reference issue I (match.py, S04)
   F.type == I.type
   and set(F.affected_ids) == set(I.affected_ids)        (canonical ids, no subsets/supersets)
   and F.currency == I.currency
   and |F.claimed_amount_minor - I.expected_amount_minor| <= I.amount_tolerance_minor
                                                           (inclusive; tolerance from truth:
                                                            0 for all types except fx_conversion)
   and some S in I.evidence_sets has S <= set(F.evidence_ids)
                                                           (a whole evidence set; extra ids are
                                                            fine, the 10-id cap bounds them)

4. Matching (match.py, S04-S06)
   Bipartite graph: findings x issues, an edge per compatible pair. TP is the size of a maximum
   matching, FP = findings - TP (unmatched and repeated findings), FN = issues - TP. The size of
   a maximum matching does not depend on the order of findings or on how ids are spelled.
   Choice when several findings fit (published tie-breaking): findings are offered to Kuhn's
   augmenting-path algorithm in this order, and a finding once matched stays matched, so the
   earliest finding in this order that can be matched is reported as the TP:
     a. smaller |claimed - expected| to its closest compatible issue;
     b. fewer distinct evidence ids;
     c. smaller claimed amount;
     d. normalized content (type, affected ids, amount, currency, evidence ids) in lexicographic
        order;
     e. earlier position in the submission - reached only by findings with identical content.
   Issues are tried in truth order (Truth sorts them by (type, affected_ids)). Keys d and e
   only choose which of two equally scored findings is labelled TP in `matches`; TP, FP, FN and
   both rewards never depend on them. The `matches` list is sorted by finding index.

5. Metrics (reward.py, S07), exact fractions
   P = TP/(TP+FP), R = TP/(TP+FN), 0 on a zero denominator. F1 = 2TP/(2TP+FP+FN); truth and
   findings both empty -> F1 = 1; truth empty, findings present -> F1 = 0.

6. Shared reward (reward.py, S08)
   reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1), computed with fractions.Fraction and
   converted to float at the end. Invalid structure -> -1. `inspection_steps` counts every step
   before submit, failed ones included; a truncated episode is graded as an empty submission.

7. Experimental reward (GradeResult.experimental_reward), reported separately
   Same formula, but a pair is compatible only if the satisfied evidence set is also wholly
   inside `viewed_ids`, the ids the agent actually requested. It defeats guessed evidence ids.
   It is None without `viewed_ids` and never enters the shared reward: the organizers' oracle
   submits without exploring and must still receive the maximum shared reward.
"""

from auditgym.grading.grade import INVALID_REWARD, grade_submission
from auditgym.grading.match import (
    MatchResult,
    compatible,
    evidence_satisfied,
    match_findings,
    normalize,
)
from auditgym.grading.parse import (
    SubmissionError,
    SubmissionErrorCode,
    empty_submission,
    parse_submission,
)
from auditgym.grading.reward import STEP_COST_FRACTION, metrics, reward_exact, shared_reward

__all__ = [
    "INVALID_REWARD",
    "STEP_COST_FRACTION",
    "MatchResult",
    "SubmissionError",
    "SubmissionErrorCode",
    "compatible",
    "empty_submission",
    "evidence_satisfied",
    "grade_submission",
    "match_findings",
    "metrics",
    "normalize",
    "parse_submission",
    "reward_exact",
    "shared_reward",
]
