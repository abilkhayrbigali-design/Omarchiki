"""AuditGym grader (WP4): findings parsing, one-to-one matcher, metrics, terminal reward.

This package is the published normalization, tolerance and tie-breaking code required by S06.
It depends only on `auditgym.contracts` and `auditgym.util`, keeps no state and does no I/O:
the same input always gives the same GradeResult.

1. Structure (parse.py, S01-S03)
   The `submit` argument must match schemas/submission.schema.json exactly and carry the current
   `episode_id`: no missing or extra keys, every value of its JSON type (`claimed_amount_minor` a
   JSON integer - not bool, not a string, not 5.5; 5.0 is read as 5), at most 30 findings, at
   most 10 `affected_ids` and 10 `evidence_ids` per finding (raw list length, before removing
   repeats), `explanation` at most 1000 Unicode code points. Any deviation: valid=False,
   reward -1, the reason code goes to the grader log only. A structurally valid finding with a
   wrong value is not an error of the submission; it simply matches no issue (section 3), as
   the jury's matcher conventions require (docs/REQUIREMENTS.md, section 8).

2. Normalization
   - `type`: exact string; only the four ISSUE_TYPES can match. No case folding, no aliases; any
     other string is a false positive.
   - `affected_ids`, `evidence_ids`: sets. Order and repeats inside one list do not matter.
     Ids are opaque and compared by exact string equality: no trimming, no case folding, no
     Unicode normalization ("d-i3" and "D-I3 " are not "D-I3").
   - `claimed_amount_minor`: integer minor units of `currency`, a magnitude like the reference
     `expected_amount_minor` (docs/DOMAIN.md, section 6); a negative claim matches nothing.
   - `currency`: compared exactly with the reference code; "kzt" or "KZT " is a false positive.
   - `explanation`: kept, never read. Text cannot change any count or reward (S09).

3. Compatibility of a finding F and a reference issue I (match.py, S04)
   F.type == I.type
   and set(F.affected_ids) == set(I.affected_ids)        (canonical ids, no subsets/supersets)
   and F.currency == I.currency
   and |F.claimed_amount_minor - I.expected_amount_minor| <= I.amount_tolerance_minor
                                                           (inclusive; the tolerance is fixed per
                                                            type and published in
                                                            policy.tolerances: 0 for all types
                                                            except fx_conversion; a truth file
                                                            without it means 0)
   and some S in I.evidence_sets has S <= set(F.evidence_ids)
                                                           (a whole evidence set; extra ids are
                                                            fine)
   and every id in F.evidence_ids is an id of the open data
                                                           (PublicEpisode.open_ids: record ids
                                                            and the ids they carry; one unknown
                                                            id makes the finding unmatched, M04)

4. Matching (match.py, S04-S06)
   Bipartite graph: findings x issues, an edge per compatible pair, repeated findings as
   separate vertices. TP is the size of a maximum matching (Kuhn's augmenting-path algorithm),
   FP = findings - TP (unmatched and repeated findings), FN = issues - TP. The size of a maximum
   matching does not depend on the order of findings or on how ids are spelled.
   Choice among maximum matchings (the jury's published rule): sort the reference issues by
   issue_id (Unicode code points) and take the lexicographically smallest sorted sequence of
   (issue_id, finding index) pairs, indices counted from 0 in the submitted list. Computed by
   walking the issues in that order and giving each the smallest finding index that still
   leaves a maximum matching for the issues after it. The choice is reported in `matches`
   (sorted by finding index) and never changes TP, FP, FN or a reward.

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
