# Трассировка требований

Каждое требование из [REQUIREMENTS.md](REQUIREMENTS.md): где реализовано и какие тесты его проверяют (по имени теста, `test_<код>_…`). Файл собирается командой `python -m auditgym traceability` из списка тестов `pytest --collect-only`.

Тестовых функций: 508; с параметризацией проверок больше тысячи.

| Код | Где реализовано | Тесты |
|---|---|---|
| C01 | вся архитектура: среда, а не приложение | архитектурное требование, см. DESIGN_NOTE §1 |
| C02 | env/, trajectories/, grading/ | `test_requirements.py` (1) |
| C03 | только stdlib в рантайме | `test_requirements.py` (1) |
| C04 | DESIGN_NOTE §8 | `test_requirements.py` (1) |
| G01 | inject/ (4 инжектора, config.errors) | `test_data_truth.py` (4), `test_verify_scenarios.py` (1), `test_verify_tiny.py` (1) |
| G02 | inject/ (2 сценария, config.interactions) | `test_data_truth.py` (2), `test_verify_scenarios.py` (2) |
| G03 | world/ (X1–X10), config.clean | `test_data_truth.py` (3), `test_verify_scenarios.py` (2) |
| G04 | world → books → inject, provenance.changes | `test_data_provenance.py` (2) |
| G05 | truth.evidence_sets, verify/evidence.py, policy.evidence | `test_requirements.py` (1) |
| G06 | world/, present/, policy | `test_data_determinism.py` (3), `test_data_review.py` (1), `test_verify_scenarios.py` (1) |
| G07 | ловушки X1–X10, harness/probes.py | `test_data_review.py` (3), `test_trick_probes.py` (2) |
| D01 | config.size, max_issues ≤ 12 | `test_data_presentation.py` (1), `test_data_provenance.py` (2), `test_data_review.py` (3), `test_data_truth.py` (1) |
| D02 | contracts/meta.py make_manifest | `test_requirements.py` (1) |
| D03 | util/money.py, util/timeutil.py | `test_requirements.py` (1) |
| D04 | contracts/types.py LedgerRow | `test_data_presentation.py` (1), `test_requirements.py` (1) |
| D05 | contracts/types.py Invoice | `test_requirements.py` (1) |
| D06 | contracts/types.py Payment | `test_requirements.py` (1) |
| D07 | contracts/types.py FxRate | `test_requirements.py` (1) |
| D08 | contracts/meta.py make_policy | `test_verify_scenarios.py` (4) |
| D09 | canon.csv_text, types, policy.formats | `test_requirements.py` (1) |
| D10 | truth.json, provenance.json | `test_data_provenance.py` (1), `test_data_truth.py` (2) |
| D11 | server/ (закрытый режим, расписание), env/session.py | `test_env_server_schedule.py` (13), `test_leak_server_process.py` (11) |
| D12 | env/observe.py, contracts/errors.py | `test_data_presentation.py` (2), `test_leak_blind_swap.py` (3), `test_leak_scan.py` (5), `test_leak_server_process.py` (2) |
| E01 | AuditEnv.reset | `test_env_reset_lifecycle.py` (3), `test_env_reset_step.py` (4) |
| E02 | AuditEnv.step | `test_env_reset_step.py` (2) |
| E03 | util/rng.py, util/canon.py | `test_env_isolation_determinism.py` (6), `test_env_reset_lifecycle.py` (2) |
| E04 | env/actions.py | `test_env_reset_step.py` (6) |
| E05 | env/store.py | `test_env_pagination.py` (9) |
| E06 | env/calculator.py | `test_env_calculator.py` (12), `test_leak_import_boundary.py` (1) |
| E07 | contracts/errors.py | `test_env_errors.py` (7), `test_env_protocol.py` (3) |
| E08 | env/observe.py | `test_env_reset_step.py` (2), `test_leak_import_boundary.py` (1) |
| E09 | env/session.py | `test_env_steps.py` (5) |
| E10 | env/session.py truncate(), server/host.py (лимит времени) | `test_env_reset_lifecycle.py` (4), `test_env_server_schedule.py` (4), `test_env_steps.py` (5), `test_env_time_limit.py` (18), `test_leak_import_boundary.py` (1) |
| E11 | env/session.py | `test_env_steps.py` (1) |
| E12 | mode=closed, grader_log | `test_env_modes.py` (7), `test_env_reset_lifecycle.py` (1), `test_env_server_schedule.py` (2), `test_leak_scan.py` (1) |
| S01 | grading/parse.py | `test_env_reset_step.py` (2), `test_reward_parse.py` (4), `test_reward_schema_crosscheck.py` (4) |
| S02 | grading/parse.py | `test_reward_parse.py` (7) |
| S03 | grade_submission valid=False | `test_env_errors.py` (3), `test_env_modes.py` (1), `test_env_steps.py` (2), `test_reward_never_raises.py` (5), `test_reward_parse.py` (8) |
| S04 | grading/match.py | `test_reward_hypothesis.py` (2), `test_reward_matching.py` (16), `test_reward_never_raises.py` (1), `test_reward_parse.py` (1) |
| S05 | grading/match.py | `test_env_steps.py` (1), `test_reward_hypothesis.py` (2), `test_reward_matching.py` (4) |
| S06 | docstring grading/ | `test_reward_boundaries.py` (3), `test_reward_matching.py` (5) |
| S07 | grading/reward.py | `test_reward_metrics.py` (4) |
| S08 | grading/reward.py | `test_reward_boundaries.py` (1), `test_reward_hypothesis.py` (2), `test_reward_metrics.py` (7), `test_reward_s11_attacks.py` (3) |
| S09 | explanation не читается | `test_reward_hypothesis.py` (1), `test_reward_matching.py` (2) |
| S10 | tests/reward | `test_reward_s10_properties.py` (17) |
| S11 | tests/reward | `test_reward_s11_attacks.py` (24) |
| V01 | configs/splits.json | `test_requirements.py` (1) |
| V02 | схема конфигурации, harness/transform.py | `test_requirements.py` (1) |
| V03 | configs/clean.v1.json, configs/interacting.v1.json | `test_requirements.py` (1) |
| V04 | harness/run.py aggregate | `test_requirements.py` (1) |
| K1 | harness/jury.py k1_checks | `test_judge_protocol.py` (3), `test_jury_checks_fixture.py` (1), `test_leak_import_boundary.py` (5) |
| K2 | harness/sweep.py | `test_data_review.py` (2), `test_data_sweep.py` (1), `test_data_truth.py` (3), `test_judge_protocol.py` (1), `test_verify_evidence.py` (10), `test_verify_tiny.py` (4) |
| K3 | harness/jury.py k3_checks | `test_judge_protocol.py` (2), `test_jury_checks_fixture.py` (1) |
| K4 | harness/transform.py, k4_checks | `test_data_determinism.py` (3), `test_data_presentation.py` (2), `test_judge_protocol.py` (3), `test_published_episodes.py` (1), `test_transform.py` (6), `test_verify_invariance.py` (2) |
| K5 | README, configs, trajectories, DESIGN_NOTE | `test_published_episodes.py` (1) |
| J01 | judge_adapter.py, auditgym/judge.py | `test_judge_adapter.py` (3) |
| J02 | auditgym/judge.py (оболочки и формы результатов) | `test_judge_adapter.py` (2), `test_judge_protocol.py` (1) |
| J03 | JudgeEnv.reset: episode_dir, team_config, mode | `test_judge_adapter.py` (3) |
| J04 | auditgym/judge.py score_episode | `test_judge_adapter.py` (2) |
| J05 | JudgeEnv: 60 с от начала reset | `test_judge_adapter.py` (1), `test_judge_protocol.py` (1) |
| J06 | grading/parse.py, grading/match.py (known_ids) | `test_judge_protocol.py` (2), `test_reward_jury_conventions.py` (4) |
| J07 | grading/match.py match_findings | `test_reward_jury_conventions.py` (2) |
| J08 | contracts/io.py read_episode, harness/judge_check.py block4 | `test_judge_protocol.py` (2) |
| J09 | contracts/config.py (variants), harness/judge_check.py d_matrix | `test_judge_protocol.py` (1) |
| J10 | README, раздел «Адаптер жюри» | `test_judge_adapter.py` (1) |
