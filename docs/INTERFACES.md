# Интерфейсы между пакетами

Пакеты пишутся параллельно разными агентами. Здесь — точные точки стыковки. Менять их может только интегратор; нужна правка — напиши ему.

Общее: типы записей в `auditgym.contracts.types`, лимиты в `auditgym.contracts.constants`, коды ошибок в `auditgym.contracts.errors`, конфиг в `auditgym.contracts.config`, файлы эпизода в `auditgym.contracts.io`, манифест и политика в `auditgym.contracts.meta`. Эпизод-фикстура с четырьмя ошибками: `auditgym.fixtures.tiny.build_tiny_episode()`.

## WP1 + WP2 — генератор (`world/`, `books/`, `inject/`, `episode/`)

```python
from auditgym.episode.generate import generate_episode, generate_with_clean

episode: Episode = generate_episode(seed: int, config: dict | None = None)
episode, clean = generate_with_clean(seed: int, config: dict | None = None)
#   clean: PublicEpisode — те же книги до внесения ошибок, с теми же публичными ID
```

- `config` — сырой или нормализованный словарь; внутри всегда `normalize_config`.
- Детерминизм: одинаковые `seed`, `config`, `GENERATOR_VERSION` дают побайтово одинаковый эпизод (`contracts.io.fingerprint`).
- Самопроверка: `generate_episode(..., verify_fn=None)`. Если передан `verify_fn(public) -> tuple[Issue, ...]`, генератор сравнивает его ответ с эталоном по `(type, affected_ids, expected_amount_minor, currency)` и при расхождении повторяет попытку со следующим `attempt`. Число попыток пишется в `provenance["attempts"]`. Интегратор подключит решатель WP5.
- Инварианты: `auditgym.books.invariants.check(public: PublicEpisode) -> list[Violation]`, где `Violation` содержит код инварианта (`I1`…`I7`) и ID записей. На `clean` — пустой список.

## WP3 — среда (`env/`, `server/`, `client.py`)

```python
from auditgym.env import AuditEnv

env = AuditEnv(mode="open" | "closed", generate_fn=None, grader=None, grader_log=None)
obs, info = env.reset(seed: int, config: dict | None = None)
obs, info = env.reset_with_episode(episode: Episode)          # для тестов и фикстур
obs, reward, terminated, truncated, info = env.step(action: dict)
```

- `generate_fn` по умолчанию — `auditgym.episode.generate.generate_episode` (импорт при первом вызове). `grader` по умолчанию — `auditgym.grading.grade_submission` (сигнатура в `contracts/grading_api.py`). В тестах WP3 передаёт свои заглушки.
- Действие: `{"action": name, "args": {...}}`, схема `schemas/action.schema.json`.
- Наблюдение: `{"result", "error", "steps_remaining"}`; первое — `{"task", "manifest", "policy", "steps_remaining"}`. Схема `schemas/observation.schema.json`.
- `list_records` и `get_record` по `invoices` отдают `Invoice.summary()`, `get_document` — полный документ. ID записи: `ledger` → `row_id`, `invoices` и `payments` → `doc_id`, `fx` → `FxRate.rate_id`.
- Форма `result` в наблюдении шага (при ошибке `result = null`):

  | Действие | `result` |
  |---|---|
  | `list_records` | `{"table", "records": [...], "next_cursor": str \| null}` |
  | `get_record` | `{"table", "record": {...}}` |
  | `get_document` | `{"document": {...}}` |
  | `calculator` | `{"value": "<десятичная строка>"}` |
  | `submit` | `{"submitted": true, "findings_count": n}` — ничего о правильности |
- `inspection_steps` — все шаги до `submit`, включая ошибочные. При исчерпании 100 шагов: `truncated=true`, оценка пустого ответа.
- `closed`: политике `reward=0.0`, `info={"reward_withheld": true}`. В обоих режимах, если задан `grader_log`, туда пишется `GradeResult.to_log()`.
- `env.truncate()` завершает эпизод по тайм-ауту: `truncated=true`, оценка пустого ответа. Его вызывает harness при превышении 60 секунд; сама среда часы не читает. В протоколе сервера — `{"op": "truncate"}`.
- Сервер: `python -m auditgym serve` (или `python -m auditgym.server`) читает JSON-строки `{"op": "reset", "seed", "config"}`, `{"op": "step", "action"}`, `{"op": "truncate"}`, `{"op": "close"}` и отвечает JSON-строками. Опции: `--mode open|closed`, `--grader-log PATH` (в закрытом режиме по умолчанию `runs/grader_log.jsonl`), `--episode-dir DIR` (раздавать эпизод с диска). Ошибки протокола — отдельные коды `invalid_request`, `invalid_config`, `internal_error`; кривая строка запроса шаг не тратит. `auditgym.client.RemoteEnv` — те же `reset/step/truncate` поверх него, ошибки — `RemoteEnvError(code)`.
- Решения среды, принятые при реализации:
  - кривые аргументы `submit` (нет `findings`, лишние ключи) — `invalid_arguments`, шаг тратится, эпизод идёт дальше; содержимое `findings` целиком уходит оценщику, и кривая структура там завершает эпизод с наградой −1;
  - результат `submit` не сообщает, валиден ли ответ; в открытом режиме единственный сигнал — награда;
  - исключение в оценщике: эпизод завершается с наградой 0.0, в журнале `grader_failure: true`;
  - `info`: при сбросе `{episode_id, mode, env_version, max_steps}`, на шагах исследования `{}`, на последнем шаге `{"reward_withheld": bool}`;
  - `viewed_ids` для экспериментальной награды — все ID, которые агенту показали: строки страниц `list_records`, `get_record`, `get_document`.

## WP4 — оценщик (`grading/`)

```python
from auditgym.grading import grade_submission, parse_submission, metrics, shared_reward

result: GradeResult = grade_submission(payload, truth, *, episode_id, inspection_steps, viewed_ids=None)
findings: tuple[Finding, ...] = parse_submission(payload, episode_id)   # SubmissionError при ошибке
p, r, f1 = metrics(tp, fp, fn, n_truth, n_findings)
reward: float = shared_reward(f1, inspection_steps)
```

- Структура строго по `schemas/submission.schema.json`. Любое отклонение: `valid=False`, `reward=-1.0`.
- Арифметика награды точная (`Fraction`), в `float` — только на выходе. `metrics()` возвращает `Fraction`.
- Дополнительно экспортируются `empty_submission`, `reward_exact`, `match_findings`, `compatible`, `SubmissionError`, `SubmissionErrorCode`, `STEP_COST_FRACTION`, `INVALID_REWARD`.
- `grade_submission` не бросает исключений ни на каком входе: любая кривая структура даёт `valid=False`, награда −1.
- Целое, записанное как `5.0`, принимается и переводится в `int`, как в JSON Schema; дробное отвергается. Валюта — ровно три заглавные латинские буквы.
- Зависит только от `contracts/` и `util/`.

## WP5 — независимый решатель (`verify/`)

```python
from auditgym.verify import solve
from auditgym.verify.evidence import check_evidence

issues: tuple[Issue, ...] = solve(public: PublicEpisode)
check: EvidenceCheck = check_evidence(public, issue_type, affected_ids, evidence_ids)
#   EvidenceCheck(ok: bool, amount_minor: int | None, currency: str | None, reason: str)
```

- Читает только открытые данные и `docs/DOMAIN.md`. Не импортирует и не читает `world/`, `inject/`, `episode/`, `books/`.
- `solve` возвращает ровно те ошибки, что есть в данных. `issue_id` — любой стабильный, сравнение идёт по `(type, affected_ids, expected_amount_minor, currency)`.
- `check_evidence` пересчитывает тип и сумму, используя только записи из `evidence_ids`, манифест и политику.

## WP6 — политики и запуск (`policies/`, `harness/`)

```python
class Policy(Protocol):
    name: str
    def reset(self, observation: dict, info: dict) -> None: ...
    def act(self, observation: dict) -> dict: ...
```

Oracle получает эталон от harness при создании, не через среду.
