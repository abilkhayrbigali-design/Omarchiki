# Интерфейсы между пакетами

Пакеты пишутся параллельно разными агентами. Здесь — точные точки стыковки. Менять их может только интегратор; нужна правка — напиши ему.

Общее: типы записей в `auditgym.contracts.types`, лимиты в `auditgym.contracts.constants`, коды ошибок в `auditgym.contracts.errors`, конфиг в `auditgym.contracts.config`, файлы эпизода в `auditgym.contracts.io`, манифест и политика в `auditgym.contracts.meta`. Эпизод-фикстура с четырьмя ошибками: `auditgym.fixtures.tiny.build_tiny_episode()`.

## WP1 + WP2 — генератор (`world/`, `books/`, `inject/`, `episode/`)

```python
from auditgym.episode.generate import generate_episode, generate_with_clean

episode: Episode = generate_episode(seed: int, config: dict | None = None)
episode, clean = generate_with_clean(seed: int, config: dict | None = None, *, verify_fn=None)
#   clean: PublicEpisode — те же книги до внесения ошибок, с теми же публичными ID
```

- Ключ случайных потоков строится от `seed`, конфига без секции `presentation`, версии генератора и номера попытки. Поэтому `presentation.id_salt` и `presentation.order` меняют только ID и порядок строк, а не содержание эпизода.
- Секция `variants` (какие варианты каждого типа ошибки разрешены, [DOMAIN.md](DOMAIN.md), раздел 6) входит в ключ, только если что-то ограничивает. Без неё или с полным набором нормализованный конфиг, ключ и эпизод те же, что до её появления. Вариант каждой внесённой ошибки — в `provenance["injections"][i]["variant"]`.
- `provenance["changes"]` — упорядоченный список изменений на уровне записей с публичными ID: `{"issue_ids", "table", "id", "before", "after"}`, где `before = null` для добавленной записи и `after = null` для удалённой. Откат изменений в обратном порядке на опубликованных данных воспроизводит чистые книги.
- `provenance["scenarios"]` — список взаимодействующих сценариев с их `issue_ids`.

- `config` — сырой или нормализованный словарь; внутри всегда `normalize_config`.
- Детерминизм: одинаковые `seed`, `config`, `GENERATOR_VERSION` дают побайтово одинаковый эпизод (`contracts.io.fingerprint`).
- Самопроверка: `generate_episode(..., verify_fn=None)`. Если передан `verify_fn(public) -> tuple[Issue, ...]`, генератор сравнивает его ответ с эталоном по `(type, affected_ids, expected_amount_minor, currency)` и при расхождении повторяет попытку со следующим `attempt`. Число попыток пишется в `provenance["attempts"]`. Интегратор подключит решатель WP5.
- Инварианты: `auditgym.books.invariants.check(public: PublicEpisode) -> list[Violation]`, где `Violation` содержит `code` (`I1`…`I7`), `ids`, `amount_minor`, `currency`, `detail`. На `clean` — пустой список.

## WP3 — среда (`env/`, `server/`, `client.py`)

```python
from auditgym.env import AuditEnv

env = AuditEnv(mode="open" | "closed", generate_fn=None, grader=None, grader_log=None)
obs, info = env.reset(seed: int | None = None, config: dict | None = None, *, options: dict | None = None)
obs, info = env.reset_with_episode(episode: Episode)          # для тестов, фикстур, эпизодов с диска
obs, reward, terminated, truncated, info = env.step(action: dict)
obs, reward, terminated, truncated, info = env.truncate()     # лимит времени (решает вызывающий)
obs, reward, terminated, truncated, info = env.abandon()      # вызывающий бросил эпизод
env.active                                                    # идёт ли эпизод
```

- `reset` в стиле Gymnasium: `env.reset(seed=1, options=None)` работает; `options` может содержать только `{"config": ...}` (конфиг и в аргументе, и в `options` — `ValueError`). `seed=None` даёт детерминированную последовательность: 0 при первом сбросе, дальше предыдущий seed + 1 (`reset_with_episode` её не сдвигает).
- Неудачный `reset` ничего не ломает: новый эпизод генерируется и проверяется до того, как трогается текущий. Плохой конфиг — `ConfigError` (или `TypeError`/`ValueError` на кривые аргументы), текущий эпизод продолжается как был и не пишется в журнал.
- Успешный `reset`/`reset_with_episode` посреди эпизода завершает старый как брошенный: `truncated=true`, оценка пустого ответа, `inspection_steps` = потраченные шаги, запись в журнале с `"end": "abandoned"`. Это делается, когда задан `grader_log`; без журнала (быстрые тесты, обучение в процессе) старый эпизод просто отбрасывается. `env.abandon()` делает то же явно и оценивает всегда.
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
- `closed`: политике `reward=0.0`, `info={"reward_withheld": true}`. В обоих режимах, если задан `grader_log`, туда пишется `GradeResult.to_log()` плюс заголовок `{episode_id, step_count, end, env_version, mode}`; `end` — `submit`, `step_limit`, `timeout` или `abandoned`.
- `env.truncate()` завершает эпизод по тайм-ауту: `truncated=true`, оценка пустого ответа, шаг не тратится, `end: "timeout"`. Его вызывают сервер и harness; сама среда и генератор часы не читают.
- Время. Лимит на эпизод считается монотонными часами от ответа на `reset` и живёт только в транспорте: в сервере (`auditgym.server.host.EpisodeHost`, опция `--time-limit`, часы подставляются в тестах) и в `harness.run.run_episode` (`time_limit_s`, `clock`). Запрос шага, пришедший после лимита, не выполняется: сервер вызывает `env.truncate()` и отвечает его результатом. В harness время `policy.act` входит в лимит: если действие вернулось после лимита, оно отбрасывается и вызывается `env.truncate()`.
- Сервер: `python -m auditgym.server [опции]` (или `python -m auditgym serve -- [опции]`), JSON-строки на stdin/stdout.
  - Запросы: `{"op": "reset"}` (следующий эпизод расписания), `{"op": "reset", "seed", "config"}` (только открытый режим без расписания), `{"op": "step", "action"}`, `{"op": "truncate"}` (сдаться: оценка пустого ответа), `{"op": "close"}`.
  - Опции: `--mode open|closed`; `--grader-log PATH`; `--schedule FILE` — JSON-список `{"seed": int, "config": путь | объект | null}`, относительный путь конфига — от каталога файла расписания, конфиги проверяются при старте; `--episode-dir DIR` (повторяемая) — каталог одного эпизода или каталог, подкаталоги которого — эпизоды (по порядку имён, каталоги с точкой пропускаются; нужен `private/`); источники раздаются в порядке опций командной строки; `--time-limit SECONDS` — в закрытом режиме по умолчанию 60, в открытом по умолчанию выключен.
  - Закрытый режим без расписания (`--schedule`/`--episode-dir`) или без абсолютного `--grader-log` не стартует (код 2, сообщение в stderr). Путь журнала по умолчанию нет ни в каком режиме; журнал не может совпадать с каналом stdout. В stdout в закрытом режиме нет ничего об оценках.
  - С расписанием (в закрытом режиме всегда) `reset` с `seed` или `config` (даже `null`) или любым другим полем — `invalid_request`, состояние не меняется. Каждый `reset` берёт следующий эпизод; эпизод расписания не выдаётся повторно, даже если не загрузился. После последнего — `schedule_exhausted`. В открытом режиме без расписания клиент сам выбирает `seed` и `config` (обучение); `reset` без `seed` — `invalid_request`.
  - Без бесплатных повторов, в обоих режимах: если эпизод идёт, когда приходит корректный `reset`, `close`, EOF (или SIGTERM серверу), он сначала завершается как `truncated` с пустым ответом, оценивается и пишется в журнал с `end: "abandoned"` (или `"timeout"`, если лимит уже вышел). Только потом выполняется сам запрос.
  - Ошибки протокола — фиксированные коды `invalid_request`, `invalid_config`, `schedule_exhausted`, `internal_error`; кривая строка запроса шаг не тратит.
  - `auditgym.client.RemoteEnv` — те же `reset(seed=None, config=None, *, options=None)/step/truncate` поверх сервера, ошибки — `RemoteEnvError(code)`. `RemoteEnv(...)` сам запускает сервер (обучение, тесты); `RemoteEnv.from_streams(reader, writer)` подключается к каналу, который выдал организатор (закрытый прогон).
- Решения среды, принятые при реализации:
  - любое действие-объект с `"action": "submit"` завершает эпизод (`terminated=true`). Если аргументы не по схеме действия (нет `args`, `args` не объект, нет `findings`, лишние ключи, слишком глубокая вложенность), наблюдение — `invalid_arguments`, оценщику уходит сырое значение `args` (или `null`, если `args` нет), и он оценивает это как неверную структуру: награда −1 (в закрытом режиме удерживается), в журнале `valid: false` и `action_error: "invalid_arguments"`. Исключение: если `args` сам по себе — объект ответа `{"episode_id", "findings"}` без обёртки `findings`, оценщику уходит `null`, чтобы кривое действие никогда не засчитывалось как верный ответ. Если `args` по схеме, содержимое `findings` целиком уходит оценщику, и кривая структура там тоже завершает эпизод с наградой −1;
  - результат `submit` не сообщает, валиден ли ответ; в открытом режиме единственный сигнал — награда;
  - исключение в оценщике: эпизод завершается с наградой 0.0, в журнале `grader_failure: true`;
  - `info`: при сбросе `{episode_id, mode, env_version, max_steps}`, на шагах исследования `{}`, на последнем шаге `{"reward_withheld": bool}`;
  - `viewed_ids` для экспериментальной награды — все ID, которые агенту показали: строки страниц `list_records`, `get_record`, `get_document`;
  - калькулятор принимает обычную десятичную запись: унарный плюс и минус (`+5`, `-+5`), `.5`, `5.`; отвергает экспоненты, `**`, `_`, имена, не-ASCII цифры, одинокую точку; лимиты длины (256 символов) и цифр (40 на число) прежние.

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
