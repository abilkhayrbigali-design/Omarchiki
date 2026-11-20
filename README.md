# AuditGym

Воспроизводимая RL-среда для дообучения ИИ-аудитора. Genesis Hackathon 2026, трек FinTech, кейс «RL-среды для дообучения моделей финансового аудита».

Среда генерирует вымышленную компанию за один отчётный период: проводки, счета, платежи, курсы валют и политику учёта. В книги внесены скрытые ошибки четырёх типов с записанным эталоном. Агент расследует их пятью инструментами за 100 шагов и отправляет выводы с доказательствами. Оценщик сопоставляет выводы с эталоном и считает награду по формуле кейса.

Все компании, люди, счета и суммы вымышлены.

## Проверка одной командой

В контейнере, без сети, в лимитах кейса:

```bash
docker build -t auditgym .
docker run --rm --network none --cpus 2 --memory 4g auditgym
```

В этих лимитах на Docker Engine 29.9.2 `check` завершился с кодом 0 за 70 с: 1308 тестов, прогон генератора и имитатор жюри ([reports/docker_check.txt](reports/docker_check.txt)). Медленные тесты (6 штук, прогоны по многим seed) включает `check --full`.

Локально (Python 3.11, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
make check
```

`check` делает четыре вещи и завершается с кодом 0, только если всё прошло:

1. **Тесты** среды, генератора, оценщика, решателя и воспроизводимости.
2. **Прогон генератора по объёму.** По 100 seed на каждую конфигурацию. Каждый эпизод проверяется независимым решателем и проверкой достаточности доказательств.
3. **Имитация закрытой проверки жюри** на свежих seed по протоколу из разделов 06 и 10 кейса: 20 эпизодов (5 чистых, 5 со взаимодействующими ошибками), четыре базовые политики, проверки K1–K4, чек-лист K5.
4. **Опубликованные определения проверок через `judge_adapter.py`** (отдельно: `python -m auditgym judge-check`): контрактные тесты C01–C20, закрытые эпизоды D01–D20, тесты matcher и награды 1–20 и примеры M01–M12, парные проверки P01–P20. Эпизоды пишутся на диск, преобразования для пар делаются переписыванием файлов, как у жюри.

Отчёт пишется в `runs/check/check_report.json`.

## Адаптер жюри

Жюри подключается через [judge_adapter.py](judge_adapter.py) в корне репозитория (реализация — [src/auditgym/judge.py](src/auditgym/judge.py)). Рабочая директория в контейнере — `/app`, `src/` уже на пути импорта:

```bash
docker run --rm --network none --cpus 2 --memory 4g --entrypoint python auditgym \
    -c "from judge_adapter import generate_episode, make_env, score_episode"
```

| Точка входа | Что делает |
|---|---|
| `generate_episode(seed, config, output_dir)` | пишет эпизод: `manifest.json`, `policy.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv` и закрытые `private/truth.json`, `private/provenance.json`. `config` — наша конфигурация ([docs/DOMAIN.md](docs/DOMAIN.md), [configs/](configs/)) или `{"team_config": {...}}` |
| `make_env()` | объект с `reset(seed, config)` и `step(action)` |
| `reset(seed, config)` | `config = {"mode": "closed", "episode_dir": DIR}` загружает каталог как есть, без генерации и смены ID; повторный `reset` начинает его заново. `{"mode": "train", "team_config": {...}}` генерирует эпизод из `seed`. Первое наблюдение: `{task, manifest, policy, remaining_steps}` |
| `step(action)` | `action = {"name": ..., "args": {...}}` (принимается и наша форма `{"action": ...}`). Возвращает `(observation, reward, terminated, truncated, info)`, наблюдение — `{result, error, remaining_steps}` |
| `score_episode(episode_dir, findings, inspection_steps)` | `{tp, fp, fn, precision, recall, f1, return}`; для структурно неверного ответа `return = -1`, остальные `null` |

**Результаты действий.** `list_records(table, cursor, limit)` → `{"records": [...], "next_cursor": str | null}`, таблицы `ledger`, `invoices`, `payments`, `fx`, не больше 50 строк. `get_record(table, id)` → запись: `id` — это `row_id` для `ledger`, `doc_id` для `invoices` и `payments`, `"YYYY-MM-DD/SRC/TGT"` для `fx` (дата и пара валют, не суррогатный ID). `get_document(doc_id)` → счёт со строками или платёж. `calculator(expression)` → десятичная строка, например `"0.3"` для `0.1+0.2`. `submit(findings)` → `{"accepted": true}`; `findings` — объект findings.json `{episode_id, findings: [...]}`, список выводов без обёртки тоже принимается. Ошибка — `{"code", "message"}` с кодом из фиксированного списка ([schemas/](schemas/)), шаг расходуется.

**Четыре типа ошибок** (поле `type`, точные строки):

| `type` | Ошибка кейса | `affected_ids` | `expected_amount_minor` |
|---|---|---|---|
| `duplicate_invoice` | дубль счёта: тот же `invoice_id` во втором документе или новый номер при том же заказе и строках | `doc_id` всех копий группы | `(k − 1) × |total_minor|`, валюта счёта |
| `payment_allocation` | платёж не распределён на счёт, распределён частично или на другой счёт | `doc_id` платежа и `doc_id` счёта | недораспределённая сумма, валюта платежа |
| `fx_conversion` | конвертация по курсу не той даты или не той валютной пары | `doc_id` платежа | разница в базовой валюте |
| `period_cutoff` | проводка документа попала в другой период, чем сам документ | `doc_id` документа | сумма документа, его валюта |

Точные определения и наборы доказательств по типам — в [docs/DOMAIN.md](docs/DOMAIN.md), разделы 6–7, и в `policy.evidence` каждого эпизода. Допуск суммы опубликован в `policy.tolerances`: 0 для всех типов, кроме `fx_conversion` (1 минимальная единица на округление). В файле эталона, где допуска нет, он равен 0.

**Как заказать сценарий** (`config` для `generate_episode` или `team_config` для `reset`; полная схема — [docs/DOMAIN.md](docs/DOMAIN.md), примеры — [configs/](configs/)). Каждый тип ошибки, сценарий взаимодействия и вид ловушки включается отдельно, число — целое или `[min, max]`; не указанное берётся по умолчанию:

```json
{"errors": {"duplicate_invoice": 0, "payment_allocation": 1, "fx_conversion": 0, "period_cutoff": 0},
 "interactions": {"fx_partial_at_boundary": 0, "duplicate_misallocated": 0},
 "variants": {"payment_allocation": ["partial"]}}
```

| Что нужно | Настройка |
|---|---|
| чистая компания | `"clean": true`; ловушки — `"distractors": {"split_payment": 3, "credit_note_refund": 2, "same_amount_orders": 2, "subscription": 3, "boundary_offset": 3, "fx_spike": 1, ...}` |
| вариант ошибки | `"variants"`: `duplicate_invoice` — `same_invoice_id`, `new_invoice_id`; `payment_allocation` — `none` (пропуск), `partial` (частичное), `other_invoice` (на чужой счёт); `fx_conversion` — `wrong_date`, `wrong_pair`; `period_cutoff` — `invoice_posted_late`, `payment_posted_late` |
| сценарий взаимодействия | `"interactions": {"fx_partial_at_boundary": 1}` (частичный иностранный платёж у границы периода, три ошибки) или `{"duplicate_misallocated": 1}` (дубль, на который ушёл платёж, две ошибки) |
| закрытые seed | `"secret_salt": "<секрет>"`: без соли открытый генератор позволяет восстановить эпизод перебором seed |

Полученный вариант записан в `private/provenance.json` (`injections[].variant`). Матрица D01–D20, которой мы проверяем себя, — `d_matrix()` в [src/auditgym/harness/judge_check.py](src/auditgym/harness/judge_check.py).

**Режимы и время.** В `closed` политика получает `reward = 0.0` до конца, включая завершение; в `train` — терминальную награду. Лимит — 60 секунд от начала `reset`: шаг после лимита не выполняется, эпизод завершается с `truncated=true` и оценивается как пустой ответ.

**Журнал оценщика.** Оценка каждого завершённого эпизода и каждого вызова `score_episode` добавляется в `env.grade_log` / `judge_adapter.SCORE_LOG` и, если задана переменная окружения `AUDITGYM_GRADER_LOG`, в этот JSONL-файл. В наблюдения и `info` она не попадает.

**Сопоставление** следует опубликованным соглашениям жюри ([docs/REQUIREMENTS.md](docs/REQUIREMENTS.md), раздел 8; код и правила — [src/auditgym/grading/](src/auditgym/grading/__init__.py)):
- неверный тип JSON-значения, отсутствующее поле или превышение лимита (30 выводов, 10 ID доказательств, 1000 символов объяснения, 10 `affected_ids`) — ответ недействителен, −1;
- неверное значение в структурно правильном выводе — FP; ID и валюта сравниваются точно, вывод со ссылкой на несуществующий `evidence_id` не сопоставляется;
- при равных максимальных сопоставлениях выбор идёт по `issue_id` и индексу вывода.

## Использование

```python
from auditgym.env import AuditEnv

env = AuditEnv(mode="open")              # "closed": награда остаётся у оценщика
obs, info = env.reset(seed=1000, config=None)
obs, reward, terminated, truncated, info = env.step(
    {"action": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}}
)
```

Лимит времени при запуске в одном процессе, без сервера: `AuditEnv(mode="closed", time_limit_s=60)`. Шаг, пришедший после лимита, не выполняется; эпизод завершается с `truncated=true` и пустым ответом. По умолчанию лимит выключен, и среда часы не читает.

Действия: `list_records(table, cursor, limit)`, `get_record(table, id)`, `get_document(doc_id)`, `calculator(expression)`, `submit(findings)`. Таблицы: `ledger`, `invoices`, `payments`, `fx`. Форматы — в [schemas/](schemas/).

### Закрытый прогон

Эталон живёт только в отдельном процессе-оценщике. Агент получает только канал JSON-строк на stdin/stdout и не выбирает эпизоды.

- **Сервер запускает организатор**, не агент, и в настоящей проверке — от другого пользователя ОС, чтобы агент не мог читать эпизоды с `private/`, расписание и журнал.
- **Эпизоды задаёт организатор**: `--schedule FILE` (JSON-список `{"seed": N, "config": "путь" | {...}}`) и/или `--episode-dir DIR` (каталог эпизода или каталог с эпизодами, повторяемая опция). Каждый `{"op": "reset"}` даёт следующий эпизод; `reset` с `seed` или `config` отклоняется (`invalid_request`), после последнего эпизода — `schedule_exhausted`.
- **Журнал оценщика — у организатора**: `--grader-log` обязателен и должен быть абсолютным путём вне досягаемости агента. Награда, метрики и сопоставления пишутся только туда; агенту в закрытом режиме возвращается `reward=0.0`.
- **Без повторов**: эпизод, брошенный через `reset`, `close` или обрыв канала, оценивается как пустой ответ и пишется в журнал (`end: "abandoned"`). Лимит времени `--time-limit` (по умолчанию 60 с) считает сервер; шаг после лимита не выполняется, эпизод завершается с `truncated=true` и пустым ответом (`end: "timeout"`).
- **Секретная соль**: генератор открыт, поэтому для скрытых seed задайте в конфигурации расписания `"secret_salt": "<секрет организатора>"`. Без соли эпизод можно пересобрать перебором seed; с солью — нельзя, а тот же seed с той же солью воспроизводит эпизод побайтово.

```bash
# организатор, от пользователя grader:
cat > /srv/grader/schedule.json <<'EOF'
[{"seed": 734119, "config": {"config_id": "hidden.v1", "secret_salt": "change-me"}},
 {"seed": 902211, "config": {"config_id": "hidden.v1", "clean": true, "secret_salt": "change-me"}}]
EOF
python -m auditgym serve --mode closed \
    --schedule /srv/grader/schedule.json \
    --grader-log /srv/grader/logs/grader_log.jsonl
```

Обвязка организатора соединяет stdout сервера со stdin агента и stdout агента со stdin сервера. Агент подключается к выданному каналу:

```python
import sys
from auditgym.client import RemoteEnv

env = RemoteEnv.from_streams(sys.stdin.buffer, sys.stdout.buffer)  # stdout агента — это канал
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step({"action": "list_records", "args": {"table": "ledger"}})
```

Для обучения и тестов `RemoteEnv(mode="open")` запускает сервер сам и выбирает `seed`/`config`; это не закрытая проверка, потому что процессом-оценщиком тогда владеет агент.

## Команды

| Команда | Что делает |
|---|---|
| `python -m auditgym generate --seed 7 --config configs/train.v1.json --out runs/ep7` | сгенерировать эпизод на диск |
| `python -m auditgym demo --seed 1000` | пошаговый показ эпизода со скриптовой политикой |
| `python -m auditgym run --policy reconcile --seed 1000 --out runs/t.jsonl` | прогнать политику, записать траекторию |
| `python -m auditgym sweep --config configs/dev.v1.json --seeds 1000` | прогон генератора по объёму с независимой проверкой |
| `python -m auditgym baselines --config configs/dev.v1.json --seeds 300` | четыре базовые политики на многих эпизодах, P, R, F1, возврат, шаги, время |
| `python -m auditgym jury` | имитация проверки жюри с оценкой по критериям и пробами на трюки |
| `python -m auditgym judge-check` | опубликованные определения проверок (блоки 1–4) через `judge_adapter.py` |
| `python -m auditgym transform --input DIR --out DIR2 --salt s` | перестановка строк и смена ID готового эпизода |
| `python -m auditgym export` | открытые train/dev-эпизоды и `trajectories/trajectories.jsonl` |

Политики: `random` (случайный исследователь с фиксированным seed), `reconcile` (сверка по опубликованным правилам), `flag_all` (атака «пометить всё»), `oracle` (эталон, только для оценщика).

## Что где лежит

| Путь | Содержимое |
|---|---|
| `src/auditgym/world`, `books`, `inject`, `present`, `episode` | генератор: история компании, проводки, ошибки, публичные ID, сборка эпизода |
| `src/auditgym/env`, `server`, `client.py` | среда `reset/step`, инструменты, калькулятор, процесс-оценщик |
| `judge_adapter.py`, `src/auditgym/judge.py` | адаптер жюри: `generate_episode`, `make_env`, `score_episode` |
| `src/auditgym/grading` | разбор ответа, сопоставление с эталоном, метрики, награда |
| `src/auditgym/verify` | независимый решатель и проверка достаточности доказательств |
| `src/auditgym/policies`, `harness` | базовые политики, прогоны, отчёты, имитатор жюри |
| `configs/` | версионированные конфигурации и открытые seed |
| `episodes/` | 10 train- и 10 dev-эпизодов; эталон каждого — в его `private/` |
| `trajectories/trajectories.jsonl` | записанные траектории базовых политик на train-эпизодах |
| `reports/` | сохранённые прогоны, из которых взяты все числа записки |
| `docs/` | [записка о дизайне](docs/DESIGN_NOTE.md), [требования](docs/REQUIREMENTS.md), [доменная модель](docs/DOMAIN.md), [архитектура](docs/ARCHITECTURE.md), [трассировка требований к тестам](docs/TRACEABILITY.md) |

## Использованные инструменты

Рантайм — только стандартная библиотека Python 3.11. Для тестов: pytest, hypothesis, jsonschema; для стиля: ruff; окружение: uv. Код написан командой с помощью ИИ-ассистентов для программирования (Claude Code).

## Лицензия

MIT, см. [LICENSE](LICENSE).
