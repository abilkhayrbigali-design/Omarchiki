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

`check` делает три вещи и завершается с кодом 0, только если всё прошло:

1. **Тесты** среды, генератора, оценщика, решателя и воспроизводимости.
2. **Прогон генератора по объёму.** По 100 seed на каждую конфигурацию. Каждый эпизод проверяется независимым решателем и проверкой достаточности доказательств.
3. **Имитация закрытой проверки жюри** на свежих seed по протоколу из разделов 06 и 10 кейса: 20 эпизодов (5 чистых, 5 со взаимодействующими ошибками), четыре базовые политики, проверки K1–K4, чек-лист K5. Отчёт пишется в `runs/check/check_report.json`.

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
| `python -m auditgym transform --input DIR --out DIR2 --salt s` | перестановка строк и смена ID готового эпизода |
| `python -m auditgym export` | открытые train/dev-эпизоды и `trajectories/trajectories.jsonl` |

Политики: `random` (случайный исследователь с фиксированным seed), `reconcile` (сверка по опубликованным правилам), `flag_all` (атака «пометить всё»), `oracle` (эталон, только для оценщика).

## Что где лежит

| Путь | Содержимое |
|---|---|
| `src/auditgym/world`, `books`, `inject`, `present`, `episode` | генератор: история компании, проводки, ошибки, публичные ID, сборка эпизода |
| `src/auditgym/env`, `server`, `client.py` | среда `reset/step`, инструменты, калькулятор, процесс-оценщик |
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
