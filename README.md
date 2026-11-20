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

Действия: `list_records(table, cursor, limit)`, `get_record(table, id)`, `get_document(doc_id)`, `calculator(expression)`, `submit(findings)`. Таблицы: `ledger`, `invoices`, `payments`, `fx`. Форматы — в [schemas/](schemas/).

Отдельный процесс-оценщик для закрытого прогона (эталон живёт только в нём, политика общается с ним JSON-строками):

```bash
uv run python -m auditgym serve
```

```python
from auditgym.client import RemoteEnv
```

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
| `docs/` | [записка о дизайне](docs/DESIGN_NOTE.md), [требования](docs/REQUIREMENTS.md), [доменная модель](docs/DOMAIN.md), [архитектура](docs/ARCHITECTURE.md) |

## Использованные инструменты

Рантайм — только стандартная библиотека Python 3.11. Для тестов: pytest, hypothesis, jsonschema; для стиля: ruff; окружение: uv. Код написан командой с помощью ИИ-ассистентов для программирования (Claude Code).

## Лицензия

MIT, см. [LICENSE](LICENSE).
