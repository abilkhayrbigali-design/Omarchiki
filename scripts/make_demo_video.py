#!/usr/bin/env python3
"""Render the AuditGym demo video from real command output.

    python scripts/make_demo_video.py            # -> runs/video/auditgym_demo.mp4

Every terminal scene runs its commands for real (bash -c, from the repository root) and
shows their stdout; no number on screen is typed by hand. Captions carry no measured
numbers; the "expect" patterns below fail the build if the output stops supporting them.

Needs: python 3.11 (stdlib only), uv, ffmpeg with libx264, ImageMagick 7 (`magick`),
Noto Sans and Noto Sans Mono fonts. Writes only under runs/video/ and is idempotent.

Editing the video: change SCENES. A terminal scene is
    {"caption": str, "sub": str, "steps": [step, ...], "hold": seconds}
and a step is
    {"sh": "command as shown and run",   # may use "\\\n" continuations
     "line_delay": s,                    # seconds per revealed output line
     "highlight": [(regex, style[, pause_s]), ...],
     "squeeze": (regex with 3 groups, width),  # shorten group 2 to width with "…"
     "expect": [regex, ...],             # must match the output or the build fails
     "same_episode": True}               # output names the same episode id as the others
A card scene is {"kind": "card", ...} with title, lines, table, command, footer.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO / "runs" / "video"
CLEAN_BEFORE_RUN = ["runs/video/ep", "runs/video/ep_again", "runs/video/ep_1001"]

EPISODE_ID = re.compile(r"\bep_[0-9A-Z]+\b")
FORMULA = "награда = clip(2·F1 − 1 − 0.002 · шаги проверки, −1, 1)"

SCENES: list[dict] = [
    {
        "kind": "card",
        "title": "AuditGym — RL-среда для ИИ-аудитора",
        "lines": [
            "Генератор вымышленных компаний со скрытыми бухгалтерскими ошибками,",
            "среда reset/step с пятью инструментами и оценщик, сверяющий выводы с эталоном",
        ],
        "footer": "Genesis Hackathon 2026 · трек FinTech · все данные вымышлены",
        "hold": 8,
    },
    {
        "caption": "Генерация эпизода по seed",
        "sub": "Вымышленная компания за период: проводки, счета, платежи, курсы, политика учёта",
        "steps": [
            {
                "sh": "uv run python -m auditgym generate --seed 1000 \\\n"
                "  --config configs/train.v1.json --out runs/video/ep",
                "same_episode": True,
                "highlight": [(r"fingerprint", "good")],
                "expect": [r"fingerprint [0-9a-f]+"],
            },
            {
                "sh": "find runs/video/ep -type f | sort",
                "line_delay": 0.3,
                "highlight": [(r"/private/", "private")],
                "expect": [r"private/truth\.json", r"private/provenance\.json"],
            },
            {"sh": "head -n 4 runs/video/ep/ledger.csv", "line_delay": 0.35},
        ],
        "hold": 7,
    },
    {
        "caption": "Эталон лежит в private/ — его видит только оценщик",
        "sub": "Ошибка в эталоне: тип, затронутые записи, сумма и допустимые наборы доказательств",
        "steps": [
            {
                "sh": "head -n 20 runs/video/ep/private/truth.json",
                "line_delay": 0.16,
                "highlight": [(r'"(type|expected_amount_minor)"', "warn")],
                "expect": [r'"issues"', r'"type"'],
            },
        ],
        "hold": 8,
    },
    {
        "caption": "Скриптовая политика расследует эпизод инструментами среды",
        "sub": FORMULA,
        "steps": [
            {
                "sh": "uv run python -m auditgym demo --seed 1000",
                "same_episode": True,
                "line_delay": 0.3,
                "squeeze": (r"^(\s*step\s+\d+\s+\S+\s+)(.*?)\s+(-> .*)$", 48),
                "highlight": [
                    (r"^Reward ", "warn", 1.5),
                    (r"^\s*step\s+\d+\s+submit", "good", 0.6),
                    (r"^(Episode|Records|Truth)", "dim"),
                ],
                "expect": [r"^Reward \S+\s+F1 \S+\s+TP \d+ FP \d+ FN \d+"],
            },
        ],
        "hold": 9,
    },
    {
        "caption": "Трюки не окупаются",
        "sub": "«Пометить всё» (flag_all) и случайный обход (random) на том же эпизоде",
        "steps": [
            {
                "sh": "uv run python -m auditgym run --policy flag_all --seed 1000 \\\n"
                "  | grep -wE '\"(episode_id|policy|tp|fp|fn|f1|reward)\"'",
                "same_episode": True,
                "highlight": [(r'"reward"', "bad", 0.6), (r'"policy"', "cmd")],
                "expect": [r'"reward": -', r'"policy": "flag_all"'],
            },
            {
                "sh": "uv run python -m auditgym run --policy random --seed 1000 \\\n"
                "  | grep -wE '\"(episode_id|policy|tp|fp|fn|f1|reward)\"'",
                "same_episode": True,
                "highlight": [(r'"reward"', "bad", 0.6), (r'"policy"', "cmd")],
                "expect": [r'"reward": -', r'"policy": "random'],
            },
        ],
        "hold": 7,
    },
    {
        "caption": "Четыре базовые политики на dev-эпизодах",
        "sub": "flag_all и random — атаки, reconcile — сверка по правилам, oracle — эталон",
        "steps": [
            {
                "sh": "uv run python -m auditgym baselines \\\n"
                "  --config configs/dev.v1.json --seeds 100 | tail -n 4",
                "line_delay": 0.6,
                "highlight": [
                    (r"^\s*(flag_all|random)", "bad"),
                    (r"^\s*reconcile", "good"),
                    (r"^\s*oracle", "dim"),
                ],
                "expect": [r"flag_all\s+reward -", r"random\w*\s+reward -", r"reconcile\s+reward"],
            },
        ],
        "font": 36,
        "hold": 9,
    },
    {
        "caption": "Воспроизводимость: тот же seed — тот же эпизод",
        "sub": "Повтор даёт тот же отпечаток и побайтово те же файлы; соседний seed — для сравнения",
        "steps": [
            {
                "sh": "uv run python -m auditgym generate --seed 1000 \\\n"
                "  --config configs/train.v1.json --out runs/video/ep_again",
                "same_episode": True,
                "highlight": [(r"fingerprint", "good")],
                "expect": [r"fingerprint [0-9a-f]+"],
            },
            {
                "sh": "diff -r runs/video/ep runs/video/ep_again && echo identical",
                "highlight": [(r"^identical$", "good", 0.8)],
                "expect": [r"^identical$"],
            },
            {
                "sh": "uv run python -m auditgym generate --seed 1001 \\\n"
                "  --config configs/train.v1.json --out runs/video/ep_1001",
                "expect": [r"fingerprint [0-9a-f]+"],
            },
        ],
        "hold": 6,
    },
    {
        "caption": "Проверки жюри через judge_adapter.py",
        "sub": "Опубликованные определения: контракт, 20 закрытых эпизодов, matcher и награда, пары",
        "steps": [
            {
                "sh": "uv run python -m auditgym judge-check",
                "line_delay": 0.7,
                "highlight": [
                    (r"\b(\d+)/\1\b", "good"),
                    (r"FAIL", "bad"),
                ],
                "expect": [
                    r"Block 1 contract C01-C20\s+\d+/20",
                    r"Block 2 closed episodes D01-D20\s+\d+/20",
                    r"Block 3 matcher and reward T01-T20\s+\d+/20",
                    r"Block 4 pairs P01-P20\s+\d+/20",
                ],
            },
        ],
        "font": 34,
        "hold": 8,
    },
    {
        "caption": "Проверка одной командой",
        "sub": "Контейнер без сети, 2 CPU, 4 ГБ · сохранённый прогон reports/docker_check.txt",
        "steps": [
            {
                "sh": "grep -vE '^\\.|probe|Score|reward' reports/docker_check.txt",
                "line_delay": 0.3,
                "highlight": [
                    (r"RESULT: PASS", "good", 1.0),
                    (r"exit code 0|\d+ passed|\bok \d+/\d+|K\d: \d+/\d+ passed", "good"),
                    (r"^==|^# make", "cmd"),
                    (r"^#", "dim"),
                ],
                "expect": [r"RESULT: PASS", r"--network none"],
            },
        ],
        "hold": 9,
    },
    {
        "kind": "card",
        "title": "Что в репозитории",
        "table": [
            ("src/auditgym/", "генератор, среда, оценщик, независимый решатель"),
            ("configs/", "версионированные конфигурации и открытые seed"),
            ("episodes/", "открытые train- и dev-эпизоды, эталон в private/"),
            ("trajectories/", "траектории базовых политик"),
            ("reports/", "сохранённые прогоны — источник всех чисел"),
        ],
        "command": "make check",
        "command_note": "тесты · прогон генератора · имитатор жюри — одной командой",
        "footer": "Genesis Hackathon 2026 · трек FinTech · все данные вымышлены",
        "hold": 11,
    },
]

# ---------------------------------------------------------------- look and timing

W, H, FPS = 1920, 1080, 30
BG = "#0b0e14"
PANEL = "#11161e"
PANEL_BORDER = "#2b3442"
TITLEBAR = "#19202a"
CAPTION_COLOR = "#f0f6fc"
SUB_COLOR = "#9aa7b4"
DIM = "#7d8590"
ACCENT = "#7ee787"
CURSOR = "#c9d1d9"

FONT_FILES = {
    "mono": "NotoSansMono-Regular.ttf",
    "mono_bold": "NotoSansMono-Bold.ttf",
    "sans": "NotoSans-Regular.ttf",
    "sans_bold": "NotoSans-Bold.ttf",
}
STYLES = {  # style -> (color, font key)
    "prompt": (ACCENT, "mono_bold"),
    "cmd": ("#f0f6fc", "mono_bold"),
    "out": ("#c9d1d9", "mono"),
    "dim": ("#8b949e", "mono"),
    "good": ("#7ee787", "mono_bold"),
    "warn": ("#f2cc60", "mono_bold"),
    "bad": ("#ff7b72", "mono_bold"),
    "private": ("#ffa657", "mono_bold"),
}

# terminal panel geometry
PANEL_BOX = (40, 176, 1880, 1048)
TITLEBAR_H = 44
TEXT_PAD_X, TEXT_PAD_Y = 30, 14
MONO_ADVANCE = 0.6  # Noto Sans Mono advance width, em
FONT_MIN, FONT_MAX = 26, 32
FONT_FIT_MIN = 28  # shrink down to this to avoid scrolling; below it, scroll instead
WRAP_AT = 104  # longer lines wrap and do not shrink the font
LINE_SPACING = 1.3

INTRO = 1.4
TYPE_PER_CHAR, TYPE_MIN, TYPE_MAX = 0.03, 0.7, 1.8
ENTER = 0.4
LINE_DELAY = 0.25
AFTER_STEP = 1.1
BLINK = 0.5
FADE_IN, FADE_OUT = 0.6, 1.0


# ---------------------------------------------------------------- running commands


def run_step(sh: str) -> str:
    """Run one shown command for real and return its stdout."""
    proc = subprocess.run(["bash", "-c", sh], cwd=REPO, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        sys.exit(f"command failed ({proc.returncode}): {sh}\n{proc.stderr[-2000:]}")
    return proc.stdout


def run_all_commands() -> dict[str, str]:
    for rel in CLEAN_BEFORE_RUN:
        path = (REPO / rel).resolve()
        if VIDEO_DIR.resolve() not in path.parents:
            sys.exit(f"refusing to delete outside runs/video: {path}")
        shutil.rmtree(path, ignore_errors=True)
    # warm up uv so its environment sync does not happen inside a shown command
    subprocess.run(
        ["uv", "run", "python", "-m", "auditgym", "version"],
        cwd=REPO,
        capture_output=True,
        check=True,
    )
    outputs: dict[str, str] = {}
    episodes: set[str | None] = set()
    for scene in SCENES:
        for step in scene.get("steps", []):
            print(f"  $ {step['sh'].splitlines()[0]}", flush=True)
            out = run_step(step["sh"])
            for pattern in step.get("expect", []):
                if not re.search(pattern, out, re.MULTILINE):
                    sys.exit(f"output of `{step['sh']}` does not match {pattern!r}:\n{out}")
            outputs[step["sh"]] = out
            if step.get("same_episode"):
                found = EPISODE_ID.search(out)
                episodes.add(found.group(0) if found else None)
    if len(episodes) > 1:
        sys.exit(f"captions say these steps share one episode, outputs differ: {episodes}")
    for scene in SCENES:
        for path, _ in scene.get("table", []):
            if not (REPO / path).is_dir():
                sys.exit(f"closing card names a missing directory: {path}")
    return outputs


# ---------------------------------------------------------------- text model

Segment = tuple[str, str]  # (text, style)
Line = tuple[Segment, ...]


@dataclass(frozen=True)
class Frame:
    scene: int
    lines: tuple[Line, ...] = ()
    cursor: bool = False


@dataclass
class TermLayout:
    size: int
    cols: int
    rows: int
    line_h: int
    left: int
    top: int

    @property
    def adv(self) -> float:
        return self.size * MONO_ADVANCE


def squeeze(line: str, spec: tuple[str, int] | None) -> str:
    if not spec:
        return line
    pattern, width = spec
    m = re.match(pattern, line)
    if not m:
        return line
    head, mid, tail = m.groups()
    mid = mid.rstrip()
    mid = mid.ljust(width) if len(mid) <= width else mid[: width - 1] + "…"
    return f"{head}{mid} {tail}"


def style_for(line: str, rules: list) -> tuple[str, float]:
    for rule in rules:
        if re.search(rule[0], line):
            return rule[1], (rule[2] if len(rule) > 2 else 0.0)
    return "out", 0.0


def wrap(text: str, cols: int) -> list[str]:
    if len(text) <= cols:
        return [text]
    indent = len(text) - len(text.lstrip(" ")) + 2
    parts, rest, width = [], text, cols
    while len(rest) > width:
        cut = rest.rfind(" ", 0, width + 1)
        if cut > 0 and rest[cut + 1 : cut + 2] != "-":
            prev = rest.rfind(" ", 0, cut)  # keep "--option value" together
            if rest[prev + 1 : prev + 3] == "--" and prev > width // 2:
                cut = prev
        if cut <= width // 2:
            cut = width
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip(" ")
        rest = " " * indent + rest
        width = cols
    parts.append(rest)
    return parts


def command_lines(sh: str) -> list[str]:
    """Shell text as shown: first line after '$ ', continuations after '> '."""
    return sh.split("\n")


def scene_text(scene: dict, outputs: dict[str, str]) -> list[str]:
    """All lines the scene will show at its end (commands with prompts, outputs, prompt)."""
    lines = []
    for step in scene["steps"]:
        lines += ["$ " + x for x in command_lines(step["sh"])]
        lines += [squeeze(x, step.get("squeeze")) for x in output_lines(outputs[step["sh"]])]
    return lines + ["$ "]


def make_layout(size: int) -> TermLayout:
    x0, y0, x1, y1 = PANEL_BOX
    avail_w = x1 - x0 - 2 * TEXT_PAD_X
    avail_h = y1 - y0 - TITLEBAR_H - 2 * TEXT_PAD_Y
    line_h = round(size * LINE_SPACING)
    return TermLayout(
        size=size,
        cols=int(avail_w / (size * MONO_ADVANCE)),
        rows=avail_h // line_h,
        line_h=line_h,
        left=x0 + TEXT_PAD_X,
        top=y0 + TITLEBAR_H + TEXT_PAD_Y,
    )


def scene_layout(scene: dict, outputs: dict[str, str]) -> TermLayout:
    """Largest font that fits the lines up to WRAP_AT columns; smaller if that avoids scrolling."""
    if scene.get("font"):
        return make_layout(scene["font"])
    text = scene_text(scene, outputs)
    fitting = [len(x) for x in text if len(x) <= WRAP_AT] or [WRAP_AT]
    avail_w = PANEL_BOX[2] - PANEL_BOX[0] - 2 * TEXT_PAD_X
    size = max(FONT_MIN, min(FONT_MAX, int(avail_w / (MONO_ADVANCE * max(fitting)))))
    for smaller in range(size, FONT_FIT_MIN - 1, -1):
        lay = make_layout(smaller)
        if sum(len(wrap(x, lay.cols)) for x in text) <= lay.rows:
            return lay
    return make_layout(size)


def output_lines(out: str) -> list[str]:
    return out.rstrip("\n").split("\n")


# ---------------------------------------------------------------- timelines


def frames_count(seconds: float) -> int:
    return max(1, round(seconds * FPS))


@dataclass
class Timeline:
    items: list[tuple[Frame, int]] = field(default_factory=list)  # (frame, n video frames)

    def add(self, frame: Frame, seconds: float) -> None:
        self.items.append((frame, frames_count(seconds)))

    def blink(self, frame: Frame, seconds: float) -> None:
        """Hold a frame whose last line ends in a cursor, blinking it."""
        left, on = seconds, True
        while left > 1e-6:
            d = min(BLINK, left)
            self.add(Frame(frame.scene, frame.lines, cursor=on), d)
            left -= d
            on = not on

    @property
    def seconds(self) -> float:
        return sum(n for _, n in self.items) / FPS


def terminal_timeline(
    idx: int, scene: dict, outputs: dict[str, str], lay: TermLayout, tl: Timeline
):
    lines: list[Line] = []
    prompt: Line = (("$ ", "prompt"),)

    def snap(extra: list[Line] = (), cursor: bool = False) -> Frame:
        return Frame(idx, tuple(lines) + tuple(extra), cursor)

    tl.blink(snap([prompt]), INTRO)
    for step in scene["steps"]:
        shown = command_lines(step["sh"])
        total = sum(len(x) for x in shown)
        type_s = min(TYPE_MAX, max(TYPE_MIN, total * TYPE_PER_CHAR))
        n_chunks = max(4, min(14, total // 5))
        for k in range(1, n_chunks + 1):
            budget = math.ceil(total * k / n_chunks)
            partial: list[Line] = []
            for j, text in enumerate(shown):
                if budget <= 0 and j > 0:
                    break
                piece = text[: max(0, budget)]
                budget -= len(text)
                lead = "$ " if j == 0 else "> "
                for w in wrap(piece, lay.cols - 2) or [""]:
                    partial.append(((lead, "prompt"), (w, "cmd")))
                    lead = "  "
            tl.add(snap(partial, cursor=True), type_s / n_chunks)
        tl.add(snap(partial, cursor=True), ENTER)
        lines.extend(partial)
        delay = step.get("line_delay", LINE_DELAY)
        for raw in output_lines(outputs[step["sh"]]):
            text = squeeze(raw, step.get("squeeze"))
            style, pause = style_for(text, step.get("highlight", []))
            for w in wrap(text, lay.cols):
                lines.append(((w, style),))
            tl.add(snap(), delay + pause)
        tl.blink(snap([prompt]), AFTER_STEP)
    tl.blink(snap([prompt]), scene["hold"])


# ---------------------------------------------------------------- rendering


def find_fonts() -> dict[str, str]:
    found: dict[str, str] = {}
    roots = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".local/share/fonts")]
    wanted = {v: k for k, v in FONT_FILES.items()}
    for root in roots:
        for dirpath, _, files in sorted(os.walk(root)):
            for name in sorted(files):
                if name in wanted and wanted[name] not in found:
                    found[wanted[name]] = os.path.join(dirpath, name)
    missing = sorted(set(FONT_FILES) - set(found))
    if missing:
        sys.exit(f"fonts not found: {[FONT_FILES[m] for m in missing]}")
    return found


def esc(text: str) -> str:
    """Escape text for ImageMagick -annotate / label: (backslash, percent, leading @)."""
    text = text.replace("\\", "\\\\").replace("%", "%%")
    return "\\" + text if text.startswith("@") else text


class Renderer:
    def __init__(self, work: Path, fonts: dict[str, str]):
        self.work = work
        self.fonts = fonts
        self._width_cache: dict[tuple[str, str, int], int] = {}

    def text_width(self, text: str, font: str, size: int) -> int:
        key = (text, font, size)
        if key not in self._width_cache:
            out = subprocess.run(
                [
                    "magick",
                    "-font",
                    self.fonts[font],
                    "-pointsize",
                    str(size),
                    f"label:{esc(text)}",
                    "-format",
                    "%w",
                    "info:",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self._width_cache[key] = int(out.strip())
        return self._width_cache[key]

    def fit_size(self, text: str, font: str, size: int, max_w: int) -> int:
        while size > 12 and self.text_width(text, font, size) > max_w:
            size -= 1
        return size

    def annotate(
        self, text: str, font: str, size: int, color: str, x: float, y: float
    ) -> list[str]:
        return [
            "-font",
            self.fonts[font],
            "-pointsize",
            str(size),
            "-fill",
            color,
            "-annotate",
            f"+{round(x)}+{round(y)}",
            esc(text),
        ]

    def magick(self, args: list[str], out: Path) -> None:
        env = dict(os.environ, MAGICK_THREAD_LIMIT="1")
        subprocess.run(
            ["magick", *args, "-define", "png:compression-level=1", str(out)], check=True, env=env
        )

    # -- backgrounds

    def terminal_background(self, idx: int, scene: dict, counter: str) -> Path:
        x0, y0, x1, y1 = PANEL_BOX
        tb = y0 + TITLEBAR_H
        cap_size = self.fit_size(scene["caption"], "sans_bold", 50, W - 2 * 56 - 160)
        args = [
            "-size",
            f"{W}x{H}",
            f"xc:{BG}",
            "-fill",
            PANEL,
            "-stroke",
            PANEL_BORDER,
            "-strokewidth",
            "2",
            "-draw",
            f"roundrectangle {x0},{y0} {x1},{y1} 16,16",
            "-stroke",
            "none",
            "-fill",
            TITLEBAR,
            "-draw",
            f"roundrectangle {x0 + 1},{y0 + 1} {x1 - 1},{tb} 15,15",
            "-draw",
            f"rectangle {x0 + 1},{tb - 16} {x1 - 1},{tb}",
            "-fill",
            PANEL_BORDER,
            "-draw",
            f"rectangle {x0 + 1},{tb} {x1 - 1},{tb + 1}",
        ]
        for i, color in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
            cx, cy = x0 + 30 + i * 28, y0 + TITLEBAR_H // 2
            args += ["-fill", color, "-draw", f"circle {cx},{cy} {cx + 8},{cy}"]
        args += ["-gravity", "North"]
        args += self.annotate("auditgym — bash", "sans", 22, DIM, 0, y0 + 9)
        args += ["-gravity", "NorthWest"]
        args += self.annotate(scene["caption"], "sans_bold", cap_size, CAPTION_COLOR, 56, 34)
        if scene.get("sub"):
            sub_size = self.fit_size(scene["sub"], "sans", 32, W - 2 * 56)
            args += self.annotate(scene["sub"], "sans", sub_size, SUB_COLOR, 56, 112)
        args += ["-gravity", "NorthEast"]
        args += self.annotate(counter, "mono", 26, DIM, 56, 46)
        out = self.work / f"bg_{idx:02d}.png"
        self.magick(args, out)
        return out

    def card(self, idx: int, scene: dict) -> Path:
        args = ["-size", f"{W}x{H}", f"xc:{BG}", "-gravity", "North"]
        y = 340 if scene.get("table") is None else 120
        title_size = self.fit_size(scene["title"], "sans_bold", 76, W - 200)
        args += self.annotate(scene["title"], "sans_bold", title_size, CAPTION_COLOR, 0, y)
        y += title_size + 60
        args += ["-fill", ACCENT, "-draw", f"rectangle {W // 2 - 90},{y} {W // 2 + 90},{y + 5}"]
        y += 60
        for text in scene.get("lines", []):
            size = self.fit_size(text, "sans", 40, W - 240)
            args += self.annotate(text, "sans", size, "#c9d1d9", 0, y)
            y += round(size * 1.6)
        if scene.get("table"):
            names = max(len(n) for n, _ in scene["table"])
            size, gap = 36, 40
            adv = size * MONO_ADVANCE
            name_w = names * adv
            desc_w = max(self.text_width(d, "sans", size) for _, d in scene["table"])
            x = (W - (name_w + gap + desc_w)) / 2
            args += ["-gravity", "NorthWest"]
            for name, desc in scene["table"]:
                args += self.annotate(name, "mono_bold", size, ACCENT, x, y)
                args += self.annotate(desc, "sans", size, "#c9d1d9", x + name_w + gap, y - 2)
                y += round(size * 1.7)
            args += ["-gravity", "North"]
        if scene.get("command"):
            y += 50
            size = 64
            box_w = len("$ " + scene["command"]) * size * MONO_ADVANCE + 120
            bx0, bx1 = (W - box_w) / 2, (W + box_w) / 2
            args += [
                "-fill",
                PANEL,
                "-stroke",
                PANEL_BORDER,
                "-strokewidth",
                "2",
                "-draw",
                f"roundrectangle {bx0:.0f},{y} {bx1:.0f},{y + 120} 16,16",
                "-stroke",
                "none",
                "-gravity",
                "NorthWest",
            ]
            args += self.annotate("$", "mono_bold", size, ACCENT, bx0 + 60, y + 22)
            args += self.annotate(
                scene["command"],
                "mono_bold",
                size,
                "#f0f6fc",
                bx0 + 60 + 2 * size * MONO_ADVANCE,
                y + 22,
            )
            args += ["-gravity", "North"]
            y += 150
            if scene.get("command_note"):
                args += self.annotate(scene["command_note"], "sans", 34, SUB_COLOR, 0, y)
        if scene.get("footer"):
            args += self.annotate(scene["footer"], "sans", 28, DIM, 0, H - 90)
        out = self.work / f"card_{idx:02d}.png"
        self.magick(args, out)
        return out

    # -- terminal frames

    def terminal_frame(self, frame: Frame, bg: Path, lay: TermLayout, out: Path) -> None:
        args = [str(bg), "-gravity", "NorthWest"]
        visible = frame.lines[-lay.rows :]
        for row, line in enumerate(visible):
            y = lay.top + row * lay.line_h
            col = 0
            for text, style in line:
                body = text.lstrip(" ")
                if body.strip():
                    color, font = STYLES[style]
                    x = lay.left + (col + len(text) - len(body)) * lay.adv
                    args += self.annotate(body.rstrip(), font, lay.size, color, x, y)
                col += len(text)
            if frame.cursor and row == len(visible) - 1:
                cx = lay.left + col * lay.adv
                cy = y + round(lay.size * 0.12)
                args += [
                    "-fill",
                    CURSOR,
                    "-draw",
                    f"rectangle {cx:.0f},{cy} {cx + lay.adv - 2:.0f},{cy + round(lay.size * 1.15)}",
                ]
        self.magick(args, out)


# ---------------------------------------------------------------- main


def build(out_path: Path, keep_frames: bool) -> None:
    for tool in ("ffmpeg", "magick", "uv", "bash"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH")
    print("running commands", flush=True)
    outputs = run_all_commands()

    work = VIDEO_DIR / "frames"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    renderer = Renderer(work, find_fonts())

    print("rendering frames", flush=True)
    tl = Timeline()
    static: dict[int, Path] = {}  # card scenes
    layouts: dict[int, TermLayout] = {}
    backgrounds: dict[int, Path] = {}
    n_scenes = len(SCENES)
    for idx, scene in enumerate(SCENES):
        start = tl.seconds
        if scene.get("kind") == "card":
            static[idx] = renderer.card(idx, scene)
            tl.add(Frame(idx), scene["hold"])
        else:
            lay = scene_layout(scene, outputs)
            layouts[idx] = lay
            backgrounds[idx] = renderer.terminal_background(idx, scene, f"{idx + 1} / {n_scenes}")
            terminal_timeline(idx, scene, outputs, lay, tl)
        title = scene.get("caption") or scene.get("title")
        print(f"  scene {idx + 1}: {start:6.1f} - {tl.seconds:6.1f} s  {title}", flush=True)

    files: dict[Frame, Path] = {}
    jobs = []
    for frame, _ in tl.items:
        if frame in files:
            continue
        if frame.scene in static:
            files[frame] = static[frame.scene]
            continue
        digest = hashlib.sha256(repr(frame).encode()).hexdigest()[:16]
        path = work / f"f_{frame.scene:02d}_{digest}.png"
        files[frame] = path
        jobs.append((frame, path))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        list(
            pool.map(
                lambda j: renderer.terminal_frame(
                    j[0], backgrounds[j[0].scene], layouts[j[0].scene], j[1]
                ),
                jobs,
            )
        )
    print(f"  {len(jobs)} terminal frames, {tl.seconds:.1f} s of video", flush=True)

    concat = work / "list.ffconcat"
    entries = ["ffconcat version 1.0"]
    for frame, n in tl.items:
        entries += [f"file '{files[frame].name}'", f"duration {n / FPS:.6f}"]
    entries.append(f"file '{files[tl.items[-1][0]].name}'")
    concat.write_text("\n".join(entries) + "\n", encoding="utf-8")

    total = tl.seconds
    vf = (
        f"fps={FPS},fade=t=in:st=0:d={FADE_IN},"
        f"fade=t=out:st={total - FADE_OUT:.3f}:d={FADE_OUT},format=yuv420p"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("encoding", flush=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-vf",
            vf,
            "-r",
            str(FPS),
            "-t",
            f"{total:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "14",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(out_path),
        ],
        check=True,
    )
    if not keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {out_path.relative_to(REPO)} ({total:.1f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(VIDEO_DIR / "auditgym_demo.mp4"))
    parser.add_argument(
        "--keep-frames", action="store_true", help="keep the rendered PNGs in runs/video/frames"
    )
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if VIDEO_DIR.resolve() not in out.parents:
        sys.exit("the video must be written under runs/video/")
    build(out, args.keep_frames)


if __name__ == "__main__":
    main()
