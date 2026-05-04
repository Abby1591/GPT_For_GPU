"""
tool_definitions.py
===================
Inference-time tool executors and the TOOL_REGISTRY that ties them together.

This file contains **only** the code needed at inference time:
  - ToolDef dataclass
  - One executor function per tool  (str -> str, never raises)
  - TOOL_REGISTRY dict

Training-data generation (templates, generators, fetch_tool_training) lives
in build_dataset.py, which imports the executors from here to keep results
consistent between training and inference.

Available tools
---------------
  calc    -- arithmetic via eval with a math whitelist
  convert -- unit conversions (length, weight, temp, volume, data)
  date    -- date queries using stdlib only, no network
  search  -- first sentence of a Wikipedia article
  lookup  -- entity lookup via Wikipedia (same backend as search)

Adding a new tool
-----------------
1. Write an executor: ``str -> str``, never raises, returns ``"error:..."``
   on failure.
2. Add it to TOOL_REGISTRY at the bottom.
3. In build_dataset.py add a template bank + generator, then wire into
   fetch_tool_training().

Programmatic usage::

    from model import MiniGPT
    from tool_definitions import TOOL_REGISTRY

    model = MiniGPT.load("gpt_weights.json")

    # Tools are on by default when the model was trained with tool data
    print(model.generate("The square root of 144 is"))

    # Opt out of specific tools
    print(model.generate("5 km equals", skip_tools={"search", "lookup"}))

    # Opt out entirely
    print(model.generate("Democracy is", tool_registry=None))

    # Call an executor directly (no model needed)
    from tool_definitions import _tool_exec_calc
    print(_tool_exec_calc("sqrt(144)"))   # "12"
"""

from __future__ import annotations

import calendar
import datetime
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Optional

TOOL_MAX_RESULT: int = 200  # max chars kept from any single tool result


# ---------------------------------------------------------------------------
#  HTTP helper (search / lookup only)
# ---------------------------------------------------------------------------

# Reuse the rate-limited fetcher from build_dataset when available so the
# search executor doesn't bypass the politeness delay during dataset generation.
try:
    from build_dataset import _get_json as _fetch_json
except ImportError:
    def _fetch_json(url: str) -> Optional[dict]:  # type: ignore[misc]
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "miniGPT-tools/1.0 (educational)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                return json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception:
            return None


# ---------------------------------------------------------------------------
#  ToolDef
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """Describes one tool the model can call at inference time."""
    name:        str
    executor:    Callable[[str], str]  # called at inference; never raises
    description: str
    example:     str                   # sample [TOOL:name|arg][RESULT:...]


# ---------------------------------------------------------------------------
#  Executors
# ---------------------------------------------------------------------------

def _tool_exec_calc(expr: str) -> str:
    """
    Evaluate a simple arithmetic expression.

    Accepts digits, ``+ - * / ** % ( ) .`` and the named functions
    ``sqrt``, ``log``, ``abs``, ``round``, plus constants ``pi`` and ``e``.

    :param expr: Expression string, e.g. ``"sqrt(144)"`` or ``"3**8"``.
    :return: String result, or ``"error: ..."`` on failure.

    .. code-block:: python

        _tool_exec_calc("sqrt(144)")  # "12"
        _tool_exec_calc("3 * 7")      # "21"
        _tool_exec_calc("2**10")      # "1024"
    """
    if not re.match(r'^[0-9\+\-\*/\(\)\.\^ \tsqrtpilogabsroun]+$', expr.lower()):
        return "error: unsafe expression"
    try:
        result = eval(
            expr, {"__builtins__": {}},
            {"sqrt": math.sqrt, "pi": math.pi, "e": math.e,
             "log": math.log, "abs": abs, "round": round},
        )
        if isinstance(result, float) and result == int(result):
            s = str(int(result))
        elif isinstance(result, float):
            s = str(round(result, 6))
        else:
            s = str(result)
        return "error: result too large" if len(s.lstrip("-")) > 15 else s
    except Exception as exc:
        return f"error: {exc}"


def _tool_exec_convert(expr: str) -> str:
    """
    Convert between common units.

    Argument format: ``"<value> <src> to <dst>"``, e.g. ``"5 km to miles"``.

    Supported pairs: km/miles, m/ft, cm/inches, kg/lbs, g/oz,
    C/F/K, l/gal, mb/kb, gb/mb, tb/gb.

    :param expr: Conversion string.
    :return: Result string, or ``"unknown: ..."`` for unsupported pairs.

    .. code-block:: python

        _tool_exec_convert("100 km to miles")  # "62.1371 miles"
        _tool_exec_convert("0 C to F")         # "32 F"
    """
    m = re.match(r"([\d\.]+)\s*(\w+)\s+(?:to|in)\s+(\w+)", expr.lower().strip())
    if not m:
        return "unknown conversion"
    val, src, dst = float(m.group(1)), m.group(2), m.group(3)
    table: Dict[tuple, Callable] = {
        ("km",    "miles"): lambda x: x * 0.621371,
        ("miles", "km"):    lambda x: x * 1.60934,
        ("m",     "ft"):    lambda x: x * 3.28084,
        ("ft",    "m"):     lambda x: x / 3.28084,
        ("cm",    "inches"):lambda x: x * 0.393701,
        ("inches","cm"):    lambda x: x / 0.393701,
        ("kg",    "lbs"):   lambda x: x * 2.20462,
        ("lbs",   "kg"):    lambda x: x / 2.20462,
        ("g",     "oz"):    lambda x: x * 0.035274,
        ("f",     "c"):     lambda x: (x - 32) * 5 / 9,
        ("c",     "f"):     lambda x: x * 9 / 5 + 32,
        ("k",     "c"):     lambda x: x - 273.15,
        ("c",     "k"):     lambda x: x + 273.15,
        ("l",     "gal"):   lambda x: x * 0.264172,
        ("gal",   "l"):     lambda x: x / 0.264172,
        ("mb",    "kb"):    lambda x: x * 1024,
        ("gb",    "mb"):    lambda x: x * 1024,
        ("tb",    "gb"):    lambda x: x * 1024,
    }
    fn = table.get((src, dst))
    if fn is None:
        return f"unknown: {src} to {dst}"
    r = fn(val)
    return f"{int(round(r))} {dst}" if abs(r - round(r)) < 1e-9 else f"{round(r, 4)} {dst}"


def _tool_exec_date(expr: str) -> str:
    """
    Answer simple date questions using only the stdlib (no network).

    Supported queries:

    - ``"today"`` / ``"current date"`` / ``"what date"`` → ``"YYYY-MM-DD"``
    - ``"current year"`` / ``"what year"`` → ``"YYYY"``
    - ``"day of the week"`` / ``"weekday"`` → e.g. ``"Monday"``
    - ``"days in <month> <year>"`` → number of days in that month
    - ``"YYYY-MM-DD to YYYY-MM-DD"`` → ``"N days"``

    :param expr: Natural-language date query.
    :return: Answer string, or ``"unknown date query"``.

    .. code-block:: python

        _tool_exec_date("today")                     # "2025-08-14"
        _tool_exec_date("days in February 2024")     # "29"
        _tool_exec_date("2020-01-01 to 2020-12-31")  # "365 days"
    """
    el  = expr.lower()
    now = datetime.datetime.now()
    if any(k in el for k in ("today", "current date", "what date")):
        return now.strftime("%Y-%m-%d")
    if "year" in el and any(k in el for k in ("current", "what")):
        return str(now.year)
    if "day of the week" in el or "weekday" in el:
        return now.strftime("%A")
    if "days in" in el:
        m = re.search(r"(\w+)\s+(\d{4})", el)
        if m:
            months = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
            }
            mon = months.get(m.group(1))
            if mon:
                return str(calendar.monthrange(int(m.group(2)), mon)[1])
    m = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})", expr)
    if m:
        try:
            d1 = datetime.date.fromisoformat(m.group(1))
            d2 = datetime.date.fromisoformat(m.group(2))
            return f"{abs((d2 - d1).days)} days"
        except Exception:
            pass
    return "unknown date query"


def _tool_exec_search(query: str) -> str:
    """
    Return the first sentence of the matching Wikipedia article.

    :param query: Article title or search phrase.
    :return: First sentence (≤ TOOL_MAX_RESULT chars), or ``"no result"``.

    .. code-block:: python

        _tool_exec_search("Marie Curie")
        # "Marie Curie was a Polish and naturalized-French physicist..."
    """
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={urllib.parse.quote(query)}"
        "&prop=extracts&exintro=1&explaintext=1&exsentences=2"
        "&format=json&redirects=1"
    )
    data = _fetch_json(url)
    if not data:
        return "no result"
    try:
        page = next(iter(data["query"]["pages"].values()))
        extract = page.get("extract", "").strip()
        if not extract or "may refer to" in extract:
            return "no result"
        return re.split(r"(?<=[.!?])\s", extract)[0][:TOOL_MAX_RESULT]
    except Exception:
        return "no result"


def _tool_exec_lookup(entity: str) -> str:
    """
    Entity lookup via Wikipedia — same backend as search, different
    training context (teaches the model to resolve proper nouns/concepts).

    :param entity: Entity name, e.g. ``"Stonewall riots"``.
    :return: First sentence of the Wikipedia article, or ``"no result"``.
    """
    return _tool_exec_search(entity)


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, ToolDef] = {
    "calc": ToolDef(
        name="calc",
        executor=_tool_exec_calc,
        description="Arithmetic: digits, +-*/**, sqrt, log, pi.",
        example="[TOOL:calc|sqrt(144)][RESULT:12]",
    ),
    "convert": ToolDef(
        name="convert",
        executor=_tool_exec_convert,
        description="Unit conversion: length, weight, temperature, volume, data.",
        example="[TOOL:convert|100 km to miles][RESULT:62.1371 miles]",
    ),
    "date": ToolDef(
        name="date",
        executor=_tool_exec_date,
        description="Date queries: today, days between dates, days in a month.",
        example="[TOOL:date|2020-03-01 to 2020-06-15][RESULT:106 days]",
    ),
    "search": ToolDef(
        name="search",
        executor=_tool_exec_search,
        description="First sentence of a Wikipedia article.",
        example="[TOOL:search|Marie Curie][RESULT:Marie Curie was a Polish-French physicist...]",
    ),
    "lookup": ToolDef(
        name="lookup",
        executor=_tool_exec_lookup,
        description="Entity lookup (person, place, concept) via Wikipedia.",
        example="[TOOL:lookup|Stonewall riots][RESULT:The Stonewall riots were a series of...]",
    ),
}

# Weights used by build_dataset.py fetch_tool_training() to split the budget.
# Must sum to 1.0.
TOOL_WEIGHTS: Dict[str, float] = {
    "calc": 0.30, "convert": 0.20, "date": 0.15, "search": 0.20, "lookup": 0.15,
}