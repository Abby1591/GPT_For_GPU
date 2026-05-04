"""
tool_definitions.py
===================
The tools the model can call at inference time, plus the training-data
generators that teach it when and how to call them.

HOW TO ADD A NEW TOOL
---------------------
1. Write an executor:  str -> str, never raises, returns "error:..." on failure.

2. Write template bank: list of (context, query_expr, continuation) triples.
   The continuation MUST contain [TOOL:name|...][RESULT:{result}].
   Include prose BEFORE and AFTER the call — the "after" is what teaches the
   model to actually use the result (key Toolformer finding).

3. Write a generator: (n: int) -> List[str].  Calls the executor, skips
   results that start with "error".

4. Add to TOOL_REGISTRY at the bottom.  default_weight is the fraction of
   the Phase 8 budget; all weights should sum to ~1.0.

5. In build_dataset.py Phase 8 nothing changes — it reads TOOL_REGISTRY.

6. At inference wire tools into NeuralNetwork:
       from tool_definitions import TOOL_REGISTRY
       from Neural_Network import ensure_tool_vocab
       char2idx = ensure_tool_vocab(char2idx)   # adds [ ] : | if missing
       for name, tdef in TOOL_REGISTRY.items():
           nn.register_tool(name, tdef.executor)
       out_ids, log = nn.generate_with_tools(prompt_ids, idx2char, char2idx)

HOW THE MODEL LEARNS TO USE TOOLS
----------------------------------
No architectural changes needed.  The model is character-level, so it learns
purely by seeing the [TOOL:name|arg][RESULT:...] format in training data.

  generate_with_tools() in Neural_Network.py watches the output stream for
  [TOOL:...] patterns, calls the registered handler, injects [RESULT:...]
  back into context, and continues generating.

  ensure_tool_vocab() (also in Neural_Network.py) makes sure [ ] : | are in
  the vocab — if they're missing the model can never generate the format.

The only connection between training and inference is the format string.
"""

from __future__ import annotations

import calendar
import datetime
import math
import random
import re
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Reuse rate-limited HTTP helper from build_dataset when available.
try:
    from build_dataset import _get_json as _fetch_json
except ImportError:
    def _fetch_json(url: str) -> Optional[dict]:  # type: ignore[misc]
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "miniGPT-tools/1.0 (educational)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                import json
                return json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception:
            return None

TOOL_MAX_RESULT: int = 200   # max chars kept from any single result


@dataclass
class ToolDef:
    name:           str
    executor:       Callable[[str], str]        # runs at inference time
    generator:      Callable[[int], List[str]]  # builds training examples
    default_weight: float                        # fraction of Phase 8 budget
    description:    str
    example:        str                          # one sample [TOOL:...][RESULT:...]


# =============================================================================
#  Executors
# =============================================================================

def _tool_exec_calc(expr: str) -> str:
    # Whitelist before eval — only digits, operators, parens, named math fns.
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
    # Argument: "<value> <src> to <dst>", e.g. "5 km to miles"
    m = re.match(r"([\d\.]+)\s*(\w+)\s+(?:to|in)\s+(\w+)", expr.lower().strip())
    if not m:
        return "unknown conversion"
    val, src, dst = float(m.group(1)), m.group(2), m.group(3)
    table: Dict[tuple, Callable] = {
        ("km","miles"): lambda x: x*0.621371,  ("miles","km"): lambda x: x*1.60934,
        ("m","ft"):     lambda x: x*3.28084,   ("ft","m"):     lambda x: x/3.28084,
        ("cm","inches"):lambda x: x*0.393701,  ("inches","cm"):lambda x: x/0.393701,
        ("kg","lbs"):   lambda x: x*2.20462,   ("lbs","kg"):   lambda x: x/2.20462,
        ("g","oz"):     lambda x: x*0.035274,
        ("f","c"):      lambda x: (x-32)*5/9,  ("c","f"):      lambda x: x*9/5+32,
        ("k","c"):      lambda x: x-273.15,    ("c","k"):      lambda x: x+273.15,
        ("l","gal"):    lambda x: x*0.264172,  ("gal","l"):    lambda x: x/0.264172,
        ("mb","kb"):    lambda x: x*1024,       ("gb","mb"):    lambda x: x*1024,
        ("tb","gb"):    lambda x: x*1024,
    }
    fn = table.get((src, dst))
    if fn is None:
        return f"unknown: {src} to {dst}"
    r = fn(val)
    return f"{int(round(r))} {dst}" if abs(r-round(r)) < 1e-9 else f"{round(r,4)} {dst}"


def _tool_exec_date(expr: str) -> str:
    # Date questions via stdlib only — no network.
    el = expr.lower()
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
            months = {"january":1,"february":2,"march":3,"april":4,"may":5,
                      "june":6,"july":7,"august":8,"september":9,
                      "october":10,"november":11,"december":12}
            mon = months.get(m.group(1))
            if mon:
                return str(calendar.monthrange(int(m.group(2)), mon)[1])
    m = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{4}-\d{2}-\d{2})", expr)
    if m:
        try:
            d1, d2 = (datetime.date.fromisoformat(m.group(1)),
                      datetime.date.fromisoformat(m.group(2)))
            return f"{abs((d2-d1).days)} days"
        except Exception:
            pass
    return "unknown date query"


def _tool_exec_search(query: str) -> str:
    # First sentence of the matching Wikipedia article.
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
    # Entity lookup — same backend as search, different training context.
    return _tool_exec_search(entity)


# =============================================================================
#  Template banks
#  Each entry: (context_template, query_template, continuation_template)
#  continuation MUST contain the [TOOL:name|...][RESULT:{result}] call.
# =============================================================================

_CALC_TEMPLATES = [
    ("The total cost was {a} items at {b} dollars each.",
     "{a}*{b}",
     "The calculation confirms the total is [TOOL:calc|{a}*{b}][RESULT:{result}] dollars."),
    ("A rectangle measures {a} metres by {b} metres.",
     "{a}*{b}",
     "Its area is [TOOL:calc|{a}*{b}][RESULT:{result}] square metres."),
    ("If you divide {a} equally among {b} people,",
     "{a}/{b}",
     "each person receives [TOOL:calc|{a}/{b}][RESULT:{result}]."),
    ("The sum of {a} and {b} is",
     "{a}+{b}",
     "[TOOL:calc|{a}+{b}][RESULT:{result}]."),
    ("To find the square root of {a},",
     "sqrt({a})",
     "we compute [TOOL:calc|sqrt({a})][RESULT:{result}]."),
    ("Raising {a} to the power of {b} gives",
     "{a}**{b}",
     "[TOOL:calc|{a}**{b}][RESULT:{result}]."),
    ("{a} percent of {b} is",
     "{a}/100*{b}",
     "[TOOL:calc|{a}/100*{b}][RESULT:{result}]."),
    ("She owed {a} dollars and paid back {b}.",
     "{a}-{b}",
     "The remaining balance was [TOOL:calc|{a}-{b}][RESULT:{result}] dollars."),
    ("At {b} miles per hour, covering {a} miles takes",
     "{a}/{b}",
     "[TOOL:calc|{a}/{b}][RESULT:{result}] hours."),
]

_CONVERT_TEMPLATES = [
    ("The distance between the two cities is {a} kilometres.",
     "{a} km to miles",
     "That is [TOOL:convert|{a} km to miles][RESULT:{result}]."),
    ("The package weighs {a} kilograms.",
     "{a} kg to lbs",
     "In pounds, that is [TOOL:convert|{a} kg to lbs][RESULT:{result}]."),
    ("The temperature outside was {a} degrees Celsius.",
     "{a} C to F",
     "In Fahrenheit that is [TOOL:convert|{a} C to F][RESULT:{result}]."),
    ("The recipe calls for {a} litres of water.",
     "{a} l to gal",
     "That is approximately [TOOL:convert|{a} l to gal][RESULT:{result}]."),
    ("The shelf is {a} feet long.",
     "{a} ft to m",
     "In metric units that is [TOOL:convert|{a} ft to m][RESULT:{result}] metres."),
    ("The download is {a} gigabytes.",
     "{a} gb to mb",
     "That equals [TOOL:convert|{a} gb to mb][RESULT:{result}] megabytes."),
    ("The athlete ran {a} miles.",
     "{a} miles to km",
     "That is [TOOL:convert|{a} miles to km][RESULT:{result}] kilometres."),
    ("The oven was set to {a} degrees Fahrenheit.",
     "{a} F to C",
     "That is [TOOL:convert|{a} F to C][RESULT:{result}] degrees Celsius."),
]

_DATE_TEMPLATES = [
    ("To determine the current date,",
     "what is today's date",
     "we check: [TOOL:date|what is today's date][RESULT:{result}]."),
    ("The event is scheduled for this year.",
     "what is the current year",
     "Specifically, [TOOL:date|what is the current year][RESULT:{result}]."),
    ("She asked what day of the week it was.",
     "day of the week today",
     "The answer was [TOOL:date|day of the week today][RESULT:{result}]."),
    ("February {y} had a specific number of days.",
     "days in February {y}",
     "There were [TOOL:date|days in February {y}][RESULT:{result}] days in February {y}."),
    ("The number of days between {d1} and {d2} matters for the calculation.",
     "{d1} to {d2}",
     "There are [TOOL:date|{d1} to {d2}][RESULT:{result}] days between those dates."),
    ("The project started on {d1} and ended on {d2}.",
     "{d1} to {d2}",
     "It lasted [TOOL:date|{d1} to {d2}][RESULT:{result}]."),
    ("She needed to know how many days were in March {y}.",
     "days in March {y}",
     "[TOOL:date|days in March {y}][RESULT:{result}] days made up that month."),
]

_SEARCH_TEMPLATES = [
    ("The article discussed {q}.",
     "{q}",
     "[TOOL:search|{q}][RESULT:{result}] This background helped readers understand the topic."),
    ("Many people wondered about {q}.",
     "{q}",
     "According to available information, [TOOL:search|{q}][RESULT:{result}]"),
    ("To answer the question about {q},",
     "{q}",
     "one can look it up: [TOOL:search|{q}][RESULT:{result}]"),
    ("The student researched {q} for her essay.",
     "{q}",
     "She found that [TOOL:search|{q}][RESULT:{result}]"),
    ("He wanted to fact-check the claim about {q}.",
     "{q}",
     "The search confirmed: [TOOL:search|{q}][RESULT:{result}]"),
]

_LOOKUP_TEMPLATES = [
    ("The author mentioned {e} in passing.",
     "{e}",
     "To clarify: [TOOL:lookup|{e}][RESULT:{result}]"),
    ("Few people knew who {e} was.",
     "{e}",
     "[TOOL:lookup|{e}][RESULT:{result}] This context is important."),
    ("The concept of {e} was central to the argument.",
     "{e}",
     "Specifically, [TOOL:lookup|{e}][RESULT:{result}]"),
    ("The professor asked the class about {e}.",
     "{e}",
     "The definition: [TOOL:lookup|{e}][RESULT:{result}]"),
    ("The report referenced {e} without explanation.",
     "{e}",
     "Looking it up: [TOOL:lookup|{e}][RESULT:{result}]"),
]


# =============================================================================
#  Generators
# =============================================================================

def _generate_calc_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        ctx_t, expr_t, cont_t = random.choice(_CALC_TEMPLATES)
        a, b = random.randint(2, 999), random.randint(2, 99)
        if "{a}/{b}" in expr_t and random.random() < 0.5:
            a = b * random.randint(2, 20)
        expr   = expr_t.format(a=a, b=b)
        result = _tool_exec_calc(expr)
        if result.startswith("error"):
            continue
        examples.append(f"{ctx_t.format(a=a,b=b)}\n{cont_t.format(a=a,b=b,result=result)}")
    return examples


def _generate_convert_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        ctx_t, q_t, cont_t = random.choice(_CONVERT_TEMPLATES)
        a = int(round(random.uniform(1,500))) if random.random() < 0.6 \
            else round(random.uniform(1,500), 1)
        result = _tool_exec_convert(q_t.format(a=a))
        if "unknown" in result or "error" in result:
            continue
        examples.append(f"{ctx_t.format(a=a)}\n{cont_t.format(a=a,result=result)}")
    return examples


def _generate_date_examples(n: int) -> List[str]:
    examples = []
    years = list(range(1990, 2025))
    pairs = []
    for _ in range(40):
        d = datetime.date(random.randint(2000,2023), random.randint(1,12), random.randint(1,28))
        pairs.append((str(d), str(d + datetime.timedelta(days=random.randint(10,500)))))
    for _ in range(n):
        ctx_t, q_t, cont_t = random.choice(_DATE_TEMPLATES)
        if "{y}" in q_t:
            y = random.choice(years)
            result = _tool_exec_date(q_t.format(y=y))
            ctx, cont = ctx_t.format(y=y), cont_t.format(y=y, result=result)
        elif "{d1}" in q_t:
            d1, d2 = random.choice(pairs)
            result = _tool_exec_date(q_t.format(d1=d1, d2=d2))
            ctx, cont = ctx_t.format(d1=d1,d2=d2), cont_t.format(d1=d1,d2=d2,result=result)
        else:
            result = _tool_exec_date(q_t)
            ctx, cont = ctx_t, cont_t.format(result=result)
        if result == "unknown date query":
            continue
        examples.append(f"{ctx}\n{cont}")
    return examples


def _generate_search_lookup_examples(n: int, live: bool = True) -> List[str]:
    try:
        from build_dataset import (
            LGBTQ_ARTICLES, LEFT_ARTICLES, DIVERSE_ARTICLES, ACADEMIC_ARTICLES,
        )
        pool = LGBTQ_ARTICLES[:20] + LEFT_ARTICLES[:20] + \
               DIVERSE_ARTICLES[:20] + ACADEMIC_ARTICLES[:20]
    except ImportError:
        pool = ["Marxism", "Feminism", "Alan Turing", "Marie Curie",
                "Climate change", "Stonewall riots", "DNA", "Black hole"]

    topics, examples = random.sample(pool, min(n*3, len(pool))), []
    for topic in topics:
        if len(examples) >= n:
            break
        result = _tool_exec_search(topic) if live else f"{topic} is a notable subject."
        if result == "no result":
            continue
        if random.random() < 0.5:
            ctx_t, _, cont_t = random.choice(_SEARCH_TEMPLATES)
            examples.append(
                f"{ctx_t.format(q=topic)}\n"
                f"{cont_t.format(q=topic, result=result[:TOOL_MAX_RESULT])}")
        else:
            ctx_t, _, cont_t = random.choice(_LOOKUP_TEMPLATES)
            examples.append(
                f"{ctx_t.format(e=topic)}\n"
                f"{cont_t.format(e=topic, result=result[:TOOL_MAX_RESULT])}")
        print(f"    search/lookup OK: {topic[:55]}")
    return examples


# =============================================================================
#  Registry  --  single place that ties everything together
# =============================================================================

TOOL_REGISTRY: Dict[str, ToolDef] = {}

TOOL_REGISTRY["calc"] = ToolDef(
    name="calc", executor=_tool_exec_calc,
    generator=_generate_calc_examples, default_weight=0.30,
    description="Evaluate arithmetic: digits, +-*/**, sqrt, log, pi.",
    example="[TOOL:calc|sqrt(144)][RESULT:12]",
)
TOOL_REGISTRY["convert"] = ToolDef(
    name="convert", executor=_tool_exec_convert,
    generator=_generate_convert_examples, default_weight=0.20,
    description="Unit conversion: length, weight, temperature, volume, data.",
    example="[TOOL:convert|100 km to miles][RESULT:62.1371 miles]",
)
TOOL_REGISTRY["date"] = ToolDef(
    name="date", executor=_tool_exec_date,
    generator=_generate_date_examples, default_weight=0.15,
    description="Date queries: today, days between dates, days in a month.",
    example="[TOOL:date|2020-03-01 to 2020-06-15][RESULT:106 days]",
)
TOOL_REGISTRY["search"] = ToolDef(
    name="search", executor=_tool_exec_search,
    generator=lambda n: _generate_search_lookup_examples(n, live=True),
    default_weight=0.20,
    description="First sentence of a Wikipedia article.",
    example="[TOOL:search|Marie Curie][RESULT:Marie Curie was a Polish-French physicist...]",
)
TOOL_REGISTRY["lookup"] = ToolDef(
    name="lookup", executor=_tool_exec_lookup,
    generator=lambda n: _generate_search_lookup_examples(n, live=True),
    default_weight=0.15,
    description="Entity lookup (person, place, concept) via Wikipedia.",
    example="[TOOL:lookup|Stonewall riots][RESULT:The Stonewall riots were a series of...]",
)
# weights: 0.30 + 0.20 + 0.15 + 0.20 + 0.15 = 1.00


# =============================================================================
#  fetch_tool_training  --  called by build_dataset.py Phase 8
# =============================================================================

def fetch_tool_training(
    target_chars: int,
    live_search:  bool = True,
    **weights: float,    # per-tool overrides, e.g. calc=0.50; auto-renormalised
) -> List[str]:
    resolved = {n: weights.get(n, td.default_weight) for n, td in TOOL_REGISTRY.items()}
    total_w  = sum(resolved.values()) or 1.0
    for k in resolved:
        resolved[k] /= total_w

    if not live_search:
        no_live = lambda n: _generate_search_lookup_examples(n, live=False)
        for name in ("search", "lookup"):
            if name in TOOL_REGISTRY:
                td = TOOL_REGISTRY[name]
                TOOL_REGISTRY[name] = ToolDef(
                    td.name, td.executor, no_live,
                    td.default_weight, td.description, td.example,
                )

    n_total = max(50, target_chars // 220)
    print(f"\n[8/8] Tool training  (target {target_chars:,} chars, "
          f"live={'yes' if live_search else 'no'})")
    print("  Tools: " + "  ".join(f"{n}={resolved[n]:.0%}" for n in TOOL_REGISTRY))

    all_examples: List[str] = []
    for name, td in TOOL_REGISTRY.items():
        n_this = int(n_total * resolved[name])
        if n_this > 0:
            print(f"  {name} ({n_this})...")
            all_examples.extend(td.generator(n_this))

    random.shuffle(all_examples)
    chunks, total = [], 0
    for ex in all_examples:
        if total >= target_chars:
            break
        chunks.append(ex); total += len(ex)

    counts: Counter = Counter(
        m.group(1)
        for ex in chunks
        for m in re.finditer(r'\[TOOL:(\w+)\|', ex)
    )
    print(f"  Done: {len(chunks):,} examples  ({total:,} chars)")
    print(f"  Counts: " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return chunks


# Smoke test: python tool_definitions.py  [--live]
if __name__ == "__main__":
    import sys
    random.seed(42)
    live = "--live" in sys.argv
    print("Registered tools:", list(TOOL_REGISTRY.keys()))
    chunks = fetch_tool_training(3000, live_search=live)
    print(f"\nGot {len(chunks)} examples.")
    print("Sample:", chunks[0])