"""
build_dataset.py
================
Builds a diverse plain-text training dataset for miniGPT from free public sources.
No API keys. No HuggingFace. Runs entirely locally.

Sources
-------
  1. Project Gutenberg  -- classic books, public domain
  2. Wikipedia          -- factual articles, curated for diversity + random fill
  3. Simple Wikipedia   -- plain English articles, great for training
  4. Wikiquote          -- short quotes from diverse voices

Usage
-----
    python build_dataset.py
    python build_dataset.py --target_chars 3000000 --output my_data.txt
    python build_dataset.py --no_gutenberg   # skip Gutenberg (slow connection)

Output
------
    diverse_dataset.txt  -- ready to pass directly to miniGPT --train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import List, Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# =============================================================================
#  Article lists
# =============================================================================

# Gutenberg book IDs. Shuffled at runtime so each run picks a different subset
# when the total target is smaller than the full list.
# Format: (gutenberg_id, description)
GUTENBERG_BOOKS = [
    # Fiction
    (1342,  "Pride and Prejudice"),
    (11,    "Alice in Wonderland"),
    (84,    "Frankenstein"),
    (98,    "A Tale of Two Cities"),
    (1661,  "Sherlock Holmes"),
    (2701,  "Moby Dick"),
    (76,    "Huckleberry Finn"),
    (174,   "The Picture of Dorian Gray"),
    (345,   "Dracula"),
    (5200,  "Metamorphosis"),
    (2542,  "A Doll's House"),
    (1260,  "Jane Eyre"),
    (514,   "Little Women"),
    (203,   "Uncle Tom's Cabin"),
    (25344, "The Jungle"),
    (1952,  "The Yellow Wallpaper"),
    (768,   "Wuthering Heights"),
    (1184,  "The Count of Monte Cristo"),
    # Non-fiction / philosophy
    (7370,  "Common Sense - Paine"),
    (61,    "The Communist Manifesto"),
    (4705,  "Meditations - Marcus Aurelius"),
    (2130,  "The Republic - Plato"),
    (1497,  "The Symposium - Plato"),
    (17192, "The Art of War"),
    (1998,  "Thus Spoke Zarathustra"),
    (1228,  "On the Origin of Species"),
    # Black American voices
    (2852,  "My Bondage and My Freedom - Douglass"),
    (23,    "Narrative of Frederick Douglass"),
    (33,    "The Souls of Black Folk - Du Bois"),
    # World literature
    (8492,  "Gitanjali - Tagore"),
    (7333,  "Tao Te Ching"),
    (2413,  "Arabian Nights vol 1"),
    (6130,  "The Iliad"),
    (1727,  "The Odyssey"),
]

# LGBTQ+ Wikipedia articles -- fetched FIRST, no cap, always in the dataset.
LGBTQ_ARTICLES = [
    "Stonewall riots",
    "Harvey Milk",
    "Marsha P. Johnson",
    "Sylvia Rivera",
    "Bayard Rustin",
    "Alan Turing",
    "Oscar Wilde",
    "James Baldwin",
    "Audre Lorde",
    "Laverne Cox",
    "Christine Jorgensen",
    "Matthew Shepard",
    "History of homosexuality",
    "Transgender history",
    "LGBT rights by country or territory",
    "LGBT rights in the United States",
    "Same-sex marriage",
    "Transgender rights in the United States",
    "Legal recognition of non-binary gender",
    "Decriminalization of homosexuality",
    "Pride parade",
    "Don't ask, don't tell",
    "ACT UP",
    "HIV/AIDS",
    "Human Rights Campaign",
    "PFLAG",
    "Lambda Legal",
    "Giovanni's Room",
    "The Well of Loneliness",
    "Angels in America",
    "The Normal Heart",
]

# Diverse Wikipedia articles -- fetched after LGBTQ+, up to 60% of wiki target.
# No LGBTQ+ duplicates here since those are always guaranteed above.
DIVERSE_ARTICLES = [
    # Civil rights / racial equality
    "Martin Luther King Jr.",
    "Rosa Parks",
    "Malcolm X",
    "Harriet Tubman",
    "Frederick Douglass",
    "Nelson Mandela",
    "Apartheid",
    "Black Lives Matter",
    "Civil Rights Act of 1964",
    "Voting Rights Act of 1965",
    "Brown v. Board of Education",
    "Montgomery bus boycott",
    "March on Washington",
    "Selma to Montgomery marches",
    "Emmett Till",
    "John Lewis",
    "Fannie Lou Hamer",
    "Thurgood Marshall",
    "Shirley Chisholm",
    "NAACP",
    "Black Panther Party",
    "Tulsa race massacre",
    "Juneteenth",
    "Jim Crow laws",
    "Slavery in the United States",
    "Underground Railroad",
    "Reconstruction era",
    "Redlining",
    "Loving v. Virginia",
    # Women's rights
    "Feminism",
    "Women's suffrage",
    "Simone de Beauvoir",
    "Mary Wollstonecraft",
    "bell hooks",
    "Gloria Steinem",
    "Betty Friedan",
    "Sojourner Truth",
    "Susan B. Anthony",
    "Malala Yousafzai",
    "Ruth Bader Ginsburg",
    "Seneca Falls Convention",
    "Me Too movement",
    "Gender pay gap",
    "Reproductive rights",
    "Roe v. Wade",
    "Women's liberation movement",
    # Science (diverse)
    "Marie Curie",
    "Katherine Johnson",
    "Mae Jemison",
    "Rosalind Franklin",
    "Chien-Shiung Wu",
    "Ada Lovelace",
    "Grace Hopper",
    "Charles Darwin",
    "Albert Einstein",
    "Carl Sagan",
    "Jane Goodall",
    "Wangari Maathai",
    "Neil deGrasse Tyson",
    "Tu Youyou",
    # Global / decolonisation
    "Mahatma Gandhi",
    "Indian independence movement",
    "Decolonization",
    "Atlantic slave trade",
    "Colonialism",
    "Desmond Tutu",
    "Chinua Achebe",
    "Frantz Fanon",
    "Kwame Nkrumah",
    "Jawaharlal Nehru",
    "Ho Chi Minh",
    "Universal Declaration of Human Rights",
    "Amnesty International",
    # Climate / environment
    "Climate change",
    "Global warming",
    "Paris Agreement",
    "Greta Thunberg",
    "Renewable energy",
    "Environmental justice",
    "Biodiversity",
    # Disability / neurodiversity
    "Disability rights movement",
    "Neurodiversity",
    "Mental health",
    # History / general
    "World War II",
    "Holocaust",
    "Cold War",
    "French Revolution",
    "American Revolution",
    "Democracy",
    "Human rights",
    "Poverty",
    "Wealth inequality",
    "Philosophy",
]

# Wikiquote pages -- quotes from diverse voices
WIKIQUOTE_PAGES = [
    # Science
    "Albert Einstein", "Marie Curie", "Carl Sagan", "Richard Feynman",
    "Stephen Hawking", "Ada Lovelace", "Grace Hopper", "Nikola Tesla",
    "Katherine Johnson", "Mae Jemison",
    # Civil rights
    "Martin Luther King Jr.", "Malcolm X", "Rosa Parks", "Nelson Mandela",
    "Mahatma Gandhi", "Harriet Tubman", "Frederick Douglass",
    "Sojourner Truth", "Harvey Milk", "Bayard Rustin",
    # Women's rights / feminism
    "Simone de Beauvoir", "Virginia Woolf", "Audre Lorde",
    "bell hooks", "Gloria Steinem", "Mary Wollstonecraft", "Malala Yousafzai",
    # LGBTQ+ voices
    "Oscar Wilde", "James Baldwin", "Audre Lorde", "Alan Turing",
    # Philosophy / literature
    "Aristotle", "Voltaire", "Bertrand Russell", "Hannah Arendt",
    "Maya Angelou", "Toni Morrison", "Chinua Achebe", "Rumi",
    # Topics
    "Justice", "Freedom", "Love", "Knowledge", "Art",
]


# =============================================================================
#  HTTP
# =============================================================================

_REQUEST_DELAY = 3.0   # seconds between requests -- keeps us under Wikipedia's limit
_last_request  = 0.0

_HEADERS = {
    # Wikipedia's API policy requires a descriptive User-Agent
    "User-Agent": "miniGPT-DatasetBuilder/2.0 (educational project)",
    "Accept-Encoding": "gzip",
}


def _get(url: str, timeout: int = 20) -> Optional[str]:
    """
    Fetch a URL. Enforces a polite delay between requests and retries
    on HTTP 429 (rate limited) with exponential backoff.
    Returns the response text or None on failure.
    """
    global _last_request

    # Enforce minimum gap between requests
    wait = _REQUEST_DELAY - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()

    for attempt in range(6):
        try:
            if _HAS_REQUESTS:
                r = requests.get(url, headers=_HEADERS, timeout=timeout)
                if r.status_code == 429:
                    # Back off: 15s, 30s, 60s, 120s, 240s, 480s
                    backoff = max(int(r.headers.get("Retry-After", 0)),
                                  15 * (2 ** attempt))
                    print(f"  [429] Rate limited, waiting {backoff}s...")
                    time.sleep(backoff)
                    _last_request = time.time()
                    continue
                r.raise_for_status()
                return r.text
            else:
                req = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8", errors="ignore")

        except Exception as e:
            if "429" in str(e):
                backoff = 15 * (2 ** attempt)
                print(f"  [429] Rate limited, waiting {backoff}s...")
                time.sleep(backoff)
                _last_request = time.time()
                continue
            # Any other error: log and give up immediately
            print(f"  [WARN] {url[:70]}: {e}")
            return None

    print(f"  [WARN] Gave up after 6 retries: {url[:70]}")
    return None


def _get_json(url: str) -> Optional[dict]:
    """Fetch URL and parse JSON. Returns None on any failure."""
    text = _get(url)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


# =============================================================================
#  Text cleaning
# =============================================================================

def clean(text: str) -> str:
    """
    Strip markup artifacts and normalise whitespace.
    Removes HTML tags, wiki template syntax, citation markers,
    then collapses whitespace and drops very short lines.
    """
    text = re.sub(r'<[^>]+>',              ' ',    text)  # HTML tags
    text = re.sub(r'\[\[([^\]|]*\|)?([^\]]+)\]\]', r'\2', text)  # [[link|text]]
    text = re.sub(r'\{\{[^}]+\}\}',        '',     text)  # {{templates}}
    text = re.sub(r'==+[^=]+=+',           '',     text)  # == Headings ==
    text = re.sub(r"'{2,}",                '',     text)  # ''bold''
    text = re.sub(r'https?://\S+',         '',     text)  # URLs
    text = re.sub(r'\[\d+\]',              '',     text)  # [1] citations
    text = re.sub(r'[ \t]+',               ' ',    text)  # multiple spaces
    text = re.sub(r'\n{3,}',               '\n\n', text)  # excessive newlines
    # Drop lines shorter than 40 chars (navigation fragments, stray numbers, etc.)
    lines = [l.strip() for l in text.split('\n')]
    text  = '\n'.join(l for l in lines if len(l) >= 40 or l == '')
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header and footer from a downloaded book."""
    start_re = re.compile(r'\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG', re.I)
    end_re   = re.compile(r'\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG',   re.I)

    m = start_re.search(text)
    start = text.find('\n', m.end()) + 1 if m else 0

    m = end_re.search(text)
    end = m.start() if m else len(text)

    return text[start:end].strip()


# =============================================================================
#  Source fetchers
# =============================================================================

def fetch_gutenberg(target_chars: int) -> List[str]:
    """
    Download Gutenberg books until target_chars is reached.
    The last book is trimmed at a paragraph boundary so we land exactly on target.
    """
    print(f"\n[1/4] Gutenberg  (target {target_chars:,} chars)")
    books  = random.sample(GUTENBERG_BOOKS, len(GUTENBERG_BOOKS))
    chunks = []
    total  = 0

    for book_id, title in books:
        if total >= target_chars:
            break

        # Gutenberg serves books at a few different URL patterns
        urls = [
            f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
            f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        ]
        text = None
        for url in urls:
            raw = _get(url)
            if raw and len(raw) > 1000:
                text = clean(strip_gutenberg_boilerplate(raw))
                if len(text) > 500:
                    break

        if not text:
            print(f"  SKIP  {title}")
            continue

        # Trim last book to land on target
        remaining = target_chars - total
        if len(text) > remaining:
            cut  = text.rfind('\n\n', 0, remaining)
            text = text[:cut] if cut > 0 else text[:remaining]

        chunks.append(text)
        total += len(text)
        print(f"  OK    {title}  ({len(text):,} chars)  total={total:,}/{target_chars:,}")

    print(f"  Done: {total:,} chars from {len(chunks)} books")
    return chunks


def _fetch_wiki_article(title: str, base: str = "https://en.wikipedia.org") -> Optional[str]:
    """
    Fetch one Wikipedia (or Simple Wikipedia) article by title.
    Returns clean text or None if the article doesn't exist / is too short.
    """
    url = (
        f"{base}/w/api.php"
        f"?action=query&titles={urllib.parse.quote(title)}"
        "&prop=extracts&explaintext=1&exsectionformat=plain"
        "&format=json&redirects=1"
    )
    data = _get_json(url)
    if not data:
        return None
    try:
        page = next(iter(data["query"]["pages"].values()))
        if "extract" not in page or page.get("ns", 0) != 0:
            return None
        text = clean(page["extract"])
        return text if len(text) >= 200 else None
    except Exception:
        return None


def _fetch_wiki_random_titles(base: str = "https://en.wikipedia.org",
                               count: int = 20) -> List[str]:
    """Return a list of random article titles from Wikipedia."""
    url = (
        f"{base}/w/api.php"
        f"?action=query&list=random&rnlimit={count}"
        "&rnnamespace=0&format=json"
    )
    data = _get_json(url)
    if not data:
        return []
    try:
        return [p["title"] for p in data["query"]["random"]]
    except Exception:
        return []


def fetch_wikipedia(target_chars: int) -> List[str]:
    """
    Fetch Wikipedia articles in three passes:

    Pass 1 -- LGBTQ+ guaranteed (no cap, always fetched)
        Every article in LGBTQ_ARTICLES is fetched unconditionally first.
        This ensures LGBTQ+ content is always in the dataset.

    Pass 2 -- Diverse topics (up to 60% of target)
        Articles from DIVERSE_ARTICLES in random order.

    Pass 3 -- Random fill (remaining 40%)
        Random Wikipedia articles until target is reached.
    """
    print(f"\n[2/4] Wikipedia  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    seen   = set()

    # ------------------------------------------------------------------
    # Pass 1: LGBTQ+ articles -- always fetched, no cap
    # ------------------------------------------------------------------
    print(f"  Pass 1: LGBTQ+ articles ({len(LGBTQ_ARTICLES)} titles, guaranteed)")
    ok = fail = 0
    for title in LGBTQ_ARTICLES:
        seen.add(title)
        text = _fetch_wiki_article(title)
        if text:
            chunks.append(text)
            total += len(text)
            ok += 1
            print(f"    OK   [{ok:2d}/{len(LGBTQ_ARTICLES)}] {title}  ({len(text):,} chars)")
        else:
            fail += 1
            print(f"    SKIP [{ok:2d}/{len(LGBTQ_ARTICLES)}] {title}  -- not found")
    print(f"  Pass 1 done: {ok} fetched, {fail} skipped  |  {total:,} chars so far")

    # ------------------------------------------------------------------
    # Pass 2: Diverse topics, up to 60% of target
    # ------------------------------------------------------------------
    cap = int(target_chars * 0.6)
    print(f"  Pass 2: diverse topics (up to {cap:,} chars)")
    ok = fail = 0
    for title in random.sample(DIVERSE_ARTICLES, len(DIVERSE_ARTICLES)):
        if total >= cap:
            print(f"    Cap reached ({total:,}) -- moving to random fill")
            break
        if title in seen:
            continue
        seen.add(title)
        text = _fetch_wiki_article(title)
        if text:
            chunks.append(text)
            total += len(text)
            ok += 1
            print(f"    OK   {title}  ({len(text):,} chars)  total={total:,}")
        else:
            fail += 1
            print(f"    SKIP {title}  -- not found")
    print(f"  Pass 2 done: {ok} fetched, {fail} skipped  |  {total:,} chars so far")

    # ------------------------------------------------------------------
    # Pass 3: Random articles to fill remaining quota
    # ------------------------------------------------------------------
    print(f"  Pass 3: random fill ({total:,} -> {target_chars:,})")
    ok = fail = 0
    while total < target_chars and fail < 50:
        for title in _fetch_wiki_random_titles():
            if title in seen or total >= target_chars:
                continue
            seen.add(title)
            text = _fetch_wiki_article(title)
            if text:
                chunks.append(text)
                total += len(text)
                ok += 1
                print(f"    OK   [{ok}] {title}  ({len(text):,} chars)  "
                      f"total={total:,}/{target_chars:,}")
            else:
                fail += 1
    print(f"  Pass 3 done: {ok} fetched  |  {total:,} chars total")
    print(f"  Wikipedia done: {total:,} chars from {len(chunks)} articles")
    return chunks


def fetch_simple_wikipedia(target_chars: int) -> List[str]:
    """
    Fetch random Simple English Wikipedia articles.
    Simple Wikipedia uses shorter, cleaner sentences -- great training data.
    Uses a single API call per batch (random list + extract in one request).
    """
    print(f"\n[3/4] Simple Wikipedia  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    seen   = set()
    fails  = 0

    while total < target_chars and fails < 30:
        # Fetch 20 random titles and their extracts in a single API call
        url = (
            "https://simple.wikipedia.org/w/api.php"
            "?action=query&list=random&rnlimit=20&rnnamespace=0"
            "&prop=extracts&explaintext=1&format=json"
        )
        data = _get_json(url)
        if not data:
            fails += 1
            continue

        titles = []
        try:
            titles = [p["title"] for p in data["query"]["random"]]
        except Exception:
            fails += 1
            continue

        for title in titles:
            if title in seen or total >= target_chars:
                continue
            seen.add(title)
            # One article fetch per title (simple wiki doesn't support batch extract+random)
            text = _fetch_wiki_article(title, base="https://simple.wikipedia.org")
            if text and len(text) >= 100:
                chunks.append(text)
                total += len(text)

    print(f"  Simple Wikipedia done: {total:,} chars from {len(chunks)} articles")
    return chunks


def fetch_wikiquote(target_chars: int) -> List[str]:
    """
    Fetch quote pages from Wikiquote.
    Short varied sentences from diverse voices -- good for vocabulary breadth.
    """
    print(f"\n[4/4] Wikiquote  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    pages  = random.sample(WIKIQUOTE_PAGES, len(WIKIQUOTE_PAGES))

    for name in pages:
        if total >= target_chars:
            break
        url = (
            "https://en.wikiquote.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(name)}"
            "&prop=extracts&explaintext=1&format=json"
        )
        data = _get_json(url)
        if not data:
            continue
        try:
            page = next(iter(data["query"]["pages"].values()))
            if "extract" not in page:
                continue
            text = clean(page["extract"])
            if len(text) < 50:
                continue
            chunks.append(f"{name}:\n{text}")
            total += len(chunks[-1])
        except Exception:
            continue

    print(f"  Wikiquote done: {total:,} chars from {len(chunks)} pages")
    return chunks


# =============================================================================
#  Assemble
# =============================================================================

def build_dataset(target_chars: int, output_file: str) -> None:
    """
    Collect from all sources, shuffle documents, write to file.

    Source split:
        40% Gutenberg  -- long-form prose and narrative
        35% Wikipedia  -- factual, diverse topics
        15% Simple Wikipedia -- clean plain prose
        10% Wikiquote  -- short diverse quotes
    """
    print("=" * 60)
    print("miniGPT dataset builder")
    print(f"Target : {target_chars:,} chars -> {output_file}")
    print("=" * 60)

    # Brief startup pause so Wikipedia's rate limiter is clear
    # if this is a re-run shortly after a previous one.
    print("\nWaiting 15s before starting (clears any Wikipedia rate limit)...")
    for i in range(15, 0, -5):
        print(f"  {i}s...", end="\r", flush=True)
        time.sleep(5)
    print("  Starting.         ")

    gutenberg_chars    = int(target_chars * 0.40)
    wikipedia_chars    = int(target_chars * 0.35)
    simple_wiki_chars  = int(target_chars * 0.15)
    wikiquote_chars    = int(target_chars * 0.10)

    chunks = []
    chunks.extend(fetch_gutenberg(gutenberg_chars))
    chunks.extend(fetch_wikipedia(wikipedia_chars))
    chunks.extend(fetch_simple_wikipedia(simple_wiki_chars))
    chunks.extend(fetch_wikiquote(wikiquote_chars))

    if not chunks:
        print("\nERROR: no text collected. Check your internet connection.")
        sys.exit(1)

    # Shuffle so sources are interleaved -- prevents the model from seeing
    # all Gutenberg first, then all Wikipedia, etc.
    random.shuffle(chunks)

    full_text = "\n\n".join(chunks)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}',  ' ',    full_text)
    full_text = full_text.strip()

    # Trim to target at a word boundary
    if len(full_text) > target_chars:
        cut       = full_text.rfind(' ', 0, target_chars)
        full_text = full_text[:cut] if cut > 0 else full_text[:target_chars]

    # Write
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    from collections import Counter
    size_mb = os.path.getsize(output_file) / 1024 / 1024

    print("\n" + "=" * 60)
    print("Dataset complete")
    print("=" * 60)
    print(f"  File      : {output_file}  ({size_mb:.1f} MB)")
    print(f"  Chars     : {len(full_text):,}")
    print(f"  Words     : {len(full_text.split()):,}")
    print(f"  Paragraphs: {full_text.count(chr(10)*2):,}")
    print(f"  Documents : {len(chunks):,}")
    print(f"  Vocab     : {len(set(full_text))} unique chars")
    top = Counter(full_text).most_common(8)
    print(f"  Top chars : {top}")
    print(f"\nReady for training:")
    print(f"  python miniGPT/cli.py --train {output_file} --simple_vocab ...")
    print("=" * 60)


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a diverse local dataset for miniGPT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--target_chars", type=int, default=2_200_000,
                        help="Target character count (default: 2,200,000)")
    parser.add_argument("--output",       type=str, default="diverse_dataset.txt",
                        help="Output filename (default: diverse_dataset.txt)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no_gutenberg",   action="store_true", help="Skip Gutenberg")
    parser.add_argument("--no_wikipedia",   action="store_true", help="Skip Wikipedia")
    parser.add_argument("--no_simple_wiki", action="store_true", help="Skip Simple Wikipedia")
    parser.add_argument("--no_wikiquote",   action="store_true", help="Skip Wikiquote")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.no_gutenberg:   GUTENBERG_BOOKS.clear()
    if args.no_wikipedia:   LGBTQ_ARTICLES.clear(); DIVERSE_ARTICLES.clear()
    if args.no_wikiquote:   WIKIQUOTE_PAGES.clear()

    build_dataset(args.target_chars, args.output)