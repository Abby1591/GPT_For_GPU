"""
build_dataset.py
================
Builds a diverse plain-text training dataset for miniGPT from free public sources.
No API keys. No HuggingFace. Runs entirely locally.

Sources
-------
  1. Project Gutenberg  -- classic books + left-wing / radical literature
  2. Wikipedia LGBTQ+   -- guaranteed, no cap, always first
  3. Wikipedia Left     -- socialism, communism, labour, anarchism (guaranteed)
  4. Wikipedia Diverse  -- civil rights, science, history, women, environment
  5. Wikipedia Academic -- chemistry, physics, biology, maths, history of science
  6. Simple Wikipedia   -- plain English, shorter sentences
  7. Wikiquote          -- short quotes from diverse, radical, and scientific voices
  8. Reddit             -- conversational, informal register (local .jsonl dumps)
  9. Tool training      -- Toolformer-style [TOOL:name|arg][RESULT:...] examples

Source split (default 200M chars  ~200 MB):
    18% Gutenberg         -- long-form prose, vocabulary breadth
    28% Wikipedia (LGBTQ+ + Left guaranteed, then Diverse + Academic)
    10% Simple Wikipedia
     5% Wikiquote
     7% Wikibooks         -- programming + science textbooks
    12% Wiktionary        -- clean word definitions, near-full dictionary
    12% Reddit            -- conversational, informal, programming Q&A
     8% Tool training     -- teaches model WHEN and HOW to call tools

Usage
-----
    python build_dataset.py
    python build_dataset.py --target_chars 500000000 --output big_dataset.txt
    python build_dataset.py --no_gutenberg
    python build_dataset.py --no_wikipedia
    python build_dataset.py --no_reddit
    python build_dataset.py --phase 7   # Reddit only
    python build_dataset.py --phase 8   # Tool training only
    python build_dataset.py --no_tools  # Skip tool training phase

Reddit (phase 7) requires local .jsonl dump files:
    RC_2016-01.jsonl   (comments)
    RS_2016-01.jsonl   (submissions)

Output
------
    diverse_dataset.txt  -- ready to pass directly to miniGPT --train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import gzip
import bz2
import gc
import html
import mwparserfromhell
import hashlib

from collections import Counter, deque
from typing import Iterator, List, Optional, Dict

import unicodedata

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# =============================================================================
#  Gutenberg books
# =============================================================================

GUTENBERG_BOOKS = [
    # --- Left-wing / radical / socialist thought ---
    (61,    "The Communist Manifesto - Marx & Engels"),
    (7370,  "Common Sense - Thomas Paine"),
    (3296,  "Rights of Man - Thomas Paine"),
    (31,    "The Age of Reason - Thomas Paine"),
    (1250,  "Mutual Aid - Kropotkin"),
    (32011, "The Conquest of Bread - Kropotkin"),
    (36276, "Fields Factories and Workshops - Kropotkin"),
    (25344, "The Jungle - Upton Sinclair"),
    (1155,  "The Iron Heel - Jack London"),
    (3800,  "The People of the Abyss - Jack London"),
    (1139,  "The War of the Classes - Jack London"),
    (36429, "Looking Backward - Edward Bellamy"),
    (6762,  "News from Nowhere - William Morris"),
    (30,    "The Ragged Trousered Philanthropists"),
    (3693,  "Walden - Thoreau"),
    (243,   "Civil Disobedience - Thoreau"),
    # --- Black American voices ---
    (23,    "Narrative of Frederick Douglass"),
    (2852,  "My Bondage and My Freedom - Douglass"),
    (33,    "The Souls of Black Folk - Du Bois"),
    (408,   "Up From Slavery - Booker T. Washington"),
    (203,   "Uncle Tom's Cabin - Stowe"),
    (45,    "Anne of Green Gables"),
    # --- Women's voices / feminist ---
    (1952,  "The Yellow Wallpaper - Charlotte Perkins Gilman"),
    (514,   "Little Women - Alcott"),
    (1260,  "Jane Eyre - Charlotte Bronte"),
    (768,   "Wuthering Heights - Emily Bronte"),
    (161,   "Sense and Sensibility - Austen"),
    (1342,  "Pride and Prejudice - Austen"),
    (34902, "A Vindication of the Rights of Woman - Wollstonecraft"),
    (3268,  "Herland - Charlotte Perkins Gilman"),
    # --- Classic fiction (vocabulary breadth) ---
    (11,    "Alice in Wonderland"),
    (84,    "Frankenstein - Shelley"),
    (98,    "A Tale of Two Cities - Dickens"),
    (1400,  "Great Expectations - Dickens"),
    (46,    "A Christmas Carol - Dickens"),
    (1661,  "Adventures of Sherlock Holmes"),
    (2701,  "Moby Dick - Melville"),
    (76,    "Adventures of Huckleberry Finn - Twain"),
    (74,    "The Adventures of Tom Sawyer - Twain"),
    (174,   "The Picture of Dorian Gray - Wilde"),
    (345,   "Dracula - Stoker"),
    (5200,  "The Metamorphosis - Kafka"),
    (2542,  "A Doll's House - Ibsen"),
    (1184,  "The Count of Monte Cristo - Dumas"),
    (1727,  "The Odyssey - Homer"),
    (6130,  "The Iliad - Homer"),
    (4085,  "The Time Machine - H.G. Wells"),
    (35,    "The War of the Worlds - H.G. Wells"),
    (5230,  "The Invisible Man - H.G. Wells"),
    # --- Philosophy ---
    (1998,  "Thus Spoke Zarathustra - Nietzsche"),
    (4705,  "Meditations - Marcus Aurelius"),
    (2130,  "The Republic - Plato"),
    (1497,  "The Symposium - Plato"),
    (1228,  "On the Origin of Species - Darwin"),
    (2009,  "The Origin of the Family - Engels"),
    # --- World literature ---
    (8492,  "Gitanjali - Tagore"),
    (7333,  "Tao Te Ching - Lao Tzu"),
    (2413,  "Arabian Nights vol 1"),
    (17192, "The Art of War - Sun Tzu"),
]


# =============================================================================
#  LGBTQ+ Wikipedia articles (guaranteed, no cap)
# =============================================================================

LGBTQ_ARTICLES = [
    "Stonewall riots", "LGBT history", "Timeline of LGBT history",
    "Homosexuality in ancient Greece", "Homosexuality in ancient Rome",
    "Two-spirit", "Hijra (South Asia)", "History of homosexuality",
    "History of transgender people", "Transgender history", "Sodomy laws",
    "Decriminalization of homosexuality", "Bowers v. Hardwick",
    "Lawrence v. Texas", "Obergefell v. Hodges",
    "Harvey Milk", "Marsha P. Johnson", "Sylvia Rivera", "Bayard Rustin",
    "Alan Turing", "Oscar Wilde", "James Baldwin", "Audre Lorde",
    "Laverne Cox", "Christine Jorgensen", "Renee Richards", "Josephine Baker",
    "Frida Kahlo", "Langston Hughes", "Adrienne Rich", "Kate Millett",
    "Leslie Feinberg", "Quentin Crisp", "Del Martin and Phyllis Lyon",
    "Frank Kameny", "Harry Hay", "Barbara Gittings", "Edie Windsor",
    "Matthew Shepard", "Brandon Teena", "Pulse nightclub shooting",
    "LGBT rights in the United States", "LGBT rights by country or territory",
    "Same-sex marriage", "Same-sex marriage in the United States",
    "Transgender rights in the United States", "LGBT adoption",
    "Don't ask, don't tell", "Employment Non-Discrimination Act",
    "Hate crime laws in the United States",
    "Legal recognition of non-binary gender", "Intersex human rights",
    "LGBT rights in Russia", "LGBT rights in Uganda", "LGBT rights in China",
    "LGBT rights in India", "Section 28",
    "Pride parade", "Gay liberation", "Lesbian feminism", "Queer theory",
    "Gender studies", "LGBT community", "Bisexuality", "Non-binary gender",
    "Genderqueer", "Asexuality", "Pansexuality", "Drag (gender expression)",
    "Ballroom culture", "Coming out", "LGBT culture", "Pink triangle",
    "Rainbow flag (LGBT)", "Camp (style)",
    "HIV/AIDS", "AIDS crisis", "ACT UP", "Gay Men's Health Crisis", "Ryan White",
    "Human Rights Campaign", "PFLAG", "Lambda Legal", "GLSEN",
    "National LGBTQ Task Force", "Gay-Straight Alliance",
    "Giovanni's Room", "The Well of Loneliness", "Angels in America",
    "The Normal Heart", "Fun Home", "Maurice (novel)", "Brokeback Mountain",
    "Paris Is Burning (film)", "The L Word", "Queer as Folk",
]


# =============================================================================
#  Left-wing / socialist articles (guaranteed, no cap)
# =============================================================================

LEFT_ARTICLES = [
    "Marxism", "Socialism", "Communism", "Anarchism", "Marxism–Leninism",
    "Trotskyism", "Libertarian socialism", "Democratic socialism",
    "Social democracy", "Anarcho-communism", "Syndicalism",
    "Revolutionary socialism", "Leninism", "Stalinism", "Maoism",
    "Feminism", "Socialist feminism", "Marxist feminism", "Intersectionality",
    "Critical theory", "Frankfurt School", "Hegelian dialectics",
    "Historical materialism", "Dialectical materialism", "Class consciousness",
    "False consciousness", "Alienation (Marx)", "Mode of production",
    "Means of production", "Base and superstructure", "Surplus value",
    "Capital (Marx)",
    "Karl Marx", "Friedrich Engels", "Vladimir Lenin", "Leon Trotsky",
    "Rosa Luxemburg", "Emma Goldman", "Peter Kropotkin", "Mikhail Bakunin",
    "Antonio Gramsci", "Georg Wilhelm Friedrich Hegel", "Eugene V. Debs",
    "Mother Jones", "Big Bill Haywood", "Alexandra Kollontai", "Che Guevara",
    "Fidel Castro", "Ho Chi Minh", "Mao Zedong", "Salvador Allende",
    "Hugo Chavez", "Angela Davis", "Huey P. Newton", "Fred Hampton",
    "Claudia Jones", "C. L. R. James", "Paul Robeson", "Howard Zinn",
    "Noam Chomsky", "Herbert Marcuse", "Jean-Paul Sartre",
    "Simone de Beauvoir", "Frantz Fanon", "Walter Rodney", "bell hooks",
    "Russian Revolution", "October Revolution", "Paris Commune",
    "Spanish Civil War", "Cuban Revolution", "Chinese Revolution",
    "Haitian Revolution", "Labour movement", "Trade union", "General strike",
    "Industrial Workers of the World", "International Workers' Day",
    "Black Panther Party", "Young Lords", "American Indian Movement",
    "Chicano movement", "Anti-capitalism", "Anti-imperialism",
    "Decolonization", "Third-worldism", "Non-Aligned Movement",
    "Zapatista Army of National Liberation", "Occupy movement",
    "Anti-globalization movement", "Socialist International",
    "Communist International",
    "Capitalism", "Neoliberalism", "Imperialism", "Colonialism",
    "Wealth inequality", "Poverty", "Universal basic income", "Welfare state",
    "Mixed economy", "Planned economy", "Market socialism", "Worker cooperative",
    "Common ownership", "Nationalization", "Privatization",
    "Soviet Union", "Cuba", "Yugoslavia under Tito", "Allende's Chile",
    "Bolivarian Revolution",
]


# =============================================================================
#  Diverse curated Wikipedia articles
# =============================================================================

DIVERSE_ARTICLES = [
    "Martin Luther King Jr.", "Rosa Parks", "Malcolm X", "Harriet Tubman",
    "Frederick Douglass", "Nelson Mandela", "Apartheid", "Black Lives Matter",
    "Civil Rights Act of 1964", "Voting Rights Act of 1965",
    "Brown v. Board of Education", "Montgomery bus boycott",
    "March on Washington", "Selma to Montgomery marches", "Emmett Till",
    "John Lewis", "Fannie Lou Hamer", "Thurgood Marshall", "Shirley Chisholm",
    "NAACP", "Tulsa race massacre", "Juneteenth", "Jim Crow laws",
    "Slavery in the United States", "Underground Railroad",
    "Reconstruction era", "Redlining", "Loving v. Virginia",
    "Trayvon Martin", "George Floyd protests",
    "Women's suffrage", "Mary Wollstonecraft", "Gloria Steinem",
    "Betty Friedan", "Sojourner Truth", "Susan B. Anthony", "Malala Yousafzai",
    "Ruth Bader Ginsburg", "Seneca Falls Convention", "Me Too movement",
    "Gender pay gap", "Reproductive rights", "Roe v. Wade",
    "Women's liberation movement", "Equal Rights Amendment", "Title IX",
    "Marie Curie", "Katherine Johnson", "Mae Jemison", "Rosalind Franklin",
    "Chien-Shiung Wu", "Ada Lovelace", "Grace Hopper", "Charles Darwin",
    "Albert Einstein", "Carl Sagan", "Jane Goodall", "Wangari Maathai",
    "Neil deGrasse Tyson", "Tu Youyou", "Nikola Tesla", "Alan Turing",
    "Subrahmanyan Chandrasekhar",
    "Mahatma Gandhi", "Indian independence movement", "Atlantic slave trade",
    "Desmond Tutu", "Chinua Achebe", "Kwame Nkrumah", "Jawaharlal Nehru",
    "Universal Declaration of Human Rights", "Amnesty International",
    "Rwandan genocide", "Armenian genocide", "Indigenous peoples",
    "Climate change", "Global warming", "Paris Agreement", "Greta Thunberg",
    "Renewable energy", "Environmental justice", "Biodiversity",
    "Amazon rainforest", "Climate justice",
    "Disability rights movement", "Neurodiversity", "Mental health",
    "Deinstitutionalisation",
    "World War II", "Holocaust", "Cold War", "French Revolution",
    "American Revolution", "Democracy", "Human rights", "United Nations",
    "Vietnam War", "Korean War", "Iraq War", "Afghanistan War",
    "Nuclear weapons", "Nuclear disarmament",
]


# =============================================================================
#  Academic / knowledge Wikipedia articles
# =============================================================================

ACADEMIC_ARTICLES = [
    "Chemistry", "Atom", "Chemical element", "Periodic table", "Chemical bond",
    "Covalent bond", "Ionic bonding", "Molecule", "Chemical reaction",
    "Acid–base reaction", "Oxidation state", "Organic chemistry", "Polymer",
    "Protein", "DNA", "RNA", "Enzyme", "Photosynthesis", "Cellular respiration",
    "Thermodynamics", "Entropy", "Gibbs free energy", "Electrolysis",
    "Radioactive decay", "Nuclear fission", "Nuclear fusion",
    "Physics", "Classical mechanics", "Quantum mechanics", "Special relativity",
    "General relativity", "Electromagnetism", "Wave–particle duality",
    "Uncertainty principle", "Standard Model", "Black hole", "Big Bang",
    "Dark matter", "Gravity", "Speed of light", "Electromagnetic spectrum",
    "Biology", "Cell (biology)", "Evolution", "Natural selection", "Genetics",
    "Gene", "Chromosome", "Mutation", "Ecology", "Ecosystem", "Food chain",
    "Nervous system", "Immune system", "Virus", "Bacteria",
    "Antibiotic resistance", "CRISPR", "Stem cell",
    "Mathematics", "Calculus", "Linear algebra", "Statistics", "Probability",
    "Prime number", "Topology", "Set theory", "Mathematical proof",
    "Cryptography",
    "Earth", "Plate tectonics", "Atmosphere of Earth", "Ocean", "Solar System",
    "Galaxy", "Milky Way", "Exoplanet", "Asteroid", "Comet",
    "Scientific revolution", "Age of Enlightenment", "History of chemistry",
    "History of physics", "History of biology", "History of mathematics",
    "Copernican heliocentrism", "Isaac Newton", "Galileo Galilei",
    "Johannes Kepler",
    "Ancient Egypt", "Ancient Greece", "Ancient Rome", "Byzantine Empire",
    "Islamic Golden Age", "Renaissance", "Industrial Revolution", "World War I",
    "Great Depression", "Transatlantic slave trade", "Silk Road",
    "Medieval Europe", "Ming dynasty", "Ottoman Empire", "British Empire",
    "Philosophy", "Ethics", "Epistemology", "Metaphysics",
    "Philosophy of science", "Utilitarianism", "Kantian ethics",
    "Existentialism", "Empiricism", "Rationalism", "Phenomenology",
    "Social contract", "Justice", "Political philosophy", "Anarchist philosophy",
    "Keynesian economics", "Austerity", "Unemployment", "Inflation",
    "Minimum wage", "Housing", "Homelessness", "Food security", "Healthcare",
    "Universal healthcare", "Education", "Mass incarceration",
    "Prison–industrial complex",
]


# =============================================================================
#  Wikiquote pages
# =============================================================================

WIKIQUOTE_PAGES = [
    "Albert Einstein", "Marie Curie", "Carl Sagan", "Richard Feynman",
    "Stephen Hawking", "Ada Lovelace", "Grace Hopper", "Nikola Tesla",
    "Katherine Johnson", "Mae Jemison", "Charles Darwin", "Alan Turing",
    "Richard Dawkins", "Neil deGrasse Tyson", "Rachel Carson",
    "Karl Marx", "Friedrich Engels", "Vladimir Lenin", "Rosa Luxemburg",
    "Emma Goldman", "Eugene V. Debs", "Antonio Gramsci", "Leon Trotsky",
    "Howard Zinn", "Noam Chomsky", "Angela Davis", "Fred Hampton",
    "Che Guevara", "Nelson Mandela", "Frantz Fanon", "James Connolly",
    "Mahatma Gandhi", "Ho Chi Minh",
    "Martin Luther King Jr.", "Malcolm X", "Rosa Parks", "Harriet Tubman",
    "Frederick Douglass", "Sojourner Truth", "Fannie Lou Hamer",
    "John Lewis", "Audre Lorde", "James Baldwin", "Langston Hughes",
    "Maya Angelou", "Toni Morrison", "Zora Neale Hurston",
    "Simone de Beauvoir", "Virginia Woolf", "bell hooks", "Gloria Steinem",
    "Mary Wollstonecraft", "Malala Yousafzai", "Adrienne Rich",
    "Sylvia Plath", "Susan B. Anthony",
    "Oscar Wilde", "Harvey Milk", "Bayard Rustin", "Quentin Crisp",
    "Aristotle", "Voltaire", "Bertrand Russell", "Hannah Arendt",
    "Chinua Achebe", "Rumi", "Pablo Neruda", "Bertolt Brecht",
    "Jean-Paul Sartre", "Albert Camus", "George Orwell",
    "Justice", "Freedom", "Knowledge", "Love", "Art", "Revolution",
    "Democracy", "Equality",
]

# =============================================================================
#  HTTP helpers
# =============================================================================

_REQUEST_DELAY  = 3.5
_last_request   = 0.0
_request_count  = 0
_COOLDOWN_EVERY = 150

_HEADERS = {
    "User-Agent": "miniGPT-DatasetBuilder/4.0 (educational, non-commercial)",
    "Accept-Encoding": "gzip",
}


def _get(url: str, timeout: int=25) -> Optional[str]:
    global _last_request, _request_count

    _request_count += 1

    if _request_count % _COOLDOWN_EVERY == 0:
        print(f'  [cooldown] {_request_count} requests made, pausing 30s...')
        time.sleep(30)

    wait = _REQUEST_DELAY - (time.time() - _last_request)

    if wait > 0:
        time.sleep(wait)

    _last_request = time.time()

    for attempt in range(7):
        try:

            if _HAS_REQUESTS:
                r = requests.get(url, headers=_HEADERS, timeout=timeout)

                if r.status_code == 429:
                    backoff = max(
                        int(r.headers.get('Retry-After', 0)),
                        20 * 2 ** attempt
                    )

                    print(f'  [429] Rate limited, waiting {backoff}s...')
                    time.sleep(backoff)
                    _last_request = time.time()
                    continue

                if r.status_code == 503:
                    time.sleep(30)
                    continue

                r.raise_for_status()
                return r.text

            else:
                req = urllib.request.Request(url, headers=_HEADERS)

                with urllib.request.urlopen(req, timeout=timeout) as resp:

                    data = resp.read()

                    if resp.headers.get('Content-Encoding') == 'gzip':
                        data = gzip.decompress(data)

                    return data.decode('utf-8', errors='ignore')

                    print(resp.headers.get('Content-Encoding'))

        except Exception as e:

            if '429' in str(e) or '503' in str(e):
                backoff = 20 * 2 ** attempt

                print(f'  [rate limit] waiting {backoff}s...')
                time.sleep(backoff)

                _last_request = time.time()
                continue

            print(f'  [WARN] {url[:80]}: {e}')
            return None

    print(f'  [WARN] Gave up after 7 retries: {url[:80]}')
    return None

def _get_json(url: str) -> Optional[dict]:
    text = _get(url)

    if text is None:
        return None

    if text.lstrip().startswith('<'):
        print("HTML returned instead of JSON")
        print(text[:500])
        return None

    try:
        return json.loads(text)

    except Exception as e:
        print("JSON decode failed:", e)
        print(text[:500])
        return None

# =============================================================================
#  Text cleaning
# =============================================================================

def clean(text: str) -> str:
    text = re.sub(r'<[^>]+>',                       ' ',   text)
    text = re.sub(r'\[\[([^\]|]*\|)?([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\{\{[^}]*\}\}',                 '',    text)
    text = re.sub(r'==+[^=]+=+',                    '',    text)
    text = re.sub(r"'{2,}",                          '',    text)
    text = re.sub(r'https?://\S+',                  '',    text)
    text = re.sub(r'\[\d+\]',                       '',    text)
    text = re.sub(r'\^?\[\w+\]',                    '',    text)
    text = re.sub(r'thumb\|[^\|]+\|',              '',    text)
    text = re.sub(r'File:[^\n]+',                   '',    text)
    text = re.sub(r'[ \t]+',                        ' ',   text)
    text = re.sub(r'\n{3,}',                        '\n\n',text)
    lines = [l.strip() for l in text.split('\n')]
    text  = '\n'.join(l for l in lines if len(l) >= 40 or l == '')
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    start_re = re.compile(r'\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG', re.I)
    end_re   = re.compile(r'\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG',   re.I)
    m_start  = start_re.search(text)
    start    = text.find('\n', m_start.end()) + 1 if m_start else 0
    m_end    = end_re.search(text)
    end      = m_end.start() if m_end else len(text)
    return text[start:end].strip()


# =============================================================================
#  Source fetchers (phases 1-7, unchanged)
# =============================================================================

def fetch_gutenberg(target_chars: int) -> List[str]:
    print(f"\n[1/8] Gutenberg  (target {target_chars:,} chars, {len(GUTENBERG_BOOKS)} books available)")
    books  = random.sample(GUTENBERG_BOOKS, len(GUTENBERG_BOOKS))
    chunks = []
    total  = 0

    for book_id, title in books:
        if total >= target_chars:
            break
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
        remaining = target_chars - total
        if len(text) > remaining:
            cut  = text.rfind('\n\n', 0, remaining)
            text = text[:cut] if cut > 0 else text[:remaining]
        chunks.append(text)
        total += len(text)
        print(f"  OK    {title}  ({len(text):,} chars)  total={total:,}/{target_chars:,}")

    print(f"  Gutenberg done: {total:,} chars from {len(chunks)} books")
    return chunks


def _fetch_wiki_article(title: str,
                        base: str = "https://en.wikipedia.org") -> Optional[str]:
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


def _fetch_pass(label: str, articles: List[str],
                seen: set, cap: Optional[int] = None) -> tuple:
    chunks = []
    total  = 0
    ok = fail = 0
    for title in articles:
        if cap is not None and total >= cap:
            print(f"    Cap {cap:,} reached -- stopping this pass")
            break
        if title in seen:
            continue
        seen.add(title)
        text = _fetch_wiki_article(title)
        if text:
            chunks.append(text)
            total += len(text)
            ok += 1
            print(f"    OK   [{ok}] {title}  ({len(text):,} chars)  running={total:,}")
        else:
            fail += 1
            print(f"    SKIP {title}")
    print(f"  {label}: {ok} fetched, {fail} skipped  |  {total:,} chars")
    return chunks, total


def fetch_wikipedia(target_chars: int) -> List[str]:
    print(f"\n[2/8] Wikipedia  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    seen   = set()

    print(f"\n  Pass 1: LGBTQ+ ({len(LGBTQ_ARTICLES)} articles, guaranteed)")
    c, t = _fetch_pass("LGBTQ+", LGBTQ_ARTICLES, seen)
    chunks.extend(c); total += t

    print(f"\n  Pass 2: Left/socialist ({len(LEFT_ARTICLES)} articles, guaranteed)")
    c, t = _fetch_pass("Left", LEFT_ARTICLES, seen)
    chunks.extend(c); total += t

    print(f"\n  Pass 3: Diverse curated (cap {int(target_chars * 0.40):,})")
    c, t = _fetch_pass("Diverse", random.sample(DIVERSE_ARTICLES, len(DIVERSE_ARTICLES)),
                        seen, cap=int(target_chars * 0.40))
    chunks.extend(c); total += t

    if total < target_chars:
        print(f"\n  Pass 4: Academic/science (cap {int(target_chars * 0.30):,})")
        c, t = _fetch_pass("Academic", random.sample(ACADEMIC_ARTICLES, len(ACADEMIC_ARTICLES)),
                            seen, cap=int(target_chars * 0.30))
        chunks.extend(c); total += t

    if total < target_chars:
        print(f"\n  Pass 5: random fill ({total:,} -> {target_chars:,})")
        ok = fail = 0
        while total < target_chars and fail < 60:
            for title in _fetch_wiki_random_titles():
                if title in seen or total >= target_chars:
                    continue
                seen.add(title)
                text = _fetch_wiki_article(title)
                if text:
                    chunks.append(text); total += len(text); ok += 1
                    print(f"    RAND [{ok}] {title}  ({len(text):,} chars)  total={total:,}")
                else:
                    fail += 1
        print(f"  Random fill done: {ok} articles  |  {total:,} chars total")

    print(f"\n  Wikipedia done: {total:,} chars from {len(chunks)} articles")
    return chunks


def fetch_simple_wikipedia(target_chars: int) -> List[str]:
    print(f"\n[3/8] Simple Wikipedia  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    seen   = set()
    ok = fail = 0

    while total < target_chars and fail < 30:
        url = (
            "https://simple.wikipedia.org/w/api.php"
            "?action=query&list=random&rnlimit=20&rnnamespace=0&format=json"
        )
        data = _get_json(url)
        if not data:
            fail += 1
            continue
        try:
            titles = [p["title"] for p in data["query"]["random"]]
        except Exception:
            fail += 1
            continue
        batch_added = 0
        for title in titles:
            if title in seen or total >= target_chars:
                continue
            seen.add(title)
            text = _fetch_wiki_article(title, base="https://simple.wikipedia.org")
            if text and len(text) >= 100:
                chunks.append(text); total += len(text); batch_added += 1; ok += 1
                if ok % 5 == 0:
                    print(f"    [{ok}] {title}  ({len(text):,} chars)  total={total:,}/{target_chars:,}")
            else:
                fail += 1
        if batch_added == 0:
            break

    print(f"  Simple Wikipedia done: {total:,} chars from {len(chunks)} articles")
    return chunks


def fetch_wikiquote(target_chars: int) -> List[str]:
    print(f"\n[4/8] Wikiquote  (target {target_chars:,} chars)")
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
            print(f"  OK  {name}  ({len(chunks[-1]):,} chars)")
        except Exception:
            continue

    print(f"  Wikiquote done: {total:,} chars from {len(chunks)} pages")
    return chunks


def fetch_wikibooks(target_chars: int) -> List[str]:
    print(f"\n[5/8] Wikibooks  (target {target_chars:,} chars)")
    BOOKS = [
        "Chemistry", "Physics Study Guide", "Biology",
        "Human Physiology", "History of Western Civilisation",
        "World History", "Introduction to Sociology",
        "Introduction to Philosophy", "Economics",
        "Calculus", "Linear Algebra",
    ]
    chunks = []
    total  = 0

    for book in random.sample(BOOKS, len(BOOKS)):
        if total >= target_chars:
            break
        url = (
            "https://en.wikibooks.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(book)}"
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
            if len(text) < 200:
                continue
            chunks.append(text)
            total += len(text)
            print(f"  OK  {book}  ({len(text):,} chars)")
        except Exception:
            continue

    print(f"  Wikibooks done: {total:,} chars from {len(chunks)} books")
    return chunks

def fetch_reddit(
    target_chars:     int,
    comments_file:    str  = "Dumps/comments/RC_2016-01.jsonl",
    submissions_file: str  = "Dumps/submissions/RS_2016-01.jsonl",
    reddit_output:    str  = "Dataset/Dataset part backup/phase_7_reddit.txt",
    no_submissions:   bool = False,
    resume:           bool = False,
    min_score:        int  = 4,
) -> list[str]:
    print(f"\n[7/8] Reddit  (target {target_chars:,} chars)")

    # ------------------------------------------------------------------ #
    # Path Fix                                                             #
    # ------------------------------------------------------------------ #
    base = os.path.dirname(os.path.abspath(__file__))
    def _rel(p): return p if os.path.isabs(p) else os.path.join(base, p)
    comments_file    = _rel(comments_file)
    submissions_file = _rel(submissions_file)
    reddit_output    = _rel(reddit_output)

    if not os.path.exists(comments_file) and not os.path.exists(submissions_file):
        print(f"  No dump files found — skipping.")
        print(f"  Expected: {comments_file}  /  {submissions_file}")
        return []

    # ------------------------------------------------------------------ #
    # Constants                                                            #
    # ------------------------------------------------------------------ #
    MIN_SCORE_COMMENT    = min_score
    MIN_SCORE_SUBMISSION = max(1, min_score - 1)
    MIN_LEN, MAX_LEN     = 60, 3000
    DEDUP_CACHE          = 200_000
    COMMENT_RATIO        = 0.80          # 80% comments, 20% submissions
    AVG_CHARS_PER_LINE   = 200           # rough estimate for target_lines
    MAX_THREAD_CHARS = 2000

    target_lines       = max(1000, target_chars // AVG_CHARS_PER_LINE)
    target_comments    = int(target_lines * COMMENT_RATIO)
    target_submissions = 0 if no_submissions else (target_lines - target_comments)

    # ------------------------------------------------------------------ #
    # Allowed Subreddits                                                   #
    # ------------------------------------------------------------------ #
    GOOD_SUBS: set[str] = {
        # Programming / CS / ML
        "programming", "compsci", "softwareengineering", "learnprogramming",
        "Python", "javascript", "cpp", "rust", "golang", "java", "haskell",
        "MachineLearning", "learnmachinelearning", "deeplearning", "artificial",
        "datascience", "algorithms", "computerscience", "coding", "webdev",
        "gamedev", "devops", "sysadmin", "linux", "opensource", "commandline",
        # Math / Science
        "math", "askmath", "statistics", "physics", "chemistry", "biology",
        "AskScience", "science", "neuroscience", "genetics",
        # Conversation / Q&A
        "AskReddit", "explainlikeimfive", "NoStupidQuestions", "answers",
        "AskHistorians", "TrueReddit", "DepthHub", "changemyview",
        "philosophy", "Philosophy", "Ethics",
        # Left / Politics / Social Justice
        "socialism", "communism", "anarchism", "Marxism", "labour",
        "PoliticalDiscussion", "progressive", "SocialJustice",
        "feminism", "GenderCritical", "lgbt", "ainbow", "trans", "nonbinary",
        "BlackLives", "antiracism", "labor",
        # Culture / Knowledge
        "history", "Economics", "books", "writing", "literature", "technology", "Futurology",
    }

    LOW_EFFORT_PATTERNS = re.compile(
        r"^(same|this|based|facts)[.!]*$",
        re.I,
    )

    MEME_PATTERNS = re.compile(
        r"(nobody:|me when|tfw|mfw|\bshitpost\b)",
        re.I,
    )

    # ------------------------------------------------------------------ #
    # Quality Filter Patterns                                              #
    # ------------------------------------------------------------------ #
    BOT_PHRASES = re.compile(
        r"(i am a bot|this action was performed automatically|"
        r"beep boop|please contact the moderators|"
        r"\^this|AutoModerator|I'm a bot)",
        re.IGNORECASE,
    )
    REPEATED_CHARS = re.compile(r"(.)\1{6,}")
    ALL_CAPS_RATIO = 0.60

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _clean(text: str) -> str:

        text = html.unescape(text)

        text = unicodedata.normalize("NFKC", text)

        text = "".join(
            c for c in text
            if unicodedata.category(c)[0] != "C"
            or c in "\n\t"
        )

        # URLs
        text = re.sub(r"http\S+", "", text)

        # Reddit refs
        text = re.sub(r"/u/\w+", "", text)
        text = re.sub(r"/r/\w+", "", text)

        # Markdown links
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Quotes
        text = re.sub(r"(?m)^>\s?.*$", "", text)

        # Inline/code blocks
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.S)

        # Edit signatures
        text = re.sub(r"(?i)\bedit\s*:\s*.*", "", text)

        # normalize line endings
        text = text.replace("\r\n", "\n")

        # collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # collapse spaces/tabs only
        text = re.sub(r"[ \t]+", " ", text)

        # trim lines
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        text = text.strip()

        return text.strip()

    def _quality(text: str) -> bool:
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            return False

        words = text.split()

        if len(words) < 4:
            return False

        if LOW_EFFORT_PATTERNS.match(text.strip()):
            return False

        if MEME_PATTERNS.search(text):
            return False

        if BOT_PHRASES.search(text):
            return False

        if REPEATED_CHARS.search(text):
            return False

        letters = [c for c in text if c.isalpha()]
        if letters:
            caps_ratio = sum(c.isupper() for c in letters) / len(letters)
            if caps_ratio > ALL_CAPS_RATIO:
                return False

        sentences = re.split(r"[.!?]+", text)

        if len(sentences) == 1 and len(words) > 80:
            return False

        # lexical diversity
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.35:
            return False

        return True

    def _fingerprint(text: str) -> str:
        norm = re.sub(r"\s+", " ", text.lower()).strip()

        return hashlib.blake2b(
            norm.encode("utf-8"),
            digest_size=8,
        ).hexdigest()

    def _format_thread(parent_thread: str | None,parent_text: str | None,reply: str,) -> str:

        reply = reply.strip()

        if not parent_text:
            return reply

        if parent_thread:
            if len(parent_thread) > MAX_THREAD_CHARS:
                parent_thread = parent_thread[-MAX_THREAD_CHARS:]

            return (
                f"{parent_thread}\n\n"
                f"Assistant: {reply}"
            )

        return (
            f"User: {parent_text.strip()}\n\n"
            f"Assistant: {reply}"
        )

    # ------------------------------------------------------------------ #
    # Streaming Iterator                                                   #
    # ------------------------------------------------------------------ #
    def _stream_jsonl(path: str, target: int, is_submission: bool):
        """Yield (text, total_skipped) for each accepted record in a JSONL dump."""
        min_score_  = MIN_SCORE_SUBMISSION if is_submission else MIN_SCORE_COMMENT
        boost_score = max(min_score + 2, 6 if not is_submission else 4) # score needed to bypass GOOD_SUBS
        kept = skipped = 0
        comment_cache: dict[str, dict[str, str]] = {}

        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                if kept >= target:
                    break
                try:
                    obj = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    skipped += 1
                    continue

                score = obj.get("score", 0)
                sub   = obj.get("subreddit", "")

                if is_submission:
                    title = obj.get("title", "")
                    body  = obj.get("selftext") or obj.get("body", "")
                    if not body or body in ("[deleted]", "[removed]"):
                        skipped += 1; continue
                    raw_text = f"{title}\n{body}" if title else body
                else:
                    body = obj.get("body", "")
                    if body in ("", "[deleted]", "[removed]"):
                        skipped += 1; continue
                    raw_text = body

                if score < min_score_ or (sub not in GOOD_SUBS and score < boost_score):
                    skipped += 1
                    continue

                cleaned = _clean(raw_text)

                parent_text = None
                parent_thread = None

                if not is_submission:
                    parent_id = obj.get("parent_id", "")

                    if parent_id.startswith("t1_"):
                        pid = parent_id[3:]

                        parent = comment_cache.get(pid)

                        if parent:
                            parent_text = parent["text"]
                            parent_thread = parent["thread"]

                text = _format_thread(parent_thread,parent_text,cleaned,)

                if _quality(text):
                    kept += 1
                    if not is_submission:
                        cid = obj.get("id")

                        if cid:
                            comment_cache[cid] = {"text": cleaned,"thread": text,}

                            if len(comment_cache) > 200_000:
                                comment_cache.pop(next(iter(comment_cache)))

                    yield text, skipped
                else:
                    skipped += 1

    # ------------------------------------------------------------------ #
    # Dedup Writer                                                         #
    # ------------------------------------------------------------------ #
    class _DedupWriter:
        def __init__(self, fh):
            self._fh     = fh
            self._seen   = set()
            self._queue  = deque()
            self.written = 0

        def try_write(self, text: str) -> bool:
            fp = _fingerprint(text)
            if fp in self._seen:
                return False
            self._seen.add(fp)
            self._queue.append(fp)
            if len(self._queue) > DEDUP_CACHE:
                self._seen.discard(self._queue.popleft())
            self._fh.write(text + "\n")
            self.written += 1
            return True

    # ------------------------------------------------------------------ #
    # Resume Logic                                                         #
    # ------------------------------------------------------------------ #
    already = 0
    if resume and os.path.exists(reddit_output):
        with open(reddit_output, "r", encoding="utf-8") as f:
            already = sum(1 for _ in f)
        if already >= target_lines:
            print(f"  Already at {already:,} lines — nothing to do.")
        else:
            print(f"  Resuming from {already:,} lines.")
            already_comments = int(already * COMMENT_RATIO)
            already_subs = already - already_comments

            target_comments = max(0, target_comments - already_comments)
            target_submissions = max(0, target_submissions - already_subs)

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #
    mode = "a" if (resume and already > 0) else "w"
    with open(reddit_output, mode, encoding="utf-8") as out:
        writer = _DedupWriter(out)

        total_chars = 0
        comment_chars = 0
        submission_chars = 0

        if target_comments > 0 and os.path.exists(comments_file):
            for text, skipped in _stream_jsonl(comments_file, target_comments, is_submission=False):
                if writer.try_write(text):
                    chars = len(text)

                    total_chars += chars
                    comment_chars += chars

                    if writer.written % 10_000 == 0:
                        print(
                            f"  [{writer.written:,} comments | "
                            f"{skipped:,} skipped | "
                            f"{comment_chars:,} chars]"
                        )

        comments_written = writer.written

        if target_submissions > 0 and os.path.exists(submissions_file):
            for text, skipped in _stream_jsonl(submissions_file, target_submissions, is_submission=True):
                subs = writer.written - comments_written
                if writer.try_write(text) and subs % 5_000 == 0 and subs > 0:
                    print(f"  [{subs:,} submissions | {skipped:,} skipped]")

    print(f"  Reddit done: {writer.written:,} chunks  "
          f"(comments: {comments_written:,} | submissions: {writer.written - comments_written:,})")

    # ------------------------------------------------------------------ #
    # Read Back                                                            #
    # ------------------------------------------------------------------ #
    chunks = []
    chars  = 0
    with open(reddit_output, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            chunks.append(line)
            chars += len(line)
            if chars >= target_chars:
                break

    print(f"  Read back: {len(chunks):,} chunks  ({chars:,} chars)")
    return chunks

def fetch_wiktionary(
    target_chars: int,
    dump_file: str = 'dumps/enwiktionary-latest-pages-articles.xml.bz2',
    ascii_only: bool = False,
) -> list[str]:

    print(f'\n[6/8] Wiktionary Dump  (target {target_chars:,} chars)')
    if not os.path.exists(dump_file):
        print(f'  Dump not found: {dump_file}')
        return []
    os.makedirs('Dataset/Dataset part backup', exist_ok=True)

    # ------------------------------------------------------------------ #
    # Constants
    # ------------------------------------------------------------------ #

    SKIP_SECTIONS = {
        'references', 'further reading', 'anagrams', 'translations',
        'related terms', 'derived terms', 'descendants', 'see also',
        'synonyms', 'antonyms', 'hypernyms', 'hyponyms', 'meronyms',
        'holonyms', 'coordinate terms', 'alternative forms',
        'declension', 'conjugation', 'inflection', 'usage notes',
        'quotations', 'external links',
    }

    POS_SECTIONS = {
        'noun', 'verb', 'adjective', 'adverb', 'pronoun', 'preposition',
        'conjunction', 'interjection', 'determiner', 'article', 'phrase',
        'proper noun', 'abbreviation', 'initialism', 'numeral', 'symbol',
        'prefix', 'suffix', 'idiom', 'proverb', 'particle',
    }

    BAD_SENSE_LABELS = (
        'slang', 'internet', 'meme', 'vulgar', 'offensive',
        'racial slur', 'ethnic slur', 'childish', 'baby talk',
        'nonstandard', 'eye dialect', 'leet', 'txt', 'texting',
    )

    LEADING_SENSE_LABELS = re.compile(
        r'^(?:slang|archaic|obsolete|dated|rare|now rare|'
        r'transitive|intransitive|countable|uncountable|'
        r'figurative|literal|formal|informal|'
        r'computing|mathematics|physics|biology|chemistry|'
        r'linguistics|semantics|theology|law|legal|'
        r'nautical|military|sports?|meiosis|'
        r'UK|US|Australia|Ireland|Canada|'
        r'sometimes|usually|especially|'
        r'in the plural|plural|singular|'
        r'oath\s+\w+|outside certain phrases|'
        r'sometimes\s+\w+\s+\w+|sometimes\s+\w+)\s+'
        r'|\w+\s+outside certain phrases|outside certain phrases',
        re.IGNORECASE,
    )

    JUNK_CITATION_MARKERS = (
        'quote-book', 'quote-web', 'quote-journal', 'quote-text',
        'cite-book', 'cite-web', 'isbn', 'retrieved from',
    )

    DISPLAY_TEMPLATES = {
        'w', 'l', 'm', 'mention', 'taxlink', 'vern', 'vernacular', 'link', 'pedlink',
    }

    DROP_TEMPLATES = {
        'ipa', 'audio', 'rhymes', 'homophones', 'hyphenation', 'hyph',
        'sense', 'senseid', 'sid', 'rfdef', 'rfquote', 'rfc', 'rfd',
        'attention', 'wikipedia', 'wikispecies', 'commons',
        'quote-book', 'quote-web', 'quote-journal', 'quote-text',
        'cite-book', 'cite-web', 'rq', 'seeCites', 'defdate', 'date',
        'syn', 'ant', 'hyper', 'hypo', 'cot',
        'synonyms', 'antonyms', 'hypernyms', 'hyponyms',
        'uxi', 'uxa', 'alti', 'senseno', '+obj',
    }

    LABEL_TEMPLATES = {'lb', 'label', 'context', 'qualifier', 'q', 'qual', 'gloss'}

    INLINE_SUFFIX_TEMPLATES = {
        's', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'ness', 'ism', 'ist', 'tion', 'ize', 'ise',
    }

    INFLECTION_TAG_MAP = {
        'spast': 'simple past', 'pastp': 'past participle',
        'presp': 'present participle', 'pres': 'present',
        'p': 'plural', 's': 'singular',
        '1': 'first person', '2': 'second person', '3': 'third person',
    }

    DIACRITIC_MAP = {
        'à':'a','á':'a','â':'a','ã':'a','ä':'a','å':'a','æ':'ae','ç':'c',
        'è':'e','é':'e','ê':'e','ë':'e','ì':'i','í':'i','î':'i','ï':'i',
        'ñ':'n','ò':'o','ó':'o','ô':'o','õ':'o','ö':'o','ø':'o',
        'ù':'u','ú':'u','û':'u','ü':'u','ý':'y','ÿ':'y',
        'À':'A','Á':'A','Â':'A','Ã':'A','Ä':'A','Å':'A','Æ':'AE','Ç':'C',
        'È':'E','É':'E','Ê':'E','Ë':'E','Ì':'I','Í':'I','Î':'I','Ï':'I',
        'Ñ':'N','Ò':'O','Ó':'O','Ô':'O','Õ':'O','Ö':'O','Ø':'O',
        'Ù':'U','Ú':'U','Û':'U','Ü':'U','Ý':'Y','ß':'ss',
        'œ':'oe','Œ':'OE','š':'s','Š':'S','ž':'z','Ž':'Z',
        'č':'c','Č':'C','ř':'r','Ř':'R',
    }

    WORD_MAP = {
        'café':'cafe','naïve':'naive','résumé':'resume','Pokémon':'Pokemon',
        'piñata':'pinata','père':'pere','fiancée':'fiance','fiancé':'fiance',
        'née':'nee','rôle':'role','über':'uber','façade':'facade',
        'protégé':'protege','cliché':'cliche','déjà':'deja','naïveté':'naivete',
        'émigré':'emigre','attaché':'attache','exposé':'expose',
        'communiqué':'communique','détente':'detente','soufflé':'souffle',
        'rosé':'rose','touché':'touche','blasé':'blase','purée':'puree',
        'entrée':'entree','sauté':'saute','maté':'mate','pâté':'pate',
        'crêpe':'crepe','chalet':'chalet','rôtisserie':'rotisserie',
        'hôtel':'hotel','naïf':'naif','Zürich':'Zurich','München':'Munich','Köln':'Cologne',
    }

    KNOWN_NON_ASCII = set(DIACRITIC_MAP.keys())

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def normalize_diacritics(text: str) -> str:
        if not text:
            return text
        tokens = text.split()
        if not tokens:
            return text

        def normalize_token(word: str) -> Optional[str]:
            if word in WORD_MAP:
                return WORD_MAP[word]
            lower = word.lower()
            if lower in WORD_MAP:
                r = WORD_MAP[lower]
                return r.capitalize() if word[0].isupper() else r
            for ch in word:
                if not ch.isascii() and ch not in KNOWN_NON_ASCII:
                    return None
            return ''.join(DIACRITIC_MAP.get(ch, ch) for ch in word)

        result = [n for tok in tokens if (n := normalize_token(tok)) is not None]
        return ' '.join(result) if result else text

    def resolve_template(tpl) -> str:
        name = tpl.name.strip().lower()

        if name in ('infl of', 'inflection of'):
            params = [p for p in tpl.params if not p.showkey]
            base = str(params[1].value).strip() if len(params) > 1 else ''
            tags = [str(p.value).strip() for p in params[3:] if str(p.value).strip()]
            readable = re.sub(r' +', ' ', ' '.join(INFLECTION_TAG_MAP.get(t, '') for t in tags)).strip()
            return f'{readable} of {base}'.strip() if base else ''

        if name in DROP_TEMPLATES or name.startswith('quote-') or name.startswith('cite-'):
            return ''

        if name in DISPLAY_TEMPLATES:
            params = [p for p in tpl.params if not p.showkey]
            if len(params) >= 2:
                return str(params[-1].value).strip()
            return str(params[0].value).strip() if params else ''

        if name in ('given name', 'surname'):
            return name

        if name in LABEL_TEMPLATES:
            labels = [
                str(p.value).strip() for p in tpl.params
                if not p.showkey
                and not re.match(r'^[a-z]{2,3}$', str(p.value).strip())
                and not re.match(r'^Q\d+$', str(p.value).strip())
                and len(str(p.value).strip()) >= 2
            ]
            return ' '.join(labels)

        if name == 'taxlink':
            params = [p for p in tpl.params if not p.showkey]
            return str(params[0].value).strip() if params else ''

        if 'non-gloss' in name:
            return ''

        useful = [
            str(p.value).strip() for p in tpl.params
            if not p.showkey
            and not re.match(r'^[a-z]{1,3}$', str(p.value).strip())
            and not re.match(r'^Q\d+$', str(p.value).strip())
            and len(str(p.value).strip()) >= 2
        ]
        return ' '.join(useful[:2])

    def wikitext_to_plain(raw: str) -> str:
        try:
            wikicode = mwparserfromhell.parse(raw)
        except Exception:
            return raw

        # Wikilinks first to avoid node tree corruption
        for link in wikicode.filter_wikilinks():
            try:
                title_str = str(link.title).strip()
                if re.match(r'^(File|Image|Category):', title_str, re.IGNORECASE):
                    wikicode.replace(link, ' ')
                else:
                    text = str(link.text) if link.text else str(link.title)
                    wikicode.replace(link, f' {text.strip()} ')
            except Exception:
                pass

        for tpl in wikicode.filter_templates(recursive=True):
            try:
                name = tpl.name.strip().lower()
                replacement = resolve_template(tpl)
                if name in INLINE_SUFFIX_TEMPLATES or name in DISPLAY_TEMPLATES:
                    wikicode.replace(tpl, replacement)
                else:
                    wikicode.replace(tpl, f' {replacement} ' if replacement else ' ')
            except Exception:
                try:
                    wikicode.replace(tpl, ' ')
                except Exception:
                    pass

        text = wikicode.strip_code()
        text = html.unescape(text)
        text = re.sub(r'\b(\w{2,}) (s|es|ed|ing|er|est|ly|d|ling|lling|tion|tions)\b', r'\1\2', text)
        text = re.sub(r'\b(the|a|an|and|or|to|of|in|on|at|by|as|is)\b(?=[a-z])', r'\1 ', text)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        return text

    def clean_line(line: str) -> str:
        line = html.unescape(line)
        stripped = line.strip()
        if not stripped:
            return ''
        if re.match(r'^\*?\s*IPA', stripped) or '{{IPA' in stripped:
            return ''
        if re.match(r'^\*?\s*(audio|File|Image|thumb|rhymes|hyph|homophones?)', stripped, re.IGNORECASE):
            return ''
        if stripped.startswith('[[Category') or stripped.startswith('C|'):
            return ''
        if 'non-gloss' in stripped.lower():
            return ''

        line = wikitext_to_plain(line)

        line = re.sub(r'<ref[^>]*>.*?</ref>', ' ', line, flags=re.IGNORECASE | re.DOTALL)
        line = re.sub(r'<!--.*?-->', ' ', line, flags=re.DOTALL)
        line = re.sub(r'https?://\S+', ' ', line)
        line = re.sub(r'\[[^\]]*http[^\]]*\]', ' ', line)
        line = re.sub(r"'{2,}", '', line)
        line = re.sub(r'<[^>]+>', ' ', line)
        line = re.sub(r'\|', ' ', line)
        line = re.sub(r'\{\{[^}]*\}\}', ' ', line)
        line = re.sub(r'\[\[[^\]]*\]\]', ' ', line)
        line = re.sub(r'\bQ\d+\b', ' ', line)
        line = re.sub(r'\b\d{9,13}\b', ' ', line)
        line = re.sub(r'\b[a-zA-Z_]+\s*=\s*\S+', ' ', line)
        line = re.sub(r'\(\s*\)', ' ', line)
        line = re.sub(r'\bcontrast\s+([A-Za-z])', r'contrast: \1', line)
        line = re.sub(r'\bsee\s+([A-Z])', r'see: \1', line)
        line = re.sub(r':\s*(?:from|to|into|onto|upon)\s*$', '', line)
        line = re.sub(r'\b\w+#\w+\b', ' ', line)
        line = re.sub(r'\b(ca\.|from|since)[\s\d,\.]+s?\b', '', line, flags=re.IGNORECASE)
        line = re.sub(r'\b\d+(?:st|nd|rd|th)\s+c\.?(?:\s+\d+)?\b', '', line, flags=re.IGNORECASE)
        line = re.sub(r'_\s*', ' ', line)
        line = line.replace('&emsp;', ' ').replace('&nbsp;', ' ')

        line = normalize_diacritics(line)

        if ascii_only:
            line = ' '.join(t for t in line.split() if not re.search(r'[^\x00-\x7F]', t))

        if any(j in line.lower() for j in JUNK_CITATION_MARKERS):
            return ''

        line = re.sub(r' +([,.;:])', r'\1', line)
        line = re.sub(r'\s+(about|of|for|with|by|to|from)\s*[.!?]$', '.', line)
        line = re.sub(r'([,.;:]){2,}', r'\1', line)
        line = re.sub(r'\(\s*[,.;:]?\s*\)', ' ', line)
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        line = re.sub(r' +', ' ', line).strip()
        line = line.strip(' ,;:-')

        if len(line) < 4 or re.match(r'^[^a-zA-Z]*$', line):
            return ''
        return line

    # ------------------------------------------------------------------ #
    # Entry parsing
    # ------------------------------------------------------------------ #

    def parse_section_header(line: str) -> Optional[str]:
        m = re.match(r'^(={2,5})\s*(.+?)\s*\1$', line.strip())
        return m.group(2) if m else None

    def entry_looks_clean(text: str) -> bool:
        if '{{' in text or '}}' in text:
            return False
        if text.count('|') > 2:
            return False
        if not ascii_only:
            stripped = text.replace('\u00b7', '')
            letters = len(re.findall(r'[a-zA-Z]', stripped))
            non_ascii = len(re.findall(r'[^\x00-\x7F]', stripped))
            if letters > 0 and non_ascii / (letters + 1) > 0.6:
                return False
        return True

    def render_entry(title: str, raw_wikitext: str) -> str:

        def find_english_scope(lines: list[str]) -> list[str]:
            eng_start = eng_end = None
            for i, ln in enumerate(lines):
                s = ln.strip()
                if re.match(r'^==\s*English\s*==$', s, re.IGNORECASE):
                    eng_start = i
                elif eng_start is not None and re.match(r'^==[^=].*==$', s):
                    eng_end = i
                    break
            return lines[eng_start:eng_end] if eng_start is not None else lines

        def extract_syllable_title(base: str, scope: list[str]) -> str:
            for ln in scope:
                m = re.search(r'\{\{hyph(?:enation)?\s*\|[^}]*\}\}', ln, re.IGNORECASE)
                if not m:
                    continue
                parts = re.sub(r'^\{\{[^|]+\|', '', m.group(0)).rstrip('}').split('|')
                if parts and re.match(r'^[a-z]{2,3}$', parts[0].strip()):
                    parts = parts[1:]
                syllables = [p.strip() for p in parts if p.strip()]
                if syllables:
                    candidate = normalize_diacritics('\u00b7'.join(syllables))
                    if candidate and len(candidate) <= len(base) * 2 + 3:
                        return candidate
                break
            return base

        def parse_pos_sections(scope: list[str]) -> list[tuple[str, list[tuple[str, list[str]]]]]:
            sections: dict[str, list[tuple[str, list[str]]]] = {}
            current_pos = None
            in_skip = False
            pending_defs: list[tuple[str, list[str]]] = []
            current_defn = None
            current_examples: list[str] = []

            def flush_definition():
                nonlocal current_defn, current_examples
                if current_defn:
                    pending_defs.append((current_defn, current_examples[:]))
                current_defn = None
                current_examples = []

            def flush_pos():
                nonlocal current_pos, pending_defs
                if current_pos and pending_defs:
                    sections.setdefault(current_pos, []).extend(pending_defs)
                current_pos = None
                pending_defs = []

            for ln in scope:
                header = parse_section_header(ln)
                if header is not None:
                    hl = re.sub(r'\s+', ' ', header.lower()).strip()
                    hl_base = re.sub(r'\s*\d+$', '', hl)
                    flush_definition()
                    if hl in SKIP_SECTIONS or hl_base in SKIP_SECTIONS:
                        flush_pos(); in_skip = True
                    elif hl.startswith(('etymology', 'pronunciation')):
                        flush_pos(); in_skip = False; current_pos = None
                    elif hl_base in POS_SECTIONS:
                        flush_pos(); current_pos = hl_base; in_skip = False
                    else:
                        flush_pos(); in_skip = True
                    continue

                if in_skip or current_pos is None:
                    continue

                if ln.startswith('# ') and not ln.startswith('## '):
                    raw = ln[2:].lower()
                    if any(label in raw for label in BAD_SENSE_LABELS):
                        continue
                    flush_definition()
                    clean = clean_line(ln[2:])
                    if clean:
                        clean = LEADING_SENSE_LABELS.sub('', clean).strip()
                    if clean and 3 <= len(clean.split()) and len(clean) <= 400:
                        current_defn = clean
                    continue

                if ln.startswith('#: ') or ln.startswith('#* '):
                    raw = ln[3:].strip().lower()
                    if any(tag in raw for tag in ('{{quote-', '{{rq:', '{{rquote', '{{cite')):
                        continue
                    clean = clean_line(ln[3:])
                    if not clean or len(clean) <= 8:
                        continue
                    if '{{' in clean or '}}' in clean or clean.startswith('quote-'):
                        continue
                    if re.match(r'^\d{4},', clean):
                        continue
                    if re.search(r'\b(January|February|March|April|May|June|July|August|'
                                 r'September|October|November|December)\b', clean):
                        if re.search(r'\b\d{4}\b', clean):
                            continue
                    if re.search(r'\b(p\.|pp\.|vol\.|ibid|op\.cit)\b', clean, re.IGNORECASE):
                        continue
                    words = clean.split()
                    if len(words) <= 2:
                        continue
                    if len(words) <= 4 and clean.lower() == clean and not clean.endswith('.'):
                        continue
                    if re.fullmatch(r'[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+', clean):
                        continue
                    if len(clean) <= 400:
                        current_examples.append(clean)

            flush_definition()
            flush_pos()

            if ascii_only:
                sections = {
                    pos: [(d, [e for e in exs if not re.search(r'[^\x00-\x7F]', e)])
                          for d, exs in defs if not re.search(r'[^\x00-\x7F]', d)]
                    for pos, defs in sections.items()
                }
                sections = {k: v for k, v in sections.items() if v}

            return list(sections.items())

        lines = raw_wikitext.replace('\r\n', '\n').splitlines()
        scope = find_english_scope(lines)
        display_title = extract_syllable_title(normalize_diacritics(title) or title, scope)
        sections = parse_pos_sections(scope)

        if not sections or not any(defs for _, defs in sections):
            return ''

        blocks = [display_title]
        for pos_label, defs in sections:
            blocks.extend(['', pos_label, ''])
            for idx, (defn, examples) in enumerate(defs, 1):
                blocks.append(f'{idx}. {defn.rstrip(".:")+":"}' if examples else f'{idx}. {defn}')
                for ex in examples:
                    ex = ex.strip('"\'')
                    if '#' not in ex and len(ex.split()) > 1:
                        blocks.append(f'"{ex}"')

        rendered = '\n'.join(blocks)
        return rendered if entry_looks_clean(rendered) else ''

    def normalize_title(raw: str) -> str:
        title = normalize_diacritics(raw.strip())
        if not title:
            return ''
        pattern = r"^[A-Za-z0-9 _'\-]+$" if ascii_only else r"^[\w\s'\-\.]+$"
        if not re.match(pattern, title, 0 if ascii_only else re.UNICODE):
            return ''
        if ':' in title and not title[0].isupper():
            return ''
        return title

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    chunks: list[str] = []
    total_chars = entries_kept = entries_skipped = 0
    inside_page = inside_text = False
    title = ''
    text_lines: list[str] = []

    with bz2.open(dump_file, 'rt', encoding='utf-8', errors='ignore') as fh:
        for raw_line in fh:
            if '<page>' in raw_line:
                inside_page = True
                inside_text = False
                title = ''
                text_lines = []
                continue

            if '</page>' in raw_line:
                if title and text_lines:
                    raw_text = ''.join(text_lines).replace('\r\n', '\n')
                    if '#REDIRECT' not in raw_text.upper() and len(raw_text) < 200_000:
                        is_english = re.search(r'==\s*English\s*==', raw_text) is not None
                        has_pos = re.search(
                            r'===?\s*(Noun|Verb|Adjective|Adverb|Pronoun|Preposition'
                            r'|Conjunction|Interjection|Determiner|Article)\s*===?',
                            raw_text, re.IGNORECASE,
                        ) is not None
                        if title.startswith('Reconstruction:'):
                            entries_skipped += 1
                        elif is_english and has_pos:
                            entry = render_entry(title, raw_text)
                            if entry and len(entry) > 20:
                                chunks.append(entry)
                                total_chars += len(entry)
                                entries_kept += 1
                                if entries_kept % 5000 == 0:
                                    print(f'  [{entries_kept:,} entries | {entries_skipped:,} skipped]  '
                                          f'{total_chars:,}/{target_chars:,} chars')
                                    gc.collect()
                                if total_chars >= target_chars:
                                    break
                        else:
                            entries_skipped += 1
                inside_page = inside_text = False
                text_lines = []
                continue

            if not inside_page:
                continue

            if '<title>' in raw_line and '</title>' in raw_line:
                title = normalize_title(re.sub(r'.*<title>(.*?)</title>.*', r'\1', raw_line).strip())
                continue

            if '<text' in raw_line:
                inside_text = True
                raw_line = raw_line.split('>', 1)[-1]

            if inside_text:
                if '</text>' in raw_line:
                    raw_line = raw_line.split('</text>', 1)[0]
                    inside_text = False
                if len(text_lines) < 6000:
                    text_lines.append(raw_line)

    print(f'  Wiktionary done: {entries_kept:,} entries  '
          f'({total_chars:,} chars)  ({entries_skipped:,} non-English skipped)')
    return chunks

# =============================================================================
#  Phase 8: Tool training data
# =============================================================================
#
# Generates Toolformer-style training examples in the format:
#
#   [TOOL:name|argument][RESULT:result text]
#
# Each example is a short self-contained passage:
#   <context sentence(s)>
#   [TOOL:name|query][RESULT:result]
#   <continuation sentence(s)>
#
# The continuation after RESULT matters: it teaches the model to USE the
# result rather than ignore it (key Toolformer finding).
#
# Executors are imported from tool_definitions so training and inference
# always produce identical results for the same input.

from MiniGPT.tool_definitions import (  # noqa: E402
    _tool_exec_calc,
    _tool_exec_convert,
    _tool_exec_date,
    _tool_exec_search,
    TOOL_MAX_RESULT,
    TOOL_REGISTRY,
    TOOL_WEIGHTS,
)

_SEARCH_SEED_TOPICS = (
    LGBTQ_ARTICLES[:20] + LEFT_ARTICLES[:20] +
    DIVERSE_ARTICLES[:20] + ACADEMIC_ARTICLES[:20]
)

# ---------------------------------------------------------------------------
#  Template banks
#  (context_template, query_template, continuation_template)
#  continuation MUST contain [TOOL:name|...][RESULT:{result}].
# ---------------------------------------------------------------------------

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
]

# ---------------------------------------------------------------------------
#  Generators
# ---------------------------------------------------------------------------

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
        examples.append(f"{ctx_t.format(a=a, b=b)}\n{cont_t.format(a=a, b=b, result=result)}")
    return examples


def _generate_convert_examples(n: int) -> List[str]:
    examples = []
    for _ in range(n):
        ctx_t, q_t, cont_t = random.choice(_CONVERT_TEMPLATES)
        a = int(round(random.uniform(1, 500))) if random.random() < 0.6 \
            else round(random.uniform(1, 500), 1)
        result = _tool_exec_convert(q_t.format(a=a))
        if "unknown" in result or "error" in result:
            continue
        examples.append(f"{ctx_t.format(a=a)}\n{cont_t.format(a=a, result=result)}")
    return examples


def _generate_date_examples(n: int) -> List[str]:
    import datetime as _dt
    examples = []
    years = list(range(1990, 2025))
    pairs = []
    for _ in range(40):
        d = _dt.date(random.randint(2000, 2023), random.randint(1, 12), random.randint(1, 28))
        pairs.append((str(d), str(d + _dt.timedelta(days=random.randint(10, 500)))))
    for _ in range(n):
        ctx_t, q_t, cont_t = random.choice(_DATE_TEMPLATES)
        if "{y}" in q_t:
            y = random.choice(years)
            result = _tool_exec_date(q_t.format(y=y))
            ctx, cont = ctx_t.format(y=y), cont_t.format(y=y, result=result)
        elif "{d1}" in q_t:
            d1, d2 = random.choice(pairs)
            result = _tool_exec_date(q_t.format(d1=d1, d2=d2))
            ctx, cont = ctx_t.format(d1=d1, d2=d2), cont_t.format(d1=d1, d2=d2, result=result)
        else:
            result = _tool_exec_date(q_t)
            ctx, cont = ctx_t, cont_t.format(result=result)
        if result == "unknown date query":
            continue
        examples.append(f"{ctx}\n{cont}")
    return examples


def _generate_search_examples(n: int, live: bool = True) -> List[str]:
    examples = []
    topics = random.sample(_SEARCH_SEED_TOPICS, min(n * 3, len(_SEARCH_SEED_TOPICS)))
    for topic in topics:
        if len(examples) >= n:
            break
        result = _tool_exec_search(topic) if live else f"{topic} is a notable subject."
        if result == "no result":
            continue
        if random.random() < 0.5:
            ctx_t, _, cont_t = random.choice(_SEARCH_TEMPLATES)
            examples.append(f"{ctx_t.format(q=topic)}\n"
                            f"{cont_t.format(q=topic, result=result[:TOOL_MAX_RESULT])}")
        else:
            ctx_t, _, cont_t = random.choice(_LOOKUP_TEMPLATES)
            examples.append(f"{ctx_t.format(e=topic)}\n"
                            f"{cont_t.format(e=topic, result=result[:TOOL_MAX_RESULT])}")
        print(f"    search/lookup OK: {topic[:55]}")
    return examples


def fetch_tool_training(
    target_chars: int,
    live_search:  bool  = True,
    weights:      Optional[dict] = None,
) -> List[str]:
    """
    Phase 8: generate Toolformer-style tool-training examples.

    Produces a mix of calc, convert, date, and search/lookup examples
    in the ``[TOOL:name|arg][RESULT:...]`` format.

    Also calls ``ensure_tool_vocab`` on a sentinel dict to print a reminder
    about which characters must be present in the model's vocabulary.

    :param target_chars: Total character budget for this phase.
    :param live_search: Fetch real Wikipedia snippets for search/lookup
        examples.  Set ``False`` to use placeholders (faster, lower quality).
    :param weights: ``{tool_name: fraction}`` override.  Defaults to
        ``TOOL_WEIGHTS`` from ``tool_definitions``.  Must sum to ~1.0.
    """

    TOOL_CHARS: frozenset = frozenset("[]:|")

    def ensure_tool_vocab(char2idx: Dict[str, int], silent: bool = False) -> Dict[str, int]:
        """
        Guarantee all tool delimiter characters are in the vocabulary.

        Call this after building char2idx from your corpus, before constructing
        the NeuralNetwork.  Missing chars are appended so existing indices stay
        stable.

        :param silent: Suppress the print when chars are added. Pass ``True``
            during training -- model.py calls this automatically when tool
            patterns are detected in the corpus.

        Example
        -------
            chars    = sorted(set(corpus_text))
            char2idx = {c: i for i, c in enumerate(chars)}
            char2idx = ensure_tool_vocab(char2idx)   # adds [ ] : | if absent
            nn = NeuralNetwork(vocab_size=len(char2idx), ...)
        """
        missing = TOOL_CHARS - set(char2idx)
        if missing:
            next_idx = max(char2idx.values()) + 1
            for ch in sorted(missing):
                char2idx[ch] = next_idx
                next_idx += 1
            if not silent:
                print(f"[tool vocab] Added {len(missing)} missing chars: "
                      f"{sorted(missing)}  (new size: {len(char2idx)})")
        return char2idx

    print(f"\n[8/8] Tool training  (target {target_chars:,} chars, "
          f"live_search={'yes' if live_search else 'no'})")

    # Remind the caller which vocab chars are required at inference.
    # We pass a dummy dict so ensure_tool_vocab can report missing chars
    # without touching any real model state.
    _dummy = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    _extended = ensure_tool_vocab(_dummy)
    added = set(_extended) - set(_dummy)
    if added:
        print(f"  NOTE: train your model on a vocab that includes: {sorted(added)}")
        print(f"        or call ensure_tool_vocab(char2idx) before building the model.")

    w = weights if weights is not None else TOOL_WEIGHTS
    avg_chars = 220
    n_total   = max(50, target_chars // avg_chars)
    n_calc    = int(n_total * w.get("calc",    0.30))
    n_convert = int(n_total * w.get("convert", 0.20))
    n_date    = int(n_total * w.get("date",    0.15))
    n_search  = n_total - n_calc - n_convert - n_date

    print(f"  Generating ~{n_total} examples: "
          f"calc={n_calc} convert={n_convert} date={n_date} search/lookup={n_search}")

    all_examples: List[str] = []
    print(f"  calc ({n_calc})...")
    all_examples.extend(_generate_calc_examples(n_calc))
    print(f"  convert ({n_convert})...")
    all_examples.extend(_generate_convert_examples(n_convert))
    print(f"  date ({n_date})...")
    all_examples.extend(_generate_date_examples(n_date))
    print(f"  search/lookup ({n_search}, live={live_search})...")
    all_examples.extend(_generate_search_examples(n_search, live=live_search))

    random.shuffle(all_examples)
    chunks, total = [], 0
    for ex in all_examples:
        if total >= target_chars:
            break
        chunks.append(ex)
        total += len(ex)

    tool_counts: Counter = Counter()
    for ex in chunks:
        for m in re.finditer(r'\[TOOL:(\w+)\|', ex):
            tool_counts[m.group(1)] += 1

    print(f"  Tool training done: {len(chunks):,} examples  ({total:,} chars)")
    print(f"  Tool call counts: " + "  ".join(f"{k}={v}" for k, v in tool_counts.items()))
    return chunks


# =============================================================================
#  Checkpoint saving
# =============================================================================

def _save_checkpoint(chunks: List[str], filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))


# =============================================================================
#  Main build
# =============================================================================

def build_dataset(
    target_chars:          int  = 200_000_000,
    output_file:           str  = "Dataset.txt",
    no_gutenberg:          bool = False,
    no_wikipedia:          bool = False,
    no_simple_wiki:        bool = False,
    no_wikiquote:          bool = False,
    no_wiktionary:         bool = False,
    no_wikibooks:          bool = False,
    no_reddit:             bool = False,
    no_tools:              bool = False,
    phase:                 Optional[int] = None,
    reddit_comments:       str = "Dumps/comments/RC_2016-01.jsonl",
    reddit_submissions:    str = "Dumps/submissions/RS_2016-01.jsonl",
    reddit_output:         str = "Dataset/Dataset part backup/phase_7_reddit.txt",
    reddit_no_submissions: bool = False,
    reddit_resume:         bool = False,
    reddit_min_score:      int  = 4,
    tools_live_search:     bool = True,
    tools_calc_weight:     float = 0.30,
    tools_convert_weight:  float = 0.20,
    tools_date_weight:     float = 0.15,
    tools_search_weight:   float = 0.35,
) -> None:
    phase_names = {1: "Gutenberg", 2: "Wikipedia", 3: "Simple Wikipedia",
                   4: "Wikiquote", 5: "Wikibooks", 6: "Wiktionary",
                   7: "Reddit", 8: "Tool training"}
    print("=" * 65)
    print("  MiniGPT dataset builder  v5.0")
    print(f"  Target: {target_chars:,} chars  →  {output_file}")
    print(f"  Sources: Gutenberg · Wikipedia · Simple Wiki · Wikiquote")
    print(f"           Wikibooks · Wiktionary · Reddit · Tools")
    if phase is not None:
        print(f"  Phase : {phase} — {phase_names.get(phase, '?')} only")
    print("=" * 65)
    if phase is not None:
        print(f"Mode   : SINGLE PHASE {phase} ({phase_names.get(phase, '?')})")
        no_gutenberg   = (phase != 1)
        no_wikipedia   = (phase != 2)
        no_simple_wiki = (phase != 3)
        no_wikiquote   = (phase != 4)
        no_wikibooks   = (phase != 5)
        no_wiktionary  = (phase != 6)
        no_reddit      = (phase != 7)
        no_tools       = (phase != 8)
    print("=" * 65)

    chunks = []

    if not no_gutenberg:
        c = fetch_gutenberg(int(target_chars * 0.18))
        chunks.extend(c)
        _save_checkpoint(c, "Dataset/Dataset part backup/phase_1_gutenberg.txt")
        print("  ✓ Gutenberg checkpoint saved")

    if not no_wikipedia:
        c = fetch_wikipedia(int(target_chars * 0.28))
        chunks.extend(c)
        _save_checkpoint(c, "Dataset/Dataset part backup/phase_2_wikipedia.txt")
        print("  ✓ Wikipedia checkpoint saved")

    if not no_simple_wiki:
        c = fetch_simple_wikipedia(int(target_chars * 0.10))
        chunks.extend(c)
        _save_checkpoint(c, "Dataset/Dataset part backup/phase_3_simple_wikipedia.txt")
        print("  ✓ Simple Wikipedia checkpoint saved")

    if not no_wikiquote:
        c = fetch_wikiquote(int(target_chars * 0.05))
        chunks.extend(c)
        _save_checkpoint(c, "Dataset/Dataset part backup/phase_4_wikiquote.txt")
        print("  ✓ Wikiquote checkpoint saved")

    if not no_wikibooks:
        c = fetch_wikibooks(int(target_chars * 0.07))
        chunks.extend(c)
        _save_checkpoint(c, "Dataset/Dataset part backup/phase_5_wikibooks.txt")
        print("  ✓ Wikibooks checkpoint saved")

    if not no_wiktionary:
        c = fetch_wiktionary(int(target_chars * 0.12))
        chunks.extend(c)
        _save_checkpoint(c, "Dataset/Dataset part backup/phase_6_wiktionary.txt")
        print("  ✓ Wiktionary checkpoint saved")

    if not no_reddit:
        c = fetch_reddit(
            target_chars     = int(target_chars * 0.12),
            comments_file    = reddit_comments,
            submissions_file = reddit_submissions,
            reddit_output    = reddit_output,
            no_submissions   = reddit_no_submissions,
            resume           = reddit_resume,
            min_score        = reddit_min_score,
        )
        if c:
            chunks.extend(c)
            _save_checkpoint(c, "Dataset/Dataset part backup/phase_7_reddit.txt")
            print("  ✓ Reddit checkpoint saved")

    if not no_tools:
        c = fetch_tool_training(
            target_chars = int(target_chars * 0.08),
            live_search  = tools_live_search,
            weights      = {
                "calc":    tools_calc_weight,
                "convert": tools_convert_weight,
                "date":    tools_date_weight,
                "search":  tools_search_weight,
            },
        )
        if c:
            chunks.extend(c)
            _save_checkpoint(c, "Dataset/Dataset part backup/phase_8_tools.txt")
            print("  ✓ Tool training checkpoint saved")

    if not chunks:
        print("\nERROR: no text collected. Check internet connection.")
        sys.exit(1)

    random.shuffle(chunks)

    full_text = "\n\n".join(chunks)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}',  ' ',    full_text)
    full_text = full_text.strip()

    if len(full_text) > target_chars:
        cut       = full_text.rfind(' ', 0, target_chars)
        full_text = full_text[:cut] if cut > 0 else full_text[:target_chars]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    size_mb    = os.path.getsize(output_file) / 1024 / 1024
    text_lower = full_text.lower()
    audits = [
        ("LGBTQ+",         ["stonewall", "lgbtq", "transgender", "queer", "gay rights"]),
        ("Socialist",      ["socialism", "communism", "marxism", "working class", "proletariat"]),
        ("Civil rights",   ["civil rights", "segregation", "discrimination"]),
        ("Science",        ["chemistry", "physics", "biology", "evolution"]),
        ("Women",          ["feminism", "suffrage", "women's rights"]),
        ("Conversational", ["actually", "honestly", "i think", "in my opinion", "basically"]),
        ("Tool calls",     ["[tool:calc|", "[tool:search|", "[tool:convert|",
                            "[tool:date|",  "[tool:lookup|"]),
    ]

    print("\n" + "=" * 65)
    print("Dataset complete")
    print("=" * 65)
    print(f"  File      : {output_file}  ({size_mb:.1f} MB)")
    print(f"  Chars     : {len(full_text):,}")
    print(f"  Words     : {len(full_text.split()):,}")
    print(f"  Documents : {len(chunks):,}")
    print(f"  Vocab     : {len(set(full_text))} unique chars")
    print("\n  Content audit:")
    for label, keywords in audits:
        hits = sum(text_lower.count(kw) for kw in keywords)
        bar  = "█" * min(hits // 20, 30)
        print(f"    {label:<14} {bar}  ({hits:,} hits)")
    print(f"\nReady:")
    print(f"  python miniGPT/cli.py --train {output_file} --simple_vocab ...")
    print("=" * 65)


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a diverse local dataset for miniGPT.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--target_chars",  type=int, default=200_000_000,
                        help="Target character count (default: 200,000,000 ≈ 200 MB)")
    parser.add_argument("--output",        type=str, default="diverse_dataset.txt",
                        help="Output filename (default: diverse_dataset.txt)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no_gutenberg",      action="store_true")
    parser.add_argument("--no_wikipedia",      action="store_true")
    parser.add_argument("--no_simple_wiki",    action="store_true")
    parser.add_argument("--no_wikiquote",      action="store_true")
    parser.add_argument("--no_wikibooks",      action="store_true")
    parser.add_argument("--no_wiktionary",     action="store_true")
    parser.add_argument("--no_reddit",         action="store_true",
                        help="Skip Reddit phase (requires local .jsonl dumps)")
    parser.add_argument("--no_tools",          action="store_true",
                        help="Skip tool training phase (phase 8)")
    parser.add_argument("--reddit_comments",    type=str, default="Dumps/comments/RC_2016-01.jsonl")
    parser.add_argument("--reddit_submissions", type=str, default="Dumps/submissions/RS_2016-01.jsonl")
    parser.add_argument("--reddit_output",      type=str, default="Dataset/Dataset part backup/phase_7_reddit.txt")
    parser.add_argument("--reddit_no_submissions", action="store_true")
    parser.add_argument("--reddit_resume",     action="store_true")
    parser.add_argument("--reddit_min_score",  type=int, default=4)
    parser.add_argument("--tools_no_live_search", action="store_true",
                        help="Generate tool examples with placeholder results (faster, "
                             "skips Wikipedia fetches for search/lookup examples)")
    parser.add_argument("--tools_calc_weight",    type=float, default=0.30,
                        help="Fraction of tool examples that are calc (default 0.30)")
    parser.add_argument("--tools_convert_weight", type=float, default=0.20,
                        help="Fraction of tool examples that are unit conversions (default 0.20)")
    parser.add_argument("--tools_date_weight",    type=float, default=0.15,
                        help="Fraction of tool examples that are date queries (default 0.15)")
    parser.add_argument("--tools_search_weight",  type=float, default=0.35,
                        help="Fraction of tool examples that are search/lookup (default 0.35)")
    parser.add_argument("--phase", type=int, choices=[1,2,3,4,5,6,7,8], default=None,
                        help=(
                            "Run only one phase and save its checkpoint:\n"
                            "  1=Gutenberg  2=Wikipedia  3=Simple Wikipedia\n"
                            "  4=Wikiquote  5=Wikibooks  6=Wiktionary\n"
                            "  7=Reddit     8=Tool training\n"
                            "Overrides all --no_* flags."
                        ))
    args = parser.parse_args()

    random.seed(args.seed)
    build_dataset(
        target_chars          = args.target_chars,
        output_file           = args.output,
        no_gutenberg          = args.no_gutenberg,
        no_wikipedia          = args.no_wikipedia,
        no_simple_wiki        = args.no_simple_wiki,
        no_wikiquote          = args.no_wikiquote,
        no_wikibooks          = args.no_wikibooks,
        no_wiktionary         = args.no_wiktionary,
        no_reddit             = args.no_reddit,
        no_tools              = args.no_tools,
        phase                 = args.phase,
        reddit_comments       = args.reddit_comments,
        reddit_submissions    = args.reddit_submissions,
        reddit_output         = args.reddit_output,
        reddit_no_submissions = args.reddit_no_submissions,
        reddit_resume         = args.reddit_resume,
        reddit_min_score      = args.reddit_min_score,
        tools_live_search     = not args.tools_no_live_search,
        tools_calc_weight     = args.tools_calc_weight,
        tools_convert_weight  = args.tools_convert_weight,
        tools_date_weight     = args.tools_date_weight,
        tools_search_weight   = args.tools_search_weight,
    )