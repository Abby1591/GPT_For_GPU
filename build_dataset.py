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

Source split (default 2.2M chars):
    35% Gutenberg       -- long-form prose, vocabulary breadth
    20% Wikipedia LGBTQ+/Left (guaranteed blocks)
    20% Wikipedia Diverse/Academic (curated)
    15% Simple Wikipedia
    10% Wikiquote

Usage
-----
    python build_dataset.py
    python build_dataset.py --target_chars 5000000 --output big_dataset.txt
    python build_dataset.py --no_gutenberg
    python build_dataset.py --no_wikipedia

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
#  Gutenberg books
#  Chosen for: vocabulary breadth, prose quality, left-wing/radical thought,
#  diverse authorship, and rich descriptive language.
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
    (768,   "Wuthering Heights"),
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
#  LGBTQ+ Wikipedia articles
#  Fetched FIRST, no character cap, 100% guaranteed in every dataset.
#  Covers: history, figures, rights, culture, theory, global perspectives.
# =============================================================================

LGBTQ_ARTICLES = [
    # --- Foundational history ---
    "Stonewall riots",
    "LGBT history",
    "Timeline of LGBT history",
    "Homosexuality in ancient Greece",
    "Homosexuality in ancient Rome",
    "Two-spirit",
    "Hijra (South Asia)",
    "History of homosexuality",
    "History of transgender people",
    "Transgender history",
    "Sodomy laws",
    "Decriminalization of homosexuality",
    "Bowers v. Hardwick",
    "Lawrence v. Texas",
    "Obergefell v. Hodges",
    # --- Key figures ---
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
    "Renee Richards",
    "Josephine Baker",
    "Frida Kahlo",
    "Langston Hughes",
    "Adrienne Rich",
    "Kate Millett",
    "Leslie Feinberg",
    "Quentin Crisp",
    "Del Martin and Phyllis Lyon",
    "Frank Kameny",
    "Harry Hay",
    "Barbara Gittings",
    "Edie Windsor",
    "Matthew Shepard",
    "Brandon Teena",
    "Pulse nightclub shooting",
    # --- Rights and law ---
    "LGBT rights in the United States",
    "LGBT rights by country or territory",
    "Same-sex marriage",
    "Same-sex marriage in the United States",
    "Transgender rights in the United States",
    "LGBT adoption",
    "Don't ask, don't tell",
    "Employment Non-Discrimination Act",
    "Hate crime laws in the United States",
    "Legal recognition of non-binary gender",
    "Intersex human rights",
    "LGBT rights in Russia",
    "LGBT rights in Uganda",
    "LGBT rights in China",
    "LGBT rights in India",
    "Section 28",
    # --- Culture and community ---
    "Pride parade",
    "Gay liberation",
    "Lesbian feminism",
    "Queer theory",
    "Gender studies",
    "LGBT community",
    "Bisexuality",
    "Non-binary gender",
    "Genderqueer",
    "Asexuality",
    "Pansexuality",
    "Drag (gender expression)",
    "Ballroom culture",
    "Coming out",
    "LGBT culture",
    "Pink triangle",
    "Rainbow flag (LGBT)",
    "Camp (style)",
    # --- Health and crisis ---
    "HIV/AIDS",
    "AIDS crisis",
    "ACT UP",
    "Gay Men's Health Crisis",
    "Ryan White",
    # --- Organizations ---
    "Human Rights Campaign",
    "PFLAG",
    "Lambda Legal",
    "GLSEN",
    "National LGBTQ Task Force",
    "Gay-Straight Alliance",
    # --- Literature and arts ---
    "Giovanni's Room",
    "The Well of Loneliness",
    "Angels in America",
    "The Normal Heart",
    "Fun Home",
    "Maurice (novel)",
    "Brokeback Mountain",
    "Paris Is Burning (film)",
    "The L Word",
    "Queer as Folk",
]


# =============================================================================
#  Left-wing / socialist / communist / anarchist articles
#  Fetched as a guaranteed second pass -- always in every dataset.
# =============================================================================

LEFT_ARTICLES = [
    # --- Core theory ---
    "Marxism",
    "Socialism",
    "Communism",
    "Anarchism",
    "Marxism–Leninism",
    "Trotskyism",
    "Libertarian socialism",
    "Democratic socialism",
    "Social democracy",
    "Anarcho-communism",
    "Syndicalism",
    "Revolutionary socialism",
    "Leninism",
    "Stalinism",
    "Maoism",
    "Feminism",
    "Socialist feminism",
    "Marxist feminism",
    "Intersectionality",
    "Critical theory",
    "Frankfurt School",
    "Hegelian dialectics",
    "Historical materialism",
    "Dialectical materialism",
    "Class consciousness",
    "False consciousness",
    "Alienation (Marx)",
    "Mode of production",
    "Means of production",
    "Base and superstructure",
    "Surplus value",
    "Capital (Marx)",
    # --- People ---
    "Karl Marx",
    "Friedrich Engels",
    "Vladimir Lenin",
    "Leon Trotsky",
    "Rosa Luxemburg",
    "Emma Goldman",
    "Peter Kropotkin",
    "Mikhail Bakunin",
    "Antonio Gramsci",
    "Georg Wilhelm Friedrich Hegel",
    "Friedrich Engels",
    "Eugene V. Debs",
    "Mother Jones",
    "Big Bill Haywood",
    "Alexandra Kollontai",
    "Che Guevara",
    "Fidel Castro",
    "Ho Chi Minh",
    "Mao Zedong",
    "Salvador Allende",
    "Hugo Chavez",
    "Angela Davis",
    "Huey P. Newton",
    "Fred Hampton",
    "Claudia Jones",
    "C. L. R. James",
    "Paul Robeson",
    "Howard Zinn",
    "Noam Chomsky",
    "Herbert Marcuse",
    "Jean-Paul Sartre",
    "Simone de Beauvoir",
    "Frantz Fanon",
    "Walter Rodney",
    "bell hooks",
    # --- Movements and events ---
    "Russian Revolution",
    "October Revolution",
    "Paris Commune",
    "Spanish Civil War",
    "Cuban Revolution",
    "Chinese Revolution",
    "Haitian Revolution",
    "Labour movement",
    "Trade union",
    "General strike",
    "Industrial Workers of the World",
    "International Workers' Day",
    "Black Panther Party",
    "Young Lords",
    "American Indian Movement",
    "Chicano movement",
    "Anti-capitalism",
    "Anti-imperialism",
    "Decolonization",
    "Third-worldism",
    "Non-Aligned Movement",
    "Zapatista Army of National Liberation",
    "Occupy movement",
    "Anti-globalization movement",
    "Socialist International",
    "Communist International",
    # --- Economic concepts ---
    "Capitalism",
    "Neoliberalism",
    "Imperialism",
    "Colonialism",
    "Wealth inequality",
    "Poverty",
    "Universal basic income",
    "Welfare state",
    "Mixed economy",
    "Planned economy",
    "Market socialism",
    "Worker cooperative",
    "Common ownership",
    "Nationalization",
    "Privatization",
    # --- States and experiments ---
    "Soviet Union",
    "Cuba",
    "Yugoslavia under Tito",
    "Allende's Chile",
    "Bolivarian Revolution",
]


# =============================================================================
#  Diverse curated Wikipedia articles
#  Civil rights, science, environment, women, global history, disability.
# =============================================================================

DIVERSE_ARTICLES = [
    # --- Civil rights ---
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
    "Tulsa race massacre",
    "Juneteenth",
    "Jim Crow laws",
    "Slavery in the United States",
    "Underground Railroad",
    "Reconstruction era",
    "Redlining",
    "Loving v. Virginia",
    "Trayvon Martin",
    "George Floyd protests",
    # --- Women's rights ---
    "Women's suffrage",
    "Mary Wollstonecraft",
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
    "Equal Rights Amendment",
    "Title IX",
    # --- Scientists (diverse) ---
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
    "Nikola Tesla",
    "Alan Turing",
    "Subrahmanyan Chandrasekhar",
    # --- Global and anti-colonial ---
    "Mahatma Gandhi",
    "Indian independence movement",
    "Atlantic slave trade",
    "Colonialism",
    "Desmond Tutu",
    "Chinua Achebe",
    "Kwame Nkrumah",
    "Jawaharlal Nehru",
    "Universal Declaration of Human Rights",
    "Amnesty International",
    "Rwandan genocide",
    "Armenian genocide",
    "Indigenous peoples",
    # --- Environment / climate ---
    "Climate change",
    "Global warming",
    "Paris Agreement",
    "Greta Thunberg",
    "Renewable energy",
    "Environmental justice",
    "Biodiversity",
    "Amazon rainforest",
    "Climate justice",
    # --- Disability / neurodiversity ---
    "Disability rights movement",
    "Neurodiversity",
    "Mental health",
    "Deinstitutionalisation",
    # --- History ---
    "World War II",
    "Holocaust",
    "Cold War",
    "French Revolution",
    "American Revolution",
    "Democracy",
    "Human rights",
    "United Nations",
    "Vietnam War",
    "Korean War",
    "Iraq War",
    "Afghanistan War",
    "Nuclear weapons",
    "Nuclear disarmament",
]


# =============================================================================
#  Academic / knowledge Wikipedia articles
#  Chemistry, physics, biology, maths, history of science, philosophy of science.
#  These give the model factual, precise language and domain vocabulary.
# =============================================================================

ACADEMIC_ARTICLES = [
    # --- Chemistry ---
    "Chemistry",
    "Atom",
    "Chemical element",
    "Periodic table",
    "Chemical bond",
    "Covalent bond",
    "Ionic bonding",
    "Molecule",
    "Chemical reaction",
    "Acid–base reaction",
    "Oxidation state",
    "Organic chemistry",
    "Polymer",
    "Protein",
    "DNA",
    "RNA",
    "Enzyme",
    "Photosynthesis",
    "Cellular respiration",
    "Thermodynamics",
    "Entropy",
    "Gibbs free energy",
    "Electrolysis",
    "Radioactive decay",
    "Nuclear fission",
    "Nuclear fusion",
    # --- Physics ---
    "Physics",
    "Classical mechanics",
    "Quantum mechanics",
    "Special relativity",
    "General relativity",
    "Electromagnetism",
    "Wave–particle duality",
    "Uncertainty principle",
    "Standard Model",
    "Black hole",
    "Big Bang",
    "Dark matter",
    "Gravity",
    "Thermodynamics",
    "Entropy",
    "Speed of light",
    "Electromagnetic spectrum",
    # --- Biology ---
    "Biology",
    "Cell (biology)",
    "Evolution",
    "Natural selection",
    "Genetics",
    "Gene",
    "Chromosome",
    "Mutation",
    "Ecology",
    "Ecosystem",
    "Food chain",
    "Nervous system",
    "Immune system",
    "Virus",
    "Bacteria",
    "Antibiotic resistance",
    "CRISPR",
    "Stem cell",
    # --- Mathematics ---
    "Mathematics",
    "Calculus",
    "Linear algebra",
    "Statistics",
    "Probability",
    "Prime number",
    "Topology",
    "Set theory",
    "Mathematical proof",
    "Cryptography",
    # --- Earth and space ---
    "Earth",
    "Plate tectonics",
    "Atmosphere of Earth",
    "Ocean",
    "Solar System",
    "Galaxy",
    "Milky Way",
    "Exoplanet",
    "Asteroid",
    "Comet",
    # --- History of science ---
    "Scientific revolution",
    "Age of Enlightenment",
    "History of chemistry",
    "History of physics",
    "History of biology",
    "History of mathematics",
    "Copernican heliocentrism",
    "Isaac Newton",
    "Galileo Galilei",
    "Johannes Kepler",
    "Charles Darwin",
    # --- History general ---
    "Ancient Egypt",
    "Ancient Greece",
    "Ancient Rome",
    "Byzantine Empire",
    "Islamic Golden Age",
    "Renaissance",
    "Industrial Revolution",
    "World War I",
    "Great Depression",
    "Colonialism",
    "Transatlantic slave trade",
    "Silk Road",
    "Medieval Europe",
    "Ming dynasty",
    "Ottoman Empire",
    "British Empire",
    # --- Philosophy ---
    "Philosophy",
    "Ethics",
    "Epistemology",
    "Metaphysics",
    "Philosophy of science",
    "Utilitarianism",
    "Kantian ethics",
    "Existentialism",
    "Empiricism",
    "Rationalism",
    "Phenomenology",
    "Social contract",
    "Justice",
    "Political philosophy",
    "Anarchist philosophy",
    # --- Economics (critical) ---
    "Capitalism",
    "Keynesian economics",
    "Neoliberalism",
    "Austerity",
    "Unemployment",
    "Inflation",
    "Minimum wage",
    "Housing",
    "Homelessness",
    "Food security",
    "Healthcare",
    "Universal healthcare",
    "Education",
    "Mass incarceration",
    "Prison–industrial complex",
]


# =============================================================================
#  Wikiquote pages
#  Heavy on radical, progressive, scientific, and literary voices.
# =============================================================================

WIKIQUOTE_PAGES = [
    # Science
    "Albert Einstein", "Marie Curie", "Carl Sagan", "Richard Feynman",
    "Stephen Hawking", "Ada Lovelace", "Grace Hopper", "Nikola Tesla",
    "Katherine Johnson", "Mae Jemison", "Charles Darwin", "Alan Turing",
    "Richard Dawkins", "Neil deGrasse Tyson", "Rachel Carson",
    # Left / socialist
    "Karl Marx", "Friedrich Engels", "Vladimir Lenin", "Rosa Luxemburg",
    "Emma Goldman", "Eugene V. Debs", "Antonio Gramsci", "Leon Trotsky",
    "Howard Zinn", "Noam Chomsky", "Angela Davis", "Fred Hampton",
    "Che Guevara", "Nelson Mandela", "Frantz Fanon", "James Connolly",
    "Mahatma Gandhi", "Ho Chi Minh",
    # Civil rights
    "Martin Luther King Jr.", "Malcolm X", "Rosa Parks", "Harriet Tubman",
    "Frederick Douglass", "Sojourner Truth", "Fannie Lou Hamer",
    "John Lewis", "Audre Lorde", "James Baldwin", "Langston Hughes",
    "Maya Angelou", "Toni Morrison", "Zora Neale Hurston",
    # Women's rights / feminism
    "Simone de Beauvoir", "Virginia Woolf", "bell hooks", "Gloria Steinem",
    "Mary Wollstonecraft", "Malala Yousafzai", "Adrienne Rich",
    "Sylvia Plath", "Susan B. Anthony",
    # LGBTQ+
    "Oscar Wilde", "Harvey Milk", "Bayard Rustin", "Quentin Crisp",
    # Philosophy / literature
    "Aristotle", "Voltaire", "Bertrand Russell", "Hannah Arendt",
    "Chinua Achebe", "Rumi", "Pablo Neruda", "Bertolt Brecht",
    "Jean-Paul Sartre", "Albert Camus", "George Orwell",
    # Topics
    "Justice", "Freedom", "Knowledge", "Love", "Art", "Revolution",
    "Democracy", "Equality",
]


# =============================================================================
#  HTTP helpers
# =============================================================================

_REQUEST_DELAY = 3.5   # seconds between requests
_last_request  = 0.0
_request_count = 0
_COOLDOWN_EVERY = 150  # cool down for 30s every N requests

_HEADERS = {
    "User-Agent": "miniGPT-DatasetBuilder/3.0 (educational, non-commercial)",
    "Accept-Encoding": "gzip",
}


def _get(url: str, timeout: int = 25) -> Optional[str]:
    """
    Fetch a URL with polite delay, cooldown every 150 requests,
    and exponential backoff on 429.
    """
    global _last_request, _request_count

    _request_count += 1
    if _request_count % _COOLDOWN_EVERY == 0:
        print(f"  [cooldown] {_request_count} requests made, pausing 30s...")
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
                    backoff = max(int(r.headers.get("Retry-After", 0)),
                                  20 * (2 ** attempt))
                    print(f"  [429] Rate limited, waiting {backoff}s...")
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
                    return resp.read().decode("utf-8", errors="ignore")

        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                backoff = 20 * (2 ** attempt)
                print(f"  [rate limit] waiting {backoff}s...")
                time.sleep(backoff)
                _last_request = time.time()
                continue
            print(f"  [WARN] {url[:80]}: {e}")
            return None

    print(f"  [WARN] Gave up after 7 retries: {url[:80]}")
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
    Keeps substantive sentences, drops navigation fragments and stray numbers.
    """
    text = re.sub(r'<[^>]+>',                        ' ',    text)  # HTML tags
    text = re.sub(r'\[\[([^\]|]*\|)?([^\]]+)\]\]',  r'\2',  text)  # [[link|text]]
    text = re.sub(r'\{\{[^}]*\}\}',                  '',     text)  # {{templates}}
    text = re.sub(r'==+[^=]+=+',                     '',     text)  # == Headings ==
    text = re.sub(r"'{2,}",                           '',     text)  # ''bold''
    text = re.sub(r'https?://\S+',                   '',     text)  # URLs
    text = re.sub(r'\[\d+\]',                        '',     text)  # [1] citations
    text = re.sub(r'\^?\[\w+\]',                     '',     text)  # [note] markers
    text = re.sub(r'thumb\|[^\|]+\|',               '',     text)  # image captions
    text = re.sub(r'File:[^\n]+',                    '',     text)  # file refs
    text = re.sub(r'[ \t]+',                         ' ',    text)  # multiple spaces
    text = re.sub(r'\n{3,}',                         '\n\n', text)  # blank lines
    lines = [l.strip() for l in text.split('\n')]
    text  = '\n'.join(l for l in lines if len(l) >= 40 or l == '')
    return text.strip()


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header and footer."""
    start_re = re.compile(r'\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG', re.I)
    end_re   = re.compile(r'\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG',   re.I)
    m_start  = start_re.search(text)
    start    = text.find('\n', m_start.end()) + 1 if m_start else 0
    m_end    = end_re.search(text)
    end      = m_end.start() if m_end else len(text)
    return text[start:end].strip()


# =============================================================================
#  Source fetchers
# =============================================================================

def fetch_gutenberg(target_chars: int) -> List[str]:
    """
    Download Gutenberg books (shuffled) until target_chars is reached.
    """
    print(f"\n[1/5] Gutenberg  (target {target_chars:,} chars, {len(GUTENBERG_BOOKS)} books available)")
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
    """Fetch one Wikipedia article. Returns clean text or None."""
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
    """Return random Wikipedia article titles."""
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
    """
    Generic fetcher for a list of articles.
    Returns (chunks, total_chars_added). Stops at cap if provided.
    """
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
    """
    Four-pass Wikipedia fetch:

    Pass 1 -- LGBTQ+ articles (guaranteed, no cap)
    Pass 2 -- Left/socialist/communist articles (guaranteed, no cap)
    Pass 3 -- Diverse curated topics (up to remaining capacity)
    Pass 4 -- Academic/science articles (up to remaining capacity)
    Pass 5 -- Random fill (remaining quota)
    """
    print(f"\n[2/5] Wikipedia  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    seen   = set()

    # Pass 1: LGBTQ+ -- always, no cap
    print(f"\n  Pass 1: LGBTQ+ ({len(LGBTQ_ARTICLES)} articles, guaranteed)")
    c, t = _fetch_pass("LGBTQ+", LGBTQ_ARTICLES, seen)
    chunks.extend(c); total += t

    # Pass 2: Left-wing / socialist -- always, no cap
    print(f"\n  Pass 2: Left/socialist ({len(LEFT_ARTICLES)} articles, guaranteed)")
    c, t = _fetch_pass("Left", LEFT_ARTICLES, seen)
    chunks.extend(c); total += t

    # Pass 3: Diverse curated -- up to 40% of target
    cap3 = total + int(target_chars * 0.40)
    print(f"\n  Pass 3: Diverse curated (cap {cap3:,})")
    shuffled_diverse = random.sample(DIVERSE_ARTICLES, len(DIVERSE_ARTICLES))
    c, t = _fetch_pass("Diverse", shuffled_diverse, seen, cap=int(target_chars * 0.40))
    chunks.extend(c); total += t

    # Pass 4: Academic/science -- up to 30% of target
    if total < target_chars:
        print(f"\n  Pass 4: Academic/science (cap {int(target_chars * 0.30):,})")
        shuffled_acad = random.sample(ACADEMIC_ARTICLES, len(ACADEMIC_ARTICLES))
        c, t = _fetch_pass("Academic", shuffled_acad, seen, cap=int(target_chars * 0.30))
        chunks.extend(c); total += t

    # Pass 5: Random fill
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
                    chunks.append(text)
                    total += len(text)
                    ok += 1
                    print(f"    RAND [{ok}] {title}  ({len(text):,} chars)  total={total:,}")
                else:
                    fail += 1
        print(f"  Random fill done: {ok} articles  |  {total:,} chars total")

    print(f"\n  Wikipedia done: {total:,} chars from {len(chunks)} articles")
    return chunks


def fetch_simple_wikipedia(target_chars: int) -> List[str]:
    """
    Fetch random Simple English Wikipedia articles.
    Short, clean sentences -- good for training sentence rhythm.
    """
    print(f"\n[3/5] Simple Wikipedia  (target {target_chars:,} chars)")
    chunks = []
    total  = 0
    seen   = set()
    ok = fail = 0

    while total < target_chars and fail < 30:
        url = (
            "https://simple.wikipedia.org/w/api.php"
            "?action=query&list=random&rnlimit=20&rnnamespace=0"
            "&format=json"
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
                chunks.append(text)
                total += len(text)
                batch_added += 1
                ok += 1
                if ok % 5 == 0:
                    print(f"    [{ok}] {title}  ({len(text):,} chars)  total={total:,}/{target_chars:,}")
            else:
                fail += 1
        
        if batch_added == 0:
            break

    print(f"  Simple Wikipedia done: {total:,} chars from {len(chunks)} articles ({ok} OK, {fail} skipped)")
    return chunks


def fetch_wikiquote(target_chars: int) -> List[str]:
    """
    Fetch Wikiquote pages. Short diverse quotes -- great sentence variety.
    """
    print(f"\n[4/5] Wikiquote  (target {target_chars:,} chars)")
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
    """
    Fetch chapters from open Wikibooks textbooks (science, maths, history).
    Dense factual prose -- excellent for academic vocabulary.
    """
    print(f"\n[5/5] Wikibooks  (target {target_chars:,} chars)")

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


# =============================================================================
#  Checkpoint saving
# =============================================================================

def _save_checkpoint(chunks: List[str], filename: str) -> None:
    """Save phase chunks to checkpoint file."""
    text = "\n\n".join(chunks)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

def build_dataset(
    target_chars:    int,
    output_file:     str,
    no_gutenberg:    bool = False,
    no_wikipedia:    bool = False,
    no_simple_wiki:  bool = False,
    no_wikiquote:    bool = False,
    no_wikibooks:    bool = False,
) -> None:
    """
    Collect from all sources, shuffle documents, write to file.

    Default split (2.2M chars):
        35% Gutenberg       ~770k  -- rich prose, vocabulary breadth
        30% Wikipedia       ~660k  -- LGBTQ+ + left guaranteed, then diverse + academic
        15% Simple Wikipedia~330k  -- clean plain sentences
        10% Wikiquote       ~220k  -- short quotes, diverse voices
        10% Wikibooks       ~220k  -- dense factual/academic text
    """
    print("=" * 65)
    print("miniGPT dataset builder  v3.0")
    print(f"Target : {target_chars:,} chars  ->  {output_file}")
    print("Sources: Gutenberg | Wikipedia (LGBTQ+ + Left + Diverse + Academic)")
    print("         Simple Wikipedia | Wikiquote | Wikibooks")
    print("=" * 65)

    print("\nWaiting 10s before starting (clears any Wikipedia rate limit)...")
    for i in range(10, 0, -2):
        print(f"  {i}s...", end="\r", flush=True)
        time.sleep(2)
    print("  Starting.         ")

    chunks = []

    if not no_gutenberg:
        gutenberg_chars = int(target_chars * 0.35)
        phase_chunks = fetch_gutenberg(gutenberg_chars)
        chunks.extend(phase_chunks)
        _save_checkpoint(phase_chunks, "Dataset/Dataset part backup/phase_1_gutenberg.txt")
        print(f"  ✓ Gutenberg checkpoint saved")

    if not no_wikipedia:
        wikipedia_chars = int(target_chars * 0.30)
        phase_chunks = fetch_wikipedia(wikipedia_chars)
        chunks.extend(phase_chunks)
        _save_checkpoint(phase_chunks, "Dataset/Dataset part backup/phase_2_wikipedia.txt")
        print(f"  ✓ Wikipedia checkpoint saved")

    if not no_simple_wiki:
        simple_chars = int(target_chars * 0.15)
        phase_chunks = fetch_simple_wikipedia(simple_chars)
        chunks.extend(phase_chunks)
        _save_checkpoint(phase_chunks, "Dataset/Dataset part backup/phase_3_simple_wikipedia.txt")
        print(f"  ✓ Simple Wikipedia checkpoint saved")

    if not no_wikiquote:
        wikiquote_chars = int(target_chars * 0.10)
        phase_chunks = fetch_wikiquote(wikiquote_chars)
        chunks.extend(phase_chunks)
        _save_checkpoint(phase_chunks, "Dataset/Dataset part backup/phase_4_wikiquote.txt")
        print(f"  ✓ Wikiquote checkpoint saved")

    if not no_wikibooks:
        wikibooks_chars = int(target_chars * 0.10)
        phase_chunks = fetch_wikibooks(wikibooks_chars)
        chunks.extend(phase_chunks)
        _save_checkpoint(phase_chunks, "Dataset/Dataset part backup/phase_5_wikibooks.txt")
        print(f"  ✓ Wikibooks checkpoint saved")

    if not chunks:
        print("\nERROR: no text collected. Check internet connection.")
        sys.exit(1)

    # Shuffle so sources interleave -- model never sees a block of one style
    random.shuffle(chunks)

    full_text = "\n\n".join(chunks)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}',  ' ',    full_text)
    full_text = full_text.strip()

    # Trim to target at a word boundary
    if len(full_text) > target_chars:
        cut       = full_text.rfind(' ', 0, target_chars)
        full_text = full_text[:cut] if cut > 0 else full_text[:target_chars]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    from collections import Counter
    size_mb = os.path.getsize(output_file) / 1024 / 1024

    # Content audit -- check key topics are represented
    text_lower = full_text.lower()
    audits = [
        ("LGBTQ+",     ["stonewall", "lgbtq", "transgender", "queer", "gay rights"]),
        ("Socialist",  ["socialism", "communism", "marxism", "working class", "proletariat"]),
        ("Civil rights", ["civil rights", "segregation", "discrimination"]),
        ("Science",    ["chemistry", "physics", "biology", "evolution"]),
        ("Women",      ["feminism", "suffrage", "women's rights"]),
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
    parser.add_argument("--target_chars",  type=int, default=2_200_000,
                        help="Target character count (default: 2,200,000)")
    parser.add_argument("--output",        type=str, default="diverse_dataset.txt",
                        help="Output filename (default: diverse_dataset.txt)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no_gutenberg",   action="store_true", help="Skip Gutenberg")
    parser.add_argument("--no_wikipedia",   action="store_true", help="Skip Wikipedia")
    parser.add_argument("--no_simple_wiki", action="store_true", help="Skip Simple Wikipedia")
    parser.add_argument("--no_wikiquote",   action="store_true", help="Skip Wikiquote")
    parser.add_argument("--no_wikibooks",   action="store_true", help="Skip Wikibooks")
    args = parser.parse_args()

    random.seed(args.seed)
    build_dataset(
        target_chars   = args.target_chars,
        output_file    = args.output,
        no_gutenberg   = args.no_gutenberg,
        no_wikipedia   = args.no_wikipedia,
        no_simple_wiki = args.no_simple_wiki,
        no_wikiquote   = args.no_wikiquote,
        no_wikibooks   = args.no_wikibooks,
    )