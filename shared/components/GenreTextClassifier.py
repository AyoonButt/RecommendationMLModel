"""
GenreTextClassifier: keyword-based genre classification over TMDB overview text.

TMDB ToS Compliant: this module only ever produces a boolean/label signal consumed
by EligibilityFilter (pass/fail). Its output is never fed into Two-Tower or RL
scoring features - see MetadataEnhancer.py's compliance notes.

Lightweight keyword matching rather than a per-request ML/LLM call, consistent
with the existing keyword-matching approach in ContentFilter.py.

Matching pipeline:
1. Normalize overview text (lowercase, strip accents/punctuation, collapse whitespace)
2. Single-word keywords matched via O(1) set lookup against the normalized word set
3. Multi-word phrases matched via word co-occurrence within a small window
   (order-independent, tolerates inserted words) - NOT a rigid substring match,
   since normalization only fixes surface form (case/accents/punctuation), not
   word order or insertion. Common irregular-verb inflections (falls/fell/falling)
   are listed as separate phrase entries rather than solved generically, since
   suffix-stripping doesn't help irregular verbs.
Both indexes are built once at import time from the full TMDB genre taxonomy
(movie + TV genre lists), keyed by genre_id to stay faithful to TMDB's actual IDs.
"""

import logging
import re
import unicodedata
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("genre-text-classifier")

# Full TMDB genre taxonomy (movie genre list + TV-only genres appended),
# each with characteristic keyword/phrase cues. Matching is approximate by
# design (recall over precision) - exists to catch posts whose structured
# genreWeights miss a genre the overview text makes clear.
GENRE_KEYWORDS: Dict[int, Tuple[str, List[str]]] = {
    28: ("Action", [
        "action-packed", "explosive", "high-octane", "daring rescue", "shootout",
        "chase scene", "adrenaline-fueled", "showdown", "raid", "mercenary",
    ]),
    12: ("Adventure", [
        "adventure", "expedition", "treasure hunt", "quest", "uncharted",
        "perilous journey", "voyage", "explorer", "jungle", "shipwrecked",
    ]),
    16: ("Animation", [
        "animated", "animation", "cartoon", "pixar", "stop-motion", "anime",
        "computer-animated", "voiced by", "hand-drawn", "voice cast", "claymation",
        "cgi-animated", "animated feature", "animated musical", "dreamworks",
        "animated adventure", "toon", "3d animated", "animated film", "anime film",
        "animated series", "voice of",
    ]),
    35: ("Comedy", [
        "comedy", "hilarious", "hijinks", "misadventures", "comedic", "slapstick",
        "sitcom", "farce", "satire", "buddy comedy",
    ]),
    80: ("Crime", [
        "crime", "detective", "heist", "murder investigation", "gangster",
        "smuggling", "criminal underworld", "serial killer", "mob boss", "cartel",
        "con artist",
    ]),
    99: ("Documentary", [
        "documentary", "true story", "real-life", "archival footage",
        "interviews with", "docuseries", "investigates", "investigated", "investigating",
        "chronicles the", "chronicled the", "chronicling the",
        "based on true events", "firsthand accounts",
        "explores the life of", "explored the life of", "exploring the life of",
        "sheds light on", "shed light on", "shedding light on",
        "untold story",
        "exposes the", "exposed the", "exposing the",
        "reveals the truth", "revealed the truth", "revealing the truth",
    ]),
    18: ("Drama", [
        "drama", "emotional journey", "coming of age", "family drama", "estranged",
        "struggles to overcome", "struggled to overcome", "struggling to overcome",
        "tragedy strikes", "tragedy struck",
    ]),
    10751: ("Family", [
        "family-friendly", "for the whole family", "kid-friendly", "family adventure",
    ]),
    14: ("Fantasy", [
        "fantasy", "wizard", "sorcery", "mythical creatures", "kingdom of",
        "magical realm", "dragons", "prophecy", "enchanted", "sorcerer",
    ]),
    36: ("History", [
        "historical", "based on the life of", "set during the", "true events of",
        "period drama", "ancient civilization",
    ]),
    27: ("Horror", [
        "horror", "haunted", "demon", "possessed", "slasher", "supernatural terror",
        "curse", "exorcism", "paranormal", "monster", "zombie", "vampire",
        "werewolf", "terrifying", "nightmare", "evil spirit", "sinister force",
        "terrorized by", "terrorizes", "terrorizing", "chilling", "bloodcurdling",
        "malevolent presence", "gruesome", "ghostly", "paranormal activity",
        "occult ritual", "ancient evil", "unspeakable evil", "spine-tingling",
        "creature lurking", "creature lurks", "creature lurked",
        "stalked by", "stalks", "stalking", "possessed by an evil",
    ]),
    10402: ("Music", [
        "musical", "concert film", "band's rise", "singer-songwriter", "rock band",
        "recording career", "musical numbers", "original songs",
        "breaks into song", "broke into song", "breaking into song",
        "jukebox musical", "aspiring musician", "record deal", "battle of the bands",
    ]),
    9648: ("Mystery", [
        "mystery", "whodunit", "unsolved", "disappearance of", "secret hidden",
        "cryptic clues", "puzzling case",
    ]),
    10749: ("Romance", [
        "falls in love", "fell in love", "falling in love",
        "love story", "romance", "romantic", "soulmate",
        "wedding day", "star-crossed", "love triangle",
        "falls for", "fell for", "falling for",
        "falls hard for", "fell hard for", "falling hard for",
        "sparks fly", "swept off her feet", "swept off his feet", "true love",
        "chemistry between", "second chance at love", "forbidden love",
        "love blossoms", "passionate affair", "heartbreak",
        "high school sweetheart", "matchmaker", "romantic getaway",
        "engagement", "proposes", "proposed", "proposing",
        "unexpected romance", "love of her life", "love of his life",
        "whirlwind romance", "reunites with her ex", "reunited with her ex",
        "reunites with his ex", "reunited with his ex", "affair with",
    ]),
    878: ("Science Fiction", [
        "sci-fi", "science fiction", "spacecraft", "alien invasion",
        "dystopian future", "time travel", "artificial intelligence",
        "interstellar", "cyberpunk", "extraterrestrial",
    ]),
    10770: ("TV Movie", [
        "tv movie", "made-for-television",
    ]),
    53: ("Thriller", [
        "thriller", "conspiracy", "race against time", "cat and mouse",
        "high-stakes", "psychological tension", "double-cross",
    ]),
    10752: ("War", [
        "war", "battlefield", "soldiers", "soldier", "wartime", "combat", "invasion",
        "military campaign", "trenches", "resistance fighters", "platoon",
        "front lines", "enemy forces", "under siege", "prisoners of war",
        "warfront", "deployed to", "deploys to", "deploying to",
    ]),
    37: ("Western", [
        "western", "wild west", "gunslinger", "frontier town", "outlaw",
        "sheriff", "saloon", "cattle ranch", "bounty hunter", "lawless town",
        "cattle rustlers", "showdown at high noon",
    ]),
    # TV-only genres (distinct genre_ids from the movie taxonomy)
    10759: ("Action & Adventure", [
        "action-packed", "high-octane", "daring rescue", "adventure", "expedition",
        "quest",
    ]),
    10762: ("Kids", [
        "kids show", "children's series", "preschool", "for young viewers",
    ]),
    10763: ("News", [
        "news coverage", "breaking news", "newsroom", "current affairs",
    ]),
    10764: ("Reality", [
        "reality show", "unscripted", "competition series", "docuseries follows",
        "real people",
    ]),
    10765: ("Sci-Fi & Fantasy", [
        "sci-fi", "science fiction", "fantasy", "wizard", "alien invasion",
        "magical realm", "dragons", "interstellar",
    ]),
    10766: ("Soap", [
        "soap opera", "melodrama", "love triangle", "family secrets",
    ]),
    10767: ("Talk", [
        "talk show", "late-night", "interview series",
    ]),
    10768: ("War & Politics", [
        "war", "political drama", "geopolitical", "battlefield", "government conspiracy",
    ]),
}


def _normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse non-alphanumerics to spaces, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Max distance (in words) allowed between a phrase's constituent words for a
# co-occurrence match. Phrases match as an unordered word set within this span,
# not as a rigid substring - "falls madly in love" still matches "falls in love"
# because "madly" fits inside the window instead of breaking a substring search.
_PHRASE_WINDOW = 5


def _build_index() -> Tuple[Dict[str, Set[int]], List[Tuple[frozenset, int]]]:
    """Build a single-word lookup index and a multi-word phrase word-set list, once, from GENRE_KEYWORDS."""
    single_word_index: Dict[str, Set[int]] = {}
    phrase_index: List[Tuple[frozenset, int]] = []

    for genre_id, (_, keywords) in GENRE_KEYWORDS.items():
        for keyword in keywords:
            normalized_kw = _normalize_text(keyword)
            if not normalized_kw:
                continue
            words = normalized_kw.split()
            if len(words) > 1:
                phrase_index.append((frozenset(words), genre_id))
            else:
                single_word_index.setdefault(words[0], set()).add(genre_id)

    return single_word_index, phrase_index


_SINGLE_WORD_INDEX, _PHRASE_INDEX = _build_index()


def _phrase_matches(text_words: List[str], phrase_words: frozenset) -> bool:
    """
    Check if all words of a phrase co-occur within a _PHRASE_WINDOW-word span,
    in any order - robust to inserted words and reordering (not to inflection).
    """
    span = max(_PHRASE_WINDOW, len(phrase_words))
    for start in range(max(1, len(text_words) - span + 1)):
        if phrase_words.issubset(text_words[start:start + span]):
            return True
    return False


def classify_overview_genres(overview: str) -> Set[str]:
    """
    Classify a TMDB overview/synopsis string into zero or more genre labels.

    Args:
        overview: Post overview/synopsis text (TMDB-sourced).

    Returns:
        Set of matched genre names (subset of GENRE_KEYWORDS values). Empty if no
        overview text or no keyword/phrase matches.
    """
    if not overview:
        return set()

    normalized = _normalize_text(overview)
    if not normalized:
        return set()

    matched_ids: Set[int] = set()
    text_words = normalized.split()

    # Single-word keywords: O(1) set lookup per overview word
    for word in text_words:
        genre_ids = _SINGLE_WORD_INDEX.get(word)
        if genre_ids:
            matched_ids.update(genre_ids)

    # Multi-word phrases: word co-occurrence within a small window (order-independent,
    # tolerates inserted words) rather than a rigid substring match
    for phrase_words, genre_id in _PHRASE_INDEX:
        if genre_id not in matched_ids and _phrase_matches(text_words, phrase_words):
            matched_ids.add(genre_id)

    return {GENRE_KEYWORDS[genre_id][0] for genre_id in matched_ids}


# =============================================================================
# Sub-genre classification
#
# Finer-grained labels beyond TMDB's ~20 genres (e.g. "Heist", "Romantic Comedy").
# IDs are STABLE and MUST MATCH the Spring API's sub_genres.sub_genre_id table
# (seeded by SubGenreSeeder) so keyword-derived and text-derived classifications
# share one ID space and can be merged trivially at sync time.
#
# TMDB ToS Compliant: classification/labeling only (boolean/soft signal), never
# fed into Two-Tower or RL scoring features directly.
# =============================================================================

SUB_GENRE_KEYWORDS: Dict[int, Tuple[str, List[str]]] = {
    1: ("Heist", ["heist", "robbery", "bank job", "vault", "inside job", "the score", "con artist"]),
    2: ("Psychological Thriller", ["psychological thriller", "psychological manipulation", "unreliable narrator", "paranoia", "gaslighting", "mind game", "cat and mouse"]),
    3: ("Romantic Comedy", ["romantic comedy", "rom-com", "meet cute", "fake dating", "enemies to lovers", "friends to lovers", "love triangle"]),
    4: ("Coming of Age", ["coming of age", "growing up", "teen drama", "first love", "high school", "adolescence", "rites of passage"]),
    5: ("Survival", ["survival", "stranded", "deserted island", "fight to survive", "last one standing", "outlast"]),
    6: ("Superhero", ["superhero", "superpower", "masked vigilante", "origin story", "supervillain", "comic book"]),
    7: ("Space Opera", ["space opera", "galactic empire", "interstellar war", "star fleet", "space battle"]),
    8: ("Whodunit", ["whodunit", "murder mystery", "suspects", "the killer", "detective story", "clue"]),
    9: ("Period Piece", ["period piece", "costume drama", "victorian era", "regency", "medieval", "renaissance"]),
    10: ("Dark Comedy", ["dark comedy", "black comedy", "dark humor", "gallows humor", "cringe comedy", "absurdist"]),
    11: ("Courtroom Drama", ["courtroom drama", "trial", "jury", "lawyer", "verdict"]),
    12: ("Zombie", ["zombie", "zombie apocalypse", "undead", "outbreak"]),
    13: ("Time Travel", ["time travel", "time loop", "paradox"]),
    14: ("Dystopian", ["dystopian", "dystopian future", "dystopia", "oppressive regime", "totalitarian"]),
    15: ("True Crime", ["true crime", "serial killer", "murder investigation", "cold case", "unsolved"]),
    16: ("Martial Arts", ["martial arts", "kung fu", "karate", "mixed martial arts", "ninja"]),
    17: ("Found Footage", ["found footage", "documentary style", "handheld camera"]),
    18: ("Post Apocalyptic", ["post-apocalyptic", "apocalyptic", "wasteland", "the end of the world"]),
}


def _build_sub_genre_index() -> Tuple[Dict[str, Set[int]], List[Tuple[frozenset, int]]]:
    """Build a single-word lookup index and multi-word phrase list for SUB_GENRE_KEYWORDS."""
    single_word_index: Dict[str, Set[int]] = {}
    phrase_index: List[Tuple[frozenset, int]] = []

    for sub_genre_id, (_, keywords) in SUB_GENRE_KEYWORDS.items():
        for keyword in keywords:
            normalized_kw = _normalize_text(keyword)
            if not normalized_kw:
                continue
            words = normalized_kw.split()
            if len(words) > 1:
                phrase_index.append((frozenset(words), sub_genre_id))
            else:
                single_word_index.setdefault(words[0], set()).add(sub_genre_id)

    return single_word_index, phrase_index


_SUB_SINGLE_WORD_INDEX, _SUB_PHRASE_INDEX = _build_sub_genre_index()


def classify_overview_sub_genres(overview: str) -> Dict[int, float]:
    """
    Classify a TMDB overview/synopsis string into sub-genres.

    Uses the same keyword-matching pipeline as classify_overview_genres but
    against the sub-genre dictionary. Returns a confidence score per sub-genre
    (fraction of the sub-genre's keywords matched), keyed by the stable
    sub_genre_id that the Spring API's post_sub_genres table uses.

    Args:
        overview: Post overview/synopsis text (TMDB-sourced).

    Returns:
        Dict mapping sub_genre_id -> confidence score (0.0-1.0). Empty if no
        overview text or no keyword/phrase matches.
    """
    if not overview:
        return {}

    normalized = _normalize_text(overview)
    if not normalized:
        return {}

    text_words = normalized.split()
    matched_ids: Set[int] = set()

    for word in text_words:
        sub_genre_ids = _SUB_SINGLE_WORD_INDEX.get(word)
        if sub_genre_ids:
            matched_ids.update(sub_genre_ids)

    for phrase_words, sub_genre_id in _SUB_PHRASE_INDEX:
        if sub_genre_id not in matched_ids and _phrase_matches(text_words, phrase_words):
            matched_ids.add(sub_genre_id)

    if not matched_ids:
        return {}

    # Confidence = fraction of a sub-genre's keywords that matched (with matched
    # multi-word phrases counting as their word count for a smoother signal).
    scores: Dict[int, float] = {}
    for sub_genre_id in matched_ids:
        keywords = SUB_GENRE_KEYWORDS[sub_genre_id][1]
        total_words = sum(len(_normalize_text(kw).split()) for kw in keywords)
        if total_words == 0:
            continue
        matched_words = 0
        for keyword in keywords:
            kw_words = _normalize_text(keyword).split()
            if len(kw_words) == 1:
                if kw_words[0] in text_words:
                    matched_words += 1
            else:
                if _phrase_matches(text_words, frozenset(kw_words)):
                    matched_words += len(kw_words)
        scores[sub_genre_id] = min(1.0, matched_words / total_words)

    return scores
