"""Player-name normalization for matching sportsbook and injury-report spellings to NBA IDs.

Sportsbooks and the NBA spell names differently ("Nic Claxton" / "Nicolas Claxton",
"Luka Dončić" / "Luka Doncic", "Harper Jr., Ron" / "Ron Harper Jr."). We normalize to a
lowercase ASCII key with suffixes and punctuation removed. Exact-key matches are taken
automatically; anything else goes to the alias table or the unmatched report.
"""

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_TEAM_SUFFIX = re.compile(r"\s*\([A-Z]{2,4}\)\s*$")  # "Tarik Skubal (DET)"


def strip_team_suffix(name: str) -> str:
    return _TEAM_SUFFIX.sub("", name).strip()


def last_first_to_first_last(name: str) -> str:
    """Injury-report format "Harper Jr., Ron" -> "Ron Harper Jr."."""
    if "," not in name:
        return name.strip()
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def normalize_name(name: str) -> str:
    """Matching key: ASCII, lowercase, no punctuation, no generational suffix."""
    name = strip_team_suffix(name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower().replace("-", " ")
    name = re.sub(r"[^a-z ]", "", name)
    tokens = [t for t in name.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def compact_forms(key: str) -> list[str]:
    """Fallback match keys for names whose spaces were squashed out ("Dereck LivelyII",
    "YanicKonan Niederhauser"): the normalized key without spaces, then without a generational
    suffix glued to its end ("derecklivelyii" -> "derecklively")."""
    c = key.replace(" ", "")
    forms = [c]
    for suf in ("iii", "ii", "iv", "jr", "sr"):
        if c.endswith(suf) and len(c) > len(suf) + 3:
            forms.append(c[: -len(suf)])
            break
    return forms


def parse_provider_player_id(raw: str | None) -> int | None:
    """PropLine player ids look like "nba:1628369" (the NBA personId). Others -> None."""
    if not raw or not raw.startswith("nba:"):
        return None
    try:
        return int(raw.split(":", 1)[1])
    except ValueError:
        return None
