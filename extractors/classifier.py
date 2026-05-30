"""
extractors/classifier.py
-------------------------
Two public functions:

    extract_case_type(text) -> str
    extract_sub_type(text, case_type) -> str

Design notes
------------
* Defamation/Slander is checked BEFORE criminal to avoid false positives
  (e.g. "charged with adultery" → Torts, not Criminal).
* Scoring uses weighted keyword dicts so high-frequency words like
  "damages" or "case" don't unfairly inflate Torts scores.
* Sub-type uses ordered lists of (keyword, label) pairs — first match wins.
"""

from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Case-type weighted keyword tables
# weight ≥ 3 = strong signal,  1-2 = weak / contextual signal
# ─────────────────────────────────────────────────────────────────────────────

_DEFAMATION_KW: List[str] = [
    "slander", "defamation", "libel", "scandalous words",
    "action on the case for slander", "slanderous words",
    "for slanderous words", "words spoken", "actionable words",
    "charged with adultery", "fornication",
]

_CRIMINAL_KW: Dict[str, int] = {
    "indictment":      5, "larceny":       5, "murder":         5,
    "felony":          4, "homicide":      5, "manslaughter":   5,
    "robbery":         4, "burglary":      4, "counterfeiting": 4,
    "perjury":         4, "grand jury":    3, "misdemeanor":    3,
    "guilty":          3, "theft":         3, "assault":        3,
    "battery":         3, "crime":         2, "criminal":       3,
    "prosecution":     2, "quo warranto":  3, "contra formam statuti": 3,
    "state":           1,
}

_CIVIL_KW: Dict[str, int] = {
    "demurrer":        4, "scire facias":  4, "habeas corpus":  4,
    "certiorari":      4, "mandamus":      4, "replevin":       3,
    "nul tiel record": 3, "writ":          2, "plea":           2,
    "pleading":        2, "jurisdiction":  2, "venue":          2,
    "continuance":     2, "appeal":        2, "error":          2,
    "assumpsit":       3, "joinder":       2, "replication":    2,
    "injunction":      2, "bill of exceptions": 2,
}

_CONTRACT_KW: Dict[str, int] = {
    "promissory note":      5, "negotiable instrument": 4,
    "bond":                 3, "covenant":              3,
    "usury":                4, "sealed note":           3,
    "writing obligatory":   3, "surety":                2,
    "debt":                 2, "contract":              2,
    "payment":              2, "consideration":         2,
    "obligation":           2,
}

_PROP_EJECT_KW: Dict[str, int] = {
    "ejectment":        5, "adverse possession":  4,
    "disseisin":        4, "equity of redemption":4,
    "foreclosure":      3, "mortgage":            3,
    "real estate":      3, "title":               2,
    "deed":             2, "conveyance":          2,
    "land":             2, "possession":          2,
    "lease":            2, "demise":              2,
}

_PROP_EXEC_KW: Dict[str, int] = {
    "fieri facias":     5, "fi. fa.":             5,
    "capias":           4, "ca. sa.":             4,
    "venditioni exponas":4, "replevin bond":      4,
    "execution":        3, "sheriff":             3,
    "levy":             3, "distress":            2,
    "sale":             1,
}

_TORTS_KW: Dict[str, int] = {
    "negligence":  4, "nuisance":    4, "conversion": 4,
    "trover":      4, "trespass":    3, "mesne profits":3,
    "injury":      2, "damages":     1,
}


def _weighted_score(text_lower: str, kw_dict: Dict[str, int]) -> int:
    return sum(weight for kw, weight in kw_dict.items() if kw in text_lower)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_case_type(text: str) -> str:
    """Return the primary case-type label for *text*."""
    t = text.lower()

    # Priority 1 – Defamation/Slander (must precede criminal check)
    if any(kw in t for kw in _DEFAMATION_KW):
        return "Torts - Defamation"

    scores: Dict[str, int] = {
        "Criminal Law":                _weighted_score(t, _CRIMINAL_KW),
        "Civil Procedure":             _weighted_score(t, _CIVIL_KW),
        "Contract Law - Debt":         _weighted_score(t, _CONTRACT_KW),
        "Property Law - Ejectment":    _weighted_score(t, _PROP_EJECT_KW),
        "Property Law - Execution Sale": _weighted_score(t, _PROP_EXEC_KW),
        "Torts":                       _weighted_score(t, _TORTS_KW),
    }

    best_score = max(scores.values())
    if best_score == 0:
        return "Unclassified"

    # Return first label with the highest score
    for label, score in scores.items():
        if score == best_score:
            return label

    return "Unclassified"


def extract_sub_type(text: str, case_type: str) -> str:
    """Return a fine-grained sub-type for *text* given its *case_type*."""
    t = text.lower()

    dispatchers = {
        "Torts - Defamation":           _sub_defamation,
        "Criminal Law":                 _sub_criminal,
        "Civil Procedure":              _sub_civil,
        "Contract Law - Debt":          _sub_contract,
        "Property Law - Ejectment":     _sub_property_eject,
        "Property Law - Execution Sale":_sub_property_exec,
        "Family Law - Dower":           _sub_family,
        "Torts":                        _sub_torts,
    }

    fn = dispatchers.get(case_type)
    if fn:
        return fn(t)

    # Fallback for "Unclassified"
    return _sub_fallback(t)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-type helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_match(t: str, pairs: List[Tuple[str, str]]) -> str:
    """Return label of first keyword found in *t*, or empty string."""
    for kw, label in pairs:
        if kw in t:
            return label
    return ""


def _sub_defamation(t: str) -> str:
    if any(kw in t for kw in ["slander", "words spoken", "scandalous words", "slanderous"]):
        return "Slander"
    if "libel" in t:
        return "Libel"
    return "Defamation"


def _sub_criminal(t: str) -> str:
    pairs = [
        ("grand larceny",       "Larceny"),
        ("petit larceny",       "Larceny"),
        ("larceny",             "Larceny"),
        ("receiving stolen",    "Larceny"),
        ("stolen goods",        "Larceny"),
        ("theft",               "Larceny"),
        ("murder",              "Homicide"),
        ("homicide",            "Homicide"),
        ("manslaughter",        "Homicide"),
        ("mortal wound",        "Homicide"),
        ("assault and battery", "Assault & Battery"),
        ("assault & battery",   "Assault & Battery"),
        ("assault",             "Assault & Battery"),
        ("battery",             "Assault & Battery"),
        ("burglary",            "Burglary/Robbery"),
        ("robbery",             "Burglary/Robbery"),
        ("housebreaking",       "Burglary/Robbery"),
        ("breaking and entering","Burglary/Robbery"),
        ("counterfeiting",      "Counterfeiting"),
        ("forgery",             "Counterfeiting"),
        ("counterfeit",         "Counterfeiting"),
        ("perjury",             "Perjury"),
        ("false oath",          "Perjury"),
        ("forsworn",            "Perjury"),
        ("affray",              "Affray"),
        ("riot",                "Riot"),
    ]
    result = _first_match(t, pairs)
    if result:
        return result
    if "kill" in t and any(kw in t for kw in ["person", "man", "woman"]):
        return "Homicide"
    if "indictment" in t and "found" in t:
        return "Indictment"
    return "General Criminal"


def _sub_civil(t: str) -> str:
    pairs = [
        ("habeas corpus",  "Habeas Corpus"),
        ("scire facias",   "Scire Facias"),
        ("certiorari",     "Writ"),
        ("mandamus",       "Writ"),
        ("demurrer",       "Demurrer"),
        ("writ of error",  "Appeal"),
        ("appeal",         "Appeal"),
        ("attachment",     "Attachment"),
        ("nil dicit",      "Default Judgment"),
        ("default",        "Default Judgment"),
        ("continuance",    "Continuance"),
        ("adjournment",    "Continuance"),
        ("replication",    "Pleading"),
        ("plea",           "Pleading"),
        ("writ",           "Writ"),
    ]
    return _first_match(t, pairs) or "General Civil Procedure"


def _sub_contract(t: str) -> str:
    pairs = [
        ("promissory note",       "Promissory Note"),
        ("negotiable instrument", "Promissory Note"),
        ("usury",                 "Usury"),
        ("usurious",              "Usury"),
        ("writing obligatory",    "Bond"),
        ("sealed",                "Bond"),
        ("bond",                  "Bond"),
        ("breach",                "Breach of Contract"),
        ("non-performance",       "Breach of Contract"),
        ("failed to perform",     "Breach of Contract"),
        ("debt",                  "Debt Collection"),
        ("indebtedness",          "Debt Collection"),
    ]
    return _first_match(t, pairs) or "General Contract"


def _sub_property_eject(t: str) -> str:
    pairs = [
        ("adverse possession",    "Adverse Possession"),
        ("disseisin",             "Adverse Possession"),
        ("equity of redemption",  "Mortgage Foreclosure"),
        ("foreclosure",           "Mortgage Foreclosure"),
        ("mortgage",              "Mortgage Foreclosure"),
        ("quiet title",           "Title Dispute"),
        ("cloud on title",        "Title Dispute"),
        ("title",                 "Title Dispute"),
        ("ejectment",             "Ejectment"),
        ("recovery of land",      "Ejectment"),
    ]
    return _first_match(t, pairs) or "General Property"


def _sub_property_exec(t: str) -> str:
    pairs = [
        ("fieri facias", "Fieri Facias"),
        ("fi. fa.",      "Fieri Facias"),
        ("ca. sa.",      "Capias"),
        ("capias",       "Capias"),
        ("sheriff sale", "Sheriff Sale"),
        ("execution sale","Sheriff Sale"),
    ]
    return _first_match(t, pairs) or "General Execution"


def _sub_family(t: str) -> str:
    pairs = [
        ("dower",       "Dower Rights"),
        ("dowable",     "Dower Rights"),
        ("heir",        "Inheritance"),
        ("devise",      "Inheritance"),
        ("legatee",     "Inheritance"),
    ]
    return _first_match(t, pairs) or "General Family Law"


def _sub_torts(t: str) -> str:
    pairs = [
        ("negligence",  "Negligence"),
        ("nuisance",    "Nuisance"),
        ("conversion",  "Conversion"),
        ("trover",      "Conversion"),
        ("trespass",    "Trespass"),
        ("assault",     "Assault & Battery"),
        ("battery",     "Assault & Battery"),
    ]
    return _first_match(t, pairs) or "General Torts"


def _sub_fallback(t: str) -> str:
    """Catch-all for Unclassified cases."""
    pairs = [
        ("grand larceny",  "Larceny"), ("larceny", "Larceny"),
        ("steal",          "Larceny"), ("theft",   "Larceny"),
        ("stolen",         "Larceny"),
        ("murder",         "Homicide"), ("homicide","Homicide"),
        ("assault",        "Assault & Battery"),
        ("writ of error",  "Appeal"),   ("appeal",  "Appeal"),
        ("demurrer",       "Demurrer"),
        ("ejectment",      "Ejectment"),
        ("foreclosure",    "Mortgage Foreclosure"),
        ("mortgage",       "Mortgage Foreclosure"),
        ("slander",        "Slander"),
        ("defamation",     "Defamation"),
        ("promissory note","Promissory Note"),
        ("bond",           "Bond"),
    ]
    return _first_match(t, pairs) or "Unclassified"