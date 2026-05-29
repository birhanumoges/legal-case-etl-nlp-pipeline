"""
tests/test_classifier.py
-------------------------
Unit tests for classifier.extract_case_type() and extract_sub_type().

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractors.classifier import extract_case_type, extract_sub_type


# ── Case type ─────────────────────────────────────────────────────────────────

def test_defamation_beats_criminal():
    """A slander case should never be classified as Criminal Law."""
    text = "the plaintiff brings an action on the case for slander, charged with adultery."
    assert extract_case_type(text) == "Torts - Defamation"


def test_criminal_law_murder():
    text = "the defendant was indicted for murder in the first degree."
    assert extract_case_type(text) == "Criminal Law"


def test_criminal_law_larceny():
    text = "the grand jury returned a true bill of indictment for grand larceny."
    assert extract_case_type(text) == "Criminal Law"


def test_contract_promissory_note():
    text = "the plaintiff sues upon a promissory note made and endorsed by the defendant."
    assert extract_case_type(text) == "Contract Law - Debt"


def test_property_ejectment():
    text = "this is an action of ejectment to recover possession of land held adversely."
    assert extract_case_type(text) == "Property Law - Ejectment"


def test_civil_procedure_demurrer():
    text = "defendant filed a demurrer to the declaration on the grounds of misjoinder."
    assert extract_case_type(text) == "Civil Procedure"


def test_torts_negligence():
    text = "the plaintiff alleges negligence on the part of the defendant causing injury."
    assert extract_case_type(text) == "Torts"


def test_unclassified_generic():
    text = "parties appeared before the court."
    # minimal text – should not crash and should return a string
    result = extract_case_type(text)
    assert isinstance(result, str)


# ── Sub-type ──────────────────────────────────────────────────────────────────

def test_sub_type_larceny():
    text = "defendant was convicted of grand larceny for stealing horses."
    assert extract_sub_type(text, "Criminal Law") == "Larceny"


def test_sub_type_homicide():
    text = "the indictment charges the defendant with murder in the first degree."
    assert extract_sub_type(text, "Criminal Law") == "Homicide"


def test_sub_type_assault_battery():
    text = "the defendant committed an assault and battery upon the plaintiff."
    assert extract_sub_type(text, "Criminal Law") == "Assault & Battery"


def test_sub_type_slander():
    text = "action brought for slanderous words spoken by the defendant."
    assert extract_sub_type(text, "Torts - Defamation") == "Slander"


def test_sub_type_libel():
    text = "the plaintiff sues for libel published in the newspaper."
    assert extract_sub_type(text, "Torts - Defamation") == "Libel"


def test_sub_type_demurrer():
    text = "defendant filed a demurrer to the complaint."
    assert extract_sub_type(text, "Civil Procedure") == "Demurrer"


def test_sub_type_promissory_note():
    text = "suit on a promissory note for five hundred dollars."
    assert extract_sub_type(text, "Contract Law - Debt") == "Promissory Note"


def test_sub_type_mortgage_foreclosure():
    text = "action to foreclose mortgage on real property."
    assert extract_sub_type(text, "Property Law - Ejectment") == "Mortgage Foreclosure"


def test_sub_type_negligence():
    text = "plaintiff alleges negligence caused personal injury."
    assert extract_sub_type(text, "Torts") == "Negligence"