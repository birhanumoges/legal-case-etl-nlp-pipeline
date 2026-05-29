"""
tests/test_verdict.py
----------------------
Unit tests for verdict_extractor.extract_verdict().

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractors.verdict_extractor import extract_verdict


# ── WV Court of Claims ────────────────────────────────────────────────────────

def test_wv_award_granted_with_amount():
    text = "an award is made accordingly in the sum of $500.00 to the claimant."
    assert extract_verdict(text) == "Award Granted ($500.00)"


def test_wv_award_denied_refuse():
    text = "we refuse an award and dismiss the claim entirely."
    result = extract_verdict(text)
    assert "Award Denied" in result


def test_wv_settlement_ratified():
    text = "the settlement is ratified and confirmed in the amount of $1,200.00."
    result = extract_verdict(text)
    assert "Settlement Ratified" in result
    assert "$1,200.00" in result


def test_wv_claim_disallowed():
    text = "for the reasons stated herein, the claim is disallowed."
    assert extract_verdict(text) == "Award Denied"


# ── Connecticut Superior Court ────────────────────────────────────────────────

def test_ct_judgment_for_defendant():
    text = "judgment may be entered in favor of the defendant on all counts."
    assert extract_verdict(text) == "Judgment for Defendant"


def test_ct_judgment_for_plaintiff_with_amount():
    text = "judgment may be entered against the defendants for $3,500 in damages."
    result = extract_verdict(text)
    assert "Judgment for Plaintiff" in result
    assert "3,500" in result


def test_ct_injunction_granted():
    text = "the temporary injunction prayed for may issue."
    assert extract_verdict(text) == "Temporary Injunction Granted"


def test_ct_trust_construction():
    text = "the questions asked of the court are answered as follows: …"
    assert extract_verdict(text) == "Trust Construction - Ruling Issued"


# ── Appellate / General ───────────────────────────────────────────────────────

def test_judgment_reversed():
    text = "the judgment of the court below is reversed with costs."
    assert extract_verdict(text) == "Judgment REVERSED with costs"


def test_judgment_affirmed():
    text = "the judgment is affirmed."
    assert extract_verdict(text) == "Judgment AFFIRMED"


def test_per_curiam_reversed():
    text = "Per Curiam. The judgment is reversed."
    result = extract_verdict(text)
    assert "Per Curiam" in result
    assert "REVERSED" in result


def test_decree_reversed_bill_dismissed():
    text = "the decree is reversed, and the bill dismissed."
    assert extract_verdict(text) == "Decree REVERSED, Bill Dismissed"


# ── Criminal ─────────────────────────────────────────────────────────────────

def test_criminal_guilty():
    text = "the jury found the defendant guilty of the charge."
    assert extract_verdict(text) == "Verdict: GUILTY"


def test_criminal_not_guilty():
    text = "verdict of not guilty was returned by the jury."
    assert extract_verdict(text) == "Verdict: NOT GUILTY"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_text():
    assert extract_verdict("") == "Verdict Unknown"


def test_no_verdict_signal():
    text = "The parties appeared before the court and arguments were heard."
    assert extract_verdict(text) == "Verdict Unknown"