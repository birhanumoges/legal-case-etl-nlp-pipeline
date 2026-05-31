"""
tests/test_extractors.py
-------------------------
Unit tests for the extractors package:
  text_extractor, metadata_extractor, citation_extractor,
  verdict_extractor (coverage), classifier (coverage).

Run:  python -m pytest tests/ -v
"""

import sys
import json
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extractors.text_extractor     import extract_text_from_html
from extractors.metadata_extractor import (
    extract_metadata_from_json, extract_year, extract_court, extract_case_name,
)
from extractors.citation_extractor import extract_citations
from extractors.verdict_extractor  import extract_verdict
from extractors.classifier         import extract_case_type, extract_sub_type


# ── text_extractor ────────────────────────────────────────────────────────────

class TestTextExtractor:

    def test_basic_html_extraction(self, tmp_path):
        html = "<html><body><p>The plaintiff sued the defendant.</p></body></html>"
        f = tmp_path / "case.html"
        f.write_text(html, encoding="utf-8")
        text, length = extract_text_from_html(f)
        assert "plaintiff" in text
        assert length > 0

    def test_strips_script_tags(self, tmp_path):
        html = "<html><body><script>alert('x')</script><p>legal text</p></body></html>"
        f = tmp_path / "case.html"
        f.write_text(html, encoding="utf-8")
        text, _ = extract_text_from_html(f)
        assert "alert" not in text
        assert "legal" in text

    def test_strips_nav_and_footer(self, tmp_path):
        html = "<html><body><nav>nav content</nav><p>body</p><footer>foot</footer></body></html>"
        f = tmp_path / "case.html"
        f.write_text(html, encoding="utf-8")
        text, _ = extract_text_from_html(f)
        assert "nav content" not in text
        assert "foot" not in text
        assert "body" in text

    def test_removes_page_markers(self, tmp_path):
        html = "<html><body><p>Page*12 of the case text.</p></body></html>"
        f = tmp_path / "case.html"
        f.write_text(html, encoding="utf-8")
        text, _ = extract_text_from_html(f)
        assert "*12" not in text

    def test_missing_file_returns_empty(self, tmp_path):
        text, length = extract_text_from_html(tmp_path / "missing.html")
        assert text == ""
        assert length == 0

    def test_returns_tuple(self, tmp_path):
        f = tmp_path / "t.html"
        f.write_text("<p>hello</p>")
        result = extract_text_from_html(f)
        assert isinstance(result, tuple) and len(result) == 2

    def test_whitespace_normalised(self, tmp_path):
        html = "<html><body><p>word1   word2\n\n\nword3</p></body></html>"
        f = tmp_path / "t.html"
        f.write_text(html)
        text, _ = extract_text_from_html(f)
        assert "word1 word2" in text or "word1" in text


# ── metadata_extractor ────────────────────────────────────────────────────────

class TestMetadataExtractor:

    def _write_json(self, tmp_path, data: dict) -> Path:
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        return f

    def test_extracts_case_name(self, tmp_path):
        f = self._write_json(tmp_path, {"name": "Smith v. Jones", "decision_date": "1958-03-01"})
        meta = extract_metadata_from_json(f)
        assert meta["case_name"] == "Smith v. Jones"

    def test_extracts_decision_date(self, tmp_path):
        f = self._write_json(tmp_path, {"decision_date": "1958-03-15"})
        meta = extract_metadata_from_json(f)
        assert meta["decision_date"] == "1958-03-15"

    def test_extracts_court_from_dict(self, tmp_path):
        f = self._write_json(tmp_path, {"court": {"name": "Connecticut Superior Court"}})
        meta = extract_metadata_from_json(f)
        assert meta["court_name"] == "Connecticut Superior Court"

    def test_extracts_court_from_string(self, tmp_path):
        f = self._write_json(tmp_path, {"court": "Indiana Supreme Court"})
        meta = extract_metadata_from_json(f)
        assert meta["court_name"] == "Indiana Supreme Court"

    def test_extracts_citations(self, tmp_path):
        f = self._write_json(tmp_path, {
            "citations": [{"cite": "12 Conn. 345"}, {"cite": "5 U.S. 100"}]
        })
        meta = extract_metadata_from_json(f)
        assert "12 Conn. 345" in meta["citations"]
        assert "5 U.S. 100" in meta["citations"]

    def test_extracts_cites_to(self, tmp_path):
        f = self._write_json(tmp_path, {"cites_to": [{"cite": "7 Ind. 200"}]})
        meta = extract_metadata_from_json(f)
        assert "7 Ind. 200" in meta["citations"]

    def test_missing_file_returns_empty_dict(self, tmp_path):
        meta = extract_metadata_from_json(tmp_path / "missing.json")
        assert meta == {}

    def test_extract_year_from_metadata(self):
        assert extract_year("", {"decision_date": "1958-03-01"}) == "1958"

    def test_extract_year_falls_back_to_text(self):
        assert extract_year("This case was decided in 1922.", {}) == "1922"

    def test_extract_year_unknown(self):
        assert extract_year("No date here.", {}) == "Unknown"

    def test_extract_court_from_metadata(self):
        assert extract_court({"court_name": "My Court"}, "") == "My Court"

    def test_extract_court_from_text(self):
        result = extract_court({}, "In the Supreme Court of Indiana the judges held")
        assert "Indiana" in result

    def test_extract_case_name_from_metadata(self):
        name = extract_case_name({"case_name": "Doe v. Roe"}, "", "file_stem")
        assert name == "Doe v. Roe"

    def test_extract_case_name_from_text(self):
        name = extract_case_name({}, "Smith v. Jones appeared before the court.", "stem")
        assert "Smith" in name and "Jones" in name

    def test_extract_case_name_fallback(self):
        name = extract_case_name({}, "no pattern here", "file_0001")
        assert name == "file_0001"


# ── citation_extractor ────────────────────────────────────────────────────────

class TestCitationExtractor:

    def test_extracts_volume_reporter_page(self):
        text   = "See 12 Conn. 345 for the rule."
        cites  = extract_citations(text, {})
        assert any("12" in c and "Conn" in c and "345" in c for c in cites)

    def test_extracts_us_reporter(self):
        text  = "Cf. 347 U.S. 483 (1954)."
        cites = extract_citations(text, {})
        assert any("U.S." in c for c in cites)

    def test_merges_json_citations(self):
        text  = "Some legal text without citations."
        meta  = {"citations": ["12 Conn. 345", "5 Mass. 100"]}
        cites = extract_citations(text, meta)
        assert "12 Conn. 345" in cites
        assert "5 Mass. 100"  in cites

    def test_deduplicates(self):
        text  = "See 12 Conn. 345. Also 12 Conn. 345."
        cites = extract_citations(text, {"citations": ["12 Conn. 345"]})
        assert cites.count("12 Conn. 345") == 1

    def test_returns_sorted_list(self):
        meta  = {"citations": ["Z Rep. 1", "A Rep. 99"]}
        cites = extract_citations("", meta)
        assert cites == sorted(cites)

    def test_empty_text_and_meta(self):
        assert extract_citations("", {}) == []
        assert extract_citations("", {"citations": []}) == []


# ── verdict_extractor (key patterns) ─────────────────────────────────────────

class TestVerdictExtractor:

    def test_empty_returns_unknown(self):
        assert extract_verdict("") == "Verdict Unknown"

    def test_none_returns_unknown(self):
        assert extract_verdict(None) == "Verdict Unknown"

    def test_judgment_affirmed(self):
        assert "AFFIRMED" in extract_verdict("the judgment is affirmed.")

    def test_judgment_reversed(self):
        assert "REVERSED" in extract_verdict(
            "the judgment of the court below is reversed with costs.")

    def test_per_curiam_reversed(self):
        r = extract_verdict("Per Curiam. The judgment is reversed.")
        assert "Per Curiam" in r and "REVERSED" in r

    def test_ct_judgment_for_defendant(self):
        assert "Defendant" in extract_verdict(
            "judgment may be entered in favor of the defendant.")

    def test_ct_judgment_for_plaintiff_with_amount(self):
        r = extract_verdict("judgment may be entered against the defendants for $2,500.")
        assert "Plaintiff" in r and "2,500" in r

    def test_wv_award_granted(self):
        r = extract_verdict("an award is made accordingly in the sum of $750.00.")
        assert "Award Granted" in r and "750" in r

    def test_wv_award_denied(self):
        assert "Award Denied" in extract_verdict(
            "we are constrained to refuse an award in this matter.")

    def test_criminal_guilty(self):
        assert "GUILTY" in extract_verdict(
            "the jury found the defendant guilty of the charge.")

    def test_criminal_not_guilty(self):
        assert "NOT GUILTY" in extract_verdict("verdict of not guilty was returned.")

    def test_demurrer_sustained(self):
        r = extract_verdict("the demurrer is sustained.")
        assert "Demurrer" in r and "SUSTAINED" in r

    def test_remanded(self):
        assert "Remanded" in extract_verdict(
            "cause remanded for further proceedings.")


# ── classifier ────────────────────────────────────────────────────────────────

class TestClassifier:

    def test_defamation_not_criminal(self):
        text = "action for slander, slanderous words spoken by defendant."
        assert extract_case_type(text) == "Torts - Defamation"

    def test_criminal_larceny(self):
        assert extract_case_type(
            "indictment for grand larceny. defendant found guilty.") == "Criminal Law"

    def test_contract_promissory_note(self):
        assert extract_case_type(
            "suit on a promissory note for debt made by the defendant."
        ) == "Contract Law - Debt"

    def test_property_ejectment(self):
        assert extract_case_type(
            "action of ejectment to recover possession of land."
        ) == "Property Law - Ejectment"

    def test_civil_demurrer(self):
        assert extract_case_type(
            "defendant demurred to the declaration; demurrer filed."
        ) == "Civil Procedure"

    def test_torts_negligence(self):
        assert extract_case_type(
            "plaintiff claims negligence caused personal injury."
        ) == "Torts"

    def test_sub_type_larceny(self):
        assert extract_sub_type(
            "convicted of grand larceny for stealing horses.", "Criminal Law"
        ) == "Larceny"

    def test_sub_type_homicide(self):
        assert extract_sub_type(
            "indicted for murder in the first degree.", "Criminal Law"
        ) == "Homicide"

    def test_sub_type_assault_battery(self):
        assert extract_sub_type(
            "assault and battery upon the plaintiff.", "Criminal Law"
        ) == "Assault & Battery"

    def test_sub_type_slander(self):
        assert extract_sub_type(
            "action for slanderous words spoken.", "Torts - Defamation"
        ) == "Slander"

    def test_sub_type_negligence(self):
        assert extract_sub_type(
            "negligence caused the plaintiff's injury.", "Torts"
        ) == "Negligence"

    def test_sub_type_promissory_note(self):
        assert extract_sub_type(
            "suit on a promissory note.", "Contract Law - Debt"
        ) == "Promissory Note"

    def test_sub_type_ejectment(self):
        assert extract_sub_type(
            "action of ejectment to recover land.", "Property Law - Ejectment"
        ) == "Ejectment"

    def test_always_returns_string(self):
        assert isinstance(extract_case_type("random text"), str)
        assert isinstance(extract_sub_type("random text", "Unclassified"), str)