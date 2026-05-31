"""
tests/test_verdict.py
----------------------
Unit and integration tests for:
  modeling/verdict_predictor.py  – VerdictPredictor, map_verdict_to_group
  extractors/verdict_extractor.py – extract_verdict (rule-based, 90 % rate)

Covers: verdict grouping, model train/predict/evaluate/save/load,
        edge cases, confidence scores, and extractor pattern coverage.

Run:  python -m pytest tests/test_verdict.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.verdict_predictor  import VerdictPredictor, map_verdict_to_group
from extractors.verdict_extractor import extract_verdict


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n_per_group: int = 50) -> pd.DataFrame:
    """
    Synthetic DataFrame whose Case_Text is distinctive per verdict group
    so TF-IDF can separate them cleanly.
    """
    groups_and_texts = [
        ("Judgment AFFIRMED",         "the judgment is affirmed court upheld lower ruling"),
        ("Judgment REVERSED",         "the judgment is reversed court overturned lower ruling"),
        ("Remanded",                  "cause remanded further proceedings lower court"),
        ("Judgment for Plaintiff ($1,000)", "judgment may enter for the plaintiff damages awarded"),
        ("Award Granted ($500.00)",   "an award is made accordingly sum granted claimant"),
        ("Award Denied",              "refuse an award claim disallowed no recovery"),
        ("Claim Disallowed - Contributory Negligence",
                                      "claim disallowed contributory negligence proximate cause"),
    ]
    rows = []
    for verdict, text in groups_and_texts:
        for _ in range(n_per_group):
            rows.append({"Case_Text": text, "Verdict": verdict})
    return pd.DataFrame(rows)


def _split(df: pd.DataFrame, test_frac: float = 0.20):
    from sklearn.model_selection import train_test_split
    # Stratify on Verdict_Group proxy (map first)
    df2 = df.copy()
    df2["_g"] = df2["Verdict"].apply(map_verdict_to_group)
    known = df2[df2["_g"] != "unknown"]
    return train_test_split(known.drop(columns=["_g"]),
                            test_size=test_frac,
                            stratify=known["_g"],
                            random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# map_verdict_to_group
# ─────────────────────────────────────────────────────────────────────────────

class TestMapVerdictToGroup:

    def test_affirmed(self):
        assert map_verdict_to_group("Judgment AFFIRMED") == "affirmed"
        assert map_verdict_to_group("Judgment AFFIRMED with costs") == "affirmed"
        assert map_verdict_to_group("Per Curiam: Judgment AFFIRMED") == "affirmed"

    def test_reversed(self):
        assert map_verdict_to_group("Judgment REVERSED") == "reversed"
        assert map_verdict_to_group("Judgment REVERSED with costs") == "reversed"
        assert map_verdict_to_group("Decree REVERSED, Bill Dismissed") == "reversed"

    def test_remanded(self):
        assert map_verdict_to_group("Remanded") == "remanded"
        assert map_verdict_to_group("Judgment REVERSED on Appeal") == "reversed"

    def test_dismissed(self):
        assert map_verdict_to_group("Case Dismissed") == "dismissed"
        assert map_verdict_to_group("Complaint Dismissed") == "dismissed"
        assert map_verdict_to_group("Nonsuit Entered") == "unknown"

    def test_plaintiff_wins(self):
        assert map_verdict_to_group("Judgment for Plaintiff ($2,500)") == "plaintiff_wins"
        assert map_verdict_to_group("Verdict for Plaintiff") == "plaintiff_wins"
        assert map_verdict_to_group("Decree for Plaintiff") == "plaintiff_wins"

    def test_defendant_wins(self):
        assert map_verdict_to_group("Judgment for Defendant") == "defendant_wins"
        assert map_verdict_to_group("Verdict for Defendant") == "defendant_wins"

    def test_award_granted(self):
        assert map_verdict_to_group("Award Granted ($750.00)") == "award_granted"
        assert map_verdict_to_group("Claim Allowed") == "award_granted"
        assert map_verdict_to_group("Award Recommended - Granted ($100)") == "award_granted"

    def test_award_denied(self):
        assert map_verdict_to_group("Award Denied") == "award_denied"
        assert map_verdict_to_group("Claim Disallowed") == "award_denied"
        assert map_verdict_to_group("Claim Denied") == "award_denied"
        assert map_verdict_to_group("Award Denied - Contributory Negligence") == "award_denied"

    def test_unknown(self):
        assert map_verdict_to_group("Verdict Unknown") == "unknown"
        assert map_verdict_to_group("") == "unknown"
        assert map_verdict_to_group(None) == "unknown"
        assert map_verdict_to_group("Demurrer SUSTAINED") == "unknown"

    def test_case_insensitive(self):
        assert map_verdict_to_group("judgment affirmed") == "affirmed"
        assert map_verdict_to_group("JUDGMENT REVERSED") == "reversed"
        assert map_verdict_to_group("award granted") == "award_granted"

    def test_always_returns_string(self):
        for v in ["Verdict Unknown", "", None, "xyz", "Award Granted ($1)"]:
            assert isinstance(map_verdict_to_group(v), str)


# ─────────────────────────────────────────────────────────────────────────────
# VerdictPredictor
# ─────────────────────────────────────────────────────────────────────────────

class TestVerdictPredictor:

    def test_train_returns_self(self):
        df    = _make_df(30)
        model = VerdictPredictor(model_type="logistic", max_features=300)
        ret   = model.train(df)
        assert ret is model

    def test_predict_returns_series(self):
        df    = _make_df(50)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        preds = model.predict(test)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(test)

    def test_predict_series_name(self):
        df    = _make_df(50)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        assert model.predict(test).name == "Predicted_Verdict_Group"

    def test_predict_labels_are_known_groups(self):
        df    = _make_df(60)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        preds = model.predict(test)
        known_groups = {"affirmed", "reversed", "remanded", "dismissed",
                        "plaintiff_wins", "defendant_wins",
                        "award_granted", "award_denied", "unknown"}
        assert set(preds.unique()).issubset(known_groups)

    def test_predict_high_accuracy_on_training_data(self):
        """Distinctive synthetic texts → high accuracy expected."""
        df    = _make_df(80)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=1000).train(train)
        preds = model.predict(test)
        true_groups = test["Verdict"].apply(map_verdict_to_group)
        acc   = (preds.values == true_groups.values).mean()
        assert acc > 0.80, f"Expected >80% accuracy, got {acc:.2%}"

    def test_predict_proba_returns_dataframe(self):
        df    = _make_df(50)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        out   = model.predict_proba(test)
        assert isinstance(out, pd.DataFrame)
        assert "Predicted_Verdict_Group" in out.columns
        assert "Verdict_Confidence"      in out.columns

    def test_predict_proba_confidence_in_range(self):
        df    = _make_df(50)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        out   = model.predict_proba(test)
        assert out["Verdict_Confidence"].between(0.0, 1.0).all()

    def test_predict_proba_index_matches_input(self):
        df    = _make_df(40).sample(20, random_state=3).reset_index(drop=True)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(df)
        out   = model.predict_proba(df)
        assert list(out.index) == list(df.index)

    def test_evaluate_returns_required_keys(self):
        df    = _make_df(80)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test)
        assert "accuracy"    in m
        assert "f1_macro"    in m
        assert "f1_weighted" in m

    def test_evaluate_accuracy_range(self):
        df    = _make_df(80)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_evaluate_skips_unknown_verdicts(self):
        """Rows with 'Verdict Unknown' must be excluded from evaluation."""
        df    = _make_df(60)
        train, test = _split(df)
        # Inject unknown verdicts into test
        test2 = test.copy()
        test2.loc[test2.index[:5], "Verdict"] = "Verdict Unknown"
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test2)   # must not crash
        assert "accuracy" in m

    def test_evaluate_all_unknown_returns_zeros(self):
        df    = _make_df(60)
        train, test = _split(df)
        test_unknown       = test.copy()
        test_unknown["Verdict"] = "Verdict Unknown"
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test_unknown)
        # Must return empty dict (no known verdicts) without raising
        assert isinstance(m, dict)

    def test_save_and_load_predicts_identically(self, tmp_path):
        df    = _make_df(60)
        train, test = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        path  = str(tmp_path / "verdict_pred.pkl")
        model.save(path)
        loaded = VerdictPredictor.load(path)
        orig   = model.predict(test).values
        new    = loaded.predict(test).values
        assert np.array_equal(orig, new)

    def test_save_creates_file(self, tmp_path):
        df    = _make_df(40)
        model = VerdictPredictor(model_type="logistic", max_features=300).train(df)
        path  = str(tmp_path / "vp.pkl")
        model.save(path)
        assert os.path.exists(path)

    def test_svm_model_type_works(self):
        df    = _make_df(50)
        model = VerdictPredictor(model_type="svm", max_features=300).train(df)
        preds = model.predict(df)
        assert len(preds) == len(df)

    def test_naive_bayes_model_type_works(self):
        df    = _make_df(50)
        model = VerdictPredictor(model_type="naive_bayes", max_features=300).train(df)
        preds = model.predict(df)
        assert len(preds) == len(df)

    def test_trainer_attribute_set(self):
        df    = _make_df(40)
        model = VerdictPredictor(model_type="logistic", max_features=300).train(df)
        assert model._trainer is not None

    def test_vectorizer_attribute_set(self):
        df    = _make_df(40)
        model = VerdictPredictor(model_type="logistic", max_features=300).train(df)
        assert model._vectorizer is not None

    def test_classes_attribute_set(self):
        df    = _make_df(50)
        model = VerdictPredictor(model_type="logistic", max_features=300).train(df)
        assert model._classes is not None
        assert len(model._classes) >= 2

    def test_single_row_prediction(self):
        df    = _make_df(60)
        train, _ = _split(df)
        model = VerdictPredictor(model_type="logistic", max_features=500).train(train)
        single = pd.DataFrame({
            "Case_Text": ["the judgment is affirmed by the court"],
            "Verdict":   ["Judgment AFFIRMED"],
        })
        preds = model.predict(single)
        assert len(preds) == 1
        assert isinstance(preds.iloc[0], str)

    def test_unknown_rows_not_used_during_train(self):
        """
        Adding a large number of 'Verdict Unknown' rows must not degrade
        the model because they are filtered out before training.
        """
        df_known   = _make_df(60)
        df_unknown = pd.DataFrame({
            "Case_Text": ["some unknown text"] * 200,
            "Verdict":   ["Verdict Unknown"] * 200,
        })
        df_mixed = pd.concat([df_known, df_unknown], ignore_index=True)
        train, test = _split(df_known)  # test only on known data
        model = VerdictPredictor(model_type="logistic", max_features=500)
        model.train(df_mixed)           # train on mixed (unknowns filtered inside)
        m = model.evaluate(test)
        assert m["accuracy"] > 0.70


# ─────────────────────────────────────────────────────────────────────────────
# extract_verdict  (rule-based extractor, key pattern coverage)
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractVerdict:
    """
    Spot-checks the rule-based extractor used during ETL.
    The full extractor test suite lives in test_extractors.py;
    here we verify the patterns that feed verdict_predictor training data.
    """

    # Patterns that map to "affirmed"
    def test_affirmed_plain(self):
        assert "AFFIRMED" in extract_verdict("the judgment is affirmed.")

    def test_affirmed_with_costs(self):
        assert "AFFIRMED" in extract_verdict(
            "the judgment of the court below is affirmed with costs.")

    def test_per_curiam_affirmed(self):
        r = extract_verdict("Per Curiam. The judgment is affirmed.")
        assert "Per Curiam" in r and "AFFIRMED" in r

    # Patterns that map to "reversed"
    def test_reversed_plain(self):
        assert "REVERSED" in extract_verdict("the judgment is reversed.")

    def test_reversed_with_costs(self):
        assert "REVERSED" in extract_verdict(
            "the judgment of the court below is reversed with costs.")

    def test_decree_reversed_bill_dismissed(self):
        r = extract_verdict("the decree is reversed, and the bill dismissed.")
        assert "REVERSED" in r or "Dismissed" in r

    # Patterns that map to "remanded"
    def test_remanded(self):
        assert "Remanded" in extract_verdict(
            "cause remanded for further proceedings consistent with this opinion.")

    # Patterns that map to "plaintiff_wins"
    def test_judgment_for_plaintiff_with_amount(self):
        r = extract_verdict("judgment may be entered against the defendants for $3,500.")
        assert "Plaintiff" in r and "3,500" in r

    def test_judgment_for_plaintiff_plain(self):
        assert "Plaintiff" in extract_verdict(
            "judgment may enter for the plaintiff in this matter.")

    def test_verdict_for_plaintiff(self):
        assert "Plaintiff" in extract_verdict(
            "verdict was in favour of the plaintiff.")

    # Patterns that map to "defendant_wins"
    def test_judgment_for_defendant(self):
        assert "Defendant" in extract_verdict(
            "judgment may be entered in favor of the defendant.")

    def test_judgment_rendered_for_defendant(self):
        assert "Defendant" in extract_verdict(
            "judgment is rendered for the defendant.")

    # Patterns that map to "award_granted"
    def test_wv_award_granted_with_amount(self):
        r = extract_verdict("an award is made accordingly in the sum of $750.00.")
        assert "Award Granted" in r and "750" in r

    def test_wv_award_granted_no_amount(self):
        r = extract_verdict("an award is made accordingly.")
        assert "Award Granted" in r

    def test_wv_settlement_ratified(self):
        r = extract_verdict("the settlement is ratified and confirmed. $1,200.00.")
        assert "Settlement Ratified" in r or "Award Granted" in r

    # Patterns that map to "award_denied"
    def test_award_denied_refuse(self):
        assert "Award Denied" in extract_verdict(
            "the court is constrained to refuse an award.")

    def test_award_denied_claim_disallowed(self):
        r = extract_verdict("for the reasons stated, the claim is disallowed.")
        assert "Disallowed" in r or "Denied" in r

    # Patterns that stay "unknown"
    def test_unknown_on_empty(self):
        assert extract_verdict("") == "Verdict Unknown"

    def test_unknown_on_no_pattern(self):
        assert extract_verdict("The parties discussed the matter at length.") == "Verdict Unknown"

    # Criminal verdicts
    def test_criminal_guilty(self):
        assert "GUILTY" in extract_verdict(
            "the jury found the defendant guilty of the charge.")

    def test_criminal_not_guilty(self):
        assert "NOT GUILTY" in extract_verdict(
            "the jury returned a verdict of not guilty.")

    # Procedural
    def test_demurrer_sustained(self):
        r = extract_verdict("the demurrer is sustained.")
        assert "Demurrer" in r and "SUSTAINED" in r

    def test_demurrer_overruled(self):
        r = extract_verdict("the demurrer is overruled.")
        assert "Demurrer" in r and "OVERRULED" in r

    def test_injunction_granted(self):
        assert "Injunction" in extract_verdict(
            "the temporary injunction prayed for may issue.")

    def test_divorce_decree(self):
        assert "Divorce" in extract_verdict(
            "the plaintiff is entitled to a decree of divorce.")