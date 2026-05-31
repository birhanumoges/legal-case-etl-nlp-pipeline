"""
tests/test_classifier.py
-------------------------
Unit and integration tests for:
  modeling/case_classifier.py   – CaseClassifier
  modeling/subtype_classifier.py – SubtypeClassifier

Covers: train, predict, predict_proba, evaluate, save, load,
        cross_validate, edge cases, and persistence.

Run:  python -m pytest tests/test_classifier.py -v
"""

import os
import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.case_classifier    import CaseClassifier
from modeling.subtype_classifier import SubtypeClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n_per_class: int = 60) -> pd.DataFrame:
    """
    Synthetic legal DataFrame with three well-separated Case_Type classes
    and matching Sub_Type labels.  Texts are intentionally distinctive so
    TF-IDF can separate them cleanly.
    """
    criminal_texts = (
        ["defendant indicted grand larceny theft stolen goods guilty felony"] * n_per_class
    )
    torts_texts = (
        ["plaintiff negligence injury damages trespass nuisance conversion"] * n_per_class
    )
    contract_texts = (
        ["promissory note debt bond payment obligation consideration usury"] * n_per_class
    )

    criminal_sub  = ["Larceny"]          * n_per_class
    torts_sub     = ["Negligence"]       * n_per_class
    contract_sub  = ["Promissory Note"]  * n_per_class

    texts      = criminal_texts  + torts_texts    + contract_texts
    case_types = (["Criminal Law"] * n_per_class
                  + ["Torts"]       * n_per_class
                  + ["Contract Law - Debt"] * n_per_class)
    sub_types  = criminal_sub + torts_sub + contract_sub

    return pd.DataFrame({
        "Case_Text": texts,
        "Case_Type": case_types,
        "Sub_Type":  sub_types,
    })


def _split(df: pd.DataFrame, test_frac: float = 0.20):
    """Simple 80/20 split keeping class balance."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_frac,
                            stratify=df["Case_Type"], random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# CaseClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestCaseClassifier:

    def test_train_returns_self(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500)
        ret   = model.train(df)
        assert ret is model

    def test_predict_returns_series(self):
        df = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        preds = model.predict(df)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(df)

    def test_predict_labels_are_valid(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        preds = model.predict(df)
        valid = set(df["Case_Type"].unique())
        assert set(preds.unique()).issubset(valid)

    def test_predict_high_accuracy_on_training_data(self):
        """Clean synthetic data should train to near-perfect accuracy."""
        df    = _make_df(80)
        train, test = _split(df)
        model = CaseClassifier(model_type="logistic", max_features=1000).train(train)
        preds = model.predict(test)
        acc   = (preds.values == test["Case_Type"].values).mean()
        assert acc > 0.85, f"Expected >85% accuracy on clean data, got {acc:.2%}"

    def test_predict_proba_returns_dataframe(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        proba = model.predict_proba(df)
        assert isinstance(proba, pd.DataFrame)
        assert "Predicted_Case_Type"    in proba.columns
        assert "Case_Type_Confidence"   in proba.columns

    def test_predict_proba_confidence_in_range(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        proba = model.predict_proba(df)
        assert proba["Case_Type_Confidence"].between(0.0, 1.0).all()

    def test_predict_proba_index_matches_input(self):
        df    = _make_df().sample(20, random_state=1).reset_index(drop=True)
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        proba = model.predict_proba(df)
        assert list(proba.index) == list(df.index)

    def test_evaluate_returns_required_keys(self):
        df    = _make_df(80)
        train, test = _split(df)
        model = CaseClassifier(model_type="logistic", max_features=1000).train(train)
        m     = model.evaluate(test)
        assert "accuracy"    in m
        assert "f1_macro"    in m
        assert "f1_weighted" in m

    def test_evaluate_accuracy_range(self):
        df    = _make_df(80)
        train, test = _split(df)
        model = CaseClassifier(model_type="logistic", max_features=1000).train(train)
        m     = model.evaluate(test)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_save_and_load_predicts_identically(self, tmp_path):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        path  = str(tmp_path / "case_clf.pkl")
        model.save(path)
        loaded = CaseClassifier.load(path)
        orig_preds   = model.predict(df).values
        loaded_preds = loaded.predict(df).values
        assert np.array_equal(orig_preds, loaded_preds)

    def test_save_creates_file(self, tmp_path):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        path  = str(tmp_path / "case_clf.pkl")
        model.save(path)
        assert os.path.exists(path)

    def test_cross_validate_returns_dict(self):
        df    = _make_df(90)
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        cv    = model.cross_validate(df, cv=3)
        assert "mean" in cv and "std" in cv and "scores" in cv

    def test_cross_validate_mean_in_range(self):
        df    = _make_df(90)
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        cv    = model.cross_validate(df, cv=3)
        assert 0.0 <= cv["mean"] <= 1.0

    def test_cross_validate_score_count(self):
        df    = _make_df(90)
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        cv    = model.cross_validate(df, cv=3)
        assert len(cv["scores"]) == 3

    def test_svm_model_type_works(self):
        df    = _make_df()
        model = CaseClassifier(model_type="svm", max_features=500).train(df)
        preds = model.predict(df)
        assert len(preds) == len(df)

    def test_naive_bayes_model_type_works(self):
        df    = _make_df()
        model = CaseClassifier(model_type="naive_bayes", max_features=500).train(df)
        preds = model.predict(df)
        assert len(preds) == len(df)

    def test_classes_attribute_set_after_train(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        assert model._classes is not None
        assert len(model._classes) == 3

    def test_vectorizer_attribute_set_after_train(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        assert model._vectorizer is not None

    def test_single_row_prediction(self):
        df    = _make_df(60)
        train, _ = _split(df)
        model = CaseClassifier(model_type="logistic", max_features=500).train(train)
        single = pd.DataFrame({"Case_Text": ["defendant indicted larceny felony"]})
        preds  = model.predict(single)
        assert len(preds) == 1
        assert isinstance(preds.iloc[0], str)

    def test_predict_on_unseen_text_returns_valid_label(self):
        df    = _make_df()
        model = CaseClassifier(model_type="logistic", max_features=500).train(df)
        new   = pd.DataFrame({"Case_Text": ["brand new unseen text about law"]})
        preds = model.predict(new)
        valid = set(df["Case_Type"].unique())
        assert preds.iloc[0] in valid


# ─────────────────────────────────────────────────────────────────────────────
# SubtypeClassifier
# ─────────────────────────────────────────────────────────────────────────────

class TestSubtypeClassifier:

    def test_train_returns_self(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500)
        ret   = model.train(df)
        assert ret is model

    def test_sub_models_created_per_case_type(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        # Should have 3 sub-models (one per Case_Type group)
        assert len(model._models) == 3

    def test_predict_returns_series(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        preds = model.predict(df, case_type_col="Case_Type")
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(df)

    def test_predict_series_name(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        preds = model.predict(df, case_type_col="Case_Type")
        assert preds.name == "Predicted_Sub_Type"

    def test_predict_high_accuracy_on_training_data(self):
        df    = _make_df(100)
        train, test = _split(df)
        model = SubtypeClassifier(model_type="logistic", max_features=1000).train(train)
        preds = model.predict(test, case_type_col="Case_Type")
        acc   = (preds.values == test["Sub_Type"].values).mean()
        assert acc > 0.80, f"Expected >80% sub-type accuracy, got {acc:.2%}"

    def test_predict_full_returns_dataframe(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        out   = model.predict_full(df, case_type_col="Case_Type")
        assert isinstance(out, pd.DataFrame)
        assert "Predicted_Sub_Type"  in out.columns
        assert "Sub_Type_Confidence" in out.columns

    def test_predict_full_confidence_in_range(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        out   = model.predict_full(df, case_type_col="Case_Type")
        assert out["Sub_Type_Confidence"].between(0.0, 1.0).all()

    def test_predict_full_index_matches_input(self):
        df    = _make_df(60).sample(20, random_state=7).reset_index(drop=True)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        out   = model.predict_full(df, case_type_col="Case_Type")
        assert list(out.index) == list(df.index)

    def test_unseen_case_type_returns_unclassified(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        new   = pd.DataFrame({
            "Case_Text": ["some legal text"],
            "Case_Type": ["Family Law - Dower"],   # not in training data
        })
        preds = model.predict(new, case_type_col="Case_Type")
        assert preds.iloc[0] == "Unclassified"

    def test_evaluate_returns_required_keys(self):
        df    = _make_df(100)
        train, test = _split(df)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test)
        assert "accuracy"       in m
        assert "f1_macro"       in m
        assert "f1_weighted"    in m
        assert "num_sub_models" in m

    def test_evaluate_num_sub_models(self):
        df    = _make_df(100)
        train, test = _split(df)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test)
        assert m["num_sub_models"] == 3

    def test_evaluate_accuracy_range(self):
        df    = _make_df(100)
        train, test = _split(df)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(train)
        m     = model.evaluate(test)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_save_and_load_predicts_identically(self, tmp_path):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        path  = str(tmp_path / "sub_clf.pkl")
        model.save(path)
        loaded      = SubtypeClassifier.load(path)
        orig_preds  = model.predict(df, case_type_col="Case_Type").values
        loaded_preds= loaded.predict(df, case_type_col="Case_Type").values
        assert np.array_equal(orig_preds, loaded_preds)

    def test_save_creates_file(self, tmp_path):
        df    = _make_df(60)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        path  = str(tmp_path / "sub_clf.pkl")
        model.save(path)
        assert os.path.exists(path)

    def test_groups_below_min_samples_skipped(self):
        """Groups with fewer than MIN_SAMPLES=10 rows should not get a sub-model."""
        df_large = _make_df(60)
        # Add a tiny fourth group (2 rows) – should be skipped
        tiny = pd.DataFrame({
            "Case_Text": ["rare legal text"] * 2,
            "Case_Type": ["Family Law - Dower"] * 2,
            "Sub_Type":  ["Dower Rights"] * 2,
        })
        df = pd.concat([df_large, tiny], ignore_index=True)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        assert "Family Law - Dower" not in model._models

    def test_predict_uses_predicted_case_type_col(self):
        """When the column is named 'Predicted_Case_Type' the model must use it."""
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        df2   = df.copy()
        df2["Predicted_Case_Type"] = df2["Case_Type"]
        preds_a = model.predict(df2, case_type_col="Case_Type")
        preds_b = model.predict(df2, case_type_col="Predicted_Case_Type")
        # Both should produce the same predictions
        assert np.array_equal(preds_a.values, preds_b.values)

    def test_single_row_prediction(self):
        df    = _make_df(80)
        model = SubtypeClassifier(model_type="logistic", max_features=500).train(df)
        single = pd.DataFrame({
            "Case_Text": ["defendant indicted for larceny theft stolen"],
            "Case_Type": ["Criminal Law"],
        })
        preds = model.predict(single, case_type_col="Case_Type")
        assert len(preds) == 1

    def test_pipeline_integration_case_then_subtype(self):
        """
        Integration: CaseClassifier predicts Case_Type, then SubtypeClassifier
        uses that prediction to predict Sub_Type.
        """
        df    = _make_df(100)
        train, test = _split(df)

        case_model = CaseClassifier(model_type="logistic", max_features=500).train(train)
        sub_model  = SubtypeClassifier(model_type="logistic", max_features=500).train(train)

        # Step 1: predict Case_Type
        test2 = test.copy()
        test2["Predicted_Case_Type"] = case_model.predict(test)

        # Step 2: predict Sub_Type using predicted Case_Type
        sub_preds = sub_model.predict(test2, case_type_col="Predicted_Case_Type")

        assert isinstance(sub_preds, pd.Series)
        assert len(sub_preds) == len(test)
        # Most predictions should not be "Unclassified" on clean data
        unclassified_rate = (sub_preds == "Unclassified").mean()
        assert unclassified_rate < 0.30, (
            f"Too many Unclassified in integration test: {unclassified_rate:.1%}"
        )