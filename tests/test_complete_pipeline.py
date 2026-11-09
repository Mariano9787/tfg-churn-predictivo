"""Smoke tests that do not train models or read the heldout target."""
import numpy as np
from src.complete_pipeline import decile_table, model_factory, pipeline


def test_decile_capture_and_precision():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    s = np.arange(10, 0, -1) / 10
    d = decile_table(y, s)
    assert len(d) == 10
    assert d.iloc[0]['precision_cumulative'] == 1
    assert d.iloc[-1]['recall_cumulative'] == 1
    assert d.iloc[-1]['clients'] == 10


def test_model_families_and_feature_toggle():
    for name in ('dummy','logistic','random_forest','xgboost'):
        est, _ = model_factory(name, 0.26)
        assert est is not None
        p = pipeline(name, 0.26, include_sensitive=False)
        cat_cols = [cols for key, _, cols in p.named_steps['preprocessor'].transformers if key == 'categorical'][0]
        assert 'gender' not in cat_cols
        assert 'SeniorCitizen' not in cat_cols
