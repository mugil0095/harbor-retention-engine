"""N: API behavior. Model/data artifacts are injected directly into the
module's in-memory state so these tests do not depend on the training
pipeline having been run first (that dependency is covered separately by
the end-to-end pipeline test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from harbor.serving import api

VALID_PAYLOAD = {
    "account_id": 1,
    "plan_tier": "growth",
    "industry": "retail",
    "employee_size_band": "small",
    "region": "na",
    "tenure_months": 12.0,
    "base_mrr": 199.0,
    "current_mrr": 199.0,
    "avg_login_trailing_3m": 10.0,
    "avg_login_trailing_6m": 12.0,
    "avg_login_trailing_12m": 15.0,
    "avg_api_calls_trailing_3m": 100.0,
    "avg_api_calls_trailing_6m": 110.0,
    "avg_api_calls_trailing_12m": 120.0,
    "avg_feature_adoption_trailing_3m": 3.0,
    "avg_feature_adoption_trailing_6m": 3.5,
    "avg_feature_adoption_trailing_12m": 4.0,
    "login_trend_3v6": -0.1,
    "login_trend_6v12": -0.05,
    "ticket_count_trailing_3m": 1.0,
    "ticket_count_trailing_6m": 2.0,
    "avg_ticket_risk_proba_trailing_3m": 0.3,
    "avg_ticket_risk_proba_trailing_6m": 0.25,
    "avg_ticket_sentiment_trailing_3m": 0.0,
    "avg_ticket_sentiment_trailing_6m": 0.0,
    "low_plan_and_declining": 0,
}


class FakeChurnModel:
    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.37)
        return np.column_stack([1 - p, p])


@pytest.fixture()
def client():
    with TestClient(api.app) as c:
        yield c
    api._state["churn_model"] = None
    api._state["batch_decisions"] = None
    api._state["feature_store"] = None
    api._state["model_version"] = "unloaded"


def test_health_endpoint_reports_status_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_churn_returns_503_when_model_not_loaded(client):
    api._state["churn_model"] = None
    response = client.post("/score/churn", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_score_churn_returns_valid_probability_when_model_loaded(client):
    api._state["churn_model"] = FakeChurnModel()
    api._state["model_version"] = "test-version"
    response = client.post("/score/churn", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["account_id"] == 1
    assert body["model_version"] == "test-version"


def test_score_churn_rejects_invalid_plan_tier(client):
    api._state["churn_model"] = FakeChurnModel()
    bad_payload = {**VALID_PAYLOAD, "plan_tier": "not_a_real_plan"}
    response = client.post("/score/churn", json=bad_payload)
    assert response.status_code == 422


def test_score_churn_rejects_missing_required_field(client):
    api._state["churn_model"] = FakeChurnModel()
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "avg_login_trailing_3m"}
    response = client.post("/score/churn", json=bad_payload)
    assert response.status_code == 422


def test_get_decision_returns_503_when_not_loaded(client):
    api._state["batch_decisions"] = None
    response = client.get("/accounts/1/decision")
    assert response.status_code == 503


def test_get_decision_returns_404_for_unknown_account(client):
    api._state["batch_decisions"] = pd.DataFrame(
        {"account_id": [1], "churn_probability": [0.5], "predicted_uplift": [0.1],
         "expected_value": [10.0], "recommended_action": ["retention_outreach"],
         "scored_at": ["2026-01-01T00:00:00"], "model_version": ["v1"]}
    ).set_index("account_id")
    response = client.get("/accounts/999/decision")
    assert response.status_code == 404


def test_get_decision_returns_correct_fields_for_known_account(client):
    api._state["batch_decisions"] = pd.DataFrame(
        {"account_id": [42], "churn_probability": [0.62], "predicted_uplift": [0.08],
         "expected_value": [153.5], "recommended_action": ["retention_outreach"],
         "scored_at": ["2026-01-01T00:00:00"], "model_version": ["v1"]}
    ).set_index("account_id")
    response = client.get("/accounts/42/decision")
    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == 42
    assert body["churn_probability"] == pytest.approx(0.62)
    assert body["recommended_action"] == "retention_outreach"


def _fake_feature_store_row(account_id: int) -> pd.DataFrame:
    row = {k: v for k, v in VALID_PAYLOAD.items() if k != "account_id"}
    return pd.DataFrame([row], index=pd.Index([account_id], name="account_id"))


def test_churn_score_by_id_returns_503_when_model_not_loaded(client):
    api._state["churn_model"] = None
    api._state["feature_store"] = _fake_feature_store_row(1)
    response = client.get("/accounts/1/churn-score")
    assert response.status_code == 503


def test_churn_score_by_id_returns_503_when_feature_store_not_loaded(client):
    api._state["churn_model"] = FakeChurnModel()
    api._state["feature_store"] = None
    response = client.get("/accounts/1/churn-score")
    assert response.status_code == 503


def test_churn_score_by_id_returns_404_for_unknown_account(client):
    api._state["churn_model"] = FakeChurnModel()
    api._state["feature_store"] = _fake_feature_store_row(1)
    response = client.get("/accounts/999/churn-score")
    assert response.status_code == 404


def test_churn_score_by_id_returns_valid_probability_for_known_account(client):
    api._state["churn_model"] = FakeChurnModel()
    api._state["model_version"] = "test-version"
    api._state["feature_store"] = _fake_feature_store_row(7)
    response = client.get("/accounts/7/churn-score")
    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == 7
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"] == "test-version"


def test_churn_score_by_id_and_by_raw_features_agree_for_the_same_account(client):
    """Both real-time paths should produce the same score for equivalent
    inputs -- they're two ways to reach the same model, not two models."""
    api._state["churn_model"] = FakeChurnModel()
    api._state["feature_store"] = _fake_feature_store_row(3)
    by_id = client.get("/accounts/3/churn-score").json()
    by_features = client.post("/score/churn", json={**VALID_PAYLOAD, "account_id": 3}).json()
    assert by_id["churn_probability"] == pytest.approx(by_features["churn_probability"])
