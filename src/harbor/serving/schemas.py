"""Pydantic request/response models for the scoring API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AccountFeatures(BaseModel):
    account_id: int
    plan_tier: str = Field(pattern="^(starter|growth|enterprise)$")
    industry: str
    employee_size_band: str
    region: str
    tenure_months: float
    base_mrr: float
    current_mrr: float
    avg_login_trailing_3m: float
    avg_login_trailing_6m: float
    avg_login_trailing_12m: float
    avg_api_calls_trailing_3m: float
    avg_api_calls_trailing_6m: float
    avg_api_calls_trailing_12m: float
    avg_feature_adoption_trailing_3m: float
    avg_feature_adoption_trailing_6m: float
    avg_feature_adoption_trailing_12m: float
    login_trend_3v6: float
    login_trend_6v12: float
    ticket_count_trailing_3m: float
    ticket_count_trailing_6m: float
    avg_ticket_risk_proba_trailing_3m: float
    avg_ticket_risk_proba_trailing_6m: float
    avg_ticket_sentiment_trailing_3m: float
    avg_ticket_sentiment_trailing_6m: float
    low_plan_and_declining: int


class ChurnScoreResponse(BaseModel):
    account_id: int
    churn_probability: float
    model_version: str


class DecisionResponse(BaseModel):
    account_id: int
    churn_probability: float
    predicted_uplift: float
    expected_value: float
    recommended_action: str
    scored_at: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    churn_model_loaded: bool
    batch_decisions_loaded: bool
    feature_store_loaded: bool
