"""NLP on support-ticket text.

Two approaches are implemented and compared head-to-head, as required for
any NLP component in this project:

1. **Simple baseline** -- VADER lexicon sentiment (no training data needed)
   thresholded to flag "at-risk" tickets.
2. **Stronger approach** -- a supervised TF-IDF + logistic regression
   classifier trained to recognize risk-flavored tickets (billing / bug /
   cancellation-intent) from text alone.

The classifier predicts ticket *category-derived risk*, not churn -- it
never sees the churn label, so using it as a feature source for the churn
model introduces no target leakage. It is a text-pattern extractor, the
NLP equivalent of a hand-built keyword feature but learned from data.

An unsupervised NMF topic model is included for exploratory/communication
value (surfacing latent complaint themes), consistent with "topic discovery"
as a genuinely useful, separate technique from the supervised classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RISK_CATEGORIES = {"cancellation_intent", "bug_report", "billing"}

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def make_risk_label(category: pd.Series) -> pd.Series:
    return category.isin(RISK_CATEGORIES).astype(int)


_analyzer = SentimentIntensityAnalyzer()


def vader_compound_scores(texts: pd.Series) -> np.ndarray:
    return np.array([_analyzer.polarity_scores(t)["compound"] for t in texts])


def vader_negative_flag(texts: pd.Series, threshold: float = -0.05) -> np.ndarray:
    return (vader_compound_scores(texts) < threshold).astype(int)


@dataclass
class NlpComparisonResult:
    baseline_accuracy: float
    baseline_f1: float
    baseline_auc: float
    classifier_accuracy: float
    classifier_f1: float
    classifier_auc: float
    n_test: int


class TicketRiskClassifier:
    """TF-IDF + logistic regression classifier for at-risk ticket text."""

    def __init__(self, random_seed: int = 42, max_features: int = 2000) -> None:
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=random_seed)
        self._fitted = False

    def fit(self, texts: pd.Series, labels: pd.Series) -> TicketRiskClassifier:
        cleaned = texts.map(clean_text)
        X = self.vectorizer.fit_transform(cleaned)
        self.model.fit(X, labels)
        self._fitted = True
        return self

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TicketRiskClassifier must be fit before predicting.")
        cleaned = texts.map(clean_text)
        X = self.vectorizer.transform(cleaned)
        return self.model.predict_proba(X)[:, 1]


def compare_nlp_approaches(
    tickets: pd.DataFrame, random_seed: int = 42
) -> tuple[NlpComparisonResult, TicketRiskClassifier]:
    """Fits the supervised classifier on a train split and compares it
    against the VADER baseline on a held-out test split of tickets."""
    labels = make_risk_label(tickets["category"])
    train_df, test_df, y_train, y_test = train_test_split(
        tickets, labels, test_size=0.25, random_state=random_seed, stratify=labels
    )

    baseline_pred = vader_negative_flag(test_df["text"])
    baseline_score = vader_compound_scores(test_df["text"]) * -1  # higher = more "risk-like"

    clf = TicketRiskClassifier(random_seed=random_seed).fit(train_df["text"], y_train)
    clf_proba = clf.predict_proba(test_df["text"])
    clf_pred = (clf_proba >= 0.5).astype(int)

    result = NlpComparisonResult(
        baseline_accuracy=accuracy_score(y_test, baseline_pred),
        baseline_f1=f1_score(y_test, baseline_pred, zero_division=0),
        baseline_auc=roc_auc_score(y_test, baseline_score),
        classifier_accuracy=accuracy_score(y_test, clf_pred),
        classifier_f1=f1_score(y_test, clf_pred, zero_division=0),
        classifier_auc=roc_auc_score(y_test, clf_proba),
        n_test=len(test_df),
    )

    # refit on the full corpus for downstream feature extraction use
    final_clf = TicketRiskClassifier(random_seed=random_seed).fit(tickets["text"], labels)
    return result, final_clf


def discover_topics(
    tickets: pd.DataFrame, n_topics: int = 6, n_top_words: int = 8, random_seed: int = 42
) -> dict[int, list[str]]:
    """Unsupervised NMF topic discovery over ticket text, for exploration
    and communication (not used as a model feature)."""
    cleaned = tickets["text"].map(clean_text)
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(cleaned)
    nmf = NMF(n_components=n_topics, random_state=random_seed, init="nndsvda", max_iter=400)
    nmf.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, component in enumerate(nmf.components_):
        top_indices = component.argsort()[::-1][:n_top_words]
        topics[topic_idx] = [feature_names[i] for i in top_indices]
    return topics
