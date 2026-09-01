"""L: NLP processing."""

from __future__ import annotations

import pandas as pd

from harbor.nlp.ticket_analysis import (
    clean_text,
    compare_nlp_approaches,
    discover_topics,
    make_risk_label,
    vader_compound_scores,
)


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Cancel THIS!! Now.") == "cancel this now"


def test_clean_text_collapses_whitespace():
    assert clean_text("too    many   spaces") == "too many spaces"


def test_make_risk_label_flags_correct_categories():
    categories = pd.Series(["billing", "general_praise", "cancellation_intent", "feature_request"])
    labels = make_risk_label(categories)
    assert labels.tolist() == [1, 0, 1, 0]


def test_vader_scores_positive_and_negative_text_correctly():
    positive = pd.Series(["This is fantastic, we love the product and support team."])
    negative = pd.Series(["This is terrible, we are furious and cancelling immediately."])
    pos_score = vader_compound_scores(positive)[0]
    neg_score = vader_compound_scores(negative)[0]
    assert pos_score > 0
    assert neg_score < 0


def test_classifier_beats_random_on_held_out_tickets(synthetic_tables):
    result, _clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    assert result.classifier_auc > 0.6  # meaningfully better than chance (0.5)
    assert result.n_test > 0


def test_fitted_classifier_predicts_proba_in_valid_range(synthetic_tables):
    _, clf = compare_nlp_approaches(synthetic_tables["support_tickets"], random_seed=123)
    sample_texts = synthetic_tables["support_tickets"]["text"].head(20)
    proba = clf.predict_proba(sample_texts)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_discover_topics_returns_requested_number_of_topics(synthetic_tables):
    topics = discover_topics(synthetic_tables["support_tickets"], n_topics=4, random_seed=123)
    assert len(topics) == 4
    for words in topics.values():
        assert len(words) > 0
