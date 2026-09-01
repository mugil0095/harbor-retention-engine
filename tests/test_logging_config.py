"""configure_logging: the first-time setup path (creating the handler and
formatter) was never actually exercised by the test suite -- by the time
any test file imports harbor code, some other module (e.g. the API's
module-level `configure_logging()` call) had already added a handler to
the root logger, so every test run only ever hit the early-return branch.
"""

from __future__ import annotations

import logging

from harbor.utils.logging_config import configure_logging


def test_configure_logging_adds_a_handler_on_first_call():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers = []
    try:
        configure_logging()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
        assert root.handlers[0].formatter is not None
        assert root.level == logging.INFO
    finally:
        root.handlers = saved_handlers
        root.level = saved_level


def test_configure_logging_respects_custom_level():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers = []
    try:
        configure_logging(level=logging.DEBUG)
        assert root.level == logging.DEBUG
    finally:
        root.handlers = saved_handlers
        root.level = saved_level


def test_configure_logging_is_idempotent_does_not_duplicate_handlers():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers = []
    try:
        configure_logging()
        configure_logging()
        configure_logging()
        assert len(root.handlers) == 1  # re-entrant calls must not pile up handlers
    finally:
        root.handlers = saved_handlers
        root.level = saved_level


def test_configure_logging_formatter_produces_expected_format():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers = []
    try:
        configure_logging()
        record = logging.LogRecord(
            name="harbor.test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        formatted = root.handlers[0].formatter.format(record)
        assert "harbor.test" in formatted
        assert "INFO" in formatted
        assert "hello" in formatted
    finally:
        root.handlers = saved_handlers
        root.level = saved_level
