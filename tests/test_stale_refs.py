"""Guard against the dead _slug helper reappearing in chunkers."""

from __future__ import annotations

import importlib


def test_slug_helper_removed():
    chunkers = importlib.import_module("rag_bench.pipeline.components.chunkers")
    assert not hasattr(chunkers, "_slug")
