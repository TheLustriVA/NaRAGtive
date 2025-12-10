from __future__ import annotations

import json
from typing import Any

import pytest
import polars as pl
import numpy as np


@pytest.fixture
def sample_embedding_scores() -> list[float]:
    """Sample embedding similarity scores."""
    return [0.92, 0.87, 0.81, 0.75, 0.68]


@pytest.fixture
def sample_rerank_scores() -> list[float]:
    """Sample BGE reranker scores."""
    return [0.95, 0.88, 0.76, 0.71, 0.65]


@pytest.fixture
def sample_metadata_list() -> list[dict[str, Any]]:
    """Sample scene metadata."""
    return [
        {
            "scene_id": "scene_0001_2025-11-10",
            "date_iso": "2025-11-10",
            "location": "bridge",
            "pov_character": "Venice",
            "characters_present": json.dumps(["Venice", "Heidi"]),
            "ships": ["ThunderChild"],
            "events": ["undocking"],
            "tone": "tense",
            "emotional_intensity": 0.6,
            "action_level": 0.4,
        },
    ]


@pytest.fixture
def sample_search_results(
    sample_embedding_scores: list[float],
    sample_metadata_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sample vector store search results."""
    return {
        "ids": ["scene_0001_2025-11-10"],
        "documents": ["The Admiral stepped onto the bridge."],
        "metadatas": sample_metadata_list[:1],
        "scores": sample_embedding_scores[:1],
        "distances": [[1 - sample_embedding_scores[0]]],
    }


@pytest.fixture
def sample_reranked_results(
    sample_search_results: dict[str, Any],
    sample_rerank_scores: list[float],
) -> dict[str, Any]:
    """Sample two-stage retrieval results."""
    result = sample_search_results.copy()
    result["embedding_scores"] = result.pop("scores")
    result["rerank_scores"] = sample_rerank_scores[:1]
    result["reranked"] = True
    result["rerank_method"] = "bge-v2-m3"
    result["initial_search_count"] = 50
    return result
