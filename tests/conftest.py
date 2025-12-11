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
        {
            "scene_id": "scene_0002_2025-11-10",
            "date_iso": "2025-11-10",
            "location": "medbay",
            "pov_character": "Heidi",
            "characters_present": json.dumps(["Heidi", "Petrova"]),
            "ships": ["ThunderChild"],
            "events": ["scanning"],
            "tone": "emotional",
            "emotional_intensity": 0.8,
            "action_level": 0.1,
        },
    ]


@pytest.fixture
def sample_search_results(
    sample_embedding_scores: list[float],
    sample_metadata_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sample vector store search results."""
    return {
        "ids": ["scene_0001_2025-11-10", "scene_0002_2025-11-10"],
        "documents": [
            "The Admiral stepped onto the bridge, her presence commanding immediate attention.",
            "Dr. Rizzo ran the scans three times. The results didn't change.",
        ],
        "metadatas": sample_metadata_list[:2],
        "scores": sample_embedding_scores[:2],
        "distances": [[1 - s] for s in sample_embedding_scores[:2]],
    }


@pytest.fixture
def sample_reranked_results(
    sample_embedding_scores: list[float],
    sample_rerank_scores: list[float],
    sample_metadata_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sample two-stage retrieval results."""
    return {
        "ids": ["scene_0001_2025-11-10", "scene_0002_2025-11-10"],
        "documents": [
            "The Admiral stepped onto the bridge, her presence commanding.",
            "Dr. Rizzo ran the scans three times.",
        ],
        "metadatas": sample_metadata_list[:2],
        "embedding_scores": sample_embedding_scores[:2],
        "rerank_scores": sample_rerank_scores[:2],
        "reranked": True,
        "rerank_method": "bge-v2-m3",
        "initial_search_count": 50,
    }


@pytest.fixture
def sample_neptune_export() -> str:
    """Sample Neptune AI RP export format."""
    return """# Conversation: ThunderChild Mission Log

***11/10/2025, 4:00:41 AM - Venice:***
The Admiral stepped into the command center, her presence commanding immediate attention. I moved aside, giving her space.

---

***11/10/2025, 4:00:50 AM - Admiral Zelenskyy:***
She nods curtly, then moves to the main display. Status report. I want details on that engine burn.

"""


@pytest.fixture
def sample_polars_dataframe(sample_metadata_list: list[dict[str, Any]]) -> pl.DataFrame:
    """Sample Polars DataFrame with mock scene data."""
    embeddings = [
        np.random.randn(384).tolist() for _ in range(3)  # ← Changed from 2 to 3
    ]
    
    # Create 3rd metadata item to have 3 rows
    metadata_3 = {
        "scene_id": "scene_0003_2025-11-10",
        "date_iso": "2025-11-10",
        "location": "engine_room",
        "pov_character": "Tech",
        "characters_present": json.dumps(["Tech", "Specialist"]),
        "ships": ["ThunderChild"],
        "events": ["maintenance"],
        "tone": "technical",
        "emotional_intensity": 0.2,
        "action_level": 0.3,
    }
    
    metadata_list = list(sample_metadata_list) + [metadata_3]
    
    return pl.DataFrame({
        "id": ["scene_0001_2025-11-10", "scene_0002_2025-11-10", "scene_0003_2025-11-10"],
        "text": [
            "The Admiral stepped onto the bridge, her presence commanding immediate attention.",
            "Dr. Rizzo ran the scans three times. The results didn't change.",
            "The engine room hummed with activity as engineers made final adjustments.",
        ],
        "embedding": embeddings,
        "metadata": [json.dumps(m) for m in metadata_list],
    })



@pytest.fixture
def sample_turn_list() -> list[dict[str, Any]]:
    """Sample parsed Neptune turns (as returned by NeptuneParser)."""
    return [
        {
            "timestamp_raw": "11/10/2025, 4:00:41 AM",
            "date_iso": "2025-11-10",
            "time_display": "11/10/2025, 4:00:41 AM",
            "speaker": "Venice",
            "text": "The Admiral stepped onto the bridge, her presence commanding immediate attention. I moved aside, giving her space.",
        },
        {
            "timestamp_raw": "11/10/2025, 4:00:50 AM",
            "date_iso": "2025-11-10",
            "time_display": "11/10/2025, 4:00:50 AM",
            "speaker": "Admiral Zelenskyy",
            "text": "She nods curtly, then moves to the main display. Status report. I want details on that engine burn.",
        },
    ]


@pytest.fixture
def sample_chat_json() -> str:
    """Sample chat transcript in JSON format."""
    return json.dumps([
        {
            "timestamp": "2025-12-05T10:30:00",
            "user": "Kieran",
            "message": "The Admiral stepped into the command center, her presence commanding immediate attention.",
            "channel": "thunderchild",
        },
        {
            "timestamp": "2025-12-05T10:31:00",
            "user": "Venice",
            "message": "I moved aside, giving her space. The bridge felt smaller with her presence.",
            "channel": "thunderchild",
        },
        {
            "timestamp": "2025-12-05T10:32:00",
            "user": "Admiral",
            "message": "Status report. I want details on that engine burn. Don't leave anything out.",
            "channel": "thunderchild",
        },
    ])


@pytest.fixture
def mock_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock SentenceTransformer to return fake embeddings."""
    class MockSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
        
        def encode(self, texts: Any, **kwargs: Any) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384).astype(np.float32)
    
    monkeypatch.setattr(
        "naragtive.polars_vectorstore.SentenceTransformer",
        MockSentenceTransformer
    )
    monkeypatch.setattr(
        "naragtive.ingest_chat_transcripts.SentenceTransformer",
        MockSentenceTransformer
    )
