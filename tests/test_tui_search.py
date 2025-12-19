"""Tests for NaRAGtive TUI search functionality.

Covers:
- Search utility functions
- Widget functionality
- Screen integration
- Async search operations
- Export functionality
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest

from naragtive.tui.search_utils import (
    async_search,
    async_rerank,
    format_relevance_score,
    parse_metadata,
    truncate_text,
    format_search_query,
    SearchError,
)


class TestSearchUtils:
    """Tests for search utility functions."""

    def test_format_relevance_score_valid(self) -> None:
        """Test score formatting with valid inputs."""
        assert format_relevance_score(0.94) == "94%"
        assert format_relevance_score(0.871) == "87%"
        assert format_relevance_score(1.0) == "100%"
        assert format_relevance_score(0.0) == "0%"

    def test_format_relevance_score_clamping(self) -> None:
        """Test score clamping to [0, 1] range."""
        assert format_relevance_score(1.5) == "100%"  # Clamped to 1.0
        assert format_relevance_score(-0.5) == "0%"  # Clamped to 0.0

    def test_parse_metadata_complete(self) -> None:
        """Test metadata parsing with all fields."""
        metadata = {
            "scene_id": "scene-123",
            "date_iso": "2024-01-15",
            "location": "Throne Room",
            "pov_character": "Admiral",
            "characters_present": '["Admiral", "King", "Advisor"]',
        }

        parsed = parse_metadata(metadata)

        assert parsed["scene_id"] == "scene-123"
        assert parsed["date"] == "2024-01-15"
        assert parsed["location"] == "Throne Room"
        assert parsed["pov"] == "Admiral"
        assert parsed["characters"] == ["Admiral", "King", "Advisor"]

    def test_parse_metadata_missing_fields(self) -> None:
        """Test metadata parsing with missing fields."""
        metadata: dict[str, str] = {}

        parsed = parse_metadata(metadata)

        assert parsed["scene_id"] == "UNKNOWN"
        assert parsed["date"] == "UNKNOWN"
        assert parsed["location"] == "unknown"
        assert parsed["pov"] == "UNKNOWN"
        assert parsed["characters"] == []

    def test_parse_metadata_invalid_json(self) -> None:
        """Test metadata parsing with invalid JSON in characters."""
        metadata = {
            "characters_present": "invalid json",
        }

        parsed = parse_metadata(metadata)
        assert parsed["characters"] == []

    def test_truncate_text_no_truncation(self) -> None:
        """Test truncate with text shorter than limit."""
        text = "Short text"
        result = truncate_text(text, max_length=50)
        assert result == text

    def test_truncate_text_with_truncation(self) -> None:
        """Test truncate with text longer than limit."""
        text = "A" * 300
        result = truncate_text(text, max_length=50)
        
        assert len(result) <= 50
        assert result.endswith("[...]")

    def test_format_search_query_valid(self) -> None:
        """Test query formatting with valid input."""
        query = "  Admiral leadership  "
        result = format_search_query(query)
        assert result == "Admiral leadership"

    def test_format_search_query_too_short(self) -> None:
        """Test query formatting with too-short query."""
        with pytest.raises(SearchError):
            format_search_query("ab")

    def test_format_search_query_empty(self) -> None:
        """Test query formatting with empty query."""
        with pytest.raises(SearchError):
            format_search_query("")


class TestAsyncSearch:
    """Tests for async search operations."""

    @pytest.mark.asyncio
    async def test_async_search_success(self) -> None:
        """Test successful async search."""
        # Mock store
        mock_store = AsyncMock()
        mock_store.df = Mock()  # Loaded
        mock_store.query = Mock(
            return_value={
                "ids": ["scene-1"],
                "documents": ["Test scene"],
                "scores": [0.94],
                "metadatas": [{"scene_id": "scene-1"}],
            }
        )

        result = await async_search(mock_store, "Admiral", n_results=20)

        assert len(result["ids"]) == 1
        assert result["ids"][0] == "scene-1"
        mock_store.query.assert_called_once_with("Admiral", 20)

    @pytest.mark.asyncio
    async def test_async_search_query_too_short(self) -> None:
        """Test async search with query too short."""
        mock_store = Mock()
        
        with pytest.raises(SearchError):
            await async_search(mock_store, "ab")

    @pytest.mark.asyncio
    async def test_async_search_store_not_loaded(self) -> None:
        """Test async search with unloaded store."""
        mock_store = Mock()
        mock_store.df = None  # Not loaded
        
        with pytest.raises(SearchError):
            await async_search(mock_store, "Admiral")

    @pytest.mark.asyncio
    async def test_async_search_timeout(self) -> None:
        """Test async search timeout."""
        mock_store = AsyncMock()
        mock_store.df = Mock()
        mock_store.query = AsyncMock(
            side_effect=asyncio.sleep(100)  # Long delay
        )
        
        with pytest.raises(SearchError):
            await async_search(mock_store, "Admiral", timeout=0.1)


class TestAsyncRerank:
    """Tests for async reranking operations."""

    @pytest.mark.asyncio
    async def test_async_rerank_success(self) -> None:
        """Test successful async reranking."""
        import numpy as np
        
        # Mock reranker
        mock_reranker = Mock()
        scores = np.array([0.95, 0.87, 0.76])
        indices = np.array([0, 1, 2])
        mock_reranker.rerank = Mock(return_value=(scores, indices))

        documents = ["Doc 1", "Doc 2", "Doc 3"]
        result = await async_rerank(
            mock_reranker,
            "query",
            documents,
            top_k=2,
        )

        assert len(result) == 2
        assert result[0]["score"] == pytest.approx(0.95)
        assert result[0]["text"] == "Doc 1"

    @pytest.mark.asyncio
    async def test_async_rerank_no_documents(self) -> None:
        """Test async reranking with no documents."""
        mock_reranker = Mock()
        
        with pytest.raises(SearchError):
            await async_rerank(mock_reranker, "query", [])

    @pytest.mark.asyncio
    async def test_async_rerank_timeout(self) -> None:
        """Test async reranking timeout."""
        mock_reranker = Mock()
        mock_reranker.rerank = Mock(
            side_effect=asyncio.sleep(100)
        )
        
        with pytest.raises(SearchError):
            await async_rerank(
                mock_reranker,
                "query",
                ["Doc 1"],
                timeout=0.1,
            )


class TestSearchWidgets:
    """Tests for search widgets.
    
    Note: These are light integration tests due to Textual complexity.
    """

    def test_search_input_widget_creation(self) -> None:
        """Test search input widget creation."""
        from naragtive.tui.widgets.search_input import SearchInputWidget
        
        history = ["previous query"]
        widget = SearchInputWidget(history)
        
        assert widget.search_history == history
        assert widget.history_index is None

    def test_search_input_widget_history_management(self) -> None:
        """Test search history management."""
        from naragtive.tui.widgets.search_input import SearchInputWidget
        
        widget = SearchInputWidget(["old", "older"])
        
        # Add new query to history
        widget.search_history.insert(0, "new")
        assert widget.search_history[0] == "new"
        
        # Keep only last 50
        widget.search_history = widget.search_history[:50]
        assert len(widget.search_history) <= 50

    def test_results_table_widget_creation(self) -> None:
        """Test results table widget creation."""
        from naragtive.tui.widgets.results_table import ResultsTableWidget
        
        widget = ResultsTableWidget()
        assert widget.results_count == 0
        assert widget.cursor_type == "row"
        assert widget.zebra_stripes is True

    def test_result_detail_widget_creation(self) -> None:
        """Test result detail widget creation."""
        from naragtive.tui.widgets.result_detail import ResultDetailWidget
        
        widget = ResultDetailWidget()
        assert widget.result_data is None


class TestSearchExport:
    """Tests for export functionality."""

    def test_export_data_structure(self) -> None:
        """Test export data structure."""
        results = {
            "ids": ["scene-1", "scene-2"],
            "documents": ["Text 1", "Text 2"],
            "scores": [0.94, 0.87],
            "metadatas": [
                {"scene_id": "scene-1", "date_iso": "2024-01-15"},
                {"scene_id": "scene-2", "date_iso": "2024-01-16"},
            ],
        }

        # Build export data (mimics SearchScreen.action_export)
        export_data = {
            "query": "test query",
            "result_count": len(results["ids"]),
            "results": [
                {
                    "scene_id": scene_id,
                    "score": score,
                    "metadata": metadata,
                }
                for scene_id, score, metadata in zip(
                    results["ids"],
                    results["scores"],
                    results["metadatas"],
                )
            ],
        }

        assert export_data["result_count"] == 2
        assert len(export_data["results"]) == 2
        assert export_data["results"][0]["scene_id"] == "scene-1"
        assert export_data["results"][0]["score"] == 0.94

    def test_export_json_serialization(self, tmp_path: Path) -> None:
        """Test JSON export serialization."""
        export_data = {
            "query": "test",
            "result_count": 1,
            "results": [
                {
                    "scene_id": "scene-1",
                    "score": 0.94,
                    "metadata": {"date_iso": "2024-01-15"},
                }
            ],
        }

        # Write and read back
        export_path = tmp_path / "export.json"
        with open(export_path, "w") as f:
            json.dump(export_data, f)

        with open(export_path) as f:
            loaded = json.load(f)

        assert loaded == export_data
        assert loaded["results"][0]["scene_id"] == "scene-1"


class TestIntegration:
    """Integration tests for search workflow."""

    @pytest.mark.asyncio
    async def test_search_workflow_no_results(self) -> None:
        """Test search workflow with no results."""
        mock_store = Mock()
        mock_store.df = Mock()
        mock_store.query = Mock(
            return_value={
                "ids": [],
                "documents": [],
                "scores": [],
                "metadatas": [],
            }
        )

        # Should not raise, just return empty results
        result = await async_search(mock_store, "nonexistent")
        assert result["ids"] == []

    @pytest.mark.asyncio
    async def test_search_workflow_with_metadata(self) -> None:
        """Test search with metadata parsing."""
        results = {
            "ids": ["scene-1"],
            "documents": ["Scene text"],
            "scores": [0.94],
            "metadatas": [
                {
                    "scene_id": "scene-1",
                    "date_iso": "2024-01-15",
                    "location": "Throne Room",
                    "pov_character": "Admiral",
                    "characters_present": '["Admiral", "King"]',
                }
            ],
        }

        # Parse metadata
        parsed = parse_metadata(results["metadatas"][0])
        assert parsed["scene_id"] == "scene-1"
        assert parsed["characters"] == ["Admiral", "King"]
        
        # Format score
        score_str = format_relevance_score(results["scores"][0])
        assert score_str == "94%"


# Test fixtures and helpers

@pytest.fixture
def mock_vector_store() -> Mock:
    """Create a mock vector store."""
    store = Mock()
    store.df = Mock()  # Loaded
    store.query = Mock(
        return_value={
            "ids": ["scene-1", "scene-2"],
            "documents": ["Scene 1 text", "Scene 2 text"],
            "scores": [0.94, 0.87],
            "metadatas": [
                {
                    "scene_id": "scene-1",
                    "date_iso": "2024-01-15",
                    "location": "Throne Room",
                    "pov_character": "Admiral",
                    "characters_present": '["Admiral"]',
                },
                {
                    "scene_id": "scene-2",
                    "date_iso": "2024-01-16",
                    "location": "War Room",
                    "pov_character": "King",
                    "characters_present": '["King", "Admiral"]',
                },
            ],
        }
    )
    return store


@pytest.fixture
def mock_reranker() -> Mock:
    """Create a mock BGE reranker."""
    import numpy as np
    
    reranker = Mock()
    scores = np.array([0.95, 0.85])
    indices = np.array([0, 1])
    reranker.rerank = Mock(return_value=(scores, indices))
    return reranker
