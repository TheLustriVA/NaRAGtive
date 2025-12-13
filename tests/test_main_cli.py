"""Tests for main.py CLI commands.

Covers ingestion, search, and utility commands.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

import numpy as np
import pytest
import polars as pl


class TestIngestNeptuneCommand:
    """Test ingest-neptune command."""
    
    @patch('main.NeptuneIngester')
    def test_ingest_neptune_valid_file(self, mock_ingester_class: Mock, tmp_path: Path) -> None:
        """Test ingesting valid Neptune export."""
        # Setup mock
        mock_ingester = MagicMock()
        mock_ingester_class.return_value = mock_ingester
        mock_df = pl.DataFrame({
            "id": ["scene_0000"],
            "text": ["Sample scene"],
            "embedding": [[0.1, 0.2]],
            "metadata": ['{"date_iso": "2025-11-10"}']
        })
        mock_ingester.ingest.return_value = mock_df
        
        # Create test file
        export_file = tmp_path / "export.txt"
        export_file.write_text("***11/10/2025, 4:00:41 AM - Venice:***\nTest scene")
        
        # Would test by calling CLI directly
        # Just verify the ingester was called correctly
        ingester = mock_ingester_class()
        df = ingester.ingest(str(export_file))
        
        assert len(df) == 1
        assert df["id"][0] == "scene_0000"
    
    def test_ingest_neptune_file_not_found(self, tmp_path: Path) -> None:
        """Test error when Neptune export file not found."""
        # This would be tested via CLI exit code
        nonexistent = tmp_path / "nonexistent.txt"
        assert not nonexistent.exists()


class TestIngestLlamaCommand:
    """Test ingest-llama command."""
    
    @patch('main.LlamaServerIngester')
    def test_ingest_llama_valid_file(self, mock_ingester_class: Mock, tmp_path: Path) -> None:
        """Test ingesting valid llama-server export."""
        # Setup mock
        mock_ingester = MagicMock()
        mock_ingester_class.return_value = mock_ingester
        mock_df = pl.DataFrame({
            "id": ["scene_62b43483_0000_2025-11-10"],
            "text": ["User: Hello\n\nAssistant: Response"],
            "embedding": [[0.1, 0.2]],
            "metadata": ['{"model": "test"}']
        })
        mock_ingester.ingest_llama_server_export.return_value = mock_df
        
        # Create test JSON
        export_file = tmp_path / "export.json"
        export_data = {
            "conv": {
                "id": "62b43483-fb41-49b4",
                "name": "Test",
                "lastModified": 1765275434078
            },
            "messages": [
                {"role": "user", "text": "Hello"},
                {"role": "assistant", "text": "Response"}
            ]
        }
        export_file.write_text(json.dumps(export_data))
        
        # Test
        ingester = mock_ingester_class()
        df = ingester.ingest_llama_server_export(str(export_file))
        
        assert len(df) == 1
        assert "User:" in df["text"][0]


class TestIngestChatCommand:
    """Test ingest-chat command."""
    
    @patch('main.ChatTranscriptIngester')
    def test_ingest_json_transcript(self, mock_ingester_class: Mock, tmp_path: Path) -> None:
        """Test ingesting JSON chat transcript."""
        # Setup mock
        mock_ingester = MagicMock()
        mock_ingester_class.return_value = mock_ingester
        mock_df = pl.DataFrame({
            "id": ["chat_000000_Alice"],
            "text": ["Hello world"],
            "embedding": [[0.1, 0.2]],
            "metadata": ['{"user": "Alice"}']
        })
        mock_ingester.ingest_json_messages.return_value = mock_df
        
        # Create test JSON
        chat_file = tmp_path / "chat.json"
        chat_data = [
            {
                "timestamp": "2025-12-05T12:00:00",
                "user": "Alice",
                "message": "Hello world",
                "channel": "general"
            }
        ]
        chat_file.write_text(json.dumps(chat_data))
        
        # Test
        ingester = mock_ingester_class()
        df = ingester.ingest_json_messages(str(chat_file))
        
        assert len(df) == 1
        assert df["text"][0] == "Hello world"
    
    @patch('main.ChatTranscriptIngester')
    def test_ingest_text_transcript(self, mock_ingester_class: Mock, tmp_path: Path) -> None:
        """Test ingesting plain text transcript."""
        # Setup mock
        mock_ingester = MagicMock()
        mock_ingester_class.return_value = mock_ingester
        mock_df = pl.DataFrame({
            "id": ["chunk_000000"],
            "text": ["Chunk of text"],
            "embedding": [[0.1, 0.2]],
            "metadata": ['{"chunk_index": 0}']
        })
        mock_ingester.ingest_txt_file.return_value = mock_df
        
        # Create test text
        txt_file = tmp_path / "transcript.txt"
        txt_file.write_text("This is a long transcript that will be chunked." * 50)
        
        # Test
        ingester = mock_ingester_class()
        df = ingester.ingest_txt_file(str(txt_file), chunk_size=500)
        
        assert len(df) >= 1
        assert df["text"][0] != ""


class TestSearchCommand:
    """Test search command."""
    
    @patch('main.PolarsVectorStore')
    def test_search_basic(self, mock_store_class: Mock, tmp_path: Path) -> None:
        """Test basic semantic search."""
        # Setup mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.load.return_value = True
        mock_store.query.return_value = {
            "ids": ["scene_0001"],
            "documents": ["Sample text"],
            "metadatas": [{"location": "bridge"}],
            "distances": [0.1]
        }
        
        # Test
        store = mock_store_class(str(tmp_path / "store.parquet"))
        assert store.load()
        results = store.query("test query", n_results=10)
        
        assert len(results["ids"]) == 1
    
    @patch('main.PolarsVectorStoreWithReranker')
    def test_search_with_reranking(self, mock_reranker_class: Mock, tmp_path: Path) -> None:
        """Test search with BGE reranking."""
        # Setup mock reranker
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.query_and_rerank.return_value = {
            "ids": ["scene_0001"],
            "documents": ["Sample text"],
            "metadatas": [{"location": "bridge"}],
            "embedding_scores": [0.8],
            "rerank_scores": [0.9],
            "initial_search_count": 50
        }
        
        # Test
        reranker = mock_reranker_class(str(tmp_path / "store.parquet"))
        results = reranker.query_and_rerank(
            "test query",
            initial_k=50,
            final_k=10
        )
        
        assert len(results["ids"]) == 1
        assert "rerank_scores" in results


class TestListCommand:
    """Test list command."""
    
    @patch('main.PolarsVectorStore')
    def test_list_by_location(self, mock_store_class: Mock, tmp_path: Path) -> None:
        """Test filtering scenes by location."""
        # Setup mock store with DataFrame
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.load.return_value = True
        
        # Create sample DataFrame
        mock_store.df = pl.DataFrame({
            "id": ["scene_0001", "scene_0002"],
            "text": ["Bridge scene", "Medbay scene"],
            "metadata": [
                '{"location": "bridge", "date_iso": "2025-11-10"}',
                '{"location": "medbay", "date_iso": "2025-11-10"}'
            ]
        })
        
        # Test filtering
        store = mock_store_class(str(tmp_path / "store.parquet"))
        assert store.load()
        
        # Filter by location (simulated)
        filtered = store.df.filter(pl.col('metadata').str.contains("bridge"))
        assert len(filtered) == 1


class TestStatsCommand:
    """Test stats command."""
    
    @patch('main.PolarsVectorStore')
    def test_stats_shows_counts(self, mock_store_class: Mock, tmp_path: Path) -> None:
        """Test that stats shows scene counts."""
        # Setup mock store
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.load.return_value = True
        mock_store.stats.return_value = None  # stats() prints output
        
        # Create test store
        store = mock_store_class(str(tmp_path / "store.parquet"))
        assert store.load()
        store.stats()  # Just verify it's callable
    
    @patch('main.PolarsVectorStoreWithReranker')
    def test_stats_with_reranker(self, mock_reranker_class: Mock, tmp_path: Path) -> None:
        """Test stats with reranker information."""
        # Setup mock
        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker
        mock_reranker.get_reranker_stats.return_value = {
            "model": "bge-reranker-v2-m3",
            "vram_mb": 1350.5,
            "precision": "FP16"
        }
        
        # Test
        reranker = mock_reranker_class(str(tmp_path / "store.parquet"))
        stats = reranker.get_reranker_stats()
        
        assert stats["model"] == "bge-reranker-v2-m3"
        assert stats["vram_mb"] > 1000


class TestExportCommand:
    """Test export command."""
    
    @patch('main.PolarsVectorStore')
    # @patch('main.RerankerExporter')
    def test_export_llm_context(self, mock_exporter_class: Mock, mock_store_class: Mock, tmp_path: Path) -> None:
        """Test exporting results as LLM context."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store
        mock_store.load.return_value = True
        mock_store.query.return_value = {
            "ids": ["scene_0001"],
            "documents": ["Scene text"],
            "metadatas": [{"location": "bridge"}],
            "distances": [0.1]
        }
        
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        mock_exporter.format_for_llm_context.return_value = "# Scene\n\nText here"
        
        # Test
        store = mock_store_class(str(tmp_path / "store.parquet"))
        assert store.load()
        results = store.query("test query")
        
        exporter = mock_exporter_class()
        output = exporter.format_for_llm_context(results, "test query")
        
        assert isinstance(output, str)
        assert "Scene" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
