from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
import polars as pl

from naragtive.polars_vectorstore import PolarsVectorStore, SceneQueryFormatter


class TestPolarsVectorStoreInit:
    """Test PolarsVectorStore initialization."""
    
    def test_init_default_path(self) -> None:
        """Test initialization with default parquet path."""
        store = PolarsVectorStore()
        assert store.parquet_path.name == "thunderchild_scenes.parquet"
        assert store.df is None
        assert store.embeddings_cache is None
    
    def test_init_custom_path(self) -> None:
        """Test initialization with custom parquet path."""
        store = PolarsVectorStore("./custom_scenes.parquet")
        assert store.parquet_path.name == "custom_scenes.parquet"
    
    def test_init_invalid_path_raises(self) -> None:
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError):
            PolarsVectorStore("")


class TestPolarsVectorStoreLoad:
    """Test PolarsVectorStore.load() method."""
    
    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test load returns False when file doesn't exist."""
        store = PolarsVectorStore(str(tmp_path / "nonexistent.parquet"))
        result = store.load()
        assert result is False
        assert store.df is None
    
    def test_load_success(self, tmp_path: Path, sample_polars_dataframe: pl.DataFrame) -> None:
        """Test successful load of parquet file."""
        parquet_path = tmp_path / "test_scenes.parquet"
        sample_polars_dataframe.write_parquet(parquet_path)
        
        store = PolarsVectorStore(str(parquet_path))
        result = store.load()
        
        assert result is True
        assert store.df is not None
        assert len(store.df) == 3
        assert store.embeddings_cache is not None
        assert store.embeddings_cache.shape == (3, 384)


class TestPolarsVectorStoreQuery:
    """Test PolarsVectorStore.query() method."""
    
    @patch('naragtive.polars_vectorstore.SentenceTransformer')
    def test_query_returns_correct_structure(
        self,
        mock_model: Mock,
        tmp_path: Path,
        sample_polars_dataframe: pl.DataFrame,
    ) -> None:
        """Test query returns dict with required keys."""
        # Setup mock
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        mock_instance.encode.return_value = np.random.randn(384).astype(np.float32)
        
        # Create store
        parquet_path = tmp_path / "test_scenes.parquet"
        sample_polars_dataframe.write_parquet(parquet_path)
        
        store = PolarsVectorStore(str(parquet_path))
        store.load()
        
        # Query
        results = store.query("test query", n_results=2)
        
        # Verify structure
        assert isinstance(results, dict)
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "scores" in results
        assert len(results["ids"]) == 2
        assert len(results["documents"]) == 2
        assert len(results["scores"]) == 2
    
    @patch('naragtive.polars_vectorstore.SentenceTransformer')
    def test_query_scores_between_0_and_1(
        self,
        mock_model: Mock,
        tmp_path: Path,
        sample_polars_dataframe: pl.DataFrame,
    ) -> None:
        """Test that query scores are normalized between 0 and 1."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        mock_instance.encode.return_value = np.random.randn(384).astype(np.float32)
        
        parquet_path = tmp_path / "test_scenes.parquet"
        sample_polars_dataframe.write_parquet(parquet_path)
        
        store = PolarsVectorStore(str(parquet_path))
        store.load()
        results = store.query("test", n_results=3)
        
        for score in results["scores"]:
            assert 0.0 <= score <= 1.0


class TestSceneQueryFormatter:
    """Test SceneQueryFormatter."""
    
    def test_format_results_with_empty_results(
        self,
        mock_embedding_model: None,
    ) -> None:
        """Test formatting empty results."""
        store = PolarsVectorStore()
        formatter = SceneQueryFormatter(store)
        
        results = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "scores": [],
        }
        
        output = formatter.format_results(results, "test query")
        assert "No results found" in output
    
    def test_format_results_with_results(
        self,
        mock_embedding_model: None,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test formatting valid results."""
        store = PolarsVectorStore()
        formatter = SceneQueryFormatter(store)
        
        output = formatter.format_results(sample_search_results, "test query")
        
        assert "SCENE SEARCH" in output
        assert "test query" in output
        assert "Result 1" in output
        assert "Relevance" in output
        assert "scene_0001" in output


class TestPolarsVectorStoreStats:
    """Test PolarsVectorStore.stats() method."""
    
    @patch('builtins.print')
    @patch('naragtive.polars_vectorstore.SentenceTransformer')
    def test_stats_output(
        self,
        mock_model: Mock,
        mock_print: Mock,
        tmp_path: Path,
        sample_polars_dataframe: pl.DataFrame,
    ) -> None:
        """Test stats() prints expected output."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        
        parquet_path = tmp_path / "test_scenes.parquet"
        sample_polars_dataframe.write_parquet(parquet_path)
        
        store = PolarsVectorStore(str(parquet_path))
        store.load()
        store.stats()
        
        # Verify print was called with stats
        call_args_str = str(mock_print.call_args_list)
        assert "VECTOR STORE STATS" in call_args_str or any(
            "documents" in str(call) for call in mock_print.call_args_list
        )
