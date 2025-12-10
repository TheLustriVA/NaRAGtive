from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np

from naragtive.bge_reranker_integration import (
    BGERerankerM3,
    PolarsVectorStoreWithReranker,
)


class TestPolarsVectorStoreWithRerankerFallback:
    """Test graceful fallback when reranker unavailable."""
    
    @patch('naragtive.bge_reranker_integration.BGERerankerM3')
    def test_init_fallback_when_reranker_fails(self, mock_reranker: Mock) -> None:
        """Test that store initializes with use_reranker=False on reranker failure."""
        mock_reranker.side_effect = Exception("GPU not available")
        
        store = PolarsVectorStoreWithReranker(use_reranker=True)
        
        assert store.use_reranker is False
        assert store.reranker is None
    
    @patch('naragtive.bge_reranker_integration.BGERerankerM3')
    def test_query_and_rerank_fallback(
        self,
        mock_reranker: Mock,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test query_and_rerank falls back to embedding search when reranker fails."""
        mock_reranker.side_effect = Exception("GPU error")
        
        store = PolarsVectorStoreWithReranker(use_reranker=True)
        
        # Mock the underlying vector store
        store.store.query = MagicMock(return_value=sample_search_results)
        
        results = store.query_and_rerank("test query")
        
        assert results["reranked"] is False
        assert "scores" in results
        assert results["rerank_method"] == "none"


class TestPolarsVectorStoreWithRerankerRerank:
    """Test two-stage retrieval with reranking."""
    
    @patch('naragtive.bge_reranker_integration.BGERerankerM3')
    def test_query_and_rerank_structure(
        self,
        mock_reranker: Mock,
        sample_search_results: dict[str, Any],
        sample_rerank_scores: list[float],
    ) -> None:
        """Test query_and_rerank returns correct result structure."""
        # Setup mocks
        rerank_instance = MagicMock()
        mock_reranker.return_value = rerank_instance
        rerank_instance.rerank.return_value = (
            np.array(sample_rerank_scores[:2]),
            np.array([0, 1]),
        )
        
        store = PolarsVectorStoreWithReranker(use_reranker=True)
        
        # Mock the underlying vector store query
        store.store.query = MagicMock(return_value=sample_search_results)
        
        results = store.query_and_rerank("test", initial_k=50, final_k=2)
        
        assert results["reranked"] is True
        assert "embedding_scores" in results
        assert "rerank_scores" in results
        assert results["rerank_method"] == "bge-v2-m3"
        assert len(results["rerank_scores"]) == 2


class TestGetRerankerStats:
    """Test get_reranker_stats() method."""
    
    def test_stats_when_disabled(self) -> None:
        """Test stats when reranker is disabled."""
        store = PolarsVectorStoreWithReranker(use_reranker=False)
        stats = store.get_reranker_stats()
        
        assert stats["status"] == "disabled"
        assert "reason" in stats
    
    @patch('naragtive.bge_reranker_integration.BGERerankerM3')
    def test_stats_when_enabled(self, mock_reranker: Mock) -> None:
        """Test stats returns expected keys when enabled."""
        mock_instance = MagicMock()
        mock_reranker.return_value = mock_instance
        mock_instance.model_name = "BAAI/bge-reranker-v2-m3"
        mock_instance.device = "cuda"
        mock_instance.use_fp16 = True
        mock_instance.model.parameters.return_value = [
            MagicMock(numel=MagicMock(return_value=1000000))
        ]
        
        store = PolarsVectorStoreWithReranker(use_reranker=True)
        stats = store.get_reranker_stats()
        
        assert stats["status"] == "enabled"
        assert "model" in stats
        assert "device" in stats
        assert "fp16" in stats
        assert "parameters" in stats
        assert "vram_mb" in stats
