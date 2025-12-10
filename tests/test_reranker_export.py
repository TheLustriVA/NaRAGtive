from __future__ import annotations

import json
from typing import Any

import pytest

from naragtive.reranker_export import RerankerExporter


class TestRerankerExporter:
    """Test export formatting."""
    
    def test_format_for_bge_reranker(
        self,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test BGE reranker format."""
        exporter = RerankerExporter()
        result = exporter.format_for_bge_reranker(sample_search_results, "test query")
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]
        assert "query" in result[0]
        assert "doc_id" in result[0]
        assert result[0]["query"] == "test query"
    
    def test_format_for_llm_context(
        self,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test LLM context format."""
        exporter = RerankerExporter()
        result = exporter.format_for_llm_context(sample_search_results, "test query")
        
        assert isinstance(result, str)
        assert "# Search Results for" in result
        assert "test query" in result
        assert "Relevance" in result
    
    def test_format_for_json_batch(
        self,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test JSONL batch format."""
        exporter = RerankerExporter()
        result = exporter.format_for_json_batch(sample_search_results, "test query")
        
        lines = result.strip().split("\n")
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "query" in obj
            assert "scene_id" in obj
            assert "text" in obj
    
    def test_format_for_rag_default(
        self,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test RAG default template."""
        exporter = RerankerExporter()
        result = exporter.format_for_retrieval_augmented_generation(
            sample_search_results,
            "test query",
            template="default"
        )
        
        assert isinstance(result, str)
        assert "Retrieved Context" in result
        assert "scene_0001" in result
    
    def test_format_for_rag_minimal(
        self,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test RAG minimal template."""
        exporter = RerankerExporter()
        result = exporter.format_for_retrieval_augmented_generation(
            sample_search_results,
            "test query",
            template="minimal"
        )
        
        assert isinstance(result, str)
        # Scene ID includes date suffix, so check for scene_0001_2025-11-10
        assert "scene_0001_2025-11-10" in result
    
    def test_format_for_rag_structured(
        self,
        sample_search_results: dict[str, Any],
    ) -> None:
        """Test RAG structured template."""
        exporter = RerankerExporter()
        result = exporter.format_for_retrieval_augmented_generation(
            sample_search_results,
            "test query",
            template="structured"
        )
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "source" in data[0]
        assert "content" in data[0]
