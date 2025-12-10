from __future__ import annotations

#!/usr/bin/env python3
"""
Export scene search results in formats suitable for reranking and LLM consumption.

Provides multiple output formats for downstream integration with:
- BGE reranker models
- Language models (LLM prompts)
- Llamafile/llama-server
- Batch processing pipelines
- RAG (Retrieval-Augmented Generation) systems
"""

import json
from typing import Any


class RerankerExporter:
    """
    Export search results in formats suitable for reranking and integration.
    
    Provides methods to convert vector store query results into formats
    optimized for different downstream consumers.
    
    Example:
        ```python
        store = PolarsVectorStore("./scenes.parquet")
        store.load()
        results = store.query("Admiral leadership", n_results=10)
        
        exporter = RerankerExporter()
        
        # For LLM context
        context = exporter.format_for_llm_context(results, "Admiral leadership")
        
        # For BGE reranker
        pairs = exporter.format_for_bge_reranker(results, "Admiral leadership")
        ```
    """
    
    @staticmethod
    def format_for_bge_reranker(
        results: dict[str, Any],
        query: str
    ) -> list[dict[str, str]]:
        """
        Format results for BGE cross-encoder reranker.
        
        BGE reranker expects query-document pairs as input.
        This method creates pairs ready for reranking.
        
        Args:
            results: Results dictionary from vector store query
            query: Original search query string
            
        Returns:
            List of query-document pair dictionaries with keys:
                - 'text': Document text
                - 'query': Query string
                - 'doc_id': Document ID
                
        Example:
            ```python
            results = store.query("Admiral command")
            pairs = exporter.format_for_bge_reranker(results, "Admiral command")
            # [{'text': '...', 'query': 'Admiral command', 'doc_id': 'scene_0001'}, ...]
            ```
        """
        pairs: list[dict[str, str]] = []
        for doc_id, text in zip(results["ids"], results["documents"]):
            pairs.append({
                "text": text,
                "query": query,
                "doc_id": doc_id,
            })
        return pairs
    
    @staticmethod
    def format_for_llm_context(
        results: dict[str, Any],
        query: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Format results as markdown context for LLM prompt injection.
        
        Creates a markdown document with proper citations, metadata,
        and text snippets suitable for including in LLM prompts.
        
        Args:
            results: Results dictionary from vector store query
            query: Original search query string
            max_tokens: Maximum tokens to include (approximate).
                Default: 4000
                
        Returns:
            Formatted markdown string with citations
            
        Example:
            ```python
            results = store.query("Admiral leadership", n_results=5)
            context = exporter.format_for_llm_context(results, "Admiral leadership")
            print(context)  # Ready to paste into LLM prompt
            ```
        """
        context: list[str] = []
        context.append(f"# Search Results for: '{query}'\n")
        context.append(f"Found {len(results['ids'])} relevant scenes:\n")
        
        token_count = 0.0
        max_chars = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars
        
        for i, (scene_id, text, metadata, score) in enumerate(
            zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                results["scores"]
            ),
            1
        ):
            # Check token budget
            if token_count > max_chars * 0.9:
                context.append("\n[... truncated due to token limit ...]")
                break
            
            section = f"""
## Source [{i}]: {scene_id}
**Relevance Score:** {score:.1%}
**Date:** {metadata.get('date_iso', 'unknown')}
**Location:** {metadata.get('location', 'unknown')}
**POV:** {metadata.get('pov_character', 'unknown')}

{text}

---
"""
            context.append(section)
            token_count += len(section) / 4
        
        return "\n".join(context)
    
    @staticmethod
    def format_for_llamafile(
        results: dict[str, Any],
        query: str
    ) -> dict[str, Any]:
        """
        Format results for llamafile/llama-server integration.
        
        Creates JSON payload with documents, metadata, and system instructions
        for llamafile/llama-cpp-python server integration.
        
        Args:
            results: Results dictionary from vector store query
            query: Original search query string
            
        Returns:
            Dictionary with:
                - 'query': Search query
                - 'document_count': Number of documents
                - 'documents': List of document dicts with content and metadata
                - 'instructions': System instructions for LLM
                
        Example:
            ```python
            results = store.query("Admiral command", n_results=5)
            payload = exporter.format_for_llamafile(results, "Admiral command")
            # Send to llamafile server
            ```
        """
        documents: list[dict[str, Any]] = []
        for scene_id, text, metadata, score in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["scores"]
        ):
            documents.append({
                "id": scene_id,
                "content": text,
                "metadata": {
                    "date": metadata.get("date_iso"),
                    "location": metadata.get("location"),
                    "pov": metadata.get("pov_character"),
                    "characters": json.loads(metadata.get("characters_present", "[]")),
                    "relevance_score": score,
                }
            })
        
        return {
            "query": query,
            "document_count": len(documents),
            "documents": documents,
            "instructions": f"""You have been provided with {len(documents)} relevant scene excerpts from an RPG narrative.
Use these to answer questions about the story, characters, and events.
When referencing specific scenes, cite the source ID (e.g., [scene_0001]).
If information contradicts between sources, note the discrepancy."""
        }
    
    @staticmethod
    def format_for_json_batch(
        results: dict[str, Any],
        query: str
    ) -> str:
        """
        Export as JSONL (JSON Lines) for batch processing.
        
        Creates one JSON object per line, suitable for processing
        with tools like jq or batch pipelines.
        
        Args:
            results: Results dictionary from vector store query
            query: Original search query string
            
        Returns:
            String with newline-separated JSON objects
            
        Example:
            ```python
            results = store.query("Admiral command")
            jsonl = exporter.format_for_json_batch(results, "Admiral command")
            with open("results.jsonl", "w") as f:
                f.write(jsonl)
            ```
        """
        lines: list[str] = []
        for scene_id, text, metadata, score in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["scores"]
        ):
            line = {
                "query": query,
                "scene_id": scene_id,
                "text": text,
                "relevance_score": score,
                "metadata": metadata,
            }
            lines.append(json.dumps(line))
        
        return "\n".join(lines)
    
    @staticmethod
    def format_for_retrieval_augmented_generation(
        results: dict[str, Any],
        query: str,
        template: str = "default"
    ) -> str:
        """
        Format for RAG (Retrieval-Augmented Generation) applications.
        
        Provides multiple templates for different RAG use cases:
        - "default": Numbered citations with full metadata
        - "minimal": Just concatenated text with minimal formatting
        - "structured": JSON array format for parsing
        
        Args:
            results: Results dictionary from vector store query
            query: Original search query string
            template: Output template ("default", "minimal", "structured").
                Default: "default"
                
        Returns:
            Formatted string in requested template
            
        Example:
            ```python
            results = store.query("Admiral leadership")
            
            # For LLM consumption
            context = exporter.format_for_retrieval_augmented_generation(
                results, "Admiral leadership", template="default"
            )
            
            # Minimal version for token-constrained scenarios
            minimal = exporter.format_for_retrieval_augmented_generation(
                results, "Admiral leadership", template="minimal"
            )
            
            # Structured for parsing
            structured = exporter.format_for_retrieval_augmented_generation(
                results, "Admiral leadership", template="structured"
            )
            ```
        """
        if template == "default":
            context = "# Retrieved Context\n\n"
            for i, (scene_id, text, metadata) in enumerate(
                zip(results["ids"], results["documents"], results["metadatas"]),
                1
            ):
                context += f"[{i}] {scene_id}\n"
                context += f"{text}\n\n"
            return context
        
        elif template == "minimal":
            # Just concatenate texts with minimal formatting
            return "\n\n".join(
                [f"[{sid}] {text}" for sid, text in zip(results["ids"], results["documents"])]
            )
        
        elif template == "structured":
            # Highly structured for parsing
            docs: list[dict[str, Any]] = []
            for sid, text, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                docs.append({
                    "source": sid,
                    "content": text,
                    "context": {
                        "date": meta.get("date_iso"),
                        "location": meta.get("location"),
                    }
                })
            return json.dumps(docs, indent=2)
        
        return "Unknown template"


def add_export_command(subparsers: Any) -> None:
    """
    Add export command to CLI argument parser.
    
    Called from main.py to register the export command.
    
    Args:
        subparsers: ArgumentParser subparsers object from argparse
        
    Example:
        ```python
        # In main.py
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_export_command(subparsers)
        ```
    """
    
    export_parser = subparsers.add_parser("export", help="Export search results for reranking/LLM")
    export_parser.add_argument("query", help="Search query")
    export_parser.add_argument(
        "-f", "--format",
        choices=["bge", "llm-context", "llamafile", "jsonl", "rag", "rag-minimal", "rag-structured"],
        default="llm-context",
        help="Export format (default: llm-context)"
    )
    export_parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)"
    )
    export_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Number of results"
    )
    export_parser.add_argument(
        "-s", "--store",
        default="./thunderchild_scenes.parquet",
        help="Path to vector store"
    )
    export_parser.set_defaults(func=export_command)


def export_command(args: Any) -> None:
    """
    Execute export command from CLI.
    
    Args:
        args: Parsed command-line arguments from argparse
    """
    import sys
    from naragtive.polars_vectorstore import PolarsVectorStore
    
    store = PolarsVectorStore(args.store)
    if not store.load():
        print("❌ Vector store not found.")
        sys.exit(1)
    
    results = store.query(args.query, n_results=args.limit)
    exporter = RerankerExporter()
    
    # Export in requested format
    if args.format == "bge":
        output = json.dumps(exporter.format_for_bge_reranker(results, args.query), indent=2)
    elif args.format == "llm-context":
        output = exporter.format_for_llm_context(results, args.query)
    elif args.format == "llamafile":
        output = json.dumps(exporter.format_for_llamafile(results, args.query), indent=2)
    elif args.format == "jsonl":
        output = exporter.format_for_json_batch(results, args.query)
    elif args.format == "rag":
        output = exporter.format_for_retrieval_augmented_generation(results, args.query, "default")
    elif args.format == "rag-minimal":
        output = exporter.format_for_retrieval_augmented_generation(results, args.query, "minimal")
    elif args.format == "rag-structured":
        output = exporter.format_for_retrieval_augmented_generation(results, args.query, "structured")
    else:
        print(f"❌ Unknown format: {args.format}")
        sys.exit(1)
    
    # Output to file or stdout
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"✅ Exported to {args.output}")
    else:
        print(output)


def main() -> None:
    """
    Demo: Show all export format examples.
    
    Demonstrates each export format with sample data.
    """
    import sys
    
    test_results = {
        "ids": ["scene_001", "scene_002"],
        "documents": ["The Admiral commanded the fleet.", "The captain saluted."],
        "metadatas": [
            {
                "date_iso": "2025-11-10",
                "location": "bridge",
                "pov_character": "Venice",
                "characters_present": '["Admiral", "Captain"]'
            },
            {
                "date_iso": "2025-11-10",
                "location": "bridge",
                "pov_character": "User",
                "characters_present": '["Captain"]'
            },
        ],
        "scores": [0.85, 0.72],
    }
    
    exporter = RerankerExporter()
    
    print("=== BGE Reranker Format ===")
    print(json.dumps(exporter.format_for_bge_reranker(test_results, "Admiral command"), indent=2))
    print("\n=== LLM Context Format ===")
    print(exporter.format_for_llm_context(test_results, "Admiral command"))
    print("\n=== Llamafile Format ===")
    print(json.dumps(exporter.format_for_llamafile(test_results, "Admiral command"), indent=2))


if __name__ == "__main__":
    main()
