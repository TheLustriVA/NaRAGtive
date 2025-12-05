#!/usr/bin/env python3
"""
Export scene search results in formats ready for reranking
Supports BGE reranker models and context packaging for LLM consumption
"""

import json
from typing import Dict, List, Any


class RerankerExporter:
    """Export search results in formats suitable for reranking pipelines"""
    
    @staticmethod
    def format_for_bge_reranker(results: Dict[str, Any], query: str) -> List[Dict[str, str]]:
        """
        Format results for BGE reranker (bge-reranker-large, bge-reranker-base, etc.)
        BGE expects: [{"text": "...", "query": "..."}, ...]
        Returns list of query-document pairs ready for reranking
        """
        pairs = []
        for doc_id, text in zip(results['ids'], results['documents']):
            pairs.append({
                "text": text,
                "query": query,
                "doc_id": doc_id,
            })
        return pairs
    
    @staticmethod
    def format_for_llm_context(results: Dict[str, Any], query: str, max_tokens: int = 4000) -> str:
        """
        Format results as markdown context for LLM prompt
        Includes metadata and proper citations
        """
        context = []
        context.append(f"# Search Results for: '{query}'\n")
        context.append(f"Found {len(results['ids'])} relevant scenes:\n")
        
        token_count = 0
        max_chars = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars
        
        for i, (scene_id, text, metadata, score) in enumerate(
            zip(
                results['ids'],
                results['documents'],
                results['metadatas'],
                results['scores']
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
    def format_for_llamafile(results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Format results for llamafile/llama-server context window
        Ready for injection into system prompt or user message
        """
        documents = []
        for scene_id, text, metadata, score in zip(
            results['ids'],
            results['documents'],
            results['metadatas'],
            results['scores']
        ):
            documents.append({
                "id": scene_id,
                "content": text,
                "metadata": {
                    "date": metadata.get('date_iso'),
                    "location": metadata.get('location'),
                    "pov": metadata.get('pov_character'),
                    "characters": json.loads(metadata.get('characters_present', '[]')),
                    "relevance_score": float(score),
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
    def format_for_json_batch(results: Dict[str, Any], query: str) -> str:
        """
        Export as JSONL (JSON Lines) for batch processing
        One JSON object per line, suitable for processing pipelines
        """
        lines = []
        for scene_id, text, metadata, score in zip(
            results['ids'],
            results['documents'],
            results['metadatas'],
            results['scores']
        ):
            line = {
                "query": query,
                "scene_id": scene_id,
                "text": text,
                "relevance_score": float(score),
                "metadata": metadata,
            }
            lines.append(json.dumps(line))
        
        return "\n".join(lines)
    
    @staticmethod
    def format_for_retrieval_augmented_generation(
        results: Dict[str, Any], 
        query: str,
        template: str = "default"
    ) -> str:
        """
        Format for RAG/prompt injection
        Default template suitable for most LLMs
        """
        if template == "default":
            context = "# Retrieved Context\n\n"
            for i, (scene_id, text, metadata) in enumerate(
                zip(results['ids'], results['documents'], results['metadatas']),
                1
            ):
                context += f"[{i}] {scene_id}\n"
                context += f"{text}\n\n"
            return context
        
        elif template == "minimal":
            # Just concatenate texts with minimal formatting
            return "\n\n".join([f"[{sid}] {text}" for sid, text in zip(results['ids'], results['documents'])])
        
        elif template == "structured":
            # Highly structured for parsing
            docs = []
            for sid, text, meta in zip(results['ids'], results['documents'], results['metadatas']):
                docs.append({
                    "source": sid,
                    "content": text,
                    "context": {
                        "date": meta.get('date_iso'),
                        "location": meta.get('location'),
                    }
                })
            return json.dumps(docs, indent=2)
        
        return "Unknown template"


def add_export_command(subparsers):
    """Add export command to CLI (call from scene_search.py main())"""
    
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


def export_command(args):
    """Execute export command"""
    from polars_vectorstore import PolarsVectorStore
    
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
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Exported to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    # Quick test
    import sys
    test_results = {
        'ids': ['scene_001', 'scene_002'],
        'documents': ['The Admiral commanded the fleet.', 'The captain saluted.'],
        'metadatas': [
            {'date_iso': '2025-11-10', 'location': 'bridge', 'pov_character': 'Venice', 'characters_present': '["Admiral", "Captain"]'},
            {'date_iso': '2025-11-10', 'location': 'bridge', 'pov_character': 'User', 'characters_present': '["Captain"]'},
        ],
        'scores': [0.85, 0.72],
    }
    
    exporter = RerankerExporter()
    
    print("=== BGE Reranker Format ===")
    print(json.dumps(exporter.format_for_bge_reranker(test_results, "Admiral command"), indent=2))
    print("\n=== LLM Context Format ===")
    print(exporter.format_for_llm_context(test_results, "Admiral command"))
    print("\n=== Llamafile Format ===")
    print(json.dumps(exporter.format_for_llamafile(test_results, "Admiral command"), indent=2))
