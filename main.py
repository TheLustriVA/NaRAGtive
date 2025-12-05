#!/usr/bin/env python3
"""
ThunderChild Scene Search CLI with BGE Reranking Support
Query your RPG narrative using Polars vector search + optional BGE cross-encoder reranking
Usage: python scene_search.py [command] [options]
"""

import argparse
import sys
import json
from tkinter import EXCEPTION
from naragtive.polars_vectorstore import PolarsVectorStore, SceneQueryFormatter
from naragtive.bge_reranker_integration import PolarsVectorStoreWithReranker


def migrate_command(args):
    """Migrate from ChromaDB to Polars"""
    print("🚀 Starting ChromaDB → Polars migration...")
    store = PolarsVectorStore(args.output)
    store.save_from_chromadb(None)
    store.stats()
    print("✅ Migration complete!")


def print_reranked_results(results: dict, query: str) -> str:
    """Format reranked results for display"""
    output = []
    output.append("\n" + "=" * 80)
    output.append(f"SCENE SEARCH: '{query}' (BGE RERANKED)")
    output.append(f"Initial embedding search: {results['initial_search_count']} docs")
    output.append(f"After reranking: {len(results['ids'])} docs")
    output.append("=" * 80)

    if not results['ids']:
        output.append("\n❌ No results found\n")
        return "\n".join(output)

    for i, (scene_id, text, metadata, emb_score, rerank_score) in enumerate(
        zip(
            results['ids'],
            results['documents'],
            results['metadatas'],
            results['embedding_scores'],
            results['rerank_scores']
        ),
        1
    ):
        output.append(f"\n[Result {i}] BGE Score: {rerank_score:.1%} (Embedding: {emb_score:.1%})")
        output.append("-" * 80)
        output.append(f"Scene ID:     {scene_id}")
        output.append(f"Date:         {metadata.get('date_iso', 'unknown')}")
        output.append(f"Location:     {metadata.get('location', 'unknown')}")
        output.append(f"POV:          {metadata.get('pov_character', 'unknown')}")

        # Characters
        try:
            chars = json.loads(metadata.get('characters_present', '[]'))
            output.append(f"Characters:   {', '.join(chars) if chars else 'None'}")
        except EXCEPTION as ex:
            print(f"{ex}")

        # Text preview
        output.append("\n📖 Text:")
        preview = text[:400] if len(text) > 400 else text
        output.append(preview)
        if len(text) > 400:
            output.append("[... truncated]")
        output.append("-" * 80)

    output.append("\n" + "=" * 80 + "\n")
    return "\n".join(output)


def query_command(args):
    """Search for scenes with optional reranking"""
    
    if args.rerank:
        store = PolarsVectorStoreWithReranker(args.store)
        results = store.query_and_rerank(
            args.query,
            initial_k=args.initial_k,
            final_k=args.limit,
            normalize=True
        )
        print(print_reranked_results(results, args.query))
    else:
        # Standard embedding-only search
        store = PolarsVectorStore(args.store)
        
        if not store.load():
            print("❌ Vector store not found. Run 'migrate' first.")
            sys.exit(1)
        
        formatter = SceneQueryFormatter(store)
        results = store.query(args.query, n_results=args.limit)
        print(formatter.format_results(results, args.query))


def list_command(args):
    """List scenes by metadata criteria"""
    import polars as pl
    
    store = PolarsVectorStore(args.store)
    
    if not store.load():
        print("❌ Vector store not found.")
        sys.exit(1)
    
    df = store.df
    
    # Filter by location
    if args.location:
        df = df.filter(pl.col('metadata').str.contains(args.location))
    
    # Filter by character
    if args.character:
        df = df.filter(pl.col('metadata').str.contains(args.character))
    
    # Filter by date
    if args.date:
        df = df.filter(pl.col('metadata').str.contains(args.date))
    
    print(f"\n📋 Found {len(df)} matching scenes:\n")
    for row in df.select(['id', 'metadata']).head(20).to_dicts():
        meta = json.loads(row['metadata'])
        print(f"  {row['id']}")
        print(f"    Date: {meta.get('date_iso', 'unknown')}")
        print(f"    Location: {meta.get('location', 'unknown')}")
        print(f"    POV: {meta.get('pov_character', 'unknown')}\n")


def stats_command(args):
    """Show vector store statistics"""
    store = PolarsVectorStore(args.store)
    store.load()
    store.stats()
    
    # Show reranker stats if available
    if args.show_reranker:
        try:
            reranker_store = PolarsVectorStoreWithReranker(args.store)
            reranker_stats = reranker_store.get_reranker_stats()
            print("\n" + "=" * 60)
            print("RERANKER STATISTICS")
            print("=" * 60)
            for key, value in reranker_stats.items():
                if isinstance(value, float):
                    print(f"{key:.<40} {value:.1f}")
                else:
                    print(f"{key:.<40} {value}")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"⚠️  Reranker not available: {e}\n")


def interactive_command(args):
    """Interactive search mode with optional reranking"""
    
    if args.rerank:
        store = PolarsVectorStoreWithReranker(args.store)
        stats = store.get_reranker_stats()
        reranker_status = f"✅ {stats['model']} ({stats['vram_mb']:.0f}MB VRAM, FP16)"
    else:
        store = PolarsVectorStore(args.store)
        if not store.load():
            print("❌ Vector store not found.")
            sys.exit(1)
        reranker_status = "disabled"
    
    print("\n" + "=" * 80)
    print("ThunderChild Scene Search - Interactive Mode")
    print(f"Reranking: {reranker_status}")
    print("=" * 80)
    print("Commands:")
    print("  search <query>    - Search for scenes")
    print("  quit              - Exit")
    print("=" * 80 + "\n")
    
    while True:
        try:
            user_input = input("🔍 Query: ").strip()
            
            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            if args.rerank:
                results = store.query_and_rerank(
                    user_input,
                    initial_k=args.initial_k,
                    final_k=args.limit
                )
                print(print_reranked_results(results, user_input))
            else:
                results = store.query(user_input, n_results=args.limit)
                formatter = SceneQueryFormatter(store)
                print(formatter.format_results(results, user_input))
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def export_command(args):
    """Export search results for reranking/LLM"""
    from naragtive.reranker_export import RerankerExporter
    
    store = PolarsVectorStore(args.store)
    
    if not store.load():
        print("❌ Vector store not found. Run 'migrate' first.")
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
        print(f"✅ Exported {len(results['ids'])} results to {args.output}")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(
        description="ThunderChild Scene Search - Query your RPG narrative",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time: migrate from ChromaDB
  python scene_search.py migrate

  # Search for scenes (embedding only)
  python scene_search.py search "Admiral command"
  
  # Search with BGE reranking (better accuracy)
  python scene_search.py search "Admiral command" --rerank
  python scene_search.py search "tactical decision" --rerank --limit 20

  # Interactive mode (keeps model in memory)
  python scene_search.py interactive
  python scene_search.py interactive --rerank

  # List scenes by criteria
  python scene_search.py list --character "Kieran"
  python scene_search.py list --location "bridge"

  # Show statistics
  python scene_search.py stats --show-reranker

  # Export for external rerankers
  python scene_search.py export "Admiral command" -f bge -o results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate from ChromaDB to Polars")
    migrate_parser.add_argument(
        "-o", "--output",
        default="./thunderchild_scenes.parquet",
        help="Output parquet file (default: ./thunderchild_scenes.parquet)"
    )
    migrate_parser.set_defaults(func=migrate_command)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for scenes")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Number of final results (default: 10)"
    )
    search_parser.add_argument(
        "-s", "--store",
        default="./thunderchild_scenes.parquet",
        help="Path to vector store"
    )
    search_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable BGE v2 M3 reranking (two-stage retrieval)"
    )
    search_parser.add_argument(
        "--initial-k",
        type=int,
        default=50,
        help="Documents to rerank from (default: 50)"
    )
    search_parser.set_defaults(func=query_command)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List scenes by metadata")
    list_parser.add_argument(
        "-c", "--character",
        help="Filter by character name"
    )
    list_parser.add_argument(
        "-l", "--location",
        help="Filter by location"
    )
    list_parser.add_argument(
        "-d", "--date",
        help="Filter by date (ISO format)"
    )
    list_parser.add_argument(
        "-s", "--store",
        default="./thunderchild_scenes.parquet",
        help="Path to vector store"
    )
    list_parser.set_defaults(func=list_command)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show store statistics")
    stats_parser.add_argument(
        "-s", "--store",
        default="./thunderchild_scenes.parquet",
        help="Path to vector store"
    )
    stats_parser.add_argument(
        "--show-reranker",
        action="store_true",
        help="Also show BGE reranker statistics"
    )
    stats_parser.set_defaults(func=stats_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive search mode")
    interactive_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Number of results per search"
    )
    interactive_parser.add_argument(
        "-s", "--store",
        default="./thunderchild_scenes.parquet",
        help="Path to vector store"
    )
    interactive_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable BGE reranking (loads ~1.3GB VRAM)"
    )
    interactive_parser.add_argument(
        "--initial-k",
        type=int,
        default=50,
        help="Documents to rerank from"
    )
    interactive_parser.set_defaults(func=interactive_command)
    
    # Export command
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
