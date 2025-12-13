#!/usr/bin/env python3
"""
NaRAGtive: Narrative RAG with Semantic Search & Reranking

Manage your fiction project's narrative using vector search and embedding-based retrieval.
Features ingestion from multiple sources, semantic search with optional reranking, and analytics.

Usage: python main.py [command] [options]

Commands:
  ingest      - Ingest narratives from Neptune exports or llama-server chats
  search      - Search ingested narratives with optional BGE reranking
  interactive - Interactive search mode with model caching
  stats       - Show vector store statistics
  list        - List scenes by metadata filters
  export      - Export search results for LLM/reranking
  migrate     - (Legacy) Migrate from ChromaDB to Polars
"""

import argparse
import sys
import json
from tkinter import EXCEPTION
from pathlib import Path

from naragtive.polars_vectorstore import PolarsVectorStore, SceneQueryFormatter
from naragtive.bge_reranker_integration import PolarsVectorStoreWithReranker
from naragtive.ingest_chat_transcripts import NeptuneIngester, ChatTranscriptIngester
from naragtive.ingest_llama_server_chat import LlamaServerIngester


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 80)
    print("NaRAGtive: Narrative RAG with Semantic Search & Reranking")
    print("=" * 80 + "\n")


# ============================================================================
# INGESTION COMMANDS
# ============================================================================

def ingest_neptune_command(args):
    """Ingest Neptune AI RPG narrative export"""
    print(f"\n📚 Ingesting Neptune export: {args.export}")
    
    if not Path(args.export).exists():
        print(f"❌ File not found: {args.export}")
        sys.exit(1)
    
    ingester = NeptuneIngester()
    try:
        df = ingester.ingest(
            args.export,
            parquet_output=args.output,
            append=args.append
        )
        print(f"\n✅ Successfully ingested {len(df)} scenes")
        print(f"📁 Saved to: {args.output}")
        
        # Show sample stats
        store = PolarsVectorStore(args.output)
        store.load()
        store.stats()
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        sys.exit(1)


def ingest_llama_command(args):
    """Ingest llama-server chat export"""
    print(f"\n💬 Ingesting llama-server export: {args.export}")
    
    if not Path(args.export).exists():
        print(f"❌ File not found: {args.export}")
        sys.exit(1)
    
    ingester = LlamaServerIngester()
    try:
        df = ingester.ingest_llama_server_export(
            args.export,
            output_parquet=args.output
        )
        print(f"\n✅ Successfully ingested {len(df)} scenes")
        print(f"📁 Saved to: {args.output}")
        
        # Show sample stats
        store = PolarsVectorStore(args.output)
        store.load()
        store.stats()
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        sys.exit(1)


def ingest_chat_command(args):
    """Ingest generic chat transcripts (JSON or text)"""
    print(f"\n💬 Ingesting chat transcript: {args.source}")
    
    if not Path(args.source).exists():
        print(f"❌ File not found: {args.source}")
        sys.exit(1)
    
    ingester = ChatTranscriptIngester()
    try:
        if args.type == "json":
            df = ingester.ingest_json_messages(
                args.source,
                parquet_output=args.output
            )
        elif args.type == "txt":
            df = ingester.ingest_txt_file(
                args.source,
                chunk_size=args.chunk_size,
                parquet_output=args.output
            )
        
        print(f"\n✅ Successfully ingested {len(df)} entries")
        print(f"📁 Saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        sys.exit(1)


def ingest_help_command(args):
    """Show detailed help for ingestion workflows"""
    help_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                       INGESTION WORKFLOWS GUIDE                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📖 THREE WAYS TO INGEST YOUR NARRATIVES:

1️⃣  NEPTUNE AI RPG NARRATIVES
   ─────────────────────────────────────────────────────────────────────────
   Perfect for: Role-playing game transcripts from Neptune AI
   
   Steps:
   a) Export your Neptune conversation (File → Export)
   b) Run ingestion:
      python main.py ingest-neptune -e your_export.txt -o scenes.parquet
   
   Options:
   -e, --export FILE       Path to Neptune export file (required)
   -o, --output FILE       Output parquet file (default: scenes.parquet)
   --no-append             Create new store instead of merging
   
   Example:
   python main.py ingest-neptune -e thunderchild_chapter3.txt -o scenes.parquet


2️⃣  LLAMA-SERVER CHAT EXPORTS
   ─────────────────────────────────────────────────────────────────────────
   Perfect for: Chat histories from llama.cpp web interface
   
   Steps:
   a) Export chat from llama-server (e.g., via /api/chat/export)
   b) Run ingestion:
      python main.py ingest-llama -e export.json -o chats.parquet
   
   Options:
   -e, --export FILE       Path to llama-server export JSON (required)
   -o, --output FILE       Output parquet file (default: llama_chats.parquet)
   
   Example:
   python main.py ingest-llama -e my_chat_2025.json -o character_scenes.parquet


3️⃣  GENERIC CHAT TRANSCRIPTS
   ─────────────────────────────────────────────────────────────────────────
   Perfect for: Discord exports, Slack logs, custom JSON, or plain text
   
   For JSON format:
   python main.py ingest-chat -s export.json --type json -o chat_store.parquet
   
   For plain text (auto-chunked):
   python main.py ingest-chat -s transcript.txt --type txt -o chat_store.parquet
   
   Options:
   -s, --source FILE       Source file (required)
   --type {json,txt}       File type (default: json)
   -o, --output FILE       Output parquet file (default: chat_transcripts.parquet)
   --chunk-size INT        Text chunk size in characters (default: 500, text only)
   
   JSON Format Expected:
   [
     {"timestamp": "2025-12-05T12:00:00", "user": "Alice", 
      "message": "Hello", "channel": "general"},
     ...
   ]


📊 AFTER INGESTION: SEARCH YOUR DATA
   ─────────────────────────────────────────────────────────────────────────
   
   Basic search:
   python main.py search "your query" -s scenes.parquet
   
   With BGE reranking (better accuracy):
   python main.py search "your query" -s scenes.parquet --rerank
   
   Interactive mode (model cached):
   python main.py interactive -s scenes.parquet --rerank
   
   View statistics:
   python main.py stats -s scenes.parquet
   
   List by filter:
   python main.py list -s scenes.parquet --location "bridge"
   
   Export for LLM:
   python main.py export "your query" -f llm-context -o context.md


💾 MERGING MULTIPLE SOURCES
   ─────────────────────────────────────────────────────────────────────────
   By default, ingestions APPEND to existing stores (merge with deduplication).
   
   To start fresh:
   python main.py ingest-neptune -e export.txt -o scenes.parquet --no-append
   
   To add more later:
   python main.py ingest-neptune -e export2.txt -o scenes.parquet  # appends


🔍 TROUBLESHOOTING
   ─────────────────────────────────────────────────────────────────────────
   
   Neptune export format wrong?
   → Make sure you export from Neptune, not copy-paste. File → Export.
   
   llama-server export empty?
   → Check that /api/chat/export endpoint returns valid JSON.
   
   Ingestion slow?
   → Large files take time for embedding. Use --chunk-size to reduce.
     First ingestion downloads embedding model (~200MB).
   
   Merge errors?
   → Use --no-append to create fresh store if having issues.
   → Or delete existing .parquet file and reimport.


═══════════════════════════════════════════════════════════════════════════════
    """
    print(help_text)


# ============================================================================
# SEARCH COMMANDS
# ============================================================================

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
    if not Path(args.store).exists():
        print(f"❌ Vector store not found: {args.store}")
        print("   First ingest narratives using: python main.py ingest-help")
        sys.exit(1)
    
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
            print("❌ Vector store not found. Ingest narratives first.")
            sys.exit(1)
        
        formatter = SceneQueryFormatter(store)
        results = store.query(args.query, n_results=args.limit)
        print(formatter.format_results(results, args.query))


def list_command(args):
    """List scenes by metadata criteria"""
    import polars as pl
    
    if not Path(args.store).exists():
        print(f"❌ Vector store not found: {args.store}")
        sys.exit(1)
    
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
    if not Path(args.store).exists():
        print(f"❌ Vector store not found: {args.store}")
        sys.exit(1)
    
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
    if not Path(args.store).exists():
        print(f"❌ Vector store not found: {args.store}")
        sys.exit(1)
    
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
    print("Interactive Search Mode")
    print(f"Store: {args.store}")
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
    
    if not Path(args.store).exists():
        print(f"❌ Vector store not found: {args.store}")
        sys.exit(1)
    
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
        print(f"✅ Exported {len(results['ids'])} results to {args.output}")
    else:
        print(output)


def migrate_command(args):
    """(Legacy) Migrate from ChromaDB to Polars"""
    print("🚀 Starting ChromaDB → Polars migration...")
    store = PolarsVectorStore(args.output)
    store.save_from_chromadb(None)
    store.stats()
    print("✅ Migration complete!")


def main():
    parser = argparse.ArgumentParser(
        description="NaRAGtive: Narrative RAG with Semantic Search & Reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:

  1. Ingest narratives:
     python main.py ingest-neptune -e export.txt
     python main.py ingest-llama -e chat_export.json
  
  2. Search:
     python main.py search "your query"
     python main.py search "your query" --rerank  # better accuracy
  
  3. Interactive mode:
     python main.py interactive --rerank
  
  For detailed help:
     python main.py ingest-help
     python main.py search --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ========== INGESTION COMMANDS ==========
    
    # Ingest help
    help_parser = subparsers.add_parser(
        "ingest-help",
        help="Show detailed ingestion workflows and examples"
    )
    help_parser.set_defaults(func=ingest_help_command)
    
    # Ingest Neptune
    neptune_parser = subparsers.add_parser(
        "ingest-neptune",
        help="Ingest Neptune AI RPG narrative export"
    )
    neptune_parser.add_argument(
        "-e", "--export",
        required=True,
        help="Path to Neptune export file"
    )
    neptune_parser.add_argument(
        "-o", "--output",
        default="./scenes.parquet",
        help="Output parquet file (default: ./scenes.parquet)"
    )
    neptune_parser.add_argument(
        "--no-append",
        action="store_true",
        help="Create new store instead of merging with existing"
    )
    neptune_parser.set_defaults(func=ingest_neptune_command, append=True)
    
    # Ingest llama-server
    llama_parser = subparsers.add_parser(
        "ingest-llama",
        help="Ingest llama-server chat export"
    )
    llama_parser.add_argument(
        "-e", "--export",
        required=True,
        help="Path to llama-server export JSON file"
    )
    llama_parser.add_argument(
        "-o", "--output",
        default="./llama_chats.parquet",
        help="Output parquet file (default: ./llama_chats.parquet)"
    )
    llama_parser.set_defaults(func=ingest_llama_command)
    
    # Ingest generic chat
    chat_parser = subparsers.add_parser(
        "ingest-chat",
        help="Ingest generic chat transcripts (JSON or text)"
    )
    chat_parser.add_argument(
        "-s", "--source",
        required=True,
        help="Path to source file (JSON or text)"
    )
    chat_parser.add_argument(
        "--type",
        choices=["json", "txt"],
        default="json",
        help="File type (default: json)"
    )
    chat_parser.add_argument(
        "-o", "--output",
        default="./chat_transcripts.parquet",
        help="Output parquet file (default: ./chat_transcripts.parquet)"
    )
    chat_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Text chunk size in characters for text files (default: 500)"
    )
    chat_parser.set_defaults(func=ingest_chat_command)
    
    # ========== SEARCH COMMANDS ==========
    
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
        default="./scenes.parquet",
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
        default="./scenes.parquet",
        help="Path to vector store"
    )
    list_parser.set_defaults(func=list_command)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show store statistics")
    stats_parser.add_argument(
        "-s", "--store",
        default="./scenes.parquet",
        help="Path to vector store"
    )
    stats_parser.add_argument(
        "--show-reranker",
        action="store_true",
        help="Also show BGE reranker statistics"
    )
    stats_parser.set_defaults(func=stats_command)
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Interactive search mode"
    )
    interactive_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Number of results per search"
    )
    interactive_parser.add_argument(
        "-s", "--store",
        default="./scenes.parquet",
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
    export_parser = subparsers.add_parser(
        "export",
        help="Export search results for reranking/LLM"
    )
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
        default="./scenes.parquet",
        help="Path to vector store"
    )
    export_parser.set_defaults(func=export_command)
    
    # Migrate command (legacy)
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="(Legacy) Migrate from ChromaDB to Polars"
    )
    migrate_parser.add_argument(
        "-o", "--output",
        default="./thunderchild_scenes.parquet",
        help="Output parquet file"
    )
    migrate_parser.set_defaults(func=migrate_command)
    
    args = parser.parse_args()
    
    if not args.command:
        print_banner()
        parser.print_help()
        print("\n" + "=" * 80)
        print("QUICK START:")
        print("=" * 80)
        print("\n1. Show ingestion help:")
        print("   python main.py ingest-help\n")
        print("2. Ingest your narratives:")
        print("   python main.py ingest-neptune -e your_export.txt")
        print("   python main.py ingest-llama -e chat_export.json\n")
        print("3. Search:")
        print("   python main.py search \"your query\"")
        print("   python main.py search \"your query\" --rerank\n")
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
