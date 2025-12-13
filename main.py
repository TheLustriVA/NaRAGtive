#!/usr/bin/env python3
"""
NaRAGtive: Narrative RAG with Semantic Search & Reranking

Manage your fiction project's narrative using vector search and embedding-based retrieval.
Features ingestion from multiple sources, semantic search with optional reranking, and analytics.
Supports multiple named vector stores with registry system.

Usage: python main.py [command] [options]

Commands:
  ingest-neptune  - Ingest Neptune AI RPG narrative exports
  ingest-llama    - Ingest llama-server chat exports
  ingest-chat     - Ingest generic chat transcripts (JSON/text)
  search          - Search ingested narratives with optional BGE reranking
  interactive     - Interactive search mode with model caching
  stats           - Show vector store statistics
  list            - List scenes by metadata filters
  export          - Export search results for LLM/reranking
  list-stores     - List all registered stores
  create-store    - Register a new named store
  set-default     - Set default store for commands
  migrate         - (Legacy) Migrate from ChromaDB to Polars
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
from naragtive.store_registry import VectorStoreRegistry


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 80)
    print("NaRAGtive: Narrative RAG with Semantic Search & Reranking")
    print("=" * 80 + "\n")


def resolve_store_path(args) -> str:
    """Resolve --store or --store-name to actual file path.
    
    Resolution order:
    1. If --store is provided and exists as file → use it (backward compat)
    2. If --store-name is provided → look up in registry
    3. If default store exists in registry → use it
    4. Raise error with helpful message
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        Path to vector store parquet file
        
    Raises:
        FileNotFoundError: If store not found
    """
    # Check explicit path first (backward compatibility)
    if hasattr(args, 'store') and args.store != "./scenes.parquet":
        if Path(args.store).exists():
            return args.store
    
    # Try registry
    try:
        registry = VectorStoreRegistry()
    except Exception:
        registry = None
    
    # Check store name from registry
    if hasattr(args, 'store_name') and args.store_name:
        if registry:
            try:
                store = registry.get(args.store_name)
                return store.parquet_path
            except KeyError as e:
                raise FileNotFoundError(f"Store '{args.store_name}' not found in registry.\n{str(e)}")
        else:
            raise FileNotFoundError(f"Registry not available for store '{args.store_name}'")
    
    # Fall back to default store
    if registry:
        default = registry.get_default()
        if default:
            try:
                store = registry.get(default)
                return store.parquet_path
            except KeyError:
                pass
    
    # No store found
    raise FileNotFoundError(
        "No vector store specified or found.\n"
        "Options:\n"
        "  1. Specify path: python main.py search 'query' -s /path/to/file.parquet\n"
        "  2. Use store name: python main.py search 'query' --store-name 'my-store'\n"
        "  3. Ingest data: python main.py ingest-neptune -e export.txt --register 'name'\n"
        "  4. List stores: python main.py list-stores"
    )


# ============================================================================
# REGISTRY COMMANDS
# ============================================================================

def list_stores_command(args):
    """List all registered vector stores"""
    registry = VectorStoreRegistry()
    registry.print_table()


def create_store_command(args):
    """Register a new named store"""
    registry = VectorStoreRegistry()
    
    try:
        metadata = registry.register(
            name=args.name,
            path=Path(args.path),
            source_type=args.source_type,
            description=args.description
        )
        
        print(f"\n✅ Registered store: {metadata.name}")
        print(f"   Path: {metadata.path}")
        print(f"   Type: {metadata.source_type}")
        print(f"   Records: {metadata.record_count}")
        
        # Set as default if first store
        if len(registry.list_stores()) == 1:
            registry.set_default(metadata.name)
            print(f"   Set as default store")
        
        print()
        registry.print_table()
        
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def set_default_command(args):
    """Set default store"""
    registry = VectorStoreRegistry()
    
    try:
        registry.set_default(args.name)
        print(f"\n✅ Set default store to: {args.name}\n")
        registry.print_table()
    except KeyError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


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
        
        # Register if requested
        if hasattr(args, 'register') and args.register:
            registry = VectorStoreRegistry()
            try:
                metadata = registry.register(
                    name=args.register,
                    path=Path(args.output),
                    source_type="neptune",
                    description=f"Neptune narrative from {Path(args.export).name}",
                    record_count=len(df)
                )
                print(f"\n📌 Registered as: {metadata.name}")
                
                # Set as default if first
                if len(registry.list_stores()) == 1:
                    registry.set_default(metadata.name)
                    print(f"   Set as default store")
                    
            except ValueError as e:
                print(f"⚠️  Warning: Could not register: {e}")
        
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
        
        # Register if requested
        if hasattr(args, 'register') and args.register:
            registry = VectorStoreRegistry()
            try:
                metadata = registry.register(
                    name=args.register,
                    path=Path(args.output),
                    source_type="llama-server",
                    description=f"Llama-server chats from {Path(args.export).name}",
                    record_count=len(df)
                )
                print(f"\n📌 Registered as: {metadata.name}")
                
                # Set as default if first
                if len(registry.list_stores()) == 1:
                    registry.set_default(metadata.name)
                    print(f"   Set as default store")
                    
            except ValueError as e:
                print(f"⚠️  Warning: Could not register: {e}")
        
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
        
        # Register if requested
        if hasattr(args, 'register') and args.register:
            registry = VectorStoreRegistry()
            try:
                source_type = args.type if args.type in ["json", "txt"] else "chat"
                metadata = registry.register(
                    name=args.register,
                    path=Path(args.output),
                    source_type=source_type,
                    description=f"Chat transcript from {Path(args.source).name}",
                    record_count=len(df)
                )
                print(f"\n📌 Registered as: {metadata.name}")
                
                # Set as default if first
                if len(registry.list_stores()) == 1:
                    registry.set_default(metadata.name)
                    print(f"   Set as default store")
                    
            except ValueError as e:
                print(f"⚠️  Warning: Could not register: {e}")
        
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
   --register NAME         Register as named store after ingestion
   
   Example:
   python main.py ingest-neptune -e thunderchild_chapter3.txt --register "campaign-1"


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
   --register NAME         Register as named store after ingestion
   
   Example:
   python main.py ingest-llama -e my_chat_2025.json --register "perplexity-research"


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
   --register NAME         Register as named store after ingestion
   
   JSON Format Expected:
   [
     {"timestamp": "2025-12-05T12:00:00", "user": "Alice", 
      "message": "Hello", "channel": "general"},
     ...
   ]


📊 AFTER INGESTION: SEARCH YOUR DATA
   ─────────────────────────────────────────────────────────────────────────
   
   Basic search:
   python main.py search "your query" --store-name campaign-1
   
   With BGE reranking (better accuracy):
   python main.py search "your query" --store-name campaign-1 --rerank
   
   Interactive mode (model cached):
   python main.py interactive --store-name campaign-1 --rerank
   
   View statistics:
   python main.py stats --store-name campaign-1
   
   List by filter:
   python main.py list --store-name campaign-1 --location "bridge"
   
   Export for LLM:
   python main.py export "your query" --store-name campaign-1 -f llm-context -o context.md


💾 MANAGING MULTIPLE STORES
   ─────────────────────────────────────────────────────────────────────────
   
   List all registered stores:
   python main.py list-stores
   
   Register a store manually:
   python main.py create-store "my-store" ./scenes.parquet neptune
   
   Set default store:
   python main.py set-default "my-store"
   
   Use default automatically:
   python main.py search "query"  # Uses default store


╔════════════════════════════════════════════════════════════════════════════╗
║                              EXAMPLES                                      ║
╚════════════════════════════════════════════════════════════════════════════╝

1. Ingest multiple sources and manage them:
   python main.py ingest-neptune -e chapter1.txt --register campaign-1
   python main.py ingest-llama -e chat.json --register perplexity-research
   python main.py list-stores
   python main.py set-default campaign-1

2. Search across your stores:
   python main.py search "Admiral command" --store-name campaign-1 --rerank
   python main.py search "ethics" --store-name perplexity-research --rerank

3. Keep using old way (backward compatible):
   python main.py search "query" -s /path/to/scenes.parquet
   # Still works!


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
    try:
        store_path = resolve_store_path(args)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    if args.rerank:
        store = PolarsVectorStoreWithReranker(store_path)
        results = store.query_and_rerank(
            args.query,
            initial_k=args.initial_k,
            final_k=args.limit,
            normalize=True
        )
        print(print_reranked_results(results, args.query))
    else:
        # Standard embedding-only search
        store = PolarsVectorStore(store_path)
        
        if not store.load():
            print("❌ Vector store not found. Ingest narratives first.")
            sys.exit(1)
        
        formatter = SceneQueryFormatter(store)
        results = store.query(args.query, n_results=args.limit)
        print(formatter.format_results(results, args.query))


def list_command(args):
    """List scenes by metadata criteria"""
    import polars as pl
    
    try:
        store_path = resolve_store_path(args)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    store = PolarsVectorStore(store_path)
    
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
    try:
        store_path = resolve_store_path(args)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    store = PolarsVectorStore(store_path)
    store.load()
    store.stats()
    
    # Show reranker stats if available
    if args.show_reranker:
        try:
            reranker_store = PolarsVectorStoreWithReranker(store_path)
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
    try:
        store_path = resolve_store_path(args)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    if args.rerank:
        store = PolarsVectorStoreWithReranker(store_path)
        stats = store.get_reranker_stats()
        reranker_status = f"✅ {stats['model']} ({stats['vram_mb']:.0f}MB VRAM, FP16)"
    else:
        store = PolarsVectorStore(store_path)
        if not store.load():
            print("❌ Vector store not found.")
            sys.exit(1)
        reranker_status = "disabled"
    
    print("\n" + "=" * 80)
    print("Interactive Search Mode")
    print(f"Store: {store_path}")
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
    
    try:
        store_path = resolve_store_path(args)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    store = PolarsVectorStore(store_path)
    
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
     python main.py ingest-neptune -e export.txt --register "campaign-1"
     python main.py ingest-llama -e chat_export.json --register "perplexity"
  
  2. List registered stores:
     python main.py list-stores
  
  3. Search by store name:
     python main.py search "your query" --store-name "campaign-1"
     python main.py search "your query" --store-name "campaign-1" --rerank
  
  4. Set default store:
     python main.py set-default "campaign-1"
     python main.py search "query"  # Uses default
  
  For detailed help:
     python main.py ingest-help
     python main.py search --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ========== REGISTRY COMMANDS ==========
    
    # List stores
    list_stores_parser = subparsers.add_parser(
        "list-stores",
        help="List all registered vector stores"
    )
    list_stores_parser.set_defaults(func=list_stores_command)
    
    # Create store
    create_store_parser = subparsers.add_parser(
        "create-store",
        help="Register a new named vector store"
    )
    create_store_parser.add_argument(
        "name",
        help="Store name (e.g., 'campaign-1')"
    )
    create_store_parser.add_argument(
        "path",
        help="Path to parquet file"
    )
    create_store_parser.add_argument(
        "source_type",
        help="Source type (neptune, llama-server, chat, etc.)"
    )
    create_store_parser.add_argument(
        "-d", "--description",
        help="Optional description"
    )
    create_store_parser.set_defaults(func=create_store_command)
    
    # Set default
    set_default_parser = subparsers.add_parser(
        "set-default",
        help="Set default store for commands"
    )
    set_default_parser.add_argument(
        "name",
        help="Store name to set as default"
    )
    set_default_parser.set_defaults(func=set_default_command)
    
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
    neptune_parser.add_argument(
        "--register",
        help="Register as named store after ingestion"
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
    llama_parser.add_argument(
        "--register",
        help="Register as named store after ingestion"
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
    chat_parser.add_argument(
        "--register",
        help="Register as named store after ingestion"
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
        help="Path to vector store (for backward compatibility)"
    )
    search_parser.add_argument(
        "--store-name",
        help="Named store from registry (overrides --store)"
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
        help="Path to vector store (for backward compatibility)"
    )
    list_parser.add_argument(
        "--store-name",
        help="Named store from registry (overrides --store)"
    )
    list_parser.set_defaults(func=list_command)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show store statistics")
    stats_parser.add_argument(
        "-s", "--store",
        default="./scenes.parquet",
        help="Path to vector store (for backward compatibility)"
    )
    stats_parser.add_argument(
        "--store-name",
        help="Named store from registry (overrides --store)"
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
        help="Path to vector store (for backward compatibility)"
    )
    interactive_parser.add_argument(
        "--store-name",
        help="Named store from registry (overrides --store)"
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
        help="Path to vector store (for backward compatibility)"
    )
    export_parser.add_argument(
        "--store-name",
        help="Named store from registry (overrides --store)"
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
        print("\n1. Ingest with auto-registration:")
        print("   python main.py ingest-neptune -e export.txt --register 'campaign-1'\n")
        print("2. List your stores:")
        print("   python main.py list-stores\n")
        print("3. Search by store name:")
        print("   python main.py search \"your query\" --store-name campaign-1 --rerank\n")
        print("4. Set and use default:")
        print("   python main.py set-default campaign-1")
        print("   python main.py search \"query\"  # Uses default\n")
        print("For detailed help: python main.py ingest-help\n")
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
