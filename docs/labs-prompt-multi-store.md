# Labs Prompt: Multi-Vector Store Implementation for NaRAGtive

**Copy & paste this ENTIRE text to Perplexity Labs**

---

## THE PROMPT

```
You have access to TheLustriVA/NaRAGtive on GitHub.

TASK: Implement Multi-Vector Store Support

CONTEXT:
NaRAGtive currently requires specifying parquet file paths for every operation.
We need a registry system to name stores, set defaults, and switch between them
for different projects (Neptune RPG chats, Perplexity conversations, text datasets).

REFERENCE ARCHITECTURE:
See docs/multi-store-architecture.md in the repo for the complete technical
specification. This is NOT a brainstorm—it's a production-ready blueprint.

DELIVERABLE:
A pull request on TheLustriVA/NaRAGtive with a feature branch containing:

1. NEW FILE: naragtive/store_registry.py (~250 lines)

   a) StoreMetadata dataclass:
      - name: str (e.g., "campaign-1")
      - path: Path (full path to .parquet file)
      - created_at: str (ISO datetime)
      - source_type: str ("neptune", "llama-server", "perplexity-chat", etc.)
      - record_count: int (number of scenes/entries)
      - description: Optional[str]
      - to_dict() and from_dict() methods for JSON serialization

   b) VectorStoreRegistry class:
      - REGISTRY_DIR = Path.home() / ".naragtive" / "stores"
      - REGISTRY_FILE = REGISTRY_DIR / "registry.json"

      Methods:
      - __init__(): Load existing registry or create new
      - register(name, path, source_type, description, record_count) → StoreMetadata
        * Validates parquet file exists
        * Detects record_count from file if not provided
        * Raises ValueError if name already exists
        * Raises FileNotFoundError if path doesn't exist
        * Saves updated registry

      - get(name) → PolarsVectorStore
        * Returns PolarsVectorStore instance pointing to correct parquet
        * Handles "default" keyword to use default store
        * Raises KeyError with helpful message if not found

      - list_stores() → List[StoreMetadata]
        * Returns all registered stores

      - get_default() → str
        * Returns default store name from ~/.naragtive/stores/default.txt
        * Returns first registered store if no explicit default
        * Returns None if no stores exist

      - set_default(name) → None
        * Validates store exists
        * Writes to ~/.naragtive/stores/default.txt
        * Raises KeyError if store not found

      - delete(name) → None
        * Removes from registry and persists

      - rename(old_name, new_name) → None
        * Renames store in registry
        * Raises KeyError if old not found
        * Raises ValueError if new already exists

      - print_table() → None
        * Pretty-print registry as formatted table
        * Shows all stores with metadata
        * Marks default store with ⭐

   Private methods:
   - _load_registry() → Dict[str, StoreMetadata]
   - _save_registry() → None (atomic writes)

   Notes:
   - Use dataclasses throughout
   - Full Python 3.13 type hints
   - Comprehensive docstrings on all methods
   - Error messages are helpful and specific
   - Create ~/.naragtive/stores/ directory automatically if needed

2. MODIFIED FILE: main.py (~300 lines of additions)

   Add imports:
   - from naragtive.store_registry import VectorStoreRegistry

   Add new helper function:
   - resolve_store_path(args) → str
     * Check if args.store exists (backward compat explicit path)
     * Check if args.store_name exists (new named store)
     * Fall back to default store from registry
     * Raise ValueError with helpful message if none available

   Add new commands:

   a) list_stores_command(args)
      - Call registry.print_table()
      - Show all registered stores

   b) create_store_command(args)
      - Args: name, path, source_type, --description
      - Register the store
      - Print confirmation with metadata
      - Set as default if first store

   c) set_default_command(args)
      - Args: name
      - Call registry.set_default(name)
      - Print confirmation
      - Call registry.print_table() to show result

   Modify existing command functions (query, stats, interactive, list, export):
   - At start, call: store_path = resolve_store_path(args)
   - Use store_path instead of args.store
   - No other behavioral changes

   Add arguments to existing parsers:
   - search_parser, stats_parser, interactive_parser, list_parser, export_parser:
     * Add: --store-name (help="Named store from registry (overrides --store)")

   - ingest_neptune_parser, ingest_llama_parser, ingest_chat_parser:
     * Add: --register (help="Register as named store after ingestion")
     * After successful ingestion, if args.register:
       - registry.register(args.register, args.output, source_type, ...)
       - If first store, set as default
       - Print confirmation

   Add new subcommands in argparse:

   a) list-stores:
      parser = subparsers.add_parser("list-stores", help="List registered stores")
      parser.set_defaults(func=list_stores_command)

   b) create-store:
      parser = subparsers.add_parser("create-store", help="Register a new store")
      parser.add_argument("name", help="Store name")
      parser.add_argument("path", help="Path to parquet file")
      parser.add_argument("source_type", help="Source type (neptune, llama-server, etc.)")
      parser.add_argument("-d", "--description", help="Optional description")
      parser.set_defaults(func=create_store_command)

   c) set-default:
      parser = subparsers.add_parser("set-default", help="Set default store")
      parser.add_argument("name", help="Store name")
      parser.set_defaults(func=set_default_command)

3. NEW FILE: tests/test_store_registry.py (~300 lines)

   Import tempfile for isolated testing

   Test classes/functions:

   a) test_register_store()
      - Create temp parquet file
      - Register it
      - Assert metadata correct
      - Assert file exists in registry.json

   b) test_get_store()
      - Register store
      - Call get(name)
      - Assert returns PolarsVectorStore instance
      - Assert parquet_path matches

   c) test_get_default()
      - Register multiple stores
      - Assert first is default
      - Call set_default(second)
      - Assert get_default() returns second

   d) test_registry_persistence()
      - Register store in one instance
      - Create new instance
      - Assert same stores still there

   e) test_duplicate_name_error()
      - Register store
      - Try to register with same name
      - Assert ValueError raised

   f) test_missing_file_error()
      - Try to register non-existent file
      - Assert FileNotFoundError raised

   g) test_metadata_serialization()
      - Create StoreMetadata
      - Call to_dict()
      - Call from_dict() on result
      - Assert matches original

   h) test_delete_store()
      - Register store
      - Call delete()
      - Assert no longer in list_stores()

   i) test_rename_store()
      - Register store
      - Rename it
      - Assert new name exists
      - Assert old name gone

   Requirements:
   - Aim for >95% code coverage on registry module
   - Use pytest fixtures for setup
   - All tests should be isolated (use temp directories)
   - Clear test names and docstrings

4. UPDATED FILE: README.md

   Add new section "Multi-Store Management" after "Quick Start" that includes:
   - Overview of registry system
   - Example: python main.py ingest-neptune -e export.txt --register "campaign-1"
   - Example: python main.py search "query" --store-name "campaign-1"
   - Example: python main.py list-stores
   - Example: python main.py set-default "campaign-1"
   - Show before/after comparison
   - Document that --store (path) still works for backward compatibility

CODE QUALITY REQUIREMENTS:

✓ Full Python 3.13 type hints on all functions and classes
✓ Comprehensive docstrings (class + all public methods)
✓ Error messages are specific and helpful
✓ No modifications to existing classes:
  - polars_vectorstore.py (PolarsVectorStore, SceneQueryFormatter) - UNTOUCHED
  - bge_reranker_integration.py - UNTOUCHED
  - ingest_chat_transcripts.py - UNTOUCHED
  - ingest_llama_server_chat.py - UNTOUCHED
  - reranker_export.py - UNTOUCHED
✓ All existing commands must still work (backward compatible)
✓ All new/modified commands must follow existing CLI patterns
✓ Use dataclasses (not NamedTuple) for StoreMetadata
✓ Atomic file writes for registry.json
✓ Create ~/.naragtive/stores/ directory automatically

NO BEHAVIORAL CHANGES:
- Existing command behavior stays identical
- Only the path resolution method changes (from explicit -s to registry)
- All search/stats/export/list functionality unchanged

COMPLETION CHECKLIST:

☐ naragtive/store_registry.py created with all specified classes/methods
☐ main.py modified with registry integration (all commands still work)
☐ tests/test_store_registry.py created with comprehensive test coverage (>95%)
☐ README.md updated with Multi-Store Management section
☐ All imports in main.py are correct
☐ All type hints present and correct
☐ All docstrings complete and helpful
☐ All error messages specific and actionable
☐ pytest runs successfully: pytest tests/test_store_registry.py -v
☐ All existing tests still pass: pytest tests/ -v
☐ PR is on a feature branch (e.g., feature/multi-vector-stores)
☐ Commit messages are clear and reference this feature

KEY FILES TO REVIEW BEFORE STARTING:
- naragtive/polars_vectorstore.py (understand PolarsVectorStore interface)
- main.py (understand CLI command structure and patterns)
- tests/test_polars_vectorstore.py (understand test patterns used)

RESULT:

After implementation and merge, users can:

  python main.py ingest-neptune -e campaign1.txt -o scenes.parquet --register "campaign-1"
  python main.py ingest-llama -e perplexity.json -o chats.parquet --register "perplexity-research"

  python main.py list-stores
  # ⭐ campaign-1       neptune      500 records
  #    perplexity-research llama   750 records

  python main.py search "Admiral command" --store-name "campaign-1"
  python main.py interactive --store-name "perplexity-research" --rerank
  python main.py set-default "perplexity-research"

  python main.py search "query"  # Now uses perplexity-research by default

All while maintaining full backward compatibility:
  python main.py search "query" -s /explicit/path/file.parquet  # Still works
```

---

## How to Send to Labs

1. **Copy everything in the code block above** (from "You have access..." to the final "```")
2. **Go to Perplexity Labs**
3. **Create new task** and paste the prompt
4. **Tell Labs**: "Reference docs/multi-store-architecture.md in the repo for detailed spec"
5. **Submit and wait** - Labs will implement in 2-4 hours

---

## What to Expect

Labs will deliver:
- ✅ Feature branch with all code complete
- ✅ All tests passing (>95% coverage)
- ✅ Full type hints
- ✅ Production-ready error handling
- ✅ PR ready to merge
- ✅ No breaking changes

**Timeline:** 2-4 hours for implementation

---

**Ready to send!** Copy the prompt and go. 🚀
