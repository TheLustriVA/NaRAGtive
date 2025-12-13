# Multi-Vector Store Architecture Design for NaRAGtive

**Date:** December 13, 2025
**Target:** Implementation via Perplexity Labs PR
**Status:** Ready for Labs Implementation

---

## Executive Summary

This document outlines how to add **multi-vector-store support** to NaRAGtive, allowing seamless switching between different parquet databases for different use cases (Neptune RPG chats, Perplexity chats, text datasets, etc.) while maintaining backward compatibility and the current architecture style.

**Key Principle:** Keep the code philosophy the same—simple, type-hinted, production-ready—while adding store management as a thin layer above the existing `PolarsVectorStore`.

---

## Current Architecture Analysis

### Strengths to Preserve

1. **Single-class simplicity** - `PolarsVectorStore` is a self-contained, understandable class
2. **Type-hinted throughout** - Python 3.13 full type hints
3. **Composition pattern** - `SceneQueryFormatter` and rerankers extend, don't modify core
4. **CLI-driven workflow** - `main.py` orchestrates operations via commands
5. **No external state** - Everything is file-based (parquet files)

### Current Limitations

1. **Hard-coded file path** - Each operation must specify `-s/--store` with full path
2. **No store listing** - Can't see what stores exist
3. **No store registry** - No way to manage/alias stores (e.g., "character-scenes", "perplexity-chats")
4. **No store metadata** - Don't know what's in each parquet file without loading it
5. **No store switching** - Must change CLI args every time
6. **Scattered across commands** - Each command implements its own path handling

---

## Recommended Solution: VectorStoreRegistry Pattern

### Overview

Implement a **`VectorStoreRegistry`** class that:

1. **Maintains a store directory** - Default: `~/.naragtive/stores/`
2. **Tracks store metadata** - Name, path, ingestion date, record count, source type
3. **Provides store aliases** - Users can name stores ("neptune-campaign", "perplexity-research")
4. **Integrates with CLI** - `--store-name` instead of `--store` path
5. **Stays minimal** - Thin wrapper, no behavioral changes to PolarsVectorStore

### Architecture Diagram

```
User (CLI)
    ↓
main.py (commands)
    ↓
VectorStoreRegistry.get_store("campaign-1")
    ↓
Returns PolarsVectorStore instance (unchanged)
    ↓
    ├─ polars_vectorstore.py (PolarsVectorStore - NO CHANGES)
    ├─ polars_vectorstore.py (SceneQueryFormatter - NO CHANGES)
    └─ bge_reranker_integration.py (PolarsVectorStoreWithReranker - NO CHANGES)
```

**Key point:** `PolarsVectorStore` remains completely unchanged. The registry just manages which file to open.

---

## Implementation Details

### 1. New File: `naragtive/store_registry.py` (~250 lines)

Key classes:

```python
@dataclass
class StoreMetadata:
    """Metadata about a vector store."""
    name: str                    # "campaign-1", "perplexity-research"
    path: Path                   # Full path to .parquet
    created_at: str              # ISO datetime
    source_type: str             # "neptune", "llama-server", "perplexity-chat"
    record_count: int            # Number of scenes
    description: Optional[str]

class VectorStoreRegistry:
    """Manage multiple vector stores."""
    REGISTRY_DIR = Path.home() / ".naragtive" / "stores"
    REGISTRY_FILE = REGISTRY_DIR / "registry.json"
    
    Methods:
    - register(name, path, source_type, description) → StoreMetadata
    - get(name) → PolarsVectorStore
    - list_stores() → List[StoreMetadata]
    - set_default(name) → None
    - get_default() → str
    - delete(name) → None
    - rename(old_name, new_name) → None
    - print_table() → None  # Pretty print registry
```

**Core responsibilities:**
- Maintain `~/.naragtive/stores/registry.json` with store metadata
- Manage `~/.naragtive/stores/default.txt` for default store
- No file-system side effects except registry/default files
- Return PolarsVectorStore instances pointing to correct parquet files

### 2. Modified `main.py` - Minimal Changes

**Changes:** ~300 lines (mostly argument additions)

**New functions:**
```python
def resolve_store_path(args) -> str:
    """Resolve --store or --store-name to actual file path."""
    # Check explicit path first (backward compat)
    # Then check named store
    # Then use default
    # Raise helpful error if none available

def list_stores_command(args):
    """List all registered stores with table formatting."""

def create_store_command(args):
    """Register a parquet file as a named store."""

def set_default_command(args):
    """Set the default store."""
```

**Modified functions:** (All search/stats/export/etc.)
```python
# Before:
def query_command(args):
    store_path = args.store  # Hard-coded path
    store = PolarsVectorStore(store_path)

# After:
def query_command(args):
    store_path = resolve_store_path(args)  # Smart resolution
    store = PolarsVectorStore(store_path)
```

**New CLI arguments:** (Add to each command that uses stores)
```python
parser.add_argument(
    "--store-name",
    help="Named store from registry (overrides --store)"
)
```

**New subcommands:**
- `list-stores` - Show all registered stores
- `create-store <name> <path> <type>` - Register a store
- `set-default <name>` - Set default store

### 3. Ingestion Integration (Minimal)

Add `--register` flag to ingestion commands:

```python
# ingest_neptune_command, ingest_llama_command, ingest_chat_command
if args.register:
    registry = VectorStoreRegistry()
    registry.register(
        name=args.register,
        path=args.output,
        source_type="neptune",  # or "llama-server", etc.
        description=f"...",
        record_count=len(df),
    )
    if len(registry._stores) == 1:
        registry.set_default(args.register)
```

---

## File Structure

```
naragtive/
├── __init__.py
├── polars_vectorstore.py            (NO CHANGES)
├── bge_reranker_integration.py      (NO CHANGES)
├── ingest_chat_transcripts.py       (NO CHANGES)
├── ingest_llama_server_chat.py      (NO CHANGES)
├── reranker_export.py               (NO CHANGES)
└── store_registry.py                (NEW ~250 lines)

main.py                              (MODIFIED ~300 lines)

docs/
└── multi-store-architecture.md      (This file)

tests/
└── test_store_registry.py           (NEW ~300 lines)

~/.naragtive/stores/                (NEW - Auto-created)
├── registry.json                   (Store metadata)
├── default.txt                     (Default store name)
├── campaign-1.parquet
├── perplexity-research.parquet
└── text-dataset.parquet
```

---

## Usage Examples

### Before (Current)
```bash
# Must remember full paths every time
python main.py ingest-neptune -e export.txt -o ./stores/campaign1.parquet
python main.py search "query" -s ./stores/campaign1.parquet --rerank
python main.py search "other" -s ./stores/campaign1.parquet --rerank
python main.py stats -s ./stores/campaign1.parquet
python main.py interactive -s ./stores/campaign1.parquet --rerank
```

### After (With Registry)
```bash
# Register stores with memorable names
python main.py ingest-neptune -e campaign1.txt -o scenes.parquet --register "campaign-1"
python main.py ingest-llama -e perplexity_export.json -o chats.parquet --register "perplexity-research"
python main.py ingest-chat -s dataset.json -o data.parquet --register "text-dataset"

# List available stores
python main.py list-stores
# Output:
# ⭐ campaign-1           neptune  500 records
#    perplexity-research llama    750 records
#    text-dataset        chat     1200 records

# Use by name (or omit for default)
python main.py search "Admiral command"  # Uses default
python main.py search "AI ethics" --store-name "perplexity-research"
python main.py stats --store-name "campaign-1"
python main.py interactive --store-name "campaign-1" --rerank
python main.py export "query" --store-name "perplexity-research" -f llm-context -o context.md

# Manage stores
python main.py set-default "perplexity-research"
python main.py list-stores  # Shows perplexity-research is now default (⭐)
```

---

## Design Principles

1. ✅ **No core changes** - `PolarsVectorStore` class is untouched
2. ✅ **Backward compatible** - Old `-s/--store /path/file` still works
3. ✅ **Minimal CLI changes** - New flags optional, defaults work
4. ✅ **Thin wrapper** - Registry is ~250 lines, super focused
5. ✅ **Type-hinted** - Full Python 3.13 typing throughout
6. ✅ **Production-ready** - Error handling, validation, tests
7. ✅ **Git workflow ready** - Easy to test, review, merge
8. ✅ **Composition over inheritance** - Uses the existing class, doesn't modify it
9. ✅ **Atomic operations** - Registry writes are safe
10. ✅ **User-friendly** - Helpful error messages, clear CLI

---

## Testing Strategy

Create `tests/test_store_registry.py` (~300 lines):

```python
import tempfile
from pathlib import Path
from naragtive.store_registry import VectorStoreRegistry, StoreMetadata

def test_register_store():
    """Test registering a new store"""
    # Create dummy parquet
    # Register it
    # Assert metadata correct

def test_get_store():
    """Test retrieving registered store instance"""
    # Register a store
    # Call get()
    # Assert returns PolarsVectorStore with correct path

def test_default_store():
    """Test setting/getting default store"""
    # Register multiple stores
    # Set default
    # Assert get_default() returns correct one

def test_registry_persistence():
    """Test that registry.json persists across instances"""
    # Register in one instance
    # Create new instance
    # Assert same stores still there

def test_duplicate_name_error():
    """Test that duplicate store names are rejected"""

def test_missing_file_error():
    """Test that registering non-existent file fails"""

def test_metadata_to_dict_from_dict():
    """Test StoreMetadata serialization"""
```

---

## Why This Design

| Aspect | Why |
|--------|-----|
| **Solves the problem** | Named stores, easy switching, persistent config |
| **Minimal code** | ~250 lines registry + ~300 lines main changes |
| **No breaking changes** | Existing `-s /path` flag keeps working |
| **Extensible** | Can add store groups/sharing/versioning later |
| **Pythonic** | Type hints, dataclasses, composition |
| **Testable** | Registry logic isolated, easy unit tests |
| **User-friendly** | Intuitive CLI, helpful errors |
| **Production-ready** | Error handling, validation, atomic writes |
| **Git workflow** | Clean PR, easy review, quick merge |

---

## Integration Checklist (For Labs)

- [ ] Implement `naragtive/store_registry.py` (~250 lines)
- [ ] Modify `main.py` to add registry integration (~300 lines)
- [ ] Create `tests/test_store_registry.py` (~300 lines, full coverage)
- [ ] Update `README.md` with multi-store examples
- [ ] Update `docs/multi-store-architecture.md` (this file) with any clarifications
- [ ] All code fully type-hinted (Python 3.13)
- [ ] All docstrings complete
- [ ] All error messages helpful
- [ ] PR ready for merge

---

## Next Steps

1. **Send to Labs** with this spec + `DELIVERABLE` section below
2. **Labs implements** using this architecture
3. **PR review** focuses on: type hints, tests, CLI UX
4. **Merge to main**
5. **You manage** multiple stores for different projects!

---

## DELIVERABLE (For Labs Prompt)

```
You have access to TheLustriVA/NaRAGtive on GitHub.

TASK: Implement Multi-Vector Store Support

CONTEXT:
NaRAGtive currently requires specifying parquet file paths for every command.
We need to add a registry system to name stores and switch between them easily,
while maintaining backward compatibility and the existing code architecture.

REFERENCE:
See docs/multi-store-architecture.md for complete design specification.
The design is production-ready—this is not a brainstorm, it's a blueprint.

DELIVERABLE:
A pull request on TheLustriVA/NaRAGtive with a feature branch including:

1. NEW FILE: naragtive/store_registry.py (~250 lines)
   - StoreMetadata dataclass
   - VectorStoreRegistry class with methods:
     * register(name, path, source_type, description, record_count)
     * get(name) → PolarsVectorStore
     * list_stores()
     * set_default(name) / get_default()
     * delete(name) / rename(old, new)
     * print_table() for CLI display
   - Persistent registry in ~/.naragtive/stores/registry.json
   - Default store tracking in ~/.naragtive/stores/default.txt

2. MODIFIED FILE: main.py (~300 lines of changes)
   - New resolve_store_path(args) function for smart store resolution
   - New commands: list-stores, create-store <name> <path> <type>, set-default <name>
   - Add --store-name argument to all store-using commands
   - Add --register flag to all ingestion commands
   - Integrate registry with existing commands (search, stats, export, interactive, list)
   - No changes to command logic, only path resolution

3. NEW FILE: tests/test_store_registry.py (~300 lines)
   - test_register_store()
   - test_get_store()
   - test_default_store()
   - test_registry_persistence()
   - test_duplicate_name_error()
   - test_missing_file_error()
   - test_metadata_serialization()
   - Achieve >95% code coverage on registry module

4. UPDATED FILE: README.md
   - Add "Multi-Store Management" section
   - Show before/after usage examples
   - Document: python main.py list-stores, create-store, set-default

5. CODE QUALITY:
   - Full Python 3.13 type hints
   - Comprehensive docstrings
   - No modifications to existing classes (PolarsVectorStore, SceneQueryFormatter, etc.)
   - Backward compatible (old -s flag still works)
   - Error handling with helpful messages

KEY REQUIREMENTS:
- Do NOT modify PolarsVectorStore, SceneQueryFormatter, or reranker classes
- Do NOT break existing command structure (all existing commands must still work)
- Registry is simple wrapper, not a redesign
- Use dataclasses and type hints consistently with existing codebase
- Create ~/.naragtive/stores/ directory automatically
- Persist registry.json and default.txt atomically

REFERENCES TO REVIEW:
- naragtive/polars_vectorstore.py (understand PolarsVectorStore interface)
- main.py (understand CLI command structure)
- tests/test_polars_vectorstore.py (understand test patterns)

RESULT:
After merge, users can:
- python main.py ingest-neptune -e export.txt --register "campaign-1"
- python main.py search "query" --store-name "campaign-1"  OR  (uses default)
- python main.py list-stores  (see all registered stores)
- python main.py set-default "campaign-1"

All while keeping existing code untouched and 100% backward compatible.
```

---

**Document Status:** Ready for Labs Implementation ✨
**Last Updated:** December 13, 2025
**Architecture Style:** Thin wrapper composition pattern, no core changes
