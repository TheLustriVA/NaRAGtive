# NaRAGtive: Narrative RAG with Semantic Search & Reranking

A Polars-based retrieval-augmented generation system for fiction projects. Ingest narratives from multiple sources (Neptune AI, llama-server, generic chat), perform semantic search with optional BGE reranking, and extract narrative insights. **Now with multi-store support for managing multiple projects!**

## Quick Start

### 1. Show Ingestion Help

```bash
python main.py ingest-help
```

This displays comprehensive guides for ingesting from three sources: Neptune AI, llama-server, and generic chat transcripts.

### 2. Ingest Your Narratives

#### Neptune AI RPG Exports

```bash
python main.py ingest-neptune -e your_export.txt -o scenes.parquet
```

Options:
- `-e, --export FILE`: Path to Neptune export file (required)
- `-o, --output FILE`: Output parquet file (default: `./scenes.parquet`)
- `--no-append`: Create new store instead of merging
- `--register NAME`: Register as named store after ingestion

**How to export from Neptune:**
1. Open your conversation in Neptune
2. Click "File" → "Export"
3. Save the `.txt` file
4. Ingest with command above

#### llama-server Chat Exports

```bash
python main.py ingest-llama -e chat_export.json -o chats.parquet
```

Options:
- `-e, --export FILE`: Path to llama-server export JSON (required)
- `-o, --output FILE`: Output parquet file (default: `./llama_chats.parquet`)
- `--register NAME`: Register as named store after ingestion

**How to export from llama-server:**
1. Use the web interface chat export feature, or
2. Hit `/api/chat/export` endpoint
3. Ingest with command above

#### Generic Chat Transcripts

```bash
# From JSON (Discord, Slack, custom)
python main.py ingest-chat -s export.json --type json -o chat_store.parquet

# From plain text (auto-chunked)
python main.py ingest-chat -s transcript.txt --type txt -o chat_store.parquet
```

Options:
- `-s, --source FILE`: Source file (required)
- `--type {json,txt}`: File type (default: `json`)
- `-o, --output FILE`: Output parquet file
- `--chunk-size INT`: Text chunk size (default: 500, text mode only)
- `--register NAME`: Register as named store after ingestion

**Expected JSON format:**
```json
[
  {
    "timestamp": "2025-12-05T12:00:00",
    "user": "Alice",
    "message": "Hello world",
    "channel": "general"
  }
]
```

### 3. Search Your Narratives

#### Basic Semantic Search

```bash
python main.py search "Admiral command" -s scenes.parquet
```

Options:
- `query`: Search query (positional, required)
- `-l, --limit INT`: Number of results (default: 10)
- `-s, --store FILE`: Path to vector store (default: `./scenes.parquet`)
- `--store-name NAME`: Use named store from registry

#### With BGE Reranking (Better Accuracy)

```bash
python main.py search "Admiral command" -s scenes.parquet --rerank
```

Options:
- `--rerank`: Enable BGE v2 M3 reranking (two-stage retrieval)
- `--initial-k INT`: Documents to rerank from (default: 50)

#### Interactive Mode (Model Cached)

```bash
python main.py interactive -s scenes.parquet --rerank
```

Enter multiple queries without reloading the model, for faster iterations.

### 4. View Statistics

```bash
python main.py stats -s scenes.parquet
```

Show vector store size, scene counts, and metadata breakdowns.

With reranker stats:
```bash
python main.py stats -s scenes.parquet --show-reranker
```

### 5. List Scenes by Criteria

```bash
python main.py list -s scenes.parquet --location "bridge"
python main.py list -s scenes.parquet --character "Kieran"
python main.py list -s scenes.parquet --date "2025-11-10"
```

Options:
- `-c, --character CHAR`: Filter by character name
- `-l, --location LOC`: Filter by location
- `-d, --date DATE`: Filter by ISO date
- `-s, --store FILE`: Path to vector store
- `--store-name NAME`: Use named store from registry

### 6. Export for LLM/Reranker

```bash
python main.py export "Admiral command" -f llm-context -o context.md
```

Formats:
- `llm-context`: Markdown for LLM prompts
- `bge`: JSON for BGE reranker
- `rag`: Multi-format RAG structure
- `llamafile`: LLamafile compatible JSON
- `jsonl`: JSONL batch format

---

## Multi-Store Management

**New Feature:** Named stores with persistent registry, allowing seamless switching between multiple projects.

### Register Stores During Ingestion

```bash
# Register during Neptune ingestion
python main.py ingest-neptune -e campaign1.txt --register "campaign-1"

# Register during llama-server ingestion
python main.py ingest-llama -e perplexity_chats.json --register "perplexity-research"

# Register generic chat
python main.py ingest-chat -s dataset.json --register "text-dataset"
```

The `--register` flag automatically:
- Saves metadata (name, path, source type, record count)
- Sets as default if it's the first store
- Creates `~/.naragtive/stores/` registry directory

### List Registered Stores

```bash
python main.py list-stores
```

Output:
```
==========================================================================================
REGISTERED VECTOR STORES
==========================================================================================
 ⭐ campaign-1           neptune        500        2025-12-13T04:00... Campaign 1 scenes
    perplexity-research llama-server  750        2025-12-13T04:05... Perplexity chats
    text-dataset        chat           1200       2025-12-13T04:10... Text dataset
==========================================================================================

Default: campaign-1
Use 'python main.py set-default <name>' to change default
```

### Search by Store Name

```bash
# Explicit store name
python main.py search "query" --store-name "campaign-1" --rerank

# Use default store
python main.py search "query"  # Uses default if no -s specified

# Interactive with named store
python main.py interactive --store-name "perplexity-research" --rerank
```

### Set/Change Default Store

```bash
# Set default
python main.py set-default "perplexity-research"

# Now all commands use this store by default
python main.py search "query"           # Uses perplexity-research
python main.py stats                   # Shows perplexity-research stats
```

### Manual Store Registration

```bash
# Register an existing parquet file
python main.py create-store "my-store" /path/to/file.parquet neptune -d "My description"
```

Options:
- `name`: Store name (required)
- `path`: Path to parquet file (required)
- `source_type`: Source origin (neptune, llama-server, chat, etc.) (required)
- `-d, --description`: Optional description

### Before/After Comparison

**Before (without registry):**
```bash
# Must remember paths
python main.py ingest-neptune -e export.txt -o /home/user/stores/campaign1.parquet
python main.py search "query" -s /home/user/stores/campaign1.parquet --rerank
python main.py search "other" -s /home/user/stores/campaign1.parquet
python main.py stats -s /home/user/stores/campaign1.parquet
```

**After (with registry):**
```bash
# Register once
python main.py ingest-neptune -e export.txt --register "campaign-1"

# Use by name
python main.py search "query" --store-name "campaign-1" --rerank
python main.py search "other" --store-name "campaign-1"
python main.py stats --store-name "campaign-1"

# Or use default
python main.py set-default "campaign-1"
python main.py search "query" --rerank
python main.py stats
```

---

## Complete Workflow Examples

### Example 1: Neptune Narrative Analysis

```bash
# 1. Export conversation from Neptune
# File → Export → save_as thunderchild_chapter3.txt

# 2. Ingest and register
python main.py ingest-neptune -e thunderchild_chapter3.txt --register "campaign-1"

# 3. Search for specific scenes
python main.py search "Admiral decision making" --store-name "campaign-1" --rerank

# 4. View all scenes on the bridge
python main.py list --store-name "campaign-1" --location bridge

# 5. Export for LLM context
python main.py export "command conflict" --store-name "campaign-1" -f llm-context -o context.md
```

### Example 2: Multi-Source Narrative

```bash
# 1. Ingest Neptune conversation
python main.py ingest-neptune -e neptune_ch1.txt --register "campaign-1"

# 2. Add llama-server chat to registry (different store)
python main.py ingest-llama -e llama_ch2.json --register "perplexity-research"

# 3. Search campaign
python main.py search "character reaction" --store-name "campaign-1" --rerank

# 4. Search perplexity
python main.py search "AI ethics" --store-name "perplexity-research" --rerank

# 5. View all stores
python main.py list-stores
```

### Example 3: Archive Chat History

```bash
# 1. Export Discord/Slack as JSON
# (Use Discord's export feature or Slack API)

# 2. Ingest generic chat
python main.py ingest-chat -s discord_archive.json --register "discord"

# 3. Interactive search
python main.py interactive --store-name "discord"
# Then enter queries:
# 🔍 Query: lore discussion
# 🔍 Query: character relationship
# 🔍 Query: quit
```

---

## Architecture

### Ingestion Pipeline

1. **Parser**: Format-specific extraction (Neptune text, llama JSON, generic)
2. **Processor**: Convert raw data into scenes (grouping, pairing)
3. **Analyzer**: Extract metadata (themes, tone, characters, locations)
4. **Embedder**: Generate embeddings using all-MiniLM-L6-v2 (384-dim)
5. **Storage**: Save to Polars parquet with metadata
6. **Registry**: Track store metadata in `~/.naragtive/stores/registry.json`

### Search Pipeline

1. **Query Embedding**: Encode search query (384-dim)
2. **Vector Search**: Find top-50 most similar scenes (fast)
3. **(Optional) BGE Reranking**: Cross-encode top-50 → top-10 (accurate)
4. **Formatting**: Display with metadata and text preview

### Supported Sources

| Source | Format | Key Fields | Example |
|--------|--------|-----------|----------|
| Neptune AI | `.txt` export | timestamp, speaker, text | `thunderchild_ch1.txt` |
| llama-server | `.json` export | conversation ID, messages, timestamps | `chat_export.json` |
| Discord | `.json` export | user, message, timestamp, channel | `discord_backup.json` |
| Plain Text | `.txt` file | chunked content | `transcript.txt` |

---

## Store Registry

Metadata stored in `~/.naragtive/stores/`:

```
~/.naragtive/stores/
├── registry.json          # Store metadata (name, path, type, count, etc.)
├── default.txt            # Current default store name
├── campaign-1.parquet     # Actual vector store file
├── perplexity-research.parquet
└── text-dataset.parquet
```

**Backward Compatibility:** Old `-s/--store /path` flag still works:
```bash
python main.py search "query" -s /explicit/path/file.parquet  # Still works!
```

---

## Performance Notes

### Memory Usage

- **Embedding generation**: ~300MB (first run, downloads model)
- **Vector store**: ~50MB per 1000 scenes
- **BGE reranker**: ~1.3GB VRAM (optional, only when using `--rerank`)
- **Registry**: <1MB (metadata only)

### Speed

- **Neptune ingestion**: ~10 scenes/sec (depends on file size)
- **Semantic search**: ~100ms (embedding only)
- **Reranked search**: ~500ms-1s (embedding + reranking)
- **Interactive mode**: ~100ms after first query (model cached)

### Scaling

- **Vector store**: Tested up to 100k scenes
- **Search latency**: Sub-second for most stores
- **Multiple stores**: No performance penalty (independent parquet files)
- **Registry**: O(n) lookup where n = number of stores (typically <100)

---

## Troubleshooting

### Import errors

```
ModuleNotFoundError: No module named 'naragtive'
```

**Solution:** Run from repo root, ensure `naragtive/` is in same directory as `main.py`

### Neptune export format wrong

```
ValueError: TURN_RE regex didn't match
```

**Solution:** Make sure you exported from Neptune (File → Export), not copy-pasted. Export creates proper `***TIMESTAMP - SPEAKER:***` headers.

### Store not found in registry

```
KeyError: Store 'campaign-1' not found in registry
```

**Solution:** List available stores: `python main.py list-stores`

Or register: `python main.py ingest-neptune -e export.txt --register "campaign-1"`

### Ingestion slow

**First run?** Model download (~200MB) happens automatically. Subsequent runs are faster.

**Large file?** Try reducing chunk size:
```bash
python main.py ingest-chat -s large.txt --chunk-size 250 --register "dataset"
```

### Vector store file corrupted

**Solution:** Delete the `.parquet` file and reingest:
```bash
rm scenes.parquet
python main.py ingest-neptune -e export.txt
```

### Registry corrupted

**Solution:** Delete registry and reingest:
```bash
rm -rf ~/.naragtive/stores/
python main.py ingest-neptune -e export.txt --register "campaign-1"
```

---

## Dependencies

- `polars`: DataFrame and parquet storage
- `sentence-transformers`: Embedding generation
- `numpy`: Array operations
- `pyarrow`: Parquet format support

All installed via `uv install` or `pip install -r requirements.txt`

---

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Registry tests
pytest tests/test_store_registry.py -v

# Ingestion tests
pytest tests/test_ingest_chat_transcripts.py -v
pytest tests/test_ingest_llama_server_chat.py -v

# With coverage
pytest tests/ --cov=naragtive
```

See [`TESTING.md`](TESTING.md) for detailed test documentation.

---

## Learning Resources

### For First-Time Users

1. Read this README
2. Run `python main.py ingest-help` for guided examples
3. Try Example 1 (Neptune workflow) above
4. Run `python main.py list-stores` to see registered stores

### For Developers

1. Read ingester docstrings: `naragtive/ingest_chat_transcripts.py`
2. Study registry code: `naragtive/store_registry.py`
3. Review test cases: `tests/test_store_registry.py`
4. Check vector store: `naragtive/polars_vectorstore.py`

### For Advanced Usage

1. Review `docs/LLAMA_SERVER_INGESTION.md` for llama-server specifics
2. Explore `naragtive/bge_reranker_integration.py` for reranking details
3. See `naragtive/reranker_export.py` for export formats
4. Check `docs/multi-store-architecture.md` for design details

---

## Recent Updates

### v1.2.0 - Multi-Store Support

- ✅ New VectorStoreRegistry for managing multiple stores
- ✅ Named store support with `--register` flag on ingestion
- ✅ `list-stores`, `create-store`, `set-default` commands
- ✅ `--store-name` argument for all search commands
- ✅ Default store tracking in `~/.naragtive/stores/default.txt`
- ✅ Persistent registry in `~/.naragtive/stores/registry.json`
- ✅ 100% backward compatible with old `-s/--store` paths
- ✅ Full test coverage for registry module

### v1.1.0 - Ingestion CLI Workflow

- ✅ New `ingest-neptune` command for Neptune AI exports
- ✅ New `ingest-llama` command for llama-server chats
- ✅ New `ingest-chat` command for generic transcripts
- ✅ New `ingest-help` command with comprehensive guides
- ✅ Improved error handling and user feedback
- ✅ Full test coverage for ingestion pipeline

### v1.0.0 - Initial Release

- Polars-based vector storage
- BGE v2 M3 reranking
- Neptune AI ingestion
- Semantic search and interactive mode

---

## License

See LICENSE file.

---

## Citation

If using NaRAGtive in your research:

```bibtex
@misc{naragtive2025,
  title={NaRAGtive: Narrative RAG for Fiction Projects},
  author={Bicheno, Kieran},
  year={2025},
  url={https://github.com/TheLustriVA/NaRAGtive}
}
```
