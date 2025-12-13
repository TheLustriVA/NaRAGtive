# NaRAGtive: Narrative RAG with Semantic Search & Reranking

A Polars-based retrieval-augmented generation system for fiction projects. Ingest narratives from multiple sources (Neptune AI, llama-server, generic chat), perform semantic search with optional BGE reranking, and extract narrative insights.

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

## Complete Workflow Examples

### Example 1: Neptune Narrative Analysis

```bash
# 1. Export conversation from Neptune
# File → Export → save_as thunderchild_chapter3.txt

# 2. Ingest scenes
python main.py ingest-neptune -e thunderchild_chapter3.txt -o thunderchild.parquet

# 3. Search for specific scenes
python main.py search "Admiral decision making" -s thunderchild.parquet --rerank

# 4. View all scenes on the bridge
python main.py list -s thunderchild.parquet --location bridge

# 5. Export for LLM context
python main.py export "command conflict" -s thunderchild.parquet -f llm-context -o context.md
```

### Example 2: Multi-Source Narrative

```bash
# 1. Ingest Neptune conversation
python main.py ingest-neptune -e neptune_ch1.txt -o combined.parquet

# 2. Add llama-server chat to same store
python main.py ingest-llama -e llama_ch2.json -o combined.parquet

# 3. Search across both sources
python main.py search "character reaction" -s combined.parquet --rerank

# 4. View statistics (shows both sources merged)
python main.py stats -s combined.parquet --show-reranker
```

### Example 3: Archive Chat History

```bash
# 1. Export Discord/Slack as JSON
# (Use Discord's export feature or Slack API)

# 2. Ingest generic chat
python main.py ingest-chat -s discord_archive.json -o discord.parquet

# 3. Interactive search
python main.py interactive -s discord.parquet
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

## Merging Narratives

By default, ingestions **append** to existing stores (with duplicate detection):

```bash
# First ingestion creates store
python main.py ingest-neptune -e export1.txt -o scenes.parquet

# Second ingestion merges
python main.py ingest-neptune -e export2.txt -o scenes.parquet
# (automatically deduplicates by scene ID)
```

To start fresh:

```bash
python main.py ingest-neptune -e export.txt -o scenes.parquet --no-append
# or delete the .parquet file first
```

---

## Metadata Extraction

After ingestion, each scene includes:

### From Neptune/llama-server
- `scene_id`: Unique identifier
- `date_iso`: Scene date
- `pov_character`: Point of view
- `location`: Where the scene happens
- `characters_present`: List of characters
- `tone`: "tense", "emotional", "neutral"
- `emotional_intensity`: 0.0–1.0 score
- `action_level`: 0.0–1.0 score

### From Generic Chat
- `timestamp`: Message time
- `user`: Speaker name
- `channel`: Discord channel / Slack workspace
- `word_count`: Message length
- `character_count`: Text length

---

## Performance Notes

### Memory Usage

- **Embedding generation**: ~300MB (first run, downloads model)
- **Vector store**: ~50MB per 1000 scenes
- **BGE reranker**: ~1.3GB VRAM (optional, only when using `--rerank`)

### Speed

- **Neptune ingestion**: ~10 scenes/sec (depends on file size)
- **Semantic search**: ~100ms (embedding only)
- **Reranked search**: ~500ms-1s (embedding + reranking)
- **Interactive mode**: ~100ms after first query (model cached)

### Scaling

- **Vector store**: Tested up to 100k scenes
- **Search latency**: Sub-second for most stores
- **Memory**: Linear with scene count

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

### Ingestion slow

**First run?** Model download (~200MB) happens automatically. Subsequent runs are faster.

**Large file?** Try reducing chunk size:
```bash
python main.py ingest-chat -s large.txt --chunk-size 250 -o output.parquet
```

### Merge errors

```
Duplicate IDs detected
```

**Solution:** Start fresh with `--no-append` or delete existing `.parquet`:
```bash
python main.py ingest-neptune -e export.txt -o scenes.parquet --no-append
```

### Vector store file corrupted

**Solution:** Delete the `.parquet` file and reingest:
```bash
rm scenes.parquet
python main.py ingest-neptune -e export.txt -o scenes.parquet
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

# Specific ingestion tests
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

### For Developers

1. Read ingester docstrings: `naragtive/ingest_chat_transcripts.py`
2. Study test cases: `tests/test_ingest_*.py`
3. Check vector store: `naragtive/polars_vectorstore.py`

### For Advanced Usage

1. Review `docs/LLAMA_SERVER_INGESTION.md` for llama-server specifics
2. Explore `naragtive/bge_reranker_integration.py` for reranking details
3. See `naragtive/reranker_export.py` for export formats

---

## Recent Updates

### v1.1.0 - Ingestion CLI Workflow

- ✅ New `ingest-neptune` command for Neptune AI exports
- ✅ New `ingest-llama` command for llama-server chats
- ✅ New `ingest-chat` command for generic transcripts
- ✅ New `ingest-help` command with comprehensive guides
- ✅ Improved error handling and user feedback
- ✅ Full test coverage for ingestion pipeline
- ✅ Enhanced README with workflows and examples

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
