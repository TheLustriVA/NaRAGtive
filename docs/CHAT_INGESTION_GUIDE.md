# Adding Chat Transcripts to Your Polars Vector Store

## Overview

You have **two separate collections**:
1. **Scene transcripts** (your RPG narrative) - already migrated from ChromaDB
2. **Chat transcripts** (Discord, Slack, conversations) - new, add as needed

Both are separate `.parquet` files that can be queried independently or merged.

## Quick Start: Ingest Chat Data

### Method 1: JSON Format (Discord/Slack Export)

**Expected format:**
```json
[
    {
        "timestamp": "2025-12-05T10:30:00",
        "user": "username",
        "message": "text here",
        "channel": "channel_name"
    }
]
```

**Ingest:**
```python
from ingest_chat_transcripts import ChatTranscriptIngester

ingester = ChatTranscriptIngester()
df = ingester.ingest_json_messages(
    "discord_export.json",
    parquet_output="./chat_transcripts.parquet"
)
```

### Method 2: Plain Text File

```python
ingester = ChatTranscriptIngester()
df = ingester.ingest_txt_file(
    "transcript.txt",
    chunk_size=500,  # characters per chunk
    parquet_output="./chat_transcripts.parquet"
)
```

### Method 3: Merge With Existing Data

```python
# New data
new_df = ingester.ingest_json_messages("new_chat.json", parquet_output="temp.parquet")

# Merge with existing (automatically handles duplicates)
merged = ingester.merge_with_existing(new_df, existing_parquet="./chat_transcripts.parquet")
```

## File Structure After Setup

```
your_project/
├── polars_vectorstore.py          # Core vector store
├── bge_reranker_integration.py     # BGE reranking
├── reranker_export.py             # Export functions
├── scene_search.py                # CLI tool
├── ingest_chat_transcripts.py      # Chat ingestion ← NEW
├── thunderchild_scenes.parquet     # Scene data (443 docs)
├── chat_transcripts.parquet        # Chat data ← NEW
└── README.md
```

## What Gets Stored

Each message in the vector store has:

**Text:** Full message content

**Embedding:** 384-dimensional vector (all-MiniLM-L6-v2)

**Metadata (JSON):**
```json
{
    "timestamp": "2025-12-05T10:30:00",
    "user": "Kieran",
    "channel": "thunderchild",
    "word_count": 42,
    "character_count": 245,
    "ingestion_date": "2025-12-05T11:00:00"
}
```

## Search Both Collections

You can query either collection independently:

```bash
# Search scenes only
python scene_search.py search "Admiral command"

# To query chat transcripts, modify scene_search.py:
# Change: PolarsVectorStore("./thunderchild_scenes.parquet")
# To:     PolarsVectorStore("./chat_transcripts.parquet")
```

## Or Create a Multi-Collection Search

```python
from polars_vectorstore import PolarsVectorStore

# Load both
scenes = PolarsVectorStore("./thunderchild_scenes.parquet")
chat = PolarsVectorStore("./chat_transcripts.parquet")

# Search both
scene_results = scenes.query("Admiral command", n_results=5)
chat_results = chat.query("Admiral command", n_results=5)

# Combine and rerank
combined = {
    'ids': scene_results['ids'] + chat_results['ids'],
    'documents': scene_results['documents'] + chat_results['documents'],
    'scores': scene_results['scores'] + chat_results['scores'],
}
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Ingest 1000 messages | ~30 seconds | Includes embedding generation |
| Merge with existing | ~5 seconds | Automatic duplicate detection |
| Query single collection | 50-150ms | Embedding search |
| Query with reranking | 200-300ms | Two-stage retrieval |

## Typical Workflow

```bash
# 1. Export chat data from Discord/Slack as JSON

# 2. Ingest
python -c "
from ingest_chat_transcripts import ChatTranscriptIngester
ingester = ChatTranscriptIngester()
ingester.ingest_json_messages('export.json', parquet_output='./chat_transcripts.parquet')
"

# 3. Verify
python scene_search.py stats --show-reranker

# 4. Search
python scene_search.py search "query text" --rerank

# 5. Export results
python scene_search.py export "query text" -f llm-context -o context.md
```

## Troubleshooting

**"File not found"**
- Ensure you're in the right directory
- Check file paths are correct

**"Out of memory during embedding"**
- Reduce batch_size in ingest_chat_transcripts.py
- Process in smaller chunks

**"Duplicate ID error"**
- Use `merge_with_existing()` instead of direct `write_parquet()`
- It automatically handles duplicates

**"Embedding dimension mismatch"**
- Make sure all `.parquet` files use same embedding model
- Default: `all-MiniLM-L6-v2` (384 dimensions)

## Next Steps

1. ✅ Set up Polars vector store (done)
2. ✅ Add BGE reranking (done)
3. ✅ Create CLI tools (done)
4. ⏳ **Ingest chat transcripts** (this step)
5. ⏳ Feed results to LLM for agentic processing

## Files Reference

- **ingest_chat_transcripts.py** - Chat ingestion script
- **polars_vectorstore.py** - Core vector store (unchanged)
- **bge_reranker_integration.py** - BGE reranker (unchanged)
- **scene_search.py** - CLI with reranking support (unchanged)
