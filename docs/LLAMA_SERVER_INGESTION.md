# Llama-Server Chat Export Ingestion

## Overview

The `LlamaServerIngester` provides a complete pipeline to convert chat exports from the **llama.cpp Web UI** (llama-server) into a searchable, embedded vector store compatible with NaRAGtive's `PolarsVectorStore`.

This enables semantic search across narrative scenarios, creative dialogue, and conversational exchanges exported from llama-server.

## Features

- **JSON Export Parsing**: Validates and extracts structured data from llama-server exports
- **Message Grouping**: Automatically pairs user/assistant messages into coherent exchanges
- **Heuristic Analysis**: Extracts themes, tone, engagement level, and complexity from dialogue
- **Embedding Generation**: Uses SentenceTransformer for semantic search capabilities
- **Batch Processing**: Ingest multiple exports and combine into single searchable store
- **Chain-of-Thought Support**: Preserves and indexes assistant thinking content

## Installation

No additional dependencies required beyond NaRAGtive core packages:

```bash
pip install sentence-transformers polars
```

## Export Format

Llama-server exports must follow this JSON structure:

```json
{
  "conv": {
    "id": "conversation-uuid",
    "name": "Describe a scene from...",
    "lastModified": 1765279699684,
    "currNode": "node-id"
  },
  "messages": [
    {
      "convId": "conversation-uuid",
      "id": "message-uuid",
      "role": "user" | "assistant",
      "content": "The message text",
      "type": "text",
      "timestamp": 1765275434078,
      "thinking": "",
      "children": ["child-message-id"],
      "parent": "parent-message-id",
      "model": "model-name",
      "timings": { /* optional timing data */ }
    },
    ...
  ]
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `conv.id` | Unique conversation identifier (UUID) |
| `conv.name` | Conversation title/prompt (can be very long) |
| `conv.lastModified` | Unix timestamp in milliseconds |
| `role` | Either `"user"` or `"assistant"` |
| `type` | Message type (usually `"text"`) |
| `timestamp` | Unix millisecond timestamp |
| `thinking` | Chain-of-thought content (for supported models) |
| `model` | Model name (assistant messages only) |
| `children` | List of child message IDs (for tree structure) |

## Quick Start

### Single Export

```python
from naragtive.ingest_llama_server_chat import LlamaServerIngester
from naragtive.polars_vectorstore import PolarsVectorStore

# Ingest export
ingester = LlamaServerIngester()
df = ingester.ingest_llama_server_export(
    "path/to/export.json",
    output_parquet="./llama_scenes.parquet"
)

print(f"✅ Ingested {len(df)} scenes")

# Create vector store
store = PolarsVectorStore("./llama_scenes.parquet")
store.load()

# Search
results = store.query("Admiral commanding battle", n_results=5)
for scene_id, text, score in zip(
    results["ids"],
    results["documents"],
    results["scores"]
):
    print(f"{scene_id}: {score:.1%}")
    print(f"  {text[:100]}...\n")
```

### Multiple Exports

```python
ingester = LlamaServerIngester()
df = ingester.ingest_multiple_exports([
    "chat_1.json",
    "chat_2.json",
    "chat_3.json",
], output_parquet="./combined_scenes.parquet")

print(f"✅ Combined {len(df)} unique scenes")
```

## DataFrame Output

The ingester produces a Polars DataFrame with this schema:

```python
pl.DataFrame({
    "id": list[str],           # Unique scene ID
    "text": list[str],         # Formatted dialogue
    "embedding": list[list[float]],  # 384-dim vectors
    "metadata": list[str],     # JSON metadata strings
})
```

### Metadata Structure

Each row's metadata contains:

```json
{
  "scene_id": "scene_62b43483_0000_2025-11-10",
  "conversation_id": "62b43483-fb41-49b4-b769-582f8853cb64",
  "conversation_name": "Describe a scene from...",
  "date_iso": "2025-11-10",
  "timestamp": 1765275434078,
  "model": "TheDrummer_Magidonia-24B-v4.2.0-Q6_K.gguf",
  "has_thinking": false,
  "thinking_preview": "",
  "themes": ["creative", "technical"],
  "tone": "formal",
  "engagement_level": 0.75,
  "complexity": 0.68,
  "exchange_index": 0,
  "source_file": "export.json"
}
```

## Heuristic Analysis Details

### Theme Extraction

Themes are detected based on keyword presence:

- **creative**: story, describe, fiction, narrative, scene, dialogue
- **technical**: code, python, algorithm, implement, debug
- **analytical**: analyze, research, explain, compare, framework
- **conversational**: chat, discuss, think, feel, opinion
- **instructional**: teach, learn, guide, tutorial, help

### Tone Analysis

Dominant tone is determined by keyword scoring:

- **formal**: professional language, academic terms
- **casual**: conversational markers, contractions
- **technical**: technical vocabulary
- **creative**: vivid descriptive language
- **neutral**: default when no clear tone

### Engagement Level (0.0-1.0)

Based on:
- Text length (longer = higher engagement)
- Punctuation (? and ! indicate engagement)
- Dialogue markers (quoted speech)

### Complexity (0.0-1.0)

Based on:
- Average word length
- Sentence structure
- Vocabulary diversity

## Class Reference

### LlamaServerParser

```python
parser = LlamaServerParser()

# Parse and validate JSON export
data = parser.parse_export("export.json")
# Returns: {"conv": {...}, "messages": [...]}

# Clean conversation name
name = parser.extract_conversation_name(long_name)

# Convert timestamp to datetime
dt = LlamaServerParser.timestamp_to_datetime(1765275434078)
```

### LlamaServerExchangeGrouper

```python
grouper = LlamaServerExchangeGrouper()

# Group messages into user/assistant pairs
exchanges = grouper.group_into_exchanges(messages)

# Create scene from exchange
scene = grouper.create_scene_from_exchange(
    exchange,
    conversation_id,
    conversation_name
)
```

### LlamaServerHeuristicAnalyzer

```python
analyzer = LlamaServerHeuristicAnalyzer()

# Extract themes
themes = analyzer.extract_themes(text)
# Returns: ["creative", "technical", ...]

# Analyze tone
tone = analyzer.analyze_tone(text)
# Returns: "formal" | "casual" | "technical" | "creative" | "neutral"

# Calculate engagement (0.0-1.0)
engagement = analyzer.analyze_engagement_level(text)

# Calculate complexity (0.0-1.0)
complexity = analyzer.analyze_complexity(text)
```

### LlamaServerIngester

```python
ingester = LlamaServerIngester(embedding_model="all-MiniLM-L6-v2")

# Ingest single export
df = ingester.ingest_llama_server_export(
    file_path="export.json",
    output_parquet="./scenes.parquet"
)
# Returns: Polars DataFrame

# Ingest multiple exports
df = ingester.ingest_multiple_exports(
    file_paths=["export1.json", "export2.json"],
    output_parquet="./combined.parquet"
)
```

## Integration with PolarsVectorStore

The ingester outputs are directly compatible with NaRAGtive's vector store:

```python
from naragtive.ingest_llama_server_chat import LlamaServerIngester
from naragtive.polars_vectorstore import PolarsVectorStore, SceneQueryFormatter

# Ingest
ingester = LlamaServerIngester()
df = ingester.ingest_llama_server_export("export.json", "scenes.parquet")

# Load store
store = PolarsVectorStore("scenes.parquet")
store.load()
store.stats()

# Query
results = store.query("Admiral command", n_results=10)

# Format results
formatter = SceneQueryFormatter(store)
print(formatter.format_results(results, "Admiral command"))
```

## Advanced Usage

### Custom Embedding Model

```python
from naragtive.ingest_llama_server_chat import LlamaServerIngester

ingester = LlamaServerIngester(
    embedding_model="sentence-transformers/mpnet-base-v2"  # 768-dim
)
```

### Processing Large Exports

For exports with 100+ exchanges:

```python
ingester = LlamaServerIngester()

# Process and save
df = ingester.ingest_llama_server_export(
    "large_export.json",
    output_parquet="./large_scenes.parquet"
)

# Store stats
store = PolarsVectorStore("./large_scenes.parquet")
store.load()
store.stats()
```

### Merging with Existing Vector Store

```python
from naragtive.polars_vectorstore import PolarsVectorStore
import polars as pl

# Load existing
existing_df = pl.read_parquet("existing_scenes.parquet")

# Ingest new
ingester = LlamaServerIngester()
new_df = ingester.ingest_llama_server_export("new_export.json")

# Merge (remove duplicates by ID)
combined = pl.concat([existing_df, new_df]).unique(subset=["id"], keep="first")
combined.write_parquet("merged_scenes.parquet")

print(f"✅ Merged: {len(combined)} total scenes")
```

## Examples

### Example 1: Sci-Fi Narrative Ingestion

```python
ingester = LlamaServerIngester()

# Ingest battle scene descriptions
df = ingester.ingest_llama_server_export(
    "scifi_battle_scenes.json",
    output_parquet="./scifi_scenes.parquet"
)

store = PolarsVectorStore("./scifi_scenes.parquet")
store.load()

# Find dramatic moments
results = store.query("Admiral sacrifice defiant charge", n_results=3)

for metadata in results["metadatas"]:
    print(f"Themes: {metadata['themes']}")
    print(f"Engagement: {metadata['engagement_level']:.1%}")
    print(f"Complexity: {metadata['complexity']:.1%}")
```

### Example 2: Multi-Conversation Analysis

```python
import json
from pathlib import Path

ingester = LlamaServerIngester()

# Ingest all JSON files in directory
exports = list(Path("./exports").glob("*.json"))
df = ingester.ingest_multiple_exports(
    [str(f) for f in exports],
    output_parquet="./all_scenes.parquet"
)

# Analyze by conversation
for conv_id in df["metadata"].to_list():
    metadata = json.loads(conv_id)
    print(f"{metadata['conversation_name']}: {metadata['tone']}")
```

## Troubleshooting

### "Invalid llama-server export: missing 'conv' or 'messages' keys"

The JSON file doesn't match llama-server format. Verify:
- File has `"conv"` object with `id`, `name`, `lastModified`
- File has `"messages"` array

### "No new unique entries to merge"

All scenes in the new export already exist in the vector store (duplicate IDs). This is expected when re-ingesting the same export.

### Low engagement/complexity scores

Scores are normalized to 0.0-1.0 range based on heuristics. Use threshold filtering if needed:

```python
import polars as pl
import json

df = pl.read_parquet("scenes.parquet")

# Filter high-complexity scenes
high_complexity = df.filter(
    pl.col("metadata").map_elements(
        lambda x: json.loads(x)["complexity"] > 0.7
    )
)
```

## Performance Notes

- **Embedding generation**: ~10-20ms per scene (depends on model and CPU/GPU)
- **Typical export**: 100 scenes → ~1-2 seconds ingestion
- **Large export**: 1000 scenes → ~15-30 seconds
- **Parquet file size**: ~1MB per 50 scenes (embeddings + metadata)

## Contributing

To extend the ingester:

1. Add new heuristics to `LlamaServerHeuristicAnalyzer`
2. Modify `create_scene_from_exchange()` for different scene formatting
3. Add support for additional message types in `LlamaServerExchangeGrouper`

See `tests/test_ingest_llama_server_chat.py` for test patterns.

## License

Same as NaRAGtive main project.
