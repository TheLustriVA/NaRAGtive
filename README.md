# NaRAGtive

A Polars-based RAG for fiction projects using Python and a Reranker.

## Example Uses

Embedding search (fast, baseline)
`python scene_search.py search "Admiral command"`

With BGE reranking (accurate, slight latency)
`python scene_search.py search "Admiral command" --rerank`

Interactive with reranking (model cached in memory, instant queries)
`python scene_search.py interactive --rerank`

Stats with reranker info
`python scene_search.py stats --show-reranker`

Export for LLM context
`python scene_search.py export "query" -f llm-context -o context.md`
