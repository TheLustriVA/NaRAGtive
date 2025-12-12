#!/usr/bin/env python3
"""
Example: Ingest llama-server chat exports into searchable vector store.

Demonstrates:
1. Parsing and ingesting single llama-server JSON export
2. Extracting scenes from user/assistant exchanges
3. Analyzing themes, tone, engagement, complexity
4. Creating searchable vector store with embeddings
5. Querying and retrieving relevant scenes

Usage:
    python examples/ingest_llama_server_example.py <path_to_export.json>
"""

import json
import sys
from pathlib import Path

from naragtive.ingest_llama_server_chat import LlamaServerIngester
from naragtive.polars_vectorstore import PolarsVectorStore, SceneQueryFormatter


def main(export_path: str | None = None) -> None:
    """
    Main example workflow.
    
    Args:
        export_path: Path to llama-server JSON export (optional for demo)
    """
    print("\n" + "=" * 70)
    print("Llama-Server Chat Ingestion Example")
    print("=" * 70)

    # If no path provided, create demo data
    if not export_path:
        print("\n📝 Creating demo export data...")
        export_path = "demo_export.json"
        demo_data = {
            "conv": {
                "id": "demo-conv-001",
                "name": "Dramatic Space Battle Narrative",
                "lastModified": 1765279699684,
                "currNode": "final-scene",
            },
            "messages": [
                {
                    "id": "msg-1",
                    "convId": "demo-conv-001",
                    "role": "user",
                    "content": "Describe a pivotal moment where an Admiral makes a desperate sacrifice to save their fleet.",
                    "type": "text",
                    "timestamp": 1765275434078,
                    "thinking": "",
                    "children": ["msg-2"],
                    "parent": None,
                },
                {
                    "id": "msg-2",
                    "convId": "demo-conv-001",
                    "role": "assistant",
                    "content": (
                        "The Admiral's eyes narrowed as she studied the tactical display. "
                        "The enemy fleet was closing in, and there was only one way to save the evacuation shuttles. "
                        "With a deep breath, she plotted a direct intercept course, committing her ship and crew "
                        "to one final, defiant act. The order was given with quiet resolve: 'Charge.'"
                    ),
                    "type": "text",
                    "timestamp": 1765275434200,
                    "thinking": "This needs to capture the dramatic tension and sacrifice...",
                    "model": "llama-70b-instruct",
                    "children": ["msg-3"],
                    "parent": "msg-1",
                },
                {
                    "id": "msg-3",
                    "convId": "demo-conv-001",
                    "role": "user",
                    "content": "What happens next? How does the battle conclude?",
                    "type": "text",
                    "timestamp": 1765276000000,
                    "thinking": "",
                    "children": ["msg-4"],
                    "parent": "msg-2",
                },
                {
                    "id": "msg-4",
                    "convId": "demo-conv-001",
                    "role": "assistant",
                    "content": (
                        "The ship tore through the alien formation like a spear. "
                        "Weapons blazed with stunning intensity, each shot claiming enemy vessels. "
                        "But the damage was mounting. Hull breaches cascaded through sections as enemy fire "
                        "concentrated on the charging warship. In the final moments, the Admiral overloaded the reactors. "
                        "The resulting explosion created a magnetic pulse so intense it crumpled the enemy fleet like paper, "
                        "buying precious seconds for the last evacuees to escape. In the aftermath, "
                        "only a perfect sphere of crystallized energy remained, a monument to sacrifice."
                    ),
                    "type": "text",
                    "timestamp": 1765276100000,
                    "thinking": "Building to the climactic moment with appropriate pacing and impact...",
                    "model": "llama-70b-instruct",
                    "children": [],
                    "parent": "msg-3",
                },
            ],
        }

        with open(export_path, "w") as f:
            json.dump(demo_data, f, indent=2)
        print(f"✅ Demo data created: {export_path}")
    else:
        print(f"\n📖 Using export: {export_path}")

    # Step 1: Parse and validate
    print("\n" + "-" * 70)
    print("STEP 1: Parse and Validate Export")
    print("-" * 70)

    ingester = LlamaServerIngester()
    output_parquet = "demo_scenes.parquet"

    # Step 2: Ingest
    print("\n" + "-" * 70)
    print("STEP 2: Ingest and Create Scenes")
    print("-" * 70)

    df = ingester.ingest_llama_server_export(
        export_path,
        output_parquet=output_parquet,
    )

    # Step 3: Display DataFrame info
    print("\n" + "-" * 70)
    print("STEP 3: DataFrame Summary")
    print("-" * 70)
    print(f"\nTotal scenes: {len(df)}")
    print(f"Columns: {df.columns}")
    print(f"Parquet file: {output_parquet}")

    # Step 4: Show scene details
    print("\n" + "-" * 70)
    print("STEP 4: Scene Details")
    print("-" * 70)

    for i, row in enumerate(df.row_tuples(named=True), 1):
        print(f"\nScene {i}: {row['id']}")
        print(f"Text preview: {row['text'][:150]}...")

        # Parse and display metadata
        metadata = json.loads(row["metadata"])
        print(f"\n  Conversation: {metadata['conversation_name']}")
        print(f"  Model: {metadata['model']}")
        print(f"  Date: {metadata['date_iso']}")
        print(f"  Themes: {', '.join(metadata['themes'])}")
        print(f"  Tone: {metadata['tone']}")
        print(f"  Engagement: {metadata['engagement_level']:.1%}")
        print(f"  Complexity: {metadata['complexity']:.1%}")
        print(f"  Has Thinking: {metadata['has_thinking']}")

    # Step 5: Load vector store
    print("\n" + "-" * 70)
    print("STEP 5: Create Vector Store")
    print("-" * 70)

    store = PolarsVectorStore(output_parquet)
    if store.load():
        store.stats()

    # Step 6: Perform queries
    print("\n" + "-" * 70)
    print("STEP 6: Semantic Search Examples")
    print("-" * 70)

    queries = [
        "Admiral sacrifice and defiance",
        "dramatic naval battle",
        "escape and survival",
    ]

    formatter = SceneQueryFormatter(store)

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = store.query(query, n_results=3)
        print(formatter.format_results(results, query))

    # Step 7: Advanced analysis
    print("\n" + "-" * 70)
    print("STEP 7: Metadata Analysis")
    print("-" * 70)

    print("\nAll Themes Found:")
    all_themes = set()
    for metadata_str in df["metadata"].to_list():
        metadata = json.loads(metadata_str)
        all_themes.update(metadata["themes"])
    for theme in sorted(all_themes):
        print(f"  - {theme}")

    print("\nAll Tones Found:")
    all_tones = set()
    for metadata_str in df["metadata"].to_list():
        metadata = json.loads(metadata_str)
        all_tones.add(metadata["tone"])
    for tone in sorted(all_tones):
        print(f"  - {tone}")

    print("\nEngagement Level Statistics:")
    engagement_values = [
        json.loads(m)["engagement_level"] for m in df["metadata"].to_list()
    ]
    print(f"  Min: {min(engagement_values):.2f}")
    print(f"  Max: {max(engagement_values):.2f}")
    print(f"  Avg: {sum(engagement_values) / len(engagement_values):.2f}")

    print("\nComplexity Statistics:")
    complexity_values = [
        json.loads(m)["complexity"] for m in df["metadata"].to_list()
    ]
    print(f"  Min: {min(complexity_values):.2f}")
    print(f"  Max: {max(complexity_values):.2f}")
    print(f"  Avg: {sum(complexity_values) / len(complexity_values):.2f}")

    print("\n" + "=" * 70)
    print("✅ Example Complete!")
    print("=" * 70)
    print(f"\nParquet store saved to: {output_parquet}")
    print("\nNext steps:")
    print("1. Use PolarsVectorStore to query the scenes")
    print("2. Integrate with your RAG pipeline")
    print("3. Ingest additional exports with ingest_multiple_exports()")
    print("4. See docs/LLAMA_SERVER_INGESTION.md for more details\n")


if __name__ == "__main__":
    export_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(export_file)
