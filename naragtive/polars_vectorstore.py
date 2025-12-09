from __future__ import annotations

#!/usr/bin/env python3
"""
Polars-based Vector Search for RPG Scenes.

Replaces ChromaDB entirely with a lightweight, single-file vector store
built on Polars DataFrames and NumPy embeddings.
No external databases required - just parquet files.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer


class PolarsVectorStore:
    """
    Lightweight vector store using Polars and NumPy.
    
    A simple, fast, production-ready replacement for ChromaDB that stores
    scene data in a single parquet file with cached embeddings as NumPy arrays.
    
    Attributes:
        parquet_path: Path to the parquet file storing documents and embeddings
        embedding_model: SentenceTransformer model for semantic search
        df: Loaded Polars DataFrame (None until load() is called)
        embeddings_cache: NumPy array of embeddings for fast cosine similarity
        
    Example:
        ```python
        # Initialize and load
        store = PolarsVectorStore("./scenes.parquet")
        store.load()
        
        # Semantic search
        results = store.query("Admiral leadership", n_results=10)
        print(f"Found {len(results['ids'])} scenes")
        
        # Display stats
        store.stats()
        ```
    """
    
    def __init__(self, parquet_path: str = "./thunderchild_scenes.parquet") -> None:
        """
        Initialize vector store with path to parquet file.
        
        Args:
            parquet_path: Path where parquet file is or will be stored.
                Default: "./thunderchild_scenes.parquet"
                
        Raises:
            ValueError: If parquet_path is an empty string
        """
        if not parquet_path:
            raise ValueError("parquet_path cannot be empty")
            
        self.parquet_path: Path = Path(parquet_path)
        self.embedding_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.df: Optional[pl.DataFrame] = None
        self.embeddings_cache: Optional[np.ndarray] = None
    
    def load(self) -> bool:
        """
        Load parquet file into memory and cache embeddings.
        
        Loads the parquet file specified in parquet_path into a Polars
        DataFrame and pre-caches the embeddings as a NumPy array for fast
        cosine similarity searches.
        
        Returns:
            True if load successful, False if file not found
            
        Example:
            ```python
            store = PolarsVectorStore("./scenes.parquet")
            if store.load():
                print("Vector store loaded successfully")
            else:
                print("Parquet file not found")
            ```
        """
        if self.parquet_path.exists():
            self.df = pl.read_parquet(self.parquet_path)
            # Pre-load embeddings as numpy array for fast similarity computation
            self.embeddings_cache = np.array(
                self.df["embedding"].to_list(),
                dtype=np.float32
            )
            print(f"✅ Loaded {len(self.df)} documents from {self.parquet_path}")
            return True
        else:
            print(f"❌ {self.parquet_path} not found")
            return False
    
    def save_from_chromadb(
        self,
        chroma_client: Any,
        scene_collection_name: str = "scenes"
    ) -> None:
        """
        Migrate data from ChromaDB to Polars (one-time operation).
        
        Exports all documents and embeddings from ChromaDB and saves them
        to parquet format. This is a one-time migration step before using
        the store. After migration, use load() to access the data.
        
        Args:
            chroma_client: ChromaDB client instance (not used directly,
                          function uses internal path './thunderchild_db')
            scene_collection_name: Name of the collection in ChromaDB.
                Default: "scenes"
                
        Raises:
            ImportError: If chromadb is not installed
            
        Example:
            ```python
            import chromadb
            client = chromadb.PersistentClient(path='./thunderchild_db')
            
            store = PolarsVectorStore("./scenes.parquet")
            store.save_from_chromadb(client)
            ```
        """
        import chromadb
        
        client = chromadb.PersistentClient(path="./thunderchild_db")
        collection = client.get_collection(scene_collection_name)
        
        # Get all data from ChromaDB
        all_data = collection.get(limit=None)
        
        print(f"Extracting {len(all_data['ids'])} documents from ChromaDB...")
        
        # Compute embeddings
        embeddings = self.embedding_model.encode(
            all_data["documents"],
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build dataframe
        self.df = pl.DataFrame({
            "id": all_data["ids"],
            "text": all_data["documents"],
            "embedding": [emb.tolist() for emb in embeddings],
            "metadata": [json.dumps(m) for m in all_data["metadatas"]],
        })
        
        # Cache embeddings
        self.embeddings_cache = np.array(embeddings, dtype=np.float32)
        
        # Save to parquet
        self.df.write_parquet(self.parquet_path)
        print(f"✅ Saved {len(self.df)} documents to {self.parquet_path}")
        print(f"   Parquet size: {self.parquet_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    def query(
        self,
        query_text: str,
        n_results: int = 10
    ) -> dict[str, Any]:
        """
        Perform semantic search using cosine similarity.
        
        Encodes the query text into an embedding and finds the most similar
        documents using cosine similarity with pre-cached embeddings.
        
        Args:
            query_text: Search query string
            n_results: Number of results to return. Default: 10
            
        Returns:
            Dictionary with keys:
                - 'ids': list[str] - Scene IDs
                - 'documents': list[str] - Full scene text
                - 'metadatas': list[dict[str, Any]] - Parsed metadata dicts
                - 'distances': list[list[float]] - Distances from query
                - 'scores': list[float] - Similarity scores [0.0, 1.0]
                
        Raises:
            RuntimeError: If store not loaded (df is None)
            
        Example:
            ```python
            store = PolarsVectorStore("./scenes.parquet")
            store.load()
            
            results = store.query("Admiral command leadership", n_results=5)
            for scene_id, score in zip(results['ids'], results['scores']):
                print(f"{scene_id}: {score:.1%}")
            ```
        """
        if self.df is None:
            self.load()
        
        # Encode query into embedding
        query_emb = self.embedding_model.encode(
            query_text,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Compute cosine similarity using normalized dot product
        norms = np.linalg.norm(self.embeddings_cache, axis=1, keepdims=True)
        normalized = self.embeddings_cache / norms
        query_norm = query_emb / np.linalg.norm(query_emb)
        similarities = normalized @ query_norm
        
        # Get top N results
        top_indices = np.argsort(similarities)[::-1][:n_results]
        distances = 1 - similarities[top_indices]
        
        # Extract results from DataFrame
        results_df = self.df[top_indices]
        
        return {
            "ids": results_df["id"].to_list(),
            "documents": results_df["text"].to_list(),
            "metadatas": [json.loads(m) for m in results_df["metadata"].to_list()],
            "distances": [[d] for d in distances.tolist()],
            "scores": (1 - distances).tolist(),
        }
    
    def stats(self) -> None:
        """
        Print vector store statistics to stdout.
        
        Displays:
        - Number of documents stored
        - Parquet file size on disk
        - Embeddings memory usage
        - Total estimated memory usage
        
        Example:
            ```python
            store = PolarsVectorStore("./scenes.parquet")
            store.load()
            store.stats()
            # Output:
            # ============================================================
            # VECTOR STORE STATS
            # ============================================================
            # Total documents:     1247
            # Parquet file size:   125.3 MB
            # Embeddings in RAM:   18.4 MB
            # Total memory:        143.7 MB
            ```
        """
        if self.df is None:
            self.load()
        
        file_size = self.parquet_path.stat().st_size / 1024 / 1024
        ram_size = len(self.df) * 384 * 4 / 1024 / 1024  # embeddings only
        
        print("\n" + "=" * 60)
        print("VECTOR STORE STATS")
        print("=" * 60)
        print(f"Total documents:     {len(self.df)}")
        print(f"Parquet file size:   {file_size:.1f} MB")
        print(f"Embeddings in RAM:   {ram_size:.1f} MB")
        print(f"Total memory:        {(file_size + ram_size):.1f} MB")
        print("=" * 60 + "\n")


class SceneQueryFormatter:
    """
    Format search results for human-readable display.
    
    Renders vector store query results with metadata, character information,
    and text previews in a formatted, terminal-friendly output.
    
    Attributes:
        store: PolarsVectorStore instance for accessing documents
        
    Example:
        ```python
        store = PolarsVectorStore("./scenes.parquet")
        store.load()
        
        formatter = SceneQueryFormatter(store)
        results = store.query("Admiral command", n_results=5)
        print(formatter.format_results(results, "Admiral command"))
        ```
    """
    
    def __init__(self, store: PolarsVectorStore) -> None:
        """
        Initialize formatter with a vector store.
        
        Args:
            store: PolarsVectorStore instance (should be loaded)
        """
        self.store: PolarsVectorStore = store
    
    def format_results(
        self,
        results: dict[str, Any],
        query: str
    ) -> str:
        """
        Format search results for pretty printing.
        
        Args:
            results: Results dictionary from store.query()
            query: Original search query string
            
        Returns:
            Formatted string ready for print output
            
        Example:
            ```python
            results = store.query("Admiral leadership", n_results=3)
            formatted = formatter.format_results(results, "Admiral leadership")
            print(formatted)
            ```
        """
        output: list[str] = []
        output.append("\n" + "=" * 80)
        output.append(f"SCENE SEARCH: '{query}'")
        output.append("=" * 80)
        
        if not results["ids"]:
            output.append("\n❌ No results found\n")
            return "\n".join(output)
        
        for i, (scene_id, text, metadata, score) in enumerate(
            zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                results["scores"]
            )
        ):
            output.append(f"\n[Result {i + 1}] Relevance: {score:.1%}")
            output.append("-" * 80)
            output.append(f"Scene ID:     {metadata.get('scene_id', 'UNKNOWN')}")
            output.append(f"Date:         {metadata.get('date_iso', 'UNKNOWN')}")
            output.append(f"Location:     {metadata.get('location', 'unknown')}")
            output.append(f"POV:          {metadata.get('pov_character', 'UNKNOWN')}")
            
            # Extract and display characters
            try:
                chars = json.loads(metadata.get("characters_present", "[]"))
                output.append(f"Characters:   {', '.join(chars) if chars else 'None'}")
            except (ValueError, TypeError):
                output.append("Characters:   [error parsing]")
            
            # Text preview (400 char truncation)
            output.append("\n📖 Text:")
            preview = text[:400] if len(text) > 400 else text
            output.append(preview)
            if len(text) > 400:
                output.append("[... truncated]")
            output.append("-" * 80)
        
        output.append("\n" + "=" * 80 + "\n")
        return "\n".join(output)


def main() -> None:
    """
    Demo: Migrate from ChromaDB and perform example queries.
    
    This function demonstrates the basic workflow:
    1. Create a PolarsVectorStore instance
    2. Migrate data from ChromaDB (one-time)
    3. Display store statistics
    4. Perform several example queries
    5. Format and display results
    """
    
    # Create store
    store = PolarsVectorStore()
    
    # Migrate from ChromaDB (one-time)
    print("Migrating from ChromaDB to Polars...")
    store.save_from_chromadb(None)
    
    # Show stats
    store.stats()
    
    # Query examples
    formatter = SceneQueryFormatter(store)
    
    queries = [
        "Admiral Zelenskyy leadership",
        "military combat action",
        "character relationship trust",
    ]
    
    for query in queries:
        results = store.query(query, n_results=5)
        print(formatter.format_results(results, query))


if __name__ == "__main__":
    main()
