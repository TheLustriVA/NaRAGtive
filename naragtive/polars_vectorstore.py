#!/usr/bin/env python3
"""
Polars-based Vector Search for RPG Scenes
Replaces ChromaDB entirely. No dependencies beyond Polars + sentence-transformers
"""

import json
import polars as pl
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
from pathlib import Path


class PolarsVectorStore:
    """Simple, fast vector store using Polars + numpy"""
    
    def __init__(self, parquet_path: str = "./thunderchild_scenes.parquet"):
        self.parquet_path = Path(parquet_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = None
        self.embeddings_cache = None
    
    def load(self):
        """Load parquet into memory"""
        if self.parquet_path.exists():
            self.df = pl.read_parquet(self.parquet_path)
            # Pre-load embeddings as numpy array
            self.embeddings_cache = np.array(
                self.df['embedding'].to_list(), 
                dtype=np.float32
            )
            print(f"✅ Loaded {len(self.df)} documents from {self.parquet_path}")
        else:
            print(f"❌ {self.parquet_path} not found")
            return False
        return True
    
    def save_from_chromadb(self, chroma_client, scene_collection_name: str = "scenes"):
        """Migrate from ChromaDB to Polars"""
        import chromadb
        
        client = chromadb.PersistentClient(path='./thunderchild_db')
        collection = client.get_collection(scene_collection_name)
        
        # Get all data
        all_data = collection.get(limit=None)
        
        print(f"Extracting {len(all_data['ids'])} documents from ChromaDB...")
        
        # Compute embeddings
        embeddings = self.embedding_model.encode(
            all_data['documents'],
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build dataframe
        self.df = pl.DataFrame({
            'id': all_data['ids'],
            'text': all_data['documents'],
            'embedding': [emb.tolist() for emb in embeddings],
            'metadata': [json.dumps(m) for m in all_data['metadatas']],
        })
        
        # Cache embeddings
        self.embeddings_cache = np.array(embeddings, dtype=np.float32)
        
        # Save
        self.df.write_parquet(self.parquet_path)
        print(f"✅ Saved {len(self.df)} documents to {self.parquet_path}")
        print(f"   Parquet size: {self.parquet_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    def query(self, query_text: str, n_results: int = 10) -> Dict[str, Any]:
        """Semantic search"""
        if self.df is None:
            self.load()
        
        # Encode query
        query_emb = self.embedding_model.encode(query_text, convert_to_numpy=True).astype(np.float32)
        
        # Cosine similarity (fast!)
        norms = np.linalg.norm(self.embeddings_cache, axis=1, keepdims=True)
        normalized = self.embeddings_cache / norms
        query_norm = query_emb / np.linalg.norm(query_emb)
        similarities = normalized @ query_norm
        
        # Top N
        top_indices = np.argsort(similarities)[::-1][:n_results]
        distances = 1 - similarities[top_indices]
        
        # Extract results
        results_df = self.df[top_indices]
        
        return {
            'ids': results_df['id'].to_list(),
            'documents': results_df['text'].to_list(),
            'metadatas': [json.loads(m) for m in results_df['metadata'].to_list()],
            'distances': [[d] for d in distances.tolist()],
            'scores': (1 - distances).tolist(),
        }
    
    def stats(self):
        """Print store statistics"""
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
    """Format search results for display"""
    
    def __init__(self, store: PolarsVectorStore):
        self.store = store
    
    def format_results(self, results: Dict[str, Any], query: str) -> str:
        """Pretty print results"""
        output = []
        output.append("\n" + "=" * 80)
        output.append(f"SCENE SEARCH: '{query}'")
        output.append("=" * 80)
        
        if not results['ids']:
            output.append("\n❌ No results found\n")
            return "\n".join(output)
        
        for i, (scene_id, text, metadata, score) in enumerate(
            zip(
                results['ids'],
                results['documents'],
                results['metadatas'],
                results['scores']
            )
        ):
            output.append(f"\n[Result {i + 1}] Relevance: {score:.1%}")
            output.append("-" * 80)
            output.append(f"Scene ID:     {metadata.get('scene_id', 'UNKNOWN')}")
            output.append(f"Date:         {metadata.get('date_iso', 'UNKNOWN')}")
            output.append(f"Location:     {metadata.get('location', 'unknown')}")
            output.append(f"POV:          {metadata.get('pov_character', 'UNKNOWN')}")
            
            # Characters
            try:
                chars = json.loads(metadata.get('characters_present', '[]'))
                output.append(f"Characters:   {', '.join(chars) if chars else 'None'}")
            except ValueError as ve:
                print(f"Value error in Characters formatting: {ve}")
            
            # Text preview
            output.append("\n📖 Text:")
            preview = text[:400] if len(text) > 400 else text
            output.append(preview)
            if len(text) > 400:
                output.append("[... truncated]")
            output.append("-" * 80)
        
        output.append("\n" + "=" * 80 + "\n")
        return "\n".join(output)


def main():
    """Demo: Migrate from ChromaDB and query"""
    
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