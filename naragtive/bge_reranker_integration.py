#!/usr/bin/env python3
"""
BGE Reranker v2 M3 integration for Polars vector store
Direct in-process reranking with FP16 optimization
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer


class BGERerankerM3:
    """
    BGE Reranker v2 M3 - Optimized for multilingual reranking
    Direct reranker without FlagEmbedding dependency (pure transformers)
    Supports FP16 inference for 2-3x speedup
    """
    
    def __init__(self, device: str = "cuda", use_fp16: bool = True):
        """
        Initialize BGE reranker
        
        Args:
            device: "cuda" or "cpu"
            use_fp16: Use half-precision (float16) for speed. Requires CUDA.
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.model_name = 'BAAI/bge-reranker-v2-m3'
        
        # Use SentenceTransformer which handles cross-encoder internally
        # This is just the base - we'll use direct transformers for better control
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        print(f"📥 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        if use_fp16 and device == "cuda":
            self.model = self.model.half()
            print("✅ FP16 mode enabled (2-3x speedup)")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.torch = torch
        print(f"✅ {self.model_name} loaded on {device}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> Tuple[List[float], List[int]]:
        """
        Rerank documents against query
        
        Args:
            query: Search query
            documents: List of documents to rerank
            batch_size: Number of query-doc pairs to score at once
            normalize: Apply sigmoid to scores (maps to [0, 1])
        
        Returns:
            scores: List of relevance scores
            indices: Original indices sorted by score (descending)
        """
        all_scores = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            
            # Tokenize
            with self.torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                # Forward pass
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            
            all_scores.extend(scores.cpu().numpy().tolist())
        
        scores = np.array(all_scores)
        
        # Normalize if requested
        if normalize:
            # Sigmoid function: 1 / (1 + e^-x)
            scores = 1 / (1 + np.exp(-scores))
        
        # Sort by score descending
        indices = np.argsort(-scores)
        
        return scores, indices


class PolarsVectorStoreWithReranker:
    """Enhanced Polars vector store with built-in BGE reranking"""
    
    def __init__(self, parquet_path: str = "./thunderchild_scenes.parquet", use_reranker: bool = True):
        """
        Args:
            parquet_path: Path to parquet file
            use_reranker: Enable BGE reranking (add ~2GB VRAM)
        """
        from polars_vectorstore import PolarsVectorStore
        
        self.store = PolarsVectorStore(parquet_path)
        self.reranker = None
        self.use_reranker = use_reranker
        
        if use_reranker:
            try:
                self.reranker = BGERerankerM3(device="cuda", use_fp16=True)
            except Exception as e:
                print(f"⚠️  Reranker init failed: {e}")
                print("   Falling back to embedding-only search")
                self.use_reranker = False
    
    def query_and_rerank(
        self,
        query_text: str,
        initial_k: int = 50,
        final_k: int = 10,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Two-stage retrieval: embedding search → reranking
        
        Args:
            query_text: Search query
            initial_k: Initial embedding search results to rerank
            final_k: Final results after reranking
            normalize: Normalize reranker scores to [0, 1]
        
        Returns:
            Results dict with reranked documents and new scores
        """
        # Stage 1: Fast embedding search
        print(f"🔍 Stage 1: Embedding search (top {initial_k})...")
        results = self.store.query(query_text, n_results=initial_k)
        
        if not self.use_reranker or not self.reranker:
            # Return embedding search results
            return {
                'ids': results['ids'],
                'documents': results['documents'],
                'metadatas': results['metadatas'],
                'scores': results['scores'],
                'reranked': False,
                'rerank_method': 'none'
            }
        
        # Stage 2: Cross-encoder reranking
        print(f"📊 Stage 2: BGE reranking (top {final_k})...")
        rerank_scores, indices = self.reranker.rerank(
            query_text,
            results['documents'],
            normalize=normalize
        )
        
        # Keep top final_k
        top_indices = indices[:final_k]
        
        # Reorder all results
        reranked = {
            'ids': [results['ids'][i] for i in top_indices],
            'documents': [results['documents'][i] for i in top_indices],
            'metadatas': [results['metadatas'][i] for i in top_indices],
            'embedding_scores': [results['scores'][i] for i in top_indices],
            'rerank_scores': rerank_scores[top_indices].tolist(),
            'reranked': True,
            'rerank_method': 'bge-v2-m3',
            'initial_search_count': initial_k,
        }
        
        return reranked
    
    def get_reranker_stats(self) -> Dict[str, Any]:
        """Return reranker model statistics"""
        if not self.reranker:
            return {'status': 'disabled', 'reason': 'reranker not initialized'}
        
        return {
            'status': 'enabled',
            'model': self.reranker.model_name,
            'device': self.reranker.device,
            'fp16': self.reranker.use_fp16,
            'parameters': sum(p.numel() for p in self.reranker.model.parameters()),
            'vram_mb': sum(p.numel() * (2 if self.reranker.use_fp16 else 4) for p in self.reranker.model.parameters()) / 1024 / 1024,
        }


# Example usage
if __name__ == "__main__":
    import time
    
    # Initialize store with reranker
    store = PolarsVectorStoreWithReranker()
    store.store.load()
    
    # Show reranker stats
    print("\n" + "=" * 80)
    print("RERANKER STATISTICS")
    print("=" * 80)
    stats = store.get_reranker_stats()
    for key, value in stats.items():
        print(f"{key:.<40} {value}")
    
    # Compare: embedding-only vs embedding+reranking
    queries = [
        "Admiral command leadership",
        "tactical military decision",
        "character reunion emotional",
    ]
    
    print("\n" + "=" * 80)
    print("COMPARISON: Embedding vs Embedding+Reranking")
    print("=" * 80)
    
    for query in queries:
        print(f"\n📌 Query: '{query}'\n")
        
        # Embedding only
        start = time.time()
        embed_results = store.store.query(query, n_results=10)
        embed_time = time.time() - start
        
        print("Embedding-only (top 5):")
        for i, (scene_id, score) in enumerate(zip(embed_results['ids'][:5], embed_results['scores'][:5])):
            print(f"  {i+1}. {scene_id:.<30} {score:.1%}")
        print(f"  Time: {embed_time*1000:.1f}ms\n")
        
        # Embedding + reranking
        start = time.time()
        reranked_results = store.query_and_rerank(query, initial_k=20, final_k=5)
        rerank_time = time.time() - start
        
        print("Embedding + BGE Reranking (top 5):")
        for i, (scene_id, emb_score, rerank_score) in enumerate(
            zip(
                reranked_results['ids'][:5],
                reranked_results['embedding_scores'][:5],
                reranked_results['rerank_scores'][:5]
            )
        ):
            print(f"  {i+1}. {scene_id:.<25} Emb:{emb_score:.1%} → Rerank:{rerank_score:.1%}")
        print(f"  Time: {rerank_time*1000:.1f}ms\n")
        
        print("-" * 80)
