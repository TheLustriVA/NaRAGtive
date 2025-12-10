#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Any, Optional

"""
BGE Reranker v2 M3 integration for Polars vector store.

Provides direct in-process reranking with FP16 optimization for two-stage
retrieval-augmented generation (RAG) workflows.

The BGE (BAAI General Embeddings) reranker v2 M3 is a cross-encoder that
re-ranks embedding search results for improved accuracy.
"""


class BGERerankerM3:
    """
    BGE Reranker v2 M3 cross-encoder for multilingual reranking.
    
    A direct cross-encoder implementation using transformers for accurate
    reranking of document candidates. Supports FP16 inference for 2-3x speedup
    on CUDA-enabled GPUs.
    
    Attributes:
        device: "cuda" or "cpu" device to load model on
        use_fp16: Whether to use half-precision float16 (requires CUDA)
        model_name: HuggingFace model identifier
        tokenizer: Model tokenizer for encoding text
        model: Transformers model for scoring document pairs
        torch: PyTorch module reference
        
    Example:
        ```python
        # Initialize reranker
        reranker = BGERerankerM3(device="cuda", use_fp16=True)
        
        # Rerank documents
        scores, indices = reranker.rerank(
            query="Admiral command",
            documents=["scene text 1", "scene text 2", ...],
            normalize=True
        )
        ```
    """
    
    def __init__(
        self,
        device: str = "cuda",
        use_fp16: bool = True
    ) -> None:
        """
        Initialize BGE reranker with specified configuration.
        
        Args:
            device: Device to load model on. Default: "cuda"
                Choices: "cuda" (GPU), "cpu"
            use_fp16: Use half-precision (float16) for 2-3x speedup.
                Default: True. Requires CUDA-capable GPU.
                
        Raises:
            ImportError: If transformers or torch not installed
            RuntimeError: If device not available
            
        Example:
            ```python
            # GPU with FP16 optimization
            reranker = BGERerankerM3(device="cuda", use_fp16=True)
            
            # CPU fallback
            reranker = BGERerankerM3(device="cpu", use_fp16=False)
            ```
        """
        self.device: str = device
        self.use_fp16: bool = use_fp16
        self.model_name: str = "BAAI/bge-reranker-v2-m3"
        
        # Import transformers here to make it optional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        print(f"📥 Loading {self.model_name}...")
        self.tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Apply FP16 optimization if requested
        if use_fp16 and device == "cuda":
            self.model = self.model.half()
            print("✅ FP16 mode enabled (2-3x speedup)")
        
        # Move to device
        self.model = self.model.to(device)
        self.model.eval()
        
        self.torch: Any = torch
        print(f"✅ {self.model_name} loaded on {device}")
    
    def rerank(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Rerank documents against query using cross-encoder.
        
        Scores each query-document pair and returns relevance scores
        sorted in descending order. Processes in batches for memory efficiency.
        
        Args:
            query: Search query string
            documents: List of document strings to rerank
            batch_size: Number of query-doc pairs to process at once.
                Default: 32. Higher = faster but more memory.
            normalize: Apply sigmoid normalization to map scores to [0, 1].
                Default: True
                
        Returns:
            Tuple of (scores, indices):
                - scores: np.ndarray of shape (len(documents),)
                    Relevance scores, optionally normalized to [0, 1]
                - indices: np.ndarray of shape (len(documents),)
                    Indices sorted by score descending
                    
        Example:
            ```python
            reranker = BGERerankerM3()
            query = "Admiral leadership"
            docs = ["text 1", "text 2", "text 3", "text 4", "text 5"]
            
            scores, indices = reranker.rerank(query, docs, normalize=True)
            
            # Top 3 results
            for i in indices[:3]:
                print(f"Doc {i}: score {scores[i]:.1%}")
            ```
        """
        all_scores: list[float] = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            
            # Tokenize and score
            with self.torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                # Forward pass through cross-encoder
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            
            all_scores.extend(scores.cpu().numpy().tolist())
        
        scores = np.array(all_scores)
        
        # Normalize to [0, 1] range using sigmoid
        if normalize:
            # Sigmoid function: 1 / (1 + e^-x)
            scores = 1 / (1 + np.exp(-scores))
        
        # Sort by score descending
        indices = np.argsort(-scores)
        
        return scores, indices


class PolarsVectorStoreWithReranker:
    """
    Enhanced Polars vector store with integrated BGE cross-encoder reranking.
    
    Implements a two-stage retrieval pipeline:
    1. Fast embedding search (Stage 1) - get top-k candidates
    2. Accurate cross-encoder reranking (Stage 2) - return top-n re-ranked results
    
    This approach provides both speed (Stage 1) and accuracy (Stage 2),
    ideal for RAG and question-answering systems.
    
    Attributes:
        store: Underlying PolarsVectorStore for embedding search
        reranker: BGERerankerM3 instance (None if disabled)
        use_reranker: Whether reranking is available
        
    Example:
        ```python
        # Initialize with reranking
        store = PolarsVectorStoreWithReranker(use_reranker=True)
        store.store.load()
        
        # Two-stage retrieval
        results = store.query_and_rerank(
            "Admiral command",
            initial_k=50,  # Get 50 from embedding search
            final_k=10,    # Return top 10 after reranking
        )
        
        for i, (scene_id, score) in enumerate(zip(results['ids'], results['rerank_scores'])):
            print(f"{i+1}. {scene_id}: {score:.1%}")
        ```
    """
    
    def __init__(
        self,
        parquet_path: str = "./thunderchild_scenes.parquet",
        use_reranker: bool = True
    ) -> None:
        """
        Initialize vector store with optional reranking.
        
        Args:
            parquet_path: Path to parquet vector store.
                Default: "./thunderchild_scenes.parquet"
            use_reranker: Enable BGE reranking for improved accuracy.
                Default: True. Adds ~2.6GB VRAM (FP32) or ~1.3GB (FP16).
                
        Example:
            ```python
            # With reranking enabled
            store = PolarsVectorStoreWithReranker(use_reranker=True)
            
            # Fallback to embedding-only if reranker unavailable
            store = PolarsVectorStoreWithReranker(use_reranker=False)
            ```
        """
        from naragtive.polars_vectorstore import PolarsVectorStore
        
        self.store: PolarsVectorStore = PolarsVectorStore(parquet_path)
        self.reranker: Optional[BGERerankerM3] = None
        self.use_reranker: bool = use_reranker
        
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
    ) -> dict[str, Any]:
        """
        Two-stage retrieval: embedding search followed by reranking.
        
        Performs semantic search followed by cross-encoder reranking for
        improved accuracy. If reranking is disabled or unavailable, returns
        embedding search results only.
        
        Args:
            query_text: Search query string
            initial_k: Number of candidates from Stage 1 embedding search.
                Default: 50. Used as input to reranker.
            final_k: Number of final results after Stage 2 reranking.
                Default: 10. Must be <= initial_k.
            normalize: Normalize reranker scores to [0, 1] using sigmoid.
                Default: True
                
        Returns:
            Dictionary with keys:
                - 'ids': list[str] - Scene IDs (reranked)
                - 'documents': list[str] - Full scene text (reranked)
                - 'metadatas': list[dict] - Metadata dicts (reranked)
                - 'embedding_scores': list[float] - Original Stage 1 scores
                - 'rerank_scores': list[float] - Stage 2 reranker scores (if reranked)
                - 'reranked': bool - Whether reranking was applied
                - 'rerank_method': str - Name of reranking method (or 'none')
                - 'initial_search_count': int - Number of candidates reranked
                
        Example:
            ```python
            store = PolarsVectorStoreWithReranker()
            store.store.load()
            
            # Two-stage retrieval with defaults
            results = store.query_and_rerank("Admiral command")
            
            # Custom parameters
            results = store.query_and_rerank(
                "military combat action",
                initial_k=100,  # More candidates
                final_k=20,     # More results
            )
            
            # Check if reranking was applied
            if results['reranked']:
                for i, (id, score) in enumerate(zip(results['ids'][:5], results['rerank_scores'][:5])):
                    print(f"{i+1}. {id}: {score:.1%}")
            ```
        """
        # Stage 1: Fast embedding search
        print(f"🔍 Stage 1: Embedding search (top {initial_k})...")
        results = self.store.query(query_text, n_results=initial_k)
        
        if not self.use_reranker or not self.reranker:
            # Return embedding search results only
            return {
                "ids": results["ids"],
                "documents": results["documents"],
                "metadatas": results["metadatas"],
                "scores": results["scores"],
                "reranked": False,
                "rerank_method": "none"
            }
        
        # Stage 2: Cross-encoder reranking
        print(f"📈 Stage 2: BGE reranking (top {final_k})...")
        rerank_scores, indices = self.reranker.rerank(
            query_text,
            results["documents"],
            normalize=normalize
        )
        
        # Keep top final_k results
        top_indices = indices[:final_k]
        
        # Reorder all results
        reranked: dict[str, Any] = {
            "ids": [results["ids"][i] for i in top_indices],
            "documents": [results["documents"][i] for i in top_indices],
            "metadatas": [results["metadatas"][i] for i in top_indices],
            "embedding_scores": [results["scores"][i] for i in top_indices],
            "rerank_scores": rerank_scores[top_indices].tolist(),
            "reranked": True,
            "rerank_method": "bge-v2-m3",
            "initial_search_count": initial_k,
        }
        
        return reranked
    
    def get_reranker_stats(self) -> dict[str, Any]:
        """
        Get reranker model statistics and configuration.
        
        Returns information about the reranker model including device,
        precision mode, parameter count, and VRAM usage.
        
        Returns:
            Dictionary with keys:
                - 'status': str - "enabled" or "disabled"
                - 'model': str - Model name (if enabled)
                - 'device': str - Device (cuda/cpu) if enabled
                - 'fp16': bool - FP16 mode enabled if applicable
                - 'parameters': int - Number of parameters if enabled
                - 'vram_mb': float - Estimated VRAM in MB if enabled
                - 'reason': str - Reason for disabling if applicable
                
        Example:
            ```python
            store = PolarsVectorStoreWithReranker()
            stats = store.get_reranker_stats()
            
            for key, value in stats.items():
                print(f"{key:.<30} {value}")
            # Output:
            # status............................. enabled
            # model............................. BAAI/bge-reranker-v2-m3
            # device............................. cuda
            # fp16.............................. True
            # parameters........................ 278000000
            # vram_mb........................... 1342.0
            ```
        """
        if not self.reranker:
            return {
                "status": "disabled",
                "reason": "reranker not initialized"
            }
        
        return {
            "status": "enabled",
            "model": self.reranker.model_name,
            "device": self.reranker.device,
            "fp16": self.reranker.use_fp16,
            "parameters": sum(p.numel() for p in self.reranker.model.parameters()),
            "vram_mb": sum(
                p.numel() * (2 if self.reranker.use_fp16 else 4)
                for p in self.reranker.model.parameters()
            ) / 1024 / 1024,
        }


def main() -> None:
    """
    Demo: Compare embedding-only search vs two-stage retrieval with reranking.
    
    Initializes a reranker, loads the vector store, and demonstrates
    the quality difference between embedding-only and reranked results.
    """
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
        for i, (scene_id, score) in enumerate(zip(embed_results["ids"][:5], embed_results["scores"][:5])):
            print(f"  {i+1}. {scene_id:.<30} {score:.1%}")
        print(f"  Time: {embed_time*1000:.1f}ms\n")
        
        # Embedding + reranking
        start = time.time()
        reranked_results = store.query_and_rerank(query, initial_k=20, final_k=5)
        rerank_time = time.time() - start
        
        print("Embedding + BGE Reranking (top 5):")
        for i, (scene_id, emb_score, rerank_score) in enumerate(
            zip(
                reranked_results["ids"][:5],
                reranked_results["embedding_scores"][:5],
                reranked_results["rerank_scores"][:5]
            )
        ):
            print(f"  {i+1}. {scene_id:.<25} Emb:{emb_score:.1%} → Rerank:{rerank_score:.1%}")
        print(f"  Time: {rerank_time*1000:.1f}ms\n")
        
        print("-" * 80)


if __name__ == "__main__":
    main()
