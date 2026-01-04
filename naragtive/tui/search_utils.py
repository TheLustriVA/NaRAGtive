"""Utility functions for search operations in NaRAGtive TUI.

Provides async wrappers and formatting helpers for vector store queries,
reranking operations, and result display.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import numpy as np


class SearchError(Exception):
    """Raised when a search operation fails."""

    pass


async def async_search(
    store: Any,
    query: str,
    n_results: int = 20,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Perform async semantic search on vector store.

    Runs store.query() in executor to avoid blocking the event loop.

    Args:
        store: PolarsVectorStore instance (must be loaded)
        query: Search query string
        n_results: Number of results to return. Default: 20
        timeout: Maximum time to wait in seconds. Default: 30.0

    Returns:
        Dictionary with keys: 'ids', 'documents', 'scores', 'metadatas'

    Raises:
        SearchError: If store is not loaded or query fails
        asyncio.TimeoutError: If search exceeds timeout

    Example:
        ```python
        store = PolarsVectorStore("./scenes.parquet")
        store.load()
        results = await async_search(store, "Admiral leadership", n_results=10)
        ```
    """
    if not query or len(query) < 3:
        raise SearchError("Query must be at least 3 characters")

    loop = asyncio.get_event_loop()

    try:
        # Validate store is loaded
        if store.df is None:
            raise SearchError("Vector store not loaded")

        # Run search in executor with timeout
        results = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: store.query(query, n_results)),
            timeout=timeout,
        )

        return results
    except asyncio.TimeoutError as e:
        raise SearchError(
            f"Search timeout after {timeout}s. Try a more specific query."
        ) from e
    except AttributeError as e:
        raise SearchError(f"Store error: {e}") from e


async def async_rerank(
    reranker: Any,
    query: str,
    documents: list[str],
    top_k: int = 10,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Perform async reranking using BGE reranker.

    Runs reranker.rerank() in executor to avoid blocking the event loop.

    Args:
        reranker: BGERerankerM3 instance (must be initialized)
        query: Search query string
        documents: List of document strings to rerank
        top_k: Number of top results to return. Default: 10
        timeout: Maximum time to wait in seconds. Default: 60.0

    Returns:
        List of dicts with keys: 'index', 'score', 'text'
        Sorted by score descending

    Raises:
        SearchError: If reranking fails
        asyncio.TimeoutError: If reranking exceeds timeout

    Example:
        ```python
        reranker = BGERerankerM3(device="cuda")
        reranked = await async_rerank(
            reranker,
            "Admiral command",
            documents=["text 1", "text 2"],
            top_k=5
        )
        ```
    """
    if not documents:
        raise SearchError("No documents to rerank")

    loop = asyncio.get_event_loop()

    try:
        # Run reranking in executor
        scores, indices = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: reranker.rerank(query, documents, normalize=True)
            ),
            timeout=timeout,
        )

        # Get top_k indices
        top_indices = indices[:top_k]

        # Build result list
        results = [
            {
                "index": int(idx),
                "score": float(scores[idx]),
                "text": documents[int(idx)],
            }
            for idx in top_indices
        ]

        return results
    except asyncio.TimeoutError as e:
        raise SearchError(f"Reranking timeout after {timeout}s") from e
    except Exception as e:
        raise SearchError(f"Reranking failed: {e}") from e


def apply_filters(
    results: dict[str, Any],
    location: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    character: str | None = None,
) -> dict[str, Any]:
    """Filter search results using set intersection on metadata.

    Filters search results by location, date range, and character.
    Uses case-insensitive matching for location and character.
    Dates should be ISO format (YYYY-MM-DD).

    Args:
        results: Dictionary with keys: 'ids', 'documents', 'scores', 'metadatas'
        location: Filter by location (case-insensitive partial match). Default: None
        date_start: Filter by start date ISO format. Default: None
        date_end: Filter by end date ISO format. Default: None
        character: Filter by character presence (case-insensitive). Default: None

    Returns:
        Filtered results with same structure as input
        Empty lists if no results match

    Example:
        ```python
        results = await async_search(store, "Admiral leadership")
        filtered = apply_filters(
            results,
            location="Throne",
            date_start="2024-01-01",
            date_end="2024-12-31",
            character="Admiral"
        )
        # Returns only results matching all filters
        ```
    """
    if not results or not results.get("metadatas"):
        return {"ids": [], "documents": [], "scores": [], "metadatas": []}

    # Build filter indices
    matching_indices = set(range(len(results["metadatas"])))

    # Filter by location
    if location:
        location_lower = location.lower()
        location_indices = set()
        for i, metadata in enumerate(results["metadatas"]):
            loc = metadata.get("location", "").lower()
            if location_lower in loc:
                location_indices.add(i)
        matching_indices &= location_indices

    # Filter by date range
    if date_start or date_end:
        date_indices = set()
        for i, metadata in enumerate(results["metadatas"]):
            date_str = metadata.get("date_iso", "")
            try:
                if date_start and date_str < date_start:
                    continue
                if date_end and date_str > date_end:
                    continue
                date_indices.add(i)
            except (ValueError, TypeError):
                pass
        matching_indices &= date_indices

    # Filter by character
    if character:
        character_lower = character.lower()
        character_indices = set()
        for i, metadata in enumerate(results["metadatas"]):
            chars_str = metadata.get("characters_present", "[]")
            try:
                if isinstance(chars_str, str):
                    chars = json.loads(chars_str)
                else:
                    chars = chars_str
                if any(
                    character_lower in c.lower() for c in chars if isinstance(c, str)
                ):
                    character_indices.add(i)
            except (json.JSONDecodeError, TypeError):
                pass
        matching_indices &= character_indices

    # Build filtered results
    filtered = {"ids": [], "documents": [], "scores": [], "metadatas": []}
    for idx in sorted(matching_indices):
        filtered["ids"].append(results["ids"][idx])
        filtered["documents"].append(results["documents"][idx])
        filtered["scores"].append(results["scores"][idx])
        filtered["metadatas"].append(results["metadatas"][idx])

    return filtered


def format_relevance_score(score: float) -> str:
    """Format relevance score as percentage.

    Args:
        score: Score in range [0.0, 1.0]

    Returns:
        Formatted string like "94%"

    Example:
        ```python
        format_relevance_score(0.94)  # Returns "94%"
        format_relevance_score(0.871)  # Returns "87%"
        ```
    """
    # Clamp to [0, 1] range
    clamped = max(0.0, min(1.0, float(score)))
    percentage = int(clamped * 100)
    return f"{percentage}%"


def parse_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Parse and format metadata for display.

    Handles JSON string parsing for characters_present field.

    Args:
        metadata: Metadata dictionary from search results

    Returns:
        Formatted metadata with parsed values

    Example:
        ```python
        raw = {
            "scene_id": "scene-123",
            "date_iso": "2024-01-15",
            "location": "Throne Room",
            "pov_character": "Admiral",
            "characters_present": '["Admiral", "King"]'
        }
        parsed = parse_metadata(raw)
        # parsed["characters"] == ["Admiral", "King"]
        ```
    """
    formatted: dict[str, Any] = {}

    # Copy simple fields
    formatted["scene_id"] = metadata.get("scene_id", "UNKNOWN")
    formatted["date"] = metadata.get("date_iso", "UNKNOWN")
    formatted["location"] = metadata.get("location", "unknown")
    formatted["pov"] = metadata.get("pov_character", "UNKNOWN")

    # Parse characters_present JSON string
    try:
        chars_str = metadata.get("characters_present", "[]")
        if isinstance(chars_str, str):
            chars = json.loads(chars_str)
        else:
            chars = chars_str

        if isinstance(chars, list):
            formatted["characters"] = chars
        else:
            formatted["characters"] = []
    except (json.JSONDecodeError, TypeError):
        formatted["characters"] = []

    return formatted


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis if needed.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation. Default: 200

    Returns:
        Truncated text with "[...]" appended if needed

    Example:
        ```python
        long_text = "A" * 300
        truncate_text(long_text, max_length=50)
        # Returns: "A" * 47 + "[...]"
        ```
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 5] + "[...]"


def format_search_query(query: str) -> str:
    """Validate and format search query.

    Strips whitespace and checks minimum length.

    Args:
        query: Raw search query

    Returns:
        Formatted query string

    Raises:
        SearchError: If query is too short

    Example:
        ```python
        format_search_query("  Admiral leadership  ")
        # Returns: "Admiral leadership"
        ```
    """
    formatted = query.strip()
    if len(formatted) < 3:
        raise SearchError("Query must be at least 3 characters")
    return formatted
