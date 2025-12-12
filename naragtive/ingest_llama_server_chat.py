from __future__ import annotations

#!/usr/bin/env python3
"""
Llama-Server Chat Export Ingestion Pipeline for Polars Vector Store.

Load and process chat exports from llama-server (llama.cpp web interface)
into a vector-indexed Polars parquet store for semantic search.

Supports:
- Llama-server JSON export format with conversation and message trees
- User/assistant role separation and exchange grouping
- Heuristic analysis for themes, tone, engagement, and complexity
- Automatic embedding generation and storage
- Chain-of-thought (thinking) content preservation

Example:
    ```python
    ingester = LlamaServerIngester()
    df = ingester.ingest_llama_server_export("export.json")
    store = PolarsVectorStore()
    store.load()  # With ingested data
    results = store.query("sci-fi battle", n_results=5)
    ```
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl
from sentence_transformers import SentenceTransformer


class LlamaServerParser:
    """
    Parser for llama-server Web UI chat export JSON format.
    
    Validates and extracts structured data from llama-server exports,
    including conversation metadata and message trees.
    
    Attributes:
        export_data: Parsed JSON export data
    """
    
    def parse_export(self, file_path: str) -> dict[str, Any]:
        """
        Load and validate llama-server JSON export file.
        
        Args:
            file_path: Path to llama-server export JSON file
            
        Returns:
            Dictionary with keys:
                - 'conv': Conversation metadata (id, name, lastModified, currNode)
                - 'messages': List of message dictionaries
                
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
            ValueError: If required fields are missing
            
        Example:
            ```python
            parser = LlamaServerParser()
            data = parser.parse_export("my_chat.json")
            print(f"Conversation: {data['conv']['name']}")
            print(f"Messages: {len(data['messages'])}")
            ```
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Export file not found: {file_path}")
        
        print(f"📖 Loading llama-server export from {file_path}...")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate required structure
        if "conv" not in data or "messages" not in data:
            raise ValueError("Invalid llama-server export: missing 'conv' or 'messages' keys")
        
        conv = data["conv"]
        required_conv_fields = {"id", "name", "lastModified"}
        if not required_conv_fields.issubset(conv.keys()):
            raise ValueError(f"Missing conv fields: {required_conv_fields - conv.keys()}")
        
        print(f"✅ Loaded export with {len(data['messages'])} messages")
        
        return data
    
    def extract_conversation_name(self, name: str) -> str:
        """
        Clean and truncate conversation name for storage.
        
        Args:
            name: Raw conversation name (can be very long prompt)
            
        Returns:
            Cleaned name (first 200 characters)
            
        Example:
            ```python
            name = "Describe a scene from a long-running, near-future..."
            clean = parser.extract_conversation_name(name)
            # Returns: "Describe a scene from a long-running, near-future..."
            ```
        """
        # Truncate to first 200 characters, preserve meaning
        if len(name) > 200:
            return name[:197] + "..."
        return name
    
    @staticmethod
    def timestamp_to_datetime(timestamp_ms: int) -> datetime:
        """
        Convert millisecond timestamp to UTC datetime.
        
        Args:
            timestamp_ms: Millisecond Unix timestamp
            
        Returns:
            datetime object in UTC timezone
            
        Example:
            ```python
            dt = LlamaServerParser.timestamp_to_datetime(1765275434078)
            print(dt.isoformat())  # 2025-11-10T...
            ```
        """
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


class LlamaServerExchangeGrouper:
    """
    Group user/assistant messages into conversational exchanges.
    
    Organizes linear message stream into user-initiated, assistant-responded
    exchanges, creating coherent dialogue scenes.
    
    Attributes:
        messages: Full list of messages from export
    """
    
    def group_into_exchanges(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Group consecutive user/assistant message pairs into exchanges.
        
        Args:
            messages: List of message dictionaries from export
            
        Returns:
            List of exchange dictionaries with:
                - user_id, user_content, user_timestamp
                - assistant_id, assistant_content, assistant_timestamp
                - model, has_thinking, thinking_content
                - exchange_index
                
        Example:
            ```python
            grouper = LlamaServerExchangeGrouper()
            exchanges = grouper.group_into_exchanges(messages)
            for ex in exchanges:
                print(f"Exchange {ex['exchange_index']}:")
                print(f"User: {ex['user_content'][:100]}...")
                print(f"Assistant: {ex['assistant_content'][:100]}...")
            ```
        """
        exchanges: list[dict[str, Any]] = []
        exchange_index = 0
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            # Skip non-text messages
            if msg.get("type") != "text":
                i += 1
                continue
            
            # Look for user message
            if msg.get("role") == "user":
                user_id = msg.get("id", f"user_{exchange_index}")
                user_content = msg.get("content", "")
                user_timestamp = msg.get("timestamp", 0)
                
                # Look for next assistant message
                assistant_id = None
                assistant_content = None
                assistant_timestamp = None
                model = None
                has_thinking = False
                thinking_content = ""
                
                # Search in children first (preferred parent-child relationship)
                children = msg.get("children", [])
                if children:
                    # Find the first assistant message in children
                    for child_id in children:
                        for j in range(i + 1, len(messages)):
                            if messages[j].get("id") == child_id and messages[j].get("role") == "assistant":
                                assistant_id = messages[j].get("id")
                                assistant_content = messages[j].get("content", "")
                                assistant_timestamp = messages[j].get("timestamp", user_timestamp)
                                model = messages[j].get("model", "unknown")
                                thinking = messages[j].get("thinking", "")
                                has_thinking = bool(thinking and thinking.strip())
                                thinking_content = thinking
                                break
                        if assistant_id:
                            break
                
                # Fallback: take next assistant message
                if not assistant_id and i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "assistant" and next_msg.get("type") == "text":
                        assistant_id = next_msg.get("id")
                        assistant_content = next_msg.get("content", "")
                        assistant_timestamp = next_msg.get("timestamp", user_timestamp)
                        model = next_msg.get("model", "unknown")
                        thinking = next_msg.get("thinking", "")
                        has_thinking = bool(thinking and thinking.strip())
                        thinking_content = thinking
                        i += 1  # Skip the assistant message in main loop
                
                # Create exchange if we have both messages
                if assistant_content:
                    exchanges.append({
                        "exchange_index": exchange_index,
                        "user_id": user_id,
                        "user_content": user_content,
                        "user_timestamp": user_timestamp,
                        "assistant_id": assistant_id or f"assistant_{exchange_index}",
                        "assistant_content": assistant_content,
                        "assistant_timestamp": assistant_timestamp or user_timestamp,
                        "model": model or "unknown",
                        "has_thinking": has_thinking,
                        "thinking_content": thinking_content,
                    })
                    exchange_index += 1
            
            i += 1
        
        return exchanges
    
    def create_scene_from_exchange(
        self,
        exchange: dict[str, Any],
        conversation_id: str,
        conversation_name: str,
    ) -> dict[str, Any]:
        """
        Create a scene document from a user/assistant exchange.
        
        Args:
            exchange: Exchange dictionary from group_into_exchanges()
            conversation_id: ID of parent conversation
            conversation_name: Name/title of parent conversation
            
        Returns:
            Scene dictionary with:
                - id: Unique scene ID
                - text: Formatted dialogue text
                - conversation_id, conversation_name
                - date_iso, model, timestamp, has_thinking
                
        Example:
            ```python
            scene = grouper.create_scene_from_exchange(
                exchange,
                "conv-123",
                "My Sci-Fi Story"
            )
            print(scene["text"])
            ```
        """
        # Create scene ID combining conversation and exchange index
        scene_index = exchange["exchange_index"]
        timestamp = exchange["assistant_timestamp"]
        date_obj = LlamaServerParser.timestamp_to_datetime(timestamp)
        date_iso = date_obj.date().isoformat()
        
        scene_id = f"scene_{conversation_id[:8]}_{scene_index:04d}_{date_iso}"
        
        # Format text as dialogue
        text = f"User: {exchange['user_content']}\n\nAssistant: {exchange['assistant_content']}"
        
        # Optionally include thinking in metadata
        thinking_preview = ""
        if exchange["has_thinking"]:
            thinking_preview = exchange["thinking_content"][:200]
        
        return {
            "scene_id": scene_id,
            "text": text,
            "conversation_id": conversation_id,
            "conversation_name": conversation_name,
            "date_iso": date_iso,
            "model": exchange["model"],
            "timestamp": timestamp,
            "has_thinking": exchange["has_thinking"],
            "thinking_preview": thinking_preview,
        }


class LlamaServerHeuristicAnalyzer:
    """
    Extract metadata from llama-server conversations using heuristics.
    
    Analyzes dialogue content to determine themes, tone, engagement level,
    and complexity for improved search and discovery.
    """
    
    THEME_KEYWORDS = {
        "creative": {
            "imaginative", "story", "describe", "write", "fiction",
            "scene", "dialogue", "character", "world", "setting", "narrative"
        },
        "technical": {
            "code", "python", "javascript", "api", "database", "function",
            "algorithm", "technical", "implement", "debug", "error", "system"
        },
        "analytical": {
            "analyze", "explain", "discuss", "compare", "evaluate", "research",
            "theory", "concept", "framework", "evidence", "conclusion"
        },
        "conversational": {
            "chat", "discuss", "talk", "think", "opinion", "feel", "believe",
            "seem", "interesting", "curious", "wonder"
        },
        "instructional": {
            "teach", "learn", "guide", "step", "instruction", "how", "tutorial",
            "example", "exercise", "practice", "help"
        },
    }
    
    TONE_KEYWORDS = {
        "formal": {
            "hereby", "consequently", "moreover", "furthermore", "regarding",
            "however", "thus", "thereof", "professional"
        },
        "casual": {
            "yeah", "okay", "cool", "lol", "haha", "awesome", "gonna",
            "kinda", "like", "so", "basically"
        },
        "technical": {
            "algorithm", "variable", "function", "parameter", "return",
            "debug", "optimize", "performance", "buffer"
        },
        "creative": {
            "vivid", "magnificent", "glittering", "shimmering", "ethereal",
            "mysterious", "haunting", "exquisite", "breathtaking"
        },
    }
    
    def extract_themes(self, text: str) -> list[str]:
        """
        Extract theme tags from text content.
        
        Args:
            text: Combined user + assistant text
            
        Returns:
            List of theme tags: 'creative', 'technical', 'analytical', etc.
            
        Example:
            ```python
            analyzer = LlamaServerHeuristicAnalyzer()
            themes = analyzer.extract_themes("Write me a Python function...")
            # Returns: ['creative', 'technical']
            ```
        """
        low = text.lower()
        found_themes: list[str] = []
        
        for theme, keywords in self.THEME_KEYWORDS.items():
            if any(keyword in low for keyword in keywords):
                found_themes.append(theme)
        
        return found_themes if found_themes else ["conversational"]
    
    def analyze_tone(self, text: str) -> str:
        """
        Determine dominant tone of the conversation.
        
        Args:
            text: Combined user + assistant text
            
        Returns:
            Tone string: 'formal', 'casual', 'technical', 'creative', 'neutral'
            
        Example:
            ```python
            tone = analyzer.analyze_tone(scene_text)
            print(f"Tone: {tone}")
            ```
        """
        low = text.lower()
        scores = {}
        
        for tone, keywords in self.TONE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in low)
            scores[tone] = score
        
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "neutral"
    
    def analyze_engagement_level(self, text: str) -> float:
        """
        Calculate engagement level based on content characteristics.
        
        Factors: text length, question marks, exclamation marks, word variation
        
        Args:
            text: Combined user + assistant text
            
        Returns:
            Engagement score 0.0-1.0
            
        Example:
            ```python
            engagement = analyzer.analyze_engagement_level(scene_text)
            print(f"Engagement: {engagement:.1%}")
            ```
        """
        if not text:
            return 0.0
        
        # Length factor (longer = more engaged)
        length_score = min(len(text) / 1000.0, 1.0)  # Adjusted threshold from 2000
        
        # Punctuation factor (questions and exclamations = more engaged)
        question_count = text.count("?")
        exclamation_count = text.count("!")
        punct_score = min((question_count * 0.2 + exclamation_count * 0.15), 1.0)  # Increased weights
        
        # Dialogue/narrative richness
        dialogue_markers = text.count('"') + text.count("'")
        dialogue_score = min(dialogue_markers / 10.0, 1.0)  # Adjusted threshold from 20
        
        # Combined score with adjusted weights
        engagement = (length_score * 0.3 + punct_score * 0.4 + dialogue_score * 0.3)  # Increased punct weight
        
        return min(engagement, 1.0)
    
    def analyze_complexity(self, text: str) -> float:
        """
        Calculate text complexity based on vocabulary and structure.
        
        Factors: average word length, sentence length, unique words
        
        Args:
            text: Combined user + assistant text
            
        Returns:
            Complexity score 0.0-1.0
            
        Example:
            ```python
            complexity = analyzer.analyze_complexity(scene_text)
            print(f"Complexity: {complexity:.1%}")
            ```
        """
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / len(words)
        word_length_score = min(avg_word_length / 5.5, 1.0)  # Adjusted from 8.0 (lower threshold = lower scores)
        
        # Sentence complexity (average sentence length)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_score = min(avg_sentence_length / 15.0, 1.0)  # Adjusted from 25 (lower threshold)
        else:
            sentence_score = 0.0
        
        # Vocabulary diversity
        unique_words = len(set(w.lower() for w in words))
        diversity_score = min(unique_words / (len(words) * 0.4), 1.0) if words else 0.0  # Adjusted from 0.6
        
        # Combined complexity with adjusted weights
        complexity = (word_length_score * 0.35 + sentence_score * 0.35 + diversity_score * 0.3)  # Reduced weights
        
        return min(complexity, 1.0)


class LlamaServerIngester:
    """
    Main orchestrator for llama-server chat export ingestion.
    
    Coordinates parsing, grouping, analysis, and embedding generation
    to create searchable vector store from llama-server exports.
    
    Attributes:
        model: SentenceTransformer for embedding generation
        parser: LlamaServerParser instance
        grouper: LlamaServerExchangeGrouper instance
        analyzer: LlamaServerHeuristicAnalyzer instance
        
    Example:
        ```python
        ingester = LlamaServerIngester()
        df = ingester.ingest_llama_server_export("export.json")
        
        # Save to vector store
        store = PolarsVectorStore("./llama_chats.parquet")
        # (use df to populate store)
        store.load()
        
        # Search
        results = store.query("sci-fi battle scene", n_results=5)
        ```
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize ingester with embedding model.
        
        Args:
            embedding_model: HuggingFace model ID for embeddings.
                Default: "all-MiniLM-L6-v2" (384-dim, fast, good quality)
        """
        self.embedding_model: SentenceTransformer = SentenceTransformer(embedding_model)
        self.embedding_dim: int = 384
        self.parser: LlamaServerParser = LlamaServerParser()
        self.grouper: LlamaServerExchangeGrouper = LlamaServerExchangeGrouper()
        self.analyzer: LlamaServerHeuristicAnalyzer = LlamaServerHeuristicAnalyzer()
    
    def ingest_llama_server_export(
        self,
        file_path: str,
        output_parquet: str = "./llama_chats.parquet",
    ) -> pl.DataFrame:
        """
        Ingest single llama-server export file into vector store.
        
        Complete pipeline: parse → group exchanges → analyze → embed → save
        
        Args:
            file_path: Path to llama-server JSON export
            output_parquet: Output parquet file path
            
        Returns:
            Polars DataFrame with columns:
                - id: Scene ID
                - text: Dialogue text
                - embedding: 384-dim embedding vector
                - metadata: JSON string with themes, tone, complexity, etc.
                
        Example:
            ```python
            ingester = LlamaServerIngester()
            df = ingester.ingest_llama_server_export("my_chat.json")
            print(f"Ingested {len(df)} scenes")
            ```
        """
        # Parse export
        export_data = self.parser.parse_export(file_path)
        conv = export_data["conv"]
        messages = export_data["messages"]
        
        conversation_id = conv["id"]
        conversation_name = self.parser.extract_conversation_name(conv["name"])
        
        # Group into exchanges
        print(f"📈 Grouping {len(messages)} messages into exchanges...")
        exchanges = self.grouper.group_into_exchanges(messages)
        print(f"🎞 Created {len(exchanges)} exchanges")
        
        # Prepare data for embedding
        ids: list[str] = []
        texts: list[str] = []
        metadata_list: list[str] = []
        
        for exchange in exchanges:
            scene = self.grouper.create_scene_from_exchange(
                exchange,
                conversation_id,
                conversation_name,
            )
            
            # Analyze scene
            combined_text = exchange["user_content"] + " " + exchange["assistant_content"]
            themes = self.analyzer.extract_themes(combined_text)
            tone = self.analyzer.analyze_tone(combined_text)
            engagement = self.analyzer.analyze_engagement_level(combined_text)
            complexity = self.analyzer.analyze_complexity(combined_text)
            
            ids.append(scene["scene_id"])
            texts.append(scene["text"])
            
            # Build metadata
            metadata = {
                "scene_id": scene["scene_id"],
                "conversation_id": scene["conversation_id"],
                "conversation_name": scene["conversation_name"],
                "date_iso": scene["date_iso"],
                "timestamp": scene["timestamp"],
                "model": scene["model"],
                "has_thinking": scene["has_thinking"],
                "thinking_preview": scene["thinking_preview"],
                "themes": themes,
                "tone": tone,
                "engagement_level": engagement,
                "complexity": complexity,
                "exchange_index": exchange["exchange_index"],
                "source_file": str(file_path),
            }
            metadata_list.append(json.dumps(metadata))
        
        # Generate embeddings
        print(f"🧠 Generating {len(texts)} embeddings...")
        embeddings_array = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = [emb.tolist() for emb in embeddings_array]
        
        # Create DataFrame
        df = pl.DataFrame({
            "id": ids,
            "text": texts,
            "embedding": embeddings,
            "metadata": metadata_list,
        })
        
        # Save
        df.write_parquet(output_parquet)
        print(f"✅ Saved {len(df)} scenes to {output_parquet}")
        
        return df
    
    def ingest_multiple_exports(
        self,
        file_paths: list[str],
        output_parquet: str = "./llama_chats_combined.parquet",
    ) -> pl.DataFrame:
        """
        Ingest multiple llama-server exports and combine into single store.
        
        Args:
            file_paths: List of paths to llama-server JSON exports
            output_parquet: Output parquet file path
            
        Returns:
            Combined Polars DataFrame from all exports
            
        Example:
            ```python
            ingester = LlamaServerIngester()
            df = ingester.ingest_multiple_exports([
                "chat1.json",
                "chat2.json",
                "chat3.json",
            ])
            print(f"Combined: {len(df)} scenes")
            ```
        """
        print(f"📚 Ingesting {len(file_paths)} export files...\n")
        
        all_dfs: list[pl.DataFrame] = []
        for file_path in file_paths:
            try:
                df = self.ingest_llama_server_export(
                    file_path,
                    output_parquet="./temp_llama_export.parquet",  # Use temp file
                )
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("No exports were successfully processed")
        
        # Combine all DataFrames
        combined_df = pl.concat(all_dfs)
        
        # Remove duplicates by ID (prefer first occurrence)
        combined_df = combined_df.unique(subset=["id"], keep="first")
        
        # Save combined
        combined_df.write_parquet(output_parquet)
        print(f"\n✅ Combined {len(combined_df)} unique scenes to {output_parquet}")
        
        return combined_df


def ingest_llama_server_export_to_parquet(
    file_path: str,
    output_parquet: str = "./llama_chats.parquet",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> pl.DataFrame:
    """
    Legacy convenience function for llama-server export ingestion.
    
    Use LlamaServerIngester class for new code.
    
    Args:
        file_path: Path to llama-server JSON export
        output_parquet: Output parquet file path
        embedding_model: Embedding model to use
        
    Returns:
        Polars DataFrame with ingested data
        
    Example:
        ```python
        df = ingest_llama_server_export_to_parquet("my_export.json")
        ```
    """
    ingester = LlamaServerIngester(embedding_model=embedding_model)
    return ingester.ingest_llama_server_export(file_path, output_parquet)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Llama-Server Chat Ingestion Example")
    print("=" * 60)
    
    ingester = LlamaServerIngester()
    
    print("\nTo ingest a llama-server export:")
    print("```python")
    print("ingester = LlamaServerIngester()")
    print("df = ingester.ingest_llama_server_export('path/to/export.json')")
    print("```")
    
    print("\nTo ingest multiple exports:")
    print("```python")
    print("df = ingester.ingest_multiple_exports([")
    print("    'export1.json',")
    print("    'export2.json',")
    print("])")
    print("```")
    
    print("\nDataFrame will contain:")
    print("- id: Scene ID")
    print("- text: Formatted dialogue")
    print("- embedding: 384-dimensional vector")
    print("- metadata: JSON with themes, tone, complexity, etc.")
