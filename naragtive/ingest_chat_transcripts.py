from __future__ import annotations

#!/usr/bin/env python3
"""
Chat Transcript Ingestion Pipeline for Polars Vector Store.

Load and process chat transcripts from various sources (Discord, Slack, Neptune AI, text files)
into a vector-indexed Polars parquet store for semantic search.

Supports:
- Neptune AI RPG narrative exports
- Generic chat transcripts (JSON, text files)
- Metadata extraction via heuristic analysis
- Automatic embedding generation and storage
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl
from sentence_transformers import SentenceTransformer


# Default domain knowledge for Neptune AI RP (customizable)
DEFAULT_KNOWN_SHIPS = {
    "ThunderChild",
    "Stonewall",
    "Belleau",
    "Invincible",
    "Vengeance",
}

DEFAULT_LOCATION_HINTS = {
    "bridge": "bridge",
    "medbay": "medbay",
    "captains' lounge": "captains_lounge",
    "captain's lounge": "captains_lounge",
    "spaceport": "spaceport",
    "umbilical": "umbilical",
    "docking": "docking",
    "airlock": "airlock",
    "ganymede": "ganymede",
    "ceres-7": "ceres-7",
    "titan": "titan",
}

DEFAULT_EVENT_HINTS = {
    "undock": "undocking",
    "undocking": "undocking",
    "burn": "engine_burn",
    "flip": "midcourse_flip",
    "slingshot": "gravity_assist",
    "reunion": "reunion",
    "scan": "scanning",
    "engage": "engagement",
}

DEFAULT_CHAR_NAME_CANDIDATES = {
    "Heidi",
    "Jansen",
    "Eva Rostova",
    "Rostova",
    "Admiral Zelenskyy",
    "Zelenskyy",
    "Petrova",
    "Rizzo",
    "Thorne",
    "Li",
    "Dunlap",
    "Kegan",
    "Sarah Longwell",
    "Destin Sandlin",
    "Naomi",
    "Commander Costello",
    "Costello",
}


class BaseIngester(ABC):
    """
    Abstract base class for all document ingesters.
    
    Provides shared functionality for:
    - Generating embeddings using Sentence Transformers
    - Creating and saving Polars DataFrames
    - Merging new data with existing vector stores
    
    Subclasses implement specific ingestion pipelines for different
    source formats (JSON, text files, Neptune exports, etc.)
    
    Attributes:
        model: SentenceTransformer model for generating embeddings
        embedding_dim: Dimension of embeddings (384 for all-MiniLM-L6-v2)
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize ingester with embedding model.
        
        Args:
            embedding_model: HuggingFace model identifier for embeddings.
                Default: "all-MiniLM-L6-v2" (384-dim, fast, good quality)
        """
        self.model: SentenceTransformer = SentenceTransformer(embedding_model)
        self.embedding_dim: int = 384

    def _embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once.
                Default: 32
                
        Returns:
            List of embeddings, each as list[float]
            
        Example:
            ```python
            texts = ["scene 1 text", "scene 2 text", ...]
            embeddings = ingester._embed_texts(texts)
            ```
        """
        print("🧠 Generating embeddings...")
        embeddings_array = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=batch_size
        )
        return [emb.tolist() for emb in embeddings_array]

    def _create_dataframe(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadata_list: list[str],
    ) -> pl.DataFrame:
        """
        Create Polars DataFrame from ingested data.
        
        Args:
            ids: Unique document identifiers
            texts: Document text content
            embeddings: Pre-computed embeddings for each document
            metadata_list: JSON-serialized metadata for each document
            
        Returns:
            Polars DataFrame with columns: id, text, embedding, metadata
        """
        return pl.DataFrame(
            {
                "id": ids,
                "text": texts,
                "embedding": embeddings,
                "metadata": metadata_list,
            }
        )

    def _save_dataframe(
        self,
        df: pl.DataFrame,
        parquet_output: str
    ) -> None:
        """
        Save DataFrame to parquet file.
        
        Args:
            df: Polars DataFrame to save
            parquet_output: Output file path
        """
        df.write_parquet(parquet_output)
        print(f"✅ Saved {len(df)} entries to {parquet_output}")

    def _merge_with_existing(
        self,
        new_df: pl.DataFrame,
        existing_parquet: str
    ) -> pl.DataFrame:
        """
        Merge new data with existing vector store, avoiding duplicates.
        
        Checks for duplicate IDs and removes them from new data before
        concatenating with the existing DataFrame.
        
        Args:
            new_df: New DataFrame to merge
            existing_parquet: Path to existing parquet file
            
        Returns:
            Merged DataFrame
        """
        print(f"📚 Merging with existing {existing_parquet}...")

        if not Path(existing_parquet).exists():
            print("   No existing store, saving new data...")
            new_df.write_parquet(existing_parquet)
            return new_df

        # Load existing store
        existing_df = pl.read_parquet(existing_parquet)

        # Check for duplicates
        existing_ids = set(existing_df["id"].to_list())
        new_ids = set(new_df["id"].to_list())

        duplicates = existing_ids & new_ids
        if duplicates:
            print(
                f"   ⚠️  Found {len(duplicates)} duplicate IDs, removing from new data"
            )
            new_df = new_df.filter(~pl.col("id").is_in(list(duplicates)))

        # Concatenate
        merged = pl.concat([existing_df, new_df])

        # Save
        merged.write_parquet(existing_parquet)
        print(f"✅ Merged store now has {len(merged)} total entries")

        return merged

    @abstractmethod
    def ingest(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Abstract method to be implemented by subclasses."""
        pass


class ChatTranscriptIngester(BaseIngester):
    """
    Ingest generic chat transcripts (Discord, Slack, JSON, text files).
    
    Supports multiple input formats for generic chat data, with configurable
    chunking and metadata extraction.
    
    Example:
        ```python
        ingester = ChatTranscriptIngester()
        df = ingester.ingest_json_messages("discord_export.json")
        ```
    """

    def ingest_json_messages(
        self,
        json_file: str,
        collection_name: str = "chat_transcripts",
        parquet_output: str = "./chat_transcripts.parquet",
    ) -> pl.DataFrame:
        """
        Ingest chat messages from JSON file.
        
        Expected JSON format:
        ```json
        [
            {
                "timestamp": "2025-12-05T12:00:00",
                "user": "username",
                "message": "chat text here",
                "channel": "general"
            }
        ]
        ```
        
        Args:
            json_file: Path to JSON file with messages
            collection_name: Name for the collection (metadata only)
            parquet_output: Output parquet file path
            
        Returns:
            Polars DataFrame with ingested data
        """
        print(f"📖 Loading messages from {json_file}...")

        with open(json_file, "r") as f:
            messages = json.load(f)

        print(f"📈 Processing {len(messages)} messages...")

        # Prepare data
        ids: list[str] = []
        texts: list[str] = []
        metadata_list: list[str] = []

        for i, msg in enumerate(messages):
            # Create unique ID
            msg_id = f"chat_{i:06d}_{msg.get('user', 'unknown')}"
            ids.append(msg_id)

            # Extract text
            text = msg.get("message", "")
            texts.append(text)

            # Create metadata
            meta = {
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "user": msg.get("user", "unknown"),
                "channel": msg.get("channel", "general"),
                "msg_id": msg.get("id", i),
                "character_count": len(text),
                "word_count": len(text.split()),
            }
            metadata_list.append(json.dumps(meta))

        # Generate embeddings
        embeddings = self._embed_texts(texts)

        # Create DataFrame
        df = self._create_dataframe(ids, texts, embeddings, metadata_list)

        # Save
        self._save_dataframe(df, parquet_output)

        return df

    def ingest_txt_file(
        self,
        txt_file: str,
        chunk_size: int = 500,
        parquet_output: str = "./chat_transcripts.parquet",
    ) -> pl.DataFrame:
        """
        Ingest plain text file, split into chunks.
        
        Useful for logs, transcripts, and other plain text sources.
        
        Args:
            txt_file: Path to text file
            chunk_size: Characters per chunk. Default: 500
            parquet_output: Output parquet file path
            
        Returns:
            Polars DataFrame with chunks as documents
        """
        print(f"📖 Loading text from {txt_file}...")

        with open(txt_file, "r") as f:
            content = f.read()

        # Split into chunks
        chunks = [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ]
        chunks = [c.strip() for c in chunks if c.strip()]

        print(f"📈 Split into {len(chunks)} chunks...")

        # Prepare data
        ids: list[str] = []
        texts: list[str] = []
        metadata_list: list[str] = []

        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i:06d}")
            texts.append(chunk)

            meta = {
                "chunk_index": i,
                "source_file": txt_file,
                "chunk_size": chunk_size,
                "character_count": len(chunk),
                "word_count": len(chunk.split()),
                "ingestion_date": datetime.now().isoformat(),
            }
            metadata_list.append(json.dumps(meta))

        # Generate embeddings
        embeddings = self._embed_texts(texts)

        # Create DataFrame
        df = self._create_dataframe(ids, texts, embeddings, metadata_list)

        # Save
        self._save_dataframe(df, parquet_output)

        return df

    def ingest(
        self,
        source: str,
        source_type: str = "json",
        **kwargs: Any
    ) -> pl.DataFrame:
        """
        Generic ingest method that routes to specific handlers.
        
        Args:
            source: Path to source file
            source_type: Type of source ("json" or "txt")
            **kwargs: Additional arguments passed to handler
            
        Returns:
            Polars DataFrame with ingested data
            
        Raises:
            ValueError: If source_type is unsupported
        """
        if source_type == "json":
            return self.ingest_json_messages(source, **kwargs)
        elif source_type == "txt":
            return self.ingest_txt_file(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")


class NeptuneParser:
    """
    Parser for Neptune AI RP narrative export format.
    
    Extracts structured turn data from Neptune's markdown-like export format,
    including timestamps, speakers, and turn content.
    
    Attributes:
        TURN_RE: Regex pattern for parsing turn headers
        SCENE_SPLIT: Delimiter for scene boundaries
    """

    TURN_RE = re.compile(r"^\*{3}(.+?)\s*-\s*(.+?):\*{3}\s*$", re.M)
    SCENE_SPLIT = "\n---\n"

    def parse_file(self, path: str) -> dict[str, Any]:
        """
        Parse Neptune export file into structured data.
        
        Args:
            path: Path to Neptune export file
            
        Returns:
            Dictionary with keys:
                - 'title': Conversation title (if present)
                - 'turns': List of parsed turn dictionaries
        """
        text = Path(path).read_text(encoding="utf-8", errors="ignore")

        # Extract title if present
        title = self._extract_title(text)

        # Split into blocks and parse turns
        blocks = text.split(self.SCENE_SPLIT)
        turns: list[dict[str, Any]] = []

        for block in blocks:
            turn = self._parse_turn_block(block)
            if turn:
                turns.append(turn)

        return {"title": title, "turns": turns}

    def _extract_title(self, text: str) -> Optional[str]:
        """
        Extract conversation title from first line.
        
        Args:
            text: Full export text
            
        Returns:
            Title string or None if not found
        """
        if text.startswith("# Conversation:"):
            first_line = text.splitlines()[0]
            return first_line.replace("# Conversation:", "").strip()
        return None

    def _parse_turn_block(self, block: str) -> Optional[dict[str, Any]]:
        """
        Parse a single turn block.
        
        Args:
            block: Text block containing a single turn
            
        Returns:
            Turn dictionary or None if parsing failed
        """
        # Find the header line "***TIMESTAMP - SPEAKER:***"
        m = self.TURN_RE.search(block)
        if not m:
            return None

        ts_raw = m.group(1).strip()
        speaker = m.group(2).strip()

        # Extract message body after header
        body = block[m.end() :].strip()

        # Remove trailing prompt if present
        if "What do you do?" in body:
            body = body.split("What do you do?")[0].rstrip()

        # Parse timestamp
        date_iso, time_display = self._parse_timestamp(ts_raw)

        return {
            "timestamp_raw": ts_raw,
            "date_iso": date_iso,
            "time_display": time_display,
            "speaker": speaker,
            "text": body.strip(),
        }

    def _parse_timestamp(self, ts_raw: str) -> tuple[Optional[str], str]:
        """
        Parse timestamp string into ISO date and display format.
        
        Args:
            ts_raw: Raw timestamp string
            
        Returns:
            Tuple of (iso_date_str, display_format)
        """
        try:
            # Example format: 11/10/2025, 4:00:41 AM
            dt = datetime.strptime(ts_raw, "%m/%d/%Y, %I:%M:%S %p")
            return dt.date().isoformat(), ts_raw
        except Exception:
            return None, ts_raw


class SceneProcessor:
    """
    Process parsed turns into scenes by pairing consecutive exchanges.
    
    Combines consecutive turns to create scenes, with configurable pairing
    logic for two-speaker exchanges.
    """

    def pair_turns_into_scenes(
        self,
        turns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Pair consecutive turns into scenes.
        
        Args:
            turns: List of parsed turn dictionaries
            
        Returns:
            List of scene dictionaries
        """
        scenes: list[dict[str, Any]] = []
        i = 0
        idx = 0

        while i < len(turns):
            scene = self._create_scene_from_turns(
                turns[i],
                turns[i + 1] if i + 1 < len(turns) else None,
                idx,
            )
            scenes.append(scene)

            # Advance by 2 if we paired, otherwise 1
            if scene["paired"]:
                i += 2
            else:
                i += 1
            idx += 1

        return scenes

    def _create_scene_from_turns(
        self,
        t1: dict[str, Any],
        t2: Optional[dict[str, Any]],
        scene_index: int
    ) -> dict[str, Any]:
        """
        Create a scene from one or two turns.
        
        Args:
            t1: First turn
            t2: Second turn (optional)
            scene_index: Index for this scene
            
        Returns:
            Scene dictionary
        """
        combined_text = t1["text"]
        speakers = [t1["speaker"]]
        date_iso = t1.get("date_iso")
        time_display = t1.get("time_display")
        paired = False

        # Try to pair with second turn if available and non-empty
        if t2 and t2["text"]:
            combined_text = (t1["text"] + "\n\n" + t2["text"]).strip()
            speakers.append(t2["speaker"])
            paired = True

            # Use earliest available date/time
            if not date_iso:
                date_iso = t2.get("date_iso")
            if not time_display:
                time_display = t2.get("time_display")

        return {
            "scene_index": scene_index,
            "date_iso": date_iso,
            "time_display": time_display,
            "speakers": speakers,
            "text": combined_text,
            "paired": paired,
        }


class HeuristicAnalyzer:
    """
    Extract metadata from scene text using domain-specific heuristics.
    
    Identifies named entities (characters, locations, ships, events) and
    analyzes tone and emotional intensity using keyword matching.
    
    Attributes:
        ships: Set of known ship names
        locations: Dict mapping location keywords to canonical names
        events: Dict mapping event keywords to canonical names
        characters: Set of known character names
    """

    def __init__(
        self,
        ships: Optional[set[str]] = None,
        locations: Optional[dict[str, str]] = None,
        events: Optional[dict[str, str]] = None,
        characters: Optional[set[str]] = None,
    ) -> None:
        """
        Initialize with domain knowledge.
        
        Args:
            ships: Known ship names (default: Neptune AI ships)
            locations: Location keyword mappings (default: common locations)
            events: Event keyword mappings (default: common events)
            characters: Known character names (default: Neptune cast)
        """
        self.ships: set[str] = ships or DEFAULT_KNOWN_SHIPS
        self.locations: dict[str, str] = locations or DEFAULT_LOCATION_HINTS
        self.events: dict[str, str] = events or DEFAULT_EVENT_HINTS
        self.characters: set[str] = characters or DEFAULT_CHAR_NAME_CANDIDATES

    def analyze_scene(self, text: str, speakers: list[str]) -> dict[str, Any]:
        """
        Extract all metadata from scene text.
        
        Args:
            text: Scene text content
            speakers: List of speaker names in the scene
            
        Returns:
            Dictionary with extracted metadata
        """
        return {
            "characters": self._extract_characters(text),
            "location": self._extract_location(text),
            "ships": self._extract_ships(text),
            "events": self._extract_events(text),
            "tone": self._analyze_tone(text),
            "emotional_intensity": self._analyze_emotional_intensity(text),
            "action_level": self._analyze_action_level(text),
            "pov": self._determine_pov(speakers, text),
        }

    def _extract_characters(self, text: str) -> list[str]:
        """
        Extract character names from text.
        
        Args:
            text: Scene text
            
        Returns:
            List of character names found
        """
        found: set[str] = set()

        # Check known characters
        for name in self.characters:
            if name in text:
                found.add(name)

        # Heuristic: proper nouns with filtering
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
            cand = m.group(1)
            if self._is_valid_character_name(cand):
                found.add(cand)

        return sorted(found)

    def _is_valid_character_name(self, name: str) -> bool:
        """
        Filter out common words that aren't character names.
        
        Args:
            name: Candidate name
            
        Returns:
            True if valid character name
        """
        if len(name) < 2:
            return False

        # Skip common titles and pronouns
        excluded = {
            "You", "I", "We", "He", "She", "Captain", "Commander",
            "Lieutenant", "SRF", "ThunderChild", "User", "Venice",
        }
        if name in excluded:
            return False

        # Skip single-word common terms
        if len(name.split()) == 1 and name.lower() in {"the", "and", "it", "its", "this"}:
            return False

        return True

    def _extract_location(self, text: str) -> str:
        """
        Extract location from text.
        
        Args:
            text: Scene text
            
        Returns:
            Location name or "unknown"
        """
        low = text.lower()
        for keyword, location in self.locations.items():
            if keyword in low:
                return location
        return "unknown"

    def _extract_ships(self, text: str) -> list[str]:
        """
        Extract ship names from text.
        
        Args:
            text: Scene text
            
        Returns:
            List of ship names
        """
        ships: list[str] = []

        # Check known ships
        for ship in self.ships:
            if ship in text:
                ships.append(ship)

        # Also detect SRV <Name> pattern
        for m in re.finditer(r"\bSRV\s+([A-Z][A-Za-z0-9_-]+)\b", text):
            ships.append(m.group(1))

        return sorted(set(ships))

    def _extract_events(self, text: str) -> list[str]:
        """
        Extract events from text.
        
        Args:
            text: Scene text
            
        Returns:
            List of event names
        """
        low = text.lower()
        events: set[str] = set()

        for keyword, event_tag in self.events.items():
            if keyword in low:
                events.add(event_tag)

        return sorted(events)

    def _analyze_tone(self, text: str) -> str:
        """
        Analyze overall tone of text.
        
        Args:
            text: Scene text
            
        Returns:
            Tone classification ("tense", "emotional", or "neutral")
        """
        low = text.lower()

        action_terms = sum(
            w in low
            for w in [
                "burn", "flip", "engage", "railgun", "combat",
                "attack", "orders", "slingshot",
            ]
        )

        emo_terms = sum(
            w in low
            for w in ["smile", "laugh", "humiliation", "shock", "joy", "relief", "fear"]
        )

        if action_terms > 1:
            return "tense"
        elif emo_terms > 1:
            return "emotional"
        else:
            return "neutral"

    def _analyze_emotional_intensity(self, text: str) -> float:
        """
        Calculate emotional intensity score.
        
        Args:
            text: Scene text
            
        Returns:
            Score between 0.0 and 1.0
        """
        low = text.lower()
        exclaim = text.count("!")

        emo_terms = sum(
            w in low
            for w in ["smile", "laugh", "humiliation", "shock", "joy", "relief", "fear"]
        )

        return min(1.0, 0.1 * emo_terms + 0.05 * exclaim)

    def _analyze_action_level(self, text: str) -> float:
        """
        Calculate action level score.
        
        Args:
            text: Scene text
            
        Returns:
            Score between 0.0 and 1.0
        """
        low = text.lower()

        action_terms = sum(
            w in low
            for w in [
                "burn", "flip", "engage", "railgun", "combat",
                "attack", "orders", "slingshot",
            ]
        )

        return min(1.0, 0.15 * action_terms)

    def _determine_pov(self, speakers: list[str], text: str) -> str:
        """
        Determine point-of-view character.
        
        Args:
            speakers: List of speakers in scene
            text: Scene text
            
        Returns:
            POV character name
        """
        first_person = len(re.findall(r"\bI\b", text)) >= 2
        second_person = len(re.findall(r"\byou\b", text, flags=re.I)) >= 2

        if first_person and not second_person:
            return "User"
        if second_person and not first_person:
            return "Venice"

        # Fallback to first speaker
        return speakers[0] if speakers else "UNKNOWN"


class NeptuneIngester(BaseIngester):
    """
    Ingest Neptune AI RP narrative exports into Polars vector store.
    
    Complete ingestion pipeline combining parsing, scene processing,
    metadata extraction, embedding generation, and storage.
    
    Example:
        ```python
        ingester = NeptuneIngester()
        df = ingester.ingest(
            "export.txt",
            parquet_output="./scenes.parquet",
            append=True
        )
        print(f"Ingested {len(df)} scenes")
        ```
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        ships: Optional[set[str]] = None,
        locations: Optional[dict[str, str]] = None,
        events: Optional[dict[str, str]] = None,
        characters: Optional[set[str]] = None,
    ) -> None:
        """
        Initialize Neptune ingester with optional custom domain knowledge.
        
        Args:
            embedding_model: HuggingFace model for embeddings
            ships: Custom set of known ship names
            locations: Custom location keyword mappings
            events: Custom event keyword mappings
            characters: Custom set of known character names
        """
        super().__init__(embedding_model)
        self.parser: NeptuneParser = NeptuneParser()
        self.scene_processor: SceneProcessor = SceneProcessor()
        self.analyzer: HeuristicAnalyzer = HeuristicAnalyzer(
            ships=ships,
            locations=locations,
            events=events,
            characters=characters,
        )

    def ingest(
        self,
        export_path: str,
        parquet_output: str = "./thunderchild_scenes.parquet",
        append: bool = True,
    ) -> pl.DataFrame:
        """
        Ingest Neptune export file into vector store.
        
        Args:
            export_path: Path to Neptune export file
            parquet_output: Output parquet file path
            append: If True, merge with existing data. Default: True
            
        Returns:
            Polars DataFrame with ingested scenes
            
        Example:
            ```python
            ingester = NeptuneIngester()
            df = ingester.ingest(
                "neptune_export.txt",
                parquet_output="./scenes.parquet",
                append=True
            )
            ```
        """
        print(f"📖 Loading Neptune export from {export_path}...")

        # Parse file
        parsed = self.parser.parse_file(export_path)
        print(f"📈 Processing {len(parsed['turns'])} turns...")

        # Process into scenes
        scenes = self.scene_processor.pair_turns_into_scenes(parsed["turns"])
        print(f"🎞 Created {len(scenes)} scenes...")

        # Prepare data for embedding
        ids: list[str] = []
        texts: list[str] = []
        metadata_list: list[str] = []

        for scene in scenes:
            # Analyze scene for metadata
            analysis = self.analyzer.analyze_scene(scene["text"], scene["speakers"])

            # Create scene ID
            date_iso = scene.get("date_iso") or "UNKNOWN"
            scene_id = f"scene_{scene['scene_index']:04d}_{date_iso}"
            ids.append(scene_id)

            # Add text
            texts.append(scene["text"])

            # Build metadata
            metadata = {
                "scene_id": scene_id,
                "time_display": scene.get("time_display", "UNKNOWN"),
                "date_iso": date_iso,
                "pov_character": analysis["pov"],
                "location": analysis["location"],
                "speakers": scene["speakers"],
                "characters_present": json.dumps(analysis["characters"]),
                "ships": analysis["ships"],
                "events": analysis["events"],
                "tone": analysis["tone"],
                "emotional_intensity": analysis["emotional_intensity"],
                "action_level": analysis["action_level"],
                "plot_significance": 0.5,
                "source_title": parsed.get("title"),
                "source_file": str(export_path),
            }
            metadata_list.append(json.dumps(metadata))

        # Generate embeddings
        embeddings = self._embed_texts(texts)

        # Create DataFrame
        new_df = self._create_dataframe(ids, texts, embeddings, metadata_list)

        # Save or merge
        if append:
            return self._merge_with_existing(new_df, parquet_output)
        else:
            self._save_dataframe(new_df, parquet_output)
            return new_df


def ingest_neptune_export_to_parquet(
    export_path: str,
    parquet_out: str = "./thunderchild_scenes.parquet",
    append: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> pl.DataFrame:
    """
    Legacy function for backward compatibility.
    
    Use NeptuneIngester class for new code.
    
    Args:
        export_path: Path to Neptune export file
        parquet_out: Output parquet file path
        append: Merge with existing data
        embedding_model: Embedding model to use
        
    Returns:
        Polars DataFrame with ingested data
    """
    ingester = NeptuneIngester(embedding_model=embedding_model)
    return ingester.ingest(export_path, parquet_out, append)


def example_usage() -> None:
    """
    Example: Ingest Discord/Slack chat export and Neptune AI RP.
    
    Demonstrates basic usage patterns for both chat ingestion
    and Neptune RP ingestion.
    """

    # Example 1: Generic chat transcript ingestion
    print("=" * 60)
    print("Example 1: Generic Chat Transcript Ingestion")
    print("=" * 60)

    chat_ingester = ChatTranscriptIngester()

    example_messages = [
        {
            "timestamp": "2025-12-05T10:30:00",
            "user": "Kieran",
            "message": "Admiral command leadership",
            "channel": "thunderchild",
        },
        {
            "timestamp": "2025-12-05T10:31:00",
            "user": "Venice",
            "message": "The Admiral's face transformed, a mask of professional curiosity becoming cold and calculating.",
            "channel": "thunderchild",
        },
        {
            "timestamp": "2025-12-05T10:32:00",
            "user": "Petrova",
            "message": "Reunion after the mission was emotional and chaotic.",
            "channel": "thunderchild",
        },
    ]

    # Save example to JSON
    with open("example_chat.json", "w") as f:
        json.dump(example_messages, f, indent=2)

    # Ingest
    df = chat_ingester.ingest_json_messages("example_chat.json")
    print("\n📋 Ingested DataFrame:")
    print(df.select(["id", "text"]).head())

    # Example 2: Neptune AI RP ingestion (if export file exists)
    print("\n" + "=" * 60)
    print("Example 2: Neptune AI RP Ingestion")
    print("=" * 60)
    print("\nTo ingest Neptune exports:")
    print("```python")
    print("neptune_ingester = NeptuneIngester()")
    print("df = neptune_ingester.ingest('path/to/neptune_export.txt')")
    print("```")


if __name__ == "__main__":
    example_usage()
