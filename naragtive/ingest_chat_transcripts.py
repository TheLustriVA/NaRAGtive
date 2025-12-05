#!/usr/bin/env python3
"""
Chat Transcript Ingestion for Polars Vector Store
Load chat transcripts (from Discord, Slack, text files, etc.) into your Polars vector store
"""

import json
import re
import polars as pl
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from datetime import datetime


NEPTUNE_TURN_RE = re.compile(r"^\*{3}(.+?)\s*-\s*(.+?):\*{3}\s*$", re.M)
SCENE_SPLIT = "\n---\n"

# Optional domain lists you can expand over time
KNOWN_SHIPS = {
    "ThunderChild",
    "Stonewall",
    "Belleau",
    "Invincible",
    "Vengeance",
}
LOCATION_HINTS = {
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
EVENT_HINTS = {
    "undock": "undocking",
    "undocking": "undocking",
    "burn": "engine_burn",
    "flip": "midcourse_flip",
    "slingshot": "gravity_assist",
    "reunion": "reunion",
    "scan": "scanning",
    "engage": "engagement",
}
CHAR_NAME_CANDIDATES = {
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


class ChatTranscriptIngester:
    """Ingest chat transcripts into Polars vector store"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize ingester"""
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # all-MiniLM output size

    def ingest_json_messages(
        self,
        json_file: str,
        collection_name: str = "chat_transcripts",
        parquet_output: str = "./chat_transcripts.parquet",
    ) -> pl.DataFrame:
        """
        Ingest chat messages from JSON file

        Expected JSON format:
        [
            {
                "timestamp": "2025-12-05T12:00:00",
                "user": "username",
                "message": "chat text here",
                "channel": "general",  # optional
            },
            ...
        ]
        """
        print(f"📖 Loading messages from {json_file}...")

        with open(json_file, "r") as f:
            messages = json.load(f)

        print(f"📊 Processing {len(messages)} messages...")

        # Prepare data
        ids = []
        texts = []
        embeddings = []
        metadata_list = []

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

        # Batch embed
        print("🧠 Generating embeddings...")
        embeddings_array = self.model.encode(
            texts, show_progress_bar=True, batch_size=32
        )

        # Convert to list of lists
        embeddings = [emb.tolist() for emb in embeddings_array]

        # Create DataFrame
        df = pl.DataFrame(
            {
                "id": ids,
                "text": texts,
                "embedding": embeddings,
                "metadata": metadata_list,
            }
        )

        # Save to parquet
        df.write_parquet(parquet_output)
        print(f"✅ Saved {len(df)} messages to {parquet_output}")

        return df

    def ingest_txt_file(
        self,
        txt_file: str,
        chunk_size: int = 500,  # characters per chunk
        parquet_output: str = "./chat_transcripts.parquet",
    ) -> pl.DataFrame:
        """
        Ingest plain text file, split into chunks
        Useful for logs, transcripts, etc.
        """
        print(f"📖 Loading text from {txt_file}...")

        with open(txt_file, "r") as f:
            content = f.read()

        # Split into chunks
        chunks = [
            content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
        ]
        chunks = [c.strip() for c in chunks if c.strip()]

        print(f"📊 Split into {len(chunks)} chunks...")

        # Prepare data
        ids = []
        texts = []
        embeddings = []
        metadata_list = []

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

        # Batch embed
        print("🧠 Generating embeddings...")
        embeddings_array = self.model.encode(
            texts, show_progress_bar=True, batch_size=32
        )
        embeddings = [emb.tolist() for emb in embeddings_array]

        # Create DataFrame
        df = pl.DataFrame(
            {
                "id": ids,
                "text": texts,
                "embedding": embeddings,
                "metadata": metadata_list,
            }
        )

        # Save to parquet
        df.write_parquet(parquet_output)
        print(f"✅ Saved {len(df)} chunks to {parquet_output}")

        return df

    def merge_with_existing(
        self, new_df: pl.DataFrame, existing_parquet: str
    ) -> pl.DataFrame:
        """
        Merge new data with existing vector store
        Avoids duplicates by checking IDs
        """
        print(f"📚 Merging with existing {existing_parquet}...")

        if not Path(existing_parquet).exists():
            print("   No existing store, saving new data...")
            new_df.write_parquet(existing_parquet)
            return new_df

        # Load existing
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

def parse_neptune_export(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # Title (optional)
    title = None
    if text.startswith("# Conversation:"):
        first = text.splitlines()[0]
        title = first.replace("# Conversation:", "").strip()
    # Split blocks by '---'
    blocks = text.split(SCENE_SPLIT)
    turns = []
    for block in blocks:
        # Find the header line "***TIMESTAMP - SPEAKER:***"
        m = NEPTUNE_TURN_RE.search(block)
        if not m:
            continue
        ts_raw, speaker = m.group(1).strip(), m.group(2).strip()
        # Extract message body after header
        after = block[m.end() :].strip()
        # Drop “What do you do?” trailing prompt if present
        body = after
        if "What do you do?" in body:
            body = body.split("What do you do?")[0].rstrip()
        # Parse timestamp
        date_iso, time_display = None, ts_raw
        try:
            # Example format: 11/10/2025, 4:00:41 AM
            dt = datetime.strptime(ts_raw, "%m/%d/%Y, %I:%M:%S %p")
            date_iso = dt.date().isoformat()
        except Exception:
            pass
        turns.append(
            {
                "timestamp_raw": ts_raw,
                "date_iso": date_iso,
                "time_display": time_display,
                "speaker": speaker,
                "text": body.strip(),
            }
        )
    return {"title": title, "turns": turns}

def pair_turns_into_scenes(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scenes = []
    i = 0
    idx = 0
    while i < len(turns):
        # Pair User + next non-empty turn (often 'Venice'), else single
        t1 = turns[i]
        t2 = turns[i + 1] if i + 1 < len(turns) else None
        combined_text = t1["text"]
        speakers = [t1["speaker"]]
        date_iso = t1.get("date_iso")
        time_display = t1.get("time_display")
        if t2 and t2["text"]:
            combined_text = (t1["text"] + "\n\n" + t2["text"]).strip()
            speakers.append(t2["speaker"])
            # prefer earliest date
            if not date_iso:
                date_iso = t2.get("date_iso")
            if not time_display:
                time_display = t2.get("time_display")
            i += 2
        else:
            i += 1
        scenes.append(
            {
                "scene_index": idx,
                "date_iso": date_iso,
                "time_display": time_display,
                "speakers": speakers,
                "text": combined_text,
            }
        )
        idx += 1
    return scenes

def heuristic_characters_present(text: str) -> List[str]:
    found = set()
    for name in CHAR_NAME_CANDIDATES:
        if name in text:
            found.add(name)
    # Light proper-noun heuristic for additional names (filter noise)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
        cand = m.group(1)
        if len(cand) < 2:
            continue
        # skip common words that happen to be capitalized (You, Captain, Commander, etc.)
        if cand in {
            "You",
            "I",
            "We",
            "He",
            "She",
            "Captain",
            "Commander",
            "Lieutenant",
            "SRF",
            "ThunderChild",
            "User",
            "Venice",
        }:
            continue
        if len(cand.split()) == 1 and cand.lower() in {
            "the",
            "and",
            "It",
            "Its",
            "This",
        }:
            continue
        if len(cand) > 1:
            found.add(cand)
    return sorted(found)

def heuristic_location(text: str) -> str:
    low = text.lower()
    for k, v in LOCATION_HINTS.items():
        if k in low:
            return v
    return "unknown"

def heuristic_ships(text: str) -> List[str]:
    ships = []
    for s in KNOWN_SHIPS:
        if s in text:
            ships.append(s)
    # also detect SRV <Name> mentions
    for m in re.finditer(r"\bSRV\s+([A-Z][A-Za-z0-9_-]+)\b", text):
        ships.append(m.group(1))
    return sorted(set(ships))

def heuristic_events(text: str) -> List[str]:
    low = text.lower()
    events = set()
    for k, tag in EVENT_HINTS.items():
        if k in low:
            events.add(tag)
    return sorted(events)

def heuristic_tone_and_intensity(text: str) -> Dict[str, Any]:
    low = text.lower()
    exclaim = text.count("!")
    action_terms = sum(
        w in low
        for w in [
            "burn",
            "flip",
            "engage",
            "railgun",
            "combat",
            "attack",
            "orders",
            "slingshot",
        ]
    )
    emo_terms = sum(
        w in low
        for w in ["smile", "laugh", "humiliation", "shock", "joy", "relief", "fear"]
    )
    tone = (
        "tense"
        if action_terms > 1
        else ("emotional" if emo_terms > 1 else "neutral")
    )
    emotional_intensity = min(1.0, 0.1 * emo_terms + 0.05 * exclaim)
    action_level = min(1.0, 0.15 * action_terms)
    return {
        "tone": tone,
        "emotional_intensity": emotional_intensity,
        "action_level": action_level,
    }

def choose_pov(speakers: List[str], text: str) -> str:
    # If narrator uses “I ...” frequently and “you ...” appears in the response, assume User POV
    first_person = len(re.findall(r"\bI\b", text)) >= 2
    second_person = len(re.findall(r"\byou\b", text, flags=re.I)) >= 2
    if first_person and not second_person:
        return "User"
    if second_person and not first_person:
        return "Venice"
    # fallback to first speaker
    return speakers[0] if speakers else "UNKNOWN"

def ingest_neptune_export_to_parquet(
    export_path: str,
    parquet_out: str = "./thunderchild_scenes.parquet",
    append: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> pl.DataFrame:
    parsed = parse_neptune_export(export_path)
    turns = parsed["turns"]
    scenes = pair_turns_into_scenes(turns)

    # Build rows + embeddings
    model = SentenceTransformer(embedding_model)
    docs = [s["text"] for s in scenes]
    emb = model.encode(docs, batch_size=32, show_progress_bar=True)
    rows = []
    for s, e in zip(scenes, emb):
        chars = heuristic_characters_present(s["text"])
        loc = heuristic_location(s["text"])
        ships = heuristic_ships(s["text"])
        events = heuristic_events(s["text"])
        tone_bits = heuristic_tone_and_intensity(s["text"])
        pov = choose_pov(s["speakers"], s["text"])
        date_iso = s.get("date_iso") or "UNKNOWN"
        scene_id = f"scene_{s['scene_index']:04d}_{date_iso}"

        metadata = {
            "scene_id": scene_id,
            "time_display": s.get("time_display", "UNKNOWN"),
            "date_iso": date_iso,
            "pov_character": pov,
            "location": loc,
            "speakers": s["speakers"],
            "characters_present": json.dumps(chars),
            "ships": ships,
            "events": events,
            "tone": tone_bits["tone"],
            "emotional_intensity": tone_bits["emotional_intensity"],
            "action_level": tone_bits["action_level"],
            "plot_significance": 0.5,  # tune later
            "source_title": parsed.get("title"),
            "source_file": str(export_path),
        }
        rows.append(
            {
                "id": scene_id,
                "text": s["text"],
                "embedding": e.tolist(),
                "metadata": json.dumps(metadata),
            }
        )

    new_df = pl.DataFrame(rows)
    if append and Path(parquet_out).exists():
        old_df = pl.read_parquet(parquet_out)
        # de-dup by id
        existing = set(old_df["id"].to_list())
        new_df = new_df.filter(~pl.col("id").is_in(list(existing)))
        merged = pl.concat([old_df, new_df], how="vertical_relaxed")
        merged.write_parquet(parquet_out)
        return merged
    else:
        new_df.write_parquet(parquet_out)
        return new_df


def example_usage():
    """Example: Ingest Discord/Slack chat export"""

    ingester = ChatTranscriptIngester()

    # Example 1: JSON messages (Discord/Slack export)
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
    df = ingester.ingest_json_messages("example_chat.json")
    print("\n📋 Ingested DataFrame:")
    print(df.select(["id", "text"]).head())


if __name__ == "__main__":
    example_usage()
