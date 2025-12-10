from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch, MagicMock

import pytest
import polars as pl

from naragtive.ingest_chat_transcripts import (
    NeptuneParser,
    SceneProcessor,
    HeuristicAnalyzer,
    ChatTranscriptIngester,
    NeptuneIngester,
)


class TestNeptuneParser:
    """Test Neptune export parsing."""
    
    def test_parse_timestamp_valid(self) -> None:
        """Test parsing valid timestamp."""
        parser = NeptuneParser()
        date_iso, time_display = parser._parse_timestamp("11/10/2025, 4:00:41 AM")
        
        assert date_iso == "2025-11-10"
        assert time_display == "11/10/2025, 4:00:41 AM"
    
    def test_parse_timestamp_invalid(self) -> None:
        """Test parsing invalid timestamp."""
        parser = NeptuneParser()
        date_iso, time_display = parser._parse_timestamp("invalid")
        
        assert date_iso is None
        assert time_display == "invalid"
    
    def test_extract_title(self) -> None:
        """Test extracting conversation title."""
        parser = NeptuneParser()
        text = "# Conversation: ThunderChild Mission\n\n***timestamp***"
        title = parser._extract_title(text)
        
        assert title == "ThunderChild Mission"
    
    def test_parse_turn_block_valid(self) -> None:
        """Test parsing valid turn block."""
        parser = NeptuneParser()
        block = "***11/10/2025, 4:00:41 AM - Venice:***\nSome dialogue text here"
        turn = parser._parse_turn_block(block)
        
        assert turn is not None
        assert turn["speaker"] == "Venice"
        assert "dialogue text" in turn["text"]
        assert turn["date_iso"] == "2025-11-10"
    
    def test_parse_turn_block_with_prompt_removed(self) -> None:
        """Test that 'What do you do?' prompt is removed."""
        parser = NeptuneParser()
        block = "***11/10/2025, 4:00:41 AM - Venice:***\nDialogue text\n\nWhat do you do?"
        turn = parser._parse_turn_block(block)
        
        assert turn is not None
        assert "What do you do?" not in turn["text"]
        assert "Dialogue text" in turn["text"]


class TestSceneProcessor:
    """Test scene pairing logic."""
    
    def test_pair_single_turn(self, sample_turn_list: list[dict[str, Any]]) -> None:
        """Test pairing when only one turn available."""
        processor = SceneProcessor()
        turns = [sample_turn_list[0]]
        scenes = processor.pair_turns_into_scenes(turns)
        
        assert len(scenes) == 1
        assert scenes[0]["paired"] is False
        assert scenes[0]["speakers"] == ["Venice"]
    
    def test_pair_consecutive_turns(self, sample_turn_list: list[dict[str, Any]]) -> None:
        """Test pairing consecutive turns."""
        processor = SceneProcessor()
        turns = sample_turn_list[:2]
        scenes = processor.pair_turns_into_scenes(turns)
        
        assert len(scenes) == 1
        assert scenes[0]["paired"] is True
        assert len(scenes[0]["speakers"]) == 2
        assert "Venice" in scenes[0]["speakers"]
        assert "Admiral Zelenskyy" in scenes[0]["speakers"]


class TestHeuristicAnalyzer:
    """Test metadata extraction."""
    
    def test_extract_characters_known(self) -> None:
        """Test extracting known character names."""
        analyzer = HeuristicAnalyzer()
        text = "Venice moved to the side. Heidi checked the readout."
        characters = analyzer._extract_characters(text)
        
        # Note: Venice is not in the default character list, only Heidi is
        assert "Heidi" in characters
    
    def test_extract_location_bridge(self) -> None:
        """Test extracting bridge location."""
        analyzer = HeuristicAnalyzer()
        text = "The Admiral stepped onto the bridge, commanding the crew."
        location = analyzer._extract_location(text)
        
        assert location == "bridge"
    
    def test_extract_location_medbay(self) -> None:
        """Test extracting medbay location."""
        analyzer = HeuristicAnalyzer()
        text = "In the medbay, Dr. Rizzo ran tests."
        location = analyzer._extract_location(text)
        
        assert location == "medbay"
    
    def test_extract_ships(self) -> None:
        """Test extracting ship names."""
        analyzer = HeuristicAnalyzer()
        text = "The ThunderChild engaged its engines. Stonewall followed behind."
        ships = analyzer._extract_ships(text)
        
        assert "ThunderChild" in ships
        assert "Stonewall" in ships
    
    def test_extract_events(self) -> None:
        """Test extracting events."""
        analyzer = HeuristicAnalyzer()
        text = "We began the undocking sequence. The engine burn was controlled."
        events = analyzer._extract_events(text)
        
        assert "undocking" in events
        assert "engine_burn" in events
    
    def test_analyze_tone_tense(self) -> None:
        """Test tone analysis - tense."""
        analyzer = HeuristicAnalyzer()
        text = "Combat engage! Railgun systems online. Attack formation!"
        tone = analyzer._analyze_tone(text)
        
        assert tone == "tense"
    
    def test_analyze_tone_emotional(self) -> None:
        """Test tone analysis - emotional."""
        analyzer = HeuristicAnalyzer()
        text = "She smiled with joy. The relief was evident. Fear melted away."
        tone = analyzer._analyze_tone(text)
        
        assert tone == "emotional"
    
    def test_analyze_emotional_intensity(self) -> None:
        """Test emotional intensity scoring."""
        analyzer = HeuristicAnalyzer()
        text_high = "Joy!! Relief!! Happiness!!"
        intensity_high = analyzer._analyze_emotional_intensity(text_high)
        
        text_low = "The object was placed on the surface."
        intensity_low = analyzer._analyze_emotional_intensity(text_low)
        
        assert intensity_high > intensity_low
    
    def test_analyze_action_level(self) -> None:
        """Test action level scoring."""
        analyzer = HeuristicAnalyzer()
        text_high = "Engage combat! Attack! Orders confirmed! Railgun fire!"
        action_high = analyzer._analyze_action_level(text_high)
        
        text_low = "They sat quietly in the room."
        action_low = analyzer._analyze_action_level(text_low)
        
        assert action_high > action_low


class TestChatTranscriptIngester:
    """Test generic chat ingestion."""
    
    @patch('naragtive.ingest_chat_transcripts.SentenceTransformer')
    def test_ingest_json_creates_dataframe(
        self,
        mock_model: Mock,
        sample_chat_json: str,
        tmp_path: Path,
    ) -> None:
        """Test that JSON ingestion creates valid DataFrame."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        mock_instance.encode.return_value = []
        
        json_file = tmp_path / "chat.json"
        json_file.write_text(sample_chat_json)
        
        ingester = ChatTranscriptIngester()
        output_path = tmp_path / "output.parquet"
        df = ingester.ingest_json_messages(str(json_file), parquet_output=str(output_path))
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert "id" in df.columns
        assert "text" in df.columns
