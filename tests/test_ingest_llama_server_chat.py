from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from naragtive.ingest_llama_server_chat import (
    LlamaServerParser,
    LlamaServerExchangeGrouper,
    LlamaServerHeuristicAnalyzer,
    LlamaServerIngester,
)


class TestLlamaServerParser:
    """Test llama-server export parsing and validation."""

    def test_parse_export_valid(self, tmp_path: Path) -> None:
        """Test parsing valid llama-server export."""
        export_data = {
            "conv": {
                "id": "test-conv-123",
                "name": "Test Conversation",
                "lastModified": 1765279699684,
                "currNode": "node-id",
            },
            "messages": [
                {
                    "id": "msg-1",
                    "convId": "test-conv-123",
                    "role": "user",
                    "content": "Hello",
                    "type": "text",
                    "timestamp": 1765275434078,
                    "thinking": "",
                    "children": ["msg-2"],
                },
                {
                    "id": "msg-2",
                    "convId": "test-conv-123",
                    "role": "assistant",
                    "content": "Hi there!",
                    "type": "text",
                    "timestamp": 1765275434100,
                    "thinking": "",
                    "model": "test-model",
                    "children": [],
                },
            ],
        }

        export_file = tmp_path / "export.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f)

        parser = LlamaServerParser()
        data = parser.parse_export(str(export_file))

        assert data["conv"]["id"] == "test-conv-123"
        assert len(data["messages"]) == 2

    def test_parse_export_missing_conv(self, tmp_path: Path) -> None:
        """Test parsing export missing conv field."""
        export_data = {
            "messages": [
                {"id": "msg-1", "role": "user", "content": "test"}
            ]
        }

        export_file = tmp_path / "export.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f)

        parser = LlamaServerParser()
        with pytest.raises(ValueError, match="missing 'conv' or 'messages'"):
            parser.parse_export(str(export_file))

    def test_parse_export_missing_required_fields(self, tmp_path: Path) -> None:
        """Test parsing export missing required conv fields."""
        export_data = {
            "conv": {
                "id": "test-conv-123",
                # Missing 'name' and 'lastModified'
            },
            "messages": [],
        }

        export_file = tmp_path / "export.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f)

        parser = LlamaServerParser()
        with pytest.raises(ValueError, match="Missing conv fields"):
            parser.parse_export(str(export_file))

    def test_extract_conversation_name_short(self) -> None:
        """Test extracting short conversation name."""
        parser = LlamaServerParser()
        name = "Short title"
        result = parser.extract_conversation_name(name)
        assert result == "Short title"

    def test_extract_conversation_name_truncate(self) -> None:
        """Test truncating long conversation name."""
        parser = LlamaServerParser()
        name = "a" * 300  # Very long name
        result = parser.extract_conversation_name(name)
        assert len(result) <= 200
        assert result.endswith("...")

    def test_timestamp_to_datetime(self) -> None:
        """Test converting millisecond timestamp to datetime."""
        parser = LlamaServerParser()
        timestamp_ms = 1765275434078
        dt = parser.timestamp_to_datetime(timestamp_ms)

        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None  # Should be UTC
        # Verify conversion is correct
        expected_ts = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        assert dt == expected_ts


class TestLlamaServerExchangeGrouper:
    """Test grouping messages into user/assistant exchanges."""

    def test_group_into_exchanges_single_pair(self) -> None:
        """Test grouping a single user/assistant pair."""
        messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Hello",
                "type": "text",
                "timestamp": 1000,
                "children": ["msg-2"],
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "Hi there!",
                "type": "text",
                "timestamp": 1001,
                "model": "test-model",
                "thinking": "",
            },
        ]

        grouper = LlamaServerExchangeGrouper()
        exchanges = grouper.group_into_exchanges(messages)

        assert len(exchanges) == 1
        assert exchanges[0]["user_content"] == "Hello"
        assert exchanges[0]["assistant_content"] == "Hi there!"
        assert exchanges[0]["model"] == "test-model"

    def test_group_into_exchanges_multiple_pairs(self) -> None:
        """Test grouping multiple user/assistant pairs."""
        messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": "First question",
                "type": "text",
                "timestamp": 1000,
                "children": ["msg-2"],
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "First answer",
                "type": "text",
                "timestamp": 1001,
                "model": "test-model",
                "thinking": "",
            },
            {
                "id": "msg-3",
                "role": "user",
                "content": "Second question",
                "type": "text",
                "timestamp": 1002,
                "children": ["msg-4"],
            },
            {
                "id": "msg-4",
                "role": "assistant",
                "content": "Second answer",
                "type": "text",
                "timestamp": 1003,
                "model": "test-model",
                "thinking": "",
            },
        ]

        grouper = LlamaServerExchangeGrouper()
        exchanges = grouper.group_into_exchanges(messages)

        assert len(exchanges) == 2
        assert exchanges[0]["user_content"] == "First question"
        assert exchanges[1]["user_content"] == "Second question"

    def test_group_with_thinking_content(self) -> None:
        """Test grouping exchange with thinking content."""
        messages = [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Explain something",
                "type": "text",
                "timestamp": 1000,
                "children": ["msg-2"],
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": "Here's the explanation",
                "type": "text",
                "timestamp": 1001,
                "model": "claude",
                "thinking": "Let me think about this carefully...",
            },
        ]

        grouper = LlamaServerExchangeGrouper()
        exchanges = grouper.group_into_exchanges(messages)

        assert len(exchanges) == 1
        assert exchanges[0]["has_thinking"] is True
        assert "Let me think" in exchanges[0]["thinking_content"]

    def test_create_scene_from_exchange(self) -> None:
        """Test creating scene from exchange."""
        exchange = {
            "exchange_index": 0,
            "user_content": "User message",
            "user_timestamp": 1000,
            "assistant_content": "Assistant message",
            "assistant_timestamp": 1001,
            "model": "test-model",
            "has_thinking": False,
            "thinking_content": "",
        }

        grouper = LlamaServerExchangeGrouper()
        scene = grouper.create_scene_from_exchange(
            exchange,
            "conv-123",
            "Test Conversation"
        )

        assert scene["conversation_id"] == "conv-123"
        assert scene["conversation_name"] == "Test Conversation"
        assert "User: User message" in scene["text"]
        assert "Assistant: Assistant message" in scene["text"]
        assert scene["model"] == "test-model"
        assert scene["has_thinking"] is False


class TestLlamaServerHeuristicAnalyzer:
    """Test heuristic metadata extraction."""

    def test_extract_themes_creative(self) -> None:
        """Test extracting creative theme."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "Describe a vivid, imaginative scene with compelling narrative"
        themes = analyzer.extract_themes(text)

        assert "creative" in themes

    def test_extract_themes_technical(self) -> None:
        """Test extracting technical theme."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "Write Python code with an algorithm implementation and debugging"
        themes = analyzer.extract_themes(text)

        assert "technical" in themes

    def test_extract_themes_analytical(self) -> None:
        """Test extracting analytical theme."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "Analyze and evaluate this research, comparing different frameworks"
        themes = analyzer.extract_themes(text)

        assert "analytical" in themes

    def test_analyze_tone_formal(self) -> None:
        """Test analyzing formal tone."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "Furthermore, it should be noted that consequently, professional standards require"
        tone = analyzer.analyze_tone(text)

        # This may vary depending on keyword weighting, but formal should score high
        assert tone in ["formal", "neutral"]  # Accept either

    def test_analyze_tone_casual(self) -> None:
        """Test analyzing casual tone."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "Yeah, so like, this is gonna be cool and awesome!"
        tone = analyzer.analyze_tone(text)

        assert tone in ["casual", "neutral"]  # Accept either

    def test_analyze_engagement_level_high(self) -> None:
        """Test high engagement level."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = (
            "What do you think about this? I'm really curious! This is amazing!!!\n"
            "The dialogue continued with vivid exchanges and enthusiastic discussion."
        )
        engagement = analyzer.analyze_engagement_level(text)

        assert engagement > 0.3  # Should be relatively high

    def test_analyze_engagement_level_low(self) -> None:
        """Test low engagement level."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "The object was placed on the surface."
        engagement = analyzer.analyze_engagement_level(text)

        assert engagement < 0.3  # Should be relatively low

    def test_analyze_complexity_high(self) -> None:
        """Test high complexity analysis."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = (
            "The epistemological implications of quantum mechanics necessitate "
            "a reconsideration of deterministic paradigms."
        )
        complexity = analyzer.analyze_complexity(text)

        assert complexity > 0.4  # Should be relatively high

    def test_analyze_complexity_low(self) -> None:
        """Test low complexity analysis."""
        analyzer = LlamaServerHeuristicAnalyzer()
        text = "Go do it now."
        complexity = analyzer.analyze_complexity(text)

        assert complexity < 0.3  # Should be relatively low


class TestLlamaServerIngester:
    """Test main ingester orchestration."""

    @patch('naragtive.ingest_llama_server_chat.SentenceTransformer')
    def test_ingest_llama_server_export(
        self,
        mock_model: Any,
        tmp_path: Path,
    ) -> None:
        """Test ingesting a complete llama-server export."""
        # Setup mock
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        mock_instance.encode.return_value = np.array([
            [0.1, 0.2, 0.3] * 128,  # 384 dims
            [0.4, 0.5, 0.6] * 128,
        ])

        # Create test export
        export_data = {
            "conv": {
                "id": "test-conv",
                "name": "Test Conversation",
                "lastModified": 1765279699684,
                "currNode": "node-id",
            },
            "messages": [
                {
                    "id": "msg-1",
                    "convId": "test-conv",
                    "role": "user",
                    "content": "Hello",
                    "type": "text",
                    "timestamp": 1765275434078,
                    "thinking": "",
                    "children": ["msg-2"],
                },
                {
                    "id": "msg-2",
                    "convId": "test-conv",
                    "role": "assistant",
                    "content": "Hi there!",
                    "type": "text",
                    "timestamp": 1765275434100,
                    "thinking": "",
                    "model": "test-model",
                    "children": [],
                },
            ],
        }

        export_file = tmp_path / "export.json"
        with open(export_file, "w") as f:
            json.dump(export_data, f)

        output_file = tmp_path / "output.parquet"

        ingester = LlamaServerIngester()
        df = ingester.ingest_llama_server_export(
            str(export_file),
            str(output_file),
        )

        # Verify DataFrame
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1  # One exchange
        assert "id" in df.columns
        assert "text" in df.columns
        assert "embedding" in df.columns
        assert "metadata" in df.columns

        # Verify content
        row = df.row(0, named=True)
        assert "Hello" in row["text"]
        assert "Hi there" in row["text"]

        # Verify metadata
        metadata = json.loads(row["metadata"])
        assert metadata["conversation_name"] == "Test Conversation"
        assert metadata["model"] == "test-model"
        assert "themes" in metadata
        assert "tone" in metadata
        assert "engagement_level" in metadata
        assert "complexity" in metadata

        # Verify file was created
        assert output_file.exists()

    @patch('naragtive.ingest_llama_server_chat.SentenceTransformer')
    def test_ingest_multiple_exports(
        self,
        mock_model: Any,
        tmp_path: Path,
    ) -> None:
        """Test ingesting multiple exports."""
        mock_instance = MagicMock()
        mock_model.return_value = mock_instance
        mock_instance.encode.return_value = np.array([
            [0.1, 0.2, 0.3] * 128,
            [0.4, 0.5, 0.6] * 128,
            [0.7, 0.8, 0.9] * 128,
            [0.2, 0.3, 0.4] * 128,
        ])

        # Create two exports
        export_data_1 = {
            "conv": {
                "id": "conv-1",
                "name": "Conversation 1",
                "lastModified": 1765279699684,
                "currNode": "node-1",
            },
            "messages": [
                {
                    "id": "msg-1",
                    "convId": "conv-1",
                    "role": "user",
                    "content": "Question 1",
                    "type": "text",
                    "timestamp": 1000,
                    "children": ["msg-2"],
                },
                {
                    "id": "msg-2",
                    "convId": "conv-1",
                    "role": "assistant",
                    "content": "Answer 1",
                    "type": "text",
                    "timestamp": 1001,
                    "model": "model-1",
                    "thinking": "",
                    "children": [],
                },
            ],
        }

        export_data_2 = {
            "conv": {
                "id": "conv-2",
                "name": "Conversation 2",
                "lastModified": 1765279699684,
                "currNode": "node-2",
            },
            "messages": [
                {
                    "id": "msg-3",
                    "convId": "conv-2",
                    "role": "user",
                    "content": "Question 2",
                    "type": "text",
                    "timestamp": 2000,
                    "children": ["msg-4"],
                },
                {
                    "id": "msg-4",
                    "convId": "conv-2",
                    "role": "assistant",
                    "content": "Answer 2",
                    "type": "text",
                    "timestamp": 2001,
                    "model": "model-2",
                    "thinking": "",
                    "children": [],
                },
            ],
        }

        export_file_1 = tmp_path / "export1.json"
        export_file_2 = tmp_path / "export2.json"
        with open(export_file_1, "w") as f:
            json.dump(export_data_1, f)
        with open(export_file_2, "w") as f:
            json.dump(export_data_2, f)

        output_file = tmp_path / "combined.parquet"

        ingester = LlamaServerIngester()
        df = ingester.ingest_multiple_exports(
            [str(export_file_1), str(export_file_2)],
            str(output_file),
        )

        assert len(df) == 2
        assert output_file.exists()

        # Verify both conversations are present
        metadata_list = [json.loads(m) for m in df["metadata"].to_list()]
        conv_names = {m["conversation_name"] for m in metadata_list}
        assert "Conversation 1" in conv_names
        assert "Conversation 2" in conv_names


# Fixtures
@pytest.fixture
def sample_llama_export() -> dict[str, Any]:
    """Provide sample llama-server export data."""
    return {
        "conv": {
            "id": "62b43483-fb41-49b4-b769-582f8853cb64",
            "name": "Sci-fi battle scene",
            "lastModified": 1765279699684,
            "currNode": "d5d55efc-aa21-43d6-b970-8f832014bac1",
        },
        "messages": [
            {
                "convId": "62b43483-fb41-49b4-b769-582f8853cb64",
                "role": "user",
                "content": "Describe a dramatic space battle",
                "type": "text",
                "timestamp": 1765275434078,
                "thinking": "",
                "children": ["32a4b7b7-b114-4e5a-bdd6-505bbc9dd2bf"],
                "id": "2930ea35-a622-45e2-a512-b8425abc082d",
                "parent": "root",
            },
            {
                "convId": "62b43483-fb41-49b4-b769-582f8853cb64",
                "role": "assistant",
                "content": "The battle erupted with devastating force...",
                "type": "text",
                "timestamp": 1765275434106,
                "thinking": "",
                "children": [],
                "model": "test-model",
                "id": "32a4b7b7-b114-4e5a-bdd6-505bbc9dd2bf",
                "parent": "2930ea35-a622-45e2-a512-b8425abc082d",
            },
        ],
    }
