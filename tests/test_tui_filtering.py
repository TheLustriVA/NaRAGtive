"""Tests for metadata filtering on search results.

Coverage includes:
- Location filtering with case-insensitive matching
- Date range filtering (start, end, both)
- Character filtering with JSON parsing
- Combined multi-filter scenarios
- Edge cases (empty results, invalid dates, etc.)
"""

import json
import pytest
from naragtive.tui.search_utils import apply_filters


@pytest.fixture
def sample_results():
    """Fixture providing sample search results."""
    return {
        "ids": ["1", "2", "3", "4", "5"],
        "documents": [
            "The Admiral commands the fleet",
            "A throne room conversation",
            "Journey through the mountain pass",
            "The Admiral meets the King",
            "A secret council meeting",
        ],
        "scores": [0.95, 0.87, 0.72, 0.91, 0.68],
        "metadatas": [
            {
                "scene_id": "scene-1",
                "location": "Command Bridge",
                "date_iso": "2024-01-15",
                "pov_character": "Admiral",
                "characters_present": json.dumps(["Admiral", "Ensign", "Captain"]),
            },
            {
                "scene_id": "scene-2",
                "location": "Throne Room",
                "date_iso": "2024-01-20",
                "pov_character": "King",
                "characters_present": json.dumps(["King", "Advisor", "Guard"]),
            },
            {
                "scene_id": "scene-3",
                "location": "Mountain Trail",
                "date_iso": "2024-02-01",
                "pov_character": "Scout",
                "characters_present": json.dumps(["Scout", "Merchant", "Soldier"]),
            },
            {
                "scene_id": "scene-4",
                "location": "Royal Chamber",
                "date_iso": "2024-02-10",
                "pov_character": "Admiral",
                "characters_present": json.dumps(["Admiral", "King", "Queen"]),
            },
            {
                "scene_id": "scene-5",
                "location": "Secret Passage",
                "date_iso": "2024-02-15",
                "pov_character": "Spy",
                "characters_present": json.dumps(["Spy", "Agent", "Handler"]),
            },
        ],
    }


class TestLocationFiltering:
    """Tests for location filtering."""

    def test_filter_by_location_exact_match(self, sample_results):
        """Test filtering by exact location match."""
        filtered = apply_filters(sample_results, location="Command Bridge")
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "1"
        assert "Command Bridge" in filtered["metadatas"][0]["location"]

    def test_filter_by_location_partial_match(self, sample_results):
        """Test filtering by partial location match."""
        filtered = apply_filters(sample_results, location="Room")
        assert len(filtered["ids"]) == 2
        assert "1" not in filtered["ids"]
        ids = set(filtered["ids"])
        assert ids == {"2", "4"}

    def test_filter_by_location_case_insensitive(self, sample_results):
        """Test location filtering is case-insensitive."""
        filtered = apply_filters(sample_results, location="throne")
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "2"

    def test_filter_by_location_no_matches(self, sample_results):
        """Test location filter with no matches."""
        filtered = apply_filters(sample_results, location="NonExistent")
        assert len(filtered["ids"]) == 0

    def test_filter_by_location_empty_string(self, sample_results):
        """Test location filter with empty string returns all."""
        filtered = apply_filters(sample_results, location="")
        assert len(filtered["ids"]) == len(sample_results["ids"])


class TestDateFiltering:
    """Tests for date range filtering."""

    def test_filter_by_date_start_only(self, sample_results):
        """Test filtering by start date only."""
        filtered = apply_filters(sample_results, date_start="2024-02-01")
        assert len(filtered["ids"]) == 3  # scenes 3, 4, 5
        assert all(meta["date_iso"] >= "2024-02-01" for meta in filtered["metadatas"])

    def test_filter_by_date_end_only(self, sample_results):
        """Test filtering by end date only."""
        filtered = apply_filters(sample_results, date_end="2024-02-01")
        assert len(filtered["ids"]) == 3  # scenes 1, 2, 3
        assert all(meta["date_iso"] <= "2024-02-01" for meta in filtered["metadatas"])

    def test_filter_by_date_range(self, sample_results):
        """Test filtering by date range (start and end)."""
        filtered = apply_filters(
            sample_results, date_start="2024-01-20", date_end="2024-02-10"
        )
        assert len(filtered["ids"]) == 3  # scenes 2, 3, 4
        for meta in filtered["metadatas"]:
            assert meta["date_iso"] >= "2024-01-20"
            assert meta["date_iso"] <= "2024-02-10"

    def test_filter_by_exact_date(self, sample_results):
        """Test filtering by exact date (start == end)."""
        filtered = apply_filters(
            sample_results, date_start="2024-02-01", date_end="2024-02-01"
        )
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "3"

    def test_filter_by_date_no_matches(self, sample_results):
        """Test date filter with no matches."""
        filtered = apply_filters(
            sample_results, date_start="2025-01-01", date_end="2025-12-31"
        )
        assert len(filtered["ids"]) == 0

    def test_filter_by_date_invalid_format(self, sample_results):
        """Test date filter with invalid format is skipped."""
        filtered = apply_filters(sample_results, date_start="invalid-date")
        # Invalid dates should be skipped, so all results returned
        assert len(filtered["ids"]) == len(sample_results["ids"])


class TestCharacterFiltering:
    """Tests for character filtering."""

    def test_filter_by_character_exact_match(self, sample_results):
        """Test filtering by character with exact match."""
        filtered = apply_filters(sample_results, character="Admiral")
        assert len(filtered["ids"]) == 2  # scenes 1 and 4
        ids = set(filtered["ids"])
        assert ids == {"1", "4"}

    def test_filter_by_character_case_insensitive(self, sample_results):
        """Test character filtering is case-insensitive."""
        filtered = apply_filters(sample_results, character="admiral")
        assert len(filtered["ids"]) == 2
        ids = set(filtered["ids"])
        assert ids == {"1", "4"}

    def test_filter_by_character_partial_match(self, sample_results):
        """Test character filtering with partial match."""
        filtered = apply_filters(sample_results, character="Gua")
        # Should match "Guard"
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "2"

    def test_filter_by_character_no_matches(self, sample_results):
        """Test character filter with no matches."""
        filtered = apply_filters(sample_results, character="NonExistent")
        assert len(filtered["ids"]) == 0

    def test_filter_by_character_with_malformed_json(self):
        """Test character filter handles malformed JSON gracefully."""
        results = {
            "ids": ["1", "2"],
            "documents": ["doc1", "doc2"],
            "scores": [0.9, 0.8],
            "metadatas": [
                {
                    "characters_present": "not-valid-json",
                },
                {
                    "characters_present": json.dumps(["Admiral"]),
                },
            ],
        }
        filtered = apply_filters(results, character="Admiral")
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "2"


class TestCombinedFiltering:
    """Tests for combining multiple filters."""

    def test_filter_location_and_date(self, sample_results):
        """Test combining location and date filters."""
        filtered = apply_filters(
            sample_results,
            location="Room",
            date_start="2024-02-01",
        )
        # Only "Royal Chamber" after 2024-02-01
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "4"

    def test_filter_location_and_character(self, sample_results):
        """Test combining location and character filters."""
        filtered = apply_filters(
            sample_results,
            location="Room",
            character="King",
        )
        # Locations with "Room" that have "King"
        # Throne Room (has King) and Royal Chamber (has King)
        assert len(filtered["ids"]) == 2
        ids = set(filtered["ids"])
        assert ids == {"2", "4"}

    def test_filter_all_dimensions(self, sample_results):
        """Test combining all filter dimensions."""
        filtered = apply_filters(
            sample_results,
            location="Room",
            date_start="2024-02-01",
            date_end="2024-02-15",
            character="King",
        )
        # Royal Chamber, Feb 1-15, has King
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "4"
        assert filtered["metadatas"][0]["location"] == "Royal Chamber"

    def test_filter_no_intersection(self, sample_results):
        """Test filters with no intersection."""
        filtered = apply_filters(
            sample_results,
            location="Command Bridge",  # scene 1
            character="King",  # not in scene 1
        )
        assert len(filtered["ids"]) == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_results(self):
        """Test filtering empty results."""
        empty_results = {"ids": [], "documents": [], "scores": [], "metadatas": []}
        filtered = apply_filters(empty_results, location="test")
        assert len(filtered["ids"]) == 0

    def test_none_results(self):
        """Test filtering None results."""
        filtered = apply_filters(None, location="test")
        assert len(filtered["ids"]) == 0

    def test_missing_metadata_fields(self):
        """Test filtering when metadata is incomplete."""
        results = {
            "ids": ["1", "2"],
            "documents": ["doc1", "doc2"],
            "scores": [0.9, 0.8],
            "metadatas": [
                {"scene_id": "1"},  # missing location, date_iso, etc.
                {
                    "location": "Room",
                    "date_iso": "2024-01-01",
                    "characters_present": json.dumps(["King"]),
                },
            ],
        }
        filtered = apply_filters(results, location="Room")
        assert len(filtered["ids"]) == 1
        assert filtered["ids"][0] == "2"

    def test_preserve_order(self, sample_results):
        """Test that filtering preserves result order."""
        filtered = apply_filters(sample_results, character="Admiral")
        # Results should be in original order (ids 1, 4)
        assert filtered["ids"] == ["1", "4"]
        assert filtered["scores"] == [sample_results["scores"][0], sample_results["scores"][3]]

    def test_all_filters_none(self, sample_results):
        """Test with all filters as None returns all results."""
        filtered = apply_filters(
            sample_results,
            location=None,
            date_start=None,
            date_end=None,
            character=None,
        )
        assert len(filtered["ids"]) == len(sample_results["ids"])
        assert filtered["ids"] == sample_results["ids"]
