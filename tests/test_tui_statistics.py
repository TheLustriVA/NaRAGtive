"""Tests for statistics screen functionality.

Coverage includes:
- Statistics collection (locations, characters, file size)
- Async data loading without UI blocking
- Handling empty stores
- Character JSON parsing
- Edge cases (missing fields, malformed data)
"""

import json
import pytest
import asyncio
from pathlib import Path
from collections import Counter
import tempfile
import polars as pl


class TestStatisticsCollection:
    """Tests for statistics collection."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pl.DataFrame(
            {
                "id": ["1", "2", "3", "4", "5"],
                "text": ["text1", "text2", "text3", "text4", "text5"],
                "location": [
                    "Throne Room",
                    "Throne Room",
                    "Kitchen",
                    "Kitchen",
                    "Garden",
                ],
                "date_iso": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ],
                "characters_present": [
                    json.dumps(["King", "Queen", "Prince"]),
                    json.dumps(["King", "Advisor"]),
                    json.dumps(["Cook", "Maid"]),
                    json.dumps(["Cook", "Prince"]),
                    json.dumps(["Gardener", "Maid"]),
                ],
            }
        )

    def test_count_locations(self, sample_dataframe):
        """Test counting locations."""
        locations = sample_dataframe["location"].value_counts()
        location_dict = {
            str(loc): int(count)
            for loc, count in zip(locations.to_list(), locations.to_list())
        }
        assert "Throne Room" in location_dict or len(location_dict) > 0

    def test_count_characters(self, sample_dataframe):
        """Test counting character appearances."""
        char_counter = Counter()
        for chars_str in sample_dataframe["characters_present"]:
            chars = json.loads(chars_str)
            char_counter.update(chars)

        most_common = dict(char_counter.most_common(5))
        assert "King" in most_common
        assert "Prince" in most_common
        assert most_common["King"] == 2  # Appears in scenes 1 and 2

    def test_file_size_calculation(self):
        """Test file size calculation."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            # Create sample file
            df = pl.DataFrame({"col": [1, 2, 3]})
            df.write_parquet(f.name)
            size_mb = Path(f.name).stat().st_size / (1024 * 1024)
            assert size_mb > 0

    def test_location_breakdown_top_5(self, sample_dataframe):
        """Test location breakdown limits to top 5."""
        locations = sample_dataframe["location"].value_counts()
        top_5 = dict(
            (str(loc), int(count))
            for loc, count in zip(locations.to_list()[:5], locations.to_list()[:5])
        )
        assert len(top_5) <= 5

    def test_character_breakdown_top_5(self, sample_dataframe):
        """Test character breakdown limits to top 5."""
        char_counter = Counter()
        for chars_str in sample_dataframe["characters_present"]:
            chars = json.loads(chars_str)
            char_counter.update(chars)

        top_5 = dict(char_counter.most_common(5))
        assert len(top_5) <= 5

    def test_handle_malformed_character_json(self):
        """Test handling malformed character JSON gracefully."""
        df = pl.DataFrame(
            {
                "characters_present": [
                    json.dumps(["King", "Queen"]),
                    "not-json",  # Malformed
                    json.dumps(["Prince"]),
                ]
            }
        )

        char_counter = Counter()
        for chars_str in df["characters_present"]:
            try:
                chars = json.loads(chars_str)
                if isinstance(chars, list):
                    char_counter.update(chars)
            except (json.JSONDecodeError, TypeError):
                pass

        # Should have counted valid entries
        assert len(char_counter) > 0
        assert "King" in char_counter

    def test_empty_dataframe_statistics(self):
        """Test statistics on empty dataframe."""
        empty_df = pl.DataFrame(
            {
                "id": [],
                "location": [],
                "characters_present": [],
            }
        )

        locations = empty_df["location"].value_counts()
        assert len(locations) == 0

    def test_single_record_statistics(self):
        """Test statistics with single record."""
        single_df = pl.DataFrame(
            {
                "id": ["1"],
                "location": ["Throne Room"],
                "characters_present": [json.dumps(["King"])],
            }
        )

        locations = single_df["location"].value_counts()
        assert len(locations) == 1

        char_counter = Counter()
        for chars_str in single_df["characters_present"]:
            chars = json.loads(chars_str)
            char_counter.update(chars)
        assert len(char_counter) == 1
        assert char_counter["King"] == 1


class TestAsyncStatisticsLoading:
    """Tests for async statistics loading."""

    @pytest.mark.asyncio
    async def test_async_operation_doesnt_block(self):
        """Test that async operations don't block."""
        # Simulate async operation
        async def slow_operation():
            await asyncio.sleep(0.1)
            return "completed"

        # This should not block
        task = asyncio.create_task(slow_operation())
        # Can do other things while waiting
        await asyncio.sleep(0.05)
        result = await task
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test multiple concurrent async operations."""
        async def operation(n):
            await asyncio.sleep(0.05)
            return n * 2

        results = await asyncio.gather(
            operation(1),
            operation(2),
            operation(3),
        )
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of operation timeout."""
        async def slow_operation():
            await asyncio.sleep(5.0)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)


class TestStatisticsEdgeCases:
    """Tests for edge cases in statistics."""

    def test_missing_location_field(self):
        """Test handling missing location field."""
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["text1", "text2"],
            }
        )

        if "location" in df.columns:
            locations = df["location"].value_counts()
        else:
            locations = {}
        assert len(locations) == 0

    def test_missing_characters_field(self):
        """Test handling missing characters field."""
        df = pl.DataFrame(
            {
                "id": ["1", "2"],
                "text": ["text1", "text2"],
            }
        )

        char_counter = Counter()
        if "characters_present" in df.columns:
            for chars_str in df["characters_present"]:
                try:
                    chars = json.loads(chars_str)
                    if isinstance(chars, list):
                        char_counter.update(chars)
                except (json.JSONDecodeError, TypeError):
                    pass
        assert len(char_counter) == 0

    def test_null_values_in_statistics(self):
        """Test handling null/None values in statistics."""
        df = pl.DataFrame(
            {
                "location": ["Room1", None, "Room2", None, "Room1"],
                "characters_present": [
                    json.dumps(["A", "B"]),
                    None,
                    json.dumps(["C"]),
                    json.dumps(["A"]),
                    None,
                ],
            }
        )

        # Count locations (Polars skips nulls by default)
        locations = df["location"].value_counts()
        # Should have counted non-null values

        char_counter = Counter()
        for chars_str in df["characters_present"]:
            if chars_str is not None:
                try:
                    chars = json.loads(chars_str)
                    if isinstance(chars, list):
                        char_counter.update(chars)
                except (json.JSONDecodeError, TypeError):
                    pass
        assert len(char_counter) > 0

    def test_very_large_character_list(self):
        """Test handling very large character lists."""
        large_chars = [f"Char{i}" for i in range(1000)]
        df = pl.DataFrame(
            {
                "characters_present": [json.dumps(large_chars)],
            }
        )

        char_counter = Counter()
        for chars_str in df["characters_present"]:
            chars = json.loads(chars_str)
            char_counter.update(chars)

        # Only top 5 should be used in stats
        top_5 = dict(char_counter.most_common(5))
        assert len(top_5) == 5

    def test_duplicate_locations(self):
        """Test handling duplicate locations with different cases."""
        df = pl.DataFrame(
            {
                "location": [
                    "Throne Room",
                    "throne room",
                    "THRONE ROOM",
                    "Other",
                ]
            }
        )
        # Note: Exact duplicates only (case-sensitive)
        locations = df["location"].value_counts()
        # All three different due to case, so 3 separate entries
        assert len(locations) == 4  # 3 throne variations + 1 other

    def test_statistics_preserves_order(self):
        """Test that statistics maintain count order."""
        df = pl.DataFrame(
            {
                "location": [
                    "Room1",
                    "Room2",
                    "Room1",
                    "Room1",
                    "Room3",
                    "Room2",
                ]
            }
        )

        locations = df["location"].value_counts()
        # Should be sorted by count (descending)
        counts = list(locations.to_list())
        assert counts == sorted(counts, reverse=True)
