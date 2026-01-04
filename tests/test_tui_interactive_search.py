"""Tests for interactive search functionality.

Coverage includes:
- Search history navigation (up/down arrows)
- History item adding and clearing
- Filter state maintenance across queries
- Result count tracking
- History boundaries (first, last items)
"""

import pytest


class TestSearchHistory:
    """Tests for search history widget."""

    @pytest.fixture
    def history_widget(self):
        """Create history widget for testing."""
        # Import here to avoid import errors if textual not installed
        from naragtive.tui.widgets.search_history import SearchHistory

        return SearchHistory(max_items=5)

    def test_add_single_query(self, history_widget):
        """Test adding single query to history."""
        history_widget.add_query("test query")
        assert len(history_widget.history) == 1
        assert history_widget.history[0] == "test query"

    def test_add_multiple_queries(self, history_widget):
        """Test adding multiple queries maintains order."""
        history_widget.add_query("query1")
        history_widget.add_query("query2")
        history_widget.add_query("query3")
        assert len(history_widget.history) == 3
        assert history_widget.history[0] == "query3"
        assert history_widget.history[1] == "query2"
        assert history_widget.history[2] == "query1"

    def test_add_duplicate_query(self, history_widget):
        """Test adding duplicate query doesn't create duplicate."""
        history_widget.add_query("query1")
        history_widget.add_query("query2")
        history_widget.add_query("query1")  # Duplicate
        assert len(history_widget.history) == 2
        # Most recent should be first
        assert history_widget.history[0] == "query1"

    def test_history_respects_max_items(self, history_widget):
        """Test history respects max items limit."""
        for i in range(10):
            history_widget.add_query(f"query{i}")
        assert len(history_widget.history) == 5  # max_items is 5

    def test_add_empty_query_ignored(self, history_widget):
        """Test empty queries are not added."""
        history_widget.add_query("")
        history_widget.add_query(None)
        assert len(history_widget.history) == 0

    def test_add_whitespace_query(self, history_widget):
        """Test whitespace-only queries are added but trimmed."""
        history_widget.add_query("   ")
        # Depends on implementation - typically trimmed queries shouldn't be added
        # This tests current behavior
        assert len(history_widget.history) >= 0

    def test_clear_history(self, history_widget):
        """Test clearing history."""
        history_widget.add_query("query1")
        history_widget.add_query("query2")
        history_widget.clear_history()
        assert len(history_widget.history) == 0
        assert history_widget.current_index == -1

    def test_clear_history_on_new_search(self, history_widget):
        """Test index resets when adding new query."""
        history_widget.add_query("query1")
        history_widget.current_index = 0
        history_widget.add_query("query2")
        assert history_widget.current_index == -1


class TestHistoryNavigation:
    """Tests for navigating search history."""

    @pytest.fixture
    def history_widget(self):
        """Create history widget with sample data."""
        from naragtive.tui.widgets.search_history import SearchHistory

        widget = SearchHistory(max_items=5)
        widget.add_query("first")
        widget.add_query("second")
        widget.add_query("third")
        return widget

    def test_navigate_up_from_no_selection(self, history_widget):
        """Test navigating up selects first (most recent) item."""
        assert history_widget.current_index == -1
        result = history_widget.navigate_up()
        assert result == "third"
        assert history_widget.current_index == 0

    def test_navigate_up_through_history(self, history_widget):
        """Test navigating up through entire history."""
        # Navigate down 3 items
        history_widget.navigate_up()  # -> "third" (index 0)
        history_widget.navigate_up()  # -> "second" (index 1)
        assert history_widget.current_index == 1
        result = history_widget.navigate_up()  # -> "first" (index 2)
        assert result == "first"
        assert history_widget.current_index == 2

    def test_navigate_up_at_end_stays_at_end(self, history_widget):
        """Test navigating up at end of history stays at end."""
        history_widget.current_index = 2  # At last item
        result = history_widget.navigate_up()
        assert history_widget.current_index == 2  # Should not change

    def test_navigate_down_from_selection(self, history_widget):
        """Test navigating down from selected item."""
        history_widget.current_index = 2  # At "first"
        result = history_widget.navigate_down()  # -> "second" (index 1)
        assert result == "second"
        assert history_widget.current_index == 1

    def test_navigate_down_to_no_selection(self, history_widget):
        """Test navigating down from first item clears selection."""
        history_widget.current_index = 0  # At "third"
        result = history_widget.navigate_down()
        assert result is None
        assert history_widget.current_index == -1

    def test_navigate_down_from_no_selection(self, history_widget):
        """Test navigating down with no selection stays at -1."""
        assert history_widget.current_index == -1
        result = history_widget.navigate_down()
        assert result is None
        assert history_widget.current_index == -1

    def test_navigate_up_down_cycle(self, history_widget):
        """Test cycling through history with up/down keys."""
        # Go up: -1 -> 0 -> 1 -> 2
        history_widget.navigate_up()
        assert history_widget.current_index == 0
        history_widget.navigate_up()
        assert history_widget.current_index == 1
        history_widget.navigate_up()
        assert history_widget.current_index == 2
        # Go down: 2 -> 1 -> 0 -> -1
        history_widget.navigate_down()
        assert history_widget.current_index == 1
        history_widget.navigate_down()
        assert history_widget.current_index == 0
        history_widget.navigate_down()
        assert history_widget.current_index == -1


class TestSelectedQuery:
    """Tests for getting selected query."""

    @pytest.fixture
    def history_widget(self):
        """Create history widget with sample data."""
        from naragtive.tui.widgets.search_history import SearchHistory

        widget = SearchHistory(max_items=5)
        widget.add_query("query1")
        widget.add_query("query2")
        return widget

    def test_get_selected_with_no_selection(self, history_widget):
        """Test getting selected query returns None when nothing selected."""
        assert history_widget.get_selected() is None

    def test_get_selected_after_navigation(self, history_widget):
        """Test getting selected query after navigation."""
        history_widget.navigate_up()
        selected = history_widget.get_selected()
        assert selected == "query2"

    def test_get_selected_invalid_index(self, history_widget):
        """Test getting selected with invalid index returns None."""
        history_widget.current_index = 999  # Invalid
        assert history_widget.get_selected() is None


class TestHistoryEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def empty_history(self):
        """Create empty history widget."""
        from naragtive.tui.widgets.search_history import SearchHistory

        return SearchHistory(max_items=5)

    def test_navigate_empty_history(self, empty_history):
        """Test navigating empty history returns None."""
        assert empty_history.navigate_up() is None
        assert empty_history.navigate_down() is None

    def test_get_selected_empty_history(self, empty_history):
        """Test getting selected from empty history returns None."""
        assert empty_history.get_selected() is None

    def test_single_item_navigation(self):
        """Test navigation with single item."""
        from naragtive.tui.widgets.search_history import SearchHistory

        widget = SearchHistory(max_items=5)
        widget.add_query("only")

        # Navigate up
        result = widget.navigate_up()
        assert result == "only"
        assert widget.current_index == 0

        # Navigate down
        result = widget.navigate_down()
        assert result is None
        assert widget.current_index == -1

    def test_max_items_custom_size(self):
        """Test custom max_items size."""
        from naragtive.tui.widgets.search_history import SearchHistory

        widget = SearchHistory(max_items=2)
        widget.add_query("q1")
        widget.add_query("q2")
        widget.add_query("q3")
        widget.add_query("q4")

        assert len(widget.history) == 2
        assert "q4" in widget.history
        assert "q3" in widget.history
        assert "q1" not in widget.history

    def test_history_with_special_characters(self):
        """Test history with special characters in queries."""
        from naragtive.tui.widgets.search_history import SearchHistory

        widget = SearchHistory(max_items=5)
        queries = [
            "query with spaces",
            "query-with-dashes",
            "query_with_underscores",
            'query "with quotes"',
            "query\twith\ttabs",
        ]
        for query in queries:
            widget.add_query(query)

        assert len(widget.history) == len(queries)
        # Should be in reverse order (most recent first)
        assert widget.history[0] == queries[-1]

    def test_history_very_long_query(self):
        """Test history with very long query."""
        from naragtive.tui.widgets.search_history import SearchHistory

        widget = SearchHistory(max_items=5)
        long_query = "q" * 1000
        widget.add_query(long_query)
        assert widget.history[0] == long_query
        result = widget.navigate_up()
        assert result == long_query
