"""Filter panel widget for NaRAGtive TUI.

Provides UI for filtering search results by location, date range, and character.
Emits FilterChanged message when filters are modified.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widgets import Input, Label, Static
from textual.message import Message


class FilterPanel(Static):
    """Panel for filtering search results.

    Allows filtering by:
    - Location (partial match, case-insensitive)
    - Date range (ISO format YYYY-MM-DD)
    - Character name (case-insensitive)

    Emits FilterChanged message when filters update.

    Attributes:
        location: Current location filter
        date_start: Current start date filter
        date_end: Current end date filter
        character: Current character filter
    """

    CSS = """
    FilterPanel {
        width: 100%;
        height: auto;
        padding: 1 2;
        border-bottom: solid $accent;
        background: $panel;
    }

    FilterPanel .filter-group {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    FilterPanel Label {
        width: 100%;
        height: 1;
        text-style: bold;
        color: $accent;
    }

    FilterPanel Input {
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }

    FilterPanel #filter-info {
        width: 100%;
        height: 1;
        text-style: dim;
    }
    """

    class FilterChanged(Message):
        """Message emitted when filters change.

        Attributes:
            location: Location filter or None
            date_start: Start date filter or None
            date_end: End date filter or None
            character: Character filter or None
            total: Total results count
            filtered: Filtered results count
        """

        def __init__(
            self,
            location: Optional[str] = None,
            date_start: Optional[str] = None,
            date_end: Optional[str] = None,
            character: Optional[str] = None,
            total: int = 0,
            filtered: int = 0,
        ) -> None:
            """Initialize message.

            Args:
                location: Location filter
                date_start: Start date filter
                date_end: End date filter
                character: Character filter
                total: Total results count
                filtered: Filtered results count
            """
            super().__init__()
            self.location = location
            self.date_start = date_start
            self.date_end = date_end
            self.character = character
            self.total = total
            self.filtered = filtered

    # Reactive attributes
    location: reactive[str] = reactive("")
    date_start: reactive[str] = reactive("")
    date_end: reactive[str] = reactive("")
    character: reactive[str] = reactive("")

    def __init__(self, show_labels: bool = True) -> None:
        """Initialize filter panel.

        Args:
            show_labels: Whether to show filter labels. Default: True
        """
        super().__init__()
        self.show_labels = show_labels
        self.total_results = 0
        self.filtered_results = 0

    def compose(self) -> ComposeResult:
        """Compose filter UI.

        Yields:
            Filter input fields
        """
        with Vertical():
            if self.show_labels:
                yield Label("Filters (f to toggle | c to clear)")

            with Container(classes="filter-group"):
                yield Input(
                    placeholder="Location...",
                    id="filter-location",
                )

            with Container(classes="filter-group"):
                yield Input(
                    placeholder="Start date (YYYY-MM-DD)...",
                    id="filter-date-start",
                )

            with Container(classes="filter-group"):
                yield Input(
                    placeholder="End date (YYYY-MM-DD)...",
                    id="filter-date-end",
                )

            with Container(classes="filter-group"):
                yield Input(
                    placeholder="Character name...",
                    id="filter-character",
                )

            yield Label("", id="filter-info")

    def on_mount(self) -> None:
        """Set up focus and handlers on mount."""
        self.query_one("#filter-location", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes.

        Args:
            event: Input changed event
        """
        input_widget = event.input

        if input_widget.id == "filter-location":
            self.location = input_widget.value
        elif input_widget.id == "filter-date-start":
            self.date_start = input_widget.value
        elif input_widget.id == "filter-date-end":
            self.date_end = input_widget.value
        elif input_widget.id == "filter-character":
            self.character = input_widget.value

        self._emit_filter_changed()

    def watch_location(self, value: str) -> None:
        """Watch location filter changes.

        Args:
            value: New location value
        """
        pass

    def watch_date_start(self, value: str) -> None:
        """Watch start date filter changes.

        Args:
            value: New start date value
        """
        pass

    def watch_date_end(self, value: str) -> None:
        """Watch end date filter changes.

        Args:
            value: New end date value
        """
        pass

    def watch_character(self, value: str) -> None:
        """Watch character filter changes.

        Args:
            value: New character value
        """
        pass

    def get_filters(self) -> dict[str, Optional[str]]:
        """Get current filter values.

        Returns:
            Dictionary with filter keys and values (None if empty)
        """
        return {
            "location": self.location if self.location else None,
            "date_start": self.date_start if self.date_start else None,
            "date_end": self.date_end if self.date_end else None,
            "character": self.character if self.character else None,
        }

    def clear_filters(self) -> None:
        """Clear all filters."""
        self.query_one("#filter-location", Input).value = ""
        self.query_one("#filter-date-start", Input).value = ""
        self.query_one("#filter-date-end", Input).value = ""
        self.query_one("#filter-character", Input).value = ""
        self.location = ""
        self.date_start = ""
        self.date_end = ""
        self.character = ""
        self._emit_filter_changed()

    def set_result_counts(self, total: int, filtered: int) -> None:
        """Update result count display.

        Args:
            total: Total results before filtering
            filtered: Results after filtering
        """
        self.total_results = total
        self.filtered_results = filtered
        info_label = self.query_one("#filter-info", Label)
        if total > 0:
            info_label.update(
                f"Results: {filtered} of {total} ({100*filtered//total}%)"
            )
        else:
            info_label.update("No results")

    def _emit_filter_changed(self) -> None:
        """Emit FilterChanged message."""
        filters = self.get_filters()
        self.post_message(
            self.FilterChanged(
                location=filters["location"],
                date_start=filters["date_start"],
                date_end=filters["date_end"],
                character=filters["character"],
                total=self.total_results,
                filtered=self.filtered_results,
            )
        )
