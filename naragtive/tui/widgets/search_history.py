"""Search history widget for NaRAGtive TUI.

Displays list of recent search queries for navigation with arrow keys.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Static, Label


class SearchHistory(Static):
    """Widget showing search history.

    Displays most recent searches (typically 3-5) in a scrollable list.
    User can navigate with up/down arrows and select to run again.

    Attributes:
        max_items: Maximum number of items to show. Default: 5
        history: List of search queries
        current_index: Index of selected query (-1 for none)
    """

    CSS = """
    SearchHistory {
        width: 100%;
        height: auto;
        padding: 1 2;
        border-bottom: solid $accent;
        background: $surface;
    }

    #history-label {
        width: 100%;
        height: 1;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #history-items {
        width: 100%;
        height: auto;
    }

    .history-item {
        width: 100%;
        height: 1;
        padding: 0 1;
    }

    .history-item.selected {
        background: $accent;
        color: $text;
    }

    .history-item.unselected {
        background: $surface;
        color: $text-secondary;
    }
    """

    history: reactive[list[str]] = reactive([])
    current_index: reactive[int] = reactive(-1)

    def __init__(self, max_items: int = 5) -> None:
        """Initialize search history.

        Args:
            max_items: Maximum number of items to show. Default: 5
        """
        super().__init__()
        self.max_items = max_items
        self.history = []
        self.current_index = -1

    def compose(self) -> ComposeResult:
        """Compose history UI.

        Yields:
            Label for title and container for items
        """
        yield Label("Recent Searches", id="history-label")
        yield Container(id="history-items")

    def add_query(self, query: str) -> None:
        """Add query to history.

        Args:
            query: Query string to add
        """
        if query and query not in self.history:
            self.history.insert(0, query)
            # Keep only max_items
            if len(self.history) > self.max_items:
                self.history = self.history[:self.max_items]
            self.current_index = -1
            self._update_display()

    def clear_history(self) -> None:
        """Clear all history."""
        self.history = []
        self.current_index = -1
        self._update_display()

    def navigate_up(self) -> Optional[str]:
        """Navigate up in history.

        Returns:
            Selected query or None if no selection
        """
        if not self.history:
            return None

        if self.current_index < len(self.history) - 1:
            self.current_index += 1
        self._update_display()
        return self.history[self.current_index]

    def navigate_down(self) -> Optional[str]:
        """Navigate down in history.

        Returns:
            Selected query or None if at bottom
        """
        if not self.history:
            return None

        if self.current_index > 0:
            self.current_index -= 1
        elif self.current_index == 0:
            self.current_index = -1

        self._update_display()
        return (
            self.history[self.current_index]
            if self.current_index >= 0
            else None
        )

    def get_selected(self) -> Optional[str]:
        """Get currently selected query.

        Returns:
            Selected query or None
        """
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index]
        return None

    def watch_history(self, value: list[str]) -> None:
        """Watch history changes.

        Args:
            value: New history list
        """
        self._update_display()

    def watch_current_index(self, value: int) -> None:
        """Watch index changes.

        Args:
            value: New index
        """
        self._update_display()

    def _update_display(self) -> None:
        """Update history display."""
        try:
            items_container = self.query_one("#history-items", Container)
            items_container.query(Label).remove()

            for i, query in enumerate(self.history):
                is_selected = i == self.current_index
                classes = "history-item selected" if is_selected else "history-item unselected"
                marker = "► " if is_selected else "  "
                items_container.mount(
                    Label(f"{marker}{query}", classes=classes)
                )
        except Exception:
            pass
