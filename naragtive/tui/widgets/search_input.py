"""Search input modal widget for NaRAGtive TUI.

Provides a text input field with search history navigation.
"""

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Input, Label, Static
from textual.message import Message

if TYPE_CHECKING:
    pass


class SearchRequested(Message):
    """Posted when user submits a search query.
    
    Attributes:
        query: The search query string
    """

    def __init__(self, query: str) -> None:
        """Initialize search requested message.
        
        Args:
            query: Search query string
        """
        super().__init__()
        self.query = query


class SearchInputWidget(Static):
    """Search input widget with history support.
    
    Displays a text input for search queries with keybindings for
    history navigation.
    
    Attributes:
        search_history: List of previous search queries
        history_index: Current position in history (None = new query)
    """

    CSS = """
    SearchInputWidget {
        width: 100%;
        height: auto;
        background: $panel;
        border: solid $primary;
        padding: 1;
    }

    SearchInputWidget Label {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }

    SearchInputWidget Input {
        width: 100%;
        margin-bottom: 1;
    }

    SearchInputWidget .info {
        width: 100%;
        height: auto;
        color: $text-muted;
        font-size: 80%;
    }
    """

    BINDINGS = [
        ("up", "history_back", "Previous"),
        ("down", "history_forward", "Next"),
        ("ctrl+u", "clear_input", "Clear"),
    ]

    def __init__(self, search_history: list[str] | None = None) -> None:
        """Initialize search input widget.
        
        Args:
            search_history: Optional list of previous searches
        """
        super().__init__()
        self.search_history = search_history or []
        self.history_index: int | None = None
        self.input_widget: Input | None = None

    def compose(self) -> ComposeResult:
        """Compose search input UI.
        
        Yields:
            Label, Input, and info widgets
        """
        yield Label("Enter search query:")
        self.input_widget = Input(placeholder="e.g., Admiral leadership")
        yield self.input_widget
        
        if self.search_history:
            yield Static(
                f"[↑↓] to browse history ({len(self.search_history)} saved)",
                classes="info",
            )
        else:
            yield Static("[enter] to search, [esc] to cancel", classes="info")

    def on_mount(self) -> None:
        """Handle widget mount.
        
        Focuses the input field.
        """
        if self.input_widget:
            self.input_widget.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission.
        
        Args:
            event: Input submitted event
        """
        query = event.value.strip()
        if query:
            # Add to history if not duplicate
            if not self.search_history or self.search_history[0] != query:
                self.search_history.insert(0, query)
                # Keep only last 50 searches
                self.search_history = self.search_history[:50]
            
            # Post search requested message
            self.post_message(SearchRequested(query))

    def action_history_back(self) -> None:
        """Navigate to previous search in history.
        
        Moves backward through search history, wrapping to end.
        """
        if not self.search_history or not self.input_widget:
            return
        
        if self.history_index is None:
            # Start at beginning of history
            self.history_index = 0
        else:
            # Move backward, wrap at end
            self.history_index = (self.history_index + 1) % len(self.search_history)
        
        # Update input with historical query
        self.input_widget.value = self.search_history[self.history_index]
        self.input_widget.cursor_position = len(self.input_widget.value)

    def action_history_forward(self) -> None:
        """Navigate to next search in history.
        
        Moves forward through search history, wrapping to start.
        """
        if not self.search_history or not self.input_widget:
            return
        
        if self.history_index is None:
            # Start at end of history
            self.history_index = len(self.search_history) - 1
        else:
            # Move forward, wrap at start
            self.history_index = (self.history_index - 1) % len(self.search_history)
        
        # Update input with historical query
        self.input_widget.value = self.search_history[self.history_index]
        self.input_widget.cursor_position = len(self.input_widget.value)

    def action_clear_input(self) -> None:
        """Clear the input field."""
        if self.input_widget:
            self.input_widget.value = ""
            self.history_index = None

    def get_history(self) -> list[str]:
        """Get current search history.
        
        Returns:
            List of search queries
        """
        return self.search_history.copy()

    def set_history(self, history: list[str]) -> None:
        """Set search history.
        
        Args:
            history: List of search queries
        """
        self.search_history = history[:50]  # Keep max 50
