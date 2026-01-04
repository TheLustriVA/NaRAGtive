"""Interactive search screen for NaRAGtive TUI.

Provides mode for executing multiple queries without returning to dashboard.
Maintains search history and filter state across queries.
"""

import asyncio
from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Button, DataTable, Static, Label
from rich.text import Text

from naragtive.store_registry import VectorStoreRegistry
from naragtive.polars_vectorstore import PolarsVectorStore
from naragtive.tui.search_utils import async_search, apply_filters, format_relevance_score, parse_metadata
from naragtive.tui.widgets.search_history import SearchHistory
from naragtive.tui.widgets.filter_panel import FilterPanel


class InteractiveSearchScreen(Screen[None]):
    """Screen for interactive multi-query search mode.

    Features:
    - Execute multiple queries without returning to dashboard
    - Navigate query history with up/down arrows
    - Maintain filter state across queries
    - Display result count
    - Show search results in table

    Key bindings:
        'Enter': Execute search
        'Up': Navigate history up
        'Down': Navigate history down
        'f': Toggle filter pane
        'c': Clear filters
        'Escape': Exit to dashboard

    Attributes:
        TITLE: Screen title
        BINDINGS: Key bindings
    """

    TITLE = "Interactive Search"

    BINDINGS = [
        Binding("escape", "dismiss", "Exit", show=True),
        Binding("f", "toggle_filters", "Filters", show=True),
        Binding("c", "clear_filters", "Clear", show=True),
    ]

    CSS = """
    InteractiveSearchScreen {
        layout: vertical;
    }

    #search-input-bar {
        width: 100%;
        height: auto;
        padding: 1 2;
        border-bottom: solid $accent;
    }

    #search-input {
        width: 1fr;
        height: 1;
    }

    #filter-panel {
        width: 100%;
        height: auto;
    }

    #history-widget {
        width: 100%;
        height: auto;
    }

    #results-table {
        width: 100%;
        height: 1fr;
    }

    #status-bar {
        width: 100%;
        height: 1;
        dock: bottom;
        padding: 0 2;
        background: $surface;
    }
    """

    def __init__(self) -> None:
        """Initialize interactive search screen."""
        super().__init__()
        self.registry = VectorStoreRegistry()
        self.store: Optional[PolarsVectorStore] = None
        self.current_results: dict[str, Any] = {
            "ids": [],
            "documents": [],
            "scores": [],
            "metadatas": [],
        }
        self.filters_visible = False
        self._searching = False

    def compose(self) -> ComposeResult:
        """Compose screen UI.

        Yields:
            Header, search input, filter panel, history, results table, status, Footer
        """
        yield Header()

        with Container(id="search-input-bar"):
            with Horizontal():
                yield Input(
                    placeholder="Enter search query...",
                    id="search-input",
                )
                yield Button("Search", id="search-btn", variant="primary")

        yield FilterPanel(id="filter-panel")
        yield SearchHistory(id="history-widget")
        yield DataTable(id="results-table", show_header=True, show_cursor=True)
        yield Label("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.run_worker(self._init_store())
        self.query_one("#search-input", Input).focus()
        self.query_one("#filter-panel", FilterPanel).display = False
        self._setup_table()

    def _setup_table(self) -> None:
        """Setup results table columns."""
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Relevance", "Location", "Date", "Preview")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search submission.

        Args:
            event: Input submitted event
        """
        self._execute_search()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        if event.button.id == "search-btn":
            self._execute_search()

    def action_toggle_filters(self) -> None:
        """Action to toggle filter pane visibility."""
        filter_panel = self.query_one("#filter-panel", FilterPanel)
        filter_panel.display = not filter_panel.display
        self.filters_visible = filter_panel.display

    def action_clear_filters(self) -> None:
        """Action to clear all filters."""
        filter_panel = self.query_one("#filter-panel", FilterPanel)
        filter_panel.clear_filters()
        self._apply_filters()

    def action_dismiss(self) -> None:
        """Action to exit to dashboard."""
        self.dismiss()

    def on_key(self, event) -> None:
        """Handle key presses.

        Args:
            event: Key event
        """
        if event.key == "up":
            search_input = self.query_one("#search-input", Input)
            if search_input.has_focus:
                history = self.query_one("#history-widget", SearchHistory)
                query = history.navigate_up()
                if query:
                    search_input.value = query
                    event.stop()
        elif event.key == "down":
            search_input = self.query_one("#search-input", Input)
            if search_input.has_focus:
                history = self.query_one("#history-widget", SearchHistory)
                query = history.navigate_down()
                if query:
                    search_input.value = query
                else:
                    search_input.value = ""
                event.stop()

    async def _init_store(self) -> None:
        """Initialize vector store."""
        try:
            default_name = self.registry.get_default()
            if not default_name:
                self._update_status("[error]No default store[/error]")
                return

            metadata = self.registry._stores.get(default_name)
            if not metadata:
                self._update_status("[error]Store not found[/error]")
                return

            loop = asyncio.get_event_loop()
            self.store = await loop.run_in_executor(
                None, lambda: PolarsVectorStore(str(metadata.path))
            )
            await loop.run_in_executor(None, lambda: self.store.load())
            self._update_status(f"Ready: {default_name}")
        except Exception as e:
            self._update_status(f"[error]Error: {str(e)}[/error]")

    async def _execute_search(self) -> None:
        """Execute search query."""
        if self._searching or not self.store:
            return

        search_input = self.query_one("#search-input", Input)
        query = search_input.value.strip()

        if not query or len(query) < 3:
            self._update_status("[error]Query must be at least 3 characters[/error]")
            return

        self._searching = True
        self._update_status("Searching...")

        try:
            results = await async_search(self.store, query, n_results=50)
            self.current_results = results

            history = self.query_one("#history-widget", SearchHistory)
            history.add_query(query)

            self._apply_filters()

        except Exception as e:
            self._update_status(f"[error]Search failed: {str(e)}[/error]")
        finally:
            self._searching = False

    def _apply_filters(self) -> None:
        """Apply filters to current results."""
        filter_panel = self.query_one("#filter-panel", FilterPanel)
        filters = filter_panel.get_filters()

        filtered = apply_filters(
            self.current_results,
            location=filters["location"],
            date_start=filters["date_start"],
            date_end=filters["date_end"],
            character=filters["character"],
        )

        total = len(self.current_results.get("ids", []))
        filtered_count = len(filtered.get("ids", []))
        filter_panel.set_result_counts(total, filtered_count)

        self._update_results_table(filtered)
        self._update_status(
            f"Results: {filtered_count} of {total}"
        )

    def _update_results_table(self, results: dict[str, Any]) -> None:
        """Update results table with data.

        Args:
            results: Filtered search results
        """
        table = self.query_one("#results-table", DataTable)
        table.clear()

        for i in range(len(results.get("ids", []))):
            score = results["scores"][i]
            metadata = results["metadatas"][i]
            document = results["documents"][i]

            parsed = parse_metadata(metadata)

            preview = document[:50] + "..." if len(document) > 50 else document

            table.add_row(
                format_relevance_score(score),
                parsed["location"],
                parsed["date"],
                preview,
            )

    def _update_status(self, message: str) -> None:
        """Update status bar message.

        Args:
            message: Status message (supports markup)
        """
        status = self.query_one("#status-bar", Label)
        status.update(Text.from_markup(message))
