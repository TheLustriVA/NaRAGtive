"""Search screen for NaRAGtive TUI.

Provides comprehensive search functionality with results display,
reranking, and export capabilities.
"""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Label, Input, Button
from textual.screen import Screen
from textual.reactive import reactive

from naragtive.store_registry import VectorStoreRegistry
from naragtive.tui.screens.base import BaseScreen
from naragtive.tui.widgets import (
    SearchInputWidget,
    SearchRequested,
    ResultsTableWidget,
    ResultSelected,
    ResultDetailWidget,
)
from naragtive.tui.search_utils import (
    async_search,
    async_rerank,
    format_search_query,
    SearchError,
)

if TYPE_CHECKING:
    from naragtive.polars_vectorstore import PolarsVectorStore


class SearchScreen(BaseScreen):
    """Main search screen with results and detail view.
    
    Provides:
    - Search input with history
    - Results DataTable (sorted by relevance)
    - Result detail panel
    - Reranking toggle
    - Export to JSON/CSV
    - Async search operations
    
    Attributes:
        current_query: Current search query
        search_results: Current search results
        reranking_enabled: Whether reranking is enabled
        reranking_in_progress: Whether reranking is currently running
        search_history: List of previous searches
        store: Vector store instance
    """

    BINDINGS = [
        ("/", "open_search", "Search"),
        ("r", "toggle_rerank", "Rerank"),
        ("e", "export", "Export"),
        ("escape", "back", "Back"),
    ] + BaseScreen.BINDINGS

    CSS = """
    SearchScreen {
        layout: vertical;
    }

    SearchScreen #search-header {
        width: 100%;
        height: auto;
        background: $boost;
        border-bottom: solid $primary;
        padding: 1;
    }

    SearchScreen #search-title {
        width: 100%;
        height: auto;
        text-align: center;
        text-style: bold;
    }

    SearchScreen #search-status {
        width: 100%;
        height: auto;
        background: $panel;
        padding: 1;
        border-bottom: solid $primary;
        color: $text-muted;
    }

    SearchScreen #main-content {
        width: 100%;
        height: 1fr;
        layout: horizontal;
    }

    SearchScreen #results-panel {
        width: 60%;
        height: 1fr;
    }

    SearchScreen #detail-panel {
        width: 40%;
        height: 1fr;
        border-left: solid $primary;
    }

    SearchScreen #search-footer {
        width: 100%;
        height: auto;
        background: $panel;
        border-top: solid $primary;
        padding: 1;
    }
    """

    current_query: reactive[str] = reactive("")
    search_results: reactive[dict[str, Any] | None] = reactive(None)
    reranking_enabled: reactive[bool] = reactive(False)
    reranking_in_progress: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        """Initialize search screen."""
        super().__init__()
        self.search_history: list[str] = []
        self.store: "PolarsVectorStore | None" = None
        self._current_sort = "relevance"

    def compose(self) -> ComposeResult:
        """Compose search screen UI.
        
        Yields:
            Header, search widgets, results, and footer
        """
        # Header
        with Container(id="search-header"):
            yield Label("NaRAGtive Search", id="search-title")

        # Status
        yield Label("Ready to search. Press / to enter query.", id="search-status")

        # Main content - results and detail side by side
        with Horizontal(id="main-content"):
            # Results table
            yield ResultsTableWidget(id="results-panel")

            # Detail panel
            yield ResultDetailWidget(id="detail-panel")

        # Footer
        yield Label(
            "[/] Search  [r] Rerank  [e] Export  [esc] Back",
            id="search-footer",
        )

    async def on_mount(self) -> None:
        """Handle screen mount.
        
        Loads vector store and initializes search.
        """
        # Load store
        await self._load_store()

    async def _load_store(self) -> None:
        """Load vector store asynchronously."""
        loop = asyncio.get_event_loop()
        try:
            registry = await loop.run_in_executor(None, VectorStoreRegistry)
            store_name = await loop.run_in_executor(None, registry.get_default)

            if not store_name:
                self.app.notify(
                    "No default store set. Go to dashboard.",
                    severity="warning",
                    timeout=5,
                )
                self.action_back()
                return

            self.store = await loop.run_in_executor(None, registry.get, store_name)

            if self.store is None:
                self.app.notify(
                    "Failed to load store",
                    severity="error",
                    timeout=5,
                )
                self.action_back()
                return

            # Ensure store is loaded
            await loop.run_in_executor(None, self.store.load)
            self.app.notify(f"Store loaded: {store_name}", timeout=3)

        except Exception as e:
            self.app.notify(
                f"Error loading store: {e}",
                severity="error",
                timeout=5,
            )
            self.action_back()

    def action_open_search(self) -> None:
        """Open search input modal."""
        self.app.push_screen(
            SearchInputScreen(self.search_history, self._on_search_submitted)
        )

    async def _on_search_submitted(self, query: str) -> None:
        """Handle search query submission.
        
        Args:
            query: Search query string
        """
        try:
            # Validate and format query
            formatted_query = format_search_query(query)
            self.current_query = formatted_query

            # Update status
            status = self.query_one("#search-status", Label)
            status.update(f"Searching: {formatted_query}...")

            # Perform async search
            if not self.store:
                raise SearchError("Store not loaded")

            results = await async_search(
                self.store,
                formatted_query,
                n_results=20,
            )

            # Display results
            results_table = self.query_one("#results-panel", ResultsTableWidget)
            results_table.update_results(results)

            # Update status
            result_count = len(results.get("ids", []))
            status.update(
                f"Results: {result_count} scenes for '{formatted_query}'"
            )

            self.search_results = results
            self.app.notify(
                f"Found {result_count} results",
                timeout=3,
            )

        except SearchError as e:
            self.app.notify(
                f"Search error: {e}",
                severity="error",
                timeout=5,
            )

    def on_results_table_widget_result_selected(
        self, message: ResultSelected
    ) -> None:
        """Handle result selection.
        
        Args:
            message: Result selected message
        """
        if not self.search_results:
            return

        results = self.search_results
        result_index = message.result_index

        # Get result data
        if result_index < len(results["ids"]):
            detail_panel = self.query_one("#detail-panel", ResultDetailWidget)
            detail_panel.display_result(
                results["ids"][result_index],
                results["documents"][result_index],
                results["metadatas"][result_index],
                results.get(
                    "rerank_scores" if self.reranking_enabled else "scores", []
                )[result_index],
            )

    def action_toggle_rerank(self) -> None:
        """Toggle reranking mode.
        
        Note: Actual reranking would be triggered here.
        """
        if not self.search_results:
            self.app.notify("No results to rerank", severity="warning", timeout=3)
            return

        self.reranking_enabled = not self.reranking_enabled
        status = self.query_one("#search-status", Label)
        status.update(
            f"Reranking: {'ENABLED' if self.reranking_enabled else 'DISABLED'}"
        )
        self.app.notify(
            f"Reranking {'enabled' if self.reranking_enabled else 'disabled'}",
            timeout=3,
        )

    def action_export(self) -> None:
        """Export results to JSON file."""
        if not self.search_results:
            self.app.notify("No results to export", severity="warning", timeout=3)
            return

        try:
            # Create export data
            export_data = {
                "query": self.current_query,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(self.search_results["ids"]),
                "reranked": self.reranking_enabled,
                "results": [
                    {
                        "scene_id": scene_id,
                        "score": score,
                        "metadata": metadata,
                    }
                    for scene_id, score, metadata in zip(
                        self.search_results["ids"],
                        self.search_results.get(
                            "rerank_scores"
                            if self.reranking_enabled
                            else "scores",
                            [],
                        ),
                        self.search_results["metadatas"],
                    )
                ],
            }

            # Write to file
            export_path = Path(f"naragtive_results_{datetime.now():%Y%m%d_%H%M%S}.json")
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            self.app.notify(
                f"Exported to {export_path.name}",
                timeout=3,
            )

        except Exception as e:
            self.app.notify(
                f"Export error: {e}",
                severity="error",
                timeout=5,
            )


class SearchInputScreen(Screen):
    """Modal screen for search input with history.
    
    Attributes:
        search_history: List of previous searches
        callback: Callback function when search is submitted
    """

    CSS = """
    SearchInputScreen {
        align: center middle;
    }

    SearchInputScreen > Container {
        width: 60%;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    """

    def __init__(
        self,
        search_history: list[str],
        callback: Any,
    ) -> None:
        """Initialize search input screen.
        
        Args:
            search_history: List of previous searches
            callback: Callback when search is submitted
        """
        super().__init__()
        self.search_history = search_history
        self.callback = callback

    def compose(self) -> ComposeResult:
        """Compose search input UI.
        
        Yields:
            Container with search input widget
        """
        with Container():
            yield SearchInputWidget(self.search_history)

    def on_search_input_widget_search_requested(
        self, message: SearchRequested
    ) -> None:
        """Handle search request.
        
        Args:
            message: Search requested message
        """
        # Call callback
        asyncio.create_task(self.callback(message.query))
        # Close modal
        self.app.pop_screen()
