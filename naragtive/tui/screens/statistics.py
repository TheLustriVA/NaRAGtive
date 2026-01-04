"""Statistics screen for NaRAGtive TUI.

Displays store metadata, scene breakdowns, and embedding info.
Loads data asynchronously to avoid UI blocking.
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Footer, Header, Static, Label
from rich.panel import Panel
from rich.text import Text
from rich.console import Console

from naragtive.store_registry import VectorStoreRegistry
from naragtive.polars_vectorstore import PolarsVectorStore


class StatisticsScreen(Screen[None]):
    """Screen displaying store statistics and metadata.

    Shows:
    - Store metadata (path, type, records, size, created date)
    - Scene breakdown by location (top 5 + other)
    - Scene breakdown by character (top 5)
    - Embedding model info (all-MiniLM-L6-v2, 384 dims)
    - Reranker info if available

    Async data collection prevents UI blocking.

    Key bindings:
        'Escape': Exit to dashboard

    Attributes:
        TITLE: Screen title
        BINDINGS: Key bindings
    """

    TITLE = "Store Statistics"

    BINDINGS = [
        Binding("escape", "dismiss", "Exit", show=True),
    ]

    CSS = """
    StatisticsScreen {
        layout: vertical;
    }

    #stats-header {
        width: 100%;
        height: auto;
        padding: 1 2;
        border-bottom: solid $accent;
    }

    #stats-content {
        width: 100%;
        height: 1fr;
        overflow: auto;
    }

    .stat-section {
        width: 100%;
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
        border: solid $accent;
        background: $surface;
    }

    .stat-title {
        width: 100%;
        height: 1;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .stat-item {
        width: 100%;
        height: auto;
        padding: 0 1;
        color: $text;
    }

    .loading {
        width: 100%;
        height: 1;
        content-align: center middle;
        color: $text-secondary;
    }
    """

    def __init__(self) -> None:
        """Initialize statistics screen."""
        super().__init__()
        self.registry = VectorStoreRegistry()
        self.store: Optional[PolarsVectorStore] = None
        self.stats: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        """Compose screen UI.

        Yields:
            Header, content container, Footer
        """
        yield Header()
        yield Label("Loading statistics...", id="stats-header", classes="loading")
        yield ScrollableContainer(id="stats-content")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.load_worker(self._load_statistics())

    def action_dismiss(self) -> None:
        """Action to exit to dashboard."""
        self.dismiss()

    async def _load_statistics(self) -> None:
        """Load all statistics asynchronously."""
        try:
            # Get default store
            default_name = self.registry.get_default()
            if not default_name:
                self._show_error("No default store set")
                return

            metadata = self.registry._stores.get(default_name)
            if not metadata:
                self._show_error("Store not found")
                return

            # Load store in executor
            loop = asyncio.get_event_loop()
            self.store = await loop.run_in_executor(
                None, lambda: PolarsVectorStore(str(metadata.path))
            )
            await loop.run_in_executor(None, lambda: self.store.load())

            # Collect statistics
            self.stats = await loop.run_in_executor(
                None, self._collect_statistics
            )

            # Update UI
            self._render_statistics()
        except Exception as e:
            self._show_error(f"Error loading statistics: {str(e)}")

    def _collect_statistics(self) -> dict[str, Any]:
        """Collect statistics from store.

        Returns:
            Dictionary with collected statistics
        """
        if not self.store:
            return {}

        stats = {}
        try:
            # Get dataframe
            df = self.store.data
            if df is None:
                return {}

            stats["total_records"] = len(df)
            stats["file_size_mb"] = self.store.path.stat().st_size / (1024 * 1024)

            # Location breakdown
            if "location" in df.columns:
                locations = df["location"].value_counts()
                location_dict = {
                    str(loc): int(count)
                    for loc, count in zip(locations.to_list(), locations.to_list())
                }
                # Top 5 + other
                top_5_locations = dict(sorted(
                    location_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
                other_count = sum(
                    v for k, v in location_dict.items()
                    if k not in top_5_locations
                )
                if other_count > 0:
                    top_5_locations["Other"] = other_count
                stats["locations"] = top_5_locations

            # Character breakdown
            if "characters_present" in df.columns:
                char_counter = Counter()
                for chars_str in df["characters_present"]:
                    if chars_str is not None:
                        try:
                            chars = json.loads(chars_str)
                            if isinstance(chars, list):
                                char_counter.update(chars)
                        except (json.JSONDecodeError, TypeError):
                            pass
                top_5_chars = dict(char_counter.most_common(5))
                stats["characters"] = top_5_chars

            # Model info
            stats["embedding_model"] = "all-MiniLM-L6-v2"
            stats["embedding_dims"] = 384
            stats["reranker_model"] = None
            stats["reranker_vram"] = None

        except Exception as e:
            print(f"Error collecting statistics: {e}")

        return stats

    def _render_statistics(self) -> None:
        """Render statistics in the UI."""
        try:
            content = self.query_one("#stats-content", ScrollableContainer)
            header = self.query_one("#stats-header", Label)

            # Update header
            default_name = self.registry.get_default() or "Unknown"
            header.update(f"Store: {default_name}")

            # Clear content
            content.query(Static).remove()

            # Add metadata section
            if self.store:
                meta_text = f"""
Path: {self.store.path}
Records: {self.stats.get('total_records', 'N/A')}
Size: {self.stats.get('file_size_mb', 0):.2f} MB
            """
                content.mount(Static(meta_text, id="meta-section", classes="stat-section"))

            # Add location breakdown
            if "locations" in self.stats:
                loc_text = "Scenes by Location:\n"
                for loc, count in self.stats["locations"].items():
                    loc_text += f"  {loc}: {count}\n"
                content.mount(Static(loc_text, id="location-section", classes="stat-section"))

            # Add character breakdown
            if "characters" in self.stats:
                char_text = "Top Characters:\n"
                for char, count in self.stats["characters"].items():
                    char_text += f"  {char}: {count}\n"
                content.mount(Static(char_text, id="char-section", classes="stat-section"))

            # Add model info
            model_text = f"""
Embedding Model: {self.stats.get('embedding_model', 'N/A')}
Dimensions: {self.stats.get('embedding_dims', 'N/A')}
            """
            content.mount(Static(model_text, id="model-section", classes="stat-section"))

        except Exception as e:
            self._show_error(f"Error rendering: {str(e)}")

    def _show_error(self, message: str) -> None:
        """Display error message.

        Args:
            message: Error message to display
        """
        try:
            header = self.query_one("#stats-header", Label)
            header.update(f"[error]{message}[/error]")
        except Exception:
            pass
