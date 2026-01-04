"""Statistics screen for NaRAGtive TUI.

Displays detailed information about the current store including:
- Store metadata (path, type, records, size, created date)
- Scene breakdown by location (top 5 + other)
- Scene breakdown by character (top 5)
- Embedding model info (all-MiniLM-L6-v2, 384 dims)
- Reranker info if available
"""

import asyncio
from pathlib import Path
from typing import Any, Optional
from collections import Counter
import json

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Button, Static
from rich.text import Text
from rich.table import Table

from naragtive.store_registry import VectorStoreRegistry
from naragtive.polars_vectorstore import PolarsVectorStore


class StatisticsScreen(Screen[None]):
    """Screen displaying store statistics and metadata.

    Key bindings:
        'Escape': Return to dashboard
        'r': Refresh statistics

    Attributes:
        TITLE: Screen title
        BINDINGS: Key bindings
    """

    TITLE = "Store Statistics"

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("escape", "dismiss", "Back", show=True),
    ]

    CSS = """
    StatisticsScreen {
        layout: vertical;
    }

    #stats-container {
        width: 100%;
        height: 1fr;
        overflow: auto;
    }

    #stats-section {
        width: 100%;
        height: auto;
        padding: 1 2;
        border-bottom: solid $accent;
    }

    #stats-section Label {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    #button-bar {
        width: 100%;
        height: auto;
        dock: bottom;
        layout: horizontal;
    }

    #button-bar Button {
        flex: 1;
        margin-right: 1;
    }

    #button-bar Button:last-child {
        margin-right: 0;
    }
    """

    def __init__(self) -> None:
        """Initialize statistics screen."""
        super().__init__()
        self.registry = VectorStoreRegistry()
        self._loading = False

    def compose(self) -> ComposeResult:
        """Compose screen UI.

        Yields:
            Header, statistics container, button bar, Footer
        """
        yield Header()
        yield Container(id="stats-container")
        with Horizontal(id="button-bar"):
            yield Button("Refresh", id="refresh-btn", variant="primary")
            yield Button("Back", id="back-btn", variant="default")
        yield Footer()

    def on_mount(self) -> None:
        """Load statistics on mount."""
        self.load_worker(self._load_statistics())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        if event.button.id == "refresh-btn":
            self.action_refresh()
        elif event.button.id == "back-btn":
            self.action_dismiss()

    def action_refresh(self) -> None:
        """Action to refresh statistics."""
        self.load_worker(self._load_statistics())

    def action_dismiss(self) -> None:
        """Action to return to dashboard."""
        self.dismiss()

    async def _load_statistics(self) -> None:
        """Load and display statistics asynchronously."""
        if self._loading:
            return

        self._loading = True
        container = self.query_one("#stats-container", Container)
        container.query(Label).remove()

        try:
            # Get default store
            default_name = self.registry.get_default()
            if not default_name:
                container.mount(Label("[error]No default store set[/error]"))
                return

            metadata = self.registry._stores.get(default_name)
            if not metadata:
                container.mount(Label("[error]Store not found[/error]"))
                return

            # Load store in executor to avoid blocking
            loop = asyncio.get_event_loop()
            store = await loop.run_in_executor(
                None, lambda: PolarsVectorStore(str(metadata.path))
            )
            await loop.run_in_executor(None, lambda: store.load())

            # Collect statistics
            stats = await self._collect_statistics(store, metadata)

            # Display statistics
            container.query(Label).remove()
            self._display_statistics(container, stats)

        except Exception as e:
            container.query(Label).remove()
            container.mount(Label(f"[error]Error loading statistics: {str(e)}[/error]"))
        finally:
            self._loading = False

    async def _collect_statistics(self, store: Any, metadata: Any) -> dict[str, Any]:
        """Collect store statistics.

        Args:
            store: Loaded PolarsVectorStore instance
            metadata: Store metadata

        Returns:
            Dictionary with statistics
        """
        stats = {
            "store_name": metadata.name,
            "store_type": metadata.source_type,
            "record_count": metadata.record_count,
            "created_at": metadata.created_at,
            "path": str(metadata.path),
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dims": 384,
            "locations": {},
            "characters": {},
        }

        try:
            # Get dataframe
            if not store.df or len(store.df) == 0:
                return stats

            df = store.df

            # Count locations
            if "location" in df.columns:
                locations = df["location"].value_counts()
                stats["locations"] = {
                    str(loc): int(count)
                    for loc, count in zip(locations.to_list()[:5], locations.to_list()[:5])
                }

            # Count characters
            if "characters_present" in df.columns:
                char_counter = Counter()
                for chars_str in df["characters_present"]:
                    try:
                        if isinstance(chars_str, str):
                            chars = json.loads(chars_str)
                        else:
                            chars = chars_str
                        if isinstance(chars, list):
                            char_counter.update(chars)
                    except (json.JSONDecodeError, TypeError):
                        pass

                stats["characters"] = dict(char_counter.most_common(5))

            # File size
            try:
                file_size = Path(metadata.path).stat().st_size
                stats["file_size_mb"] = file_size / (1024 * 1024)
            except Exception:
                pass

        except Exception as e:
            pass

        return stats

    def _display_statistics(self, container: Container, stats: dict[str, Any]) -> None:
        """Display statistics in container.

        Args:
            container: Container to display statistics in
            stats: Statistics dictionary
        """
        # Store metadata section
        with container.mount(Container(id="stats-section")):
            header = Text(f"Store: {stats['store_name']}", style="bold $accent")
            yield Label(header)
            yield Label(
                f"Type: {stats['store_type']}  |  "
                f"Records: {stats['record_count']:,}  |  "
                f"Created: {stats['created_at'].split('T')[0]}"
            )
            if "file_size_mb" in stats:
                yield Label(f"File Size: {stats['file_size_mb']:.2f} MB")
            yield Label(f"Path: {stats['path']}")

        # Embedding model section
        with container.mount(Container(id="stats-section")):
            yield Label("[bold $accent]Embedding Model[/bold]")
            yield Label(
                f"Model: {stats['embedding_model']}\n"
                f"Dimensions: {stats['embedding_dims']}"
            )

        # Locations breakdown
        if stats["locations"]:
            with container.mount(Container(id="stats-section")):
                yield Label("[bold $accent]Top Locations[/bold]")
                for loc, count in stats["locations"].items():
                    yield Label(f"  {loc}: {count} scenes")

        # Characters breakdown
        if stats["characters"]:
            with container.mount(Container(id="stats-section")):
                yield Label("[bold $accent]Top Characters[/bold]")
                for char, count in stats["characters"].items():
                    yield Label(f"  {char}: {count} scenes")
