"""Result detail panel widget for displaying full scene text.

Provides scrollable text display with metadata and character information.
"""

import json
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, TextArea, Label
from textual.message import Message

from naragtive.tui.search_utils import parse_metadata


class DetailPanelClosed(Message):
    """Posted when user closes detail panel."""

    def __init__(self) -> None:
        """Initialize detail panel closed message."""
        super().__init__()


class ResultDetailWidget(Static):
    """Panel displaying full result details.
    
    Shows:
    - Full scene text (scrollable)
    - Metadata (date, location, POV, characters)
    - Relevance score
    
    Attributes:
        result_data: Current result being displayed
    """

    CSS = """
    ResultDetailWidget {
        width: 100%;
        height: 1fr;
        border: solid $primary;
        background: $surface;
    }

    ResultDetailWidget .metadata-header {
        width: 100%;
        height: auto;
        background: $boost;
        padding: 1;
        border-bottom: solid $primary;
    }

    ResultDetailWidget .metadata-grid {
        width: 100%;
        height: auto;
        background: $panel;
        padding: 1;
        border-bottom: solid $primary;
    }

    ResultDetailWidget Label {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    ResultDetailWidget .content {
        width: 100%;
        height: 1fr;
        overflow: auto;
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "action_close", "Close"),
        ("y", "action_copy_id", "Copy ID"),
        ("c", "action_copy_text", "Copy Text"),
    ]

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize result detail widget.
        
        Args:
            name: Name of widget
            id: ID of widget
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.result_data: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        """Compose detail panel UI.
        
        Yields:
            Metadata and content widgets
        """
        # Title area
        yield Label("[No Result Selected]", id="detail-title")
        
        # Metadata grid
        with Vertical(id="detail-metadata"):
            yield Label("Date: --", id="detail-date")
            yield Label("Location: --", id="detail-location")
            yield Label("POV: --", id="detail-pov")
            yield Label("Characters: --", id="detail-chars")
            yield Label("Score: --", id="detail-score")
        
        # Scene text area
        yield Label("\n📖 Scene Text:", id="detail-text-label")
        yield Static(
            "[No content to display]",
            id="detail-content",
            classes="content",
        )

    def display_result(
        self,
        result_id: str,
        document_text: str,
        metadata: dict[str, Any],
        score: float,
    ) -> None:
        """Display a result in detail panel.
        
        Args:
            result_id: Scene ID
            document_text: Full scene text
            metadata: Metadata dict from search results
            score: Relevance score
        """
        self.result_data = {
            "id": result_id,
            "text": document_text,
            "metadata": metadata,
            "score": score,
        }

        # Parse metadata
        parsed = parse_metadata(metadata)

        # Update title
        title_label = self.query_one("#detail-title", Label)
        title_label.update(f"Scene: {parsed['scene_id']}")

        # Update metadata fields
        self.query_one("#detail-date", Label).update(
            f"Date: {parsed['date']}"
        )
        self.query_one("#detail-location", Label).update(
            f"Location: {parsed['location']}"
        )
        self.query_one("#detail-pov", Label).update(
            f"POV: {parsed['pov']}"
        )

        # Format characters
        chars = parsed.get("characters", [])
        chars_str = ", ".join(chars) if chars else "(none)"
        self.query_one("#detail-chars", Label).update(
            f"Characters: {chars_str}"
        )

        # Format score
        score_pct = int(score * 100) if isinstance(score, float) else 0
        self.query_one("#detail-score", Label).update(
            f"Relevance: {score_pct}%"
        )

        # Update content with scene text
        content = self.query_one("#detail-content", Static)
        content.update(document_text)

    def action_close(self) -> None:
        """Close detail panel.
        
        Posts DetailPanelClosed message to parent screen.
        """
        self.post_message(DetailPanelClosed())

    def action_copy_id(self) -> None:
        """Copy scene ID to clipboard."""
        if self.result_data:
            scene_id = self.result_data["id"]
            # Copy to app clipboard
            self.app.copy_to_clipboard(scene_id)
            self.app.notify(
                f"Copied: {scene_id}",
                timeout=2,
            )

    def action_copy_text(self) -> None:
        """Copy full scene text to clipboard."""
        if self.result_data:
            text = self.result_data["text"]
            # Copy to app clipboard
            self.app.copy_to_clipboard(text)
            self.app.notify(
                "Scene text copied to clipboard",
                timeout=2,
            )

    def clear(self) -> None:
        """Clear the detail panel."""
        self.result_data = None
        self.query_one("#detail-title", Label).update("[No Result Selected]")
        self.query_one("#detail-content", Static).update(
            "[No content to display]"
        )
