"""Results table widget for displaying search results.

Provides a DataTable widget with search results sorted by relevance.
"""

from typing import TYPE_CHECKING, Any
from enum import Enum

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Static
from textual.message import Message
from textual.reactive import reactive

from naragtive.tui.search_utils import format_relevance_score, parse_metadata

if TYPE_CHECKING:
    pass


class SortColumn(Enum):
    """Column to sort results by."""

    RELEVANCE = "relevance"
    DATE = "date"
    SCENE_ID = "id"


class ResultSelected(Message):
    """Posted when a result row is selected.
    
    Attributes:
        result_index: Index in results list
        result_id: Scene ID
    """

    def __init__(self, result_index: int, result_id: str) -> None:
        """Initialize result selected message.
        
        Args:
            result_index: Index in results
            result_id: Scene ID
        """
        super().__init__()
        self.result_index = result_index
        self.result_id = result_id


class ResultsTableWidget(DataTable):
    """DataTable widget displaying search results.
    
    Shows relevance scores, IDs, dates, locations, and POV characters.
    Supports sorting by different columns.
    
    Attributes:
        results: Current search results
        current_sort: Current sort column
        results_count: Number of results
    """

    CSS = """
    ResultsTableWidget {
        width: 100%;
        height: 1fr;
        border: solid $primary;
    }
    """

    BINDINGS = [
        ("enter", "select_row", "View"),
        ("r", "toggle_rerank", "Rerank"),
        ("s", "cycle_sort", "Sort"),
    ]

    results: reactive[list[dict[str, Any]]] = reactive([], recompose=False)
    current_sort: reactive[SortColumn] = reactive(SortColumn.RELEVANCE)
    results_count: reactive[int] = reactive(0)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize results table widget.
        
        Args:
            name: Name of widget
            id: ID of widget
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self._results_data: list[dict[str, Any]] = []
        self.cursor_type = "row"
        self.zebra_stripes = True

    def on_mount(self) -> None:
        """Handle widget mount.
        
        Sets up table columns.
        """
        # Add columns
        self.add_columns(
            "Relevance",
            "ID",
            "Date",
            "Location",
            "POV",
        )
        self.focus()

    def update_results(
        self,
        results: dict[str, Any],
        reranked: bool = False,
    ) -> None:
        """Update table with new search results.
        
        Args:
            results: Results dict from store.query() with keys:
                'ids', 'documents', 'scores', 'metadatas'
            reranked: Whether results are reranked
        """
        self.clear()
        self._results_data = []

        ids = results.get("ids", [])
        scores = results.get(
            "rerank_scores" if reranked else "scores", []
        )
        metadatas = results.get("metadatas", [])

        # Build data tuples for display
        for i, (scene_id, score, metadata) in enumerate(
            zip(ids, scores, metadatas)
        ):
            parsed = parse_metadata(metadata)
            
            # Format relevance score
            relevance_text = Text(
                format_relevance_score(score),
                style="green" if score > 0.8 else "yellow" if score > 0.6 else "red",
            )

            # Build row data
            row_data = (
                relevance_text,
                parsed["scene_id"],
                parsed["date"],
                parsed["location"],
                parsed["pov"],
            )

            # Store metadata for later retrieval
            self._results_data.append(
                {
                    "index": i,
                    "id": scene_id,
                    "score": score,
                    "metadata": metadata,
                    "parsed": parsed,
                }
            )

            # Add row to table
            self.add_row(*row_data, key=scene_id)

        self.results_count = len(self._results_data)

        # Show status
        if self.results_count == 0:
            self.post_message(
                Static("No results found", id="results-status")
            )

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected
    ) -> None:
        """Handle row selection.
        
        Args:
            event: Row selected event
        """
        # Find result by row key
        for result in self._results_data:
            if result["id"] == event.cursor_row:
                self.post_message(
                    ResultSelected(
                        result["index"],
                        result["id"],
                    )
                )
                break

    def action_select_row(self) -> None:
        """Trigger selection of current row."""
        if self.cursor_row >= 0 and self.cursor_row < len(self._results_data):
            result = self._results_data[self.cursor_row]
            self.post_message(
                ResultSelected(
                    result["index"],
                    result["id"],
                )
            )

    def action_toggle_rerank(self) -> None:
        """Trigger reranking action.
        
        This is handled by parent screen.
        """
        self.app.post_message(Static("Reranking..."))

    def action_cycle_sort(self) -> None:
        """Cycle through sort columns.
        
        Cycles: RELEVANCE -> DATE -> ID -> RELEVANCE
        """
        if self.current_sort == SortColumn.RELEVANCE:
            self.current_sort = SortColumn.DATE
        elif self.current_sort == SortColumn.DATE:
            self.current_sort = SortColumn.SCENE_ID
        else:
            self.current_sort = SortColumn.RELEVANCE

        self._resort_results()
        self.app.notify(
            f"Sorted by: {self.current_sort.value}",
            timeout=2,
        )

    def _resort_results(self) -> None:
        """Resort results based on current sort column."""
        if self.current_sort == SortColumn.DATE:
            # Sort by date (column index 2)
            self.sort("Date")
        elif self.current_sort == SortColumn.SCENE_ID:
            # Sort by ID (column index 1)
            self.sort("ID")
        else:
            # Sort by Relevance descending (default)
            # DataTable sorts ascending by default, so we need to reverse
            self.sort("Relevance", reverse=True)

    def get_selected_result(self) -> dict[str, Any] | None:
        """Get the currently selected result.
        
        Returns:
            Result dict or None if no selection
        """
        if self.cursor_row >= 0 and self.cursor_row < len(self._results_data):
            return self._results_data[self.cursor_row]
        return None

    def get_all_results(self) -> list[dict[str, Any]]:
        """Get all results data.
        
        Returns:
            List of result dicts
        """
        return self._results_data.copy()
