"""Store manager screen for NaRAGtive TUI.

Provides interface to view, create, and delete vector stores.
"""

import asyncio
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Button, DataTable, Static, Label, Input
from rich.text import Text

from naragtive.store_registry import VectorStoreRegistry
from naragtive.tui.widgets.dialogs import ConfirmDialog, InfoDialog
from naragtive.tui.widgets.store_form import PathValidator


class StoreManagerScreen(Screen[None]):
    """Screen for managing vector stores.

    Features:
    - View all stores with metadata
    - Create new store
    - Delete store with confirmation
    - Set default store

    Key bindings:
        'n': New store
        'd': Delete selected store
        's': Set as default
        'Escape': Exit to dashboard

    Attributes:
        TITLE: Screen title
        BINDINGS: Key bindings
    """

    TITLE = "Store Manager"

    BINDINGS = [
        Binding("escape", "dismiss", "Exit", show=True),
        Binding("n", "new_store", "New", show=True),
        Binding("d", "delete_store", "Delete", show=True),
        Binding("s", "set_default", "Set Default", show=True),
    ]

    CSS = """
    StoreManagerScreen {
        layout: vertical;
    }

    #manager-header {
        width: 100%;
        height: auto;
        padding: 1 2;
        border-bottom: solid $accent;
    }

    #stores-table {
        width: 100%;
        height: 1fr;
    }

    #manager-status {
        width: 100%;
        height: 1;
        dock: bottom;
        padding: 0 2;
        background: $surface;
    }
    """

    def __init__(self) -> None:
        """Initialize store manager screen."""
        super().__init__()
        self.registry = VectorStoreRegistry()
        self.selected_row: Optional[int] = None

    def compose(self) -> ComposeResult:
        """Compose screen UI.

        Yields:
            Header, table, status bar, Footer
        """
        yield Header()
        yield Label(
            "Vector Stores - Press 'n' to create new, 'd' to delete, 's' to set default",
            id="manager-header",
        )
        yield DataTable(id="stores-table", show_header=True, show_cursor=True)
        yield Label("", id="manager-status")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.load_worker(self._load_stores())

    def action_dismiss(self) -> None:
        """Action to exit to dashboard."""
        self.dismiss()

    def action_new_store(self) -> None:
        """Action to create new store."""
        # For now, show info dialog
        # In full implementation, would show form
        self.app.push_screen(
            InfoDialog(
                "New Store",
                "Store creation form (Phase 3 TODO)",
            )
        )

    def action_delete_store(self) -> None:
        """Action to delete selected store."""
        if self.selected_row is None:
            self._update_status("[error]No store selected[/error]")
            return

        table = self.query_one("#stores-table", DataTable)
        try:
            # Get store name from first column
            store_name = table.get_cell(self.selected_row, 0)

            def on_confirm(confirmed: bool) -> None:
                if confirmed:
                    self.call_later(self._delete_store, store_name)

            self.app.push_screen(
                ConfirmDialog(
                    "Delete Store",
                    f"Delete store '{store_name}'? This cannot be undone.",
                    confirm_text="Delete",
                    cancel_text="Cancel",
                ),
                on_confirm,
            )
        except Exception as e:
            self._update_status(f"[error]Error: {str(e)}[/error]")

    def action_set_default(self) -> None:
        """Action to set selected store as default."""
        if self.selected_row is None:
            self._update_status("[error]No store selected[/error]")
            return

        table = self.query_one("#stores-table", DataTable)
        try:
            store_name = table.get_cell(self.selected_row, 0)
            self.call_later(self._set_default, store_name)
        except Exception as e:
            self._update_status(f"[error]Error: {str(e)}[/error]")

    async def _load_stores(self) -> None:
        """Load stores from registry."""
        try:
            loop = asyncio.get_event_loop()
            stores = await loop.run_in_executor(None, self.registry.list_stores)
            default = await loop.run_in_executor(None, self.registry.get_default)

            self._populate_table(stores, default)
            self._update_status(f"Loaded {len(stores)} store(s)")
        except Exception as e:
            self._update_status(f"[error]Error loading stores: {str(e)}[/error]")

    def _populate_table(
        self, stores: list, default: Optional[str] = None
    ) -> None:
        """Populate stores table.

        Args:
            stores: List of store metadata
            default: Name of default store
        """
        table = self.query_one("#stores-table", DataTable)
        table.clear()
        table.add_columns("Name", "Type", "Records", "Created", "Default")

        for store in stores:
            is_default = "✓" if store.name == default else ""
            table.add_row(
                store.name,
                store.source_type,
                str(store.record_count),
                str(store.created_at.date()) if store.created_at else "N/A",
                is_default,
            )

    async def _delete_store(self, store_name: str) -> None:
        """Delete a store.

        Args:
            store_name: Name of store to delete
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.registry.delete, store_name)
            self._update_status(f"Deleted store: {store_name}")
            await self._load_stores()
        except Exception as e:
            self._update_status(f"[error]Delete failed: {str(e)}[/error]")

    async def _set_default(self, store_name: str) -> None:
        """Set store as default.

        Args:
            store_name: Name of store to set as default
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.registry.set_default, store_name)
            self._update_status(f"Default store set to: {store_name}")
            await self._load_stores()
        except Exception as e:
            self._update_status(f"[error]Error: {str(e)}[/error]")

    def on_data_table_row_selected(self, event) -> None:
        """Handle row selection.

        Args:
            event: Row selected event
        """
        self.selected_row = event.cursor_row

    def _update_status(self, message: str) -> None:
        """Update status bar.

        Args:
            message: Status message (supports markup)
        """
        try:
            status = self.query_one("#manager-status", Label)
            status.update(Text.from_markup(message))
        except Exception:
            pass
