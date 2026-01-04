"""Store management screen for NaRAGtive TUI.

Provides UI for managing registered vector stores including viewing,
creating, deleting, and setting default store.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static

from naragtive.store_registry import VectorStoreRegistry
from naragtive.tui.widgets.dialogs import ConfirmDialog, InfoDialog
from naragtive.tui.widgets.store_form import StoreForm


class StoreManagerScreen(Screen[None]):
    """Screen for managing vector stores.

    Displays list of registered stores and provides actions to:
    - View store details (name, type, records, created date)
    - Create new store
    - Delete store (with confirmation)
    - Set default store (keystroke 's')

    Key bindings:
        'n': New store
        'd': Delete selected store
        's': Set selected as default
        'Escape': Return to dashboard

    Attributes:
        TITLE: Screen title
        BINDINGS: Key bindings
    """

    TITLE = "Store Manager"

    BINDINGS = [
        Binding("n", "new_store", "New Store", show=True),
        Binding("d", "delete_store", "Delete", show=True),
        Binding("s", "set_default", "Set Default", show=True),
        Binding("escape", "dismiss", "Back", show=True),
    ]

    CSS = """
    StoreManagerScreen {
        layout: vertical;
    }

    #store-table {
        width: 100%;
        height: 1fr;
    }

    #store-info {
        width: 100%;
        height: auto;
        border-top: solid $accent;
        padding: 1;
        background: $surface;
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
        """Initialize store manager screen."""
        super().__init__()
        self.registry = VectorStoreRegistry()
        self._current_form: Optional[StoreForm] = None

    def compose(self) -> ComposeResult:
        """Compose screen UI.

        Yields:
            Header, DataTable showing stores, info panel, and button bar
        """
        yield Header()

        with Vertical():
            yield DataTable(id="store-table", show_header=True, show_cursor=True)
            yield Label("", id="store-info")

        with Horizontal(id="button-bar"):
            yield Button("New", id="new-btn", variant="primary")
            yield Button("Delete", id="delete-btn", variant="warning")
            yield Button("Set Default", id="default-btn", variant="accent")
            yield Button("Back", id="back-btn", variant="default")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize table and load stores on mount."""
        table = self.query_one("#store-table", DataTable)
        table.add_columns(
            "Store Name", "Type", "Records", "Created", "Description"
        )
        self._refresh_stores()
        table.focus()

    def on_data_table_row_selected(
        self, event: DataTable.RowSelected
    ) -> None:
        """Handle store selection in table.

        Args:
            event: Row selected event
        """
        self._update_info_panel()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        if event.button.id == "new-btn":
            self.action_new_store()
        elif event.button.id == "delete-btn":
            self.action_delete_store()
        elif event.button.id == "default-btn":
            self.action_set_default()
        elif event.button.id == "back-btn":
            self.action_dismiss()

    def action_new_store(self) -> None:
        """Action to create new store."""
        def on_store_created(msg: StoreForm.StoreCreated) -> None:
            """Handle store creation.

            Args:
                msg: Store created message
            """
            try:
                path = Path(msg.path).expanduser().resolve()
                self.registry.register(
                    name=msg.name,
                    path=path,
                    source_type=msg.source_type,
                )
                self._refresh_stores()
                self.post_message(
                    InfoDialog(
                        "Success",
                        f"Store '{msg.name}' created successfully!",
                    )
                )
            except Exception as e:
                self.post_message(
                    InfoDialog(
                        "Error",
                        f"Failed to create store: {str(e)}",
                    )
                )

        def on_cancel() -> None:
            """Handle form cancellation."""
            self.pop_screen()

        form = StoreForm()
        self._current_form = form
        self.push_screen(form)

    def action_delete_store(self) -> None:
        """Action to delete selected store."""
        table = self.query_one("#store-table", DataTable)
        if table.row_count == 0:
            self.notify("No stores to delete", severity="error")
            return

        try:
            cursor_row = table.cursor_row
            stores = self.registry.list_stores()
            if cursor_row >= len(stores):
                return

            store = stores[cursor_row]

            def confirm_delete(result: bool) -> None:
                """Confirm deletion.

                Args:
                    result: True if user confirmed
                """
                if result:
                    try:
                        self.registry.delete(store.name)
                        self._refresh_stores()
                        self.notify(
                            f"Store '{store.name}' deleted",
                            severity="warning",
                        )
                    except Exception as e:
                        self.notify(f"Delete failed: {str(e)}", severity="error")

            dialog = ConfirmDialog(
                "Delete Store",
                f"Delete '{store.name}'? This cannot be undone.",
                confirm_text="Delete",
                cancel_text="Cancel",
            )
            self.push_screen(dialog, confirm_delete)
        except Exception as e:
            self.notify(f"Error: {str(e)}", severity="error")

    def action_set_default(self) -> None:
        """Action to set selected store as default."""
        table = self.query_one("#store-table", DataTable)
        if table.row_count == 0:
            self.notify("No stores available", severity="error")
            return

        try:
            cursor_row = table.cursor_row
            stores = self.registry.list_stores()
            if cursor_row >= len(stores):
                return

            store = stores[cursor_row]
            self.registry.set_default(store.name)
            self._refresh_stores()
            self.notify(f"Default store set to '{store.name}'")
        except Exception as e:
            self.notify(f"Error: {str(e)}", severity="error")

    def action_dismiss(self) -> None:
        """Action to return to dashboard."""
        self.dismiss()

    def _refresh_stores(self) -> None:
        """Reload stores from registry and update table."""
        table = self.query_one("#store-table", DataTable)
        table.clear()

        try:
            stores = self.registry.list_stores()
            default = self.registry.get_default()

            for store in stores:
                marker = "⭐ " if store.name == default else ""
                created_date = store.created_at.split("T")[0]

                table.add_row(
                    f"{marker}{store.name}",
                    store.source_type,
                    str(store.record_count),
                    created_date,
                    store.description or "-",
                )

            if stores:
                self._update_info_panel()
        except Exception as e:
            self.notify(f"Failed to load stores: {str(e)}", severity="error")

    def _update_info_panel(self) -> None:
        """Update info panel with selected store details."""
        table = self.query_one("#store-table", DataTable)
        info = self.query_one("#store-info", Label)

        try:
            if table.row_count == 0:
                info.update("No stores registered")
                return

            cursor_row = table.cursor_row
            stores = self.registry.list_stores()
            if cursor_row >= len(stores):
                return

            store = stores[cursor_row]
            is_default = "⭐ YES" if store.name == self.registry.get_default() else "-"

            info_text = (
                f"Store: [bold]{store.name}[/bold]  "
                f"Type: {store.source_type}  "
                f"Records: {store.record_count}  "
                f"Default: {is_default}"
            )
            info.update(Text.from_markup(info_text))
        except Exception:
            pass
