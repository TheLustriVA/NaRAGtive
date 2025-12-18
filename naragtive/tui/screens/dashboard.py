"""Dashboard screen showing registered vector stores."""

import asyncio
from typing import TYPE_CHECKING

from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button, Label
from textual.reactive import reactive

from naragtive.store_registry import StoreMetadata, VectorStoreRegistry
from naragtive.tui.screens.base import BaseScreen
from naragtive.tui.widgets import StoreListWidget
from naragtive.tui.widgets.store_list import StorePressedMessage

if TYPE_CHECKING:
    from naragtive.tui.app import NaRAGtiveApp


class SearchScreenPlaceholder(BaseScreen):
    """Placeholder for Search screen (Phase 2)."""

    BINDINGS = [
        ("escape", "back", "Back"),
    ] + BaseScreen.BINDINGS

    def compose(self):
        """Compose search screen."""
        yield Static("Search Screen (Phase 2)", id="placeholder-content")


class IngestScreenPlaceholder(BaseScreen):
    """Placeholder for Ingest screen (Phase 2)."""

    BINDINGS = [
        ("escape", "back", "Back"),
    ] + BaseScreen.BINDINGS

    def compose(self):
        """Compose ingest screen."""
        yield Static("Ingest Screen (Phase 2)", id="placeholder-content")


class ManageStoresScreenPlaceholder(BaseScreen):
    """Placeholder for Manage Stores screen (Phase 2)."""

    BINDINGS = [
        ("escape", "back", "Back"),
    ] + BaseScreen.BINDINGS

    def compose(self):
        """Compose manage stores screen."""
        yield Static("Manage Stores Screen (Phase 2)", id="placeholder-content")


class DashboardScreen(BaseScreen):
    """Dashboard screen showing registered vector stores.
    
    Displays all registered stores with metadata and provides quick access
    to search, ingest, and store management functions.
    
    Attributes:
        stores: List of registered stores
        selected_store: Currently selected store name
        default_store: Name of default store
    """

    BINDINGS = [
        ("s", "search", "Search"),
        ("i", "ingest", "Ingest"),
        ("m", "manage", "Manage"),
        ("r", "refresh", "Refresh"),
        ("enter", "set_default", "Set Default"),
        ("tab", "focus_next", "Focus Next"),
        ("shift+tab", "focus_previous", "Focus Prev"),
    ] + BaseScreen.BINDINGS

    CSS = """
    DashboardScreen {
        layout: vertical;
    }
    
    #dashboard-header {
        width: 1fr;
        height: auto;
        background: $boost;
    }
    
    #dashboard-title {
        width: 1fr;
        height: auto;
        content-align: center middle;
        text-style: bold;
        background: $boost;
    }
    
    #store-info {
        width: 1fr;
        height: auto;
        background: $panel;
        padding: 1;
    }
    
    #store-list-container {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
        background: $surface;
    }
    
    #action-buttons {
        width: 1fr;
        height: auto;
        layout: horizontal;
        background: $panel;
        padding: 1;
    }
    
    #action-buttons Button {
        margin: 0 1;
    }
    """

    stores: reactive[list[StoreMetadata]] = reactive([], recompose=True)
    selected_store: reactive[str | None] = reactive(None)
    default_store: reactive[str | None] = reactive(None)

    def __init__(self) -> None:
        """Initialize dashboard screen."""
        super().__init__()
        self._registry: VectorStoreRegistry | None = None

    def compose(self):
        """Compose dashboard UI.
        
        Yields:
            Dashboard widgets and containers
        """
        with Vertical(id="dashboard-container"):
            yield Static("NaRAGtive - Vector Store Dashboard", id="dashboard-title")
            
            # Store info bar
            yield Label("Click on a store to select it. Press 'r' to refresh.", id="store-info")
            
            # Store list
            self.store_list = StoreListWidget()
            yield self.store_list
            
            # Action buttons
            with Horizontal(id="action-buttons"):
                yield Button("Search (s)", id="btn-search", variant="primary")
                yield Button("Ingest (i)", id="btn-ingest", variant="primary")
                yield Button("Manage (m)", id="btn-manage", variant="default")
                yield Button("Refresh (r)", id="btn-refresh", variant="default")

    async def on_mount(self) -> None:
        """Handle screen mount."""
        await self._load_stores()
        self.set_focus(self.store_list)

    async def _load_stores(self) -> None:
        """Load stores from registry.
        
        Runs registry access in executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        try:
            # Run blocking registry access in executor
            registry = await loop.run_in_executor(None, VectorStoreRegistry)
            stores = await loop.run_in_executor(None, registry.list_stores)
            default = await loop.run_in_executor(None, registry.get_default)
            
            self.stores = stores
            self.default_store = default
            
            # Update widget
            if hasattr(self, "store_list"):
                self.store_list.update_stores(stores, default)
            
            # Update status
            self._update_store_info()
        except Exception as e:
            self.app.notify(f"Error loading stores: {e}", severity="error", timeout=5)

    def _update_store_info(self) -> None:
        """Update store info display."""
        if not self.stores:
            info_text = "No stores registered. Use 'i' to ingest data."
        else:
            count = len(self.stores)
            default = f" (default: {self.default_store})" if self.default_store else ""
            info_text = f"{count} store(s) registered{default}"
        
        info_widget = self.query_one("#store-info", Label)
        info_widget.update(info_text)

    def on_store_list_widget_store_pressed_message(
        self, message: StorePressedMessage
    ) -> None:
        """Handle store selection.
        
        Args:
            message: Store pressed message containing store name
        """
        self.selected_store = message.store_name

    def action_search(self) -> None:
        """Open search screen."""
        self.app.push_screen(SearchScreenPlaceholder())

    def action_ingest(self) -> None:
        """Open ingest screen."""
        self.app.push_screen(IngestScreenPlaceholder())

    def action_manage(self) -> None:
        """Open manage stores screen."""
        self.app.push_screen(ManageStoresScreenPlaceholder())

    def action_refresh(self) -> None:
        """Refresh store list."""
        self.call_later(self._load_stores)

    def action_set_default(self) -> None:
        """Set selected store as default.
        
        Uses run_in_executor to avoid blocking the event loop.
        """
        if not self.selected_store:
            self.app.notify("No store selected", severity="warning", timeout=3)
            return
        
        async def _set_default() -> None:
            loop = asyncio.get_event_loop()
            try:
                registry = await loop.run_in_executor(None, VectorStoreRegistry)
                await loop.run_in_executor(
                    None, registry.set_default, self.selected_store
                )
                self.app.notify(
                    f"Default store set to: {self.selected_store}",
                    timeout=3
                )
                await self._load_stores()
            except Exception as e:
                self.app.notify(
                    f"Error setting default: {e}",
                    severity="error",
                    timeout=5
                )
        
        self.call_later(_set_default)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.
        
        Args:
            event: Button pressed event
        """
        button_id = event.button.id
        if button_id == "btn-search":
            self.action_search()
        elif button_id == "btn-ingest":
            self.action_ingest()
        elif button_id == "btn-manage":
            self.action_manage()
        elif button_id == "btn-refresh":
            self.action_refresh()
