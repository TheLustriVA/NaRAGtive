"""Widget for displaying and managing vector store list."""

from typing import Callable, Optional
from datetime import datetime

from textual.widget import Widget
from textual.reactive import reactive
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button
from textual.message import Message

from naragtive.store_registry import StoreMetadata


class StoreItem(Static):
    """Single store item in the list.
    
    Attributes:
        metadata: Store metadata to display
        is_default: Whether this is the default store
        is_selected: Whether this item is currently selected
    """

    DEFAULT_CSS = """
    StoreItem {
        width: 1fr;
        height: auto;
        border: solid $secondary;
        padding: 0 1;
        background: $panel;
    }
    
    StoreItem.selected {
        background: $primary;
        color: $text-selected;
        text-style: bold;
    }
    
    StoreItem:hover {
        background: $boost;
    }
    """

    is_selected: reactive[bool] = reactive(False, recompose=True)
    is_default: reactive[bool] = reactive(False, recompose=True)

    def __init__(
        self,
        metadata: StoreMetadata,
        is_default: bool = False,
        on_select: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize store item.
        
        Args:
            metadata: Store metadata to display
            is_default: Whether this is the default store
            on_select: Callback when item is selected
        """
        super().__init__()
        self.metadata = metadata
        self.is_default = is_default
        self.on_select = on_select

    def render(self) -> str:
        """Render the store item.
        
        Returns:
            Formatted string representation of the store
        """
        marker = "⭐" if self.is_default else " "
        
        # Format record count with thousands separator
        count_str = f"{self.metadata.record_count:,}"
        
        # Parse and format creation date
        try:
            created = datetime.fromisoformat(self.metadata.created_at)
            date_str = created.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = "unknown"
        
        # Truncate description to fit
        desc = self.metadata.description[:30] if self.metadata.description else ""
        
        # Format: [marker] [name] [type] [count] [date] [desc]
        line = (
            f"{marker} {self.metadata.name:<15} "
            f"[{self.metadata.source_type:<10}] "
            f"{count_str:>8} records  {date_str}  {desc}"
        )
        
        return line

    def on_mount(self) -> None:
        """Handle widget mount."""
        if self.is_selected:
            self.add_class("selected")

    def on_click(self) -> None:
        """Handle click on store item."""
        if self.on_select:
            self.on_select(self.metadata.name)


class StorePressedMessage(Message):
    """Posted when a store is selected.
    
    Attributes:
        store_name: Name of the selected store
    """

    def __init__(self, store_name: str) -> None:
        """Initialize message.
        
        Args:
            store_name: Name of the selected store
        """
        super().__init__()
        self.store_name = store_name


class StoreListWidget(Widget):
    """Display list of registered vector stores.
    
    Displays all stores from VectorStoreRegistry with metadata including
    name, type, record count, creation date, and default indicator.
    
    Attributes:
        stores: List of store metadata to display
        selected_index: Index of currently selected store
    """

    DEFAULT_CSS = """
    StoreListWidget {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
        background: $surface;
    }
    
    StoreListWidget > Vertical {
        width: 1fr;
        height: 1fr;
    }
    
    StoreListWidget > Vertical > Static {
        width: 1fr;
        height: auto;
    }
    """

    stores: reactive[list[StoreMetadata]] = reactive([], recompose=True)
    selected_index: reactive[int] = reactive(0, recompose=True)

    def __init__(self, stores: list[StoreMetadata] | None = None) -> None:
        """Initialize store list widget.
        
        Args:
            stores: Initial list of stores to display
        """
        super().__init__()
        self.stores = stores or []
        self.selected_index = 0

    def compose(self) -> list[Widget]:
        """Compose store list UI.
        
        Yields:
            Vertical container with store items
        """
        with Vertical():
            for idx, store in enumerate(self.stores):
                yield StoreItem(
                    store,
                    is_default=False,  # Will be updated by parent
                    on_select=self._on_store_selected,
                )

    def _on_store_selected(self, store_name: str) -> None:
        """Handle store selection.
        
        Args:
            store_name: Name of selected store
        """
        # Find index
        for idx, store in enumerate(self.stores):
            if store.name == store_name:
                self.selected_index = idx
                self.post_message(StorePressedMessage(store_name))
                break

    def update_stores(
        self,
        stores: list[StoreMetadata],
        default_store: str | None = None,
    ) -> None:
        """Update store list.
        
        Args:
            stores: New list of stores
            default_store: Name of default store
        """
        self.stores = stores
        self.default_store = default_store

    def get_selected_store(self) -> StoreMetadata | None:
        """Get currently selected store.
        
        Returns:
            Selected store metadata or None if empty
        """
        if 0 <= self.selected_index < len(self.stores):
            return self.stores[self.selected_index]
        return None
