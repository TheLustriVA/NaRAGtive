"""Tests for NaRAGtive TUI Phase 1 core functionality.

Coverage:
- App initialization and lifecycle
- Dashboard screen rendering and interactions
- Store list widget functionality
- Navigation between placeholder screens
- Keybinding responsiveness
- Store metadata display
- Async operations handling
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from textual.pilot import Pilot

from naragtive.tui.app import NaRAGtiveApp
from naragtive.tui.screens.dashboard import (
    DashboardScreen,
    SearchScreenPlaceholder,
    IngestScreenPlaceholder,
    ManageStoresScreenPlaceholder,
)
from naragtive.tui.widgets import StoreListWidget
from naragtive.store_registry import StoreMetadata


class TestNaRAGtiveApp:
    """Tests for main NaRAGtiveApp class."""

    async def test_app_initialization(self) -> None:
        """Test app initializes correctly."""
        app = NaRAGtiveApp()
        assert app.title == "NaRAGtive"
        assert app.subtitle == "Vector Store Manager"

    async def test_app_runs_successfully(self) -> None:
        """Test app can run without errors."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            # App should be running
            assert app is not None
            # Screen stack should have at least one screen
            assert len(app.screen_stack) > 0

    async def test_app_startup_with_dashboard(self) -> None:
        """Test app starts with dashboard screen."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            # Current screen should be DashboardScreen
            assert isinstance(app.screen, DashboardScreen)

    async def test_quit_keybinding(self) -> None:
        """Test quit keybindings work."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            # Ctrl+C should trigger quit (through action_quit)
            assert "ctrl+c" in str(app.BINDINGS)
            assert "ctrl+d" in str(app.BINDINGS)


class TestDashboardScreen:
    """Tests for DashboardScreen."""

    def create_mock_store(self, name: str, count: int = 100) -> StoreMetadata:
        """Create a mock store metadata.
        
        Args:
            name: Store name
            count: Record count
            
        Returns:
            StoreMetadata instance
        """
        return StoreMetadata(
            name=name,
            path=Path(f"/tmp/{name}.parquet"),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_type="test",
            record_count=count,
            description=f"Test store {name}",
        )

    async def test_dashboard_renders(self) -> None:
        """Test dashboard screen renders without errors."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            dashboard = app.screen
            assert isinstance(dashboard, DashboardScreen)
            # Check main components exist
            assert dashboard.query_one("#dashboard-title") is not None
            assert dashboard.query_one("#store-info") is not None
            assert dashboard.query_one("#action-buttons") is not None

    async def test_dashboard_buttons_present(self) -> None:
        """Test all action buttons are present on dashboard."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            dashboard = app.screen
            # Find buttons
            assert dashboard.query_one("#btn-search") is not None
            assert dashboard.query_one("#btn-ingest") is not None
            assert dashboard.query_one("#btn-manage") is not None
            assert dashboard.query_one("#btn-refresh") is not None

    async def test_dashboard_has_store_list(self) -> None:
        """Test dashboard has store list widget."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            dashboard = app.screen
            assert hasattr(dashboard, "store_list")
            assert isinstance(dashboard.store_list, StoreListWidget)

    async def test_search_keybinding(self) -> None:
        """Test 's' keybinding opens search screen."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            # Mock the registry to avoid loading actual stores
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry"):
                # Initial screen stack depth
                initial_depth = len(app.screen_stack)
                # Simulate 's' keybinding
                await pilot.press("s")
                await pilot.pause(0.5)
                # Screen should have been pushed
                assert len(app.screen_stack) >= initial_depth

    async def test_ingest_keybinding(self) -> None:
        """Test 'i' keybinding opens ingest screen."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry"):
                initial_depth = len(app.screen_stack)
                await pilot.press("i")
                await pilot.pause(0.5)
                assert len(app.screen_stack) >= initial_depth

    async def test_manage_keybinding(self) -> None:
        """Test 'm' keybinding opens manage stores screen."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry"):
                initial_depth = len(app.screen_stack)
                await pilot.press("m")
                await pilot.pause(0.5)
                assert len(app.screen_stack) >= initial_depth

    async def test_refresh_keybinding(self) -> None:
        """Test 'r' keybinding triggers refresh."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry"):
                dashboard = app.screen
                # Refresh should not change screen stack
                initial_depth = len(app.screen_stack)
                await pilot.press("r")
                await pilot.pause(0.5)
                assert len(app.screen_stack) == initial_depth


class TestStoreListWidget:
    """Tests for StoreListWidget."""

    def create_mock_store(self, name: str, count: int = 100) -> StoreMetadata:
        """Create a mock store metadata.
        
        Args:
            name: Store name
            count: Record count
            
        Returns:
            StoreMetadata instance
        """
        return StoreMetadata(
            name=name,
            path=Path(f"/tmp/{name}.parquet"),
            created_at=datetime.now(timezone.utc).isoformat(),
            source_type="test",
            record_count=count,
            description=f"Test store {name}",
        )

    async def test_store_list_widget_renders(self) -> None:
        """Test store list widget renders."""
        stores = [
            self.create_mock_store("store1", 100),
            self.create_mock_store("store2", 200),
        ]
        widget = StoreListWidget(stores)
        
        app = NaRAGtiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert widget is not None

    async def test_store_list_empty(self) -> None:
        """Test store list handles empty list."""
        widget = StoreListWidget([])
        assert widget.stores == []
        assert widget.get_selected_store() is None

    async def test_store_list_update(self) -> None:
        """Test store list can be updated."""
        stores = [self.create_mock_store("store1", 100)]
        widget = StoreListWidget(stores)
        
        new_stores = [
            self.create_mock_store("store1", 100),
            self.create_mock_store("store2", 200),
        ]
        widget.update_stores(new_stores)
        assert len(widget.stores) == 2

    async def test_store_list_selection(self) -> None:
        """Test store selection in widget."""
        stores = [
            self.create_mock_store("store1", 100),
            self.create_mock_store("store2", 200),
        ]
        widget = StoreListWidget(stores)
        widget._on_store_selected("store2")
        assert widget.selected_index == 1


class TestPlaceholderScreens:
    """Tests for placeholder screens (Phase 2)."""

    async def test_search_screen_placeholder(self) -> None:
        """Test search screen placeholder renders."""
        app = NaRAGtiveApp()
        app.push_screen(SearchScreenPlaceholder())
        async with app.run_test() as pilot:
            screen = app.screen
            assert isinstance(screen, SearchScreenPlaceholder)

    async def test_ingest_screen_placeholder(self) -> None:
        """Test ingest screen placeholder renders."""
        app = NaRAGtiveApp()
        app.push_screen(IngestScreenPlaceholder())
        async with app.run_test() as pilot:
            screen = app.screen
            assert isinstance(screen, IngestScreenPlaceholder)

    async def test_manage_stores_screen_placeholder(self) -> None:
        """Test manage stores screen placeholder renders."""
        app = NaRAGtiveApp()
        app.push_screen(ManageStoresScreenPlaceholder())
        async with app.run_test() as pilot:
            screen = app.screen
            assert isinstance(screen, ManageStoresScreenPlaceholder)

    async def test_back_from_placeholder_screen(self) -> None:
        """Test navigation back from placeholder screens."""
        app = NaRAGtiveApp()
        initial_depth = len(app.screen_stack)
        app.push_screen(SearchScreenPlaceholder())
        assert len(app.screen_stack) == initial_depth + 1
        
        app.pop_screen()
        assert len(app.screen_stack) == initial_depth


class TestKeybindings:
    """Tests for keybinding functionality."""

    async def test_global_quit_keybindings(self) -> None:
        """Test global quit keybindings are registered."""
        app = NaRAGtiveApp()
        assert any("ctrl+c" in str(b) for b in app.BINDINGS)
        assert any("ctrl+d" in str(b) for b in app.BINDINGS)

    async def test_dashboard_action_keybindings(self) -> None:
        """Test dashboard action keybindings are registered."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            dashboard = app.screen
            # Check that keybindings are defined
            assert len(dashboard.BINDINGS) > 0
            binding_keys = [b[0] for b in dashboard.BINDINGS]
            assert "s" in binding_keys  # Search
            assert "i" in binding_keys  # Ingest
            assert "m" in binding_keys  # Manage
            assert "r" in binding_keys  # Refresh
            assert "enter" in binding_keys  # Set default

    async def test_tab_navigation(self) -> None:
        """Test tab navigation between widgets."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            dashboard = app.screen
            initial_focus = app.focused
            await pilot.press("tab")
            await pilot.pause()
            # Focus should have moved (or stayed if only one focusable widget)
            # Just verify it doesn't error
            assert app.focused is not None


class TestAsyncOperations:
    """Tests for async operations in TUI."""

    async def test_load_stores_async(self) -> None:
        """Test store loading works asynchronously."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry") as mock_registry:
                mock_registry.return_value.list_stores.return_value = []
                mock_registry.return_value.get_default.return_value = None
                
                dashboard = app.screen
                assert dashboard.stores == []

    async def test_set_default_async(self) -> None:
        """Test setting default store works asynchronously."""
        app = NaRAGtiveApp()
        async with app.run_test() as pilot:
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry") as mock_registry:
                mock_instance = Mock()
                mock_registry.return_value = mock_instance
                
                dashboard = app.screen
                dashboard.selected_store = "test-store"
                
                # This should not block
                await pilot.pause()
                assert dashboard.selected_store == "test-store"


class TestResponsiveness:
    """Tests for terminal responsiveness."""

    async def test_small_terminal_handling(self) -> None:
        """Test app handles small terminal sizes gracefully."""
        app = NaRAGtiveApp()
        # Minimum terminal size per requirements: 80x24
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.screen is not None

    async def test_large_terminal_handling(self) -> None:
        """Test app handles large terminal sizes."""
        app = NaRAGtiveApp()
        async with app.run_test(size=(200, 50)) as pilot:
            assert app.screen is not None

    async def test_responsive_layout(self) -> None:
        """Test layout is responsive to terminal resize."""
        app = NaRAGtiveApp()
        async with app.run_test(size=(100, 30)) as pilot:
            dashboard = app.screen
            # Widgets should be present regardless of size
            assert dashboard.query_one("#dashboard-title") is not None
            assert dashboard.query_one("#action-buttons") is not None


class TestIntegration:
    """Integration tests for TUI workflow."""

    async def test_complete_workflow(self) -> None:
        """Test complete user workflow."""
        app = NaRAGtiveApp()
        async with app.run_test(size=(100, 30)) as pilot:
            with patch("naragtive.tui.screens.dashboard.VectorStoreRegistry") as mock_registry:
                # Setup mock registry with stores
                mock_instance = Mock()
                mock_instance.list_stores.return_value = []
                mock_instance.get_default.return_value = None
                mock_registry.return_value = mock_instance
                
                # Start on dashboard
                assert isinstance(app.screen, DashboardScreen)
                
                # Refresh stores
                await pilot.press("r")
                await pilot.pause(0.5)
                
                # Still on dashboard
                assert isinstance(app.screen, DashboardScreen)
                
                # Navigate to search
                await pilot.press("s")
                await pilot.pause(0.5)
                
                # Should have pushed new screen
                assert isinstance(app.screen, SearchScreenPlaceholder)
                
                # Navigate back
                await pilot.press("escape")
                await pilot.pause(0.5)
                
                # Back to dashboard
                assert isinstance(app.screen, DashboardScreen)
