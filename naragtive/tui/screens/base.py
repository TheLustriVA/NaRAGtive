"""Base screen class for NaRAGtive TUI screens."""

from typing import TYPE_CHECKING

from textual.screen import Screen
from textual.containers import Container

if TYPE_CHECKING:
    from naragtive.tui.app import NaRAGtiveApp


class BaseScreen(Screen[None]):
    """Base screen for all NaRAGtive screens.
    
    Provides common functionality for navigation and app access.
    
    Attributes:
        BINDINGS: Global key bindings for this screen
    """

    BINDINGS = [
        ("ctrl+c", "app.quit", "Quit"),
        ("ctrl+d", "app.quit", "Quit"),
        ("f1", "show_help", "Help"),
    ]

    @property
    def app(self) -> "NaRAGtiveApp":
        """Get the app instance.
        
        Returns:
            The NaRAGtiveApp instance
        """
        return super().app  # type: ignore

    def action_show_help(self) -> None:
        """Show help/keybindings."""
        # TODO: Implement help screen in Phase 2
        self.notify("Help not yet implemented", title="Help", timeout=3)

    def action_back(self) -> None:
        """Navigate back to previous screen."""
        self.app.pop_screen()
