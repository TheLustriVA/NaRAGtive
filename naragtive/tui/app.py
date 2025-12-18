"""Main NaRAGtive TUI application using Textual.

Provides a modern, keyboard-driven terminal interface for managing
vector stores and performing RAG operations.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer

from naragtive.tui.styles import APP_CSS
from naragtive.tui.screens.dashboard import DashboardScreen


class NaRAGtiveApp(App[None]):
    """Main NaRAGtive TUI application.
    
    A modern, keyboard-driven terminal interface for managing vector stores
    and performing RAG (Retrieval-Augmented Generation) operations.
    
    Attributes:
        TITLE: Application title
        SUBTITLE: Application subtitle
        CSS_PATH: Path to TCSS stylesheet
        BINDINGS: Global key bindings
    """

    TITLE = "NaRAGtive"
    SUBTITLE = "Vector Store Manager"
    
    # Load CSS from file
    CSS_PATH = str(APP_CSS)
    
    # Global keybindings
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+d", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose app UI.
        
        Creates the header, footer, and main content area.
        
        Yields:
            Header and Footer widgets
        """
        yield Header()
        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount.
        
        Starts with the dashboard screen when app starts.
        """
        self.push_screen(DashboardScreen())


def main() -> None:
    """Entry point for the TUI application.
    
    Can be run with:
        python -m naragtive.tui.app
    """
    app = NaRAGtiveApp()
    app.run()


if __name__ == "__main__":
    main()
