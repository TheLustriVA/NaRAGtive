"""Reusable dialog widgets for NaRAGtive TUI.

Provides ConfirmDialog and InfoDialog for common user interactions.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmDialog(ModalScreen[bool]):
    """Confirmation dialog that returns True (confirm) or False (cancel).

    A modal dialog for confirmation prompts. User can press:
    - Enter or click 'Confirm' to return True
    - Escape or click 'Cancel' to return False

    Attributes:
        title: Dialog title
        message: Confirmation message
    """

    CSS = """
    ConfirmDialog {
        align: center middle;
    }

    #dialog-container {
        width: 60;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #dialog-message {
        width: 100%;
        margin-bottom: 1;
        text-align: left;
    }

    #dialog-buttons {
        width: 100%;
        height: auto;
        layout: horizontal;
        dock: bottom;
    }

    #dialog-buttons Button {
        flex: 1;
        margin-right: 1;
    }

    #dialog-buttons Button:last-child {
        margin-right: 0;
    }
    """

    def __init__(
        self,
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
    ) -> None:
        """Initialize confirmation dialog.

        Args:
            title: Dialog title
            message: Confirmation message
            confirm_text: Text for confirm button. Default: "Confirm"
            cancel_text: Text for cancel button. Default: "Cancel"
        """
        super().__init__()
        self.dialog_title = title
        self.dialog_message = message
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text

    def compose(self) -> ComposeResult:
        """Compose dialog UI.

        Yields:
            Container with message and button widgets
        """
        with Container(id="dialog-container"):
            yield Label(self.dialog_message, id="dialog-message")
            with Container(id="dialog-buttons"):
                yield Button(self.confirm_text, id="confirm-btn", variant="primary")
                yield Button(self.cancel_text, id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        """Set title and focus confirm button on mount."""
        self.title = self.dialog_title
        self.query_one("#confirm-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        if event.button.id == "confirm-btn":
            self.dismiss(True)
        elif event.button.id == "cancel-btn":
            self.dismiss(False)

    def action_dismiss_confirm(self) -> None:
        """Action to dismiss as confirmed (triggered by Enter)."""
        self.dismiss(True)

    def action_dismiss_cancel(self) -> None:
        """Action to dismiss as cancelled (triggered by Escape)."""
        self.dismiss(False)


class InfoDialog(ModalScreen[None]):
    """Information display dialog.

    A modal dialog for displaying information. User can press:
    - Enter or click 'OK' to close
    - Escape to close

    Attributes:
        title: Dialog title
        message: Information message
    """

    CSS = """
    InfoDialog {
        align: center middle;
    }

    #info-container {
        width: 70;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #info-message {
        width: 100%;
        margin-bottom: 1;
        text-align: left;
        height: auto;
    }

    #info-buttons {
        width: 100%;
        height: auto;
        layout: horizontal;
        dock: bottom;
    }

    #info-buttons Button {
        width: 100%;
    }
    """

    def __init__(
        self,
        title: str,
        message: str,
        ok_text: str = "OK",
    ) -> None:
        """Initialize info dialog.

        Args:
            title: Dialog title
            message: Information message
            ok_text: Text for OK button. Default: "OK"
        """
        super().__init__()
        self.info_title = title
        self.info_message = message
        self.ok_text = ok_text

    def compose(self) -> ComposeResult:
        """Compose dialog UI.

        Yields:
            Container with message and OK button
        """
        with Container(id="info-container"):
            yield Label(self.info_message, id="info-message")
            with Container(id="info-buttons"):
                yield Button(self.ok_text, id="ok-btn", variant="primary")

    def on_mount(self) -> None:
        """Set title and focus OK button on mount."""
        self.title = self.info_title
        self.query_one("#ok-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press.

        Args:
            event: Button pressed event
        """
        if event.button.id == "ok-btn":
            self.dismiss()

    def action_dismiss_ok(self) -> None:
        """Action to dismiss (triggered by Enter or Escape)."""
        self.dismiss()
