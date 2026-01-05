"""Reusable dialog widgets for NaRAGtive TUI.

Provides ConfirmDialog, InfoDialog, and other modal dialogs.
"""

from typing import Optional, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ConfirmDialog(ModalScreen[bool]):
    """Confirmation dialog that returns True/False.

    Allows user to confirm or cancel an action.

    Attributes:
        title: Dialog title
        message: Confirmation message
        confirm_text: Text for confirm button (default: "Confirm")
        cancel_text: Text for cancel button (default: "Cancel")
    """

    CSS = """
    ConfirmDialog {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #confirm-title {
        width: 100%;
        height: auto;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #confirm-message {
        width: 100%;
        height: auto;
        margin-bottom: 2;
    }

    #confirm-buttons {
        width: 100%;
        height: auto;
        layout: horizontal;
        align: right middle;
    }

    #confirm-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
    ) -> None:
        """Initialize confirm dialog.

        Args:
            title: Dialog title
            message: Confirmation message
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
        """
        super().__init__()
        self.title_text = title
        self.message_text = message
        self.confirm_button_text = confirm_text
        self.cancel_button_text = cancel_text

    def compose(self) -> ComposeResult:
        """Compose dialog UI.

        Yields:
            Dialog container with title, message, and buttons
        """
        with Vertical(id="confirm-dialog"):
            yield Label(self.title_text, id="confirm-title")
            yield Label(self.message_text, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button(
                    self.cancel_button_text,
                    id="btn-cancel",
                    variant="default",
                )
                yield Button(
                    self.confirm_button_text,
                    id="btn-confirm",
                    variant="primary",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        button_id = event.button.id
        if button_id == "btn-confirm":
            self.dismiss(True)
        elif button_id == "btn-cancel":
            self.dismiss(False)


class InfoDialog(ModalScreen[None]):
    """Information dialog for displaying messages.

    Provides a modal dialog to show information with an OK button.

    Attributes:
        title: Dialog title
        message: Information message
        ok_text: Text for OK button (default: "OK")
    """

    CSS = """
    InfoDialog {
        align: center middle;
    }

    #info-dialog {
        width: 60;
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #info-title {
        width: 100%;
        height: auto;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #info-message {
        width: 100%;
        height: auto;
        margin-bottom: 2;
    }

    #info-buttons {
        width: 100%;
        height: auto;
        align: center middle;
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
            ok_text: Text for OK button
        """
        super().__init__()
        self.title_text = title
        self.message_text = message
        self.ok_button_text = ok_text

    def compose(self) -> ComposeResult:
        """Compose dialog UI.

        Yields:
            Dialog container with title, message, and OK button
        """
        with Vertical(id="info-dialog"):
            yield Label(self.title_text, id="info-title")
            yield Label(self.message_text, id="info-message")
            with Horizontal(id="info-buttons"):
                yield Button(
                    self.ok_button_text,
                    id="btn-ok",
                    variant="primary",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        if event.button.id == "btn-ok":
            self.dismiss()
