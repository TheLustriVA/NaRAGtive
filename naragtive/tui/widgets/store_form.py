"""Store creation form widget for NaRAGtive TUI.

Provides form for registering new vector stores with validation.
"""

from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Input, Label, Static
from textual.validation import Integer, Function


class StoreNameValidator:
    """Validator for store names.

    Store names must:
    - Be 1-50 characters
    - Contain only alphanumeric, underscore, hyphen
    - Start with letter or underscore
    """

    @staticmethod
    def validate(value: str) -> bool:
        """Validate store name.

        Args:
            value: Store name to validate

        Returns:
            True if valid, False otherwise
        """
        if not value or len(value) > 50:
            return False
        if not (value[0].isalpha() or value[0] == "_"):
            return False
        return all(c.isalnum() or c in "_-" for c in value)


class PathValidator:
    """Validator for parquet file paths.

    Paths must:
    - End with .parquet
    - Point to an existing file
    """

    @staticmethod
    def validate(value: str) -> bool:
        """Validate parquet file path.

        Args:
            value: File path to validate

        Returns:
            True if valid, False otherwise
        """
        if not value or not value.endswith(".parquet"):
            return False
        try:
            path = Path(value).expanduser().resolve()
            return path.exists() and path.is_file()
        except Exception:
            return False


class StoreForm(Static):
    """Form for creating new vector store.

    Collects store name, file path, and source type.
    Emits StoreCreated message on successful submission.

    Attributes:
        DEFAULT_CSS: Form styling
    """

    CSS = """
    StoreForm {
        width: 100%;
        height: auto;
        padding: 1 2;
    }

    StoreForm .form-group {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    StoreForm Label {
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }

    StoreForm Input {
        width: 100%;
        height: 1;
        margin-bottom: 1;
    }

    StoreForm #form-buttons {
        width: 100%;
        height: 1;
        layout: horizontal;
        dock: bottom;
    }

    StoreForm Button {
        flex: 1;
        margin-right: 1;
    }

    StoreForm Button:last-child {
        margin-right: 0;
    }
    """

    class StoreCreated:
        """Message emitted when store is created.

        Attributes:
            name: Store name
            path: File path
            source_type: Source type
        """

        def __init__(self, name: str, path: str, source_type: str) -> None:
            """Initialize message.

            Args:
                name: Store name
                path: File path
                source_type: Source type
            """
            self.name = name
            self.path = path
            self.source_type = source_type

    class StoreCreationCancelled:
        """Message emitted when form is cancelled."""

        pass

    def __init__(self, show_cancel: bool = True) -> None:
        """Initialize store form.

        Args:
            show_cancel: Whether to show cancel button. Default: True
        """
        super().__init__()
        self.show_cancel = show_cancel

    def compose(self) -> ComposeResult:
        """Compose form UI.

        Yields:
            Form input fields and buttons
        """
        with Vertical():
            with Container(classes="form-group"):
                yield Label("Store Name")
                yield Input(
                    id="store-name",
                    placeholder="e.g., campaign-1",
                )

            with Container(classes="form-group"):
                yield Label("File Path")
                yield Input(
                    id="store-path",
                    placeholder="e.g., /path/to/scenes.parquet",
                )

            with Container(classes="form-group"):
                yield Label("Source Type")
                yield Input(
                    id="store-type",
                    placeholder="e.g., neptune, llama-server, chat",
                )

            with Horizontal(id="form-buttons"):
                yield Button("Create", id="create-btn", variant="primary")
                if self.show_cancel:
                    yield Button("Cancel", id="cancel-btn", variant="default")

    def on_mount(self) -> None:
        """Focus first input on mount."""
        self.query_one("#store-name", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button pressed event
        """
        if event.button.id == "create-btn":
            self._submit_form()
        elif event.button.id == "cancel-btn":
            self.post_message(self.StoreCreationCancelled())

    def _submit_form(self) -> None:
        """Submit form with validation."""
        name = self.query_one("#store-name", Input).value.strip()
        path = self.query_one("#store-path", Input).value.strip()
        source_type = self.query_one("#store-type", Input).value.strip()

        # Validate name
        if not name:
            self._show_error("Store name is required")
            return

        if not StoreNameValidator.validate(name):
            self._show_error(
                "Invalid store name. Use letters, numbers, _ or -. Max 50 chars."
            )
            return

        # Validate path
        if not path:
            self._show_error("File path is required")
            return

        if not PathValidator.validate(path):
            self._show_error("Invalid path. Must be existing .parquet file.")
            return

        # Validate source type
        if not source_type:
            self._show_error("Source type is required")
            return

        # All valid - emit message
        self.post_message(self.StoreCreated(name, path, source_type))

    def _show_error(self, message: str) -> None:
        """Show error message.

        Args:
            message: Error message to display
        """
        # Could integrate with app notification system
        # For now, just print to console
        import sys

        print(f"Form error: {message}", file=sys.stderr)
