"""Store form widget for NaRAGtive TUI.

Provides form for creating and editing vector stores with validation.
"""

import re
from pathlib import Path
from typing import Optional


class StoreNameValidator:
    """Validator for store names.

    Rules:
    - Length: 1-50 characters
    - Start with letter or underscore
    - Contains only alphanumeric, underscore, hyphen
    - No spaces or special characters
    """

    MIN_LENGTH = 1
    MAX_LENGTH = 50
    PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_-]*$"

    @classmethod
    def validate(cls, name: str) -> bool:
        """Validate store name.

        Args:
            name: Store name to validate

        Returns:
            True if valid, False otherwise
        """
        if not name:
            return False

        if len(name) < cls.MIN_LENGTH or len(name) > cls.MAX_LENGTH:
            return False

        return bool(re.match(cls.PATTERN, name))

    @classmethod
    def get_error_message(cls, name: str) -> Optional[str]:
        """Get validation error message if any.

        Args:
            name: Store name to validate

        Returns:
            Error message or None if valid
        """
        if not name:
            return "Name cannot be empty"

        if len(name) < cls.MIN_LENGTH:
            return f"Name must be at least {cls.MIN_LENGTH} character"

        if len(name) > cls.MAX_LENGTH:
            return f"Name must be at most {cls.MAX_LENGTH} characters"

        if not re.match(cls.PATTERN, name):
            return "Name must start with letter or underscore, contain only alphanumeric, underscore, or hyphen"

        return None


class PathValidator:
    """Validator for file paths.

    Rules:
    - File must exist
    - File must have .parquet extension
    - File must be readable
    """

    REQUIRED_EXTENSION = ".parquet"

    @classmethod
    def validate(cls, path_str: str) -> bool:
        """Validate file path.

        Args:
            path_str: Path to validate

        Returns:
            True if valid, False otherwise
        """
        if not path_str:
            return False

        # Expand tilde
        path = Path(path_str).expanduser()

        # Check existence
        if not path.exists():
            return False

        # Check is file
        if not path.is_file():
            return False

        # Check extension
        if path.suffix.lower() != cls.REQUIRED_EXTENSION:
            return False

        # Check readable
        if not path.readable():
            return False

        return True

    @classmethod
    def get_error_message(cls, path_str: str) -> Optional[str]:
        """Get validation error message if any.

        Args:
            path_str: Path to validate

        Returns:
            Error message or None if valid
        """
        if not path_str:
            return "Path cannot be empty"

        path = Path(path_str).expanduser()

        if not path.exists():
            return f"File not found: {path}"

        if not path.is_file():
            return f"Not a file: {path}"

        if path.suffix.lower() != cls.REQUIRED_EXTENSION:
            return f"File must have {cls.REQUIRED_EXTENSION} extension"

        if not path.readable():
            return f"File is not readable: {path}"

        return None
