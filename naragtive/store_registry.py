#!/usr/bin/env python3
"""
Vector Store Registry: Manage multiple parquet stores with named aliases.

Provides a registry system for organizing and switching between multiple
vector stores, with persistent metadata tracking and default store support.

Usage:
    registry = VectorStoreRegistry()
    store_meta = registry.register(
        name="campaign-1",
        path=Path("scenes.parquet"),
        source_type="neptune",
        description="Campaign 1 scenes"
    )
    store = registry.get("campaign-1")
    registry.set_default("campaign-1")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import polars as pl
from naragtive.polars_vectorstore import PolarsVectorStore


@dataclass
class StoreMetadata:
    """Metadata about a registered vector store.
    
    Attributes:
        name: Unique store identifier (e.g., "campaign-1", "perplexity-research")
        path: Full path to the parquet file
        created_at: ISO datetime string when store was registered
        source_type: Origin of data ("neptune", "llama-server", "chat", etc.)
        record_count: Number of scenes/entries in the store
        description: Optional human-readable description
    """
    name: str
    path: str
    created_at: str
    source_type: str
    record_count: int
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of metadata
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StoreMetadata:
        """Create from dictionary.
        
        Args:
            data: Dictionary with metadata fields
            
        Returns:
            StoreMetadata instance
        """
        return cls(**data)


class VectorStoreRegistry:
    """Registry for managing multiple vector stores.
    
    Maintains a registry of named stores with metadata, allowing users to
    reference stores by name instead of full file paths. Stores metadata
    in ~/.naragtive/stores/registry.json and tracks default store in
    ~/.naragtive/stores/default.txt.
    
    Attributes:
        REGISTRY_DIR: Directory for store metadata (~/.naragtive/stores)
        REGISTRY_FILE: Path to registry.json file
        DEFAULT_FILE: Path to default.txt file
    """

    REGISTRY_DIR = Path.home() / ".naragtive" / "stores"
    REGISTRY_FILE = REGISTRY_DIR / "registry.json"
    DEFAULT_FILE = REGISTRY_DIR / "default.txt"

    def __init__(self) -> None:
        """Initialize registry and load existing stores."""
        # Create registry directory if needed
        self.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing stores
        self._stores: Dict[str, StoreMetadata] = self._load_registry()

    def _load_registry(self) -> Dict[str, StoreMetadata]:
        """Load registry from JSON file.
        
        Returns:
            Dictionary mapping store names to StoreMetadata
        """
        if not self.REGISTRY_FILE.exists():
            return {}
        
        try:
            with open(self.REGISTRY_FILE, "r") as f:
                data = json.load(f)
            return {name: StoreMetadata.from_dict(meta) for name, meta in data.items()}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Warning: Could not load registry: {e}")
            return {}

    def _save_registry(self) -> None:
        """Save registry to JSON file atomically.
        
        Writes to a temporary file first, then atomically moves to registry.json.
        """
        # Create temp file in same directory (atomic rename on same filesystem)
        temp_file = self.REGISTRY_FILE.with_suffix(".tmp")
        
        data = {name: meta.to_dict() for name, meta in self._stores.items()}
        
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename
        temp_file.replace(self.REGISTRY_FILE)

    def register(
        self,
        name: str,
        path: Path | str,
        source_type: str,
        description: Optional[str] = None,
        record_count: Optional[int] = None,
    ) -> StoreMetadata:
        """Register a new vector store.
        
        Args:
            name: Unique identifier for this store (e.g., "campaign-1")
            path: Path to the parquet file
            source_type: Source origin ("neptune", "llama-server", "chat", etc.)
            description: Optional description of store contents
            record_count: Number of records (auto-detected if not provided)
            
        Returns:
            StoreMetadata for the registered store
            
        Raises:
            ValueError: If store name already exists
            FileNotFoundError: If parquet file doesn't exist
        """
        path = Path(path)
        
        # Validate store name
        if name in self._stores:
            raise ValueError(f"Store '{name}' already exists. Use a different name.")
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        
        # Auto-detect record count if not provided
        if record_count is None:
            try:
                df = pl.read_parquet(path)
                record_count = len(df)
            except Exception as e:
                raise ValueError(f"Could not read parquet file {path}: {e}")
        
        # Create metadata
        metadata = StoreMetadata(
            name=name,
            path=str(path),
            created_at=datetime.utcnow().isoformat(),
            source_type=source_type,
            record_count=record_count,
            description=description,
        )
        
        # Store and persist
        self._stores[name] = metadata
        self._save_registry()
        
        return metadata

    def get(self, name: str) -> PolarsVectorStore:
        """Get a PolarsVectorStore instance by name.
        
        Args:
            name: Store name, or "default" to use default store
            
        Returns:
            PolarsVectorStore instance pointing to correct parquet
            
        Raises:
            KeyError: If store not found
        """
        # Handle "default" keyword
        if name == "default":
            default_name = self.get_default()
            if default_name is None:
                raise KeyError(
                    "No default store set. Register a store first: "
                    "python main.py ingest-neptune -e export.txt --register 'store-name'"
                )
            name = default_name
        
        # Look up store
        if name not in self._stores:
            available = ", ".join(sorted(self._stores.keys()))
            raise KeyError(
                f"Store '{name}' not found.\n"
                f"Available stores: {available}\n"
                f"Register new store: python main.py ingest-neptune -e file.txt --register 'name'"
            )
        
        # Return PolarsVectorStore instance
        metadata = self._stores[name]
        return PolarsVectorStore(metadata.path)

    def list_stores(self) -> List[StoreMetadata]:
        """Get all registered stores.
        
        Returns:
            List of StoreMetadata sorted by name
        """
        return sorted(self._stores.values(), key=lambda m: m.name)

    def get_default(self) -> Optional[str]:
        """Get default store name.
        
        Returns:
            Default store name, or None if no stores exist
            
        Priority:
        1. Store name in ~/.naragtive/stores/default.txt (if exists)
        2. First registered store (alphabetically)
        3. None if no stores
        """
        # Check for explicit default file
        if self.DEFAULT_FILE.exists():
            name = self.DEFAULT_FILE.read_text().strip()
            if name in self._stores:
                return name
        
        # Fall back to first store
        if self._stores:
            return sorted(self._stores.keys())[0]
        
        return None

    def set_default(self, name: str) -> None:
        """Set the default store.
        
        Args:
            name: Store name to set as default
            
        Raises:
            KeyError: If store not found
        """
        if name not in self._stores:
            raise KeyError(f"Store '{name}' not found. Run 'list-stores' to see available stores.")
        
        # Write default file atomically
        temp_file = self.DEFAULT_FILE.with_suffix(".tmp")
        temp_file.write_text(name)
        temp_file.replace(self.DEFAULT_FILE)

    def delete(self, name: str) -> None:
        """Delete a store from registry.
        
        Note: This only removes from registry, doesn't delete the parquet file.
        
        Args:
            name: Store name to delete
            
        Raises:
            KeyError: If store not found
        """
        if name not in self._stores:
            raise KeyError(f"Store '{name}' not found.")
        
        del self._stores[name]
        self._save_registry()
        
        # Clear default if this was the default store
        if self.DEFAULT_FILE.exists():
            if self.DEFAULT_FILE.read_text().strip() == name:
                self.DEFAULT_FILE.unlink()

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a registered store.
        
        Args:
            old_name: Current store name
            new_name: New store name
            
        Raises:
            KeyError: If old store not found
            ValueError: If new name already exists
        """
        if old_name not in self._stores:
            raise KeyError(f"Store '{old_name}' not found.")
        
        if new_name in self._stores:
            raise ValueError(f"Store '{new_name}' already exists.")
        
        # Rename
        metadata = self._stores.pop(old_name)
        metadata.name = new_name
        self._stores[new_name] = metadata
        self._save_registry()
        
        # Update default if needed
        if self.DEFAULT_FILE.exists():
            if self.DEFAULT_FILE.read_text().strip() == old_name:
                self.set_default(new_name)

    def print_table(self) -> None:
        """Print registry as a formatted table.
        
        Shows all stores with their metadata. Marks default store with ⚠️.
        """
        if not self._stores:
            print("\nNo registered stores. Ingest narratives with:")
            print("  python main.py ingest-neptune -e export.txt --register 'name'")
            return
        
        default = self.get_default()
        
        print("\n" + "=" * 90)
        print("REGISTERED VECTOR STORES")
        print("=" * 90)
        
        # Header
        print(f"{'':3} {'Name':<25} {'Type':<15} {'Records':<10} {'Created':<19} {'Description':<20}")
        print("-" * 90)
        
        # Rows
        for store in self.list_stores():
            marker = "⚠️" if store.name == default else " "
            desc = (store.description[:18] + "...") if store.description and len(store.description) > 20 else (store.description or "")
            print(
                f"{marker:3} {store.name:<25} {store.source_type:<15} "
                f"{store.record_count:<10} {store.created_at:<19} {desc:<20}"
            )
        
        print("=" * 90)
        print(f"\nDefault: {default or 'None'}")
        print("Use 'python main.py set-default <name>' to change default\n")
