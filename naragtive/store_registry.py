"""
Vector Store Registry - Manage multiple named vector stores.

Provides a persistent registry system for tracking and managing multiple
Polars vector stores, enabling seamless switching between different projects
and narrative sources (Neptune RPG chapters, Perplexity chats, text datasets, etc.).

Storage:
  ~/.naragtive/stores/registry.json       - Store metadata and paths
  ~/.naragtive/stores/default.txt        - Current default store name

Usage:
  registry = VectorStoreRegistry()
  
  # Register a store
  metadata = registry.register(
      name="campaign-1",
      path=Path("./scenes.parquet"),
      source_type="neptune",
      description="Campaign 1 narrative scenes"
  )
  
  # Get a store
  store = registry.get("campaign-1")
  
  # Set default
  registry.set_default("campaign-1")
  
  # List all stores
  for meta in registry.list_stores():
      print(f"{meta.name}: {meta.record_count} records")
  
  # Print formatted table
  registry.print_table()
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import polars as pl

from naragtive.polars_vectorstore import PolarsVectorStore


@dataclass
class StoreMetadata:
    """Metadata for a registered vector store.
    
    Attributes:
        name: Unique store identifier (e.g., "campaign-1")
        path: Full path to parquet file
        created_at: ISO datetime when registered
        source_type: Origin type ("neptune", "llama-server", "chat", etc.)
        record_count: Number of scenes/entries in store
        description: Optional human-readable description
    """
    
    name: str
    path: Path
    created_at: str
    source_type: str
    record_count: int
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary with all fields, path converted to string
        """
        data = asdict(self)
        data['path'] = str(data['path'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoreMetadata':
        """Create from dictionary (e.g., from JSON).
        
        Args:
            data: Dictionary with store metadata
            
        Returns:
            StoreMetadata instance
        """
        data = data.copy()
        data['path'] = Path(data['path'])
        return cls(**data)


class VectorStoreRegistry:
    """Persistent registry for managing multiple vector stores.
    
    Manages registration, lookup, and switching between multiple vector stores
    stored as Polars parquet files. Provides default store functionality and
    persistent storage in ~/.naragtive/stores/.
    
    Attributes:
        REGISTRY_DIR: Path to registry directory (~/.naragtive/stores/)
        REGISTRY_FILE: Path to registry.json file
        DEFAULT_FILE: Path to default.txt file
    """
    
    REGISTRY_DIR: Path = Path.home() / ".naragtive" / "stores"
    REGISTRY_FILE: Path = REGISTRY_DIR / "registry.json"
    DEFAULT_FILE: Path = REGISTRY_DIR / "default.txt"
    
    def __init__(self) -> None:
        """Initialize registry, loading from disk or creating if needed.
        
        Creates ~/.naragtive/stores/ directory if it doesn't exist.
        Loads existing registry.json or starts with empty registry.
        """
        # Ensure registry directory exists
        self.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or start fresh
        self._stores: Dict[str, StoreMetadata] = self._load_registry()
    
    def register(
        self,
        name: str,
        path: Path,
        source_type: str,
        description: str = "",
        record_count: Optional[int] = None
    ) -> StoreMetadata:
        """Register a new vector store.
        
        Args:
            name: Unique store identifier
            path: Path to parquet file
            source_type: Source type (e.g., "neptune", "llama-server")
            description: Optional human-readable description
            record_count: Number of records. If None, auto-detect from parquet
            
        Returns:
            StoreMetadata for the registered store
            
        Raises:
            ValueError: If name already exists
            FileNotFoundError: If parquet file doesn't exist
        """
        # Validate name is unique
        if name in self._stores:
            raise ValueError(
                f"Store name '{name}' already exists in registry. "
                f"Use rename() to change existing store name."
            )
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path}\n"
                f"Make sure file exists before registering."
            )
        
        # Auto-detect record count if not provided
        if record_count is None:
            try:
                df = pl.read_parquet(path, columns=['id'])
                record_count = len(df)
            except Exception as e:
                raise ValueError(
                    f"Could not read parquet file {path}: {e}\n"
                    f"Provide record_count manually: "
                    f"registry.register(..., record_count=N)"
                )
        
        # Create metadata
        metadata = StoreMetadata(
            name=name,
            path=path,
            created_at=datetime.utcnow().isoformat(),
            source_type=source_type,
            record_count=record_count,
            description=description
        )
        
        # Save to registry
        self._stores[name] = metadata
        self._save_registry()
        
        return metadata
    
    def get(self, name: str) -> PolarsVectorStore:
        """Get PolarsVectorStore instance by name.
        
        Args:
            name: Store name or "default" for default store
            
        Returns:
            PolarsVectorStore instance configured for the store
            
        Raises:
            KeyError: If store name not found
        """
        # Handle "default" keyword
        if name == "default":
            default_name = self.get_default()
            if default_name is None:
                raise KeyError(
                    "No default store set. "
                    "Use: registry.set_default('store_name')"
                )
            name = default_name
        
        # Look up store
        if name not in self._stores:
            available = ", ".join(self._stores.keys()) if self._stores else "none"
            raise KeyError(
                f"Store '{name}' not found in registry.\n"
                f"Available stores: {available}\n"
                f"Use: python main.py list-stores"
            )
        
        metadata = self._stores[name]
        return PolarsVectorStore(str(metadata.path))
    
    def list_stores(self) -> List[StoreMetadata]:
        """List all registered stores.
        
        Returns:
            List of StoreMetadata objects, sorted by name
        """
        return sorted(self._stores.values(), key=lambda m: m.name)
    
    def get_default(self) -> Optional[str]:
        """Get current default store name.
        
        Returns:
            Store name, or None if no default set.
            If no explicit default, returns first store by name.
        """
        # Check for explicit default
        if self.DEFAULT_FILE.exists():
            try:
                default = self.DEFAULT_FILE.read_text().strip()
                if default in self._stores:
                    return default
            except Exception:
                pass
        
        # Return first store if any exist
        if self._stores:
            return sorted(self._stores.keys())[0]
        
        return None
    
    def set_default(self, name: str) -> None:
        """Set default store.
        
        Args:
            name: Store name to set as default
            
        Raises:
            KeyError: If store name doesn't exist
        """
        if name not in self._stores:
            available = ", ".join(self._stores.keys())
            raise KeyError(
                f"Store '{name}' not found in registry.\n"
                f"Available stores: {available}"
            )
        
        self.DEFAULT_FILE.write_text(name)
    
    def delete(self, name: str) -> None:
        """Delete a store from registry.
        
        Note: This only removes from registry, doesn't delete the parquet file.
        
        Args:
            name: Store name to delete
            
        Raises:
            KeyError: If store name doesn't exist
        """
        if name not in self._stores:
            raise KeyError(f"Store '{name}' not found in registry")
        
        del self._stores[name]
        self._save_registry()
        
        # Clear default if this was the default store
        if self.get_default() == name:
            if self.DEFAULT_FILE.exists():
                self.DEFAULT_FILE.unlink()
    
    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a registered store.
        
        Args:
            old_name: Current store name
            new_name: New store name
            
        Raises:
            KeyError: If old name doesn't exist
            ValueError: If new name already exists
        """
        if old_name not in self._stores:
            raise KeyError(f"Store '{old_name}' not found in registry")
        
        if new_name in self._stores:
            raise ValueError(
                f"Store '{new_name}' already exists. "
                f"Delete it first or use different name."
            )
        
        # Move metadata
        metadata = self._stores[old_name]
        metadata.name = new_name
        self._stores[new_name] = metadata
        del self._stores[old_name]
        self._save_registry()
        
        # Update default if this was the default store
        if self.get_default() == old_name:
            self.set_default(new_name)
    
    def print_table(self) -> None:
        """Print formatted table of all registered stores.
        
        Shows store name, source type, record count, creation date, and description.
        Marks default store with ⭐ symbol.
        """
        stores = self.list_stores()
        default = self.get_default()
        
        if not stores:
            print("\n" + "=" * 90)
            print("REGISTERED VECTOR STORES")
            print("=" * 90)
            print("\nNo stores registered.")
            print("Register your first store with:")
            print("  python main.py ingest-neptune -e export.txt --register 'store-name'")
            print("=" * 90 + "\n")
            return
        
        # Calculate column widths
        name_width = max(len(s.name) for s in stores) + 4
        type_width = max(len(s.source_type) for s in stores) + 2
        
        # Print header
        print("\n" + "=" * 90)
        print("REGISTERED VECTOR STORES")
        print("=" * 90)
        print()
        
        # Print each store
        for store in stores:
            marker = "⭐" if store.name == default else " "
            record_str = f"{store.record_count:,}"
            date_str = store.created_at[:10]  # ISO date only
            desc_str = store.description[:40] if store.description else ""
            
            print(
                f"{marker} {store.name:<{name_width}} {store.source_type:<{type_width}} "
                f"{record_str:>8} records  {date_str}  {desc_str}"
            )
        
        print()
        print("=" * 90)
        print(f"Default: {default}")
        print("Use 'python main.py set-default <name>' to change default")
        print("=" * 90 + "\n")
    
    # ========== Private Methods ==========
    
    def _load_registry(self) -> Dict[str, StoreMetadata]:
        """Load registry from disk or return empty if doesn't exist.
        
        Returns:
            Dictionary mapping store names to StoreMetadata
        """
        if not self.REGISTRY_FILE.exists():
            return {}
        
        try:
            data = json.loads(self.REGISTRY_FILE.read_text())
            return {
                name: StoreMetadata.from_dict(meta)
                for name, meta in data.items()
            }
        except Exception as e:
            print(
                f"⚠️  Warning: Could not load registry: {e}\n"
                f"   Starting with empty registry."
            )
            return {}
    
    def _save_registry(self) -> None:
        """Save registry to disk atomically.
        
        Writes to temporary file first, then renames to ensure atomicity.
        This prevents corruption if process is interrupted.
        """
        # Ensure directory exists
        self.REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        
        # Build data
        data = {
            name: meta.to_dict()
            for name, meta in self._stores.items()
        }
        
        # Write atomically (write to temp, then rename)
        temp_file = self.REGISTRY_FILE.with_suffix('.tmp')
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(self.REGISTRY_FILE)
