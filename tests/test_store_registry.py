"""Tests for VectorStoreRegistry multi-store management.

Covers registration, retrieval, persistence, and error handling.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import polars as pl

from naragtive.store_registry import VectorStoreRegistry, StoreMetadata


@pytest.fixture
def temp_registry_dir(tmp_path: Path) -> Path:
    """Create temporary registry directory."""
    registry_dir = tmp_path / ".naragtive" / "stores"
    registry_dir.mkdir(parents=True, exist_ok=True)
    return registry_dir


@pytest.fixture
def temp_parquet(tmp_path: Path) -> Path:
    """Create a temporary parquet file for testing."""
    df = pl.DataFrame({
        "id": ["scene_0001", "scene_0002"],
        "text": ["Sample scene 1", "Sample scene 2"],
        "embedding": [[0.1, 0.2], [0.3, 0.4]],
        "metadata": ['{"location": "bridge"}', '{"location": "medbay"}']
    })
    parquet_path = tmp_path / "test_store.parquet"
    df.write_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def registry_with_custom_dir(temp_registry_dir: Path, temp_parquet: Path, monkeypatch):
    """Create registry with temporary directory."""
    registry = VectorStoreRegistry()
    monkeypatch.setattr(registry, "REGISTRY_DIR", temp_registry_dir)
    monkeypatch.setattr(registry, "REGISTRY_FILE", temp_registry_dir / "registry.json")
    monkeypatch.setattr(registry, "DEFAULT_FILE", temp_registry_dir / "default.txt")
    registry._stores = {}  # Reset stores
    return registry, temp_parquet


class TestStoreMetadata:
    """Test StoreMetadata dataclass."""
    
    def test_metadata_creation(self) -> None:
        """Test creating metadata instance."""
        metadata = StoreMetadata(
            name="test-store",
            path="/path/to/store.parquet",
            created_at="2025-12-13T00:00:00",
            source_type="neptune",
            record_count=100,
            description="Test store"
        )
        
        assert metadata.name == "test-store"
        assert metadata.record_count == 100
    
    def test_metadata_to_dict(self) -> None:
        """Test metadata serialization to dict."""
        metadata = StoreMetadata(
            name="test",
            path="/path/store.parquet",
            created_at="2025-12-13T00:00:00",
            source_type="llama-server",
            record_count=50
        )
        
        data = metadata.to_dict()
        assert data["name"] == "test"
        assert data["record_count"] == 50
        assert data["description"] is None
    
    def test_metadata_from_dict(self) -> None:
        """Test metadata deserialization from dict."""
        data = {
            "name": "restored",
            "path": "/path/store.parquet",
            "created_at": "2025-12-13T00:00:00",
            "source_type": "chat",
            "record_count": 200,
            "description": "Restored store"
        }
        
        metadata = StoreMetadata.from_dict(data)
        assert metadata.name == "restored"
        assert metadata.record_count == 200
        assert metadata.description == "Restored store"


class TestVectorStoreRegistry:
    """Test VectorStoreRegistry class."""
    
    def test_register_store(self, registry_with_custom_dir) -> None:
        """Test registering a new store."""
        registry, temp_parquet = registry_with_custom_dir
        
        metadata = registry.register(
            name="campaign-1",
            path=temp_parquet,
            source_type="neptune",
            description="Campaign 1 scenes"
        )
        
        assert metadata.name == "campaign-1"
        assert metadata.source_type == "neptune"
        assert metadata.record_count == 2
        assert "campaign-1" in registry._stores
    
    def test_register_auto_detects_record_count(self, registry_with_custom_dir) -> None:
        """Test that record count is auto-detected if not provided."""
        registry, temp_parquet = registry_with_custom_dir
        
        metadata = registry.register(
            name="auto-count",
            path=temp_parquet,
            source_type="llama-server"
            # record_count not provided
        )
        
        assert metadata.record_count == 2  # Auto-detected
    
    def test_register_duplicate_name_error(self, registry_with_custom_dir) -> None:
        """Test that duplicate store names are rejected."""
        registry, temp_parquet = registry_with_custom_dir
        
        # Register first store
        registry.register(
            name="duplicate-test",
            path=temp_parquet,
            source_type="neptune"
        )
        
        # Try to register with same name
        with pytest.raises(ValueError, match="already exists"):
            registry.register(
                name="duplicate-test",
                path=temp_parquet,
                source_type="llama-server"
            )
    
    def test_register_missing_file_error(self, registry_with_custom_dir, tmp_path: Path) -> None:
        """Test that registering non-existent file fails."""
        registry, _ = registry_with_custom_dir
        
        with pytest.raises(FileNotFoundError, match="not found"):
            registry.register(
                name="missing",
                path=tmp_path / "nonexistent.parquet",
                source_type="neptune"
            )
    
    def test_get_store(self, registry_with_custom_dir) -> None:
        """Test retrieving a registered store."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register(
            name="test-store",
            path=temp_parquet,
            source_type="neptune"
        )
        
        store = registry.get("test-store")
        from naragtive.polars_vectorstore import PolarsVectorStore
        assert isinstance(store, PolarsVectorStore)
    
    def test_get_nonexistent_store_error(self, registry_with_custom_dir) -> None:
        """Test error when getting non-existent store."""
        registry, _ = registry_with_custom_dir
        
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")
    
    def test_list_stores(self, registry_with_custom_dir) -> None:
        """Test listing all registered stores."""
        registry, temp_parquet = registry_with_custom_dir
        
        # Register multiple stores
        registry.register("store-1", temp_parquet, "neptune")
        registry.register("store-2", temp_parquet, "llama-server")
        registry.register("store-3", temp_parquet, "chat")
        
        stores = registry.list_stores()
        assert len(stores) == 3
        assert [s.name for s in stores] == ["store-1", "store-2", "store-3"]
    
    def test_list_stores_empty(self, registry_with_custom_dir) -> None:
        """Test listing stores when registry is empty."""
        registry, _ = registry_with_custom_dir
        
        stores = registry.list_stores()
        assert len(stores) == 0
    
    def test_get_default_none_when_empty(self, registry_with_custom_dir) -> None:
        """Test get_default returns None when no stores."""
        registry, _ = registry_with_custom_dir
        
        default = registry.get_default()
        assert default is None
    
    def test_get_default_first_store(self, registry_with_custom_dir) -> None:
        """Test get_default returns first store when no explicit default."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("zebra", temp_parquet, "neptune")
        registry.register("alpha", temp_parquet, "llama-server")
        
        # First store alphabetically should be default
        default = registry.get_default()
        assert default == "alpha"
    
    def test_set_default(self, registry_with_custom_dir) -> None:
        """Test setting default store."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("store-1", temp_parquet, "neptune")
        registry.register("store-2", temp_parquet, "llama-server")
        
        registry.set_default("store-2")
        
        default = registry.get_default()
        assert default == "store-2"
    
    def test_set_default_nonexistent_error(self, registry_with_custom_dir) -> None:
        """Test error when setting default to non-existent store."""
        registry, _ = registry_with_custom_dir
        
        with pytest.raises(KeyError, match="not found"):
            registry.set_default("nonexistent")
    
    def test_delete_store(self, registry_with_custom_dir) -> None:
        """Test deleting a store from registry."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("to-delete", temp_parquet, "neptune")
        assert "to-delete" in registry._stores
        
        registry.delete("to-delete")
        assert "to-delete" not in registry._stores
    
    def test_delete_nonexistent_error(self, registry_with_custom_dir) -> None:
        """Test error when deleting non-existent store."""
        registry, _ = registry_with_custom_dir
        
        with pytest.raises(KeyError, match="not found"):
            registry.delete("nonexistent")
    
    def test_delete_clears_default(self, registry_with_custom_dir) -> None:
        """Test that deleting default store clears default file."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("only-store", temp_parquet, "neptune")
        registry.set_default("only-store")
        
        assert registry.get_default() == "only-store"
        
        registry.delete("only-store")
        
        # Default file should be deleted
        assert not registry.DEFAULT_FILE.exists()
    
    def test_rename_store(self, registry_with_custom_dir) -> None:
        """Test renaming a store."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("old-name", temp_parquet, "neptune")
        registry.rename("old-name", "new-name")
        
        assert "old-name" not in registry._stores
        assert "new-name" in registry._stores
    
    def test_rename_nonexistent_error(self, registry_with_custom_dir) -> None:
        """Test error when renaming non-existent store."""
        registry, _ = registry_with_custom_dir
        
        with pytest.raises(KeyError, match="not found"):
            registry.rename("nonexistent", "new-name")
    
    def test_rename_duplicate_error(self, registry_with_custom_dir, temp_parquet: Path) -> None:
        """Test error when renaming to existing name."""
        registry, _ = registry_with_custom_dir
        
        registry.register("existing", temp_parquet, "neptune")
        registry.register("to-rename", temp_parquet, "llama-server")
        
        with pytest.raises(ValueError, match="already exists"):
            registry.rename("to-rename", "existing")
    
    def test_rename_updates_default(self, registry_with_custom_dir) -> None:
        """Test that renaming default store updates default file."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("default-store", temp_parquet, "neptune")
        registry.set_default("default-store")
        
        registry.rename("default-store", "renamed-default")
        
        assert registry.get_default() == "renamed-default"


class TestVectorStoreRegistryPersistence:
    """Test registry persistence across instances."""
    
    def test_registry_persistence(self, registry_with_custom_dir) -> None:
        """Test that registry.json persists across instances."""
        registry1, temp_parquet = registry_with_custom_dir
        
        # Register store in first instance
        registry1.register(
            name="persistent",
            path=temp_parquet,
            source_type="neptune",
            description="Test persistence"
        )
        
        # Create new instance pointing to same registry directory
        registry2 = VectorStoreRegistry()
        import unittest.mock as mock
        registry2.REGISTRY_DIR = registry1.REGISTRY_DIR
        registry2.REGISTRY_FILE = registry1.REGISTRY_FILE
        registry2.DEFAULT_FILE = registry1.DEFAULT_FILE
        registry2._stores = registry2._load_registry()
        
        # Check store exists in new instance
        assert "persistent" in registry2._stores
        store_meta = registry2._stores["persistent"]
        assert store_meta.source_type == "neptune"
        assert store_meta.description == "Test persistence"
    
    def test_default_file_persistence(self, registry_with_custom_dir) -> None:
        """Test that default store choice persists."""
        registry1, temp_parquet = registry_with_custom_dir
        
        registry1.register("store-1", temp_parquet, "neptune")
        registry1.register("store-2", temp_parquet, "llama-server")
        registry1.set_default("store-2")
        
        # Create new instance
        registry2 = VectorStoreRegistry()
        registry2.REGISTRY_DIR = registry1.REGISTRY_DIR
        registry2.REGISTRY_FILE = registry1.REGISTRY_FILE
        registry2.DEFAULT_FILE = registry1.DEFAULT_FILE
        registry2._stores = registry2._load_registry()
        
        assert registry2.get_default() == "store-2"


class TestVectorStoreRegistryIntegration:
    """Integration tests with PolarsVectorStore."""
    
    def test_get_returns_working_store(self, registry_with_custom_dir) -> None:
        """Test that get() returns a working PolarsVectorStore."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("working", temp_parquet, "neptune")
        store = registry.get("working")
        
        # Should be able to load the store
        assert store.load()
        assert len(store.df) == 2
    
    def test_get_with_default_keyword(self, registry_with_custom_dir) -> None:
        """Test using 'default' keyword in get()."""
        registry, temp_parquet = registry_with_custom_dir
        
        registry.register("my-default", temp_parquet, "neptune")
        registry.set_default("my-default")
        
        store = registry.get("default")
        assert store.load()
