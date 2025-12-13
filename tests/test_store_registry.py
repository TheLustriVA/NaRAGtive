"""
Tests for VectorStoreRegistry - multi-store management system.

Tests cover:
- StoreMetadata serialization/deserialization
- Store registration with validation
- Registry persistence (load/save)
- Default store management
- Store renaming and deletion
- Integration with PolarsVectorStore
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import polars as pl

from naragtive.store_registry import StoreMetadata, VectorStoreRegistry


@pytest.fixture
def temp_registry_dir():
    """Provide isolated temp directory for registry testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry_with_temp(temp_registry_dir):
    """VectorStoreRegistry using temp directory."""
    with patch.object(VectorStoreRegistry, 'REGISTRY_DIR', temp_registry_dir):
        with patch.object(VectorStoreRegistry, 'REGISTRY_FILE', temp_registry_dir / 'registry.json'):
            with patch.object(VectorStoreRegistry, 'DEFAULT_FILE', temp_registry_dir / 'default.txt'):
                registry = VectorStoreRegistry()
                yield registry


@pytest.fixture
def temp_parquet_file(temp_registry_dir):
    """Create a temporary valid parquet file."""
    df = pl.DataFrame({
        'id': ['scene_001', 'scene_002', 'scene_003'],
        'text': ['text1', 'text2', 'text3'],
        'embedding': [[0.1] * 384, [0.2] * 384, [0.3] * 384],
        'metadata': ['{}', '{}', '{}']
    })
    parquet_path = temp_registry_dir / "test.parquet"
    df.write_parquet(parquet_path)
    return parquet_path


# ============================================================================
# StoreMetadata Tests
# ============================================================================

class TestStoreMetadata:
    """Test StoreMetadata dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = StoreMetadata(
            name="test-store",
            path=Path("/path/to/file.parquet"),
            created_at="2025-12-13T12:00:00",
            source_type="neptune",
            record_count=100,
            description="Test store"
        )
        data = meta.to_dict()
        
        assert data['name'] == "test-store"
        assert data['path'] == "/path/to/file.parquet"
        assert data['source_type'] == "neptune"
        assert data['record_count'] == 100
        assert data['description'] == "Test store"
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'name': 'test-store',
            'path': '/path/to/file.parquet',
            'created_at': '2025-12-13T12:00:00',
            'source_type': 'neptune',
            'record_count': 100,
            'description': 'Test store'
        }
        meta = StoreMetadata.from_dict(data)
        
        assert meta.name == 'test-store'
        assert meta.path == Path('/path/to/file.parquet')
        assert meta.source_type == 'neptune'
        assert meta.record_count == 100
    
    def test_serialization_roundtrip(self):
        """Test that to_dict/from_dict are inverse operations."""
        original = StoreMetadata(
            name="roundtrip",
            path=Path("/path/to/test.parquet"),
            created_at="2025-12-13T12:00:00",
            source_type="llama",
            record_count=250,
            description="Roundtrip test"
        )
        
        recovered = StoreMetadata.from_dict(original.to_dict())
        
        assert recovered.name == original.name
        assert recovered.path == original.path
        assert recovered.created_at == original.created_at
        assert recovered.source_type == original.source_type
        assert recovered.record_count == original.record_count
        assert recovered.description == original.description


# ============================================================================
# VectorStoreRegistry Tests
# ============================================================================

class TestVectorStoreRegistry:
    """Test VectorStoreRegistry core functionality."""
    
    def test_register_valid_store(self, registry_with_temp, temp_parquet_file):
        """Test registering a valid store."""
        meta = registry_with_temp.register(
            name="campaign-1",
            path=temp_parquet_file,
            source_type="neptune",
            description="Campaign 1 scenes"
        )
        
        assert meta.name == "campaign-1"
        assert meta.source_type == "neptune"
        assert meta.record_count == 3  # Our temp file has 3 records
        assert meta.description == "Campaign 1 scenes"
    
    def test_register_auto_detect_record_count(self, registry_with_temp, temp_parquet_file):
        """Test auto-detection of record count from parquet."""
        meta = registry_with_temp.register(
            name="test",
            path=temp_parquet_file,
            source_type="chat"
        )
        
        assert meta.record_count == 3  # Detected from file
    
    def test_register_explicit_record_count(self, registry_with_temp, temp_parquet_file):
        """Test explicit record count override."""
        meta = registry_with_temp.register(
            name="test",
            path=temp_parquet_file,
            source_type="chat",
            record_count=999
        )
        
        assert meta.record_count == 999
    
    def test_register_duplicate_name_error(self, registry_with_temp, temp_parquet_file):
        """Test error when registering duplicate name."""
        registry_with_temp.register(
            name="store",
            path=temp_parquet_file,
            source_type="neptune"
        )
        
        with pytest.raises(ValueError, match="already exists"):
            registry_with_temp.register(
                name="store",
                path=temp_parquet_file,
                source_type="llama"
            )
    
    def test_register_missing_file_error(self, registry_with_temp):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            registry_with_temp.register(
                name="missing",
                path=Path("/nonexistent/file.parquet"),
                source_type="neptune"
            )
    
    def test_register_invalid_parquet_error(self, registry_with_temp, temp_registry_dir):
        """Test error when parquet file is corrupted."""
        bad_file = temp_registry_dir / "bad.parquet"
        bad_file.write_text("not a parquet file")
        
        with pytest.raises(ValueError, match="Could not read"):
            registry_with_temp.register(
                name="bad",
                path=bad_file,
                source_type="neptune"
            )
    
    def test_get_by_name(self, registry_with_temp, temp_parquet_file):
        """Test retrieving store by name."""
        registry_with_temp.register(
            name="my-store",
            path=temp_parquet_file,
            source_type="neptune"
        )
        
        store = registry_with_temp.get("my-store")
        assert store is not None
        # Store is PolarsVectorStore instance
        assert hasattr(store, 'parquet_path')
    
    def test_get_nonexistent_error(self, registry_with_temp):
        """Test error when getting nonexistent store."""
        with pytest.raises(KeyError, match="not found"):
            registry_with_temp.get("nonexistent")
    
    def test_get_default_keyword(self, registry_with_temp, temp_parquet_file):
        """Test get() with 'default' keyword."""
        registry_with_temp.register(
            name="store1",
            path=temp_parquet_file,
            source_type="neptune"
        )
        registry_with_temp.set_default("store1")
        
        store = registry_with_temp.get("default")
        assert store is not None
    
    def test_list_stores_empty(self, registry_with_temp):
        """Test listing empty registry."""
        stores = registry_with_temp.list_stores()
        assert stores == []
    
    def test_list_stores_populated(self, registry_with_temp, temp_parquet_file):
        """Test listing populated registry."""
        registry_with_temp.register("store1", temp_parquet_file, "neptune")
        registry_with_temp.register("store2", temp_parquet_file, "llama")
        
        stores = registry_with_temp.list_stores()
        assert len(stores) == 2
        # Should be sorted by name
        assert stores[0].name == "store1"
        assert stores[1].name == "store2"
    
    def test_set_default(self, registry_with_temp, temp_parquet_file):
        """Test setting default store."""
        registry_with_temp.register("store1", temp_parquet_file, "neptune")
        registry_with_temp.set_default("store1")
        
        assert registry_with_temp.get_default() == "store1"
    
    def test_set_default_nonexistent_error(self, registry_with_temp):
        """Test error when setting nonexistent store as default."""
        with pytest.raises(KeyError):
            registry_with_temp.set_default("nonexistent")
    
    def test_get_default_first_store(self, registry_with_temp, temp_parquet_file):
        """Test that get_default returns first store if no explicit default."""
        registry_with_temp.register("store-a", temp_parquet_file, "neptune")
        registry_with_temp.register("store-b", temp_parquet_file, "llama")
        
        # No explicit default set, should return first by name
        default = registry_with_temp.get_default()
        assert default == "store-a"
    
    def test_delete_store(self, registry_with_temp, temp_parquet_file):
        """Test deleting a store."""
        registry_with_temp.register("store", temp_parquet_file, "neptune")
        registry_with_temp.delete("store")
        
        assert len(registry_with_temp.list_stores()) == 0
    
    def test_delete_nonexistent_error(self, registry_with_temp):
        """Test error when deleting nonexistent store."""
        with pytest.raises(KeyError):
            registry_with_temp.delete("nonexistent")
    
    def test_delete_clears_default(self, registry_with_temp, temp_parquet_file):
        """Test that deleting default store clears it."""
        registry_with_temp.register("store", temp_parquet_file, "neptune")
        registry_with_temp.set_default("store")
        registry_with_temp.delete("store")
        
        # Default should be cleared (no stores left)
        assert registry_with_temp.get_default() is None
    
    def test_rename_store(self, registry_with_temp, temp_parquet_file):
        """Test renaming a store."""
        registry_with_temp.register("old-name", temp_parquet_file, "neptune")
        registry_with_temp.rename("old-name", "new-name")
        
        assert "new-name" in [s.name for s in registry_with_temp.list_stores()]
        assert "old-name" not in [s.name for s in registry_with_temp.list_stores()]
    
    def test_rename_nonexistent_error(self, registry_with_temp):
        """Test error when renaming nonexistent store."""
        with pytest.raises(KeyError):
            registry_with_temp.rename("old", "new")
    
    def test_rename_duplicate_error(self, registry_with_temp, temp_parquet_file):
        """Test error when renaming to existing name."""
        registry_with_temp.register("store1", temp_parquet_file, "neptune")
        registry_with_temp.register("store2", temp_parquet_file, "llama")
        
        with pytest.raises(ValueError, match="already exists"):
            registry_with_temp.rename("store1", "store2")
    
    def test_rename_updates_default(self, registry_with_temp, temp_parquet_file):
        """Test that renaming default store updates it."""
        registry_with_temp.register("store", temp_parquet_file, "neptune")
        registry_with_temp.set_default("store")
        registry_with_temp.rename("store", "renamed")
        
        assert registry_with_temp.get_default() == "renamed"


# ============================================================================
# Persistence Tests
# ============================================================================

class TestVectorStoreRegistryPersistence:
    """Test registry persistence to disk."""
    
    def test_registry_saves_to_file(self, registry_with_temp, temp_parquet_file):
        """Test that registry is saved to file."""
        registry_with_temp.register("store", temp_parquet_file, "neptune")
        
        # File should exist and be valid JSON
        registry_file = registry_with_temp.REGISTRY_FILE
        assert registry_file.exists()
        
        data = json.loads(registry_file.read_text())
        assert "store" in data
    
    def test_registry_loads_from_file(self, temp_registry_dir, temp_parquet_file):
        """Test that registry loads from disk."""
        # Create and populate first registry
        with patch.object(VectorStoreRegistry, 'REGISTRY_DIR', temp_registry_dir):
            with patch.object(VectorStoreRegistry, 'REGISTRY_FILE', temp_registry_dir / 'registry.json'):
                with patch.object(VectorStoreRegistry, 'DEFAULT_FILE', temp_registry_dir / 'default.txt'):
                    reg1 = VectorStoreRegistry()
                    reg1.register("persistent", temp_parquet_file, "neptune")
        
        # Create new registry instance - should load from disk
        with patch.object(VectorStoreRegistry, 'REGISTRY_DIR', temp_registry_dir):
            with patch.object(VectorStoreRegistry, 'REGISTRY_FILE', temp_registry_dir / 'registry.json'):
                with patch.object(VectorStoreRegistry, 'DEFAULT_FILE', temp_registry_dir / 'default.txt'):
                    reg2 = VectorStoreRegistry()
                    stores = reg2.list_stores()
                    assert len(stores) == 1
                    assert stores[0].name == "persistent"
    
    def test_default_store_persists(self, temp_registry_dir, temp_parquet_file):
        """Test that default store setting persists."""
        with patch.object(VectorStoreRegistry, 'REGISTRY_DIR', temp_registry_dir):
            with patch.object(VectorStoreRegistry, 'REGISTRY_FILE', temp_registry_dir / 'registry.json'):
                with patch.object(VectorStoreRegistry, 'DEFAULT_FILE', temp_registry_dir / 'default.txt'):
                    reg1 = VectorStoreRegistry()
                    reg1.register("store", temp_parquet_file, "neptune")
                    reg1.set_default("store")
        
        # Load again
        with patch.object(VectorStoreRegistry, 'REGISTRY_DIR', temp_registry_dir):
            with patch.object(VectorStoreRegistry, 'REGISTRY_FILE', temp_registry_dir / 'registry.json'):
                with patch.object(VectorStoreRegistry, 'DEFAULT_FILE', temp_registry_dir / 'default.txt'):
                    reg2 = VectorStoreRegistry()
                    assert reg2.get_default() == "store"
    
    def test_atomic_writes(self, registry_with_temp, temp_parquet_file):
        """Test that registry writes are atomic (safe from interruption)."""
        # Register multiple stores
        for i in range(5):
            registry_with_temp.register(
                f"store{i}",
                temp_parquet_file,
                "neptune"
            )
        
        # Registry file should never have temp file
        assert not registry_with_temp.REGISTRY_FILE.with_suffix('.tmp').exists()
        
        # File should be valid JSON
        data = json.loads(registry_with_temp.REGISTRY_FILE.read_text())
        assert len(data) == 5


# ============================================================================
# Integration Tests
# ============================================================================

class TestVectorStoreRegistryIntegration:
    """Test integration with PolarsVectorStore."""
    
    def test_get_returns_polars_vectorstore(self, registry_with_temp, temp_parquet_file):
        """Test that get() returns proper PolarsVectorStore instance."""
        registry_with_temp.register("test", temp_parquet_file, "neptune")
        
        store = registry_with_temp.get("test")
        
        # Should be PolarsVectorStore
        assert type(store).__name__ == "PolarsVectorStore"
        assert hasattr(store, 'parquet_path')
    
    def test_multiple_stores_independent(self, registry_with_temp, temp_registry_dir):
        """Test that multiple stores are independent."""
        # Create two different parquet files
        df1 = pl.DataFrame({
            'id': ['s1_001', 's1_002'],
            'text': ['text1', 'text2'],
            'embedding': [[0.1] * 384, [0.2] * 384],
            'metadata': ['{}', '{}']
        })
        file1 = temp_registry_dir / "store1.parquet"
        df1.write_parquet(file1)
        
        df2 = pl.DataFrame({
            'id': ['s2_001', 's2_002', 's2_003'],
            'text': ['text1', 'text2', 'text3'],
            'embedding': [[0.1] * 384, [0.2] * 384, [0.3] * 384],
            'metadata': ['{}', '{}', '{}']
        })
        file2 = temp_registry_dir / "store2.parquet"
        df2.write_parquet(file2)
        
        # Register both
        registry_with_temp.register("store1", file1, "neptune")
        registry_with_temp.register("store2", file2, "llama")
        
        # Verify independent
        store1 = registry_with_temp.get("store1")
        store2 = registry_with_temp.get("store2")
        
        # Different file paths
        assert store1.parquet_path != store2.parquet_path
