"""Tests for store manager functionality.

Coverage includes:
- Store name validation (alphanumeric, length, format)
- File path validation (parquet files, existence)
- Registry operations (register, delete, set_default)
- Store creation with duplicate names
- Store deletion and cleanup
"""

import pytest
import tempfile
from pathlib import Path
import polars as pl

from naragtive.tui.widgets.store_form import (
    StoreNameValidator,
    PathValidator,
)
from naragtive.store_registry import VectorStoreRegistry, StoreMetadata


class TestStoreNameValidator:
    """Tests for store name validation."""

    def test_valid_names(self):
        """Test valid store names."""
        valid_names = [
            "campaign-1",
            "my_store",
            "Store123",
            "_private",
            "a",
        ]
        for name in valid_names:
            assert StoreNameValidator.validate(name), f"'{name}' should be valid"

    def test_invalid_empty_name(self):
        """Test empty store name."""
        assert not StoreNameValidator.validate("")

    def test_invalid_name_too_long(self):
        """Test store name exceeding max length."""
        long_name = "a" * 51
        assert not StoreNameValidator.validate(long_name)

    def test_invalid_name_starts_with_number(self):
        """Test store name starting with number."""
        assert not StoreNameValidator.validate("1store")

    def test_invalid_name_special_characters(self):
        """Test store name with invalid special characters."""
        invalid_names = [
            "store@name",
            "store.name",
            "store name",
            "store/name",
            "store#name",
        ]
        for name in invalid_names:
            assert not StoreNameValidator.validate(name), f"'{name}' should be invalid"

    def test_valid_max_length(self):
        """Test store name at max length."""
        max_name = "a" * 50
        assert StoreNameValidator.validate(max_name)

    def test_underscore_and_hyphen(self):
        """Test underscores and hyphens are allowed."""
        assert StoreNameValidator.validate("my_store-1")
        assert StoreNameValidator.validate("_leading")
        assert StoreNameValidator.validate("trailing_")


class TestPathValidator:
    """Tests for file path validation."""

    def test_valid_parquet_path(self):
        """Test valid parquet file path."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            assert PathValidator.validate(f.name)

    def test_invalid_non_parquet_extension(self):
        """Test non-parquet file is invalid."""
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            assert not PathValidator.validate(f.name)

    def test_invalid_nonexistent_file(self):
        """Test nonexistent file is invalid."""
        assert not PathValidator.validate("/nonexistent/path/file.parquet")

    def test_invalid_empty_path(self):
        """Test empty path is invalid."""
        assert not PathValidator.validate("")

    def test_invalid_no_extension(self):
        """Test path without extension is invalid."""
        with tempfile.NamedTemporaryFile() as f:
            assert not PathValidator.validate(f.name)

    def test_path_with_tilde_expansion(self):
        """Test path with tilde expansion."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", dir=Path.home()) as f:
            # Create path string with tilde
            path_str = str(f.name).replace(str(Path.home()), "~")
            # Should be valid after expansion
            assert PathValidator.validate(path_str)


class TestVectorStoreRegistry:
    """Tests for VectorStoreRegistry operations."""

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary registry directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override registry directory
            original_dir = VectorStoreRegistry.REGISTRY_DIR
            VectorStoreRegistry.REGISTRY_DIR = Path(tmpdir)
            VectorStoreRegistry.REGISTRY_FILE = Path(tmpdir) / "registry.json"
            VectorStoreRegistry.DEFAULT_FILE = Path(tmpdir) / "default.txt"

            yield Path(tmpdir)

            # Restore
            VectorStoreRegistry.REGISTRY_DIR = original_dir
            VectorStoreRegistry.REGISTRY_FILE = original_dir / "registry.json"
            VectorStoreRegistry.DEFAULT_FILE = original_dir / "default.txt"

    @pytest.fixture
    def sample_parquet(self, temp_registry_dir):
        """Create sample parquet file."""
        df = pl.DataFrame(
            {
                "id": ["1", "2", "3"],
                "embedding": [[0.1] * 384, [0.2] * 384, [0.3] * 384],
                "text": ["text1", "text2", "text3"],
            }
        )
        path = temp_registry_dir / "test.parquet"
        df.write_parquet(path)
        return path

    def test_register_store(self, temp_registry_dir, sample_parquet):
        """Test registering a new store."""
        registry = VectorStoreRegistry()
        metadata = registry.register(
            name="test-store",
            path=sample_parquet,
            source_type="test",
            description="Test store",
        )
        assert metadata.name == "test-store"
        assert metadata.source_type == "test"
        assert metadata.record_count == 3
        assert metadata.description == "Test store"

    def test_register_duplicate_name(self, temp_registry_dir, sample_parquet):
        """Test registering duplicate store name raises error."""
        registry = VectorStoreRegistry()
        registry.register(
            name="test-store",
            path=sample_parquet,
            source_type="test",
        )
        with pytest.raises(ValueError, match="already exists"):
            registry.register(
                name="test-store",
                path=sample_parquet,
                source_type="test",
            )

    def test_register_nonexistent_file(self, temp_registry_dir):
        """Test registering nonexistent file raises error."""
        registry = VectorStoreRegistry()
        with pytest.raises(FileNotFoundError):
            registry.register(
                name="test-store",
                path=Path("/nonexistent/file.parquet"),
                source_type="test",
            )

    def test_list_stores(self, temp_registry_dir, sample_parquet):
        """Test listing stores."""
        registry = VectorStoreRegistry()
        registry.register(name="store1", path=sample_parquet, source_type="test")
        registry.register(name="store2", path=sample_parquet, source_type="test")

        stores = registry.list_stores()
        assert len(stores) == 2
        names = [s.name for s in stores]
        assert "store1" in names
        assert "store2" in names

    def test_set_default_store(self, temp_registry_dir, sample_parquet):
        """Test setting default store."""
        registry = VectorStoreRegistry()
        registry.register(name="store1", path=sample_parquet, source_type="test")
        registry.register(name="store2", path=sample_parquet, source_type="test")

        registry.set_default("store2")
        assert registry.get_default() == "store2"

    def test_set_default_nonexistent_store(self, temp_registry_dir, sample_parquet):
        """Test setting nonexistent store as default raises error."""
        registry = VectorStoreRegistry()
        registry.register(name="store1", path=sample_parquet, source_type="test")

        with pytest.raises(KeyError):
            registry.set_default("nonexistent")

    def test_delete_store(self, temp_registry_dir, sample_parquet):
        """Test deleting a store."""
        registry = VectorStoreRegistry()
        registry.register(name="store1", path=sample_parquet, source_type="test")
        registry.register(name="store2", path=sample_parquet, source_type="test")

        registry.delete("store1")
        stores = registry.list_stores()
        assert len(stores) == 1
        assert stores[0].name == "store2"

    def test_delete_nonexistent_store(self, temp_registry_dir):
        """Test deleting nonexistent store raises error."""
        registry = VectorStoreRegistry()
        with pytest.raises(KeyError):
            registry.delete("nonexistent")

    def test_delete_default_store(self, temp_registry_dir, sample_parquet):
        """Test deleting default store clears default."""
        registry = VectorStoreRegistry()
        registry.register(name="store1", path=sample_parquet, source_type="test")
        registry.register(name="store2", path=sample_parquet, source_type="test")
        registry.set_default("store1")

        registry.delete("store1")
        assert registry.get_default() == "store2"

    def test_get_store(self, temp_registry_dir, sample_parquet):
        """Test getting a store returns correct path."""
        registry = VectorStoreRegistry()
        registry.register(name="test-store", path=sample_parquet, source_type="test")

        store = registry.get("test-store")
        assert store is not None
        # Store should be PolarsVectorStore pointing to correct path

    def test_get_nonexistent_store(self, temp_registry_dir):
        """Test getting nonexistent store raises error."""
        registry = VectorStoreRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_default_with_keyword(self, temp_registry_dir, sample_parquet):
        """Test getting store with 'default' keyword."""
        registry = VectorStoreRegistry()
        registry.register(name="test-store", path=sample_parquet, source_type="test")
        registry.set_default("test-store")

        store = registry.get("default")
        assert store is not None

    def test_registry_persistence(self, temp_registry_dir, sample_parquet):
        """Test registry persists to disk."""
        registry1 = VectorStoreRegistry()
        registry1.register(name="store1", path=sample_parquet, source_type="test")

        # Create new registry instance - should load from disk
        registry2 = VectorStoreRegistry()
        stores = registry2.list_stores()
        assert len(stores) == 1
        assert stores[0].name == "store1"

    def test_empty_registry_get_default(self, temp_registry_dir):
        """Test get_default on empty registry returns None."""
        registry = VectorStoreRegistry()
        assert registry.get_default() is None

    def test_auto_detect_record_count(self, temp_registry_dir, sample_parquet):
        """Test auto-detecting record count from parquet."""
        registry = VectorStoreRegistry()
        metadata = registry.register(
            name="test-store",
            path=sample_parquet,
            source_type="test",
        )
        assert metadata.record_count == 3

    def test_manual_record_count(self, temp_registry_dir, sample_parquet):
        """Test providing manual record count."""
        registry = VectorStoreRegistry()
        metadata = registry.register(
            name="test-store",
            path=sample_parquet,
            source_type="test",
            record_count=100,  # Override auto-detect
        )
        assert metadata.record_count == 100
