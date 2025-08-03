"""
Integration tests for database migration utilities.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from jarvis.database.migration import DatabaseMigrator, MigrationProgress
from jarvis.core.interfaces import IVectorDatabase, IGraphDatabase


class MockVectorDatabase(IVectorDatabase):
    """Mock vector database for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.notes = {}
        self.vaults = {}
    
    def close(self):
        pass
    
    def num_notes(self) -> int:
        return len(self.notes)
    
    def get_most_recent_seen_timestamp(self, vault_name: str) -> float:
        vault_notes = [note for note in self.notes.values() if note['vault_name'] == vault_name]
        if not vault_notes:
            return None
        return max(note['last_modified'] for note in vault_notes)
    
    def store_note(self, path, vault_name: str, last_modified: float, embedding: list, checksum=None) -> bool:
        note_id = f"{vault_name}::{str(path)}"
        self.notes[note_id] = {
            'path': str(path),
            'vault_name': vault_name,
            'last_modified': last_modified,
            'embedding': embedding,
            'checksum': checksum
        }
        return True
    
    def search(self, query_embedding, top_k: int = 10, vault_name=None):
        # Simple mock search
        results = []
        for note in self.notes.values():
            if vault_name is None or note['vault_name'] == vault_name:
                results.append((note['vault_name'], Path(note['path']), 0.8))
        return results[:top_k]
    
    def get_note_by_path(self, vault_name: str, path) -> dict:
        note_id = f"{vault_name}::{str(path)}"
        return self.notes.get(note_id)
    
    def delete_note(self, vault_name: str, path) -> bool:
        note_id = f"{vault_name}::{str(path)}"
        if note_id in self.notes:
            del self.notes[note_id]
            return True
        return False
    
    def get_vault_stats(self, vault_name: str) -> dict:
        vault_notes = [note for note in self.notes.values() if note['vault_name'] == vault_name]
        if not vault_notes:
            return {'vault_name': vault_name, 'note_count': 0, 'latest_modified': None, 'earliest_modified': None}
        
        timestamps = [note['last_modified'] for note in vault_notes]
        return {
            'vault_name': vault_name,
            'note_count': len(vault_notes),
            'latest_modified': max(timestamps),
            'earliest_modified': min(timestamps)
        }


class MockGraphDatabase(IGraphDatabase):
    """Mock graph database for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.healthy = True
        self.notes = {}
    
    @property
    def is_healthy(self) -> bool:
        return self.healthy
    
    def close(self):
        pass
    
    def create_or_update_note(self, note_data: dict) -> dict:
        note_id = note_data['path']
        self.notes[note_id] = note_data
        return {'operation': 'created', 'path': note_id}
    
    def get_note_graph(self, path: str, depth: int = 2) -> dict:
        return {'nodes': [], 'relationships': []}


class TestMigrationProgress:
    """Test MigrationProgress tracking."""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        progress = MigrationProgress()
        
        assert progress.total_items == 0
        assert progress.migrated_items == 0
        assert progress.failed_items == 0
        assert progress.errors == []
        assert progress.start_time is not None
        assert progress.end_time is None
    
    def test_add_error(self):
        """Test error tracking."""
        progress = MigrationProgress()
        
        progress.add_error("Test error")
        
        assert progress.failed_items == 1
        assert "Test error" in progress.errors
    
    def test_mark_success(self):
        """Test success tracking."""
        progress = MigrationProgress()
        
        progress.mark_success()
        
        assert progress.migrated_items == 1
    
    def test_finish(self):
        """Test migration completion."""
        progress = MigrationProgress()
        progress.total_items = 10
        progress.migrated_items = 8
        progress.failed_items = 2
        
        progress.finish()
        
        assert progress.end_time is not None
        
        summary = progress.get_summary()
        assert summary['total_items'] == 10
        assert summary['migrated_items'] == 8
        assert summary['failed_items'] == 2
        assert summary['success_rate'] == 80.0
        assert 'duration_seconds' in summary


class TestDatabaseMigrator:
    """Test DatabaseMigrator functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.migrator = DatabaseMigrator()
        
        # Create mock databases with test data
        self.source_db = MockVectorDatabase("source")
        self.target_db = MockVectorDatabase("target")
        
        # Add test data to source
        self.source_db.store_note(
            path=Path("note1.md"),
            vault_name="test_vault",
            last_modified=1234567890.0,
            embedding=[0.1] * 384
        )
        self.source_db.store_note(
            path=Path("note2.md"),
            vault_name="test_vault",
            last_modified=1234567891.0,
            embedding=[0.2] * 384
        )
    
    def test_migrate_vector_data_dry_run(self):
        """Test vector data migration in dry run mode."""
        progress = self.migrator.migrate_vector_data(
            source=self.source_db,
            target=self.target_db,
            vault_names=["test_vault"],
            dry_run=True
        )
        
        assert progress.total_items == 2
        assert progress.migrated_items == 0  # Dry run shouldn't migrate
        assert progress.failed_items == 0
        assert progress.end_time is not None
    
    def test_migrate_vector_data_actual(self):
        """Test actual vector data migration."""
        progress = self.migrator.migrate_vector_data(
            source=self.source_db,
            target=self.target_db,
            vault_names=["test_vault"],
            dry_run=False
        )
        
        assert progress.total_items == 2
        assert progress.migrated_items == 2
        assert progress.failed_items == 0
        assert progress.end_time is not None
    
    def test_migrate_vector_data_no_vault_names(self):
        """Test migration when no vault names provided."""
        progress = self.migrator.migrate_vector_data(
            source=self.source_db,
            target=self.target_db,
            vault_names=None,
            dry_run=True
        )
        
        assert progress.total_items == 0
        assert progress.end_time is not None
    
    def test_migrate_graph_data(self):
        """Test graph data migration."""
        source_graph = MockGraphDatabase("source")
        target_graph = MockGraphDatabase("target")
        
        progress = self.migrator.migrate_graph_data(
            source=source_graph,
            target=target_graph,
            dry_run=True
        )
        
        assert progress.end_time is not None
        assert progress.failed_items == 0
    
    def test_migrate_graph_data_unhealthy_source(self):
        """Test graph migration with unhealthy source."""
        source_graph = MockGraphDatabase("source")
        source_graph.healthy = False
        target_graph = MockGraphDatabase("target")
        
        with pytest.raises(Exception):
            self.migrator.migrate_graph_data(
                source=source_graph,
                target=target_graph,
                dry_run=True
            )
    
    def test_validate_migration(self):
        """Test migration validation."""
        # Ensure both databases have same data
        for note_id, note_data in self.source_db.notes.items():
            self.target_db.notes[note_id] = note_data
        
        result = self.migrator.validate_migration(
            source=self.source_db,
            target=self.target_db,
            vault_name="test_vault"
        )
        
        assert result['validation_passed'] is True
        assert result['count_match'] is True
        assert result['source_note_count'] == 2
        assert result['target_note_count'] == 2
        assert len(result['errors']) == 0
    
    def test_validate_migration_mismatch(self):
        """Test migration validation with count mismatch."""
        # Target has less data than source
        result = self.migrator.validate_migration(
            source=self.source_db,
            target=self.target_db,
            vault_name="test_vault"
        )
        
        assert result['validation_passed'] is False
        assert result['count_match'] is False
        assert result['source_note_count'] == 2
        assert result['target_note_count'] == 0
        assert len(result['errors']) > 0
    
    def test_create_backup(self):
        """Test backup creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_path = Path(temp_dir) / "backup.json"
            
            backup_info = self.migrator.create_backup(
                database=self.source_db,
                backup_path=backup_path,
                vault_names=["test_vault"]
            )
            
            assert backup_info['backup_path'] == str(backup_path)
            assert 'created_at' in backup_info
            assert backup_info['vault_names'] == ["test_vault"]
            assert backup_path.parent.exists()


@pytest.mark.integration 
class TestDatabaseMigrationIntegration:
    """Integration tests for database migration with real components."""
    
    def test_migration_with_real_duckdb(self):
        """Test migration between DuckDB instances."""
        pytest.skip("Requires full DuckDB setup")
    
    def test_migration_with_mixed_backends(self):
        """Test migration between different backend types."""
        pytest.skip("Requires multiple backend setup")


if __name__ == "__main__":
    pytest.main([__file__])