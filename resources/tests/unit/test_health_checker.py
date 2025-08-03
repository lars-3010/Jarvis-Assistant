import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from jarvis.services.health import HealthChecker
from jarvis.utils.config import JarvisSettings
from jarvis.utils.errors import ServiceError
from neo4j.exceptions import ServiceUnavailable

@pytest.fixture
def mock_settings():
    settings = JarvisSettings()
    settings.graph_enabled = True
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "password"
    settings.vector_db_path = "data/test_vector.duckdb"
    settings.vault_path = "/tmp/test_vault"
    return settings

@pytest.fixture
def health_checker(mock_settings):
    return HealthChecker(mock_settings)

@patch('jarvis.services.health.GraphDatabase')
def test_check_neo4j_health_healthy(mock_graph_database, health_checker):
    mock_driver = MagicMock()
    mock_graph_database.driver.return_value.__enter__.return_value = mock_driver
    
    result = health_checker.check_neo4j_health()
    assert result["service"] == "Neo4j"
    assert result["status"] == "HEALTHY"
    assert "Successfully connected" in result["details"]
    mock_driver.verify_connectivity.assert_called_once()

@patch('jarvis.services.health.GraphDatabase')
def test_check_neo4j_health_unavailable(mock_graph_database, health_checker):
    mock_graph_database.driver.side_effect = ServiceUnavailable("Connection refused")
    
    result = health_checker.check_neo4j_health()
    assert result["service"] == "Neo4j"
    assert result["status"] == "UNAVAILABLE"
    assert "Connection refused" in result["details"]

@patch('jarvis.services.health.GraphDatabase')
def test_check_neo4j_health_disabled(mock_graph_database, mock_settings, health_checker):
    mock_settings.graph_enabled = False
    
    result = health_checker.check_neo4j_health()
    assert result["service"] == "Neo4j"
    assert result["status"] == "DISABLED"
    assert "disabled in configuration" in result["details"]
    mock_graph_database.driver.assert_not_called()

@patch('jarvis.services.health.VectorDatabase.__init__', return_value=None)
@patch('jarvis.services.health.VectorDatabase.num_notes', return_value=100)
@patch('jarvis.services.health.VectorDatabase.close')
def test_check_vector_db_health_healthy(mock_close, mock_num_notes, mock_init, health_checker):
    result = health_checker.check_vector_db_health()
    assert result["service"] == "VectorDB"
    assert result["status"] == "HEALTHY"
    assert "Successfully connected" in result["details"]
    mock_num_notes.assert_called_once()
    mock_close.assert_called_once()

@patch('jarvis.services.health.VectorDatabase.__init__', side_effect=ServiceError("DB file corrupted"))
def test_check_vector_db_health_unavailable(mock_init, health_checker):
    result = health_checker.check_vector_db_health()
    assert result["service"] == "VectorDB"
    assert result["status"] == "UNAVAILABLE"
    assert "DB file corrupted" in result["details"]

@patch('pathlib.Path.exists', return_value=True)
@patch('pathlib.Path.is_dir', return_value=True)
def test_check_vault_health_healthy(mock_is_dir, mock_exists, health_checker):
    result = health_checker.check_vault_health()
    assert result["service"] == "Vault"
    assert result["status"] == "HEALTHY"
    assert "Vault found" in result["details"]

@patch('pathlib.Path.exists', return_value=False)
def test_check_vault_health_unavailable(mock_exists, health_checker):
    result = health_checker.check_vault_health()
    assert result["service"] == "Vault"
    assert result["status"] == "UNAVAILABLE"
    assert "does not exist" in result["details"]

def test_check_vault_health_unconfigured(health_checker, mock_settings):
    mock_settings.vault_path = ""
    result = health_checker.check_vault_health()
    assert result["service"] == "Vault"
    assert result["status"] == "UNCONFIGURED"
    assert "not configured" in result["details"]

@patch.object(HealthChecker, 'check_neo4j_health')
@patch.object(HealthChecker, 'check_vector_db_health')
@patch.object(HealthChecker, 'check_vault_health')
def test_get_overall_health_healthy(mock_check_vault, mock_check_vector, mock_check_neo4j, health_checker):
    mock_check_neo4j.return_value = {"service": "Neo4j", "status": "HEALTHY"}
    mock_check_vector.return_value = {"service": "VectorDB", "status": "HEALTHY"}
    mock_check_vault.return_value = {"service": "Vault", "status": "HEALTHY"}

    result = health_checker.get_overall_health()
    assert result["overall_status"] == "HEALTHY"
    assert len(result["services"]) == 3

@patch.object(HealthChecker, 'check_neo4j_health')
@patch.object(HealthChecker, 'check_vector_db_health')
@patch.object(HealthChecker, 'check_vault_health')
def test_get_overall_health_degraded(mock_check_vault, mock_check_vector, mock_check_neo4j, health_checker):
    mock_check_neo4j.return_value = {"service": "Neo4j", "status": "UNAVAILABLE"}
    mock_check_vector.return_value = {"service": "VectorDB", "status": "HEALTHY"}
    mock_check_vault.return_value = {"service": "Vault", "status": "HEALTHY"}

    result = health_checker.get_overall_health()
    assert result["overall_status"] == "DEGRADED"

    mock_check_neo4j.return_value = {"service": "Neo4j", "status": "HEALTHY"}
    mock_check_vector.return_value = {"service": "VectorDB", "status": "UNAVAILABLE"}
    mock_check_vault.return_value = {"service": "Vault", "status": "HEALTHY"}

    result = health_checker.get_overall_health()
    assert result["overall_status"] == "DEGRADED"

    mock_check_neo4j.return_value = {"service": "Neo4j", "status": "HEALTHY"}
    mock_check_vector.return_value = {"service": "VectorDB", "status": "HEALTHY"}
    mock_check_vault.return_value = {"service": "Vault", "status": "UNCONFIGURED"}

    result = health_checker.get_overall_health()
    assert result["overall_status"] == "DEGRADED"
