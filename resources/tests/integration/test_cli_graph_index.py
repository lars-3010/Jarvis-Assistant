import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from jarvis.main import cli
from pathlib import Path
from jarvis.utils.config import JarvisSettings

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_graph_database_class():
    with patch('jarvis.main.GraphDatabase') as mock_db_class:
        yield mock_db_class

@pytest.fixture
def mock_vault_reader_class():
    with patch('jarvis.services.graph.indexer.VaultReader') as mock_reader_class:
        yield mock_reader_class

@pytest.fixture
def mock_graph_indexer_class():
    with patch('jarvis.main.GraphIndexer') as mock_indexer_class:
        yield mock_indexer_class

def test_graph_index_command_success(
    runner,
    mock_graph_database_class,
    mock_vault_reader_class,
    mock_graph_indexer_class
):
    # Mock instances and their methods
    mock_db_instance = mock_graph_database_class.return_value
    mock_indexer_instance = mock_graph_indexer_class.return_value
    
    mock_vault_path_str = "/mock/vault"
    
    with patch('pathlib.Path.exists', return_value=True):
        with patch('jarvis.main.get_settings') as mock_get_settings:
            mock_settings = JarvisSettings(
                graph_enabled=True,
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="test_user",
                neo4j_password="test_password",
                jarvis_vault_path=mock_vault_path_str
            )
            mock_get_settings.return_value = mock_settings

            result = runner.invoke(cli, [
                'graph-index',
                '--vault', mock_vault_path_str,
                '--user', 'test_user',
                '--password', 'test_password'
            ])

    # Assertions
    assert result.exit_code == 0, result.output
    assert "Graph indexing complete." in result.output
    
    # Verify GraphDatabase was initialized and closed
    mock_graph_database_class.assert_called_once_with(mock_settings)
    mock_db_instance.close.assert_called_once()
    
    # Verify GraphIndexer was initialized and index_vault called
    mock_graph_indexer_class.assert_called_once_with(
        mock_db_instance,
        {"default": Path(mock_vault_path_str)}
    )
    mock_indexer_instance.index_vault.assert_called_once_with("default")

def test_graph_index_command_no_vault_path(runner):
    with patch('jarvis.main.get_settings') as mock_get_settings:
        mock_get_settings.return_value = JarvisSettings(jarvis_vault_path=None)
        result = runner.invoke(cli, ['graph-index'])
        assert result.exit_code == 1
        assert "Error: No vault path specified." in result.output

def test_graph_index_command_nonexistent_vault_path(runner):
    with patch('jarvis.main.get_settings') as mock_get_settings:
        mock_get_settings.return_value = JarvisSettings(jarvis_vault_path="/nonexistent/vault")
        with patch('pathlib.Path.exists', return_value=False):
            result = runner.invoke(cli, [
                'graph-index',
                '--vault', "/nonexistent/vault"
            ])
            assert result.exit_code == 1
            assert "Error: Vault path does not exist: /nonexistent/vault" in result.output

def test_graph_index_command_indexing_error(
    runner,
    mock_graph_database_class,
    mock_vault_reader_class,
    mock_graph_indexer_class
):
    mock_db_instance = mock_graph_database_class.return_value
    mock_indexer_instance = mock_graph_indexer_class.return_value
    mock_indexer_instance.index_vault.side_effect = Exception("Indexing failed")
    
    mock_vault_path_str = "/mock/vault"
    
    with patch('pathlib.Path.exists', return_value=True):
        with patch('jarvis.main.get_settings') as mock_get_settings:
            mock_settings = JarvisSettings(
                graph_enabled=True,
                jarvis_vault_path=mock_vault_path_str
            )
            mock_get_settings.return_value = mock_settings

            result = runner.invoke(cli, [
                'graph-index',
                '--vault', mock_vault_path_str,
                '--user', 'test_user',
                '--password', 'test_password'
            ])

    assert result.exit_code == 1, result.output
    assert "Error: Indexing failed" in result.output
    mock_db_instance.close.assert_called_once()
