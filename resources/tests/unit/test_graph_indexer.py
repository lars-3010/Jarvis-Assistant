import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from jarvis.services.graph.indexer import GraphIndexer
from jarvis.services.graph.database import GraphDatabase
from jarvis.services.graph.parser import MarkdownParser
from jarvis.services.vault.reader import VaultReader

@pytest.fixture
def mock_graph_database():
    return MagicMock(spec=GraphDatabase)

@pytest.fixture
def mock_vault_reader():
    return MagicMock(spec=VaultReader)

@pytest.fixture
def mock_vaults():
    mock_path = MagicMock(spec=Path)
    mock_path.rglob.return_value = [] # Default empty rglob
    mock_path.exists.return_value = True # Default exists
    mock_path.__truediv__.side_effect = lambda x: MagicMock(spec=Path, name=f"mock_path/{x}", stem=Path(x).stem, relative_to=lambda y: Path(x))
    return {"default": mock_path}

@pytest.fixture
def graph_indexer(mock_graph_database, mock_vaults, mock_vault_reader):
    # Patch VaultReader during initialization
    with patch('jarvis.services.graph.indexer.VaultReader', return_value=mock_vault_reader):
        indexer = GraphIndexer(mock_graph_database, mock_vaults)
        return indexer

def test_graph_indexer_init(mock_graph_database, mock_vaults):
    with patch('jarvis.services.graph.indexer.VaultReader') as MockVaultReader:
        indexer = GraphIndexer(mock_graph_database, mock_vaults)
        MockVaultReader.assert_called_once_with(str(mock_vaults["default"]))
        assert indexer.database == mock_graph_database
        assert indexer.vaults == mock_vaults
        assert "default" in indexer.vault_readers

def test_index_vault_no_files(graph_indexer, mock_vault_reader, mock_vaults):
    mock_vaults["default"].rglob.return_value = []
    graph_indexer.index_vault("default")
    mock_vault_reader.read_file.assert_not_called()
    graph_indexer.database.create_or_update_note.assert_not_called()

def test_index_vault_single_file(graph_indexer, mock_vault_reader, mock_vaults):
    mock_file_path = mock_vaults["default"] / "test_note.md"
    mock_vaults["default"].rglob.return_value = [mock_file_path]
    
    mock_vault_reader.read_file.return_value = ("---\ntitle: Test Note\n---\nContent.", {})
    
    with patch('jarvis.services.graph.indexer.MarkdownParser') as MockMarkdownParser:
        mock_parser_instance = MockMarkdownParser.return_value
        mock_parser_instance.parse.return_value = {
            "frontmatter": {"title": "Test Note"},
            "links": [],
            "tags": [],
            "headings": [],
            "relationships": {}
        }
        
        graph_indexer.index_vault("default")
        
        mock_vault_reader.read_file.assert_called_once_with("test_note.md")
        MockMarkdownParser.assert_called_once_with("---\ntitle: Test Note\n---\nContent.")
        mock_parser_instance.parse.assert_called_once()
        graph_indexer.database.create_or_update_note.assert_called_once_with({
            "path": "test_note.md",
            "title": "test_note",
            "tags": [],
            "links": [],
            "relationships": {}
        })

def test_index_vault_multiple_files(graph_indexer, mock_vault_reader, mock_vaults):
    mock_file_path1 = mock_vaults["default"] / "note1.md"
    mock_file_path2 = mock_vaults["default"] / "note2.md"
    mock_vaults["default"].rglob.return_value = [mock_file_path1, mock_file_path2]
    
    mock_vault_reader.read_file.side_effect = [
        ("Content 1.", {}),
        ("Content 2.", {})
    ]
    
    with patch('jarvis.services.graph.indexer.MarkdownParser') as MockMarkdownParser:
        mock_parser_instance = MockMarkdownParser.return_value
        mock_parser_instance.parse.side_effect = [
            {"frontmatter": {}, "links": [], "tags": ["tag1"], "headings": [], "relationships": {}},
            {"frontmatter": {}, "links": [], "tags": ["tag2"], "headings": [], "relationships": {}}
        ]
        
        graph_indexer.index_vault("default")
        
        assert mock_vault_reader.read_file.call_count == 2
        assert MockMarkdownParser.call_count == 2
        assert mock_parser_instance.parse.call_count == 2
        assert graph_indexer.database.create_or_update_note.call_count == 2
        
        # Verify calls for note1.md
        graph_indexer.database.create_or_update_note.assert_any_call({
            "path": "note1.md",
            "title": "note1",
            "tags": ["tag1"],
            "links": [],
            "relationships": {}
        })
        # Verify calls for note2.md
        graph_indexer.database.create_or_update_note.assert_any_call({
            "path": "note2.md",
            "title": "note2",
            "tags": ["tag2"],
            "links": [],
            "relationships": {}
        })

def test_index_vault_unknown_vault(graph_indexer):
    with pytest.raises(ValueError, match="Unknown vault: unknown_vault"):
        graph_indexer.index_vault("unknown_vault")

def test_index_vault_nonexistent_path(graph_indexer, mock_vaults):
    mock_vaults["default"].exists.return_value = False
    with pytest.raises(ValueError, match="Vault path does not exist"):
        graph_indexer.index_vault("default")

def test_index_file_read_error(graph_indexer, mock_vault_reader, mock_vaults):
    mock_file_path = mock_vaults["default"] / "error_note.md"
    mock_vaults["default"].rglob.return_value = [mock_file_path]
    mock_vault_reader.read_file.side_effect = Exception("Read error")
    
    graph_indexer.index_vault("default")
    
    mock_vault_reader.read_file.assert_called_once()
    graph_indexer.database.create_or_update_note.assert_not_called()

def test_index_file_parse_error(graph_indexer, mock_vault_reader, mock_vaults):
    mock_file_path = mock_vaults["default"] / "parse_error_note.md"
    mock_vaults["default"].rglob.return_value = [mock_file_path]
    mock_vault_reader.read_file.return_value = ("Invalid content.", {})
    
    with patch('jarvis.services.graph.indexer.MarkdownParser') as MockMarkdownParser:
        mock_parser_instance = MockMarkdownParser.return_value
        mock_parser_instance.parse.side_effect = Exception("Parse error")
        
        graph_indexer.index_vault("default")
        
        mock_vault_reader.read_file.assert_called_once()
        MockMarkdownParser.assert_called_once()
        mock_parser_instance.parse.assert_called_once()
        graph_indexer.database.create_or_update_note.assert_not_called()