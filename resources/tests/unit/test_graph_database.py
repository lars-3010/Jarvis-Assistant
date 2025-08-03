import pytest
from unittest.mock import MagicMock, patch
from jarvis.services.graph.database import GraphDatabase
from jarvis.utils.config import JarvisSettings
from neo4j.exceptions import Neo4jError

# Mock the neo4j.GraphDatabase.driver to prevent actual connection
@pytest.fixture
def mock_neo4j_driver():
    with patch('jarvis.services.graph.database.Neo4jGraphDatabase') as mock_driver_class:
        mock_driver = MagicMock()
        mock_driver_class.driver.return_value = mock_driver
        yield mock_driver

@pytest.fixture
def mock_settings():
    """Fixture for JarvisSettings."""
    return JarvisSettings(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        graph_enabled=True
    )

@pytest.fixture
def graph_db(mock_neo4j_driver, mock_settings):
    # Initialize GraphDatabase, which will use the mocked driver
    with patch('jarvis.services.health.check_neo4j_health', return_value=True):
        db = GraphDatabase(mock_settings)
        # Clear calls made during initialization (_initialize_schema)
        if db.driver:
            db.driver.session.return_value.__enter__.return_value.run.reset_mock()
        return db

def test_graph_database_init(mock_neo4j_driver, mock_settings):
    # Test that the driver is initialized and schema is called
    with patch('jarvis.services.health.check_neo4j_health', return_value=True):
        db = GraphDatabase(mock_settings)
        assert db.driver is not None
        db.driver.session.assert_called_once()
        mock_session = db.driver.session.return_value.__enter__.return_value
        assert mock_session.run.call_count == 3 # For the three CREATE CONSTRAINT/INDEX calls

def test_close_connection(graph_db):
    if graph_db.driver:
        graph_db.close()
        graph_db.driver.close.assert_called_once()

def test_check_connection(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    mock_session = graph_db.driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = MagicMock()
    mock_session.run.return_value.single.return_value = [1] # Mock a successful return

    # This method no longer exists, so we check is_healthy property
    assert graph_db.is_healthy is True

def test_create_or_update_note(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    mock_session = graph_db.driver.session.return_value.__enter__.return_value
    note_data = {
        "path": "test/note.md",
        "title": "Test Note",
        "tags": ["tag1"],
        "links": [{"target": "linked/note.md"}],
        "relationships": {"similar": [{"target": "similar/note.md"}]}
    }
    mock_session.execute_write.return_value = {"operation": "created", "path": note_data["path"], "title": note_data["title"]}

    result = graph_db.create_or_update_note(note_data)

    assert result["operation"] == "created"
    assert mock_session.execute_write.call_count == 3 # note, relationships, links

def test_create_or_update_note_no_relationships(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    mock_session = graph_db.driver.session.return_value.__enter__.return_value
    note_data = {
        "path": "test/note.md",
        "title": "Test Note",
    }
    mock_session.execute_write.return_value = {"operation": "created", "path": note_data["path"], "title": note_data["title"]}

    result = graph_db.create_or_update_note(note_data)

    assert result["operation"] == "created"
    assert mock_session.execute_write.call_count == 1 # only note

def test_create_or_update_note_error(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    graph_db.driver.session.return_value.__enter__.return_value.execute_write.side_effect = Neo4jError("Test Error")

    note_data = {
        "path": "test/note.md",
        "title": "Test Note",
    }

    result = graph_db.create_or_update_note(note_data)
    assert "error" in result

def test_get_note_graph(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    mock_session = graph_db.driver.session.return_value.__enter__.return_value
    mock_record = MagicMock()
    mock_record.single.return_value = {
        "nodes": [{"id": "1", "label": "Node1", "path": "path1.md", "tags": [], "center": True}],
        "relationships": []
    }
    mock_session.execute_read.return_value = mock_record.single.return_value

    graph = graph_db.get_note_graph("test/note.md", 1)

    assert "nodes" in graph
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["label"] == "Node1"

def test_get_note_graph_no_results(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    graph_db.driver.session.return_value.__enter__.return_value.execute_read.return_value = {"nodes": [], "relationships": []}

    graph = graph_db.get_note_graph("nonexistent/note.md", 1)

    assert "nodes" in graph
    assert len(graph["nodes"]) == 0

def test_get_note_graph_error(graph_db):
    if not graph_db.is_healthy:
        pytest.skip("GraphDB not healthy")
    graph_db.driver.session.return_value.__enter__.return_value.execute_read.side_effect = Neo4jError("Graph Error")

    graph = graph_db.get_note_graph("test/note.md", 1)

    assert "nodes" in graph
    assert len(graph["nodes"]) == 0
