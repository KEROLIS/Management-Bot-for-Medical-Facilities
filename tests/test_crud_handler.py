import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.crud_handler import MessageCrudHandler
import uuid


@pytest.fixture
def crud_handler():
    handler = MessageCrudHandler("mongodb://localhost:27017", "test_db")
    yield handler
    handler.close_connection()

def test_create_conversation(crud_handler):
    conversation_id = str(uuid.uuid4())
    result = crud_handler.create_conversation(conversation_id)
    assert isinstance(result, str)
    conversation = crud_handler.get_conversation(conversation_id)
    assert conversation["conversation_id"] == conversation_id
    assert len(conversation["messages"]) == 0
    assert isinstance(conversation["created_at"], datetime)
    assert isinstance(conversation["updated_at"], datetime)

def test_add_message(crud_handler):
    conversation_id = str(uuid.uuid4())
    crud_handler.create_conversation(conversation_id)
    
    result = crud_handler.add_message(conversation_id, "Hello", "Hi there!")
    assert result is True
    
    messages = crud_handler.get_messages(conversation_id)
    assert len(messages) == 1
    assert messages[0]["patient"] == "Hello"
    assert messages[0]["bot"] == "Hi there!"

def test_get_nonexistent_conversation(crud_handler):
    with pytest.raises(ValueError):
        crud_handler.get_conversation("nonexistent_id")


def test_delete_conversation(crud_handler):
    conversation_id = str(uuid.uuid4())
    crud_handler.create_conversation(conversation_id)
    
    result = crud_handler.delete_conversation(conversation_id)
    assert result is True
    
    with pytest.raises(ValueError):
        crud_handler.get_conversation(conversation_id)

def test_delete_nonexistent_conversation(crud_handler):
    with pytest.raises(ValueError):
        crud_handler.delete_conversation("nonexistent_id")

@patch('crud_handler.MongoClient')
def test_close_connection(mock_client):
    handler = MessageCrudHandler("mongodb://localhost:27017", "test_db")
    handler.close_connection()
    handler.client.close.assert_called_once()