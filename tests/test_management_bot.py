import pytest
import io
import json
from src.management_bot import LLMRunner

@pytest.fixture
def llm_runner():
    return LLMRunner()

def test_parse_json_from_buffer_valid_json(llm_runner):
    buffer = io.StringIO('### Response: {"intent": "add_patient", "entities": {"name": "John Doe", "gender": "male", "age": 45, "condition": "diabetes"}, "message": "Successfully added new patient John Doe to the system."}')
    result = llm_runner._LLMRunner__parse_json_from_buffer(buffer)
    expected = {
        "intent": "add_patient",
        "entities": {
            "name": "John Doe",
            "gender": "male",
            "age": 45,
            "condition": "diabetes"
        },
        "message": "Successfully added new patient John Doe to the system."
    }
    assert result == expected

def test_parse_json_from_buffer_no_json(llm_runner):
    buffer = io.StringIO('### Response: No JSON here')
    result = llm_runner._LLMRunner__parse_json_from_buffer(buffer)
    assert result is None

def test_parse_json_from_buffer_invalid_json(llm_runner):
    buffer = io.StringIO('### Response: {"intent": "add_patient", "entities": {"name": "John Doe", "gender": "male", "age": 45, "condition": "diabetes", "message": "Successfully added new patient John Doe to the system."')
    result = llm_runner._LLMRunner__parse_json_from_buffer(buffer)
    assert result is None

def test_parse_json_from_buffer_empty_buffer(llm_runner):
    buffer = io.StringIO('')
    result = llm_runner._LLMRunner__parse_json_from_buffer(buffer)
    assert result is None

def test_run_with_valid_input(llm_runner):
    messages = [
        {"nurse": "Add a new patient John Doe, male, 45 years old, with diabetes.", "bot": ""}
    ]
    prompt = "Add a new patient John Doe, male, 45 years old, with diabetes."
    result = llm_runner.run(messages=messages, prompt=prompt)
    assert result is not None
    assert "intent" in result
    assert "entities" in result
    assert "message" in result

def test_run_with_empty_input(llm_runner):
    messages = []
    prompt = ""
    result = llm_runner.run(messages=messages, prompt=prompt)
    assert result is None