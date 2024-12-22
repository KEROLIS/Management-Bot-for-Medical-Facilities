# Management Bot API

This project provides an API for handling medical facility management conversations. It uses a language model to process natural language commands and return structured JSON responses.

## Setup

### Prerequisites

- Python 3.8+
- MongoDB

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/management-bot-api.git
    cd management-bot-api
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure MongoDB is running and accessible at `mongodb://localhost:27017`.

## Usage

### Running the API

To start the API server, run:
```sh
uvicorn bot_api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### Generate Conversation ID

- **Endpoint:** `GET /generate_conversation_id`
- **Description:** Generates a unique conversation ID and initializes a new conversation.
- **Response:**
    ```json
    {
        "conversation_id": "unique-conversation-id"
    }
    ```

#### Handle Conversation

- **Endpoint:** `POST /conversation`
- **Description:** Processes user input and returns the bot's response.
- **Request Body:**
    ```json
    {
        "conversation_id": "unique-conversation-id",
        "user_input": "Add a new patient John Doe, male, 45 years old, with diabetes."
    }
    ```
- **Response:**
    ```json
    {
        "intent": "add_patient",
        "entities": {
            "name": "John Doe",
            "gender": "male",
            "age": 45,
            "condition": "diabetes"
        },
        "message": "Successfully added new patient John Doe to the system. Patient profile created with provided details."
    }
    ```

## Examples

### Adding a New Patient

1. Generate a conversation ID:
    ```sh
    curl -X GET "http://localhost:8000/generate_conversation_id"
    ```

2. Use the generated conversation ID to add a new patient:
    ```sh
    curl -X POST "http://localhost:8000/conversation" -H "Content-Type: application/json" -d '{
        "conversation_id": "unique-conversation-id",
        "user_input": "Add a new patient John Doe, male, 45 years old, with diabetes."
    }'
    ```

### Assigning Medication

1. Use the conversation ID to assign medication:
    ```sh
    curl -X POST "http://localhost:8000/conversation" -H "Content-Type: application/json" -d '{
        "conversation_id": "unique-conversation-id",
        "user_input": "Assign medication Paracetamol 500mg twice a day for John Doe."
    }'
    ```

### Scheduling a Follow-Up

1. Use the conversation ID to schedule a follow-up:
    ```sh
    curl -X POST "http://localhost:8000/conversation" -H "Content-Type: application/json" -d '{
        "conversation_id": "unique-conversation-id",
        "user_input": "Schedule a follow-up for John Doe on December 20th."
    }'
    ```

## Running Tests

To run the tests, use:
```sh
pytest
```

