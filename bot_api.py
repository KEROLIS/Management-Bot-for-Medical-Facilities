from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from conversation_handler import ConversationHandler
from typing import Dict, Any
import uuid

app = FastAPI(
    title="Management Bot API",
    description="API for handling medical facility management conversations",
    version="1.0.0"
)

class ConversationRequest(BaseModel):
    conversation_id: str
    user_input: str


conversation_handler = ConversationHandler()

@app.post("/conversation")
async def handle_conversation(request: ConversationRequest):
    """
    Handles a conversation request by processing user input and returning the bot's response.
    Args:
        request (ConversationRequest): The conversation request containing the conversation ID and user input.
    Returns:
        The bot's response to the user input.
    Raises:
        HTTPException: If an error occurs while processing the conversation, an HTTP 500 error is raised with the error details.
    """
    try:
        bot_response = conversation_handler.handle_conversation(
            request.conversation_id,
            request.user_input
        )
        
        return bot_response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing conversation: {str(e)}"
        )
    
@app.get("/generate_conversation_id")
async def generate_conversation_id():
    """
    Asynchronously generates a unique conversation ID.

    This function continuously generates a UUID until it finds one that does not
    already exist in the conversation handler's database. Once a unique UUID is
    found, it creates a new conversation with that ID and returns it.

    Returns:
        dict: A dictionary containing the generated unique conversation ID.
    """
    while True:
        conversation_id = str(uuid.uuid4())
        if not conversation_handler.crud.get_conversation(conversation_id):
            conversation_handler.crud.create_conversation(conversation_id)
            return {"conversation_id": conversation_id}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)