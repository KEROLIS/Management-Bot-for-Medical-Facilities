from src.management_bot import LLMRunner
from src.crud_handler import MessageCrudHandler


class ConversationHandler:
    def __init__(self):
        self.bot = LLMRunner()
        self.crud = MessageCrudHandler(connection_string="mongodb://localhost:27017", database_name="medical_conversations")

    def handle_conversation(self, conversation_id, user_input):
        """
        Handles a single conversation interaction by processing user input and generating a bot response.

        Args:
            conversation_id (str): Unique identifier for the conversation.
            user_input (str): The message input from the user.

        Returns:
            dict: Bot response containing the message and any additional metadata.

        Note:
            - Uses CRUD operations to maintain conversation history
            - Updates conversation with both user input and bot response
            - Relies on bot instance to generate responses based on conversation context
        """
        
        # Read the current state of the conversation
        conversation = self.crud.get_conversation(conversation_id)

        bot_response = self.bot.run(prompt=user_input, messages=conversation.get('messages', []))

        # Update the conversation with the new user input and bot response
        self.crud.add_message(conversation_id, user_input, bot_response['message'])

        return bot_response
