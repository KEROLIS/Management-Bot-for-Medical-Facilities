from openai import OpenAI
import json
import os
import re

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(
    api_key= OPENAI_KEY
)

llm_instruction_template_1 = """# System Context
You are a specialized medical assistant AI designed to help nurses manage patient information, medications, and appointments. You must process natural language commands and return structured JSON responses. Always maintain medical data privacy and accuracy in your responses.

# Previous Thread History"""

llm_instruction_template_2 = """# Task Definition
You must parse natural language commands related to nursing tasks and return structured JSON output. You handle three main types of tasks:

1. Adding new patients
2. Assigning medications
3. Scheduling follow-ups

# Response Format Requirements
- Always respond with valid JSON
- Include "intent", "entities", and "message" in every response
- Use consistent key names across responses
- Return error messages in JSON format when information is missing
- Include a human-readable confirmation message for each successful action

# Supported Intents and Required Entities
1. add_patient
   - name (string)
   - gender (string)
   - age (number)
   - condition (string)

2. assign_medication
   - patient_name (string)
   - medication (string)
   - dosage (string)
   - frequency (string)

3. schedule_followup
   - patient_name (string)
   - date (string)

# Error Handling
If any required entity is missing, respond with:
{
    "error": true,
    "missing_entities": ["entity1", "entity2"],
    "message": "Please provide the following information: [list missing items]"
}

# Examples
Input: "Add a new patient John Doe, male, 45 years old, with diabetes."
Expected Output:
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

Input: "Assign medication Paracetamol 500mg twice a day for John Doe."
Expected Output:
{
    "intent": "assign_medication",
    "entities": {
        "patient_name": "John Doe",
        "medication": "Paracetamol",
        "dosage": "500mg",
        "frequency": "twice a day"
    },
    "message": "Medication Paracetamol has been assigned to John Doe. Dosage: 500mg to be taken twice a day."
}

Input: "Schedule a follow-up for John Doe on December 20th."
Expected Output:
{
    "intent": "schedule_followup",
    "entities": {
        "patient_name": "John Doe",
        "date": "2024-12-20"
    },
    "message": "Follow-up appointment scheduled for John Doe on December 20th, 2024."
}

# Message Format Guidelines
1. add_patient messages should:
   - Confirm successful patient addition
   - Acknowledge all provided details
   - Use a professional, medical tone

2. assign_medication messages should:
   - Confirm medication assignment
   - Repeat dosage and frequency for verification
   - Include patient name for clarity

3. schedule_followup messages should:
   - Confirm appointment scheduling
   - Include full date in a clear format
   - Include patient name

# Rules
1. Never make assumptions about missing data
2. Maintain consistent entity names across all responses
3. Always validate that patient names match exactly
4. Convert all dates to ISO format (YYYY-MM-DD)
5. Preserve exact medication dosages as provided
6. Return error messages for ambiguous commands
7. Include clear, human-readable confirmation messages

# Process Flow
1. Identify the primary intent from the input
2. Extract all relevant entities
3. Validate completeness of required entities
4. Generate appropriate confirmation message
5. Format response in JSON with message
6. Include error handling if needed

Remember that you are processing nurse commands in a healthcare context. Maintain high accuracy and ask for clarification when needed."""

formatted_instruction_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""


class LLMRunner:
    def __init__(self):
        pass

    def run(self, messages: list = [], prompt: str = ''):

        if messages:
            formatted_messages = []
            for message in messages:
                nurse_line = f"NURSE: {message['nurse']}"
                bot_line = f"BOT: {message['bot']}"
                formatted_messages.extend([nurse_line, bot_line])

            full_context = '\n'.join(formatted_messages)
        else:
            full_context = ' '

        instruction = llm_instruction_template_1 + \
            full_context+llm_instruction_template_2
        formatted_prompt = formatted_instruction_prompt.format(
            str(instruction), str(prompt), "")

        # Send a request to OpenAI's GPT model
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        
        json_response = self.__parse_json_from_response(
            response.choices[0].message.content.strip())
        return json_response

    def __parse_json_from_response(self, response_text):
        """
        Extracts and parses JSON response from the model's raw output.

        Args:
            response_text (str): The response text from OpenAI's API.

        Returns:
            dict: Parsed JSON response, or None if parsing fails.
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
        

