import sys
import re
import json
import io
from contextlib import redirect_stdout
from unsloth import FastLanguageModel
from transformers import TextStreamer

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
        self.model = None
        self.tokenizer = None
        self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model = self._apply_peft_to_model(self.model)

    def _load_model_and_tokenizer(self, model_name="unsloth/Meta-Llama-3.1-8B", max_seq_length=2048, dtype=None, load_in_4bit=True):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit
        )
        return model, tokenizer

    def _apply_peft_to_model(self, model, r=16, target_modules=None, lora_alpha=16, lora_dropout=0,
                             bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
                             use_rslora=False, loftq_config=None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                              "up_proj", "down_proj"]

        model = model.get_peft_model(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config
        )
        return model

    def __parse_json_from_buffer(self,buffer):
        """
        Extracts and parses JSON response from a StringIO buffer containing model output.

        Args:
            buffer (io.StringIO): StringIO buffer containing the model response

        Returns:
            dict: Parsed JSON response, or None if parsing fails
        """
        try:
            # Read buffer contents
            content = buffer.getvalue()

            # Find everything between "### Response:" and the end of the JSON object
            pattern = r'### Response:\s*({[\s\S]*})'
            match = re.search(pattern, content)

            if not match:
                print("No JSON response found in buffer")
                return None

            # Extract and parse JSON
            json_str = match.group(1).strip()
            return json.loads(json_str)

        except AttributeError:
            print("Invalid buffer object")
            return None
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def run(self, messages: list = None, prompt: str = ''):

        if not messages:
            formatted_messages = []
            for message in messages:
                nurse_line = f"NURSE: {message['nurse']}"
                bot_line = f"BOT: {message['bot']}"
                formatted_messages.extend([nurse_line, bot_line])
                
            full_context = '\n'.join(formatted_messages)
        else:
            full_context = ''

            
        instruction = llm_instruction_template_1+full_context+llm_instruction_template_2
        
        formatted_instruction_prompt.format(str(instruction), str(prompt), "")

        inputs = self.tokenizer([
            formatted_instruction_prompt
        ], return_tensors="pt").to("cuda")

        text_streamer = TextStreamer(self.tokenizer)
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            self.model.generate(
                **inputs, streamer=text_streamer, max_new_tokens=128)
            
        json_response = self.__parse_json_from_buffer(buffer)
        
        return json_response
