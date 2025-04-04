from google.generativeai import GenerativeModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Get Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def format_context_from_memories(memories: List[Dict[str, Any]]) -> str:
    """
    Format memories into a context string for Gemini.
    
    Args:
        memories: List of memory objects with text, name, and section_title
        
    Returns:
        Formatted context string
    """
    if not memories:
        return ""
    
    context = "CONTEXT INFORMATION:\n\n"
    
    for i, memory in enumerate(memories):
        # Add memory metadata
        source = memory.get("name", "Unknown Source")
        section = memory.get("section_title", "")
        if section:
            source = f"{source} - {section}"
            
        context += f"[Document {i+1}: {source}]\n"
        context += f"{memory.get('text', '')}\n\n"
    
    return context

def query_gemini(user_input: str, context: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Query the Gemini API with user input and context.
    
    Args:
        user_input: The user's query text
        context: Formatted context string from memories
        model_name: The Gemini model to use
        
    Returns:
        Gemini's response text
    """
    try:
        # Create the system prompt with context
        system_prompt = """You are AMINT (Associative Memory Integration Network Tool), 
        an AI assistant with access to a temporal Hopfield network of memories.
        
        When answering questions, use the provided context from the Hopfield network.
        If the context doesn't contain relevant information, acknowledge this limitation.
        Always cite the specific document number when using information from the context.
        
        The following context has been retrieved from the Hopfield network based on the user's query:
        """
        
        system_prompt += context
        
        # Initialize the Gemini model
        model = GenerativeModel(model_name=model_name)
        
        # Instead of using system_instruction, we'll combine the system prompt with the user input
        complete_prompt = f"{system_prompt}\n\nUser question: {user_input}\n\nResponse:"
        
        # Send the complete prompt directly to the model
        response = model.generate_content(complete_prompt)
        
        # Extract text from the response
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts'):
            return ''.join(part.text for part in response.parts)
        else:
            return str(response)
    
    except Exception as e:
        print(f"Error querying Gemini: {str(e)}")
        return f"Error generating response: {str(e)}"