from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

MODEL_ID = "google/gemma-2-2b-it"
USE_4_BIT = False
USE_8_BIT = True
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant. Answer the question concisely and in a friendly manner.
Please provide the correct answer based on the context given below.

***EXAMPLE***
Question: Who is Miss Delmer?

Answer: the elderly spinster aunt of the Earl de Verseley and Captain Delmar
***END OF EXAMPLE***

Your Task: Answer the question based on the context provided.

Context: {context}

Question: {question}

Answer:
"""
MAX_TOKENS = 500

class OpenAIClient:
    def __init__(self, system_prompt=None, model_id=MODEL_ID, base_url=BASE_URL, api_key=API_KEY, tokenizer="google/gemma-2-2b-it"):
        load_dotenv()
        login(token=os.getenv("HUGGINGFACE_TOKEN"))

        self.model_id = model_id
        self.pipe = pipeline("text-generation", model=model_id, device=0)
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prompt_tokens = len(self.tokenizer.encode(self.system_prompt, add_special_tokens=False))

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
    
    def get_system_prompt(self):
        return self.system_prompt

    def generate_response(self, question, context):
        query = self.system_prompt.format(context=context, question=question)
        
        response = self.pipe(query)

        return response[0]['generated_text'].split("Answer:")[-1].strip()
    
    def get_max_tokens(self):
        return MAX_TOKENS
    
    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
def split_document(text, chunk_size, overlap):
    """
    Splits the document into chunks of specified size.
    
    Args:
        text (str or numpy.ndarray): The input text to be split.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping tokens between chunks.
    Returns:
        list: A list of text chunks.
    """
    # Convert numpy array to string if needed
    import numpy as np
    if isinstance(text, np.ndarray):
        if text.size == 0:
            return []
        # Join array elements if it's an array of strings
        if isinstance(text[0], str):
            text = ' '.join(text)
        else:
            text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    
    # Now split the text into tokens
    tokens = text.split()
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks