from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
import numpy as np

MODEL_ID = "google/gemma-2-2b-it"
DEFAULT_SYSTEM_PROMPT = """
You are a question answering system. Answer the question concisely.
Please provide the correct answer based on the context given below.

EXAMPLE FORMAT:
Context: Miss Delmer is the elderly spinster aunt of the Earl de Verseley and Captain Delmar.

Question: Who is Miss Delmer?

Answer: the elderly spinster aunt of the Earl de Verseley and Captain Delmar

Your Task: Answer the question based on the context provided.

Context: {context}

Question: {question}

Answer:
"""
MAX_TOKENS = 500

class OpenAIClient:
    def __init__(self, system_prompt=None, model_id=MODEL_ID, tokenizer_id=None):
        load_dotenv()
        login(token=os.getenv("HUGGINGFACE_TOKEN"))

        self.model_id = model_id
        tokenizer_id = tokenizer_id or model_id
        
        # Load tokenizer and model directly instead of using pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            device_map="auto"  # Automatically choose best device configuration
        )
        
        # Set up system prompt
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        self.prompt_tokens = len(self.tokenizer.encode(self.system_prompt, add_special_tokens=False))
        
        # Get model's max context length
        self.max_model_length = self.model.config.max_position_embeddings

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.prompt_tokens = len(self.tokenizer.encode(self.system_prompt, add_special_tokens=False))
    
    def get_system_prompt(self):
        return self.system_prompt

    def generate_response(self, question, context):
        """Generate a response for a single question"""
        query = self.system_prompt.format(context=context, question=question)
        
        # Tokenize input and move to device (GPU)
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        
        # Generate with appropriate parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_model_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Handle different response formats
        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[-1].strip()
        else:
            # If the model doesn't follow the exact format, return everything after the question
            return generated_text.split(f"Question: {question}")[-1].strip()
    
    def batch_generate_responses(self, questions_contexts, batch_size=4):
        """
        Generate responses for multiple questions in batch mode
        
        Args:
            questions_contexts: List of (question, context) tuples
            batch_size: Number of items to process in each batch
            
        Returns:
            List of generated responses
        """
        all_responses = []
        
        # Process in smaller batches to avoid CUDA OOM errors
        for i in range(0, len(questions_contexts), batch_size):
            batch = questions_contexts[i:i+batch_size]
            
            # Format all queries with system prompt
            queries = [
                self.system_prompt.format(context=context, question=question)
                for question, context in batch
            ]
            
            # Tokenize all inputs
            batch_inputs = self.tokenizer(
                queries, 
                padding=True, 
                truncation=True,
                max_length=self.max_model_length,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate all outputs in a single batch
            with torch.no_grad():
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_length=self.max_model_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Process all outputs
            batch_responses = []
            for j, (output, (question, _)) in enumerate(zip(batch_outputs, batch)):
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                
                # Handle different response formats
                if "Answer:" in generated_text:
                    response = generated_text.split("Answer:")[-1].strip()
                else:
                    # If the model doesn't follow the exact format, return everything after the question
                    response = generated_text.split(f"Question: {question}")[-1].strip()
                
                batch_responses.append(response)
            
            all_responses.extend(batch_responses)
            
        return all_responses
    
    def get_max_tokens(self):
        return self.max_model_length
    
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
    if isinstance(text, np.ndarray):
        if text.size == 0:
            return []
        if isinstance(text[0], str):
            text = ' '.join(text)
        else:
            text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    
    tokens = text.split()
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks