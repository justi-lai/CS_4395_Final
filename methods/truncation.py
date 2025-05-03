from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import pandas as pd
from methods.utils import OpenAIClient

def truncation(data_path):
    """
    Truncation method for text data.
    Args:
        data_path (str): Path to the input data file.
    Returns:
        dataset: Dataset containing the truncated text.
    """
    dataset = load_from_disk(data_path)
    client = OpenAIClient()

    processed_data = []
    for example in dataset:
        truncated_document = truncate_text(example['document'], client)
        
        questions = example['question']
        answers = example['answers']
        question_responses = []
        
        # Ensure questions is a list
        if not isinstance(questions, list):
            questions = [questions]
            answers = [answers]
            
        for question in questions:
            response = client.generate_response(question, truncated_document)
            question_responses.append(response)
        
        processed_example = {
            'document': example['document'],
            'truncated_document': truncated_document,
            'question': questions,
            'answers': answers,
            'response': question_responses
        }
        processed_data.append(processed_example)

    # Convert to pandas DataFrame
    output_df = pd.DataFrame(processed_data)
    
    # Save to CSV
    output_path = data_path.replace("/", "_").split(".")[0] + "_truncated.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Truncated data saved to {output_path}")

    return output_df

def generate_response(client, example):
    """
    Generates a response for a given example using the OpenAI client.
    Args:
        client (OpenAIClient): The OpenAI client instance for generating responses.
        example (dict): A dictionary containing 'question' and 'truncated_document' keys.
    Returns:
        dict: The example dictionary with an added 'response' key containing the generated response.
    """
    question = example['question']
    context = example['truncated_document']
    
    response = client.generate_response(question, context)
    example['response'] = response
    return example

def create_truncated_dataset(dataset, client):
    """
    Creates a new dataset with truncated documents based on the specified chunk size and overlap.
    
    Args:
        dataset (Dataset): The input dataset to be truncated.
        client (OpenAIClient): The OpenAI client instance for tokenization.
    
    Returns:
        Dataset: A new dataset with truncated documents.
    """
    # Apply truncation to each example in the dataset
    truncated_data = []
    for example in dataset:
        truncated_doc = truncate_text(example['document'], client)
        example['truncated_document'] = truncated_doc
        truncated_data.append(example)
    
    # Convert to DataFrame and then to HuggingFace Dataset
    df = pd.DataFrame(truncated_data)
    truncated_dataset = Dataset.from_pandas(df)
    
    return truncated_dataset

def truncate_text(text, client, max_tokens=500):
    """
    Truncates and tokenizes the input text to a specified maximum number of tokens.
    Then decodes the tokens back to a string.
    
    Args:
        text (str or list of str): The input text to be tokenized.
        client (OpenAIClient): The OpenAI client instance for tokenization.
        max_tokens (int): Maximum number of tokens to keep from the end of the document.
    """
    # Handle different text input types
    if isinstance(text, list):
        text = " ".join(text)
    elif not isinstance(text, str):
        raise ValueError("Input must be a string or a list of strings.")
    
    # Encode the text to tokens
    tokens = client.encode(text, add_special_tokens=False)
    
    # Calculate available tokens
    prompt_token_count = 100  # Estimate for prompt tokens
    model_max_tokens = client.get_max_tokens()
    available_tokens = model_max_tokens - prompt_token_count - 100  # Buffer
    
    # Limit to either max_tokens or available_tokens, whichever is smaller
    token_limit = min(max_tokens, available_tokens)
    
    # Take the last token_limit tokens
    if len(tokens) > token_limit:
        tokens = tokens[-token_limit:]
    
    # Decode back to text
    truncated_text = client.decode(tokens, skip_special_tokens=True)
    return truncated_text