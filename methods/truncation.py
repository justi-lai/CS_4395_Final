from transformers import AutoTokenizer
import pandas as pd
from methods.utils import OpenAIClient

def truncation(data_path):
    """
    Processes a CSV dataset by truncating document text and generating responses to questions.
    This function reads a CSV file containing questions and documents, truncates each document
    to a more manageable size, and then uses an OpenAI client to generate answers to the questions
    based on the truncated context. The results are saved to a new CSV file with "_truncated" appended
    to the original filename.
    Parameters:
        data_path (str): Path to the input CSV file containing 'question' and 'document' columns.
    Returns:
        pd.DataFrame: The processed DataFrame containing the original data plus 'truncated_document'
                      and 'response' columns.
    """
    df = pd.read_csv(data_path)
    prompt = "You are a helpful assistant. You will be given a question and a context. Your task is to answer the question based on the context provided."
    client = OpenAIClient()
    client.set_system_prompt(prompt)
    
    df['truncated_document'] = df['document'].apply(lambda x: truncate_text(x, client))

    for index, row in df.iterrows():
        question = row['question']
        context = row['truncated_document']
        
        response = client.generate_response(question, context)
        df.at[index, 'response'] = response
    output_path = data_path.replace('.csv', '_truncated.csv')
    df.to_csv(output_path, index=False)

    print(f"Truncated data saved to {output_path}")

    return df

def truncate_text(text, client):
    """
    Truncates and tokenizes the input text to a specified maximum number of tokens.
    Then decodes the tokens back to a string.
    
    Args:
        text (str or list of str): The input text to be tokenized.
        client (OpenAIClient): The OpenAI client instance for tokenization.
    """

    if isinstance(text, list):
        text = " ".join(text)
    elif not isinstance(text, str):
        raise ValueError("Input must be a string or a list of strings.")
    
    tokens = client.encode(text, add_special_tokens=False)
    prompt_token_count = client.prompt_tokens
    max_tokens = client.get_max_tokens()

    available_tokens = max_tokens - prompt_token_count - 100

    if len(tokens) > available_tokens:
        tokens = tokens[-available_tokens:]

    tokenized_text = client.decode(tokens, skip_special_tokens=True)
    return tokenized_text