from transformers import AutoTokenizer
import pandas as pd

def truncation(data_path, max_tokens):
    df = pd.read_csv(data_path)
    
    df['truncated_text'] = df['document'].apply(lambda x: truncate_text(x, max_tokens))
    print("Truncated text:")
    print(df['truncated_text'].keys())

def truncate_text(text, max_tokens):
    """
    Truncates and tokenizes the input text to a specified maximum number of tokens.
    Then decodes the tokens back to a string.
    
    Args:
        text (str or list of str): The input text to be tokenized.
        max_tokens (int): The maximum number of tokens to keep.
    """

    if isinstance(text, list):
        text = " ".join(text)
    elif not isinstance(text, str):
        raise ValueError("Input must be a string or a list of strings.")
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokens = tokenizer.encode(text, add_special_tokens=False)

    tokens = tokens[:max_tokens]
    tokenized_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return tokenized_text