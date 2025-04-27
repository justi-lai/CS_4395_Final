import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from methods.utils import OpenAIClient

def summarization(data_path, chunk_size=500, overlap=100):
    """
    Summarization method for text data.
    
    Args:
        data_path (str): Path to the input data file.
        max_tokens (int): Maximum number of tokens for the summarization.
    
    Returns:
        pd.DataFrame: DataFrame containing the summarized text.
    """

    df = pd.read_csv(data_path)
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    df = build_summary_index(df, summarizer, chunk_size, overlap)

    client = OpenAIClient()
    client.set_system_prompt("You are a helpful assistant. You will be given a question and a context. Your task is to answer the question based on the context provided.")

    for index, row in df.iterrows():
        question = row['question']
        top_summaries = row['summaries'][:2] if len(row['summaries']) >= 2 else row['summaries']
        context = " ".join(top_summaries)
        
        response = client.generate_response(question, context)
        df.at[index, 'response'] = response
    
    output_path = data_path.replace('.csv', '_summarized.csv')
    df.to_csv(output_path, index=False)
    print(f"Summarized data saved to {output_path}")

    return df

def split_document(text, chunk_size, overlap):
    """
    Splits the document into chunks of specified size.
    
    Args:
        text (str): The input text to be split.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping tokens between chunks.
    Returns:
        list: A list of text chunks.
    """
    tokens = text.split()
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def summarize_chunks(chunks, summarizer):
    """
    Summarizes a list of text chunks.
    
    Args:
        chunks (list): A list of text chunks to be summarized.
        summarizer: The summarization pipeline.
    
    Returns:
        list: A list of summarized texts.
    """
    summaries = []
    for chunk in chunks:
        summary = summarize_chunk(chunk, summarizer)
        summaries.append(summary)
    
    return summaries

def summarize_chunk(chunk, summarizer):
    """
    Summarizes a single chunk of text.
    
    Args:
        chunk (str): The text chunk to be summarized.
        summarizer: The summarization pipeline.
    
    Returns:
        str: The summarized text.
    """
    summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def build_summary_index(df, summarizer, chunk_size, overlap):
    """
    Builds a summary index for the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing text data.
        summarizer: The summarization pipeline.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping tokens between chunks.
    Returns:
        pd.DataFrame: DataFrame containing the summarized text and original text.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    df['chunks'] = df['document'].apply(lambda x: split_document(x, chunk_size, overlap))
    df['summaries'] = df['chunks'].apply(lambda x: summarize_chunks(x, summarizer))
    df['summaries'] = df.apply(lambda row: sort_summaries(row['question'], row['summaries'], embedding_model), axis=1)
    
    return df

def sort_summaries(question, summaries, embedding_model):

    """
    Sorts the summaries based on their length.
    Args:
        question (str): The question to be answered.
        summaries (list): A list of summaries to be sorted.
        embedding_model: The embedding model for similarity calculation.
    """
    
    summary_embeddings = embedding_model.encode(summaries, convert_to_tensor=True)
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(question_embedding, summary_embeddings)[0]
    sorted_indices = similarities.argsort(descending=True).tolist()
    sorted_summaries = [summaries[i] for i in sorted_indices]

    return sorted_summaries