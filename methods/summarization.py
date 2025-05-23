import pandas as pd
from transformers import pipeline
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer, util
from methods.utils import OpenAIClient, split_document
from tqdm import tqdm
import os

CHUNKS_USED = 2
BATCH_SIZE = 4  # Adjust based on your GPU memory

def summarization(data_path, chunk_size=300, overlap=50):
    """
    Summarization method for text data.
    
    Args:
        data_path (str): Path to the input data file.
        chunk_size (int): Maximum number of tokens for each chunk.
        overlap (int): Number of overlapping tokens between chunks.
    
    Returns:
        pd.DataFrame: DataFrame containing the summarized text.
    """
    tqdm.pandas()

    df = load_from_disk(data_path)
    df = df.to_pandas()
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    print("Summarizing documents...")
    df = build_summary_index(df, summarizer, chunk_size, overlap)

    client = OpenAIClient()
    
    print("Generating responses in batches...")
    
    # Prepare batches of questions and contexts
    questions_contexts = []
    for index, row in df.iterrows():
        question = row['question']
        top_summaries = row['summaries'][:CHUNKS_USED] if len(row['summaries']) >= CHUNKS_USED else row['summaries']
        context = " ".join(top_summaries)
        questions_contexts.append((question, context))
    
    # Process in batches
    all_responses = []
    for i in tqdm(range(0, len(questions_contexts), BATCH_SIZE), desc="Processing batches"):
        batch = questions_contexts[i:i+BATCH_SIZE]
        
        try:
            # Generate responses for the batch
            batch_responses = client.batch_generate_responses(batch, batch_size=BATCH_SIZE)
            all_responses.extend(batch_responses)
        except Exception as e:
            print(f"Error in batch processing: {e}")
            print("Falling back to individual processing...")
            
            # Fall back to individual processing
            for question, context in batch:
                try:
                    response = client.generate_response(question, context)
                    all_responses.append(response)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    all_responses.append("I couldn't find a good answer to this question.")
    
    # Update the DataFrame with responses
    for index, response in enumerate(all_responses):
        if index < len(df):
            df.at[index, 'response'] = response
    
    base_name = os.path.basename(data_path)
    output_path = os.path.join(os.path.dirname(data_path), f"{base_name.split('.')[0]}_summarized.csv")
    
    if os.path.isdir(data_path):
        dir_name = os.path.basename(data_path)
        output_path = os.path.join(os.path.dirname(data_path), f"{dir_name}_summarized.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Summarized data saved to {output_path}")

    return df

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
    
    print("\tSplitting documents into chunks...")
    df['chunks'] = df['document'].progress_apply(lambda x: split_document(x, chunk_size, overlap))
    print("\tGenerating summaries for the chunks...")
    df['summaries'] = df['chunks'].progress_apply(lambda x: summarize_chunks(x, summarizer))
    print("\tSorting summaries based on similarity to the question...")
    df['summaries'] = df.progress_apply(lambda row: sort_summaries(row['question'], row['summaries'], embedding_model), axis=1)
    
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