import pandas as pd
from datasets import load_from_disk, Dataset
from methods.utils import OpenAIClient, split_document
from sentence_transformers import SentenceTransformer, util

CHUNKS_USED = 2

def rag(data_path, chunk_size=300, overlap=100):
    """
    RAG (Retrieval-Augmented Generation) method for text data.
    
    Args:
        data_path (str): Path to the input data file.
        chunk_size (int): Maximum number of tokens for each chunk.
        overlap (int): Number of overlapping tokens between chunks.
    
    Returns:
        pd.DataFrame: DataFrame containing the responses from the RAG model.
    """
    
    df = load_from_disk(data_path)
    df = df.to_pandas()

    client = OpenAIClient()
    
    print("Building vector index...")
    df = build_vector_index(df, chunk_size, overlap)
    
    print("Generating responses...")
    for index, row in df.iterrows():
        question = row['question']
        top_chunks = row['chunks'][:CHUNKS_USED] if len(row['chunks']) >= CHUNKS_USED else row['chunks']
        context = "\n\n".join(top_chunks)

        response = client.generate_response(question, context)
        df.at[index, 'response'] = response
    
    output_path = data_path.replace("/", "_").split(".")[0] + "_truncated.csv"
    df.to_csv(output_path, index=False)
    print(f"RAG data saved to {output_path}")
    
    return df

def build_vector_index(df, chunk_size, overlap):
    """
    Builds a vector index for the text data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        chunk_size (int): Maximum number of tokens for each chunk.
        overlap (int): Number of overlapping tokens between chunks.
    Returns:
        pd.DataFrame: DataFrame containing the vector index.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['chunks'] = df['document'].apply(lambda x: split_document(x, chunk_size, overlap))
    df['query_embedding'] = df['question'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    
    for index, row in df.iterrows():
        query_embedding = row['query_embedding']
        chunks = row['chunks']
        df.at[index, 'chunks'] = sorting_by_similarity(chunks, query_embedding, model)
    
    return df

def sorting_by_similarity(chunks, query_embedding, embedding_model):
    """
    Sorts the chunks based on their similarity to the query embedding.

    Args:
        chunks (list): A list of text chunks to be sorted.
        query_embedding (torch.Tensor): The embedding of the query.
        embedding_model: The model used to encode the chunks.
    Returns:
        list: A sorted list of chunks based on similarity to the query embedding.
    """
    
    if not chunks:
        return []
    
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]

    sorted_indices = similarities.argsort(descending=True).tolist()
    sorted_chunks = [chunks[i] for i in sorted_indices]

    return sorted_chunks

def chunk_document(df, chunk_size, overlap):
    """
    Splits the document into chunks of specified size and overlaps.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        chunk_size (int): Maximum number of tokens for each chunk.
        overlap (int): Number of overlapping tokens between chunks.
    
    Returns:
        list: A list of text chunks.
    """
    chunks = []
    
    for index, row in df.iterrows():
        document = row['document']
        question = row['question']
        
        doc_chunks = split_document(document, chunk_size, overlap)
        
        for chunk in doc_chunks:
            chunks.append((chunk, question))
    
    return chunks