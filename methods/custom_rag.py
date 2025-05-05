import os
from dotenv import load_dotenv
import json
import nltk
import pandas as pd
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
from methods.utils import OpenAIClient
import spacy
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def custom_rag(data_path, max_chunk_length=300, k=3):
    """
    Implement a GraphRAG solution augmented with spaCy entity recognition
    with independent knowledge graphs for each document
    
    Args:
        data_path (str): Path to the dataset
        max_chunk_length (int): Maximum length of text chunks
        k (int): Number of similar chunks to retrieve
        
    Returns:
        DataFrame: DataFrame with documents, questions, answers and generated responses
    """
    load_dotenv()
    global NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, DRIVER
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # Load and prepare dataset
    ds = load_from_disk(data_path)
    df = ds.to_pandas()
    
    # Load models - ensure we use the same model for all embeddings
    nlp = spacy.load("en_core_web_sm")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    llm = OpenAIClient()
    
    # Process questions and generate answers in batches
    print("Processing documents and questions...")
    batch_size = 4  # Adjust based on your GPU memory
    questions_contexts = []
    metadata = []
    
    # Process each document independently
    for idx, row in tqdm(df.iterrows(), desc="Processing documents", total=len(df)):
        document = row['document']
        question = row['question']
        answers = row['answers']
        
        # Clear existing graph database for this document
        with DRIVER.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        # Chunk the document
        chunks = chunk_text(document, max_chunk_length)
        doc_chunk_ids = []
        
        # Process each chunk and build knowledge graph for this document only
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{idx}-{chunk_idx}"
            doc_chunk_ids.append(chunk_id)
            
            # Create embedding for the chunk - using encoder consistently
            embedding = encoder.encode(chunk)
            
            # Create chunk node in the graph
            with DRIVER.session() as session:
                session.execute_write(create_chunk_node, chunk_id, chunk, embedding)
            
            # Extract entities and create entity nodes and relationships
            entities = extract_entities(chunk, nlp)
            with DRIVER.session() as session:
                for entity, entity_type in entities:
                    # Create entity node if it doesn't exist
                    session.run("""
                        MERGE (e:Entity {text: $text, type: $type})
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:CONTAINS]->(e)
                    """, text=entity, type=entity_type, chunk_id=chunk_id)
                    
                    # Connect entities that appear in the same chunk
                    session.run("""
                        MATCH (c:Chunk {id: $chunk_id})-[:CONTAINS]->(e1:Entity)
                        MATCH (c)-[:CONTAINS]->(e2:Entity)
                        WHERE e1 <> e2
                        MERGE (e1)-[:CO_OCCURS]->(e2)
                    """, chunk_id=chunk_id)
        
        # Get chunk embeddings for similarity calculation
        all_chunk_embeddings = []
        all_chunks = []
        
        with DRIVER.session() as session:
            for chunk_id in doc_chunk_ids:
                chunk_data = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    RETURN c.text, c.embedding
                """, chunk_id=chunk_id)
                
                for record in chunk_data:
                    all_chunks.append(record['c.text'])
                    all_chunk_embeddings.append(np.array(record['c.embedding']))
        
        # Skip if no chunks were found
        if not all_chunk_embeddings:
            print(f"Warning: No chunks found for document {idx}, skipping.")
            # Add a placeholder result with error message
            metadata.append({
                'document': document,
                'question': question,
                'answers': answers,
                'context': None
            })
            continue
            
        # Get query embedding using the same encoder model
        # Ensure question is a single string, not a list or array
        if isinstance(question, list):
            question_text = question[0] if question else ""
        elif hasattr(question, 'tolist') and callable(getattr(question, 'tolist')):  # numpy array
            question_list = question.tolist()
            question_text = question_list[0] if question_list else ""
        else:
            question_text = str(question)
            
        print(f"Question type: {type(question)}, Question value: {question_text[:50]}")
        query_embedding = encoder.encode(question_text)
        
        # Make sure embeddings are correctly shaped for cosine similarity
        all_embeddings_array = np.array(all_chunk_embeddings)
        
        # Debug dimensions
        print(f"Query embedding shape: {query_embedding.shape}, Document embeddings shape: {all_embeddings_array.shape}")
        
        # Fix dimensionality issues - ensure 2D arrays for cosine_similarity
        
        # Reshape document embeddings if needed
        if len(all_embeddings_array.shape) > 2:
            # If 3D or higher, flatten to 2D (num_chunks, embedding_dim)
            num_chunks = all_embeddings_array.shape[0]
            all_embeddings_array = all_embeddings_array.reshape(num_chunks, -1)
            print(f"Reshaped document embeddings to: {all_embeddings_array.shape}")
        
        # Handle query embedding - ensure it has shape (1, embedding_dim)
        # If it's a 1D array, reshape to 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            print(f"Reshaped 1D query embedding to: {query_embedding.shape}")
        # If it has > 2 dimensions, flatten to 2D
        elif len(query_embedding.shape) > 2:
            query_embedding = query_embedding.reshape(1, -1)
            print(f"Flattened high-dimensional query embedding to: {query_embedding.shape}")
        # If it's 2D but the first dimension isn't 1, take the mean
        elif query_embedding.shape[0] > 1:
            query_embedding = np.mean(query_embedding, axis=0, keepdims=True)
            print(f"Averaged multiple query embeddings to: {query_embedding.shape}")
        
        # Calculate similarities and get top matches - now with correct dimensions
        try:
            # Use proper dimensions for similarity calculation
            similarities = cosine_similarity(query_embedding, all_embeddings_array)[0]
            most_similar_indices = np.argsort(similarities)[::-1][:k]
            similar_chunk_ids = [doc_chunk_ids[i] for i in most_similar_indices if i < len(doc_chunk_ids)]
        except Exception as e:
            print(f"Critical error during similarity calculation: {e}")
            print("Skipping this document due to similarity calculation failure")
            metadata.append({
                'document': document,
                'question': question,
                'answers': answers,
                'context': None
            })
            continue
        
        # Extract entities from query
        query_entities = extract_entities(question, nlp)
        query_entity_texts = [entity[0] for entity in query_entities]
        
        # Find additional relevant chunks using graph traversal
        additional_chunks = []
        with DRIVER.session() as session:
            for entity_text in query_entity_texts:
                result = session.run("""
                    MATCH (e:Entity {text: $text})<-[:CONTAINS]-(c:Chunk)
                    RETURN DISTINCT c.id, c.text
                """, text=entity_text)
                
                for record in result:
                    if record["c.id"] not in similar_chunk_ids and record["c.id"] not in additional_chunks:
                        additional_chunks.append(record["c.id"])
        
        # Retrieve text for all chunks
        all_relevant_chunk_ids = similar_chunk_ids + additional_chunks[:k]  # Limit to k additional chunks
        context_chunks = []
        
        with DRIVER.session() as session:
            for chunk_id in all_relevant_chunk_ids:
                try:
                    text_chunk = session.execute_read(get_chunk_by_id, chunk_id)
                    context_chunks.append(text_chunk)
                except Exception as e:
                    print(f"Error retrieving chunk {chunk_id}: {e}")
        
        # Build context from retrieved chunks
        context = "\n\n".join(context_chunks)
        
        # Save for batch processing
        questions_contexts.append((question, context))
        metadata.append({
            'document': document,
            'question': question,
            'answers': answers,
            'context': context
        })
    
    # Second phase: Process in batches
    print(f"Generating responses in batches of {batch_size}...")
    results = []
    
    # Process questions in batches
    for i in range(0, len(questions_contexts), batch_size):
        batch = questions_contexts[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]
        
        if not batch:
            continue
            
        try:
            # Generate responses for the entire batch at once
            batch_responses = llm.batch_generate_responses(batch, batch_size=batch_size)
            
            # Match responses with their metadata
            for j, response in enumerate(batch_responses):
                result_dict = batch_metadata[j].copy()
                if result_dict.get('context') is None:
                    result_dict['response'] = "No document chunks were found to answer this question."
                else:
                    result_dict['response'] = response
                del result_dict['context']  # Remove context to save memory
                results.append(result_dict)
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            print(f"Falling back to individual processing...")
            
            # Fall back to individual processing if batch fails
            for j, (question, context) in enumerate(batch):
                result_dict = batch_metadata[j].copy()
                
                if result_dict.get('context') is None:
                    result_dict['response'] = "No document chunks were found to answer this question."
                else:
                    try:
                        response = llm.generate_response(question, context)
                        result_dict['response'] = response
                    except Exception as e:
                        print(f"Error generating response: {e}")
                        result_dict['response'] = f"I couldn't find a good answer to this question."
                
                del result_dict['context']  # Remove context to save memory
                results.append(result_dict)
    
    # Return results as a DataFrame
    return pd.DataFrame(results)

def chunk_text(text, max_chunk_length=300, overlap=50):
    """
    Splits text into chunks with specified overlap between consecutive chunks.
    
    Args:
        text (str or numpy.ndarray): The input text to be chunked
        max_chunk_length (int): Maximum length of each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        list: List of text chunks with specified overlap
    """
    # Ensure text is a string, not a numpy array or other type
    if not isinstance(text, str):
        if hasattr(text, 'tolist'):  # Handle numpy arrays
            text = str(text.tolist())
        else:
            text = str(text)
    
    # Fallback sentence splitting that doesn't depend on punkt_tab
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        # Simple fallback if NLTK tokenizer fails
        sentences = []
        for sent in text.replace('!', '.').replace('?', '.').split('.'):
            if sent.strip():
                sentences.append(sent.strip() + '.')
    
    # Rest of the function remains the same
    chunks = []
    current_chunk = ""
    current_length = 0
    last_chunk_sentences = []  # To track sentences for overlap

    for sentence in sentences:
        # Add sentence to current chunk if it fits
        if current_length + len(sentence) <= max_chunk_length:
            current_chunk += " " + sentence
            current_length += len(sentence) + 1  # +1 for the space
            last_chunk_sentences.append(sentence)
        else:
            # Save the current chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start a new chunk with overlap from previous chunk
            if overlap > 0 and last_chunk_sentences:
                # Calculate how many sentences to include for overlap
                overlap_text = ""
                overlap_length = 0
                
                # Start from the end of the previous sentences
                for sent in reversed(last_chunk_sentences):
                    if overlap_length + len(sent) <= overlap:
                        overlap_text = sent + " " + overlap_text
                        overlap_length += len(sent) + 1
                    else:
                        break
                
                # Start new chunk with overlap + current sentence
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
                
                # Reset tracking for the new chunk
                last_chunk_sentences = [s for s in overlap_text.split(".") if s.strip()]
                last_chunk_sentences.append(sentence)
            else:
                # No overlap, just start with the current sentence
                current_chunk = sentence
                current_length = len(sentence)
                last_chunk_sentences = [sentence]
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Handle empty results by falling back to character-based chunking
    if not chunks:
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length-overlap)]
    
    return chunks

def extract_entities(text, model):
    """
    Extract named entities from text using spaCy.
    
    Args:
        text: The input text (string, numpy array, or other type)
        model: The spaCy model to use
        
    Returns:
        list: List of (entity_text, entity_type) tuples
    """
    # Convert numpy arrays or other types to string
    if not isinstance(text, str):
        if hasattr(text, 'item') and callable(getattr(text, 'item')):
            # Handle scalar numpy arrays
            try:
                text = text.item()
            except (ValueError, AttributeError):
                # For non-scalar arrays
                if hasattr(text, 'tolist') and callable(getattr(text, 'tolist')):
                    text = str(text.tolist())
                else:
                    text = str(text)
        else:
            text = str(text)
    
    # Process with spaCy
    doc = model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def create_chunk_node(tx, chunk_id, chunk_text, embedding):
    tx.run("CREATE (c:Chunk {id: $id, text: $text, embedding: $embedding})", 
           id=chunk_id, text=chunk_text, embedding=embedding.tolist())

def get_chunk_by_id(tx, chunk_id):
    result = tx.run("MATCH (c:Chunk {id: $id}) RETURN c.text", id=chunk_id)
    return result.single()[0]

def retrieve_similar_chunks(query, embeddings, model, k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    most_similar_indices = np.argsort(similarities[0])[::-1][:k]
    return most_similar_indices