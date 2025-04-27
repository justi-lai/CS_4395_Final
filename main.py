import argparse
import os
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from dotenv import load_dotenv
import pandas as pd
from methods.truncation import truncation
from methods.chunking import chunking
from methods.summarization import summarization
from methods.rag import rag
from methods.graphrag import graphrag
import re

def main():
    chunk_size = 500

    parser = argparse.ArgumentParser(description="""
            This script runs various retrieval methods on the NarrativeQA dataset.
            The methods include truncation, chunking, summarization, RAG, and GraphRAG.
            If no data file is specified, it will load a sample dataset.
            The script requires a Hugging Face token in the .env file for authentication.
            \n\n
            Data File Format:\n
            The data file should be a CSV with the following columns:\n
            - document: The text of the document.\n
            - question: The question to be answered.\n
            - answers: A list of possible answers.\n
            \n\n
            Usage:\n
            python main.py --method <method_name> [--data <data_file>]
            """)
    parser.add_argument('--method', required=True, help="Specify the retrieval method to use.")
    parser.add_argument('--chunk', type=int, default=chunk_size, help="Specify the maximum number of tokens (default: 500).")
    parser.add_argument('--data', required=False, help="Specify the data file (optional).")
    
    args = parser.parse_args()

    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    data_path = os.path.join(os.getcwd(), "data")
    if args.data:
        data_path = os.path.join(data_path, args.data)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file '{args.data}' does not exist in the data directory.")
    else:
        data_path = os.path.join(data_path, "narrative_qa.csv")
        if not os.path.exists(data_path):
            load_data(data_path)
    
    run_method(args.method, data_path, args.chunk)

def run_method(method, data_path, chunk_size):
    method = method.lower()
    methods = ('truncation', 'chunking', 'summarization', 'rag', 'graphrag')

    if method not in methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {methods}.")
    
    file_name = os.path.basename(data_path)
    print(f"Running method: '{method}' on data: '{file_name}'")

    df = None
    
    if method == 'truncation':
        df = truncation(data_path)
    elif method == 'chunking':
        df = chunking(data_path, chunk_size=chunk_size)
    elif method == 'summarization':
        df = summarization(data_path, chunk_size=chunk_size)
    elif method == 'rag':
        df = rag(data_path, chunk_size=chunk_size)
    elif method == 'graphrag':
        df = graphrag(data_path, chunk_size=chunk_size)
    else:
        raise ValueError(f"Method '{method}' is not implemented.")
    
    if df is None:
        raise ValueError("The method did not return a DataFrame. Please check the implementation.")
    
    evaluate(df)

def tokenize(text):
    if isinstance(text, str):
        return set(re.findall(r'\w+', text.lower()))
    return set()

def evaluate(df):
    """
    Evaluate the performance of the retrieval method.
    This function should be implemented based on the specific evaluation metrics you want to use.
    For example, you might want to calculate accuracy, precision, recall, etc.
    """
    
    print("Evaluating the results...")
    
    # Check if the DataFrame has the required columns
    required_cols = ['answer', 'response']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Initialize metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    # Use the tokenize function to handle punctuation properly
    for _, row in df.iterrows():
        # Convert answer list to a set of tokenized words
        if isinstance(row['answer'], list):
            # Flatten all answers into one set of tokens
            ground_truth_words = set()
            for ans in row['answer']:
                ground_truth_words.update(tokenize(ans))
        else:
            ground_truth_words = tokenize(row['answer'])
        
        # Convert response to a set of tokenized words
        response_words = tokenize(row['response']) if not pd.isna(row['response']) else set()
        
        # Calculate metrics
        if len(response_words) == 0:
            precision = 0
        else:
            precision = len(ground_truth_words.intersection(response_words)) / len(response_words)
        
        if len(ground_truth_words) == 0:
            recall = 0
        else:
            recall = len(ground_truth_words.intersection(response_words)) / len(ground_truth_words)
        
        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Compute average metrics
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    print(f"Evaluation Results:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'individual_scores': {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores
        }
    }

def load_data(data_path):
    ds = load_dataset("deepmind/narrativeqa", split="test")
    # small_sample = ds.shuffle(seed=42).select(range(10))
    small_sample = ds.select(range(10))
    
    processed_data = []
    
    for item in small_sample:
        doc_text = item['document']['text']
        
        question_text = item['question']['text']

        answer_text = [answer['text'] for answer in item['answers']]
        
        processed_data.append({
            'document': doc_text,
            'question': question_text,
            'answer': answer_text
        })

    df = pd.DataFrame(processed_data)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    df.to_csv(data_path, index=False)
    
    print(f"NarrativeQA loaded and saved to {data_path}.")

if __name__ == "__main__":
    main()