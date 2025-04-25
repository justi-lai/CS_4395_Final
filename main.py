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

MAX_TOKENS = 4096

def main():
    parser = argparse.ArgumentParser(description="Process command line inputs.")
    parser.add_argument('--method', required=True, help="Specify the retrieval method to use.")
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
    
    run_method(args.method, data_path)

def run_method(method, data_path):
    method = method.lower()
    methods = ('truncation', 'chunking', 'summarization', 'rag', 'graphrag')

    if method not in methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {methods}.")
    
    file_name = os.path.basename(data_path)
    print(f"Running method: '{method}' on data: '{file_name}'")
    
    if method == 'truncation':
        truncation(data_path, max_tokens=MAX_TOKENS)
    elif method == 'chunking':
        chunking(data_path, max_tokens=MAX_TOKENS)
    elif method == 'summarization':
        summarization(data_path, max_tokens=MAX_TOKENS)
    elif method == 'rag':
        rag(data_path, max_tokens=MAX_TOKENS)
    elif method == 'graphrag':
        graphrag(data_path, max_tokens=MAX_TOKENS)
    else:
        raise ValueError(f"Method '{method}' is not implemented.")

def load_data(data_path):
    ds = load_dataset("deepmind/narrativeqa", split="test")
    small_sample = ds.shuffle(seed=42).select(range(10))
    small_sample.to_csv(data_path, index=False)
    print(f"NarrativeQA loaded and saved to {data_path}.")

if __name__ == "__main__":
    main()