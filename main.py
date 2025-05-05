import argparse
import os
from datasets import load_dataset, Dataset
from huggingface_hub import login
from dotenv import load_dotenv
import pandas as pd
from methods.truncation import truncation
from methods.summarization import summarization
from methods.rag import rag
from methods.custom_rag import custom_rag
from methods.graphrag import graphrag
from bert_score import score
import sacrebleu
from rouge_score import rouge_scorer
import re

def main():
    chunk_size = 300

    parser = argparse.ArgumentParser(description="""
            This script runs various retrieval methods on the NarrativeQA dataset.
            The methods include truncation, summarization, RAG, and CustomRAG.
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
    parser.add_argument('--sample_size', type=int, default=10, help="Specify the sample size for the dataset (default: 10).")
    
    args = parser.parse_args()

    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    
    data_path = os.path.join(os.getcwd(), "data", "narrativeqa_arrow")
    load_data(data_path, args.sample_size)
    
    run_method(args.method, data_path, args.chunk)

def run_method(method, data_path, chunk_size, query=None):
    method = method.lower()
    methods = ('truncation', 'summarization', 'rag', 'graphrag', 'custom_rag')

    if method not in methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {methods}.")
    
    file_name = os.path.basename(data_path)
    print(f"Running method: '{method}' on data: '{file_name}'")

    df = None
    
    if method == 'truncation':
        df = truncation(data_path)
    elif method == 'summarization':
        df = summarization(data_path, chunk_size=chunk_size)
    elif method == 'rag':
        df = rag(data_path, chunk_size=chunk_size)
    elif method == 'custom_rag':
        df = custom_rag(data_path, max_chunk_length=chunk_size)
    elif method == 'graphrag':
        df = graphrag(data_path, chunk_size=chunk_size)
    else:
        raise ValueError(f"Method '{method}' is not implemented.")
    
    if df is None:
        return
    
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
    
    print("\n========== Evaluating the results... ==========")
    
    # Check if the DataFrame has the required columns
    required_cols = ['answers', 'response']
    if not all(col in df.columns for col in required_cols):
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Initialize metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # Initialize Rouge scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Use the tokenize function to handle punctuation properly
    for i, row in df.iterrows():
        # Extract and tokenize the ground truth answers
        if isinstance(row['answers'], list):
            # Flatten all answers into one set of tokens
            ground_truth_words = set()
            for ans in row['answers']:
                ground_truth_words.update(tokenize(ans))
        else:
            ground_truth_words = tokenize(str(row['answers']))
        
        # Extract and tokenize the model's response
        response = row['response']
        if isinstance(response, list):
            # If it's a list, join all responses into one string
            response_text = " ".join([str(r) for r in response if r is not None and not (isinstance(r, float) and pd.isna(r))])
            response_words = tokenize(response_text)
        else:
            # Handle single value response
            response_words = tokenize(str(response)) if response is not None else set()
        
        # Calculate precision and recall
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
        
        # Calculate BLEU score
        if isinstance(row['answers'], list):
            # Use all answers as references for BLEU calculation
            references = [ans.split() for ans in row['answers'] if isinstance(ans, str)]
            if not references:
                references = [["placeholder"]]  # Avoid empty references
        else:
            # Convert to string and split for single answer
            reference_str = str(row['answers'])
            references = [reference_str.split()]
        
        # Convert response to list of tokens for BLEU calculation
        candidate = str(response).split() if response is not None else [""]
        
        # Calculate BLEU score
        if references and candidate:
            bleu = sacrebleu.sentence_bleu(" ".join(candidate), [" ".join(ref) for ref in references]).score / 100
        else:
            bleu = 0
        bleu_scores.append(bleu)
        
        # Calculate ROUGE scores
        if isinstance(row['answers'], list) and row['answers']:
            # Use the first answer for ROUGE calculation
            reference_text = row['answers'][0] if isinstance(row['answers'][0], str) else ""
        else:
            reference_text = str(row['answers'])
        
        # Handle response for ROUGE calculation
        candidate_text = str(response) if response is not None else ""
        
        if reference_text and candidate_text:
            rouge_scores = rouge_scorer_instance.score(reference_text, candidate_text)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        else:
            rouge1_scores.append(0)
            rouge2_scores.append(0)
            rougeL_scores.append(0)
    
    # Compute average metrics
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    # Prepare for BERTScore calculation
    references = []
    candidates = []

    for _, row in df.iterrows():
        # Extract reference answer
        if isinstance(row['answers'], list):
            # If multiple answers, join them with space
            reference = " ".join([str(ans) for ans in row['answers'] if ans and str(ans).strip()])
        elif hasattr(row['answers'], 'size') and hasattr(row['answers'], 'tolist'):  # Check if it's a numpy array
            # Convert numpy array to list of strings
            answer_list = row['answers'].tolist() if hasattr(row['answers'], 'tolist') else [row['answers']]
            reference = " ".join([str(ans) for ans in answer_list if ans and str(ans).strip()])
        else:
            # For single string or other types
            reference = str(row['answers'])
        
        # Use empty reference as fallback if needed
        if not reference or not reference.strip():
            reference = "empty reference"
        
        # Extract model response, ensure it's not empty
        if isinstance(row['response'], list):
            candidate = " ".join([str(r) for r in row['response'] if r and str(r).strip()])
        else:
            candidate = str(row['response']) if row['response'] is not None else ""
        
        # Use empty response as fallback if needed
        if not candidate or not candidate.strip():
            candidate = "empty response"
        
        # Add texts to our lists
        references.append(reference)
        candidates.append(candidate)

    # Ensure we have valid data to compute scores
    valid_pairs = [(r, c) for r, c in zip(references, candidates) 
                  if r != "empty reference" and c != "empty response" 
                  and len(r.strip()) > 0 and len(c.strip()) > 0]
    
    if valid_pairs:
        valid_refs, valid_cands = zip(*valid_pairs)
        
        try:
            # Calculate BERTScore with valid inputs
            print(f"Computing BERTScore for {len(valid_pairs)} valid reference-candidate pairs...")
            P, R, F1 = score(valid_cands, valid_refs, lang="en", verbose=True, model_type="microsoft/deberta-xlarge-mnli")
            
            # Convert to Python lists
            bert_precision = P.tolist()
            bert_recall = R.tolist()
            bert_f1 = F1.tolist()
            
            # Calculate average BERTScore
            avg_bert_precision = sum(bert_precision) / len(bert_precision)
            avg_bert_recall = sum(bert_recall) / len(bert_recall)
            avg_bert_f1 = sum(bert_f1) / len(bert_f1)
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            avg_bert_precision = 0
            avg_bert_recall = 0
            avg_bert_f1 = 0
    else:
        print("Warning: Could not calculate BERTScore due to empty or invalid references/candidates")
        avg_bert_precision = 0
        avg_bert_recall = 0
        avg_bert_f1 = 0

    print()
    print(f"Evaluation Results:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print()
    
    print(f"BLEU and ROUGE Results:")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2 F1: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")
    print()

    print(f"BERTScore Results:")
    print(f"Average BERTScore Precision: {avg_bert_precision:.4f}")
    print(f"Average BERTScore Recall: {avg_bert_recall:.4f}")
    print(f"Average BERTScore F1: {avg_bert_f1:.4f}")
    print()

    print("Example Output:")
    for i in range(min(3, len(df))):
        print(f"Question: {df['question'].iloc[i]}")
        print(f"Answer: {df['answers'].iloc[i]}")
        print(f"Response: {df['response'].iloc[i]}")
        print("-" * 50)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'bert_precision': avg_bert_precision,
        'bert_recall': avg_bert_recall,
        'bert_f1': avg_bert_f1,
        'individual_scores': {
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'bleu': bleu_scores,
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores
        }
    }

def load_data(data_path, sample_size):
    ds = load_dataset("deepmind/narrativeqa", split="test")
    small_sample = ds.select(range(sample_size))
    
    small_sample = small_sample.map(transform_document, batched=False, remove_columns=small_sample.column_names)
    small_sample = small_sample.rename_column("document_text", "document")
    small_sample = small_sample.rename_column("question_text", "question")
    small_sample = small_sample.rename_column("answer_text", "answers")

    small_sample.save_to_disk(data_path)
    
    print(f"NarrativeQA saved to {data_path}.")

def transform_document(document):
    """
    Transforms a single example from the original nested format
    to the desired flat format, handling multiple answers.
    """
    doc_text = document['document']['text']
    q_text = document['question']['text']
    ans_texts = [answer['text'] for answer in document['answers']]

    num_answers = len(ans_texts)
    output = {
        'document_text': [doc_text] * num_answers,
        'question_text': [q_text] * num_answers,
        'answer_text': ans_texts
    }
    return output

if __name__ == "__main__":
    main()