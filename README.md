# Long Document Retrieval Methods

A comparative study of different approaches for retrieving information from long documents to answer questions effectively. This project evaluates several methods for handling long contexts in question-answering tasks using the NarrativeQA dataset.

## Overview

This project compares different strategies for processing and retrieving information from long documents when answering questions:

1. **Truncation**: Simple approach that keeps only the last N tokens from a document
2. **Chunking**: Splits documents into overlapping chunks and processes each separately
3. **Summarization**: Creates summaries of document chunks and ranks them by relevance to the query
4. **RAG (Retrieval-Augmented Generation)**: Uses embeddings to retrieve the most relevant passages
5. **GraphRAG**: Enhances standard RAG with a graph-based approach for more complex reasoning

## Dataset

This project uses the [NarrativeQA dataset](https://huggingface.co/datasets/deepmind/narrativeqa) from DeepMind, which contains:

- Long narrative documents (books and movie scripts)
- Questions about the content
- Human-written answers

## Requirements

- Python 3.10+
- HuggingFace API key (set in `.env` file as `HUGGINGFACE_TOKEN`)
- Required libraries (see installation steps)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [repository-url]
cd CS_4395_Final
pip install -r requirements.txt
```

Create a `.env` file in the root directory with your Hugging Face token:

```
HUGGINGFACE_TOKEN=your_token_here
```

## Usage

Run the main script with the desired method:

```bash
python main.py --method [method_name] --chunk [chunk_size] --data [optional_data_file]
```

Arguments:

- `--method`: One of `truncation`, `chunking`, `summarization`, `rag`, or `graphrag`
- `--chunk`: Maximum number of tokens for chunking methods (default: 500)
- `--data`: Optional CSV data file in the `data` directory (if not specified, a sample from NarrativeQA will be loaded)

Example:

```bash
python main.py --method summarization --chunk 750
```

## Project Structure

```
├── main.py              # Main script for running experiments
├── data/                # Dataset files
│   ├── narrative_qa.csv               # Original dataset
│   ├── narrative_qa_truncated.csv     # Results from truncation method
│   └── narrative_qa_summarized.csv    # Results from summarization method
└── methods/             # Implementation of retrieval methods
    ├── truncation.py    # Simple truncation approach
    ├── chunking.py      # Document chunking
    ├── summarization.py # Text summarization
    ├── rag.py           # Retrieval Augmented Generation
    ├── graphrag.py      # Graph-based RAG
    └── utils.py         # Utility functions
```

## Evaluation

The system evaluates each method using:

- Precision: How many of the generated words are in the ground truth answers
- Recall: How many of the ground truth answer words are in the generated response
- F1 Score: Harmonic mean of precision and recall

## Results

NOT FINISHED

## Future Work

- Implement hybrid approaches combining multiple methods
- Test with different language models
- Explore domain-specific fine-tuning

## Contributors

- Justin Lai
- 

## Acknowledgments

- DeepMind for the NarrativeQA dataset
- The University of Texas at Dallas, CS 4395 Human Language Technologies
