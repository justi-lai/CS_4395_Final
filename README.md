# Long Document Retrieval Methods

A comparative study of different approaches for retrieving information from long documents to answer questions effectively. This project evaluates several methods for handling long contexts in question-answering tasks using the NarrativeQA dataset.

## Overview

This project compares different strategies for processing and retrieving information from long documents when answering questions:

1. **Truncation**: Simple approach that keeps only the last N tokens from a document
2. **Summarization**: Creates summaries of document chunks and ranks them by relevance to the query
3. **RAG (Retrieval-Augmented Generation)**: Uses embeddings to retrieve the most relevant passages
4. **Custom RAG**: Enhances standard RAG with a simple entity-based matching approach
5. **GraphRAG**: Enhances standard RAG with a graph-based approach for more complex reasoning

## Dataset

This project uses the [NarrativeQA dataset](https://huggingface.co/datasets/deepmind/narrativeqa) from DeepMind, which contains:

- Long narrative documents (books and movie scripts)
- Questions about the content
- Human-written answers

## Requirements

- Python 3.10+
- HuggingFace API key (set in `.env` file as `HUGGINGFACE_TOKEN`)
- OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- Neo4j database for graph-based methods (set database credentials in `.env` file)
- Required libraries listed in requirements.txt

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [repository-url]
cd CS_4395_Final
pip install -r requirements.txt
```

Create a `.env` file in the root directory with your API tokens:

```
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_token_here
NEO4J_URI=your_neo4j_uri_here
NEO4J_USERNAME=your_username_here
NEO4J_PASSWORD=your_password_here
```

## Usage

Run the main script with the desired method:

```bash
python main.py --method [method_name] --chunk [chunk_size] --sample_size [sample_size]
```

Arguments:

- `--method`: One of `truncation`, `summarization`, `rag`, `custom_rag`, or `graphrag`
- `--chunk`: Maximum number of tokens for chunking methods (default: 300)
- `--sample_size`: Number of samples to use from the dataset (default: 10)

Example:

```bash
python main.py --method summarization --chunk 750 --sample_size 20
```

## Project Structure

```
├── main.py                 # Main script for running experiments
├── fine_tune.py            # Script for fine-tuning models
├── data/                   # Dataset files
│   ├── narrativeqa_arrow/  # Arrow format of the dataset
│   └── *.csv               # Processed dataset files
└── methods/                # Implementation of retrieval methods
    ├── truncation.py       # Simple truncation approach
    ├── summarization.py    # Text summarization
    ├── rag.py              # Retrieval Augmented Generation
    ├── custom_rag.py       # Entity-based RAG
    ├── graphrag.py         # Graph-based RAG
    └── utils.py            # Utility functions
```

## Evaluation

The system evaluates each method using:

- Precision: How many of the generated words are in the ground truth answers
- Recall: How many of the ground truth answer words are in the generated response
- F1 Score: Harmonic mean of precision and recall
- BLEU Score: Measures the n-gram overlap between generation and reference
- ROUGE Scores: Measures recall-oriented n-gram overlap
- BERTScore: Contextual semantic similarity using BERT embeddings

## Results

Results section to be finalized after completing all experiments.

## Future Work

- Implement hybrid approaches combining multiple methods
- Test with different language models
- Explore domain-specific fine-tuning
- Optimize graph-based approaches for better performance

## Contributors

- Justin Lai

## Acknowledgments

- DeepMind for the NarrativeQA dataset
- The University of Texas at Dallas, CS 4395 Human Language Technologies
