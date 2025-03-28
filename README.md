# LLM Workflow

## Overview

This project implements an LLM (Large Language Model) workflow using Python. It processes input data, interacts with a language model, and generates structured outputs. The script is designed to handle user queries, analyze text, and return relevant results.

## Features

- **Input Processing**: Reads and processes user input for better LLM interaction.
- **LLM Integration**: Communicates with an LLM to generate responses.
- **Data Handling**: Manages data flow efficiently for text generation.
- **Error Handling**: Implements basic error-checking mechanisms.


## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install required libraries:
```sh
pip install -r requirments.txt
```

### 2. Configure API Keys
Create a `.env` file in the project directory and add:
```
API_KEY=your_groq_api_key
LLM_MODEL=your_llm_model
```
### 3. Running the Script
Execute the script and select an option:
```sh
python LLM_Workflow.py
```
LLM Workflow Options:
- `1` for Pipeline Workflow
- `2` for DAG Workflow
- `3` for Key Points Extraction with Chain-of-Thought
- `4` for Workflow with Reflexion
- `5` for Agent-Driven Workflow
- `6` for Comparative Evaluation of Workflows
- `0` for Exit

## Implementation Documentation

- **Input Handling**: The script takes user queries and processes them into a structured format.
- **LLM Integration**: The model processes the input and generates responses.
- **Data Output**: The results are formatted and displayed to the user.
- **Error Handling**: Basic exception handling is included for robustness.

## Example Outputs

### Workflow Type 1: Direct Query Processing
**Input:** "Explain machine learning in simple terms."  
**Output:** "Machine learning is a field of AI that allows computers to learn patterns from data..."

### Workflow Type 2: Summarization Task
**Input:** Large paragraph about deep learning  
**Output:** "Deep learning is a subset of machine learning that uses neural networks..."

## Effectiveness Analysis

- **Direct Query Processing**: Provides fast and relevant responses but may lack depth.
- **Summarization Task**: Effectively condenses information but may miss nuances.
- **Other Workflows**: Additional workflows may be tested for different use cases.

## Challenges and Solutions

### Challenge 1: Handling Large Inputs
- **Issue**: The model struggled with long inputs.
- **Solution**: Implemented input truncation and summarization before processing.

### Challenge 2: Response Accuracy
- **Issue**: Some responses lacked precision.
- **Solution**: Tweaked the model parameters and added post-processing.

### Challenge 3: API Rate Limits
- **Issue**: Running multiple queries led to API throttling.
- **Solution**: Introduced request batching and retry mechanisms.

## References
- [OpenAI API Documentation](https://platform.openai.com/docs/guides/functioncalling)
