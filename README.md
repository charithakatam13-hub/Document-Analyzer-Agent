# Legal Document Analyzer with RAG

A Retrieval-Augmented Generation (RAG) application that analyzes legal documents and allows users to ask questions about uploaded PDF agreements.

The system uses:

* OpenRouter
* GPT-4o Mini
* LangChain
* FAISS Vector Database
* PDF Processing
* RAG Retrieval Pipeline

---

## Features

### Legal Document Analysis

Extracts:

* Document Summary
* Parties Involved
* Important Clauses
* Risky Clauses
* Missing Clauses
* Obligations
* Payment Terms
* Confidentiality Terms
* Termination Conditions
* Risk Rating
* Practical Recommendations

---

### Document Question Answering

After analysis, users can ask unlimited questions about the uploaded document.

Examples:

* What is the termination clause?
* What payment terms are mentioned?
* Are there confidentiality requirements?
* What penalties exist?
* What are the responsibilities of Party A?

The system answers only from the uploaded document using Retrieval-Augmented Generation (RAG).

---

## Architecture

```text
PDF Upload
     │
     ▼
PDF Loader
     │
     ▼
Text Extraction
     │
     ▼
Chunking
     │
     ▼
Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Retriever
     │
     ▼
GPT-4o Mini
     │
     ▼
Answer Generation
```

---

## Technologies Used

* Python
* LangChain
* OpenRouter
* GPT-4o Mini
* FAISS
* PyPDF
* RAG

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Environment Setup

In Google Colab:

```python
from google.colab import userdata
import os

os.environ["OPENAI_API_KEY"] = userdata.get("key")
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["MODEL_NAME"] = "openai/gpt-4o-mini"
```

Store your OpenRouter API Key in Colab Secrets:

```text
key
```

---

## Running the Project

Run:

```bash
python main.py
```

Steps:

1. Upload PDF
2. Enter agreement type
3. Receive legal analysis
4. Ask unlimited questions
5. Type exit to quit

---

## Example Questions

```text
What are the payment terms?

Who are the parties involved?

What are the termination conditions?

What penalties exist?

Is there a confidentiality clause?

What obligations does Party B have?
```

---

## Example Output

```text
Document Summary:
Employment Agreement between ABC Ltd and John Doe.

Parties:
ABC Ltd
John Doe

Risk Rating:
Medium

Recommendation:
Review termination and indemnity clauses carefully.
```

---

## Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* PDF Processing
* Prompt Engineering
* OpenRouter Integration
* Legal Document Analysis

---

## Future Improvements

* Multi-PDF Support
* Citation-Based Answers
* Contract Comparison
* Clause Extraction Agent
* Streamlit UI
* ChromaDB / Qdrant Support
* Agentic Legal Review Workflow

---

## Disclaimer

This project is for educational purposes only.

It does not provide legal advice and should not replace consultation with a qualified legal professional.

---

## Author

Charitha Katam

AI Engineering | RAG | LangChain | LLM Applications
