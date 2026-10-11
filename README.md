# Self-Correcting Hybrid RAG Agent for PDFs

An AI-powered application that allows users to upload PDF documents and ask questions about their contents using Retrieval-Augmented Generation (RAG).

## Overview

This project helps users find information in PDF documents through a conversational interface. It retrieves relevant document passages and uses a Large Language Model (LLM) to generate answers grounded in the retrieved context.

The system also includes a self-correction workflow that can evaluate retrieved context, rewrite queries, and retry retrieval when necessary.

## Features

- Upload PDF documents and ask questions about their contents.
- Hybrid retrieval using vector search and BM25 keyword search.
- Reciprocal Rank Fusion (RRF) to combine retrieval results.
- Self-correction workflow using LangGraph.
- Query rewriting and retrieval retries when needed.
- FastAPI backend with interactive API documentation.
- Separate document indexes to help prevent answers from mixing information across PDFs.

## Technology Stack

- **Language:** Python
- **Backend:** FastAPI
- **Vector Database:** ChromaDB
- **Keyword Retrieval:** BM25
- **Agent Workflow:** LangGraph
- **LLM Provider:** Groq
- **Retrieval Method:** Hybrid Search with Reciprocal Rank Fusion
- **Frontend:** HTML, CSS and JavaScript

## Architecture

1. **PDF Upload:** The user uploads a PDF through the web interface.
2. **Document Processing:** The PDF text is extracted and divided into smaller chunks.
3. **Indexing:** Document chunks are stored for vector retrieval and keyword search.
4. **Hybrid Retrieval:** Relevant passages are retrieved using vector search and BM25, then combined using RRF.
5. **Context Evaluation:** The LangGraph workflow evaluates retrieved context and may rewrite the query and retry retrieval.
6. **Answer Generation:** The LLM generates a response using the retrieved document context.
7. **Response Display:** The answer is returned to the user through the web interface.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd rag-agent
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and add the environment variables required by your application. For example:

```env
GROQ_API_KEY=your_groq_api_key
```

Check your code for the exact environment variable names. Never commit your real API key to GitHub.

### 5. Run the application

```bash
python -m uvicorn app.main:app --reload
```

Open the application at:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## Testing

The application has been manually tested with PDF uploads and questions about document contents. Multiple-document behavior has also been tested to check that answers use the appropriate document.

Automated tests have previously passed, but rerun the test suite before publishing the repository.

## Limitations

- Scanned PDFs may require OCR support.
- Answer quality depends on document text extraction and retrieval quality.
- LLM responses may still be incorrect and should be checked against the source document.
- Document storage, expiration, and upload limits depend on the current application configuration.
- Self-correction does not guarantee a more accurate answer.

## Future Improvements

- Deploy a publicly accessible demo.
- Add OCR for scanned PDFs.
- Improve evaluation using a fixed set of questions and expected answers.
- Add persistent document storage and user authentication.
- Improve source citations and answer-quality measurement.

## Author

Developed as a Python, AI and RAG project to explore document question answering, hybrid information retrieval and self-correcting AI workflows.