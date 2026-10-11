# Self-Correcting Hybrid RAG Agent for PDFs

An AI-powered PDF question-answering application built with **Python, FastAPI, ChromaDB, BM25, LangGraph, and Groq**. Upload a PDF, ask questions about its contents, and receive answers based on retrieved document passages.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that helps users interact with PDF documents through a conversational interface.

It combines vector search and keyword-based retrieval to find relevant passages. A LangGraph workflow evaluates retrieved context and can rewrite queries and retry retrieval when necessary.

## Features

- **PDF Upload:** Upload supported PDF documents through a web interface.
- **Question Answering:** Ask questions about the contents of an uploaded document.
- **Hybrid Retrieval:** Combine vector similarity search with BM25 keyword search.
- **Reciprocal Rank Fusion (RRF):** Merge results from different retrieval methods.
- **Self-Correction Workflow:** Evaluate retrieved context, rewrite queries, and retry retrieval when needed.
- **LLM-Based Responses:** Generate answers using the retrieved context and Groq-powered language models.
- **Multi-Document Support:** Maintain separate document indexes to help prevent information from different PDFs from mixing.
- **FastAPI Backend:** Provide API endpoints and interactive API documentation.
- **Automated Tests:** Test application behavior and document isolation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | Backend API and application server |
| ChromaDB | Vector storage and similarity search |
| BM25 | Keyword-based document retrieval |
| Reciprocal Rank Fusion | Combine retrieval rankings |
| LangGraph | Manage the self-correction workflow |
| Groq | Access a large language model |
| HTML, CSS, JavaScript | Frontend interface |
| Pytest | Automated testing |

## System Architecture

```text
                 User
                   |
                   v
           PDF Upload Interface
                   |
                   v
          PDF Text Extraction
                   |
                   v
           Text Chunking
                   |
                   v
          Document Indexing
             /         \
            v           v
      Vector Search   BM25 Search
            \           /
             v         v
        Reciprocal Rank Fusion
                   |
                   v
         Retrieved Context
                   |
                   v
       LangGraph Self-Correction
          |                |
          |          Rewrite Query
          |          and Retry Search
          |                |
          <----------------+
                   |
                   v
           Groq Language Model
                   |
                   v
          Answer to the User
```

### How It Works

1. **Upload:** The user uploads a PDF through the frontend.
2. **Text extraction:** The application extracts text from the document.
3. **Chunking and indexing:** The extracted text is split into smaller chunks and indexed for retrieval.
4. **Hybrid search:** Vector search and BM25 retrieve relevant passages.
5. **Result fusion:** Reciprocal Rank Fusion combines the rankings from both retrieval methods.
6. **Context evaluation:** The LangGraph workflow evaluates the retrieved context and may rewrite the query and retry retrieval.
7. **Answer generation:** The language model generates an answer using the retrieved document context.
8. **Response display:** The answer is returned to the user through the web interface.

## Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- A Groq API key
- Internet access for installing dependencies and accessing the configured language model

### 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/tasleembeleri8-oss/Self-Correcting-Hybrid-RAG-Agent-for-PDFs.git
cd Self-Correcting-Hybrid-RAG-Agent-for-PDFs
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Use the exact environment variable names expected by your application. If your code uses additional settings, configure those as well.

**Security:** Never commit your real API key or `.env` file to GitHub.

### 5. Start the application

From the project root, run:

```bash
python -m uvicorn app.main:app --reload
```

Open the application in your browser:

- **Web application:** http://127.0.0.1:8000
- **API documentation:** http://127.0.0.1:8000/docs
- **Health endpoint:** http://127.0.0.1:8000/health

The application must be running for these local URLs to work.

## Testing

Run the automated test suite from the project root:

```bash
python -m pytest -q
```

The project previously passed 15 automated tests. Rerun the tests to verify the current version.

Manual testing has also been performed with multiple PDF uploads and document-related questions.

The evaluation scripts are located in the `eval/` directory. Evaluation datasets and results containing private document content are intentionally excluded from the public repository.

## Limitations

- Scanned PDFs may require OCR because ordinary text extraction may not capture text embedded in images.
- Answer quality depends on PDF text extraction, retrieval quality, and the language model.
- Generated answers may be incorrect and should be checked against the original document.
- Self-correction does not guarantee improved accuracy; a measured comparison is needed to establish its impact.
- Document storage, expiration, upload limits, and persistence depend on the current implementation.
- The application is a project prototype and does not yet provide a complete production-grade authentication and authorization system.

## Future Improvements

- Deploy a public demo.
- Add OCR support for scanned PDFs.
- Improve source citations and passage traceability.
- Measure retrieval and answer quality using a fixed evaluation dataset.
- Compare performance with and without self-correction.
- Add persistent storage, authentication, and user-specific document management.
- Improve production deployment, logging, and error handling.

## Project Repository

[View the source code on GitHub](https://github.com/tasleembeleri8-oss/Self-Correcting-Hybrid-RAG-Agent-for-PDFs)

## Author

Developed as a Python and AI project exploring Retrieval-Augmented Generation, hybrid information retrieval, document question answering, and self-correcting AI workflows.