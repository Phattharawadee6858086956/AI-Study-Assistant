# AI Study Assistant

A small Flask app that lets you upload multiple PDF or TXT documents and ask questions across them. Upload files one at a time or select several together. It uses retrieval-augmented generation (RAG): it extracts document text, splits it into overlapping chunks, embeds the chunks with Sentence Transformers, selects the closest chunks across all uploaded documents with cosine similarity, and sends those excerpts to Ollama's `gemma3:1b` model. The prompt asks the model to stay grounded in the excerpts and say when the documents do not contain an answer.

## Requirements

- Python 3.10 or newer
- Ollama installed and running locally
- The Ollama `gemma3:1b` model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull gemma3:1b
```

Start Ollama in another terminal if it is not already running, then launch the app:

```bash
python app.py
```

Open http://127.0.0.1:5000. Upload one or more text-based PDF or UTF-8 TXT files (maximum 12 MB per upload request), then enter a question. The filenames remain listed, and later uploads are added to the current in-memory collection. The first upload downloads the sentence-transformer model if it is not cached already. Scanned PDFs without an embedded text layer are not OCR-processed.

## Run tests

```bash
pytest
```

Tests use a fake embedder and answer function, so they do not require downloading a model or running Ollama.

## Configuration

Optional environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | Local development value | Flask session signing key; set a long random value outside local development. |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL. |
| `OLLAMA_MODEL` | `gemma3:1b` | Ollama generation model. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers model used for embeddings. |

Uploaded content and embeddings are held in server memory and are not written to disk. They are cleared when the app process stops. PDF files are ignored by Git so study materials are not committed. For learning and local use, the Flask development server is convenient; do not expose it directly to the public internet.
