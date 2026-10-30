"""A small, beginner-friendly retrieval-augmented study assistant."""

from __future__ import annotations

import os
import uuid
from io import BytesIO
from typing import Any

import numpy as np
import requests
from flask import Flask, flash, redirect, render_template, request, session, url_for
from pypdf import PdfReader
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:1b"


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from a PDF or UTF-8 text file."""
    extension = os.path.splitext(filename.lower())[1]
    if extension == ".txt":
        return content.decode("utf-8-sig")
    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Please upload a PDF or TXT file.")


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into readable, overlapping character chunks."""
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a word boundary when it is reasonably close to the target size.
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


class SentenceTransformerEmbedder:
    """Load the embedding model only when the first document is uploaded."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: Any = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def retrieve(query: str, chunks: list[dict[str, Any]], embedder: Any, top_k: int = 4) -> list[dict[str, Any]]:
    """Return the most similar chunks by cosine similarity."""
    if not chunks:
        return []
    query_vector = np.asarray(embedder.encode([query])[0], dtype=np.float32)
    matrix = np.asarray([chunk["embedding"] for chunk in chunks], dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    norms = np.linalg.norm(matrix, axis=1)
    similarities = np.divide(matrix @ query_vector, norms * query_norm,
                             out=np.zeros(len(chunks), dtype=np.float32),
                             where=(norms * query_norm) != 0)
    indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[int(index)] for index in indices]


def ask_ollama(question: str, context: str, *, base_url: str = DEFAULT_OLLAMA_URL,
               model: str = DEFAULT_OLLAMA_MODEL, timeout: int = 120) -> str:
    """Ask Ollama to answer using only the supplied document excerpts."""
    prompt = (
        "You are a helpful study assistant. Answer the question using only the context below. "
        "If the answer is not in the context, say you could not find it in the uploaded document. "
        "Explain clearly for a beginner.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    answer = response.json().get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")
    return answer


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "local-development-key-change-me"),
        MAX_CONTENT_LENGTH=12 * 1024 * 1024,
        CHUNK_SIZE=900,
        CHUNK_OVERLAP=150,
        TOP_K=4,
        OLLAMA_URL=os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        OLLAMA_MODEL=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        EMBEDDING_MODEL=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    if test_config:
        app.config.update(test_config)
    app.extensions["documents"] = {}
    app.extensions["embedder"] = app.config.get("EMBEDDER") or SentenceTransformerEmbedder(app.config["EMBEDDING_MODEL"])
    app.extensions["answer_fn"] = app.config.get("ANSWER_FN") or ask_ollama

    @app.get("/")
    def index():
        documents_by_id = app.extensions["documents"]
        document_ids = session.get("document_ids", [])
        documents = [documents_by_id[doc_id] for doc_id in document_ids if doc_id in documents_by_id]
        return render_template("index.html", documents=documents, answer=session.pop("answer", None))

    @app.post("/upload")
    def upload():
        # Accept a batch from the multi-select input, and keep compatibility
        # with clients that still send the older single-file "document" field.
        uploaded_files = request.files.getlist("documents") or request.files.getlist("document")
        uploaded_files = [uploaded for uploaded in uploaded_files if uploaded and uploaded.filename]
        if not uploaded_files:
            flash("Choose a PDF or TXT file first.", "error")
            return redirect(url_for("index"))
        for uploaded in uploaded_files:
            filename = secure_filename(uploaded.filename)
            extension = os.path.splitext(filename.lower())[1]
            if extension not in ALLOWED_EXTENSIONS:
                flash(f"Skipped {filename}. Only PDF and TXT files are supported.", "error")
                continue
            try:
                text = extract_text(filename, uploaded.read())
                chunks = split_into_chunks(text, app.config["CHUNK_SIZE"], app.config["CHUNK_OVERLAP"])
                if not chunks:
                    raise ValueError("No readable text was found in that file.")
                embeddings = app.extensions["embedder"].encode(chunks)
                document_id = str(uuid.uuid4())
                app.extensions["documents"][document_id] = {
                    "name": filename,
                    "chunks": [{"name": filename, "text": chunk, "embedding": vector}
                               for chunk, vector in zip(chunks, embeddings)],
                }
                document_ids = session.get("document_ids", [])
                document_ids.append(document_id)
                session["document_ids"] = document_ids
                flash(f"Loaded {filename} ({len(chunks)} text chunks).", "success")
            except Exception as exc:
                app.logger.exception("Could not process uploaded document")
                flash(f"Could not read {filename}: {exc}", "error")
        return redirect(url_for("index"))

    @app.post("/ask")
    def ask():
        question = request.form.get("question", "").strip()
        documents_by_id = app.extensions["documents"]
        documents = [documents_by_id[doc_id] for doc_id in session.get("document_ids", [])
                     if doc_id in documents_by_id]
        if not question:
            flash("Enter a question first.", "error")
        elif not documents:
            flash("Upload a document before asking a question.", "error")
        else:
            try:
                all_chunks = [chunk for document in documents for chunk in document["chunks"]]
                relevant = retrieve(question, all_chunks, app.extensions["embedder"], app.config["TOP_K"])
                context = "\n\n".join(f"[{item['name']}] {item['text']}" for item in relevant)
                app_id = app.extensions["answer_fn"]
                session["answer"] = app_id(question, context, base_url=app.config["OLLAMA_URL"],
                                           model=app.config["OLLAMA_MODEL"])
            except Exception as exc:
                app.logger.exception("Could not answer question")
                flash(f"Could not get an answer. Check that Ollama is running: {exc}", "error")
        return redirect(url_for("index"))

    @app.errorhandler(413)
    def too_large(_error):
        flash("That file is too large. The limit is 12 MB.", "error")
        return redirect(url_for("index"))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
