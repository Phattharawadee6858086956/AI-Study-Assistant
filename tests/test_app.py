from io import BytesIO

import numpy as np
import pytest

import app as study_app
from app import create_app, extract_text, retrieve, split_into_chunks


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([1.0, 0.0] if any(word in lower for word in ("biology", "cell", "quantum")) else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def client():
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-key",
        "EMBEDDER": FakeEmbedder(),
        "ANSWER_FN": lambda question, context, **kwargs: f"Grounded answer for: {question}\n{context}",
        "CHUNK_SIZE": 120,
        "CHUNK_OVERLAP": 20,
    })
    return app.test_client()


def test_extract_text_and_split_into_chunks():
    text = extract_text("notes.txt", b"\xef\xbb\xbfCell biology studies cells.\nThey are living units.")
    assert text.startswith("Cell biology")
    chunks = split_into_chunks("word " * 100, chunk_size=80, overlap=10)
    assert len(chunks) > 1
    assert all(chunks)


def test_retrieve_ranks_matching_chunks():
    chunks = [
        {"text": "Astronomy studies stars", "embedding": np.array([0.0, 1.0])},
        {"text": "Cell biology studies cells", "embedding": np.array([1.0, 0.0])},
    ]
    result = retrieve("biology", chunks, FakeEmbedder(), top_k=1)
    assert result[0]["text"].startswith("Cell biology")


def test_upload_txt_and_ask_question(client):
    response = client.post("/upload", data={
        "document": (BytesIO(b"Biology studies living cells. Cells are the basic unit of life."), "biology.txt")
    }, content_type="multipart/form-data", follow_redirects=True)
    assert b"Ready to study" in response.data
    response = client.post("/ask", data={"question": "What are cells?"}, follow_redirects=True)
    assert b"Grounded answer for: What are cells?" in response.data
    assert b"basic unit of life" in response.data


def test_rejects_unsupported_extension(client):
    response = client.post("/upload", data={
        "document": (BytesIO(b"nope"), "notes.docx")
    }, content_type="multipart/form-data", follow_redirects=True)
    assert b"Only PDF and TXT files are supported" in response.data


def test_asking_without_document_shows_helpful_message(client):
    response = client.post("/ask", data={"question": "What is this about?"}, follow_redirects=True)
    assert b"Upload a document before asking" in response.data


def test_five_pdf_uploads_are_retained_and_searched_together(client, monkeypatch):
    # Keep the test focused on upload/session/RAG behavior, not PDF parsing.
    monkeypatch.setattr(study_app, "extract_text", lambda filename, content: content.decode("utf-8"))
    documents = [
        ("01_Introduction.pdf", "This document introduces the course."),
        ("02_Math.pdf", "This document describes equations."),
        ("03_Quantum.pdf", "Quantum tunneling lets particles cross an energy barrier."),
        ("04_Examples.pdf", "This document contains practice examples."),
        ("05_ReLU_Activation.pdf", "ReLU returns the maximum of zero and its input."),
    ]

    for filename, content in documents:
        response = client.post("/upload", data={
            "documents": (BytesIO(content.encode("utf-8")), filename)
        }, content_type="multipart/form-data", follow_redirects=True)
        assert response.status_code == 200

    # Every filename remains visible after the last upload.
    page = client.get("/")
    for filename, _content in documents:
        assert filename.encode() in page.data

    # A question about an earlier upload retrieves its chunk across the full set.
    answer = client.post("/ask", data={"question": "Explain quantum tunneling"}, follow_redirects=True)
    assert b"Quantum tunneling lets particles cross an energy barrier" in answer.data
    assert b"[03_Quantum.pdf]" in answer.data


def test_batch_pdf_upload_accepts_multiple_files(client, monkeypatch):
    monkeypatch.setattr(study_app, "extract_text", lambda filename, content: content.decode("utf-8"))
    response = client.post("/upload", data={
        "documents": [
            (BytesIO(b"First uploaded text."), "first.pdf"),
            (BytesIO(b"Second uploaded text."), "second.pdf"),
        ]
    }, content_type="multipart/form-data", follow_redirects=True)
    assert b"first.pdf" in response.data
    assert b"second.pdf" in response.data
