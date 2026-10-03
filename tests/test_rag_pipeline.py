import pytest
from unittest.mock import MagicMock, patch
from services.rag_pipeline import RAGPipeline


def test_rag_chunking_preserves_page_numbers():
    rag = RAGPipeline(api_key="mock_key")
    pages_data = [
        {
            "page_number": 1,
            "text": "This is page 1 content discussing introductory machine learning concepts and dataset labels.",
            "char_count": 80,
            "has_text": True
        },
        {
            "page_number": 2,
            "text": "This is page 2 content detailing linear regression, cost functions, and gradient descent algorithms.",
            "char_count": 95,
            "has_text": True
        }
    ]

    chunks = rag.create_chunks(pages_data, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0]["page_number"] == 1
    assert chunks[-1]["page_number"] == 2


@patch("services.rag_pipeline.Groq")
def test_rag_answer_with_page_citations(mock_groq_class):
    mock_groq_instance = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="Supervised learning uses labeled training data. Refer to [Page 1]."))
    ]
    mock_groq_instance.chat.completions.create.return_value = mock_completion

    rag = RAGPipeline(api_key="mock_key")
    rag.groq_client = mock_groq_instance

    # Mock collection retrieve
    rag.collection = MagicMock()
    rag.collection.query.return_value = {
        "documents": [["Supervised learning uses labeled dataset."]],
        "metadatas": [[{"page_number": 1, "chunk_id": "c1"}]]
    }

    res = rag.answer_question("What is supervised learning?")
    assert "answer" in res
    assert 1 in res["citations"]
    assert "Page 1" in res["answer"] or 1 in res["citations"]
