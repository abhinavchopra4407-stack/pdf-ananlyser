import pytest
import os
from unittest.mock import MagicMock, patch
from services.summarizer import SummarizerService
from services.export_service import ExportService



def test_generate_page_summary_empty_page():
    summarizer = SummarizerService(api_key="mock_key")
    summary = summarizer.generate_page_summary(page_num=5, page_text="")
    assert "little or no textual content" in summary.lower()
    assert 5 in summarizer.summary_cache


@patch("services.summarizer.Groq")
def test_generate_page_summary_with_mock_llm(mock_groq_class):
    # Mock Groq client and completion response
    mock_groq_instance = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="### Topic\nSupervised Machine Learning\n\n### Simple Explanation\nModel learns from labeled data."))
    ]
    mock_groq_instance.chat.completions.create.return_value = mock_completion
    mock_groq_class.return_value = mock_groq_instance

    summarizer = SummarizerService(api_key="mock_key")
    summarizer.client = mock_groq_instance

    sample_text = "Supervised learning is a machine learning technique in which a model learns from labeled training data."
    summary = summarizer.generate_page_summary(page_num=12, page_text=sample_text)

    assert "Supervised Machine Learning" in summary
    assert "labeled data" in summary
    assert 12 in summarizer.summary_cache

    # Test summary caching - moving back shouldn't call LLM again
    mock_groq_instance.chat.completions.create.reset_mock()
    cached_summary = summarizer.generate_page_summary(page_num=12, page_text=sample_text)
    assert cached_summary == summary
    mock_groq_instance.chat.completions.create.assert_not_called()


def test_export_service(tmp_path):
    summaries = {
        1: "### Topic\nIntro\n\nExplanation of Chapter 1.",
        2: "### Topic\nLinear Regression\n\nExplanation of Section 2."
    }

    # Patch exports directory to tmp_path
    with patch("config.EXPORTS_DIR", tmp_path):
        txt_path = ExportService.export_to_txt("test_doc.pdf", summaries, "Full summary text.")
        assert os.path.exists(txt_path)
        assert txt_path.endswith(".txt")

        md_path = ExportService.export_to_markdown("test_doc.pdf", summaries, "Full summary text.")
        assert os.path.exists(md_path)
        assert md_path.endswith(".md")
