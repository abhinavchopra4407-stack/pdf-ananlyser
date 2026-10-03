import pytest
import os
import pymupdf as fitz
from pathlib import Path
from services.pdf_processor import PDFProcessor
from utils.helpers import validate_pdf_file, format_file_size


@pytest.fixture
def sample_pdf(tmp_path) -> str:
    """Creates a temporary 3-page test PDF file."""
    pdf_path = tmp_path / "sample_textbook.pdf"
    doc = fitz.open()

    # Page 1: Chapter 1 Introduction
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Chapter 1: Supervised Learning\nSupervised learning is a machine learning technique using labeled data.")

    # Page 2: Linear Regression
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Section 1.1: Linear Regression\nLinear regression predicts continuous target variables given numerical features.")

    # Page 3: Blank/Minimal page
    doc.new_page()

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_validate_pdf_file_valid(sample_pdf):
    is_valid, err_msg = validate_pdf_file(sample_pdf)
    assert is_valid is True
    assert err_msg == ""


def test_validate_pdf_file_invalid(tmp_path):
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("Not a PDF file.")
    is_valid, err_msg = validate_pdf_file(str(invalid_file))
    assert is_valid is False
    assert "not a PDF" in err_msg


def test_pdf_processor_extraction(sample_pdf):
    processor = PDFProcessor(sample_pdf)
    metadata = processor.process()

    assert metadata["total_pages"] == 3
    assert metadata["filename"] == "sample_textbook.pdf"
    assert len(processor.pages_data) == 3

    # Test page numbers preservation (1-indexed)
    p1 = processor.get_page(1)
    assert p1 is not None
    assert p1["page_number"] == 1
    assert "Supervised Learning" in p1["text"]
    assert p1["has_text"] is True

    p2 = processor.get_page(2)
    assert p2 is not None
    assert p2["page_number"] == 2
    assert "Linear Regression" in p2["text"]

    p3 = processor.get_page(3)
    assert p3 is not None
    assert p3["page_number"] == 3
    assert p3["has_text"] is False

    processor.close()


def test_format_file_size():
    assert format_file_size(500) == "500 B"
    assert "KB" in format_file_size(2048)
    assert "MB" in format_file_size(5 * 1024 * 1024)
