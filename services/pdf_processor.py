import pymupdf as fitz
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import sys

# Add project root to sys.path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.helpers import validate_pdf_file, format_file_size


class PDFProcessor:
    """Service to process PDF files and extract text page by page with metadata."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = Path(file_path).name if file_path else ""
        self.file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
        self.doc: Optional[fitz.Document] = None
        self.pages_data: List[Dict[str, Any]] = []
        self.total_pages = 0
        self.is_scanned = False
        self.chapters: List[Dict[str, Any]] = []

    def process(self) -> Dict[str, Any]:
        """
        Processes the PDF file and returns document summary metadata & extracted pages.
        """
        is_valid, err_msg = validate_pdf_file(self.file_path)
        if not is_valid:
            raise ValueError(err_msg)

        try:
            self.doc = fitz.open(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF document: {str(e)}")

        if self.doc.is_encrypted:
            raise ValueError("The PDF document is encrypted or password-protected.")

        self.total_pages = len(self.doc)
        if self.total_pages == 0:
            raise ValueError("The PDF document contains 0 pages.")

        self.pages_data = []
        pages_with_minimal_text = 0

        # Extract bookmarks / TOC if present
        try:
            toc = self.doc.get_toc()
            for item in toc:
                # item format: [lvl, title, page_number]
                if len(item) >= 3:
                    self.chapters.append({
                        "level": item[0],
                        "title": item[1].strip(),
                        "page_number": item[2]
                    })
        except Exception:
            self.chapters = []

        for page_idx in range(self.total_pages):
            page = self.doc[page_idx]
            page_num = page_idx + 1
            text = page.get_text("text").strip()

            char_count = len(text)
            has_text = char_count >= 15

            if not has_text:
                pages_with_minimal_text += 1

            self.pages_data.append({
                "page_number": page_num,
                "text": text,
                "char_count": char_count,
                "has_text": has_text
            })

        # If > 70% of pages have minimal text, flag as scanned/OCR needed
        if self.total_pages > 0 and (pages_with_minimal_text / self.total_pages) > 0.7:
            self.is_scanned = True

        return self.get_metadata()

    def get_metadata(self) -> Dict[str, Any]:
        """Returns document level metadata."""
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "file_size": self.file_size,
            "file_size_formatted": format_file_size(self.file_size),
            "is_scanned": self.is_scanned,
            "has_toc": len(self.chapters) > 0,
            "chapters": self.chapters
        }

    def get_page(self, page_num: int) -> Optional[Dict[str, Any]]:
        """Retrieves page data for a 1-indexed page number."""
        if 1 <= page_num <= self.total_pages:
            return self.pages_data[page_num - 1]
        return None

    def close(self):
        """Close PDF document."""
        if self.doc:
            self.doc.close()
