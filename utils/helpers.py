import os
from pathlib import Path

def validate_pdf_file(file_path: str) -> tuple[bool, str]:
    """
    Validates if the given file exists, is a PDF, and is non-empty.
    Returns (is_valid, error_message)
    """
    if not file_path:
        return False, "No file provided."
    
    path = Path(file_path)
    if not path.exists():
        return False, f"File not found: {path.name}"
    
    if path.suffix.lower() != ".pdf":
        return False, "Selected file is not a PDF. Please upload a valid .pdf file."
    
    file_size = path.stat().st_size
    if file_size == 0:
        return False, "Uploaded PDF file is empty (0 bytes)."
    
    # 100 MB max file size limit check
    if file_size > 100 * 1024 * 1024:
        return False, "File size exceeds the 100 MB limit."
        
    return True, ""


def format_file_size(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent invalid characters in exports."""
    name = Path(filename).stem
    clean = "".join(c for c in name if c.isalnum() or c in ("-", "_", " ")).strip()
    return clean if clean else "document"
