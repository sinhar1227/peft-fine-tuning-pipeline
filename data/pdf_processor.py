# ============================================================
# 5. Extract text from PDF
# ============================================================

import re
import unicodedata
from typing import List, Dict, Any


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    # Extract page-level text from a PDF.
    import fitz  # PyMuPDF
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = text.strip() if text else ""
            if text:
                pages.append({
                    "page": page_index,
                    "text": text,
                    "char_count": len(text),
                })
    return pages


# ============================================================
# 6. Text cleaning utilities
# ============================================================

def clean_pdf_text(text: str) -> str:
    # Standardize Unicode text so visually similar characters are treated consistently.
    # Example: "ＡＭＰＫ" becomes "AMPK" and "ﬁ" becomes "fi".
    text = unicodedata.normalize("NFKC", text)

    # Remove invisible characters that may appear during PDF text extraction.
    text = text.replace("\u200b", "").replace("\ufeff", "")

    # Join words broken by line hyphenation, e.g., "gluconeogene-\nsis" -> "gluconeogenesis".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace multiple spaces/tabs with a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Convert three or more newlines into a standard paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that contain only page numbers.
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Split text into paragraphs, clean each paragraph, and remove empty ones.
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\n+", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)

    # Join cleaned paragraphs with one blank line between them.
    return "\n\n".join(paragraphs)


def clean_pdf_pages(pdf_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Apply clean_pdf_text to every page in the extracted PDF.
    cleaned = []
    for page in pdf_pages:
        cleaned_text = clean_pdf_text(page["text"])
        cleaned.append({
            "page": page["page"],
            "text": cleaned_text,
            "char_count": len(cleaned_text),
        })
    return cleaned


# ============================================================
# 7. Split cleaned pages into paragraphs
# ============================================================
# This step converts cleaned page-level text into paragraph-level records.

def split_into_paragraph_records(cleaned_pages: List[Dict[str, Any]], min_chars: int = 80) -> List[Dict[str, Any]]:
    records = []
    for page in cleaned_pages:
        # Split page text into paragraphs using blank lines.
        paragraphs = page["text"].split("\n\n")
        for idx, para in enumerate(paragraphs, start=1):
            # Remove extra spaces from the beginning and end.
            para = para.strip()
            # Skip very short paragraphs because they are usually headings, page numbers, or noise.
            if len(para) < min_chars:
                continue
            # Store each useful paragraph with basic metadata.
            records.append({
                "text": para,
                "source_page": page["page"],
                "paragraph_id": idx,
                "char_count": len(para),
            })
    return records
