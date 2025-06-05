from PyPDF2 import PdfReader
from docx import Document
from html import escape
import re

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def sanitize_markdown(md: str) -> str:
    """Convert a small subset of Markdown to safe HTML."""
    text = escape(md)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = text.replace("\n", "<br>")
    return text
