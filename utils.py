from PyPDF2 import PdfReader
from docx import Document
import markdown
import bleach

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(para.text for para in doc.paragraphs)


def sanitize_markdown(md: str) -> str:
    """Render markdown and sanitize the resulting HTML."""
    html = markdown.markdown(md, extensions=["nl2br"])  # preserve single newlines
    allowed_tags = [
        "p",
        "br",
        "strong",
        "em",
        "ul",
        "ol",
        "li",
        "pre",
        "code",
        "blockquote",
    ]
    return bleach.clean(html, tags=allowed_tags, strip=True)
