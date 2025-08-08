import requests
import fitz  # PyMuPDF
import docx
import re
import os
import tempfile
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser
from typing import List


def extract_text_from_pdf(content: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove(tmp_path)
        return text
    except Exception as e:
        return f"[ERROR: Failed to extract PDF text] {str(e)}"


def extract_text_from_docx(content: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        doc = docx.Document(tmp_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        os.remove(tmp_path)
        return text
    except Exception as e:
        return f"[ERROR: Failed to extract DOCX text] {str(e)}"


def extract_text_from_eml(content: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(content)
        text_parts = []

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text_parts.append(part.get_content())
        else:
            text_parts.append(msg.get_content())

        return "\n".join(text_parts)
    except Exception as e:
        return f"[ERROR: Failed to extract EML text] {str(e)}"


def extract_text_from_html(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        return text
    except Exception as e:
        return f"[ERROR: Failed to extract HTML text] {str(e)}"


def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise Exception("Failed to download document.")

        content_type = response.headers.get("Content-Type", "").lower()
        content = response.content
        extension = url.split(".")[-1].lower()

        if "application/pdf" in content_type or extension == "pdf":
            text = extract_text_from_pdf(content)

        elif (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type
            or extension == "docx"
        ):
            text = extract_text_from_docx(content)

        elif "message/rfc822" in content_type or extension == "eml":
            text = extract_text_from_eml(content)

        elif "text/html" in content_type or extension in ["html", "htm"]:
            text = extract_text_from_html(response.text)

        elif "text/plain" in content_type or extension == "txt":
            text = response.text

        else:
            text = "[Unsupported file type or content]"

        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        return cleaned_text

    except Exception as e:
        return f"[ERROR: Failed to process URL: {url}] {str(e)}"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
