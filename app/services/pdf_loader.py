from pypdf import PdfReader


def load_pdf_text(path: str) -> str:
    """
    Extract full text from PDF.
    """
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text_by_policy(text: str):
    """
    Split PDF into chunks using 'Policy:' sections.
    Ideal for HR and company policy documents.
    """
    sections = text.split("Policy:")
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        chunk = "Policy: " + sec
        chunks.append(chunk)

    return chunks