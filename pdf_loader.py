from pypdf import PdfReader
from pypdf.errors import PdfStreamError

def load_pdf_text(filepath : str) -> str:
    try:

        reader = PdfReader(filepath)
        text = ""                                           

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"


        return text
    
    except PdfStreamError:
        raise ValueError("Invalid or corrupted PDF file")