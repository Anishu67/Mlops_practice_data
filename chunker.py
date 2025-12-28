def chunk_text(text : str, chunk_size = 200, overlap = 50):
    words = text.split()
    chunks= []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks