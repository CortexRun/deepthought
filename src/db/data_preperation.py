from db.db_models import DocumentChunk
from context.deepthought_context import DeepthoughtExecContext

from pypdf import PdfReader
import uuid
import re
import unicodedata

def chunk_page_text(
    text: str, 
    page_number: int, 
    doc_title: str, 
    chunk_len: int = 500, 
    overlap: int = 100
) -> list[DocumentChunk]:

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_len
        chunk_text = text[start:end]
        
        chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                doc_title=doc_title,
                page_number=page_number,
                chunk_id=chunk_id,
                text=chunk_text
            )
        )

        chunk_id += 1
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def clean_text(text: str) -> str:
    allowed = r"[^a-zA-Z0-9äöüÄÖÜß.,:;?!\-()\"' ]"
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(allowed, " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def extract_text(path: str, doc_title: str) -> list[DocumentChunk]:
    reader = PdfReader(path)
    
    all_chunks = []

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text()
        clean = clean_text(raw)

        page_chunks = chunk_page_text(
            text=clean,
            page_number=i,
            doc_title=doc_title
        )

        all_chunks.extend(page_chunks)

    return all_chunks

def embed_chunks(ctx, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    texts = [chunk.text for chunk in chunks]

    embeddings = ctx.create_multiple_embeddings(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = emb

    return chunks