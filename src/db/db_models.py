from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentChunk:
    id: str
    doc_title: str
    page_number: int
    chunk_id: int
    text: str
    embedding: Optional[list[float]] = None