from typing import List
import logfire
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        chunks = splitter.split_text(text)

        logfire.info(f"✅ Generated {len(chunks)} chunks")

        return chunks