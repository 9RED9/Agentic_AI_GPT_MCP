"""
LangChain으로 RAG 문서 수집: 폴더 내 .md/.txt 로드 -> 청킹 -> Chroma에 저장.
실행: python -m rag.ingest data/rag_docs
"""
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def _get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for ingest")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def ingest_folder(folder_path: str, chunk_size: int = 800, chunk_overlap: int = 100):
    path = Path(folder_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder_path}")

    loader = DirectoryLoader(
        str(path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    txt_loader = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs.extend(txt_loader.load())

    if not docs:
        print("No .md or .txt files found.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "vector_data")
    collection = os.getenv("CHROMA_COLLECTION", "rag_documents")
    embeddings = _get_embeddings()

    vector_store = Chroma.from_documents(
        splits,
        embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    vector_store.persist()
    print(f"Ingested {len(splits)} chunks from {len(docs)} documents into {persist_dir}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingest <folder_path>")
        sys.exit(1)
    ingest_folder(sys.argv[1])
