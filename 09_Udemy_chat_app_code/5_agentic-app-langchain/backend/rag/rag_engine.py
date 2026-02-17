"""
RAG 엔진: LangChain Chroma로 쿼리 임베딩 후 유사 검색, LLM용 프롬프트 문자열 생성.
"""
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "vector_data")


def _get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for RAG embeddings")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def _get_vector_store():
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=_PERSIST_DIR,
    )


def build_prompt(query: str, top_k: int = 4) -> dict:
    """
    쿼리에 대해 벡터 검색 후 컨텍스트를 붙인 프롬프트 문자열 반환.
    Returns: {"prompt": str, "sources": list}
    """
    try:
        vector_store = _get_vector_store()
        docs = vector_store.similarity_search(query, k=top_k)
        if not docs:
            return {
                "prompt": f"No matching documents found for query: {query}",
                "sources": [],
            }
        context = "\n\n".join(
            f"SOURCE {i + 1}:\n{doc.page_content}\nMETA: {doc.metadata}"
            for i, doc in enumerate(docs)
        )
        prompt = f"""You are an assistant that answers questions only using the information in the context.
If the answer is not present, reply: "I don't have enough information."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        return {"prompt": prompt, "sources": [{"content": d.page_content, "metadata": d.metadata} for d in docs]}
    except Exception as e:
        return {
            "prompt": f"RAG ERROR: {e}",
            "sources": [],
        }


def rag_search(query: str, top_k: int = 4) -> str:
    """
    MCP/에이전트에서 호출: RAG 결과의 prompt 문자열만 반환 (LLM 컨텍스트로 사용).
    """
    result = build_prompt(query, top_k)
    return result["prompt"]
