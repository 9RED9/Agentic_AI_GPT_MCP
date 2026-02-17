import json
import os
from typing import List, Optional, Union

from openai import OpenAI

# OpenAI 임베딩으로 만든 FAQ 벡터 캐시 (첫 호출 시 채워짐)
_faq_vectors_cache: Optional[list] = None

EMBEDDING_MODEL = "text-embedding-3-small"


def _openai_embeddings(api_key: str, contents: Union[str, List[str]]) -> list:
    """OpenAI text-embedding-3-small로 임베딩. contents는 str 또는 str 리스트. 벡터 리스트 반환."""
    if isinstance(contents, str):
        contents = [contents]
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=contents,
        encoding_format="float",
    )
    return [e.embedding for e in response.data]


def _cosine_similarity(a, b):
    # 두 벡터의 코사인 유사도 계산 (내적 / (노름_a * 노름_b))
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_retrieved_context(
    query: str,
    data_dir: Optional[str] = None,
    top_k: int = 2,
) -> str:
    """
    OpenAI 임베딩 유사도로 상위 top-k개 FAQ 답변을 검색해 RAG 맥락 문자열로 반환합니다.
    FAQ 벡터는 모듈 캐시를 사용합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("RAG 임베딩용 OPENAI_API_KEY가 설정되지 않았습니다.")

    # RAG 데이터 디렉터리에서 faqs.json 로드
    rag = RagProvider(data_dir=data_dir)
    faq_data = rag.fetch_document_data("faqs.json")

    # FAQ 벡터가 캐시에 없으면 임베딩 생성 후 캐시에 저장
    global _faq_vectors_cache
    if _faq_vectors_cache is None:
        answers = [item["answer"] for item in faq_data]
        faq_embeddings = _openai_embeddings(api_key, answers)
        _faq_vectors_cache = []
        for i in range(len(faq_data)):
            new_item = dict(faq_data[i])
            new_item["vector"] = faq_embeddings[i]
            _faq_vectors_cache.append(new_item)

    faq_vectors = _faq_vectors_cache
    query_embedding = _openai_embeddings(api_key, query)

    # 쿼리 벡터와 FAQ 벡터 간 코사인 유사도로 정렬 후 상위 top_k개 답변만 이어서 반환
    query_vector = query_embedding[0]
    ranked = []
    for item in faq_vectors:
        score = _cosine_similarity(query_vector, item["vector"])
        ranked.append({**item, "score": score})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    top = ranked[:top_k]
    return "\n".join(item["answer"] for item in top)


# 지식 베이스 JSON 로드 및 RAG 프롬프트 조립용 클래스
class RagProvider:
    def __init__(self, data_dir=None):
        # data_dir 미지정 시 이 파일 기준 data 폴더 사용
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self._data_dir = data_dir

    def fetch_document_data(self, file_name: str):
        # 지정한 JSON 파일을 읽어 파이썬 객체로 반환
        file_path = os.path.join(self._data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_simple_rag_prompt(self, query: str) -> str:
        # knowledgeBase.json 전체를 맥락으로 넣은 단순 RAG 프롬프트 생성
        kb_data = self.fetch_document_data("knowledgeBase.json")
        context = "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in kb_data
        )
        return f"""당신은 아래 지식 베이스를 사용할 수 있는 AI 어시스턴트입니다.
    {context}
    위 지식을 바탕으로 다음 사용자 질문에 답하세요.
    User: {query}
    한 단락으로 간단히 답하세요.
    """

    def prepare_rag_prompt(self, query: str, query_vector: list, faq_vectors: list) -> str:
        # query_vector와 FAQ 벡터 유사도로 정렬한 뒤 상위 2개 답변만 맥락으로 넣은 RAG 프롬프트 생성
        ranked = []
        for item in faq_vectors:
            score = _cosine_similarity(query_vector, item["vector"])
            ranked.append({**item, "score": score})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        top2 = ranked[:2]
        context = "\n".join(item["answer"] for item in top2)
        return f"""아래 맥락을 사용해 답하세요. 맥락에 답이 없으면 "문서에는 없지만, 일반 지식으로 최대한 도와드리겠습니다."라고 말하고 일반적인 지식으로 도와주세요.

      Context:
      {context}

      User: {query}""".strip()
