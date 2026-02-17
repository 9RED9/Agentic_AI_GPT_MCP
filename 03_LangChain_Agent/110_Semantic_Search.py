# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Semantic Search (시맨틱 검색 엔진 구축)
#
# 이 튜토리얼에서는 LangChain의 문서 로더, 임베딩, 벡터 스토어를 사용하여 PDF 문서에 대한 시맨틱 검색 엔진을 구축합니다.
#
# **시맨틱 검색(Semantic Search)** 은 키워드 매칭이 아닌 의미 기반으로 문서를 검색하는 방법입니다.
# 임베딩을 사용하여 텍스트를 벡터로 변환하고, 유사도 기반으로 관련 문서를 찾습니다.
#
# **참고**: [LangChain 공식 문서 - Semantic Search](https://docs.langchain.com/oss/python/langchain/knowledge-base)
#
# ## 학습 내용
# - 문서 로더 (Document Loaders)
# - 텍스트 분할 (Text Splitting)
# - 임베딩 (Embeddings)
# - 벡터 스토어 (Vector Stores)
# - 검색기 (Retrievers)

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith 추적 (선택적)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# %% [markdown]
# ## 1. 문서 로드하기
#
# PDF 파일을 `Document` 객체로 로드합니다.
# LangChain은 다양한 문서 소스를 지원합니다: PDF, 웹페이지, 데이터베이스, API 등.

# %%
# pip install pypdf

# %%
from langchain_community.document_loaders import PyPDFLoader

# PDF 파일 로드
file_path = "./example_data/nke-10k-2023_korean.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(f"로드된 페이지 수: {len(docs)}")
print(f"\n첫 페이지 미리보기:\n{docs[0].page_content[:200]}...")
print(f"\n메타데이터: {docs[0].metadata}")

# %% [markdown]
# ## 2. 텍스트 분할 (Text Splitting)
#
# 페이지 단위는 검색에 너무 큽니다. 더 작은 청크로 분할하여 정확한 검색이 가능하도록 합니다.
#
# **중요 설정:**
# - `chunk_size`: 각 청크의 최대 크기
# - `chunk_overlap`: 청크 간 중첩 (문맥 유지)

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크 최대 1000자
    chunk_overlap=200,    # 청크 간 200자 중첩
    add_start_index=True  # 원본 문서 내 시작 위치 저장
)

# 문서 분할
all_splits = text_splitter.split_documents(docs)

print(f"분할된 청크 수: {len(all_splits)}")
print(f"\n첫 번째 청크:\n{all_splits[0].page_content[:300]}...")

# %% [markdown]
# ## 3. 임베딩 (Embeddings)
#
# 텍스트를 숫자 벡터로 변환합니다. 의미가 유사한 텍스트는 벡터 공간에서 가까운 위치에 있습니다.

# %%
# OpenAI 임베딩 사용 (권장)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 또는 Hugging Face 한국어 모델 사용
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="nlpai-lab/KURE-v1")

# 임베딩 테스트
query_vector = embeddings.embed_query("나이키의 매출")
print(f"임베딩 벡터 차원: {len(query_vector)}")
print(f"벡터 일부: {query_vector[:5]}...")

# %% [markdown]
# ## 4. 벡터 스토어 (Vector Store)
#
# 임베딩된 문서를 저장하고 유사도 검색을 수행합니다.
# 여러 벡터 스토어 옵션: InMemoryVectorStore, Chroma, Pinecone, Weaviate 등

# %% [markdown]
# ### 4.1 InMemoryVectorStore (간단한 테스트용)

# %%
from langchain_core.vectorstores import InMemoryVectorStore

# 인메모리 벡터 스토어 생성
vector_store = InMemoryVectorStore(embeddings)

# 문서 추가
document_ids = vector_store.add_documents(documents=all_splits)
print(f"저장된 문서 ID 수: {len(document_ids)}")

# %% [markdown]
# ### 4.2 Chroma (영구 저장)

# %%
from langchain_chroma import Chroma

# Chroma 벡터 스토어 생성 (로컬에 영구 저장)
vector_store = Chroma(
    collection_name="nike_10k",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"  # 로컬 저장 경로
)

# 문서 추가
document_ids = vector_store.add_documents(documents=all_splits)
print(f"Chroma에 저장된 문서 수: {len(document_ids)}")

# %% [markdown]
# ## 5. 검색 수행
#
# 벡터 스토어에서 유사도 기반 검색을 수행합니다.

# %%
# 기본 유사도 검색
query = "나이키는 미국에 몇 개의 유통 센터를 가지고 있나요?"
results = vector_store.similarity_search(query, k=3)

print(f"검색 쿼리: {query}\n")
print("=" * 80)
for i, doc in enumerate(results, 1):
    print(f"\n[결과 {i}]")
    print(f"페이지: {doc.metadata.get('page', 'N/A')}")
    print(f"내용: {doc.page_content[:300]}...")

# %% [markdown]
# ### 5.1 유사도 점수 포함 검색

# %%
# 유사도 점수와 함께 검색
results_with_scores = vector_store.similarity_search_with_score(query, k=3)

print(f"검색 쿼리: {query}\n")
print("=" * 80)
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n[결과 {i}] 유사도 점수: {score:.4f}")
    print(f"페이지: {doc.metadata.get('page', 'N/A')}")
    print(f"내용: {doc.page_content[:200]}...")

# %% [markdown]
# ## 6. 검색기 (Retriever)
#
# 검색기는 Runnable 인터페이스를 구현하여 표준화된 방식으로 검색을 수행합니다.
# 벡터 스토어의 `as_retriever()` 메서드로 검색기를 생성할 수 있습니다.

# %%
# 검색기 생성
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 상위 3개 결과 반환
)

# 단일 쿼리
results = retriever.invoke(query)
print(f"검색 결과 수: {len(results)}")

# 배치 쿼리 (여러 쿼리 동시 처리)
queries = [
    "2023년 나이키의 매출은 얼마였나요?",
    "나이키의 주요 시장은 어디인가요?"
]

batch_results = retriever.batch(queries)
for i, (q, docs) in enumerate(zip(queries, batch_results), 1):
    print(f"\n[쿼리 {i}]: {q}")
    print(f"결과 수: {len(docs)}")
    if docs:
        print(f"첫 결과: {docs[0].page_content[:150]}...")

# %% [markdown]
# ### 6.1 다양한 검색 전략

# %%
# MMR (Maximum Marginal Relevance) 검색
# 유사도와 다양성을 균형있게 고려
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # lambda_mult: 다양성 비율 (0~1)
)

mmr_results = retriever_mmr.invoke(query)
print("MMR 검색 결과:")
for i, doc in enumerate(mmr_results, 1):
    print(f"\n[{i}] {doc.page_content[:200]}...")

# %% [markdown]
# ## 7. 실전 예제: 검색 엔진 구축

# %%
class SemanticSearchEngine:
    """시맨틱 검색 엔진 클래스"""
    
    def __init__(self, vector_store, retriever):
        self.vector_store = vector_store
        self.retriever = retriever
    
    def search(self, query: str, k: int = 3):
        """검색 수행"""
        results = self.retriever.invoke(query)
        return results[:k]
    
    def search_with_scores(self, query: str, k: int = 3):
        """유사도 점수와 함께 검색"""
        return self.vector_store.similarity_search_with_score(query, k=k)

# 검색 엔진 인스턴스 생성
search_engine = SemanticSearchEngine(vector_store, retriever)

# 검색 테스트
test_queries = [
    "나이키의 재무 성과",
    "나이키의 주요 제품",
    "나이키의 글로벌 전략"
]

print("=" * 80)
print("검색 엔진 테스트")
print("=" * 80)

for query in test_queries:
    print(f"\n\n[검색어]: {query}")
    print("-" * 80)
    results = search_engine.search(query, k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n[결과 {i}]")
        print(f"출처: {doc.metadata.get('source', 'N/A')}")
        print(f"내용: {doc.page_content[:250]}...")

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **문서 로더**: 다양한 소스에서 문서를 로드
# 2. **텍스트 분할**: 검색 정확도를 위해 적절한 크기로 분할
# 3. **임베딩**: 텍스트를 벡터로 변환하여 의미 기반 검색 가능
# 4. **벡터 스토어**: 임베딩된 문서를 저장하고 검색
# 5. **검색기**: 표준화된 인터페이스로 검색 수행
#
# **다음 단계**: [120_RAG_Agent.py](120_RAG_Agent.py)에서 검색 결과를 LLM에 통합하는 RAG 패턴을 학습합니다.

# %%
