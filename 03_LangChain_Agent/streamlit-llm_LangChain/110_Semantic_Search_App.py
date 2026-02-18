#---------------------------------------------------------
# Streamlit 시맨틱 검색 엔진 앱
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="시맨틱 검색 엔진", page_icon=":mag:")
st.markdown("<h1 style='text-align: center;'>시맨틱 검색 엔진</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 사이드바: 문서 업로드 및 인덱싱
# ---------------------------------------------------------------------------------
st.sidebar.title("📚 문서 관리")

# 임베딩 초기화
if "embeddings" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 벡터 스토어 초기화
if "vector_store" not in st.session_state:
    st.session_state.vector_store = InMemoryVectorStore(st.session_state.embeddings)
    st.session_state.indexed = False

# 문서 업로드
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("문서 인덱싱"):
        with st.spinner("문서를 로드하고 인덱싱하는 중..."):
            # 임시 파일로 저장
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # PDF 로드
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # 텍스트 분할
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)
                
                # 벡터 스토어에 추가
                st.session_state.vector_store.add_documents(splits)
                st.session_state.indexed = True
                st.session_state.document_count = len(splits)
                
                st.sidebar.success(f"✅ {len(splits)}개의 청크가 인덱싱되었습니다!")
            finally:
                os.unlink(tmp_path)

# 기존 인덱스 사용
if st.sidebar.button("기본 문서 로드 (예제)"):
    with st.spinner("예제 문서를 로드하는 중..."):
        try:
            loader = PyPDFLoader("./example_data/nke-10k-2023_korean.pdf")
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            
            st.session_state.vector_store.add_documents(splits)
            st.session_state.indexed = True
            st.session_state.document_count = len(splits)
            
            st.sidebar.success(f"✅ {len(splits)}개의 청크가 인덱싱되었습니다!")
        except Exception as e:
            st.sidebar.error(f"오류: {str(e)}")

# 인덱싱 상태 표시
if st.session_state.get("indexed", False):
    st.sidebar.info(f"📊 인덱싱된 청크: {st.session_state.get('document_count', 0)}개")

# ---------------------------------------------------------------------------------
# 메인 영역: 검색
# ---------------------------------------------------------------------------------
if not st.session_state.get("indexed", False):
    st.info("👈 사이드바에서 문서를 업로드하거나 예제 문서를 로드하세요.")
else:
    # 검색 옵션
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("검색어를 입력하세요:", placeholder="예: 나이키의 매출은?")
    
    with col2:
        k_results = st.number_input("결과 수", min_value=1, max_value=10, value=3)
    
    # 검색 실행
    if st.button("🔍 검색", type="primary") and search_query:
        with st.spinner("검색 중..."):
            try:
                # 유사도 검색
                results = st.session_state.vector_store.similarity_search_with_score(
                    search_query, 
                    k=k_results
                )
                
                # 결과 표시
                st.subheader(f"검색 결과 ({len(results)}개)")
                
                for i, (doc, score) in enumerate(results, 1):
                    with st.expander(f"결과 {i} (유사도: {score:.4f})", expanded=(i==1)):
                        st.write(f"**출처:** {doc.metadata.get('source', 'N/A')}")
                        st.write(f"**페이지:** {doc.metadata.get('page', 'N/A')}")
                        st.write("---")
                        st.write(doc.page_content)
                
            except Exception as e:
                st.error(f"검색 오류: {str(e)}")
    
    # 검색 히스토리
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    if search_query and st.button("검색 실행"):
        st.session_state.search_history.insert(0, search_query)
        if len(st.session_state.search_history) > 10:
            st.session_state.search_history.pop()
    
    # 최근 검색어
    if st.session_state.search_history:
        st.sidebar.subheader("최근 검색어")
        for query in st.session_state.search_history[:5]:
            if st.sidebar.button(query, key=f"history_{query}"):
                st.experimental_rerun()
