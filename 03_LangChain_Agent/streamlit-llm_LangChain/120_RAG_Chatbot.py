#---------------------------------------------------------
# Streamlit RAG 챗봇 구현
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_chat import message

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="RAG 챗봇", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>RAG 챗봇</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------------
if "llm" not in st.session_state:
    st.session_state.llm = init_chat_model("gpt-5-nano", model_provider="openai")

if "embeddings" not in st.session_state:
    # 임베딩 모델 초기화
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if "vector_store" not in st.session_state:
    # 인메모리 벡터 스토어 생성
    st.session_state.vector_store = InMemoryVectorStore(st.session_state.embeddings)

# 모듈 레벨 참조 — @tool 함수에서 st.session_state 접근이 불안정하므로 직접 참조 사용
_vector_store = st.session_state.vector_store

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ---------------------------------------------------------------------------------
# 사이드바: 문서 인덱싱
# ---------------------------------------------------------------------------------
st.sidebar.title("📚 문서 관리")

# URL 입력으로 문서 로드
url = st.sidebar.text_input("웹 문서 URL:", placeholder="https://botpress.com/ko/blog/llm-agents")

if st.sidebar.button("문서 인덱싱"):
    if url:
        with st.spinner("문서를 로드하고 인덱싱하는 중..."):
            # 웹 문서 로드 및 청크 분할 인덱싱
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,      # 각 청크의 최대 문자 수
                    chunk_overlap=200     # 청크 간 겹치는 문자 수
                )
                splits = text_splitter.split_documents(docs)
                
                st.session_state.vector_store.add_documents(splits)
                st.session_state.indexed = True
                st.session_state.document_count = len(splits)
                
                st.sidebar.success(f"✅ {len(splits)}개의 청크가 인덱싱되었습니다!")
            except Exception as e:
                st.sidebar.error(f"오류: {str(e)}")
    else:
        st.sidebar.warning("URL을 입력하세요.")

# 인덱싱 상태
if st.session_state.indexed:
    st.sidebar.info(f"📊 인덱싱된 청크: {st.session_state.get('document_count', 0)}개")

# ---------------------------------------------------------------------------------
# 검색 도구 정의
# ---------------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """질문에 답하기 위해 관련 정보를 검색합니다.
    
    이 도구는 벡터 스토어에서 질문과 관련된 문서를 검색합니다.
    검색된 문서는 답변 생성에 사용됩니다.
    
    Args:
        query: 검색할 질문 또는 키워드
    """
    # 벡터 스토어에서 유사한 문서 검색 (모듈 레벨 참조 사용)
    retrieved_docs = _vector_store.similarity_search(query, k=2)
    if not retrieved_docs:
        return "관련 문서를 찾지 못했습니다. 사이드바에서 문서를 먼저 인덱싱하세요.", []
    
    # 검색된 문서를 문자열로 직렬화
    serialized = "\n\n".join(
        f"출처: {doc.metadata.get('source', 'N/A')}\n내용: {doc.page_content}"
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs

# ---------------------------------------------------------------------------------
# 에이전트 초기화
# ---------------------------------------------------------------------------------
if st.session_state.agent is None:
    system_prompt = (
        "당신은 블로그 게시글에서 관련 문맥(context)을 검색하는 도구에 접근할 수 있습니다. "
        "사용자의 질문에 답하기 위해 먼저 retrieve_context 도구를 사용하여 관련 정보를 검색한 후, "
        "검색된 정보를 바탕으로 정확하고 유용한 답변을 제공하세요."
    )
    
    st.session_state.agent = create_agent(
        st.session_state.llm,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )

# ---------------------------------------------------------------------------------
# 사이드바 버튼
# ---------------------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")
refresh_button = st.sidebar.button("대화 내용 초기화")

if refresh_button:
    st.session_state.messages = []
    st.experimental_rerun()

# ---------------------------------------------------------------------------------
# 메인 영역: 채팅
# ---------------------------------------------------------------------------------
if not st.session_state.indexed:
    st.warning("⚠️ 먼저 사이드바에서 문서를 인덱싱하세요.")

# 입력 폼
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_area("질문을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

# 폼 밖에서 처리 (폼 안에서 spinner 사용 시 렌더링 중복 발생)
if submit_button and user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    # 에이전트 호출 및 응답 생성
    try:
        with st.spinner("답변 생성 중..."):
            result = st.session_state.agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            })

            ai_response = result["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=ai_response))
    except Exception as e:
        error_msg = f"에러가 발생했습니다: {str(e)}"
        st.session_state.messages.append(AIMessage(content=error_msg))
        st.error(error_msg)

# 마지막 응답 표시
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if isinstance(last_msg, AIMessage):
        st.text_area("최신 답변", value=last_msg.content, height=200, disabled=True)

# 대화 이력 표시
st.subheader("대화 이력")
# 대화 이력을 순서대로 표시
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=f"user_{idx}")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=f"ai_{idx}")
