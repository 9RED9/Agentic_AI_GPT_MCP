#---------------------------------------------------------
# Streamlit SQL 챗봇 구현
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from streamlit_chat import message
from uuid import uuid4

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="SQL 챗봇", page_icon=":database:")
st.markdown("<h1 style='text-align: center;'>SQL 데이터베이스 챗봇</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------------
if "llm" not in st.session_state:
    st.session_state.llm = init_chat_model("gpt-5-nano", model_provider="openai")

if "db" not in st.session_state:
    # SQLite 데이터베이스 연결 및 툴킷 초기화
    try:
        st.session_state.db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        st.session_state.toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=st.session_state.llm)
        st.session_state.tools = st.session_state.toolkit.get_tools()
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {str(e)}")
        st.session_state.db = None

if "system_prompt" not in st.session_state and st.session_state.get("db"):
    st.session_state.system_prompt = """
당신은 SQL 데이터베이스와 상호작용하도록 설계된 에이전트입니다.
사용자로부터 질문을 입력받으면, 실행 가능한 {dialect} 구문으로
문법적으로 올바른 SQL 쿼리를 생성하세요.

그 후 쿼리를 실행한 결과를 확인하고, 그에 기반하여 답변을 반환하세요.
항상 최대 5개의 결과만 반환하도록 쿼리를 제한해야 합니다.

데이터베이스에 DML 문(INSERT, UPDATE, DELETE, DROP 등)을
절대 실행하지 마세요.
""".format(dialect=st.session_state.db.dialect)

if "agent" not in st.session_state:
    if st.session_state.db:
        st.session_state.agent = create_agent(
            st.session_state.llm,
            st.session_state.tools,
            system_prompt=st.session_state.system_prompt,
            checkpointer=InMemorySaver()
        )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "human_review" not in st.session_state:
    st.session_state.human_review = False

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"sql-chat-{uuid4()}"

# ---------------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")

hitl_enabled = st.sidebar.checkbox("Human-in-the-Loop 활성화")

# 체크박스 상태가 변경되었을 때만 에이전트 재생성 (InMemorySaver 상태 보존)
if hitl_enabled != st.session_state.human_review:
    st.session_state.human_review = hitl_enabled
    st.session_state.pending_approval = None
    st.session_state.thread_id = f"sql-chat-{uuid4()}"
    if st.session_state.db:
        if hitl_enabled:
            st.session_state.agent = create_agent(
                st.session_state.llm,
                st.session_state.tools,
                system_prompt=st.session_state.system_prompt,
                middleware=[
                    HumanInTheLoopMiddleware(
                        interrupt_on={"sql_db_query": True},
                        description_prefix="SQL 쿼리 실행 승인 대기 중",
                    ),
                ],
                checkpointer=InMemorySaver(),
            )
        else:
            st.session_state.agent = create_agent(
                st.session_state.llm,
                st.session_state.tools,
                system_prompt=st.session_state.system_prompt,
                checkpointer=InMemorySaver(),
            )

refresh_button = st.sidebar.button("대화 내용 초기화")

if refresh_button:
    st.session_state.messages = []
    st.session_state.pending_approval = None
    st.session_state.thread_id = f"sql-chat-{uuid4()}"
    st.rerun()

# 데이터베이스 정보
if st.session_state.db:
    st.sidebar.subheader("데이터베이스 정보")
    st.sidebar.write(f"방언: {st.session_state.db.dialect}")
    st.sidebar.write(f"테이블 수: {len(st.session_state.db.get_usable_table_names())}")

# ---------------------------------------------------------------------------------
# 메인 영역: 채팅
# ---------------------------------------------------------------------------------
if not st.session_state.db:
    st.error("데이터베이스에 연결할 수 없습니다. Chinook.db 파일이 있는지 확인하세요.")
else:
    # 대화 설정 (thread_id로 대화 식별)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # ----- 승인 대기 UI -----
    if st.session_state.pending_approval:
        st.warning("⚠️ SQL 쿼리 실행 승인 대기 중입니다.")

        # 인터럽트에서 SQL 쿼리 정보 추출하여 표시
        for interrupt in st.session_state.pending_approval:
            # 각 인터럽트에서 도구 실행 요청 목록 순회
            for request in interrupt.value.get("action_requests", []):
                tool_name = request.get("tool", "N/A")
                args = request.get("args", {})
                query = args.get("query", str(args))
                st.code(query, language="sql")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ 승인", type="primary"):
                with st.spinner("쿼리 실행 중..."):
                    # 승인 결정 전송
                    decision = Command(resume={"decisions": [{"type": "approve"}]})
                    result = st.session_state.agent.invoke(decision, config=config)
                    ai_response = result["messages"][-1].content
                    st.session_state.messages.append(AIMessage(content=ai_response))
                    st.session_state.pending_approval = None
                    st.rerun()

        with col2:
            if st.button("❌ 거부"):
                # 거부 결정 전송
                decision = Command(resume={"decisions": [{"type": "reject"}]})
                result = st.session_state.agent.invoke(decision, config=config)
                ai_response = result["messages"][-1].content
                st.session_state.messages.append(AIMessage(content=ai_response))
                st.session_state.pending_approval = None
                st.rerun()

    # ----- 입력 폼 -----
    with st.form(key='sql_chat_form', clear_on_submit=True):
        user_input = st.text_area("SQL 질문을 입력하세요:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))

            # 에이전트 실행 및 인터럽트 여부 확인
            try:
                with st.spinner("답변 생성 중..."):
                    result = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=config,
                    )

                    # 인터럽트 확인
                    interrupts = result.get("__interrupt__", [])
                    if interrupts:
                        st.session_state.pending_approval = interrupts
                        st.rerun()
                    else:
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
            st.subheader("최신 답변")
            st.markdown(last_msg.content)

    # 대화 이력 표시
    st.subheader("대화 이력")
    # 대화 이력을 순서대로 표시
    for idx, msg in enumerate(st.session_state.messages):
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=f"user_{idx}")
        elif isinstance(msg, AIMessage):
            message(msg.content, is_user=False, key=f"ai_{idx}")
