# ---------------------------------------------------------
# Streamlit 통합 MCP ReAct Agent 챗봇 (060 UI 스타일) - v2
#  - Chinook DB MCP 서버 + Notion MCP 서버 통합
#  - 제출 시마다 MultiServerMCPClient로 도구 수집 후 단회 호출
# ---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import os
import asyncio

import streamlit as st
from streamlit_chat import message

# ✅ 변경: 새로운 MCP 클라이언트
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------
# LLM 초기화 (LangChain 표준 init_chat_model)
# ---------------------------------------------------------
llm = init_chat_model("gpt-5-mini", model_provider="openai")

# ---------------------------------------------------------
# Streamlit 페이지 및 헤더
# ---------------------------------------------------------
st.set_page_config(page_title="통합 MCP Agent Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>통합 MCP Agent 챗봇</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 사이드바 버튼
# ---------------------------------------------------------
st.sidebar.title("😎")
refresh_button = st.sidebar.button("대화 내용 초기화")
summaries_button = st.sidebar.button("대화 내용 요약")

# ---------------------------------------------------------
# Session State 초기화
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="당신은 유용한 도우미입니다.")]

# ---------------------------------------------------------
# 환경 변수 점검 (Notion 관련)
# ---------------------------------------------------------
notion_api_key = os.getenv("NOTION_API_KEY", "")
default_page_id = os.getenv("NOTION_PAGE_ID", "")

if not notion_api_key:
    st.sidebar.warning(".env에 NOTION_API_KEY가 설정되어 있지 않습니다.")
if not default_page_id:
    st.sidebar.info(".env에 NOTION_PAGE_ID가 없으면 페이지 작업 시 ID를 직접 입력해야 합니다.")

# ---------------------------------------------------------
# MCP Agent 실행 (제출 시 일회 호출)
#   - MultiServerMCPClient로 두 MCP 서버(chinook/notion) 연결
#   - 모든 도구를 수집(get_tools) 후 에이전트 단회 실행
# ---------------------------------------------------------
async def run_agent_with_mcp(messages: list[object]) -> str:
    """
    주어진 메시지 히스토리(messages)를 입력으로 받아
    MCP 서버(Chinook DB + Notion)와 통합된 ReAct Agent를 실행하여
    최종 AI 응답 텍스트(content)를 반환합니다.
    """
    # ───────────────────────────────────────────────
    # ① 서버 스크립트 경로 설정
    #   현재 파일 경로 기준으로 Chinook MCP 서버 위치 계산
    # ───────────────────────────────────────────────
    streamlit_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(streamlit_dir)
    chinook_agent_path = os.path.abspath(
        os.path.join(project_root, "DB_MCP_Agent", "agent_server.py")
    )

    # ───────────────────────────────────────────────
    # ② MultiServerMCPClient 선언 (두 MCP 서버 통합)
    #   • Chinook: stdio (로컬 파이썬 서브프로세스)
    #   • Notion : stdio (npx로 공식 MCP 서버 실행)
    # ───────────────────────────────────────────────
    client = MultiServerMCPClient(
        {
            "chinook": {
                "transport": "stdio",
                "command": "python",
                "args": [chinook_agent_path],
            },
            "notion": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@notionhq/notion-mcp-server"],
                "env": {"NOTION_TOKEN": notion_api_key},
            },
            # 필요 시 HTTP 기반 원격 MCP도 함께 붙일 수 있습니다:
            # "weather": {
            #     "transport": "streamable_http",
            #     "url": "http://localhost:8000/mcp"
            # }
        }
    )

    # ───────────────────────────────────────────────
    # ③ 사용자 입력 메시지에 "기본 Notion 페이지 ID" 안내문 추가
    # ───────────────────────────────────────────────
    enhanced_messages: list[object] = []
    last_human_index = -1
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_index = idx

    for idx, msg in enumerate(messages):
        if idx == last_human_index and isinstance(msg, HumanMessage) and default_page_id:
            enhanced_input = (
                f"{msg.content}\n\n[참고: 기본 작업 페이지 ID는 {default_page_id}입니다. "
                f"페이지 생성/업데이트 시 이 ID를 기본 부모로 사용하세요.]"
            )
            enhanced_messages.append(HumanMessage(content=enhanced_input))
        else:
            enhanced_messages.append(msg)

    # ───────────────────────────────────────────────
    # ④ 모든 서버에서 MCP 도구 수집 → 에이전트 생성 → 단회 실행
    # ───────────────────────────────────────────────
    tools = await client.get_tools()                 # ✅ 핵심 변경: 세션 수동 연결 대신 일괄 수집
    agent = create_agent(llm, tools)                 # LangChain ReAct Agent 생성
    response = await agent.ainvoke({"messages": enhanced_messages})
    return response["messages"][-1].content          # 최종 AI 응답 텍스트

# ---------------------------------------------------------
# 사이드바 버튼 동작
# ---------------------------------------------------------
if refresh_button:
    st.session_state.messages = [SystemMessage(content="당신은 유용한 도우미입니다.")]

if summaries_button:
    conversation_text = []
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            role = "System"
        elif isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "AI"
        else:
            role = "Unknown"
        conversation_text.append(f"{role}: {msg.content}")
    joined_conversation = "\n".join(conversation_text)

    prompt_content = f"""다음 대화를 요약해주세요:\n{joined_conversation}\n--- \n요약:\n"""
    summary_response = llm.invoke([HumanMessage(content=prompt_content)])
    st.sidebar.write("**대화 요약:**")
    st.sidebar.write(summary_response.content)

# ---------------------------------------------------------
# 메인 입력 폼 및 Agent 호출
# ---------------------------------------------------------
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_area("질문을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        try:
            # Streamlit은 동기 컨텍스트이므로, 호출 시마다 이벤트루프 생성
            ai_text = asyncio.run(run_agent_with_mcp(st.session_state.messages))
            st.session_state.messages.append(AIMessage(content=ai_text))
        except Exception as e:
            st.session_state.messages.append(AIMessage(content=f"에러가 발생했습니다: {e}"))

# ---------------------------------------------------------
# 마지막 AIMessage를 입력 폼 바로 아래에 표시
# ---------------------------------------------------------
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]
    if isinstance(last_msg, AIMessage):
        st.text(last_msg.content)

# ---------------------------------------------------------
# 이전 대화 이력 표시 (060 스타일)
# ---------------------------------------------------------
st.subheader("이전 대화 이력")
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=str(idx) + "_user")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=str(idx) + "_ai")