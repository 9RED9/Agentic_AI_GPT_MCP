"""
400_DB_MCP_Agent Streamlit 앱: Chinook MCP + Streamlit UI.
agent_client.py의 Streamlit 버전. LangChain v1 (create_agent + MemorySaver).
"""
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Chinook MCP 서버 스크립트 경로 (이 폴더의 agent_server.py)
_SERVER_SCRIPT = Path(__file__).resolve().parent / "agent_server.py"


async def run_agent_once(user_message: str) -> str:
    """Chinook MCP 서버에 연결해 에이전트 단회 실행 후 최종 응답 텍스트 반환."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain.chat_models import init_chat_model
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage
    except ImportError as e:
        return f"의존성 오류: {e}. pip install langchain-mcp-adapters langgraph langchain langchain-openai"

    # 서버 스크립트 및 Chinook.db 존재 여부 확인
    if not _SERVER_SCRIPT.exists():
        return f"Chinook MCP 서버를 찾을 수 없습니다: {_SERVER_SCRIPT}. 이 폴더에 agent_server.py와 Chinook.db가 있는지 확인하세요."

    # LLM 및 MCP 클라이언트 초기화
    model = init_chat_model("gpt-5-mini", model_provider="openai")
    client = MultiServerMCPClient({
        "chinook": {
            "transport": "stdio",
            "command": "python",
            "args": [str(_SERVER_SCRIPT)],
        }
    })
    tools = await client.get_tools()
    memory = MemorySaver()
    agent = create_agent(model=model, tools=tools, checkpointer=memory)

    # 에이전트 단회 실행 (대화 세션 ID 고정)
    config = {"configurable": {"thread_id": "streamlit_session"}}
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config
    )
    # 마지막 메시지에서 텍스트 추출 (content 없을 수 있음 대비)
    messages = response.get("messages", [])
    last = messages[-1] if messages else None
    return getattr(last, "content", str(last)) or "(응답 없음)"


def run_sync(user_message: str) -> str:
    """동기 래퍼: Streamlit에서 asyncio.run으로 비동기 에이전트 호출."""
    return asyncio.run(run_agent_once(user_message))


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Chinook MCP Lab", layout="centered")
st.title("Chinook MCP Streamlit")
st.caption("Chinook DB MCP + LangChain v1 Agent")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 내역 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 새 입력 시 에이전트 호출 후 응답 표시
if prompt := st.chat_input("Chinook DB에 대해 질문하세요 (예: 고객 수는 몇 명인가요?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("응답 생성 중..."):
            reply = run_sync(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
