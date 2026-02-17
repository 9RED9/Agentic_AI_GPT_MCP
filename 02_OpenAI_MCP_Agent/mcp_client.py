# ---------------------------------------------------------
# 02_OpenAI_MCP_Agent: stdio MCP + mcp.json + Streamlit 클라이언트
# ---------------------------------------------------------
# mcp.json에 정의된 MCP 서버(mcp_local_server)를 subprocess로 자동 기동 후 stdio로 연결.
# 의류·Chinook DB 도구를 사용하는 챗봇 UI.
# ---------------------------------------------------------

import sys
import asyncio
import streamlit as st
import json
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv

load_dotenv()

# Windows에서 asyncio subprocess 호환을 위한 이벤트 루프 정책
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ---------- MCP 서버 연결 (mcp.json 기반 stdio) ----------
async def setup_mcp_servers():
    """mcp.json의 mcpServers 항목을 읽어 MCPServerStdio로 각 서버를 기동·연결."""
    servers = []
    with open("mcp.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    for server_name, server_config in config.get("mcpServers", {}).items():
        # command/args로 서버 프로세스 실행, stdio로 통신
        mcp_server = MCPServerStdio(
            params={
                "command": server_config.get("command"),
                "args": server_config.get("args", []),
            },
            cache_tools_list=True,
        )
        await mcp_server.connect()
        servers.append(mcp_server)
    return servers


# ---------- 에이전트 생성 (MCP 도구 연동) ----------
async def setup_agent():
    """MCP 서버 연결 후, 의류·Chinook DB 도구를 쓰는 OpenAI 에이전트 생성."""
    mcp_servers = await setup_mcp_servers()
    agent = Agent(
        name="Assistant",
        instructions="너는 의류 가격 조회·추가와 Chinook DB 쿼리를 도와주는 에이전트야. get_price, add_item, list_items, list_tables, get_table_schema, execute_sql_query 도구를 활용해 답해.",
        model="gpt-4o-mini",
        mcp_servers=mcp_servers,
    )
    return agent, mcp_servers


# ---------- 사용자 메시지 처리 (스트리밍 응답) ----------
async def process_user_message():
    """현재 대화 기록으로 에이전트 실행, 스트리밍으로 응답 표시 후 히스토리에 추가."""
    agent, mcp_servers = await setup_agent()
    messages = st.session_state.chat_history
    result = Runner.run_streamed(agent, input=messages)

    response_text = ""
    placeholder = st.empty()

    async for event in result.stream_events():
        # 토큰 단위 텍스트 델타: 화면에 누적 표시
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            response_text += event.data.delta or ""
            with placeholder.container():
                with st.chat_message("assistant"):
                    st.markdown(response_text)
        # 도구 호출 시 토스트로 알림
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                tool_name = item.raw_item.name
                st.toast(f"도구 활용: `{tool_name}`")

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response_text}
    )
    for server in mcp_servers:
        await server.__aexit__(None, None, None)


# ---------- Streamlit 앱 진입점 ----------
def main():
    st.set_page_config(page_title="의류/Chinook MCP", page_icon="🛒")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🛒 의류 & Chinook DB 에이전트")
    st.caption("의류 가격·재고와 Chinook DB 질문을 해보세요. (stdio MCP)")

    # 기존 대화 내역 렌더
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("질문을 입력하세요")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        asyncio.run(process_user_message())


if __name__ == "__main__":
    main()
