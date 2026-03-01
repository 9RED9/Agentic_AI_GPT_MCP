# ---------------------------------------------------------
# MCP 1: MCPServerStdio 클라이언트 (Streamlit)
# ---------------------------------------------------------
# mcp.json에 정의된 로컬 MCP 서버를 subprocess(stdio)로 연결.
# 문서 기능 반영: name, async with, tool_filter, mcp_config, cache_tools_list
# ---------------------------------------------------------

import sys
import asyncio
import json
import streamlit as st
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, create_static_tool_filter
from dotenv import load_dotenv

load_dotenv()

# Windows에서 asyncio subprocess 호환을 위한 이벤트 루프 정책
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ---------- 메인 비동기 로직 ----------
async def run_agent(messages: list[dict]) -> str:
    """async with 컨텍스트 매니저로 MCP 서버 연결 → 에이전트 실행 → 스트리밍 응답."""

    # mcp.json 읽기
    with open("mcp.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    server_config = config["mcpServers"]["mcp_local_server"]

    # ── MCPServerStdio: async with 패턴 ──
    async with MCPServerStdio(
        name="mcp_local_server",                         # name 파라미터
        params={
            "command": server_config["command"],
            "args": server_config.get("args", []),
        },
        cache_tools_list=True,                            # 도구 목록 캐싱
        # ── tool_filter: 허용할 도구만 노출 ──
        tool_filter=create_static_tool_filter(
            allowed_tool_names=[
                "get_price", "add_item", "list_items",
                "list_tables", "get_table_schema", "execute_sql_query",
            ]
        ),
    ) as server:

        # ── Agent 생성 ──
        agent = Agent(
            name="Assistant",
            instructions=(
                "너는 의류 가격 조회·추가와 Chinook DB 쿼리를 도와주는 에이전트야. "
                "get_price, add_item, list_items, list_tables, get_table_schema, "
                "execute_sql_query 도구를 활용해 답해."
            ),
            model="gpt-5-mini",
            mcp_servers=[server],
            # ── mcp_config: 스키마 strict 변환 ──
            mcp_config={"convert_schemas_to_strict": True},
        )

        # ── 스트리밍 응답 ──
        result = Runner.run_streamed(agent, input=messages)
        response_text = ""
        placeholder = st.empty()

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                response_text += event.data.delta or ""
                with placeholder.container():
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    st.toast(f"도구 활용: `{item.raw_item.name}`")

    return response_text


# ---------- Streamlit 앱 ----------
def main():
    st.set_page_config(page_title="MCP 1: Stdio Client", page_icon="🛒")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🛒 의류 & Chinook DB 에이전트 (stdio)")

    # 사이드바 안내
    st.sidebar.title("stdio MCP 설정")
    st.sidebar.info(
        "mcp.json 기반으로 로컬 MCP 서버를 subprocess로 기동합니다.\n\n"
        "**반영 기능:** name, async with, "
        "tool_filter, mcp_config, cache"
    )

    # 기존 대화 내역 렌더
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("질문을 입력하세요")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = asyncio.run(run_agent(st.session_state.chat_history))
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
