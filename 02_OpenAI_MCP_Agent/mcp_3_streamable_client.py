# ---------------------------------------------------------
# MCP 3: MCPServerStreamableHttp + MCPServerManager (Streamlit)
# ---------------------------------------------------------
# HTTP 기반 원격 MCP 서버에 연결하고, MCPServerManager로 다중 서버를
# 통합 관리. 연결 실패 처리, 서버 상태 모니터링 포함.
# 문서 기능 반영: MCPServerStreamableHttp, MCPServerManager, mcp_config
# ---------------------------------------------------------
# 실행 전 HTTP MCP 서버가 필요합니다:
#   python mcp_local_server.py --http
# ---------------------------------------------------------

import sys
import asyncio
import streamlit as st
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp, MCPServerManager
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# ---------- 메인 비동기 로직 ----------
async def run_agent(
    messages: list[dict],
    server_urls: list[str],
) -> tuple[str, int, int]:
    """MCPServerStreamableHttp + MCPServerManager로 다중 서버 연결 후 에이전트 실행."""

    # ── 서버 인스턴스 생성 ──
    servers = []
    for i, url in enumerate(server_urls):
        server = MCPServerStreamableHttp(
            name=f"MCP Server {i + 1}",
            params={"url": url},
            cache_tools_list=True,
        )
        servers.append(server)

    # ── MCPServerManager: 다중 서버 통합 관리 ──
    async with MCPServerManager(servers) as manager:
        active_count = len(manager.active_servers)
        failed_count = len(manager.failed_servers)

        if active_count == 0:
            return "연결된 MCP 서버가 없습니다. 서버 URL을 확인해주세요.", 0, failed_count

        # ── Agent 생성 ──
        agent = Agent(
            name="Multi-Server Assistant",
            instructions=(
                "너는 여러 MCP 서버의 도구를 활용하는 에이전트야. "
                "연결된 서버들의 도구를 사용해서 사용자 질문에 답해줘. "
                "한국어로 답변해."
            ),
            model="gpt-4o-mini",
            mcp_servers=manager.active_servers,
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

    return response_text, active_count, failed_count


# ---------- Streamlit 앱 ----------
def main():
    st.set_page_config(page_title="MCP 3: Streamable HTTP", page_icon="🔗")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🔗 Multi-Server MCP 에이전트 (Streamable HTTP)")

    # ── 사이드바: 서버 URL 입력 ──
    st.sidebar.title("Streamable HTTP 설정")
    st.sidebar.info(
        "HTTP MCP 서버 URL을 입력하세요.\n"
        "서버가 실행 중이어야 합니다.\n\n"
        "**반영 기능:**\n"
        "- MCPServerStreamableHttp\n"
        "- MCPServerManager (active/failed)\n"
        "- mcp_config"
    )

    server_url_1 = st.sidebar.text_input(
        "서버 1 URL",
        value="http://localhost:8000/mcp",
        help="첫 번째 MCP 서버 URL",
    )
    server_url_2 = st.sidebar.text_input(
        "서버 2 URL (선택)",
        value="",
        help="두 번째 MCP 서버 URL (비워두면 서버 1만 사용)",
    )

    # 입력된 URL 목록
    server_urls = [url.strip() for url in [server_url_1, server_url_2] if url.strip()]

    # ── 서버 상태 표시 영역 ──
    if "last_active" in st.session_state:
        st.sidebar.metric("활성 서버", st.session_state.last_active)
    if "last_failed" in st.session_state:
        st.sidebar.metric("실패 서버", st.session_state.last_failed)

    # 기존 대화 내역 렌더
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("질문을 입력하세요")
    if user_input:
        if not server_urls:
            st.error("서버 URL을 최소 1개 입력해주세요.")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response, active, failed = asyncio.run(
            run_agent(st.session_state.chat_history, server_urls)
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        st.session_state.last_active = active
        st.session_state.last_failed = failed


if __name__ == "__main__":
    main()
