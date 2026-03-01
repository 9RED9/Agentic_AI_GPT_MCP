# ---------------------------------------------------------
# MCP 2: HostedMCPTool 클라이언트 (Streamlit)
# ---------------------------------------------------------
# OpenAI Responses API가 공개 MCP 서버를 직접 호출하는 방식.
# 로컬 subprocess 없이 원격 서버의 도구를 사용.
# 문서 기능 반영: HostedMCPTool
# ---------------------------------------------------------

import sys
import asyncio
import streamlit as st
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, HostedMCPTool
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ---------- 공개 MCP 서버 목록 ----------
MCP_SERVERS = {
    "GitMCP - OpenAI Codex": "https://gitmcp.io/openai/codex",
    "GitMCP - LangChain": "https://gitmcp.io/langchain-ai/langchain",
    "GitMCP - Agentic AI GPT MCP": "https://gitmcp.io/ironmanciti/Agentic_AI_GPT_MCP",
    "직접 입력": "",
}


# ---------- URL 검증 ----------
def validate_mcp_url(url: str) -> str | None:
    """MCP 서버 URL 형식을 검증. 문제가 있으면 에러 메시지 반환."""
    if not url.strip():
        return "URL을 입력해주세요."
    if "github.com" in url:
        return (
            "github.com URL은 사용할 수 없습니다.\n\n"
            "GitMCP를 사용하려면 `gitmcp.io` 도메인으로 변환하세요:\n"
            "- ❌ `https://github.com/owner/repo`\n"
            "- ✅ `https://gitmcp.io/owner/repo`"
        )
    if not url.startswith("https://"):
        return "URL은 `https://`로 시작해야 합니다."
    return None


# ---------- 메인 비동기 로직 ----------
async def run_agent(messages: list[dict], server_url: str) -> str:
    """HostedMCPTool로 원격 MCP 서버 도구를 사용하는 에이전트 실행."""

    # ── HostedMCPTool 생성 ──
    hosted_tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": server_url,
            "require_approval": "never",
        },
    )

    # ── Agent 생성 (tools에 전달, mcp_servers가 아님) ──
    agent = Agent(
        name="Hosted MCP Assistant",
        instructions=(
            "너는 GitHub 저장소의 코드를 검색하고 분석하는 에이전트야. "
            "MCP 서버가 제공하는 도구들을 활용해서 코드를 읽고 질문에 답해줘. "
            "한국어로 답변해."
        ),
        model="gpt-5-mini",
        tools=[hosted_tool],
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
    st.set_page_config(page_title="MCP 2: Hosted Client", page_icon="🌐")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "prev_server" not in st.session_state:
        st.session_state.prev_server = ""

    st.title("🌐 Hosted MCP 에이전트")
    st.caption("HostedMCPTool · OpenAI가 직접 원격 MCP 서버 호출")

    # ── 사이드바: 서버 선택 ──
    st.sidebar.title("Hosted MCP 설정")

    selected = st.sidebar.selectbox("공개 MCP 서버 선택", list(MCP_SERVERS.keys()))

    if selected == "직접 입력":
        server_url = st.sidebar.text_input(
            "MCP 서버 URL",
            placeholder="https://gitmcp.io/owner/repo",
        )
    else:
        server_url = MCP_SERVERS[selected]
        st.sidebar.code(server_url, language=None)

    # 서버가 변경되면 대화 자동 초기화
    if server_url and server_url != st.session_state.prev_server:
        st.session_state.prev_server = server_url
        if st.session_state.chat_history:
            st.session_state.chat_history = []
            st.rerun()

    st.sidebar.info(
        "**HostedMCPTool 특징:**\n"
        "- OpenAI가 직접 원격 서버 호출\n"
        "- 로컬 subprocess 불필요\n"
        "- `tools=`에 전달 (`mcp_servers=` 아님)\n\n"
        "**GitMCP URL 형식:**\n"
        "- `https://gitmcp.io/{owner}/{repo}`"
    )

    # 기존 대화 내역 렌더
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("질문을 입력하세요 (예: 이 저장소의 주요 기능은?)")
    if user_input:
        # URL 검증
        error = validate_mcp_url(server_url)
        if error:
            st.error(error)
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            response = asyncio.run(
                run_agent(st.session_state.chat_history, server_url)
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )
        except Exception as e:
            error_msg = str(e)
            if "424" in error_msg:
                st.error(
                    f"MCP 서버 연결 실패 (424 Failed Dependency)\n\n"
                    f"서버 URL을 확인해주세요: `{server_url}`"
                )
            else:
                st.error(f"에러 발생: {error_msg}")
            st.session_state.chat_history.pop()


if __name__ == "__main__":
    main()
