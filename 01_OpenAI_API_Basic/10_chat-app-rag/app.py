"""
여행·지식 베이스 채팅 앱 (OpenAI Agents SDK + Streamlit)
- 09_Chatbot_RAG_Agent 노트북 4단계와 동일한 에이전트: 날씨(get_weather), 웹 검색(WebSearchTool), RAG(search_knowledge_base).
- 답변 생성은 OpenAI만 사용합니다. Streamlit으로 대화 화면을 제공하며, 대화 기록은 세션에 유지됩니다.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, function_tool, WebSearchTool
from rag import get_retrieved_context

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")


# 노트북과 동일: 날씨 도구 (open-meteo API)
@function_tool
def get_weather(위도: float, 경도: float) -> str:
    """주어진 위도와 경도에서의 현재 기온(섭씨)을 반환합니다."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={위도}&longitude={경도}&current=temperature_2m"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    temp = data.get("current", {}).get("temperature_2m")
    return f"{temp}°C" if temp is not None else "Unknown"


# 에이전트가 호출하는 RAG 검색 도구 (FAQ/문서 유사도 검색)
@function_tool
def search_knowledge_base(query: str) -> str:
    """사용자 질문에 맞는 FAQ/문서 맥락을 검색합니다.
    계정, 비밀번호, 환불, 고객지원, 보안, 회사 정책 관련 질문일 때 이 도구를 사용하세요."""
    return get_retrieved_context(query)


# 에이전트 생성 후 대화 메시지로 실행하고 최종 답변 문자열 반환 
def _run_agent(messages: list) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 또는 환경 변수를 설정하세요.")

    # 노트북 4단계와 동일: 여행 + 날씨 + 웹검색 + RAG
    agent = Agent(
        name="여행·지식 베이스 에이전트",
        instructions=(
            "당신은 여행 계획과 회사 FAQ를 도와주는 에이전트입니다. "
            "날씨가 필요하면 get_weather, 최신 정보는 WebSearchTool, "
            "계정·환불·정책 등은 search_knowledge_base로 지식 베이스를 검색한 뒤 답하세요. "
            "답은 간결하게 하세요."
        ),
        model=OPENAI_MODEL,
        tools=[get_weather, WebSearchTool(), search_knowledge_base],
    )
    result = Runner.run_sync(agent, input=messages)
    return str(result.final_output or "응답이 생성되지 않았습니다.")


# 세션 초기화: 대화 기록, 에러 메시지
if "messages" not in st.session_state:
    st.session_state.messages = []
if "error" not in st.session_state:
    st.session_state.error = None

st.set_page_config(page_title="OpenAI Agent AI 채팅", layout="centered")

st.title("OpenAI Agent AI 채팅")
st.caption("여행·지식 베이스 에이전트 (09 노트북과 동일 도구: 날씨, 웹검색, RAG)")

# 기존 대화 내역 표시 (사용자 오른쪽, 어시스턴트 왼쪽)
for msg in st.session_state.messages:
    left_col, right_col = st.columns(2)
    if msg["role"] == "user":
        with right_col:
            with st.chat_message("user"):
                st.markdown(msg["content"])
    else:
        with left_col:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# 이전 요청에서 발생한 에러가 있으면 표시 후 초기화
if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

# 새 메시지 입력 시: 사용자 메시지 추가 후 에이전트 호출해 답변 표시
if prompt := st.chat_input("메시지를 입력하세요..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    left_col, right_col = st.columns(2)
    with right_col:
        with st.chat_message("user"):
            st.markdown(prompt)

    left_col2, right_col2 = st.columns(2)
    with left_col2:
        with st.chat_message("assistant"):
            with st.spinner("봇이 입력 중..."):
                try:
                    # 세션의 전체 대화를 에이전트에 넘겨 실행
                    messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    reply = _run_agent(messages)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    st.markdown(reply)
                except Exception as e:
                    # API 키 미설정 등 예외 시 세션에 저장 후 화면에 표시
                    st.session_state.error = str(e)
                    st.error(str(e))
