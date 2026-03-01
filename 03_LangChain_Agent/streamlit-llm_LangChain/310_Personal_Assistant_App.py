#---------------------------------------------------------
# Streamlit 개인 비서 앱 (Subagents Pattern)
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from streamlit_chat import message

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="개인 비서", page_icon=":briefcase:")
st.markdown("<h1 style='text-align: center;'>개인 비서 AI</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 도구 정의
# ---------------------------------------------------------------------------------
@tool
def create_calendar_event(title: str, start_time: str, end_time: str, attendees: list[str], location: str = "") -> str:
    """캘린더 이벤트를 생성합니다."""
    return f"이벤트 생성 완료: {title}, {start_time} ~ {end_time}, 참석자 {len(attendees)}명"

@tool
def send_email(to: list[str], subject: str, body: str, cc: list[str] = []) -> str:
    """이메일을 전송합니다."""
    return f"이메일 전송 완료 → 수신자: {', '.join(to)} | 제목: {subject}"

# ---------------------------------------------------------------------------------
# 에이전트 생성 (일반 변수 — session_state가 아닌 직접 참조)
# ---------------------------------------------------------------------------------
llm = init_chat_model("gpt-5-nano", model_provider="openai")

# 캘린더 에이전트
calendar_agent = create_agent(
    llm,
    tools=[create_calendar_event],
    system_prompt="당신은 캘린더 일정 관리 도우미입니다."
)

# 이메일 에이전트
email_agent = create_agent(
    llm,
    tools=[send_email],
    system_prompt="당신은 이메일 작성 도우미입니다."
)

# 하위 에이전트를 도구로 래핑
@tool
def schedule_event(request: str) -> str:
    """자연어 요청을 기반으로 캘린더 이벤트를 예약합니다."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

@tool
def manage_email(request: str) -> str:
    """자연어 요청을 기반으로 이메일을 전송합니다."""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

# 슈퍼바이저 에이전트
supervisor = create_agent(
    llm,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "당신은 친절한 개인 비서입니다. "
        "캘린더 일정을 등록하고 이메일을 보낼 수 있습니다. "
        "사용자의 요청을 적절한 도구 호출로 분해하고, 그 결과를 조율하세요."
    )
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")
refresh_button = st.sidebar.button("대화 내용 초기화")

if refresh_button:
    st.session_state.messages = []
    st.rerun()

# ---------------------------------------------------------------------------------
# 메인 영역
# ---------------------------------------------------------------------------------
with st.form(key='assistant_form', clear_on_submit=True):
    user_input = st.text_area("요청을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))

        # 슈퍼바이저 에이전트 실행 및 응답 추출
        try:
            with st.spinner("처리 중..."):
                result = supervisor.invoke({
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
        st.text_area("최신 응답", value=last_msg.content, height=200, disabled=True)

# 대화 이력
st.subheader("대화 이력")
# 대화 이력을 순서대로 표시
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=f"user_{idx}")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=f"ai_{idx}")
