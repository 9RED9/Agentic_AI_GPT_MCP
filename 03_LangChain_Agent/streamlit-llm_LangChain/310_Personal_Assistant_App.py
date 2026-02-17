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


def _last_message_text(messages: list) -> str:
    """마지막 메시지의 텍스트를 반환합니다 (.content / .text 호환)."""
    if not messages:
        return ""
    last = messages[-1]
    return getattr(last, "content", None) or getattr(last, "text", str(last))


def make_schedule_event_tool(calendar_agent):
    """캘린더 에이전트를 클로저로 묶어 st.session_state 없이 동작하는 도구를 반환합니다."""
    @tool
    def schedule_event(request: str) -> str:
        """자연어 요청을 기반으로 캘린더 이벤트를 예약합니다."""
        result = calendar_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return _last_message_text(result["messages"])
    return schedule_event


def make_manage_email_tool(email_agent):
    """이메일 에이전트를 클로저로 묶어 st.session_state 없이 동작하는 도구를 반환합니다."""
    @tool
    def manage_email(request: str) -> str:
        """자연어 요청을 기반으로 이메일을 전송합니다."""
        result = email_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        return _last_message_text(result["messages"])
    return manage_email

# ---------------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------------
if "llm" not in st.session_state:
    st.session_state.llm = init_chat_model("gpt-5-nano", model_provider="openai")

if "calendar_agent" not in st.session_state:
    try:
        st.session_state.calendar_agent = create_agent(
            st.session_state.llm,
            tools=[create_calendar_event],
            system_prompt="당신은 캘린더 일정 관리 도우미입니다."
        )
    except Exception as e:
        st.sidebar.error(f"캘린더 에이전트 초기화 오류: {str(e)}")

if "email_agent" not in st.session_state:
    try:
        st.session_state.email_agent = create_agent(
            st.session_state.llm,
            tools=[send_email],
            system_prompt="당신은 이메일 작성 도우미입니다."
        )
    except Exception as e:
        st.sidebar.error(f"이메일 에이전트 초기화 오류: {str(e)}")

if "supervisor" not in st.session_state:
    if st.session_state.get("calendar_agent") and st.session_state.get("email_agent"):
        schedule_event_tool = make_schedule_event_tool(st.session_state.calendar_agent)
        manage_email_tool = make_manage_email_tool(st.session_state.email_agent)
        st.session_state.supervisor = create_agent(
            st.session_state.llm,
            tools=[schedule_event_tool, manage_email_tool],
            system_prompt=(
                "당신은 친절한 개인 비서입니다. "
                "캘린더 일정을 등록하고 이메일을 보낼 수 있습니다. "
                "사용자의 요청을 적절한 도구 호출로 분해하고, 그 결과를 조율하세요."
            )
        )
    else:
        st.session_state.supervisor = None

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
if not st.session_state.get("supervisor"):
    st.warning("⚠️ 에이전트를 초기화하는 중 오류가 발생했습니다. 사이드바 메시지를 확인한 뒤 페이지를 새로고침하세요.")

with st.form(key='assistant_form', clear_on_submit=True):
    user_input = st.text_area("요청을 입력하세요:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))

        if not st.session_state.get("supervisor"):
            st.session_state.messages.append(AIMessage(content="에이전트가 준비되지 않았습니다. 페이지를 새로고침해 주세요."))
            st.error("에이전트가 준비되지 않았습니다. 페이지를 새로고침해 주세요.")
        else:
            try:
                with st.spinner("처리 중..."):
                    result = st.session_state.supervisor.invoke({
                        "messages": [{"role": "user", "content": user_input}]
                    })
                    last_msg = result["messages"][-1]
                    ai_response = getattr(last_msg, "content", None) or getattr(last_msg, "text", str(last_msg))
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
for idx, msg in enumerate(st.session_state.messages):
    if isinstance(msg, HumanMessage):
        message(msg.content, is_user=True, key=f"user_{idx}")
    elif isinstance(msg, AIMessage):
        message(msg.content, is_user=False, key=f"ai_{idx}")
