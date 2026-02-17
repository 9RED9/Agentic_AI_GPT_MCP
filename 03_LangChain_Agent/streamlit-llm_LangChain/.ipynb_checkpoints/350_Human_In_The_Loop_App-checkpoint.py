#---------------------------------------------------------
# Streamlit Human-in-the-Loop 앱
#---------------------------------------------------------
from dotenv import load_dotenv
_ = load_dotenv()

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from uuid import uuid4

# ---------------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------------
st.set_page_config(page_title="Human-in-the-Loop", page_icon=":hand:")
st.markdown("<h1 style='text-align: center;'>Human-in-the-Loop 승인 시스템</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------
# 도구 정의
# ---------------------------------------------------------------------------------
@tool
def request_payment_approval(amount: int, purpose: str, requester: str) -> str:
    """관리자 결제 승인 요청 도구."""
    return (
        f"결제 승인 요청 발생!\n"
        f"- 요청자: {requester}\n"
        f"- 금액: {amount:,}원\n"
        f"- 목적: {purpose}\n"
        f"관리자 결정을 대기 중..."
    )

# ---------------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------------
if "llm" not in st.session_state:
    st.session_state.llm = init_chat_model("gpt-5-nano", model_provider="openai")

if "agent" not in st.session_state:
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "request_payment_approval": {
                "allowed_decisions": ["approve", "reject"],
                "description": "결제 승인 도구 실행 전 관리자 승인이 필요합니다.",
            }
        },
        description_prefix="도구 실행 승인 대기 중",
    )
    
    st.session_state.agent = create_agent(
        model=st.session_state.llm,
        tools=[request_payment_approval],
        middleware=[hitl_middleware],
        checkpointer=InMemorySaver()
    )

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"payment-{uuid4()}"

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

# ---------------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")
refresh_button = st.sidebar.button("초기화")

if refresh_button:
    st.session_state.thread_id = f"payment-{uuid4()}"
    st.session_state.pending_approval = None
    st.experimental_rerun()

# ---------------------------------------------------------------------------------
# 메인 영역
# ---------------------------------------------------------------------------------
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 승인 대기 중인 경우
if st.session_state.pending_approval:
    st.warning("⚠️ 승인 대기 중인 요청이 있습니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ 승인", type="primary"):
            decision = Command(resume={"decisions": [{"type": "approve"}]})
            result = st.session_state.agent.invoke(decision, config=config)
            st.session_state.pending_approval = None
            st.success("승인되었습니다!")
            st.experimental_rerun()
    
    with col2:
        if st.button("❌ 거부"):
            decision = Command(resume={"decisions": [{"type": "reject"}]})
            result = st.session_state.agent.invoke(decision, config=config)
            st.session_state.pending_approval = None
            st.info("거부되었습니다.")
            st.experimental_rerun()
    
    st.json(st.session_state.pending_approval)
else:
    # 결제 요청 폼
    with st.form(key='payment_form'):
        st.subheader("결제 승인 요청")
        
        requester = st.text_input("요청자", value="홍길동")
        amount = st.number_input("금액 (원)", min_value=0, value=850000)
        purpose = st.text_input("목적", value="세미나 출장비")
        
        submit_button = st.form_submit_button(label="결제 요청")
        
        if submit_button:
            with st.spinner("요청 처리 중..."):
                try:
                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                SystemMessage(
                                    content="당신은 결제 승인 Human-in-the-loop 에이전트입니다."
                                ),
                                HumanMessage(
                                    content=(
                                        f"관리자님, {requester}의 결제 요청입니다.\n"
                                        f"- 금액: {amount:,}원\n- 목적: {purpose}\n"
                                        "승인 여부를 판단해 주세요."
                                    )
                                ),
                            ]
                        },
                        config=config,
                    )
                    
                    # 인터럽트 확인
                    interrupts = response.get("__interrupt__", [])
                    if interrupts:
                        st.session_state.pending_approval = interrupts
                        st.experimental_rerun()
                    else:
                        st.success("요청이 처리되었습니다.")
                except Exception as e:
                    st.error(f"오류: {str(e)}")
