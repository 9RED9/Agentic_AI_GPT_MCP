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
# HITL 정책에 의해 호출 시 인터럽트 발생, 관리자 승인/거부 후 재개
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
# LLM 및 HITL 에이전트, 대화 스레드·승인 대기 상태를 세션에 유지
if "llm" not in st.session_state:
    st.session_state.llm = init_chat_model("gpt-5-mini", model_provider="openai")

if "agent" not in st.session_state:
    # 교재·문서 기준: approve, edit, reject 세 가지만 지원
    hitl_middleware = HumanInTheLoopMiddleware(
        interrupt_on={
            "request_payment_approval": {
                "allowed_decisions": ["approve", "edit", "reject"],
                "description": "결제 승인 도구 실행 전 관리자 승인이 필요합니다.",
            }
        },
        description_prefix="도구 실행 승인 대기 중",
    )
    # HITL 사용 시 checkpointer 필수 (인터럽트 후 재개용)
    st.session_state.agent = create_agent(
        model=st.session_state.llm,
        tools=[request_payment_approval],
        middleware=[hitl_middleware],
        checkpointer=InMemorySaver()
    )

# 대화 스레드 식별자 (재개 시 동일 스레드로 이어가기 위함)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"payment-{uuid4()}"

# 승인 대기 중인 인터럽트 정보 (있으면 승인/거부 UI 표시)
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

# ---------------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")
refresh_button = st.sidebar.button("초기화")

# 초기화 시 스레드·승인 대기 상태를 비우고 페이지 다시 그리기
if refresh_button:
    st.session_state.thread_id = f"payment-{uuid4()}"
    st.session_state.pending_approval = None
    st.rerun()

# ---------------------------------------------------------------------------------
# 메인 영역
# ---------------------------------------------------------------------------------
# 에이전트 호출 시 동일 스레드로 재개할 수 있도록 config 전달
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 인터럽트가 걸린 경우: 승인/거부 UI 표시
if st.session_state.pending_approval:
    st.warning("⚠️ 승인 대기 중인 요청이 있습니다.")

    col1, col2 = st.columns(2)

    with col1:
        # 승인 시 Command로 재개 후 에이전트 재실행
        if st.button("✅ 승인", type="primary"):
            decision = Command(resume={"decisions": [{"type": "approve"}]})
            result = st.session_state.agent.invoke(decision, config=config)
            st.session_state.pending_approval = None
            st.success("승인되었습니다!")
            st.rerun()

    with col2:
        # 거부 사유 입력 시 에이전트에 피드백으로 전달 (문서의 reject message)
        reject_reason = st.text_input(
            "거부 사유 (선택)",
            key="reject_reason",
            placeholder="거부 시 에이전트에 전달할 피드백",
        )
        if st.button("❌ 거부"):
            decision_payload = {"type": "reject"}
            if reject_reason.strip():
                decision_payload["message"] = reject_reason.strip()
            decision = Command(resume={"decisions": [decision_payload]})
            result = st.session_state.agent.invoke(decision, config=config)
            st.session_state.pending_approval = None
            st.info("거부되었습니다.")
            st.rerun()

    # 대기 중인 액션 요청 내용 미리보기
    st.json(st.session_state.pending_approval)
else:
    # 결제 요청 폼: 입력 후 에이전트 호출 → 도구 호출 시 인터럽트 발생
    with st.form(key='payment_form'):
        st.subheader("결제 승인 요청")

        requester = st.text_input("요청자", value="홍길동")
        amount = st.number_input("금액 (원)", min_value=0, value=850000)
        purpose = st.text_input("목적", value="세미나 출장비")

        submit_button = st.form_submit_button(label="결제 요청")

        if submit_button:
            with st.spinner("요청 처리 중..."):
                try:
                    # 시스템 메시지(교재와 동일): 승인 시 짧게만 응답하도록 지시
                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                SystemMessage(
                                    content=(
                                        "당신은 결제 승인 Human-in-the-loop 에이전트입니다. "
                                        "인터럽트가 승인(approve)으로 해소되면, 대기 상태를 반복 설명하지 말고 "
                                        "즉시 '승인 완료/처리 완료' 형태로만 간단히 응답하세요."
                                    )
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

                    # 응답에 인터럽트가 있으면 승인/거부 화면으로 전환
                    interrupts = response.get("__interrupt__", [])
                    if interrupts:
                        st.session_state.pending_approval = interrupts
                        st.rerun()
                    else:
                        st.success("요청이 처리되었습니다.")
                except Exception as e:
                    st.error(f"오류: {str(e)}")
