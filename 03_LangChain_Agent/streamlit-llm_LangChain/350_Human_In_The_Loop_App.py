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
    # Human-in-the-Loop 미들웨어 설정
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

if "flash_message" not in st.session_state:
    st.session_state.flash_message = None

# ---------------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------------
st.sidebar.title("⚙️ 설정")
refresh_button = st.sidebar.button("초기화")

if refresh_button:
    st.session_state.thread_id = f"payment-{uuid4()}"
    st.session_state.pending_approval = None
    st.session_state.flash_message = None
    st.rerun()

# ---------------------------------------------------------------------------------
# 메인 영역
# ---------------------------------------------------------------------------------
# flash 메시지 표시 (rerun 후 1회만 표시)
if st.session_state.flash_message:
    msg_type, msg_text = st.session_state.flash_message
    st.session_state.flash_message = None
    if msg_type == "success":
        st.success(msg_text)
    elif msg_type == "info":
        st.info(msg_text)
# 대화 스레드 설정 (thread_id로 대화 식별)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 승인 대기 중인 경우
if st.session_state.pending_approval:
    st.warning("⚠️ 승인 대기 중인 요청이 있습니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ 승인", type="primary"):
            # 승인 결정을 Command로 래핑하여 중단된 agent에 전달
            decision = Command(resume={"decisions": [{"type": "approve"}]})
            
            # 중단된 agent를 승인 결정으로 재개 (이전 thread_id 유지)
            result = st.session_state.agent.invoke(decision, config=config)
            
            # 승인 처리 완료 후 대기 중인 승인 요청 초기화
            st.session_state.pending_approval = None
            
            # 성공 플래시 메시지 설정 (다음 렌더링 시 표시)
            st.session_state.flash_message = ("success", "승인되었습니다!")
            
            # UI 새로고침 (플래시 메시지 표시 및 승인 UI 제거)
            st.rerun()

    with col2:
        if st.button("❌ 거부"):
            # 거부 결정을 Command로 래핑하여 중단된 agent에 전달
            decision = Command(resume={"decisions": [{"type": "reject"}]})
            
            # 중단된 agent를 거부 결정으로 재개
            result = st.session_state.agent.invoke(decision, config=config)
            
            # 거부 처리 완료 후 대기 중인 승인 요청 초기화
            st.session_state.pending_approval = None
            
            # 거부 플래시 메시지 설정 (다음 렌더링 시 표시)
            st.session_state.flash_message = ("info", "거부되었습니다.")
            
            # UI 새로고침 (플래시 메시지 표시 및 승인 UI 제거)
            st.rerun()
    
    # 승인 요청 상세 정보 표시
    for interrupt in st.session_state.pending_approval:
        val = interrupt.value if hasattr(interrupt, "value") else interrupt
        for req in val.get("action_requests", []):
            args = req.get("args", {})
            st.markdown(
                f"| 항목 | 내용 |\n"
                f"|------|------|\n"
                f"| **요청자** | {args.get('requester', '-')} |\n"
                f"| **금액** | {args.get('amount', 0):,}원 |\n"
                f"| **목적** | {args.get('purpose', '-')} |"
            )
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
                # 에이전트 실행 및 인터럽트 여부 확인
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
                        st.rerun()
                    else:
                        st.success("요청이 처리되었습니다.")
                except Exception as e:
                    st.error(f"오류: {str(e)}")
