# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Human-in-the-Loop (인간 개입 패턴)
#
# **Human-in-the-Loop (HITL)** 패턴은 에이전트가 중요한 작업을 수행하기 전에
# 인간의 검토와 승인을 받는 패턴입니다.
#
# 이 패턴은:
# - 보안이 중요한 작업 (결제, 데이터 삭제 등)
# - 정확성이 중요한 작업 (법률 문서 검토, 의료 진단 등)
# - 규정 준수가 필요한 작업 (규정 위반 가능성 있는 작업)
#
# **참고**: [LangChain 공식 문서 - Human-in-the-Loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# LangSmith 추적 (선택적)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangChain_V1")

# %%
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from uuid import uuid4

model = init_chat_model("gpt-5-mini", model_provider="openai")

# %% [markdown]
# ## 1. 기본 Human-in-the-Loop 설정
#
# 특정 도구 호출 시 인간의 승인을 기다리도록 설정합니다.

# %%
# 결제 승인 요청 도구
@tool
def request_payment_approval(amount: int, purpose: str, requester: str) -> str:
    """관리자 결제 승인 요청 도구.
    
    이 도구 호출은 HumanInTheLoopMiddleware의 정책에 의해
    모델이 바로 실행하지 않고, 먼저 사람(관리자)의 승인을 기다리게 됩니다.
    
    Args:
        amount: 결제 금액
        purpose: 결제 목적
        requester: 요청자 이름
    """
    return (
        f"결제 승인 요청 발생!\n"
        f"- 요청자: {requester}\n"
        f"- 금액: {amount:,}원\n"
        f"- 목적: {purpose}\n"
        f"관리자 결정을 대기 중..."
    )

# Human-in-the-Loop 미들웨어 정의
# 문서: 결정 타입은 approve, edit, reject 세 가지만 지원
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        # 이 도구가 호출되면 인터럽트 발생
        "request_payment_approval": {
            # approve(승인), edit(수정 후 실행), reject(거부)
            "allowed_decisions": ["approve", "edit", "reject"],
            # 도구 호출 시 설명 문구
            "description": "결제 승인 도구 실행 전 관리자 승인이 필요합니다.",
        }
    },
    # 전역 prefix
    description_prefix="도구 실행 승인 대기 중",
)

# 에이전트 생성
payment_agent = create_agent(
    model=model,
    tools=[request_payment_approval],
    middleware=[hitl_middleware],
    checkpointer=InMemorySaver()  # HITL은 반드시 checkpointer 필요
)

payment_agent

# %% [markdown]
# ## 2. 결제 승인 시나리오
#
# 결제 요청 → 인터럽트 → 관리자 승인/거부 → 워크플로우 재개

# %%
# 결제 요청 정보
requester = "홍길동"
amount = 850_000
purpose = "세미나 출장비"

# 스레드 ID 생성
thread_id = f"payment-{uuid4()}"
config = {"configurable": {"thread_id": thread_id}}

# 결제 승인 요청 → 인터럽트 발생
print("=" * 80)
print("1단계: 결제 요청 → 워크플로우 일시정지")
print("=" * 80)

response = payment_agent.invoke(
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

# 인터럽트 확인
interrupts = response.get("__interrupt__", [])
if interrupts:
    print("\n인터럽트 발생!")
    print(f"인터럽트 정보: {interrupts}")

# %% [markdown]
# ### 2.1 관리자 승인

# %%
# 관리자 결정: 승인
decision_type = "approve"

# HITL 재개용 Command
admin_command = Command(
    resume={
        "decisions": [
            {
                "type": decision_type,
            }
        ]
    }
)

# 워크플로우 재개
print("\n" + "=" * 80)
print("2단계: 관리자 승인 → 워크플로우 재개")
print("=" * 80)

resumed = payment_agent.invoke(admin_command, config=config)

# 승인 후 후속 처리
followup_prompt = (
    "결제 요청이 관리자에 의해 승인되었습니다. "
    "'승인 완료/처리 완료' 형식으로 짧게 결과만 안내해 주세요."
)

final = payment_agent.invoke(
    {"messages": [HumanMessage(content=followup_prompt)]},
    config=config,
)

print("\n최종 응답:")
final["messages"][-1].pretty_print()

# %% [markdown]
# ### 2.2 관리자 거부

# %%
# 새로운 스레드로 거부 시나리오 테스트
thread_id2 = f"payment-reject-{uuid4()}"
config2 = {"configurable": {"thread_id": thread_id2}}

# 결제 요청
response2 = payment_agent.invoke(
    {
        "messages": [
            SystemMessage(
                content=(
                    "당신은 결제 승인 Human-in-the-loop 에이전트입니다. "
                    "거절된 경우 '결제 거절/승인 불가' 형식으로 안내하세요."
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
    config=config2,
)

# 관리자 거부 결정
reject_command = Command(
    resume={
        "decisions": [{"type": "reject"}]
    }
)

resumed2 = payment_agent.invoke(reject_command, config=config2)

# 거부 후 후속 처리
reject_followup = (
    "결제 요청이 관리자에 의해 거절되었습니다. "
    "'결제 거절/승인 불가' 형식으로, 간단한 사유를 포함하여 짧게 안내해 주세요."
)

final2 = payment_agent.invoke(
    {"messages": [HumanMessage(content=reject_followup)]},
    config=config2,
)

print("=" * 80)
print("거부 시나리오 결과:")
print("=" * 80)
final2["messages"][-1].pretty_print()

# %% [markdown]
# ### 2.3 관리자 수정(edit)
#
# `edit` 결정으로 도구 인자를 수정한 뒤 실행할 수 있습니다.
# 문서: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

# %%
thread_id_edit = f"payment-edit-{uuid4()}"
config_edit = {"configurable": {"thread_id": thread_id_edit}}

response_edit = payment_agent.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    f"관리자님, {requester}의 결제 요청입니다.\n"
                    f"- 금액: {amount:,}원\n- 목적: {purpose}\n"
                    "승인 여부를 판단해 주세요."
                )
            ),
        ]
    },
    config=config_edit,
)

if response_edit.get("__interrupt__"):
    # edit: 금액을 800_000으로 수정한 뒤 실행
    edit_command = Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "request_payment_approval",
                        "args": {
                            "amount": 800_000,
                            "purpose": purpose,
                            "requester": requester,
                        },
                    }
                }
            ]
        }
    )
    resumed_edit = payment_agent.invoke(edit_command, config=config_edit)
    print("수정(edit) 후 재개 결과: 금액이 800,000원으로 반영되어 실행됨")

# %% [markdown]
# ## 3. 다중 도구 인터럽트
#
# 여러 도구에 대해 인터럽트를 설정할 수 있습니다.

# %%
# 여러 도구 정의
@tool
def delete_user_data(user_id: str) -> str:
    """사용자 데이터를 삭제합니다. 위험한 작업이므로 승인이 필요합니다."""
    return f"사용자 {user_id}의 데이터 삭제 요청"

@tool
def send_bulk_email(recipients: list[str], subject: str) -> str:
    """대량 이메일을 전송합니다. 승인이 필요합니다."""
    return f"{len(recipients)}명에게 이메일 전송: {subject}"

@tool
def update_database(query: str) -> str:
    """데이터베이스를 업데이트합니다. 승인이 필요합니다."""
    return f"데이터베이스 업데이트: {query}"

# 다중 도구 HITL 미들웨어
multi_hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_user_data": {
            "allowed_decisions": ["approve", "reject"],
            "description": "사용자 데이터 삭제는 위험한 작업입니다."
        },
        "send_bulk_email": {
            "allowed_decisions": ["approve", "reject"],
            "description": "대량 이메일 전송 전 승인이 필요합니다."
        },
        "update_database": {
            "allowed_decisions": ["approve", "reject"],
            "description": "데이터베이스 업데이트 전 승인이 필요합니다."
        }
    },
    description_prefix="도구 실행 승인 대기 중",
)

# 다중 도구 에이전트
# system_prompt: 사용자 요청 시 반드시 해당 도구를 호출하도록 유도 → 인터럽트가 실제로 걸리게 함
multi_agent = create_agent(
    model=model,
    tools=[delete_user_data, send_bulk_email, update_database],
    middleware=[multi_hitl],
    checkpointer=InMemorySaver(),
    system_prompt=(
        "당신은 사용자 데이터 삭제, 대량 이메일 전송, 데이터베이스 업데이트 요청을 처리하는 에이전트입니다. "
        "사용자가 삭제/이메일 전송/DB 업데이트를 요청하면 반드시 해당 도구(delete_user_data, send_bulk_email, update_database)를 호출하세요. "
        "자연어로 확인만 하지 말고, 도구를 호출하여 Human-in-the-Loop 승인 절차가 진행되도록 하세요."
    ),
)

multi_agent

# %% [markdown]
# ## 4. 스트리밍과 인터럽트
#
# 스트리밍 모드에서 인터럽트를 처리하는 방법입니다.

# %%
# 스트리밍용 스레드 ID 생성
thread_id3 = f"stream-{uuid4()}"

config3 = {
    "configurable": {
        "thread_id": thread_id3
    }
}

print("=" * 80)
print("스트리밍 모드에서 인터럽트 처리 (stream_mode=['updates', 'messages']):")
print("=" * 80)

# ------------------------------------------------------------
# 에이전트 실행 중 발생하는 상태 이벤트와 모델 응답을 동시에 실시간 처리하는 스트리밍 루프
# ------------------------------------------------------------
# stream_mode:
#   - "updates"   → 상태 변화, interrupt, tool call 등 이벤트 수신
#   - "messages"  → 모델이 생성하는 토큰 스트리밍 수신
#
# 반환값은 (mode, chunk) 형태의 generator
# SystemMessage로 이번 턴에서 delete_user_data 도구 호출을 명시 → 인터럽트 발생 보장
for mode, chunk in multi_agent.stream(
    {
        "messages": [
            SystemMessage(
                content=(
                    "사용자가 데이터 삭제를 요청하면 반드시 delete_user_data 도구를 호출하세요. "
                    "이번 요청에 대해 delete_user_data(user_id='user123')를 호출하세요."
                )
            ),
            HumanMessage(content="사용자 ID 'user123'의 데이터를 삭제해주세요."),
        ]
    },
    config=config3,
    stream_mode=["updates", "messages"],  # 두 가지 이벤트 동시 수신 (실시간 reasoning + 상태 추적)
):

    if mode == "updates":
        # interrupt 이벤트가 발생한 경우
        if "__interrupt__" in chunk:

            print("\n\n인터럽트 발생:")

            # 여러 interrupt가 동시에 발생할 수 있음
            for interrupt in chunk["__interrupt__"]:

                # interrupt 객체 또는 dict 처리
                val = (
                    getattr(interrupt, "value", interrupt)
                    if hasattr(interrupt, "value")
                    else interrupt
                )

                # 대기 중인 action 요청 출력
                for req in val.get("action_requests", []):
                    print(
                        f"  대기 중인 작업: "
                        f"{req.get('description', req.get('name', 'N/A'))}"
                    )

    # 모델 토큰 스트리밍 처리
    elif mode == "messages":

        # chunk는 (token, metadata) 형태
        token, metadata = chunk

        # token.content가 존재하면 즉시 출력
        # flush=True → 실시간 출력
        if getattr(token, "content", None):
            print(token.content, end="", flush=True)

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **인터럽트 설정**: `interrupt_on`으로 특정 도구 호출 시 인터럽트 발생
# 2. **결정 옵션**: `allowed_decisions`는 문서 기준 `approve`, `edit`, `reject` 세 가지만 지원
# 3. **체크포인터 필수**: HITL은 반드시 checkpointer 필요
# 4. **Command로 재개**: `Command(resume={"decisions": [...]})` 로 워크플로우 재개
# 5. **edit**: 수정 후 실행 시 `edited_action`에 `name`, `args` 전달
# 6. **스트리밍**: `stream_mode=["updates", "messages"]` 로 진행 상황과 토큰 스트리밍
#
# - [streamlit-llm_LangChain/310_Personal_Assistant_App.py](streamlit-llm_LangChain/350_Human_In_The_Loop_App.py)에서 웹 UI 구현

# %%
