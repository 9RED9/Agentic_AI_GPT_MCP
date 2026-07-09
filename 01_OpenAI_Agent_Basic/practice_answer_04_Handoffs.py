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
# # 실습 문제 모범답안 - 04. Handoffs
#
# `04_Handoffs` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# 아래 요구사항에 맞는 **여행 예약 지원 시스템**을 구현하세요.
#
# **에이전트 구성:**
#
# 1. **Triage Agent**: 사용자 요청을 분류하여 적절한 에이전트에 핸드오프
# 2. **Flight Agent**: 항공편 예약 및 조회 처리
# 3. **Hotel Agent**: 호텔 예약 및 조회 처리
# 4. **Cancellation Agent**: 예약 취소 처리 (`on_handoff` 콜백으로 취소 정보 로깅)
#
# **요구사항:**
# - `Cancellation Agent`로 핸드오프 시 `input_type`으로 예약 번호(`booking_id: str`)와
#   취소 사유(`reason: str`)를 전달받아 출력
# - 모든 에이전트에 `prompt_with_handoff_instructions` 적용
# - `Flight Agent`, `Hotel Agent`는 `handoff()` 함수 형태로 지정

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. 취소 요청 데이터 구조와 콜백 정의
#
# - `CancellationRequest`: LLM이 핸드오프 시 채워서 전달할 구조화된 입력 (Pydantic 모델)
# - `on_cancellation`: 핸드오프 발생 시 취소 정보를 로깅하는 콜백
#   (`input_type`과 함께 사용하므로 `async def` + 두 번째 인자로 `input_data`를 받음)

# %%
from pydantic import BaseModel
from agents import Agent, Runner, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# 취소 요청 시 전달할 데이터 구조 (LLM이 채워서 전달)
class CancellationRequest(BaseModel):
    booking_id: str  # 예약 번호
    reason: str      # 취소 사유

# Cancellation Agent로 핸드오프될 때 실행되는 콜백 (취소 정보 로깅)
async def on_cancellation(ctx: RunContextWrapper[None], input_data: CancellationRequest):
    print(f"[취소 요청 접수] 예약번호: {input_data.booking_id}, 사유: {input_data.reason}")

# %% [markdown]
# ## 2. 전문 에이전트 정의
#
# 모든 에이전트의 instructions에 `prompt_with_handoff_instructions`를 적용하여
# LLM이 핸드오프 개념을 이해하도록 합니다.

# %%
# 항공편 예약 및 조회 전문 에이전트
flight_agent = Agent(
    name="Flight_Agent",
    instructions=prompt_with_handoff_instructions(
        "항공편 예약 및 조회를 처리합니다. 출발지, 도착지, 날짜를 확인하고 예약을 안내하세요."
    ),
    model=Model,
)

# 호텔 예약 및 조회 전문 에이전트
hotel_agent = Agent(
    name="Hotel_Agent",
    instructions=prompt_with_handoff_instructions(
        "호텔 예약 및 조회를 처리합니다. 지역, 숙박 일수, 인원을 확인하고 예약을 안내하세요."
    ),
    model=Model,
)

# 예약 취소 전문 에이전트
cancellation_agent = Agent(
    name="Cancellation_Agent",
    instructions=prompt_with_handoff_instructions(
        "예약 취소 요청을 처리합니다. 고객에게 취소 절차와 환불 규정을 안내하세요."
    ),
    model=Model,
)

# %% [markdown]
# ## 3. Triage Agent 정의
#
# - `Flight Agent`, `Hotel Agent`: 요구사항에 따라 `handoff()` 함수 형태로 지정
# - `Cancellation Agent`: `on_handoff` 콜백 + `input_type`으로 취소 정보를 전달받아 로깅

# %%
triage_agent = Agent(
    name="Triage_Agent",
    instructions=prompt_with_handoff_instructions(
        "여행 예약 관련 고객 요청을 분류하세요:\n"
        "- 항공편 예약/조회 → Flight Agent\n"
        "- 호텔 예약/조회 → Hotel Agent\n"
        "- 예약 취소 → Cancellation Agent (예약번호와 취소 사유 필요)"
    ),
    model=Model,
    handoffs=[
        # handoff() 함수 형태로 지정 (요구사항)
        handoff(flight_agent),
        handoff(hotel_agent),
        handoff(
            cancellation_agent,
            on_handoff=on_cancellation,        # 핸드오프 발생 시 취소 정보 로깅
            input_type=CancellationRequest,    # LLM이 채워야 할 구조화된 입력
        ),
    ],
)

# %% [markdown]
# ## 4. 테스트
#
# **테스트 입력:**
# - `"서울-제주 항공편을 예약하고 싶습니다."` → Flight Agent 응답
# - `"제주도 호텔을 3박 예약하려고 합니다."` → Hotel Agent 응답
# - `"예약번호 BK-999 항공편을 취소하고 싶어요. 일정이 바뀌어서요."` → Cancellation Agent 핸드오프 + 콜백 출력

# %%
print("=== 테스트 1: 항공편 예약 ===")
result = await Runner.run(triage_agent, "서울-제주 항공편을 예약하고 싶습니다.")

print(result.last_agent.name)
print(result.final_output)

# %%
print("=== 테스트 2: 호텔 예약 ===")
result = await Runner.run(triage_agent, "제주도 호텔을 3박 예약하려고 합니다.")

print(result.last_agent.name)
print(result.final_output)

# %%
print("=== 테스트 3: 예약 취소 ===")
result = await Runner.run(triage_agent, "예약번호 BK-999 항공편을 취소하고 싶어요. 일정이 바뀌어서요.")

print(result.last_agent.name)
print(result.final_output)

# %% [markdown]
# ### 정리
#
# | 구성 요소 | 역할 |
# |------|------|
# | `CancellationRequest` (Pydantic) | 핸드오프 시 LLM이 채워 전달하는 구조화된 입력 (`booking_id`, `reason`) |
# | `on_cancellation` (`async def`) | 취소 핸드오프 발생 시 취소 정보 로깅 콜백 |
# | `Flight_Agent` / `Hotel_Agent` | 항공편/호텔 예약 처리, `handoff()` 함수 형태로 등록 |
# | `Cancellation_Agent` | 취소 처리, `on_handoff` + `input_type` 조합 |
# | `prompt_with_handoff_instructions` | 모든 에이전트에 핸드오프 개념을 알려주는 권장 프롬프트 |
#
# **실행 흐름:**
# ```
# 사용자
#   └─→ Triage_Agent (분류)
#         ├─→ Flight_Agent        (항공편 예약/조회)
#         ├─→ Hotel_Agent         (호텔 예약/조회)
#         └─→ Cancellation_Agent  (예약 취소)
#               ↑ 핸드오프 순간 on_cancellation 콜백 실행
#                 → "[취소 요청 접수] 예약번호: BK-999, 사유: ..." 출력
# ```
