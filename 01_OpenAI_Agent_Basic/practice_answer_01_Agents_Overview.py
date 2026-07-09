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
# # 실습 문제 모범답안 - 01. Agents Overview
#
# `01_Agents_Overview` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# 1. **분류 에이전트**:
#    사용자 입력이 '수학 문제'인지 '기타 질문'인지 분류하고,
#    분류 결과에 따라 적절한 에이전트로 handoff 하세요.
#
# 2. **수학 에이전트**:
#    수학 문제일 경우, Python 함수 `calculate_area(length: float, width: float)`를
#    도구로 등록하여 직사각형의 넓이를 계산해주는 역할을 수행하세요.
#
# 3. **일반 에이전트**:
#    기타 질문에 대해서는 "질문을 이해했지만 수학 관련 질문만 도와드릴 수 있어요." 라고 응답하세요.

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. 도구 정의
#
# 직사각형의 넓이를 계산하는 Python 함수를 `@function_tool` 데코레이터로 도구로 등록합니다.
#
# - 도구 이름은 함수 이름(`calculate_area`)에서 자동 생성됩니다.
# - 도구 설명은 docstring에서 가져옵니다.
# - 입력 스키마는 함수 인자(`length`, `width`)로부터 자동 생성됩니다.

# %%
from agents import Agent, Runner, function_tool

# 직사각형의 넓이를 계산하는 도구
@function_tool
def calculate_area(length: float, width: float) -> float:
    """직사각형의 가로(length)와 세로(width)를 받아 넓이를 반환한다."""
    print(f"** calculate_area 함수 실행 ** 가로: {length}, 세로: {width}")
    return length * width


# %% [markdown]
# ## 2. 전문 에이전트 정의
#
# - **수학 에이전트**: `calculate_area` 도구를 사용하여 직사각형의 넓이를 계산
# - **일반 에이전트**: 수학 이외의 질문에는 정해진 안내 문구로만 응답

# %%
# 수학 문제를 처리하는 에이전트 (calculate_area 도구 사용)
math_agent = Agent(
    name="Math_agent",
    handoff_description="수학 문제(직사각형 넓이 계산)를 위한 전문 에이전트",
    instructions=(
        "당신은 수학 문제를 도와주는 도우미입니다. "
        "직사각형의 넓이를 구할 때는 반드시 calculate_area 도구를 사용하세요. "
        "계산 과정을 간단히 설명하고 결과를 알려주세요."
    ),
    model=Model,
    tools=[calculate_area],
)

# 기타 질문을 처리하는 일반 에이전트
general_agent = Agent(
    name="General_agent",
    handoff_description="수학 이외의 일반 질문을 위한 에이전트",
    instructions=(
        "어떤 질문이 오더라도 반드시 다음 문장으로만 응답하세요: "
        "'질문을 이해했지만 수학 관련 질문만 도와드릴 수 있어요.'"
    ),
    model=Model,
)

# %% [markdown]
# ## 3. 분류 에이전트 정의
#
# 사용자 입력이 '수학 문제'인지 '기타 질문'인지 판단하여
# 적절한 에이전트로 handoff 하는 분류(triage) 에이전트를 정의합니다.
#
# Handoff는 LLM에게 도구로 표현됩니다.
# ```
# "Math_agent"    →  transfer_to_math_agent
# "General_agent" →  transfer_to_general_agent
# ```

# %%
# 입력을 분류하여 적절한 에이전트로 위임하는 분류 에이전트
triage_agent = Agent(
    name="Classify_agent",
    instructions=(
        "사용자 입력이 '수학 문제'인지 '기타 질문'인지 분류하세요. "
        "수학 문제(예: 직사각형 넓이 계산)이면 Math_agent에게, "
        "그 외의 질문이면 General_agent에게 넘겨주세요."
    ),
    model=Model,
    handoffs=[math_agent, general_agent],  # 분류 결과에 따라 위임할 에이전트 목록
)

# %% [markdown]
# ## 4. 테스트
#
# ### 테스트 입력 예시
#
# * `"가로 5, 세로 7인 직사각형의 넓이를 구해주세요."`
#   👉 수학 에이전트 → 함수 실행 → 넓이 출력
#
# * `"오늘 날씨 어때?"`
#   👉 일반 에이전트 응답

# %%
# 테스트 1: 수학 문제 → Math_agent로 handoff → calculate_area 도구 실행
result = await Runner.run(triage_agent, input="가로 5, 세로 7인 직사각형의 넓이를 구해주세요.")
print("Output:", result.final_output)
print("최종 처리 에이전트:", result.last_agent.name)

# %%
# 테스트 2: 기타 질문 → General_agent로 handoff → 안내 문구 응답
result = await Runner.run(triage_agent, input="오늘 날씨 어때?")
print("Output:", result.final_output)
print("최종 처리 에이전트:", result.last_agent.name)

# %% [markdown]
# ### 정리
#
# | 구성 요소 | 역할 |
# |------|------|
# | `calculate_area` (`@function_tool`) | 직사각형 넓이 계산 도구 |
# | `Math_agent` | 수학 문제 처리, 도구 호출로 넓이 계산 |
# | `General_agent` | 기타 질문에 정해진 안내 문구로만 응답 |
# | `Classify_agent` | 입력을 분류하여 적절한 에이전트로 handoff |
#
# **실행 흐름:**
# ```
# 사용자 입력
#     ↓
# Classify_agent (분류)
#     ├─ 수학 문제 → Math_agent → calculate_area 실행 → 넓이 출력
#     └─ 기타 질문 → General_agent → 안내 문구 응답
# ```
