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
# # 03. Error Handling
#
# 에이전트 루프에서 **max_turns**를 초과하면 기본적으로 `MaxTurnsExceeded` 예외가 발생하여 프로그램이 중단됩니다.  
# **error_handlers**를 사용하면 예외 대신 **제어된 최종 메시지**를 반환하도록 할 수 있습니다.
#
# ### 핵심 개념
#
# - **`error_handlers={"max_turns": handler}`**: `Runner.run()` 호출 시 전달하여, 턴 초과 시 핸들러 함수가 실행되도록 설정
# - **`RunErrorHandlerResult`**: 핸들러가 반환하는 결과 객체로, `final_output`(폴백 메시지)과 `include_in_history` 옵션을 포함
# - **`include_in_history=False`**: 폴백 메시지를 대화 기록에 남기지 않음 (사용자 안내 전용)
#
# ### 학습 내용
#
# 1. `max_turns` 초과 시 `error_handlers`로 예외 없이 안전하게 처리하기
# 2. 비동기(`async def`) / 동기(`def`) 핸들러 작성법
# 3. `include_in_history` 옵션의 역할과 활용

# %%
from dotenv import load_dotenv
load_dotenv()

Model = "gpt-5-nano"

# %% [markdown]
# ## 에이전트와 max_turns
#
# 턴 수를 3으로 제한하고, 복잡한 요청을 넣어 턴이 많이 필요하게 만듭니다.  
#
# - `error_handlers={"max_turns": on_max_turns}`: max_turns 초과 시 `on_max_turns` 호출

# %%
from agents import Agent, function_tool, RunErrorHandlerInput, RunErrorHandlerResult, Runner

@function_tool
def NumberTool(n: int) -> int:
    return n

# 핸들러 함수 정의
async def on_max_turns(ctx):
    return RunErrorHandlerResult(
        final_output="죄송합니다. 처리 한도를 초과했습니다. 잠시 후 다시 시도해주세요.",
        include_in_history=False  # 대화 기록에 저장 안 함
    )
    
agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요.",
    model=Model,
    tools=[NumberTool],
)

result = await Runner.run(
        agent,
        """
        절대 규칙:
        - 한 번의 LLM 응답에서 tool은 정확히 1번만 호출하세요.
        - tool 결과를 받은 뒤에만 다음 숫자로 진행하세요.
        - 1부터 시작해서 5까지 반복하세요.
        """,
        max_turns=2,
        error_handlers={"max_turns": on_max_turns}  # 예외 대신 핸들러 실행
    )

print(result.final_output)


# %% [markdown]
# ## max_turns 핸들러 정의
#
# `RunErrorHandlerInput`을 인자로 받고, `RunErrorHandlerResult`를 반환합니다.  
#
# - 핸들러는 `RunErrorHandlerResult(final_output=..., include_in_history=False)`를 반환
# - `include_in_history=False`: 이 메시지를 대화 히스토리에 넣지 않음 (사용자 안내만 할 때 유용)

# %%
def on_max_turns(_data: RunErrorHandlerInput[None]) -> RunErrorHandlerResult:
    return RunErrorHandlerResult(
        final_output="턴 제한 내에 완료하지 못했습니다. 요청을 더 구체적으로 줄여주세요.",
        include_in_history=False,
    )

# %% [markdown]
# ## error_handlers로 실행
#
# max_turns를 초과해도 예외가 나지 않고, 위에서 정한 메시지가 `result.final_output`으로 반환됩니다.  
#
# 에이전트 루프에서 **max_turns**를 초과하면 기본적으로 `MaxTurnsExceeded` 예외가 발생합니다.  
# **error_handlers**를 사용하면 예외 대신 **제어된 최종 메시지**를 반환하도록 할 수 있습니다.

# %%
result = await Runner.run(
    agent,
    """절대 규칙:
        - 한 번의 LLM 응답에서 tool은 정확히 1번만 호출하세요.
        - tool 결과를 받은 뒤에만 다음 숫자로 진행하세요.
        - 1부터 시작해서 5까지 반복하세요.
        """,
    max_turns=3,
    error_handlers={"max_turns": on_max_turns},
)
print(result.final_output)

# %% [markdown]
# ## 정리
#
# - `Runner.run_sync(..., max_turns=N, error_handlers={"max_turns": handler})`
# - 핸들러: `(RunErrorHandlerInput) -> RunErrorHandlerResult`
# - `include_in_history=False`: 폴백 메시지를 대화 기록에 남기지 않음
# - 지원하는 키: 현재 `"max_turns"` 등 (문서 참고)
