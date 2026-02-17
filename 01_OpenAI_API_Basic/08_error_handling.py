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
# # 08. Error Handling
#
# 에이전트 루프에서 **max_turns**를 초과하면 기본적으로 `MaxTurnsExceeded` 예외가 발생합니다.
# **error_handlers**를 사용하면 예외 대신 **제어된 최종 메시지**를 반환하도록 할 수 있습니다.
#
# - `error_handlers={"max_turns": on_max_turns}`: max_turns 초과 시 `on_max_turns` 호출
# - 핸들러는 `RunErrorHandlerResult(final_output=..., include_in_history=False)`를 반환
# - `include_in_history=False`: 이 메시지를 대화 히스토리에 넣지 않음 (사용자 안내만 할 때 유용)

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from agents import (
    Agent,
    RunErrorHandlerInput,
    RunErrorHandlerResult,
    Runner,
)

# %% [markdown]
# ## 에이전트와 max_turns
#
# 턴 수를 3으로 제한하고, 복잡한 요청을 넣어 턴이 많이 필요하게 만듭니다.

# %%
agent = Agent(name="Assistant", instructions="Be concise.")

# %% [markdown]
# ## max_turns 핸들러 정의
#
# `RunErrorHandlerInput`을 인자로 받고, `RunErrorHandlerResult`를 반환합니다.

# %%
def on_max_turns(_data: RunErrorHandlerInput[None]) -> RunErrorHandlerResult:
    return RunErrorHandlerResult(
        final_output="I couldn't finish within the turn limit. Please narrow the request.",
        include_in_history=False,
    )

# %% [markdown]
# ## error_handlers로 실행
#
# max_turns를 초과해도 예외가 나지 않고, 위에서 정한 메시지가 `result.final_output`으로 반환됩니다.

# %%
result = Runner.run_sync(
    agent,
    "Analyze this long transcript and list every topic, then summarize each in detail.",
    max_turns=3,
    error_handlers={"max_turns": on_max_turns},
)
print(result.final_output)

# %% [markdown]
# ## 핸들러 없이 실행 시 (참고)
#
# error_handlers를 주지 않으면 max_turns 초과 시 `MaxTurnsExceeded` 예외가 발생합니다.
# 필요하면 `try/except MaxTurnsExceeded`로 처리할 수 있습니다.

# %%
# from agents import MaxTurnsExceeded
# try:
#     Runner.run_sync(agent, "Very long multi-step task...", max_turns=2)
# except MaxTurnsExceeded:
#     print("Turn limit reached.")

# %% [markdown]
# ## 정리
#
# - `Runner.run_sync(..., max_turns=N, error_handlers={"max_turns": handler})`
# - 핸들러: `(RunErrorHandlerInput) -> RunErrorHandlerResult`
# - `include_in_history=False`: 폴백 메시지를 대화 기록에 남기지 않음
# - 지원하는 키: 현재 `"max_turns"` 등 (문서 참고)
