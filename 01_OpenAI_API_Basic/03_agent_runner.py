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
# # 03. Agent와 Runner
#
# **Runner**는 에이전트 루프를 돌리는 실행기입니다.
#
# - **동기**: `Runner.run_sync(agent, input)` — 스크립트/테스트에 적합
# - **비동기**: `await Runner.run(agent, input)` — Jupyter/비동기 앱에 적합
#
# **에이전트 루프** 동작:
# 1. LLM 호출 (에이전트 설정 + 메시지 히스토리)
# 2. 응답에 툴 호출이 있으면 실행 후 결과를 메시지에 추가하고 1번으로
# 3. 핸드오프가 있으면 대상 에이전트로 바꾸고 1번으로
# 4. **final_output**이 나오면 루프 종료 후 반환
# 5. `max_turns`로 최대 턴 수 제한 가능

# %% [markdown]
# ## 동기 실행 (run_sync)

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from agents import Agent, Runner

Model = "gpt-5-nano"

agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요.",
    model=Model
)

try:
    result = Runner.run_sync(agent, "2+2는 얼마일까요? 한 단어로 답하세요.")
    print(result.final_output)
except Exception as e:
    print(e)

# %% [markdown]
# ## 비동기 실행 (Runner.run)
#
# Jupyter에서는 이미 이벤트 루프가 돌고 있으므로 `await Runner.run(...)` 사용이 자연스럽습니다.
# 일반 스크립트에서는 `asyncio.run(main())`으로 감싸서 실행합니다.
#
# ```
# import asyncio
# from agents import Agent, Runner
#
# async def main():
#     agent = Agent(name="Assistant", instructions="Reply in one short sentence.")
#     result = await Runner.run(agent, "What is the capital of France?")
#     print(result.final_output)
#
# asyncio.run(main())
# ``

# %%
agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요.",
    model=Model
)

result = await Runner.run(agent, "2+2는 얼마일까요? 한 단어로 답하세요.")
print(result.final_output)

# %% [markdown]
# ## max_turns
#
# 루프가 무한히 돌지 않도록 최대 턴 수를 제한합니다.
# 턴 수를 초과하면 기본적으로 `MaxTurnsExceeded` 예외가 발생합니다.
# (예외 대신 메시지로 처리하려면 `08_error_handling.py`의 error_handlers 사용)

# %%
result = await Runner.run(
    agent,
    "1부터 5까지 각 메시지마다 숫자 하나씩 세어 보세요.",
    max_turns=2,
)
print(result.final_output)

# %% [markdown]
# ## 정리
#
# | 메서드 | 호출 방식 | 용도 |
# |--------|-----------|------|
# | `Runner.run_sync` | `Runner.run_sync(agent, input)` | 동기, 스크립트/테스트 |
# | `Runner.run` | `await Runner.run(agent, input)` | 비동기, Jupyter/앱 |
# | `max_turns` | `Runner.run_sync(..., max_turns=N)` | 루프 횟수 제한 |
