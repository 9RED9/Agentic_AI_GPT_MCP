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
# # 실습 문제 모범답안 - 02. Agent와 Runner
#
# `02_agent_runner` 노트북의 실습 문제에 대한 모범답안 예시입니다.
#
# ### 문제 요약
#
# 1. **스트리밍 시인 에이전트**:
#    사용자가 요청한 주제로 5줄 내외의 짧은 시를 짓는 `Poet` 에이전트를 만들고,
#    `Runner.run_streamed()`로 실행하여 토큰 단위로 실시간 출력(타자기 효과)하세요.
#
# 2. **진행 상황 표시**:
#    주사위를 굴려 1~6 사이의 숫자를 반환하는 도구 `roll_dice()`를 등록한 에이전트를 만들고,
#    `run_item_stream_event`를 이용해 의미 단위 진행 상황을 출력하세요.
#
# 3. **max_turns 제한**:
#    2번 에이전트에게 "주사위를 5번 굴려주세요"라고 요청하되 `max_turns=2`로 제한하고,
#    `MaxTurnsExceeded` 예외를 잡아 안내 메시지를 출력하세요.

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
Model = "gpt-5.4-mini"

# %% [markdown]
# ## 1. 스트리밍 시인 에이전트
#
# `Runner.run_streamed()`는 `await` 없이 호출하며, 즉시 `RunResultStreaming` 객체를 반환합니다.
# `stream_events()`를 `async for`로 순회하면서 `raw_response_event` 중
# 텍스트 델타(`ResponseTextDeltaEvent`)만 골라 출력하면 타자기 효과를 얻을 수 있습니다.

# %%
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

# 5줄 내외의 짧은 시를 짓는 에이전트
poet_agent = Agent(
    name="Poet",
    instructions="사용자가 요청한 주제로 5줄 내외의 짧은 시를 지어 주세요.",
    model=Model,
)

# await 없이 호출 → RunResultStreaming 객체 반환
result = Runner.run_streamed(poet_agent, "바다에 대한 시를 써주세요.")

# 이벤트 스트림을 async for로 순회하며 토큰 단위로 출력
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)  # 토큰(델타)을 즉시 출력

# %% [markdown]
# ## 2. 진행 상황 표시 (의미 단위 이벤트)
#
# 토큰 델타 대신 `run_item_stream_event`를 이용하면
# **"도구 호출됨" → "도구 결과" → "메시지 완성"** 같은 의미 단위의 진행 상황을 표시할 수 있습니다.

# %%
import random
from agents import ItemHelpers, function_tool

# 주사위를 굴려 1~6 사이의 숫자를 반환하는 도구
@function_tool
def roll_dice() -> int:
    """주사위를 굴려 1~6 사이의 숫자를 반환한다."""
    return random.randint(1, 6)

dice_agent = Agent(
    name="Dice_agent",
    instructions="먼저 roll_dice 도구를 호출해 숫자를 정한 뒤, 그 숫자를 활용해 사용자의 요청을 처리하세요.",
    model=Model,
    tools=[roll_dice],
)

result = Runner.run_streamed(dice_agent, "주사위를 굴려서 나온 숫자로 짧은 농담을 만들어줘.")

async for event in result.stream_events():
    if event.type == "raw_response_event":
        continue  # 토큰 델타는 무시 (1번 문제 참고)
    elif event.type == "agent_updated_stream_event":
        print(f"[에이전트 시작]: {event.new_agent.name}")
    elif event.type == "run_item_stream_event":
        if event.item.type == "tool_call_item":
            print("-- 도구 호출됨")
        elif event.item.type == "tool_call_output_item":
            print(f"-- 도구 결과: {event.item.output}")
        elif event.item.type == "message_output_item":
            print(f"-- 메시지 완성:\n{ItemHelpers.text_message_output(event.item)}")

print("=== 실행 완료 ===")

# %% [markdown]
# ## 3. max_turns 제한
#
# 주사위를 5번 굴리려면 (도구를 턴마다 1번씩 호출할 경우) 최소 6턴이 필요하므로,
# `max_turns=2`로 제한하면 `MaxTurnsExceeded` 예외가 발생합니다.

# %%
from agents import MaxTurnsExceeded

try:
    result = await Runner.run(
        dice_agent,
        """
        절대 규칙:
        - 한 번의 LLM 응답에서 roll_dice 도구는 정확히 1번만 호출하세요.
        - 도구 결과를 받은 뒤에만 다음 주사위로 진행하세요.
        - 주사위를 총 5번 굴려야 하며, 5번을 모두 굴리기 전에는 절대 최종 답변을 하지 마세요.
        """,
        max_turns=2,
    )
    print(result.final_output)
except MaxTurnsExceeded:
    print("턴 수 초과! max_turns를 늘려주세요.")

# %% [markdown]
# ### 정리
#
# | 구성 요소 | 역할 |
# |------|------|
# | `Poet` + `Runner.run_streamed()` | 토큰 단위 실시간 출력 (타자기 효과) |
# | `roll_dice` (`@function_tool`) | 1~6 사이의 무작위 숫자 반환 도구 |
# | `Dice_agent` + `run_item_stream_event` | 도구 호출/결과/메시지 완성 진행 상황 표시 |
# | `max_turns=2` + `MaxTurnsExceeded` | 에이전트 루프 횟수 제한 및 예외 처리 |
#
# **이벤트 처리 흐름:**
# ```
# Runner.run_streamed() 호출
#       ↓
# stream_events() 를 async for로 순회
#       ├─ raw_response_event        → 토큰 델타 (타자기 효과)
#       ├─ agent_updated_stream_event → 에이전트 교체 알림
#       └─ run_item_stream_event      → 도구 호출 / 도구 결과 / 메시지 완성
# ```
