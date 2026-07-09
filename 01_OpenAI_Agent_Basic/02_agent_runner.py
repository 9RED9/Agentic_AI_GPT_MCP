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
# # 02. Agent와 Runner
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
#
#
# ### Runner 실행 방식 3종 비교
#
# | 메서드 | 호출 방식 | 특징 | 언제 쓰나 | 핵심 차이 |
# |---|---|---|---|---|
# | `Runner.run(...)` | `await Runner.run(...)` | 비동기적, 에이전트 루프 자동 실행, 도구 & 핸드오프 지원 | FastAPI, Jupyter, 서버 환경 | `await` 필요, 비동기 환경용 |
# | `Runner.run_sync(...)` | `Runner.run_sync(...)` | 동기 호출, 내부적으로 이벤트 루프를 만들어 `Runner.run()` 실행 | 일반 Python 스크립트, 테스트 | `await` 불필요, 동기 환경용 |
# | `Runner.run_streamed(...)` | `Runner.run_streamed(...)` + `async for` | 토큰 단위 실시간 출력, 중간 이벤트를 실시간 수신 | 실시간 채팅 UI, 챗봇 서비스 | 답변을 조각조각 실시간 수신 |
#
# - Jupyter notebook은 기본적으로 이벤트 루프가 이미 실행 중이므로 `await Runner.run(...)` 사용
# - `Runner.run_streamed(...)` 는 `await` 없이 호출하며, 반환된 결과의 `stream_events()`를 `async for`로 순회하며 소비
#

# %% [markdown]
# ## 1. 동기 실행 (Runner.run_sync)
#
# `Runner.run_sync()`는 **동기 호출** 방식으로, 내부적으로 새 이벤트 루프를 만들어(`asyncio.run()` 방식) `Runner.run()`을 실행합니다.
# `await`가 필요 없으므로 **일반 Python 스크립트, 테스트** 환경에 적합합니다.
#
# 단, Jupyter처럼 **이미 이벤트 루프가 실행 중인 환경**에서는 새 루프를 만들 수 없어 예외가 발생합니다.
# 아래 셀의 try/except는 이를 확인하기 위한 것입니다. (Jupyter에서는 `await Runner.run()`을 사용해야 함)

# %%
import os
from dotenv import load_dotenv
#load_dotenv() 

# 현재 실행 중인 Python 파일의 디렉토리를 기준으로 .env 경로 설정
#current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd() # 현재경로
parent_dir = os.path.dirname(current_dir) # os.path.dirname()을 한 번 감싸주면 상위 폴더 경로가 됩니다.
dotenv_path = os.path.join(parent_dir, '.env') # 3. 상위 폴더에 있는 진짜 .env 파일 경로 지정

# 디버깅을 위한 출력
print("수정된 .env 예상 경로:", dotenv_path)

# 4. override=True 옵션을 주어 기존의 잘못된 키 값을 확실하게 덮어씁니다.
is_loaded = load_dotenv(dotenv_path, override=True)
print("Env loaded:", is_loaded)

# 5. API 키 확인 (앞뒤 글자만 확인)
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print(f"로드된 API Key: {api_key[:12]}...{api_key[-4:]}")
else:
    print("❌ 여전히 API Key를 찾을 수 없습니다. 경로를 다시 확인해주세요.")

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from agents import Agent, Runner

Model = "gpt-5.4-mini"

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
# ## 2. 비동기 실행 (Runner.run)
#
# `await Runner.run(...)`은 **비동기 호출** 방식으로, 에이전트 루프를 자동 실행하며 **도구(tool)와 핸드오프(handoff)** 를 지원합니다.
# **Jupyter, FastAPI** 등 이미 이벤트 루프가 돌고 있는 환경에 적합합니다.
#
# 일반 스크립트에서 사용하려면 `asyncio.run(main())`으로 감싸서 실행합니다.
#
# ```python
# import asyncio
# from agents import Agent, Runner
#
# async def main():
#     agent = Agent(name="Assistant", instructions="Reply in one short sentence.")
#     result = await Runner.run(agent, "What is the capital of France?")
#     print(result.final_output)
#
# asyncio.run(main())
# ```
#
# 아래 예제는 도구를 등록한 에이전트를 `await Runner.run()`으로 실행합니다.

# %%
from agents import function_tool

@function_tool
def NumberTool(n: int) -> int:
    return n
    
agent = Agent(
    name="Assistant",
    instructions="간결하게 답변해 주세요.",
    model=Model,
    tools=[NumberTool],
)

result = await Runner.run(agent, "2+2는 얼마일까요? 한 단어로 답하세요.")
print(result.final_output)

# %% [markdown]
# ## 3. 스트리밍 실행 (Runner.run_streamed)
#
# `Runner.run_streamed()`는 LLM의 응답을 **토큰 단위로 실시간 수신**할 수 있어,
# 실시간 채팅 UI처럼 **답변이 생성되는 과정을 그대로 보여주는 서비스**에 적합합니다.
#
# 사용 방법:
# 1. `result = Runner.run_streamed(agent, input)` — `await` 없이 호출 (즉시 `RunResultStreaming` 반환)
# 2. `async for event in result.stream_events():` — 이벤트를 실시간으로 순회
# 3. 스트리밍이 끝나면 `result.final_output`으로 최종 결과 접근 가능
#
# ### 스트림 이벤트 종류
#
# | 이벤트 타입 | 내용 | 용도 |
# |---|---|---|
# | `raw_response_event` | LLM이 생성하는 **토큰 단위 델타** | 타자기 효과의 실시간 텍스트 출력 |
# | `run_item_stream_event` | 메시지 완성, 도구 호출/결과 등 **의미 단위 이벤트** | 진행 상황 표시 ("도구 실행 중...") |
# | `agent_updated_stream_event` | 핸드오프로 **에이전트가 교체**될 때 | 현재 응답 주체 표시 |

# %% [markdown]
# ### 3-1. 토큰 단위 실시간 출력
#
# `raw_response_event` 중 텍스트 델타(`ResponseTextDeltaEvent`)만 골라 출력하면,
# 챗봇 UI처럼 글자가 실시간으로 찍히는 효과를 얻을 수 있습니다.

# %%
from openai.types.responses import ResponseTextDeltaEvent

story_agent = Agent(
    name="Storyteller",
    instructions="사용자가 요청한 주제로 20줄 안팎의 긴 시를 지어 주세요.",
    model=Model,
)

# await 없이 호출 → RunResultStreaming 객체 반환
result = Runner.run_streamed(story_agent, "고양이에 대한 시를 써주세요.")

# 이벤트 스트림을 async for로 순회하며 토큰 단위로 출력
async for event in result.stream_events():
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        print(event.data.delta, end="", flush=True)  # 토큰(델타)을 즉시 출력

# %%
# 스트리밍 종료 후에는 final_output으로 전체 결과에 접근 가능
print(result.final_output)

# %% [markdown]
# ### 3-2. 의미 단위 이벤트 (run item events)
#
# 토큰 델타 대신 **"도구가 호출됨", "메시지가 완성됨"** 같은 의미 단위의 진행 상황을 받아
# 사용자에게 상태를 알려줄 수도 있습니다.

# %%
import random
from agents import ItemHelpers, function_tool

@function_tool
def pick_joke_count() -> int:
    """1~5 사이의 농담 개수를 무작위로 정한다."""
    return random.randint(1, 5)

joke_agent = Agent(
    name="Joker",
    instructions="먼저 pick_joke_count 도구를 호출해 농담 개수를 정한 뒤, 그 개수만큼 짧은 농담을 들려주세요.",
    model=Model,
    tools=[pick_joke_count],
)

result = Runner.run_streamed(joke_agent, "농담 들려주세요.")

async for event in result.stream_events():
    if event.type == "raw_response_event":
        continue  # 토큰 델타는 무시 (3-1 예제 참고)
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
# ## 4. max_turns
#
# 루프가 무한히 돌지 않도록 최대 턴 수를 제한합니다.
# 턴 수를 초과하면 기본적으로 `MaxTurnsExceeded` 예외가 발생합니다.
# (예외 대신 메시지로 처리하려면 `03_error_handling.py`의 error_handlers 사용)

# %%
from agents import Agent, Runner, MaxTurnsExceeded

# max_turns 초과 시 예외 발생 → 프로그램 중단
try:
    result = await Runner.run(
        agent,
        """
        절대 규칙:
        - 한 번의 LLM 응답에서 tool은 정확히 5번만 호출하세요.
        - tool 결과를 받은 뒤에만 다음 숫자로 진행하세요.
        - 1부터 시작해서 5까지 반복하세요.
        """,
        max_turns=2,
    )
    print(result.final_output)
except MaxTurnsExceeded:
    print("에러 발생!")  

# %%

# %% [markdown]
# ### 실습 문제
#
# 1. **스트리밍 시인 에이전트**:
#    사용자가 요청한 주제로 5줄 내외의 짧은 시를 짓는 `Poet` 에이전트를 만들고,
#    `Runner.run_streamed()`로 실행하여 **토큰 단위로 실시간 출력**(타자기 효과)하세요.
#    (힌트: `raw_response_event` + `ResponseTextDeltaEvent`)
#
# 2. **진행 상황 표시**:
#    주사위를 굴려 1~6 사이의 숫자를 반환하는 도구 `roll_dice()`를 등록한 에이전트를 만들고,
#    `run_item_stream_event`를 이용해 다음과 같은 **의미 단위 진행 상황**을 출력하세요.
#    - 도구가 호출되면: `"-- 도구 호출됨"`
#    - 도구 결과가 나오면: `"-- 도구 결과: <값>"`
#    - 메시지가 완성되면: 완성된 메시지 출력
#
# 3. **max_turns 제한**:
#    2번에서 만든 에이전트에게 "주사위를 5번 굴려주세요"라고 요청하되 `max_turns=2`로 제한하고,
#    `MaxTurnsExceeded` 예외를 잡아 `"턴 수 초과! max_turns를 늘려주세요."`를 출력하세요.
#
#
# ### 테스트 입력 예시
#
# * `"바다에 대한 시를 써주세요."`
#   👉 스트리밍으로 시가 한 글자씩 출력
#
# * `"주사위를 굴려서 나온 숫자로 짧은 농담을 만들어줘."`
#   👉 도구 호출됨 → 도구 결과 → 메시지 완성 순서로 출력
#
# * `"주사위를 5번 굴려주세요."` (max_turns=2)
#   👉 `"턴 수 초과! max_turns를 늘려주세요."` 출력

# %%
